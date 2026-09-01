import jax.numpy as jnp
import numpy as np

from lifejacket import post_deployment_analysis as pda

# ---------------------------------------------------------------------------
# The refit percentile bootstrap (docs/adr/0003) decomposes into _newton_refit (a true-Newton
# root-find, re-differentiated each iterate) and refit_percentile_bootstrap (the draw loop,
# multiplicity contract, and quantiles). Each is tested against closed forms, so the expected
# values are derived, not tuned. The RN-weighted stack itself is exercised end-to-end by the
# integration test (test_RL_percentile_bootstrap_smoke) and by ADR 0003's independent-reference
# acceptance comparison, not here.
# ---------------------------------------------------------------------------


def test_newton_refit_solves_closed_form_system():
    def stack_fn(x):
        return jnp.array([x[0] ** 2 - 4.0, x[1] - 1.0])

    solution, converged, reason = pda._newton_refit(
        stack_fn, jnp.array([1.5, 0.0]), max_iterations=25, step_tolerance=1e-6
    )
    assert converged
    np.testing.assert_allclose(solution, [2.0, 1.0], atol=1e-4)
    assert reason in ("step", "residual_reduction")


def test_newton_refit_reports_failure_on_unsolvable_system():
    # tanh(x) = 2 has no root; the iteration must give up and say so rather than return
    # something plausible-looking.
    def stack_fn(x):
        return jnp.tanh(x) - 2.0

    _, converged, reason = pda._newton_refit(
        stack_fn, jnp.zeros(1), max_iterations=15, step_tolerance=1e-6
    )
    assert not converged
    # Newton chases the nonexistent root into tanh's saturating tail, where the derivative
    # underflows to a singular Jacobian -- either terminal reason is a correct diagnosis.
    assert reason in ("max_iterations", "singular_jacobian")


def test_refit_percentile_bootstrap_matches_closed_form_weighted_mean():
    # One-parameter system mean_i(m_i * (y_i - theta)) = 0 whose weighted refit has the closed
    # form theta* = sum(m*y)/sum(m). The test regenerates the multiplicities from the same seed
    # via the documented contract (np.random.default_rng(seed).poisson(1.0, size=(B, n))) and
    # requires the bootstrap's draws and interval to match the closed form exactly -- this pins
    # the multiplicity generation order, the subject-axis convention, and the quantile step all
    # at once (the same reproduce-from-seed mechanism ADR 0003's independent-reference
    # acceptance comparison relies on).
    rng = np.random.default_rng(7)
    n, num_draws, seed = 40, 60, 123
    y = jnp.asarray(rng.normal(1.5, 1.0, size=n).astype(np.float32))

    def weighted_avg_stack_fn(x, multiplicities):
        return jnp.array([jnp.mean(multiplicities * (y - x[0]))])

    result = pda.refit_percentile_bootstrap(
        weighted_avg_stack_fn,
        jnp.array([float(jnp.mean(y))]),
        theta_dim=1,
        num_subjects=n,
        num_draws=num_draws,
        alpha=0.05,
        seed=seed,
    )

    m = np.random.default_rng(seed).poisson(1.0, size=(num_draws, n)).astype(np.float64)
    expected = (m @ np.asarray(y, dtype=np.float64)) / m.sum(axis=1)

    assert result["bootstrap_num_failed_draws"] == 0
    np.testing.assert_allclose(result["theta_draws"][:, 0], expected, rtol=1e-5)
    np.testing.assert_allclose(
        result["percentile_bootstrap_ci"][0],
        np.quantile(expected, [0.025, 0.975]),
        rtol=1e-5,
    )


def test_refit_identity_multiplicities_returns_original_solution():
    # ADR 0003's weights=1 acceptance gate: with all multiplicities 1 the weighted system IS
    # the original system, whose root is the starting point -- every draw must come back at the
    # original solution (a joint 2-parameter system here, so the theta-block slicing is
    # exercised too).
    rng = np.random.default_rng(3)
    n = 30
    z = jnp.asarray(rng.normal(2.0, 1.0, size=n).astype(np.float32))
    y = jnp.asarray(rng.normal(0.5, 1.0, size=n).astype(np.float32))
    beta_hat = float(jnp.mean(z))
    theta_hat = float(jnp.mean(y)) / beta_hat

    def weighted_avg_stack_fn(x, multiplicities):
        beta, theta = x[0], x[1]
        return jnp.array(
            [
                jnp.mean(multiplicities * (z - beta)),
                jnp.mean(multiplicities * (y - theta * beta)),
            ]
        )

    num_draws = 8
    result = pda.refit_percentile_bootstrap(
        weighted_avg_stack_fn,
        jnp.array([beta_hat, theta_hat]),
        theta_dim=1,
        num_subjects=n,
        num_draws=num_draws,
        alpha=0.05,
        seed=None,
        precomputed_multiplicities=np.ones((num_draws, n)),
    )
    assert result["bootstrap_num_failed_draws"] == 0
    np.testing.assert_allclose(result["theta_draws"][:, 0], theta_hat, atol=1e-4)


def test_refit_reports_nan_interval_when_too_few_draws_converge():
    def weighted_avg_stack_fn(x, multiplicities):
        return jnp.tanh(x) - 2.0  # unsolvable regardless of the draw

    result = pda.refit_percentile_bootstrap(
        weighted_avg_stack_fn,
        jnp.zeros(1),
        theta_dim=1,
        num_subjects=5,
        num_draws=6,
        alpha=0.05,
        seed=0,
        max_newton_iterations=5,
    )
    assert result["bootstrap_num_failed_draws"] == 6
    assert np.all(np.isnan(result["percentile_bootstrap_ci"]))


# ---------------------------------------------------------------------------
# jacobian_row_chunk_size: the per-iterate Newton Jacobian is the same reverse-mode pass that
# builds the joint bread, re-run once per iteration per draw, so it obeys the package-wide
# memory-bounding policy (helper_functions.compute_row_chunked_jacobian). Chunking must be a
# pure memory decision: the draws, and therefore the interval, cannot depend on it. The system
# below is 5-dimensional on purpose -- chunk sizes 2 and 3 leave a short final chunk, which is
# where the basis padding is dropped and where an off-by-one would silently substitute zero
# rows into the Newton step.
# ---------------------------------------------------------------------------

_CHUNKING_SYSTEM_MATRIX = jnp.asarray(
    np.diag(np.full(5, 3.0)) + 0.4 * np.random.default_rng(11).normal(size=(5, 5))
)


def _coupled_residual(x):
    # Dense, non-symmetric, and genuinely nonlinear, so every Jacobian row differs and a
    # misplaced row changes the step rather than cancelling out.
    return _CHUNKING_SYSTEM_MATRIX @ x + 0.1 * x**2


def test_newton_refit_chunked_jacobian_matches_the_unchunked_path():
    target = jnp.asarray(np.linspace(-1.0, 1.0, 5))

    def stack_fn(x):
        return _coupled_residual(x) - target

    unchunked, converged, _ = pda._newton_refit(stack_fn, jnp.zeros(5), 25, 1e-6)
    assert converged

    for chunk_size in (1, 2, 3, 5, 9):
        chunked, chunk_converged, _ = pda._newton_refit(
            stack_fn,
            jnp.zeros(5),
            25,
            1e-6,
            jacobian_row_chunk_size=chunk_size,
        )
        assert chunk_converged, f"chunk_size={chunk_size} failed to converge"
        np.testing.assert_allclose(
            chunked,
            unchunked,
            rtol=1e-5,
            atol=1e-6,
            err_msg=f"chunk_size={chunk_size} reached a different root",
        )


def _chunking_bootstrap_fixture():
    num_subjects, num_draws = 12, 12
    rng = np.random.default_rng(4)
    per_subject_targets = jnp.asarray(rng.normal(size=(num_subjects, 5)))

    def weighted_avg_stack_fn(x, multiplicities):
        # mean_i m_i * (f(x) - y_i) -- the multiplicity-weighted mean stack shape
        # refit_percentile_bootstrap's contract requires, with a different root per draw.
        return jnp.mean(
            multiplicities[:, None]
            * (_coupled_residual(x)[None, :] - per_subject_targets),
            axis=0,
        )

    solution, converged, _ = pda._newton_refit(
        lambda x: weighted_avg_stack_fn(x, jnp.ones(num_subjects)),
        jnp.zeros(5),
        25,
        1e-8,
    )
    assert converged
    return {
        "weighted_avg_stack_fn": weighted_avg_stack_fn,
        "flattened_solution": jnp.asarray(solution),
        "num_subjects": num_subjects,
        "num_draws": num_draws,
        "multiplicities": np.random.default_rng(19).poisson(
            1.0, size=(num_draws, num_subjects)
        ),
    }


def test_refit_percentile_bootstrap_is_unchanged_by_jacobian_row_chunking():
    fixture = _chunking_bootstrap_fixture()

    def run(chunk_size):
        return pda.refit_percentile_bootstrap(
            fixture["weighted_avg_stack_fn"],
            fixture["flattened_solution"],
            theta_dim=2,
            num_subjects=fixture["num_subjects"],
            num_draws=fixture["num_draws"],
            alpha=0.05,
            seed=None,
            precomputed_multiplicities=fixture["multiplicities"],
            jacobian_row_chunk_size=chunk_size,
        )

    unchunked = run(None)
    chunked = run(3)  # does not divide the 5-row output basis

    assert unchunked["bootstrap_num_failed_draws"] == 0
    assert chunked["bootstrap_num_failed_draws"] == 0
    np.testing.assert_allclose(
        chunked["theta_draws"], unchunked["theta_draws"], rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        chunked["percentile_bootstrap_ci"],
        unchunked["percentile_bootstrap_ci"],
        rtol=1e-5,
        atol=1e-6,
    )


def test_refit_percentile_bootstrap_forwards_the_chunk_size_to_every_refit(monkeypatch):
    # The equivalence above would also pass if the chunk size were dropped on the floor, since
    # dropping it just means "unchunked" -- the memory bound, which is the whole point, would
    # be silently gone. So pin the plumbing directly.
    fixture = _chunking_bootstrap_fixture()
    observed = []
    real_newton_refit = pda._newton_refit

    def _spy(*args, **kwargs):
        observed.append(kwargs.get("jacobian_row_chunk_size"))
        return real_newton_refit(*args, **kwargs)

    monkeypatch.setattr(pda, "_newton_refit", _spy)

    pda.refit_percentile_bootstrap(
        fixture["weighted_avg_stack_fn"],
        fixture["flattened_solution"],
        theta_dim=2,
        num_subjects=fixture["num_subjects"],
        num_draws=3,
        alpha=0.05,
        seed=None,
        precomputed_multiplicities=fixture["multiplicities"][:3],
        jacobian_row_chunk_size=4,
    )

    assert observed == [4, 4, 4]
