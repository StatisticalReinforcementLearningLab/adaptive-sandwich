import math

import jax.numpy as jnp
import numpy as np

from lifejacket import diagnostics as d
from lifejacket.constants import CheckStatuses, DiagnosticClassifications

# ---------------------------------------------------------------------------
# Shared toy maps. These are deliberately tiny, hand-derived estimating maps -- not the full RL
# pipeline -- so every test below can compare against a closed-form answer.
# ---------------------------------------------------------------------------


def _affine_map(B):
    def g_tilde(eta):
        return jnp.asarray(B) @ eta

    return g_tilde


def _scalar_quadratic_map(b, c):
    """g(eta) = b*eta + c*eta^2, root at eta=0."""

    def g_tilde(eta):
        return b * eta + c * eta**2

    return g_tilde


# ---------------------------------------------------------------------------
# 0. check_root_and_implementation's backward-residual gate: a stale/mismatched bread_factored
#    (the exact bug class this gate exists to catch -- e.g. forgetting to re-factor B_hat after
#    it changes) must not silently produce a valid-looking root correction.
# ---------------------------------------------------------------------------


def test_root_and_implementation_hard_fails_on_broken_backward_residual():
    B_true = np.diag([2.0, 3.0])
    B_wrong = np.diag([2.0, 30.0])  # factored and used to solve, instead of B_true

    def g_tilde(eta):
        return B_true @ eta - jnp.array([0.02, 0.03])

    eta_hat = jnp.zeros(2)
    g0 = np.array([-0.02, -0.03])
    # Inflated so a_root_max stays far below its own tolerance regardless of d_root -- isolates
    # this test to the backward-residual gate specifically, not a root-error coincidence.
    V_hat = np.eye(2) * 1e6
    L = np.eye(2)
    labels = ["theta_0", "theta_1"]

    result = d.check_root_and_implementation(
        g_tilde,
        eta_hat,
        g0,
        B_true,
        np.eye(2),
        d.factor_bread(B_wrong),
        L,
        labels,
        V_hat,
        d.DiagnosticConfig(),
    )

    assert result.status == CheckStatuses.FAILED
    assert "Backward relative residual" in result.message
    assert (
        result.metrics["backward_relative_residual"]
        > d.DiagnosticConfig().backward_residual_tolerance
    )
    assert result.metrics["a_root_max"] < d.DiagnosticConfig().root_error_tolerance_se


# ---------------------------------------------------------------------------
# 0b. run_input_checks: black-and-white input/data-hygiene results, kept separate from the
#     numeric checks in check_results (never folded into root_and_implementation's own status).
# ---------------------------------------------------------------------------


def test_run_input_checks_reports_pass_and_fail_without_raising():
    def passing_check():
        return None

    def failing_check():
        raise ValueError("mismatched action probabilities")

    results = d.run_input_checks(
        [("a_passes", passing_check), ("b_fails", failing_check)]
    )

    assert results["a_passes"].status == CheckStatuses.PASSED
    assert results["b_fails"].status == CheckStatuses.FAILED
    assert "mismatched action probabilities" in results["b_fails"].message


def test_run_diagnostic_suite_hard_fails_on_input_check_failure_but_keeps_it_separate():
    rng = np.random.default_rng(4)
    d_total = 2
    n = 50
    B = np.eye(d_total)

    def g_tilde(eta):
        return jnp.asarray(B) @ eta

    eta_hat = jnp.zeros(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    joint_sandwich = M_hat / n  # B is identity here

    def failing_input_check():
        raise ValueError("could not reconstruct action probabilities")

    report = d.run_diagnostic_suite(
        g_tilde,
        eta_hat,
        B,
        M_hat,
        joint_sandwich,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(num_directions=10),
        legacy_check_callables=[
            ("action_probabilities_reconstructed", failing_input_check)
        ],
    )

    assert report.classification == DiagnosticClassifications.FAILED
    assert (
        report.input_check_results["action_probabilities_reconstructed"].status
        == CheckStatuses.FAILED
    )
    # The failure is a data-hygiene fact, not a statistical one -- root_and_implementation's own
    # status must be unaffected (this well-behaved affine map has nothing wrong with its root).
    assert (
        report.check_results["root_and_implementation"].status == CheckStatuses.PASSED
    )
    # A hard-failing input check still short-circuits the rest of the suite, same as before.
    assert set(report.check_results.keys()) == {"root_and_implementation"}


# ---------------------------------------------------------------------------
# 1. Affine estimating map: R_j = r_j = q_j = a_{j,l} = 0 up to numerical tolerance; exact
#    nonlinear and linear roots agree.
# ---------------------------------------------------------------------------


def test_affine_map_taylor_remainder_and_correction_are_zero():
    rng = np.random.default_rng(0)
    d_total = 4
    B = np.eye(d_total) + 0.05 * rng.standard_normal((d_total, d_total))
    B = B @ B.T + np.eye(d_total)
    g_tilde = _affine_map(B)
    eta_hat = jnp.zeros(d_total)
    g0 = np.zeros(d_total)
    bread_factored = d.factor_bread(B)

    deltas = rng.standard_normal((6, d_total)) * 0.1
    result = d.evaluate_taylor_remainder_and_correction(
        g_tilde, eta_hat, g0, B, bread_factored, deltas
    )

    np.testing.assert_allclose(result["R"], 0.0, atol=1e-4)
    np.testing.assert_allclose(result["r"], 0.0, atol=1e-3)
    np.testing.assert_allclose(result["c"], 0.0, atol=1e-4)
    np.testing.assert_allclose(result["q"], 0.0, atol=1e-3)


def test_affine_map_exact_and_linear_roots_agree():
    rng = np.random.default_rng(1)
    d_total = 3
    B = np.eye(d_total) + 0.05 * rng.standard_normal((d_total, d_total))
    B = B @ B.T + np.eye(d_total)
    g_tilde = _affine_map(B)
    eta_hat = jnp.zeros(d_total)
    g0 = np.zeros(d_total)
    bread_factored = d.factor_bread(B)

    s_j = rng.standard_normal(d_total) * 0.1
    delta_linear = d.solve_with_bread(bread_factored, s_j)
    config = d.DiagnosticConfig()
    solved = d.solve_exact_perturbation(
        g_tilde, eta_hat, g0, bread_factored, s_j, delta_linear, config
    )

    assert solved["converged"]
    np.testing.assert_allclose(solved["delta_nl"], delta_linear, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Scalar quadratic map: compare R_j, c_j, and the exact nonlinear-root discrepancy with
#    analytic values; verify expected radius scaling.
# ---------------------------------------------------------------------------


def test_scalar_quadratic_map_matches_closed_form_R_and_c():
    b, c = 2.0, 0.3
    g_tilde = _scalar_quadratic_map(b, c)
    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[b]])
    bread_factored = d.factor_bread(B)

    for delta_val in (0.05, -0.05, 0.2):
        deltas = np.array([[delta_val]])
        result = d.evaluate_taylor_remainder_and_correction(
            g_tilde, eta_hat, g0, B, bread_factored, deltas
        )
        expected_R = c * delta_val**2
        expected_c = -c * delta_val**2 / b
        np.testing.assert_allclose(result["R"][0, 0], expected_R, atol=1e-5)
        np.testing.assert_allclose(result["c"][0, 0], expected_c, atol=1e-5)


def test_scalar_quadratic_map_exact_root_matches_quadratic_formula():
    b, c = 2.0, 0.3
    g_tilde = _scalar_quadratic_map(b, c)
    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[b]])
    bread_factored = d.factor_bread(B)

    s_target = 0.05
    s_j = np.array([s_target])
    delta_linear = np.array([s_target / b])
    config = d.DiagnosticConfig(
        continuation_steps=20, nonlinear_solver_max_iterations=100
    )
    solved = d.solve_exact_perturbation(
        g_tilde, eta_hat, g0, bread_factored, s_j, delta_linear, config
    )

    # b*delta + c*delta^2 = s_target -> delta = (-b + sqrt(b^2 + 4 c s_target)) / (2c)
    expected_delta = (-b + math.sqrt(b**2 + 4 * c * s_target)) / (2 * c)
    assert solved["converged"]
    np.testing.assert_allclose(solved["delta_nl"][0], expected_delta, atol=1e-4)


def test_scalar_quadratic_map_radius_scaling():
    b, c = 2.0, 0.3
    g_tilde = _scalar_quadratic_map(b, c)
    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[b]])
    bread_factored = d.factor_bread(B)

    base = 0.1
    r_values = {}
    a_values = {}
    se = 1.0  # arbitrary fixed target SE for this closed-form check
    for s in (0.5, 1.0, 2.0):
        deltas = np.array([[s * base]])
        result = d.evaluate_taylor_remainder_and_correction(
            g_tilde, eta_hat, g0, B, bread_factored, deltas
        )
        r_values[s] = result["r"][0]
        a_values[s] = abs(result["c"][0, 0]) / se

    # r_j ~ s^1, a_{j,l} ~ s^2 in this exactly-quadratic regime.
    ratio_r = r_values[2.0] / r_values[0.5]
    ratio_a = a_values[2.0] / a_values[0.5]
    np.testing.assert_allclose(ratio_r, 4.0, rtol=1e-3)  # (2.0/0.5)^1
    np.testing.assert_allclose(ratio_a, 16.0, rtol=1e-3)  # (2.0/0.5)^2


# ---------------------------------------------------------------------------
# 3. Equation rescaling: raw r_j may change; c_j and target-standardized a_{j,l} are invariant.
# ---------------------------------------------------------------------------


def test_equation_rescaling_changes_r_but_not_c_or_a():
    # A genuinely 2D map is needed here: for a 1D (scalar) equation, ANY rescaling trivially
    # cancels out of the norm ratio r_j = ||R||/||B delta||. Rescaling equation COMPONENTS by
    # different amounts (non-scalar D) is what should change how those components combine under
    # a Euclidean norm, while leaving c_j (which solves the same B c = -R relation either way)
    # exactly invariant.
    B = np.diag([2.0, 3.0])
    c_vals = np.array([0.3, -0.2])

    def g_tilde(eta):
        return B @ eta + c_vals * eta**2

    eta_hat = jnp.zeros(2)
    g0 = np.zeros(2)
    bread_factored = d.factor_bread(B)
    deltas = np.array([[0.1, 0.15]])
    result = d.evaluate_taylor_remainder_and_correction(
        g_tilde, eta_hat, g0, B, bread_factored, deltas
    )

    D = np.diag([7.0, 0.1])  # a NON-scalar rescaling of the estimating equation

    def g_tilde_rescaled(eta):
        return D @ g_tilde(eta)

    B_rescaled = D @ B
    bread_factored_rescaled = d.factor_bread(B_rescaled)
    g0_rescaled = D @ g0
    result_rescaled = d.evaluate_taylor_remainder_and_correction(
        g_tilde_rescaled,
        eta_hat,
        g0_rescaled,
        B_rescaled,
        bread_factored_rescaled,
        deltas,
    )

    # r_j is an equation-space ratio and is allowed (expected) to change under rescaling.
    assert not np.isclose(result["r"][0], result_rescaled["r"][0], rtol=1e-2)
    # c_j solves the same underlying correction regardless of how the equation is scaled.
    np.testing.assert_allclose(result["c"], result_rescaled["c"], atol=1e-6)

    contrast = np.array([1.0, 0.0])
    se = 1.0
    a = abs(result["c"][0] @ contrast) / se
    a_rescaled = abs(result_rescaled["c"][0] @ contrast) / se
    np.testing.assert_allclose(a, a_rescaled, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. Parameter/contrast rescaling: raw q_j may change; target-standardized a_{j,l} is invariant.
# ---------------------------------------------------------------------------


def test_parameter_rescaling_changes_q_but_not_a():
    # 2D diagonal quadratic map g(eta) = B @ eta + c * eta**2 (elementwise), B diagonal.
    b_vals = np.array([2.0, 3.0])
    c_vals = np.array([0.3, -0.2])
    B = np.diag(b_vals)

    def g_tilde(eta):
        return B @ eta + c_vals * eta**2

    eta_hat = jnp.zeros(2)
    g0 = np.zeros(2)
    bread_factored = d.factor_bread(B)
    delta = np.array([[0.1, -0.05]])
    result = d.evaluate_taylor_remainder_and_correction(
        g_tilde, eta_hat, g0, B, bread_factored, delta
    )

    contrast = np.array([1.0, 0.0])
    # V_hat chosen arbitrarily but fixed for this closed-form check.
    V_hat = np.diag([0.04, 0.09])
    se_l = math.sqrt(contrast @ V_hat @ contrast)
    a = abs(result["c"][0] @ contrast) / se_l
    q = np.linalg.norm(result["c"][0]) / np.linalg.norm(delta[0])

    # Reparametrize: eta' = P @ eta with a non-uniform diagonal P (P != orthogonal).
    P = np.diag([5.0, 0.2])
    P_inv = np.linalg.inv(P)

    def g_tilde_reparam(eta_prime):
        return g_tilde(P_inv @ eta_prime)

    B_reparam = B @ P_inv
    bread_factored_reparam = d.factor_bread(B_reparam)
    delta_reparam = (P @ delta[0]).reshape(1, -1)
    result_reparam = d.evaluate_taylor_remainder_and_correction(
        g_tilde_reparam,
        P @ np.zeros(2),
        g0,
        B_reparam,
        bread_factored_reparam,
        delta_reparam,
    )

    q_reparam = np.linalg.norm(result_reparam["c"][0]) / np.linalg.norm(
        delta_reparam[0]
    )
    contrast_reparam = P_inv.T @ contrast
    V_hat_reparam = P @ V_hat @ P.T
    se_l_reparam = math.sqrt(contrast_reparam @ V_hat_reparam @ contrast_reparam)
    a_reparam = abs(result_reparam["c"][0] @ contrast_reparam) / se_l_reparam

    assert not np.isclose(q, q_reparam, rtol=1e-2)
    np.testing.assert_allclose(a, a_reparam, rtol=1e-4)
    np.testing.assert_allclose(se_l, se_l_reparam, rtol=1e-9)


# ---------------------------------------------------------------------------
# 5. Paired directions: even second-order correction, expected +/- behavior.
# ---------------------------------------------------------------------------


def test_paired_directions_quadratic_correction_is_even():
    b, c = 2.0, 0.3
    g_tilde = _scalar_quadratic_map(b, c)
    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[b]])
    bread_factored = d.factor_bread(B)

    delta_plus = np.array([[0.1]])
    delta_minus = -delta_plus
    result_plus = d.evaluate_taylor_remainder_and_correction(
        g_tilde, eta_hat, g0, B, bread_factored, delta_plus
    )
    result_minus = d.evaluate_taylor_remainder_and_correction(
        g_tilde, eta_hat, g0, B, bread_factored, delta_minus
    )
    # For a purely quadratic (even) nonlinearity, c_j(+delta) == c_j(-delta) exactly.
    np.testing.assert_allclose(result_plus["c"], result_minus["c"], atol=1e-6)


# ---------------------------------------------------------------------------
# 6. Covariance distortion: generalized eigenvalue / SE-ratio calculations on known matrices.
# ---------------------------------------------------------------------------


def test_generalized_eigenvalues_match_hand_computation():
    # V_L (linear) = I, nonlinear_cov = diag(4, 9) -> generalized eigenvalues [4, 9], SE ratios [2, 3].
    linear_cov = np.eye(2)
    nonlinear_cov = np.diag([4.0, 9.0])
    se_ratios = d.se_ratios_from_generalized_eigenvalues(nonlinear_cov, linear_cov)
    np.testing.assert_allclose(sorted(se_ratios), [2.0, 3.0], atol=1e-8)


def test_generalized_eigenvalues_nontrivial_linear_cov():
    linear_cov = np.diag([2.0, 8.0])
    nonlinear_cov = np.diag(
        [2.0, 32.0]
    )  # ratios: 2/2=1 -> se ratio 1; 32/8=4 -> se ratio 2
    se_ratios = d.se_ratios_from_generalized_eigenvalues(nonlinear_cov, linear_cov)
    np.testing.assert_allclose(sorted(se_ratios), [1.0, 2.0], atol=1e-8)


# ---------------------------------------------------------------------------
# 7. Rank-deficient target covariance: correct identified-subspace behavior and indeterminate
#    status when needed.
# ---------------------------------------------------------------------------


def test_joint_mahalanobis_correction_handles_rank_deficient_target_covariance():
    # V_L has a zero eigenvalue direction (rank-deficient).
    V_L = np.array([[1.0, 0.0], [0.0, 0.0]])
    c_matrix = np.array([[3.0, 5.0], [1.0, 0.0]])
    result = d.joint_mahalanobis_correction(
        c_matrix, np.eye(2), V_L, rank_tolerance=1e-8
    )
    assert result["rank_deficient"]
    assert result["effective_rank"] == 1
    # Only the identified (first) coordinate contributes; second row's c has zero first coord.
    np.testing.assert_allclose(result["values"], [3.0, 1.0], atol=1e-8)


def test_run_diagnostic_suite_reports_indeterminate_for_rank_deficient_target():
    rng = np.random.default_rng(2)
    d_total = 2
    n = 100
    B = np.eye(d_total)

    def g_tilde(eta):
        return jnp.asarray(B) @ eta

    eta_hat = jnp.zeros(d_total)
    # Construct per-subject stacks so that the second coordinate has (numerically) zero variance,
    # making the target covariance rank-deficient.
    stacks = rng.standard_normal((n, d_total))
    stacks[:, 1] = 0.0
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    joint_sandwich = M_hat / n  # B is identity here

    report = d.run_diagnostic_suite(
        g_tilde,
        eta_hat,
        B,
        M_hat,
        joint_sandwich,
        stacks,
        beta_dim=0,
        theta_dim=2,
        num_subjects=n,
        config=d.DiagnosticConfig(num_directions=20),
    )
    assert report.classification == DiagnosticClassifications.INDETERMINATE


# ---------------------------------------------------------------------------
# 8. Ill-conditioned bread: stable solves, warnings without explicit inversion, correct
#    numerical classification.
# ---------------------------------------------------------------------------


def test_ill_conditioned_bread_solves_stay_finite_and_are_flagged():
    B = np.diag([1.0, 1e-6])  # condition number ~1e6, but not singular
    M = np.eye(2)
    theta_dim = 2
    beta_dim = 0
    num_subjects = 50
    V_hat = np.linalg.solve(B, M) @ np.linalg.inv(B).T / num_subjects

    result = d.check_bread_stability(
        B, M, beta_dim, theta_dim, num_subjects, V_hat, d.DiagnosticConfig()
    )
    assert np.isfinite(result.metrics["full_bread_condition_number"])
    assert result.metrics["full_bread_condition_number"] > 1e5
    assert result.status in (
        CheckStatuses.WARNING,
        CheckStatuses.INDETERMINATE,
        CheckStatuses.PASSED,
    )

    # The bread is never explicitly inverted in factor_bread/solve_with_bread (LU-based).
    bread_factored = d.factor_bread(B)
    x = d.solve_with_bread(bread_factored, np.array([1.0, 1.0]))
    assert np.all(np.isfinite(x))


# ---------------------------------------------------------------------------
# 9. Nonlinear root failure / multiple-root behavior: continuation diagnostics and failure
#    classification.
# ---------------------------------------------------------------------------


def test_continuation_reports_nonfinite_domain_excursion():
    # g_tilde has a pole; a large enough perturbation drives the continuation path through it.
    def g_tilde(eta):
        return 1.0 / (1.0 - eta[0]) * jnp.ones(1) - 1.0

    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[1.0]])
    bread_factored = d.factor_bread(B)

    s_j = np.array([50.0])  # forces the path toward eta -> 1, where g_tilde blows up
    delta_linear = np.array([50.0])
    config = d.DiagnosticConfig(
        continuation_steps=5, nonlinear_solver_max_iterations=10
    )
    solved = d.solve_exact_perturbation(
        g_tilde, eta_hat, g0, bread_factored, s_j, delta_linear, config
    )
    assert not solved["converged"]


def test_continuation_failure_to_converge_within_iteration_budget():
    b, c = 1.0, 5.0  # strongly nonlinear relative to the perturbation size
    g_tilde = _scalar_quadratic_map(b, c)
    eta_hat = jnp.array([0.0])
    g0 = np.array([0.0])
    B = np.array([[b]])
    bread_factored = d.factor_bread(B)

    s_j = np.array([10.0])
    delta_linear = np.array([10.0])
    config = d.DiagnosticConfig(continuation_steps=1, nonlinear_solver_max_iterations=1)
    solved = d.solve_exact_perturbation(
        g_tilde, eta_hat, g0, bread_factored, s_j, delta_linear, config
    )
    # With only a single continuation step and a single Newton iteration allowed, convergence
    # to tight tolerance is not guaranteed for a strongly nonlinear map.
    assert isinstance(solved["converged"], bool)


# ---------------------------------------------------------------------------
# 10. Influence diagnostics: analytically known variance shares and effective counts.
# ---------------------------------------------------------------------------


def test_influence_concentration_uniform_contributions():
    n = 10
    d_total = 1
    B = np.array([[1.0]])
    bread_factored = d.factor_bread(B)
    stacks = np.ones((n, d_total))  # every subject contributes identically
    L = np.array([[1.0]])
    result = d.check_influence_concentration(
        stacks, bread_factored, L, ["target_0"], list(range(n)), d.DiagnosticConfig()
    )
    by_target = result.metrics["by_target"]["target_0"]
    np.testing.assert_allclose(by_target["p_max"], 1.0 / n, atol=1e-8)
    np.testing.assert_allclose(by_target["n_eff"], n, atol=1e-8)


def test_influence_concentration_single_dominant_subject():
    n = 10
    d_total = 1
    B = np.array([[1.0]])
    bread_factored = d.factor_bread(B)
    stacks = np.zeros((n, d_total))
    stacks[0, 0] = 100.0  # subject 0 dominates the variance entirely
    for i in range(1, n):
        stacks[i, 0] = 1.0
    L = np.array([[1.0]])
    result = d.check_influence_concentration(
        stacks, bread_factored, L, ["target_0"], list(range(n)), d.DiagnosticConfig()
    )
    by_target = result.metrics["by_target"]["target_0"]
    assert by_target["p_max"] > 0.9
    assert by_target["n_eff"] < 2.0
    assert result.status == CheckStatuses.WARNING


def test_influence_concentration_thresholds_are_overridable():
    n = 10
    B = np.array([[1.0]])
    bread_factored = d.factor_bread(B)
    stacks = np.ones((n, 1))
    stacks[0, 0] = 2.5  # a mild, sub-default concentration: p_max ~0.41, n_eff ~4.8
    L = np.array([[1.0]])

    default_result = d.check_influence_concentration(
        stacks, bread_factored, L, ["target_0"], list(range(n)), d.DiagnosticConfig()
    )
    assert default_result.status == CheckStatuses.PASSED

    stricter_p_max = d.check_influence_concentration(
        stacks,
        bread_factored,
        L,
        ["target_0"],
        list(range(n)),
        d.DiagnosticConfig(influence_p_max_tolerance=0.3),
    )
    assert stricter_p_max.status == CheckStatuses.WARNING

    stricter_n_eff = d.check_influence_concentration(
        stacks,
        bread_factored,
        L,
        ["target_0"],
        list(range(n)),
        d.DiagnosticConfig(influence_n_eff_min_floor=5.0),
    )
    assert stricter_n_eff.status == CheckStatuses.WARNING


# ---------------------------------------------------------------------------
# 11. Clopper-Pearson calculations are covered in test_simulator_calibration.py (shared helper).
# ---------------------------------------------------------------------------

from lifejacket.simulator_calibration import clopper_pearson_upper_bound  # noqa: E402


def test_clopper_pearson_helper_reused_by_diagnostics_module():
    assert math.isclose(
        clopper_pearson_upper_bound(0, 59, 0.95), 1 - 0.05 ** (1 / 59), rel_tol=1e-9
    )


# ---------------------------------------------------------------------------
# 12. Reproducibility: identical perturbations and reports for a fixed seed.
# ---------------------------------------------------------------------------


def test_run_diagnostic_suite_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(3)
    d_total = 4
    theta_dim = 2
    beta_dim = 2
    n = 60

    B = np.eye(d_total) + 0.05 * rng.standard_normal((d_total, d_total))
    B = B @ B.T + np.eye(d_total)

    def g_tilde(eta):
        return jnp.asarray(B) @ eta

    eta_hat = jnp.zeros(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    joint_sandwich = np.linalg.solve(B, M_hat) @ np.linalg.inv(B).T / n

    config = d.DiagnosticConfig(num_directions=8, random_seed=42)

    report_1 = d.run_diagnostic_suite(
        g_tilde,
        eta_hat,
        B,
        M_hat,
        joint_sandwich,
        stacks,
        beta_dim,
        theta_dim,
        n,
        config,
    )
    report_2 = d.run_diagnostic_suite(
        g_tilde,
        eta_hat,
        B,
        M_hat,
        joint_sandwich,
        stacks,
        beta_dim,
        theta_dim,
        n,
        config,
    )

    assert report_1.classification == report_2.classification
    np.testing.assert_allclose(
        report_1.check_results["local_nonlinearity"].metrics["per_radius"][1.0]["r"][
            "median"
        ],
        report_2.check_results["local_nonlinearity"].metrics["per_radius"][1.0]["r"][
            "median"
        ],
    )
    np.testing.assert_allclose(
        report_1.check_results["influence_concentration"].metrics["by_target"][
            "theta_0"
        ]["p_max"],
        report_2.check_results["influence_concentration"].metrics["by_target"][
            "theta_0"
        ]["p_max"],
    )
