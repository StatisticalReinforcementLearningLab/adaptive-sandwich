import dataclasses
import math
import pickle

import jax.numpy as jnp
import numpy as np
import pytest

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


def test_run_diagnostic_suite_reports_failed_for_rank_deficient_target():
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
    # Rank deficiency is a measured failure (76/76 of the undercoverage hunt's zero-width
    # collapses), not a could-not-evaluate.
    assert report.classification == DiagnosticClassifications.FAILED
    assert report.check_results["bread_stability"].status == CheckStatuses.FAILED


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
    # Both status triggers fire on this fixture: V_hat = diag(1, 1e12)/n is rank 1 at
    # rank_tolerance=1e-8 (FAILED -- the measured predictor of variance collapse), and the
    # 1e-6*||B|| perturbation doubles the small diagonal entry so the second SE moves enormously
    # (INDETERMINATE on its own). Rank deficiency wins: check_bread_stability has no WARNING
    # path at all -- PASSED, INDETERMINATE (fragile SEs) or FAILED (unidentified components).
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["target_covariance_rank_estimate"] < 2
    assert (
        result.metrics["numerical_sensitivity_max_relative_se_change"]
        > d.DiagnosticConfig().se_distortion_tolerance
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
    # With only a single continuation step and a single Newton iteration allowed, the one chord
    # step lands nowhere near the root of this strongly nonlinear map: at the warm start d = 10
    # the linear part exactly meets the target of 10 and the quadratic part leaves a residual of
    # 5*10^2 = 500, and the iteration budget is exhausted before a second attempt.
    assert not solved["converged"]
    assert solved["final_residual_norm"] > 1.0


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
# 11. Clopper-Pearson calculations (shared helper, now in helper_functions).
# ---------------------------------------------------------------------------

from lifejacket.helper_functions import clopper_pearson_upper_bound  # noqa: E402


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


# ---------------------------------------------------------------------------
# 8. Frozen-score multiplier bootstrap (check_multiplier_bootstrap) and the simulated
#    finite-draw null bands it shares with check_exact_nonlinear_perturbations. Each test uses
#    a map whose perturbed roots have a closed form, so the expected verdict is derivable, not
#    tuned.
# ---------------------------------------------------------------------------


def _scalar_bootstrap_inputs(seed=0, n=50, stack_scale=3.0, b=1.0):
    """1-D estimating problem: root at 0, bread [[b]], per-subject stacks ~ N(0, stack_scale^2).
    Returns everything check_multiplier_bootstrap needs, with V_hat = M/(n b^2) exactly."""
    rng = np.random.default_rng(seed)
    stacks = rng.standard_normal((n, 1)) * stack_scale
    M_hat = float((stacks.T @ stacks)[0, 0]) / n
    B = np.array([[b]])
    V_hat = np.array([[M_hat / (n * b * b)]])
    return stacks, B, V_hat


def test_multiplier_bootstrap_passes_on_affine_map():
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    g_tilde = _affine_map(B)
    result = d.check_multiplier_bootstrap(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(1),
        ["theta_0"],
        V_hat,
        d.DiagnosticConfig(num_bootstrap_draws=100),
    )
    assert result.status == CheckStatuses.PASSED
    ratio = result.metrics["se_ratio_by_target"]["theta_0"]
    # For an affine map the bootstrap re-solve IS the linear solve, so the only deviation from
    # 1.0 is the finite-draw noise the null band exists to absorb.
    assert 0.8 < ratio < 1.2
    assert result.metrics["num_converged_trials"] == result.metrics["num_trials"]
    band_lo, band_hi = result.metrics["se_ratio_null_band"]
    assert band_lo <= ratio <= band_hi


def test_multiplier_bootstrap_fails_when_roots_inflate_relative_to_linearization():
    # g(eta) = b*w*asinh(eta/w): B at the root is b, but the perturbed root w*sinh(s/(b*w))
    # grows faster than the linear prediction s/b, so the true resampling distribution is wider
    # than the sandwich's linearization claims -- the anticonservative direction.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]
    sd_s = math.sqrt(float((stacks.T @ stacks)[0, 0]) / n / n)
    w = (
        sd_s / 0.8
    )  # s/(b*w) has sd ~0.8 -> sinh inflates the SD by ~1.4x, far past the band

    def g_tilde(eta):
        return jnp.asarray(B) @ (w * jnp.arcsinh(eta / w))

    result = d.check_multiplier_bootstrap(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        d.factor_bread(B),
        stacks,
        n,
        np.eye(1),
        ["theta_0"],
        V_hat,
        d.DiagnosticConfig(
            num_bootstrap_draws=100,
            nonlinear_solver_max_iterations=200,
        ),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["se_ratio_by_target"]["theta_0"] > 1.2


def test_multiplier_bootstrap_warns_when_sandwich_se_is_overstated():
    # Affine map but a V_hat inflated 4x above the truth: bootstrap SEs land at ~half the
    # sandwich SEs -- the conservative/blow-up direction reports WARNING, not FAILED.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    result = d.check_multiplier_bootstrap(
        _affine_map(B),
        jnp.zeros(1),
        np.zeros(1),
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(1),
        ["theta_0"],
        4.0 * V_hat,
        d.DiagnosticConfig(num_bootstrap_draws=100),
    )
    assert result.status == CheckStatuses.WARNING
    assert result.metrics["se_ratio_by_target"]["theta_0"] < 0.7


def test_multiplier_bootstrap_fails_when_resolve_fragility_is_confirmed():
    # g(eta) = b*w*tanh(eta/w) has range (-b*w, b*w); multiplier draws routinely exceed it, so
    # many re-solves cannot converge. With >20% of 100 trials failing, the failure rate's
    # Clopper-Pearson lower bound clears bad_direction_probability_target by an order of
    # magnitude: fragility is statistically CONFIRMED on the full trial set (no
    # optimistic-subset caveat applies to counting failures), so the check FAILS -- it must
    # never read as a confident PASS from the surviving easy draws, and no longer hides a
    # measured collapse behind INDETERMINATE either.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]
    sd_s = math.sqrt(float((stacks.T @ stacks)[0, 0]) / n / n)
    w = 0.5 * sd_s  # most draws lie outside the reachable range

    def g_tilde(eta):
        return jnp.asarray(B) @ (w * jnp.tanh(eta / w))

    result = d.check_multiplier_bootstrap(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        d.factor_bread(B),
        stacks,
        n,
        np.eye(1),
        ["theta_0"],
        V_hat,
        d.DiagnosticConfig(num_bootstrap_draws=50),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["root_failure_fraction"] > 0.2
    assert any("confirmed fragility" in w for w in result.warnings)
    (rate_criterion,) = [c for c in result.criteria if "failure rate" in c.description]
    assert rate_criterion.ok is False
    assert rate_criterion.severity == "fail"
    assert "lower confidence bound" in rate_criterion.value


def test_exact_check_passes_on_affine_map_with_few_directions():
    # Regression test for the former fixed [0.95, 1.05] se_ratios band, which failed with
    # certainty at practical direction counts (100% of 6,648 ADS-142 replicates, affine cells
    # included) because it ignored the finite-J noise of the ensemble covariance. Under the
    # simulated null band, a genuinely affine map must PASS even at J=8.
    rng = np.random.default_rng(11)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    report = d.run_diagnostic_suite(
        _affine_map(B),
        jnp.zeros(d_total),
        B,
        M_hat,
        M_hat / n,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(
            num_directions=10,
            compute_exact_nonlinear_roots=True,
            num_exact_directions=8,
        ),
    )
    exact = report.check_results["exact_nonlinear_perturbation"]
    assert exact.status == CheckStatuses.PASSED
    assert exact.metrics["se_ratios_within_tolerance"] is True
    assert exact.metrics["num_converged_trials"] == exact.metrics["num_trials"]


def test_exact_check_excludes_nonconverged_solves_and_confirms_fragility():
    # g(eta) = b*clip(eta, -cap, cap): EXACTLY linear inside its solvable range, saturated
    # outside it. Directions whose targets are unreachable fail to converge; the converged ones
    # are perfectly linear. The check must exclude the failures from a^NL (no solver debris in
    # the ensemble statistics -- the converged-only a^NL is ~0 here) and report FAILED: with
    # roughly half of 30 trials failing, the failure rate's lower confidence bound clears the
    # 0.01 target, so the equation's fragility at perturbation scale is a confirmed
    # measurement, not a gap in the evidence (a PASS from the optimistically-selected subset
    # would still be unearned).
    stacks, B, V_hat = _scalar_bootstrap_inputs(seed=3)
    n = stacks.shape[0]
    sd_s = math.sqrt(float((stacks.T @ stacks)[0, 0]) / n / n)
    cap = 0.7 * sd_s  # b = 1, so ~half the |s_j| draws exceed the reachable range

    def g_tilde(eta):
        return jnp.asarray(B) @ jnp.clip(eta, -cap, cap)

    result = d.check_exact_nonlinear_perturbations(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        B,
        d.factor_bread(B),
        stacks,
        n,
        np.eye(1),
        ["theta_0"],
        V_hat,
        d.DiagnosticConfig(num_exact_directions=15),
    )
    assert result.status == CheckStatuses.FAILED
    assert any("confirmed fragility" in w for w in result.warnings)
    assert 0 < result.metrics["num_converged_trials"] < result.metrics["num_trials"]
    a_nl_max = result.metrics["a_nl_by_target"]["theta_0"]["max"]
    assert math.isfinite(a_nl_max)
    assert a_nl_max < 1e-3


# ---------------------------------------------------------------------------
# 8b. check_exact_nonlinear_perturbations' FAILURE directions. The branch's headline rewrite
#     (fixed [0.95, 1.05] band -> simulated finite-draw null band) had coverage only for PASSED
#     and INDETERMINATE, so every gate that can actually fail -- and the precedence rule that
#     keeps a failure standing under an unhealthy solver -- was untested on the very check
#     docs/adr/0002 uses as the answer key for calibrating the others.
#
#     Each test below isolates ONE gate: the other gates are either provably inert on its map
#     (an odd map has no mean shift; an affine map has no quantile shift) or explicitly opened up
#     via the config, so the assertion genuinely dies if the gate under test is disabled rather
#     than being propped up by a sibling gate firing on the same fixture.
# ---------------------------------------------------------------------------


def _run_scalar_exact_check(g_tilde, stacks, B, V_hat, config):
    return d.check_exact_nonlinear_perturbations(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        B,
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(1),
        ["theta_0"],
        V_hat,
        config,
    )


def test_exact_check_fails_when_se_ratios_exceed_the_null_band():
    # Same inflating map as the bootstrap's anticonservative test: g(eta) = b*w*asinh(eta/w) has
    # bread b at the root but exact roots w*sinh(s/(b*w)) that grow faster than the linearization
    # predicts, so the ensemble covariance is wider than V_L. w is set so s/(b*w) has sd ~1.5,
    # which puts the SE ratio far above the simulated band rather than just past its edge.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]
    sd_s = math.sqrt(float((stacks.T @ stacks)[0, 0]) / n / n)
    w = sd_s / 1.5

    def g_tilde(eta):
        return jnp.asarray(B) @ (w * jnp.arcsinh(eta / w))

    result = _run_scalar_exact_check(
        g_tilde,
        stacks,
        B,
        V_hat,
        # asinh is ODD, so the paired mean shift is exactly zero and that gate cannot fire here.
        # The quantile gate is opened up so the se_ratios band is the only live gate left.
        d.DiagnosticConfig(
            num_exact_directions=15,
            nonlinear_solver_max_iterations=200,
            quantile_shift_tolerance_se=1e6,
        ),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["se_ratios_within_tolerance"] is False
    assert max(result.metrics["se_ratios"]) > result.metrics["se_ratio_null_band"][1]
    assert result.metrics["mean_shift_se"] < result.metrics["mean_shift_threshold"]
    # No solver debris: every direction converged, so the verdict is not a censoring artifact.
    assert result.metrics["num_converged_trials"] == result.metrics["num_trials"]


def test_exact_check_fails_when_se_ratios_fall_below_the_null_band():
    # Affine map with a V_hat inflated 4x above the truth: the exact roots ARE the linear ones,
    # so the ensemble is 2x narrower than the claimed target covariance. Unlike the bootstrap
    # check (which calls the conservative direction a WARNING), the exact check treats a
    # below-band ratio on a HEALTHY solver as a failure -- there is no censoring to blame it on.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    result = _run_scalar_exact_check(
        _affine_map(B),
        stacks,
        B,
        4.0 * V_hat,
        d.DiagnosticConfig(num_exact_directions=15),
    )
    assert result.status == CheckStatuses.FAILED
    assert min(result.metrics["se_ratios"]) < result.metrics["se_ratio_null_band"][0]
    assert result.metrics["root_failure_fraction"] == 0.0


def _bounded_even_bump_map(w, curvature):
    """g(eta) = eta + c*w*(1 - exp(-(eta/w)^2)): monotone (hence globally solvable) for c < 1.16,
    with a BOUNDED even part. Bounding the even part is what makes this the clean mean-shift
    fixture: it displaces every solved root in the same direction (mean shift ~ 0.42*c SE) while
    contributing almost no extra spread (SD inflation ~ sqrt(1 + 0.11*c^2)), so the SE-ratio gate
    stays comfortably inside its null band and cannot stand in for the gate under test. A plain
    quadratic cannot do this: its even part is unbounded, so it inflates the spread and blows up
    the tail quantiles at least as fast as it shifts the mean -- and it has no root at all past
    its branch point, which censors the ensemble asymmetrically."""

    def g_tilde(eta):
        return eta + curvature * w * (1.0 - jnp.exp(-((eta / w) ** 2)))

    return g_tilde


def test_exact_check_fails_on_curvature_induced_mean_shift():
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    w = math.sqrt(float(V_hat[0, 0]))

    result = _run_scalar_exact_check(
        _bounded_even_bump_map(w, 0.35),
        stacks,
        B,
        V_hat,
        # The quantile gate is opened up (a bounded even displacement necessarily moves the tail
        # quantiles too) so that the mean-shift gate is the only thing that can produce FAILED.
        d.DiagnosticConfig(
            num_exact_directions=15,
            nonlinear_solver_max_iterations=200,
            quantile_shift_tolerance_se=1e6,
        ),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["mean_shift_gate_evaluable"] is True
    assert result.metrics["mean_shift_se"] > result.metrics["mean_shift_threshold"]
    # Under intact antithetic pairing the simulated null band collapses to zero, so the floor
    # config.mean_shift_tolerance_se is the binding threshold in this ordinary case.
    assert result.metrics["mean_shift_threshold"] == pytest.approx(
        d.DiagnosticConfig().mean_shift_tolerance_se
    )
    # The SE-ratio gate is NOT what failed here: the ratio sits inside its own null band.
    band_lo, band_hi = result.metrics["se_ratio_null_band"]
    assert band_lo < min(result.metrics["se_ratios"])
    assert max(result.metrics["se_ratios"]) < band_hi
    assert result.metrics["num_converged_trials"] == result.metrics["num_trials"]


def test_exact_check_failed_verdict_stands_when_the_solver_is_unhealthy():
    # Status-precedence rule: FAILED from the converged subset outranks the INDETERMINATE that a
    # high non-convergence rate would otherwise produce, because excluding the non-converged
    # directions selects the EASY ones and therefore biases the distortion measurement DOWN.
    # Coordinate 0 inflates (asinh); coordinate 1 has bounded range (tanh) so a few draws are
    # unreachable and the solve fails. The two stack columns are independent, so which draws are
    # censored is unrelated to how much coordinate 0 inflates.
    rng = np.random.default_rng(7)
    n = 60
    stacks = rng.standard_normal((n, 2)) * 3.0
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    B = np.eye(2)
    V_hat = M_hat / n
    sd_s = np.sqrt(np.diag(V_hat))
    w_inflating, w_bounded = sd_s[0] / 2.0, 1.3 * sd_s[1]

    def g_tilde(eta):
        return jnp.stack(
            [
                w_inflating * jnp.arcsinh(eta[0] / w_inflating),
                w_bounded * jnp.tanh(eta[1] / w_bounded),
            ]
        )

    result = d.check_exact_nonlinear_perturbations(
        g_tilde,
        jnp.zeros(2),
        np.zeros(2),
        B,
        d.factor_bread(B),
        stacks,
        n,
        np.eye(2),
        ["theta_0", "theta_1"],
        V_hat,
        d.DiagnosticConfig(
            num_exact_directions=15,
            nonlinear_solver_max_iterations=200,
            quantile_shift_tolerance_se=1e6,
            # Between this fixture's confirmed lower bound (2 failures of 30 -> 0.012) and its
            # observed fraction (0.0667): the solver reads UNHEALTHY but NOT confirmed-fragile,
            # which is the precedence this test exists to pin -- a distortion FAILED from the
            # optimistically-selected converged subset outranking the unhealthy INDETERMINATE.
            # At the default 0.01 target the confirmed-fragility rung would fire first and the
            # gates would never be consulted.
            bad_direction_probability_target=0.03,
        ),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["root_failure_fraction"] > 0.03
    assert 0 < result.metrics["num_converged_trials"] < result.metrics["num_trials"]
    assert max(result.metrics["se_ratios"]) > result.metrics["se_ratio_null_band"][1]
    assert any("optimistically selected" in w for w in result.warnings)
    # Merely unhealthy: the rate criterion stays [indeterminate]-severity, not [fail].
    (rate_criterion,) = [c for c in result.criteria if "failure rate" in c.description]
    assert rate_criterion.ok is False
    assert rate_criterion.severity == "indeterminate"


def test_multiplier_bootstrap_auto_mode_screens_on_local_nonlinearity():
    rng = np.random.default_rng(21)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n

    def run(mode):
        return d.run_diagnostic_suite(
            _affine_map(B),
            jnp.zeros(d_total),
            B,
            M_hat,
            M_hat / n,
            stacks,
            beta_dim=0,
            theta_dim=d_total,
            num_subjects=n,
            config=d.DiagnosticConfig(
                num_directions=10,
                multiplier_bootstrap=mode,
                num_bootstrap_draws=20,
            ),
        )

    # Affine map: a_{j,l} ~ 0, local check PASSED -> the auto screen skips the bootstrap.
    assert "multiplier_bootstrap" not in run("auto").check_results
    always = run("always")
    assert "multiplier_bootstrap" in always.check_results
    assert always.check_results["multiplier_bootstrap"].status == CheckStatuses.PASSED
    assert "multiplier_bootstrap" not in run("off").check_results


def test_multiplier_bootstrap_auto_mode_runs_on_a_headline_screen_exceedance():
    # The screen's POSITIVE branch, isolated to the a_{j,l} trigger: c = 0.15 puts the headline
    # a_{j,l} at ~0.074 -- above bootstrap_screen_a_jl_threshold (0.05) but below
    # nonlinear_correction_tolerance_se (0.10), so check_local_nonlinearity itself still PASSES
    # and the headline comparison is the only thing that can start the bootstrap.
    rng = np.random.default_rng(21)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n

    report = d.run_diagnostic_suite(
        lambda eta: eta + 0.15 * eta**2,
        jnp.zeros(d_total),
        B,
        M_hat,
        M_hat / n,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(
            num_directions=10, multiplier_bootstrap="auto", num_bootstrap_draws=20
        ),
    )
    local = report.check_results["local_nonlinearity"]
    headline = d._local_nonlinearity_headline_max(local.metrics)
    assert local.status == CheckStatuses.PASSED
    assert (
        d.DiagnosticConfig().bootstrap_screen_a_jl_threshold
        < headline
        < d.DiagnosticConfig().nonlinear_correction_tolerance_se
    )
    assert "multiplier_bootstrap" in report.check_results


def test_multiplier_bootstrap_auto_mode_runs_when_the_local_check_does_not_pass():
    # The screen's OTHER trigger: a perfectly affine map (headline a_{j,l} ~ 0, far below the
    # screen threshold) whose target covariance is rank-deficient, so check_local_nonlinearity
    # comes back INDETERMINATE. A non-PASSED local check is exactly when the linearization-free
    # second opinion is worth its cost, so the bootstrap must run despite the ~0 headline.
    rng = np.random.default_rng(23)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks[:, 1] = 0.0  # the second target has no identified variance
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n

    report = d.run_diagnostic_suite(
        _affine_map(B),
        jnp.zeros(d_total),
        B,
        M_hat,
        M_hat / n,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(
            num_directions=10, multiplier_bootstrap="auto", num_bootstrap_draws=20
        ),
    )
    local = report.check_results["local_nonlinearity"]
    assert local.status != CheckStatuses.PASSED
    assert (
        d._local_nonlinearity_headline_max(local.metrics)
        < d.DiagnosticConfig().bootstrap_screen_a_jl_threshold
    )
    assert "multiplier_bootstrap" in report.check_results


# ---------------------------------------------------------------------------
# 10. check_exploration_and_weights -- the suite's only always-on hard gate besides
#     root_and_implementation, previously with no direct coverage. Positivity/overlap
#     enforcement lives HERE (the legacy input check was deliberately toothless and has been
#     removed), so both directions of every gate get a case.
# ---------------------------------------------------------------------------


def _exploration_df(probs, active=None):
    import pandas as pd

    n = len(probs)
    return pd.DataFrame(
        {
            "in_study": active if active is not None else [1] * n,
            "calendar_t": [t % 2 for t in range(n)],
            "action1prob": probs,
        }
    )


def _run_exploration_check(df, config=None, **kwargs):
    return d.check_exploration_and_weights(
        df,
        "in_study",
        "calendar_t",
        "action1prob",
        config or d.DiagnosticConfig(),
        **kwargs,
    )


def test_exploration_check_passes_on_interior_probabilities():
    result = _run_exploration_check(_exploration_df([0.3, 0.7, 0.5, 0.4]))
    assert result.status == CheckStatuses.PASSED
    assert result.metrics["action_prob_global_min"] == 0.3
    assert result.metrics["action_prob_global_max"] == 0.7


def test_exploration_check_hard_fails_on_boundary_probability():
    # Exactly 1.0 violates the OPEN interval requirement -- this is where positivity/overlap
    # is actually enforced.
    result = _run_exploration_check(_exploration_df([0.3, 1.0, 0.5, 0.4]))
    assert result.status == CheckStatuses.FAILED
    assert any("outside the open interval" in w for w in result.warnings)


def test_exploration_check_hard_fails_on_nonfinite_probability():
    result = _run_exploration_check(_exploration_df([0.3, float("nan"), 0.5, 0.4]))
    assert result.status == CheckStatuses.FAILED
    assert any("Nonfinite" in w for w in result.warnings)
    # The finiteness criterion is NOT redundant with the interval criterion: a NaN passes
    # `<= 0` / `>= 1` vacuously (NaN comparisons are always false), so the interval criterion
    # reads [ok] here and ONLY the finiteness criterion catches the NaN -- with a count, since
    # the displayed [min, max] range (pandas skipna) also cannot reveal one.
    (finite_criterion,) = [c for c in result.criteria if "finite (" in c.description]
    assert finite_criterion.ok is False
    assert finite_criterion.value == "1 of 4 nonfinite"
    (interval_criterion,) = [
        c for c in result.criteria if "strictly inside" in c.description
    ]
    assert interval_criterion.ok is True


def test_exploration_check_ignores_inactive_rows():
    # An out-of-range value on an INACTIVE row is not a violation: the policy never acted there.
    result = _run_exploration_check(
        _exploration_df([0.3, 5.0, 0.5, 0.4], active=[1, 0, 1, 1])
    )
    assert result.status == CheckStatuses.PASSED
    assert result.metrics["action_prob_global_max"] == 0.5


def test_exploration_check_enforces_supplied_design_bounds_in_both_directions():
    config = d.DiagnosticConfig(exploration_floor=0.1, exploration_ceiling=0.9)
    ok = _run_exploration_check(_exploration_df([0.1, 0.5, 0.9, 0.4]), config)
    assert ok.status == CheckStatuses.PASSED
    assert ok.metrics["fraction_at_or_near_floor"] == 0.25
    assert ok.metrics["fraction_at_or_near_ceiling"] == 0.25

    below = _run_exploration_check(_exploration_df([0.05, 0.5, 0.5, 0.4]), config)
    assert below.status == CheckStatuses.FAILED
    assert any("exploration_floor" in w for w in below.warnings)

    above = _run_exploration_check(_exploration_df([0.5, 0.95, 0.5, 0.4]), config)
    assert above.status == CheckStatuses.FAILED
    assert any("exploration_ceiling" in w for w in above.warnings)


def test_exploration_check_importance_weight_trajectories():
    # Equal final weights across subjects -> normalized ESS is exactly 1; a nonfinite weight
    # under perturbation is a hard fail.
    df = _exploration_df([0.3, 0.7, 0.5, 0.4])
    healthy = _run_exploration_check(
        df, perturbed_weight_trajectories=[{0: {1: 2.0}, 1: {1: 2.0}}]
    )
    assert healthy.status == CheckStatuses.PASSED
    assert healthy.metrics["ess_over_n_by_direction"]["median"] == 1.0

    broken = _run_exploration_check(
        df, perturbed_weight_trajectories=[{0: {1: float("inf")}, 1: {1: 2.0}}]
    )
    assert broken.status == CheckStatuses.FAILED
    assert any("Nonfinite importance weight" in w for w in broken.warnings)


# ---------------------------------------------------------------------------
# 11. check_root_and_implementation -- the two documented failure modes that had doc snippets
#     but no tests (docs/diagnostics_tutorial.md section 5, cases B and C).
# ---------------------------------------------------------------------------


def test_root_correction_beyond_tolerance_hard_fails():
    # eta_hat = 0 is NOT a root of g(eta) = eta - [1, 0]; with se = 0.1 per contrast the
    # implied correction is 10 SE -- far past root_error_tolerance_se = 0.01.
    B = np.eye(2)
    result = d.check_root_and_implementation(
        lambda eta: B @ eta - jnp.array([1.0, 0.0]),
        jnp.zeros(2),
        np.array([-1.0, 0.0]),
        B,
        np.eye(2),
        d.factor_bread(B),
        np.eye(2),
        ["theta_0", "theta_1"],
        np.eye(2) * 0.01,  # se = 0.1 per contrast
        d.DiagnosticConfig(),
    )
    assert result.status == CheckStatuses.FAILED
    np.testing.assert_allclose(result.metrics["a_root_max"], 10.0, rtol=1e-3)


def test_mismatched_supplied_derivative_warns_but_does_not_fail():
    # g_tilde really is I @ eta, but the supplied "analytic" Jacobian says diag([1, 2]) -- the
    # finite-difference spot check must flag it as a warning (g0 is exactly 0, isolating this
    # from the root-error gate).
    B_true = np.eye(2)
    B_wrong = np.diag([1.0, 2.0])
    result = d.check_root_and_implementation(
        lambda eta: B_true @ eta,
        jnp.zeros(2),
        np.zeros(2),
        B_wrong,
        np.eye(2),
        d.factor_bread(B_wrong),
        np.eye(2),
        ["theta_0", "theta_1"],
        np.eye(2),
        d.DiagnosticConfig(),
    )
    assert result.status == CheckStatuses.WARNING
    assert result.metrics["finite_difference_max_relative_error"] > 0.01


# ---------------------------------------------------------------------------
# 12. check_bread_stability -- healthy pass and the rank/fragility indeterminate paths.
# ---------------------------------------------------------------------------


def test_bread_stability_passes_on_well_conditioned_system():
    n = 50
    result = d.check_bread_stability(
        np.eye(2), np.eye(2), 0, 2, n, np.eye(2) / n, d.DiagnosticConfig()
    )
    assert result.status == CheckStatuses.PASSED
    assert result.metrics["target_covariance_rank_estimate"] == 2
    assert (
        result.metrics["numerical_sensitivity_max_relative_se_change"]
        < d.DiagnosticConfig().se_distortion_tolerance
    )


def test_bread_stability_fails_on_rank_deficient_target_covariance():
    n = 50
    result = d.check_bread_stability(
        np.eye(2), np.eye(2), 0, 2, n, np.diag([1.0, 0.0]) / n, d.DiagnosticConfig()
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["target_covariance_rank_estimate"] < 2
    (rank_criterion,) = [
        c for c in result.criteria if "components identified" in c.description
    ]
    assert rank_criterion.ok is False
    assert rank_criterion.severity == "fail"


def test_bread_stability_flags_numerically_fragile_standard_errors():
    # B = diag(1, 1e-5): the check's 1e-6 * ||B||-scale perturbation is ~10% of the small
    # diagonal, so that component's SE moves past se_distortion_tolerance (~9%). M is scaled
    # so V = diag(1, 1)/n stays FULL rank: with rank deficiency now a FAILED trigger, the old
    # M = I fixture (V = diag(1, 1e10)/n, rank 1) would fail on rank and this test would stop
    # exercising the sensitivity trigger at all. Fragile SEs alone are INDETERMINATE -- the
    # ADS-142 grid measured the sensitivity gate as non-predictive of real variance failure
    # (and conservative when it fires), so it never earns FAILED on its own.
    n = 50
    B = np.diag([1.0, 1e-5])
    M = np.diag([1.0, 1e-10])
    V = np.linalg.solve(B, M) @ np.linalg.inv(B).T / n
    result = d.check_bread_stability(B, M, 0, 2, n, V, d.DiagnosticConfig())
    assert result.status == CheckStatuses.INDETERMINATE
    assert result.metrics["target_covariance_rank_estimate"] == 2
    assert (
        result.metrics["numerical_sensitivity_max_relative_se_change"]
        > d.DiagnosticConfig().se_distortion_tolerance
    )


# ---------------------------------------------------------------------------
# 13. check_jacobian_drift -- pass on an affine map, warn when the Jacobian moves by more than
#     the contraction threshold (rho >= 1) within the sampled perturbation range.
# ---------------------------------------------------------------------------


def _drift_inputs(seed=5, n=25, stack_scale=5.0):
    rng = np.random.default_rng(seed)
    stacks = rng.standard_normal((n, 1)) * stack_scale
    return stacks, np.array([[1.0]])


def test_jacobian_drift_passes_on_affine_map():
    stacks, B = _drift_inputs()
    result = d.check_jacobian_drift(
        _affine_map(B),
        jnp.zeros(1),
        B,
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        d.DiagnosticConfig(),
    )
    assert result.status == CheckStatuses.PASSED
    assert result.metrics["rho_max"] < 0.1


def test_jacobian_drift_warns_when_contraction_threshold_crossed():
    # g(eta) = eta + 10*eta^2 has Dg = 1 + 20*eta: at sandwich-scale path points (~0.2 here)
    # rho = |Dg - 1| = |20*eta| crosses 1 well inside the sampled range.
    stacks, B = _drift_inputs()

    def g_tilde(eta):
        return eta + 10.0 * eta**2

    result = d.check_jacobian_drift(
        g_tilde,
        jnp.zeros(1),
        B,
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        d.DiagnosticConfig(),
    )
    assert result.status == CheckStatuses.WARNING
    assert result.metrics["rho_max"] >= 1.0


# ---------------------------------------------------------------------------
# 14. check_multiplier_bootstrap's mean-shift gate -- even curvature displaces the resampled
#     roots systematically, which no SE ratio can see.
# ---------------------------------------------------------------------------


def _run_scalar_bootstrap(g_tilde, stacks, B, V_hat, config):
    return d.check_multiplier_bootstrap(
        g_tilde,
        jnp.zeros(1),
        np.zeros(1),
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(1),
        ["theta_0"],
        V_hat,
        config,
    )


def test_multiplier_bootstrap_fails_on_curvature_induced_mean_shift():
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    w = math.sqrt(float(V_hat[0, 0]))

    result = _run_scalar_bootstrap(
        _bounded_even_bump_map(w, 0.35),
        stacks,
        B,
        V_hat,
        d.DiagnosticConfig(
            num_bootstrap_draws=100, nonlinear_solver_max_iterations=200
        ),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["mean_shift_gate_evaluable"] is True
    assert result.metrics["mean_shift_se"] > result.metrics["mean_shift_threshold"]
    # The mean-shift gate is the sole trigger: the SE ratio stays inside its null band, and the
    # solver is healthy (every draw is globally solvable), so neither the band gate nor the
    # failure-fraction gate can be what produced this verdict.
    band_lo, band_hi = result.metrics["se_ratio_null_band"]
    assert band_lo < result.metrics["se_ratio_by_target"]["theta_0"] < band_hi
    assert result.metrics["num_converged_trials"] == result.metrics["num_trials"]


def test_multiplier_bootstrap_fails_on_confirmed_fragility_of_a_censored_ensemble():
    # TWO deliberate behavior changes, in sequence. First (older): this fixture
    # (g(eta) = eta + 0.8*eta^2) used to assert FAILED on a mean shift of 0.214 SE -- an
    # artifact: the map has NO root once a draw passes its branch point, so ~27% of the
    # re-solves fail and roughly a third of the converged rows lose their antithetic partner;
    # measured on complete +/- pairs only, the genuine displacement is ~0.088 SE, below the
    # practical-significance floor -- so the assertion moved to INDETERMINATE and the real
    # curvature-induced FAILED lives in the test above, on a globally solvable map. Second
    # (current): the 27% failure fraction is itself the finding -- its lower confidence bound
    # clears bad_direction_probability_target, so "no root past the branch point" is a
    # CONFIRMED fragility of the estimating equation and the check FAILS on that, not on the
    # censoring-artifact mean shift (whose pair-only statistic must still stay under its
    # threshold, pinned below).
    stacks, B, V_hat = _scalar_bootstrap_inputs()

    result = _run_scalar_bootstrap(
        lambda eta: eta + 0.8 * eta**2,
        stacks,
        B,
        V_hat,
        d.DiagnosticConfig(num_bootstrap_draws=100),
    )
    assert result.status == CheckStatuses.FAILED
    assert any("confirmed fragility" in w for w in result.warnings)
    assert (
        result.metrics["root_failure_fraction"]
        > d.DiagnosticConfig().bad_direction_probability_target
    )
    # Rows whose antithetic partner did not converge are dropped from the location statistic.
    assert (
        result.metrics["mean_shift_num_rows"] < result.metrics["num_converged_trials"]
    )
    assert result.metrics["mean_shift_se"] <= result.metrics["mean_shift_threshold"]
    assert any("complete +/- pairs only" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 15. sample_multiplier_perturbations' multiplier distributions. The whole frozen-score bootstrap
#     rests on nu being mean-0/variance-1: a typo in Mammen's two-point constants leaves the
#     draws looking perfectly reasonable while every re-solve carries a systematic score offset,
#     which no downstream assertion about SE ratios would notice.
# ---------------------------------------------------------------------------


def _recover_multipliers(distribution, num_subjects=200, num_draws=200, seed=0):
    """Recovers the raw nu draws. With per-subject stacks == I_n and bread == I_n, the sampler's
    s_b = (nu @ stacks)/n is exactly nu/n, so the multipliers are readable off its output."""
    stacks = np.eye(num_subjects)
    _, S = d.sample_multiplier_perturbations(
        stacks,
        d.factor_bread(np.eye(num_subjects)),
        num_subjects,
        num_draws,
        seed,
        distribution,
    )
    return S * num_subjects


def test_multiplier_distributions_are_mean_zero_unit_variance():
    for distribution in ("rademacher", "mammen", "gaussian"):
        nu = _recover_multipliers(distribution)
        # 40,000 draws: the sample mean's own sd is 0.005 and the sample variance's is ~0.007,
        # so these margins are ~10 sd wide and cannot be tripped by Monte Carlo noise alone.
        assert abs(float(nu.mean())) < 0.05, distribution
        assert abs(float(nu.var(ddof=1)) - 1.0) < 0.05, distribution


def test_mammen_multipliers_match_the_two_point_distribution():
    nu = _recover_multipliers("mammen")
    sqrt5 = math.sqrt(5.0)
    support = np.unique(nu)
    np.testing.assert_allclose(
        sorted(support), [(1.0 - sqrt5) / 2.0, (1.0 + sqrt5) / 2.0], atol=1e-12
    )
    # Mammen's defining third property, and the one that pins the point masses: with mean 0 and
    # variance 1 already fixed, E[nu^3] = 1 leaves exactly one two-point distribution, so this
    # catches a swapped or mistyped prob_low that the first two moments alone would tolerate.
    assert abs(float((nu**3).mean()) - 1.0) < 0.1


def test_rademacher_multipliers_are_plus_or_minus_one():
    np.testing.assert_allclose(
        sorted(np.unique(_recover_multipliers("rademacher"))), [-1.0, 1.0]
    )


def test_unknown_multiplier_distribution_raises():
    with pytest.raises(ValueError, match="Unknown bootstrap_multiplier_distribution"):
        d.sample_multiplier_perturbations(
            np.eye(4), d.factor_bread(np.eye(4)), 4, 2, 0, "student_t"
        )


# ---------------------------------------------------------------------------
# 16. leave_one_out_theta_sensitivity, driven through its only production caller.
# ---------------------------------------------------------------------------


def test_leave_one_out_shift_matches_the_closed_form_for_a_mean_estimator():
    # Scalar mean estimator: g_i(theta) = theta - y_i, so B = [[1]] and stacks_i = theta_hat - y_i
    # (which sum to zero at the root). Excluding subject i leaves an average of
    # -(theta_hat - y_i)/(n-1), and one Newton step from theta_hat moves theta by
    # +(theta_hat - y_i)/(n-1) -- i.e. away from the excluded subject's observation.
    rng = np.random.default_rng(5)
    n = 12
    y = rng.standard_normal(n) * 2.0
    theta_hat = float(y.mean())
    stacks = (theta_hat - y).reshape(n, 1)
    B = np.array([[1.0]])

    result = d.check_influence_concentration(
        stacks,
        d.factor_bread(B),
        np.array([[1.0]]),
        ["theta_0"],
        list(range(n)),
        d.DiagnosticConfig(
            compute_leave_one_out_sensitivity=True, leave_one_out_top_k=3
        ),
        B_hat=B,
        theta_dim=1,
    )

    loo = result.metrics["leave_one_out_sensitivity"]
    assert len(loo) == 3
    # The subjects checked are the most influential ones: the largest |y_i - theta_hat|.
    expected_subjects = set(np.argsort(-np.abs(y - theta_hat))[:3].tolist())
    assert {entry["subject_id"] for entry in loo} == expected_subjects
    for entry in loo:
        i = entry["subject_id"]
        np.testing.assert_allclose(
            entry["one_step_theta_shift"][0],
            (theta_hat - y[i]) / (n - 1),
            rtol=1e-10,
        )


def test_leave_one_out_sensitivity_is_off_by_default():
    stacks = np.arange(6.0).reshape(6, 1)
    B = np.array([[1.0]])
    result = d.check_influence_concentration(
        stacks,
        d.factor_bread(B),
        np.array([[1.0]]),
        ["theta_0"],
        list(range(6)),
        d.DiagnosticConfig(),
        B_hat=B,
        theta_dim=1,
    )
    assert "leave_one_out_sensitivity" not in result.metrics


# ---------------------------------------------------------------------------
# 17. REGRESSION: a g_tilde that is undefined off the observed point. Probing out to 1.5x the
#     sandwich scale routinely leaves the domain of a log/logit/sqrt-shaped estimating function.
#     The nonfinite rows used to reach solve_with_bread's check_finite=True LU solve, which
#     raised out of the always-on local-nonlinearity check and destroyed the ENTIRE report --
#     no diagnostic_report.pkl at all, from a condition the suite is supposed to measure.
# ---------------------------------------------------------------------------


def _censoring_inputs(seed=13, n=40, d_total=2):
    rng = np.random.default_rng(seed)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n
    return stacks, np.eye(d_total), M_hat, M_hat / n


def _run_local_nonlinearity(g_tilde, stacks, B, V_hat, num_directions):
    d_total = B.shape[0]
    return d.check_local_nonlinearity(
        g_tilde,
        jnp.zeros(d_total),
        np.zeros(d_total),
        B,
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(d_total),
        [f"theta_{i}" for i in range(d_total)],
        V_hat,
        d.DiagnosticConfig(num_directions=num_directions),
    )


def _nan_off_the_root_map(d_total):
    def g_tilde(eta):
        at_root = jnp.all(eta == 0.0)
        return jnp.where(at_root, jnp.zeros(d_total), jnp.full(d_total, jnp.nan))

    return g_tilde


def test_local_nonlinearity_fails_on_a_fully_undefined_neighborhood_without_raising():
    # TOTAL censoring is a measured failure, not an unevaluable probe set: every probe point
    # at sampling scale returned a nonfinite g_tilde, so the linearization the adjusted
    # sandwich relies on has no domain there. (Partial censoring keeps its WARNING /
    # INDETERMINATE ladder -- see the test below.)
    stacks, B, _, V_hat = _censoring_inputs()
    result = _run_local_nonlinearity(
        _nan_off_the_root_map(B.shape[0]), stacks, B, V_hat, num_directions=6
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["domain_censored_fraction"] == 1.0
    assert result.metrics["num_domain_censored_probes"] == result.metrics["num_probes"]
    assert any("undefined (nonfinite)" in w for w in result.warnings)
    assert any("nonfinite at ALL" in w for w in result.warnings)
    (survival_criterion,) = [
        c for c in result.criteria if "stayed well-defined" in c.description
    ]
    assert survival_criterion.ok is False
    assert survival_criterion.severity == "fail"


def test_local_nonlinearity_downgrades_to_warning_under_partial_censoring():
    # Undefined only in one half-space, so roughly a third of the headline-radius probes are lost
    # while the rest stay perfectly affine: every surviving statistic is finite and clean, but a
    # clean pass on an optimistically-selected subset is not a pass.
    stacks, B, _, V_hat = _censoring_inputs()
    cap = 0.5 * float(np.sqrt(V_hat[0, 0]))

    def g_tilde(eta):
        undefined = eta[0] > cap
        return jnp.where(undefined, jnp.full(B.shape[0], jnp.nan), jnp.asarray(B) @ eta)

    result = _run_local_nonlinearity(g_tilde, stacks, B, V_hat, num_directions=8)
    assert result.status == CheckStatuses.WARNING
    assert 0.0 < result.metrics["domain_censored_fraction"] < 1.0
    assert any("undefined (nonfinite)" in w for w in result.warnings)
    headline = result.metrics["per_radius"][result.metrics["headline_radius"]]
    assert 0.0 < headline["domain_censored_fraction"] <= 0.5
    # The surviving probes still support finite statistics -- censoring is reported, not fatal.
    for summary in headline["a_by_target"].values():
        assert math.isfinite(summary["max"])
        assert math.isfinite(summary["exceedance_fraction"])


def test_run_diagnostic_suite_survives_an_undefined_neighborhood():
    # The report-destroying case end to end: previously this raised out of run_diagnostic_suite.
    stacks, B, M_hat, V_hat = _censoring_inputs()
    report = d.run_diagnostic_suite(
        _nan_off_the_root_map(B.shape[0]),
        jnp.zeros(B.shape[0]),
        B,
        M_hat,
        V_hat,
        stacks,
        beta_dim=0,
        theta_dim=B.shape[0],
        num_subjects=stacks.shape[0],
        config=d.DiagnosticConfig(num_directions=6),
    )
    assert isinstance(report, d.DiagnosticReport)
    assert report.classification == DiagnosticClassifications.FAILED
    assert report.check_results["local_nonlinearity"].status == CheckStatuses.FAILED
    assert (
        report.check_results["local_nonlinearity"].metrics["domain_censored_fraction"]
        == 1.0
    )


# ---------------------------------------------------------------------------
# 18. REGRESSION: the mean-shift gate's simulated null band. With unpaired draws b_L is raw
#     sampling noise of size sqrt(chi2_dim / J) -- ~0.26 SE in one dimension at J = 15 -- which
#     the former fixed 0.10 comparison read as curvature on provably affine maps (16 of 20 seeds).
# ---------------------------------------------------------------------------


def test_mean_shift_gate_does_not_fire_on_an_affine_map_with_unpaired_directions():
    seeds = range(8)
    observed = []
    for seed in seeds:
        stacks, B, V_hat = _scalar_bootstrap_inputs(seed=seed)
        result = _run_scalar_exact_check(
            _affine_map(B),
            stacks,
            B,
            V_hat,
            d.DiagnosticConfig(random_seed=seed, paired_directions=False),
        )
        metrics = result.metrics
        assert metrics["mean_shift_gate_evaluable"] is True
        # Without antithetic cancellation the band -- not the fixed floor -- is what binds.
        assert (
            metrics["mean_shift_threshold"]
            > 2 * d.DiagnosticConfig().mean_shift_tolerance_se
        )
        assert metrics["mean_shift_threshold"] == pytest.approx(
            metrics["mean_shift_null_upper"]
        )
        observed.append(
            (
                metrics["mean_shift_se"] > metrics["mean_shift_threshold"],
                metrics["mean_shift_se"] > d.DiagnosticConfig().mean_shift_tolerance_se,
            )
        )

    # The band's nominal false-positive rate is 5%, so a lone exceedance across eight seeds is
    # expected behavior rather than a defect; the old fixed comparison fired on most of them.
    assert sum(exceeds_band for exceeds_band, _ in observed) <= 1
    assert sum(exceeds_old_fixed for _, exceeds_old_fixed in observed) >= 4


def test_mean_shift_floor_binds_under_intact_antithetic_pairing():
    # The other half of the contract: with complete +/- pairs the linear parts cancel identically,
    # so the simulated null band collapses to zero and config.mean_shift_tolerance_se -- the
    # practical-significance floor -- is the threshold that actually governs the ordinary case.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    result = _run_scalar_exact_check(
        _affine_map(B), stacks, B, V_hat, d.DiagnosticConfig()
    )
    assert result.status == CheckStatuses.PASSED
    assert result.metrics["mean_shift_null_upper"] < 1e-9
    assert result.metrics["mean_shift_threshold"] == pytest.approx(
        d.DiagnosticConfig().mean_shift_tolerance_se
    )
    assert result.metrics["mean_shift_num_rows"] == result.metrics["num_trials"]


# ---------------------------------------------------------------------------
# 19. REGRESSION: check_bread_stability's rank gate is judged on the TARGET covariance
#     (L V_hat L^T), not on the full joint sandwich. Judging it on the joint sandwich made the
#     gate unfireable at any real study scale: the healthy beta blocks alone keep the joint rank
#     well above theta_dim no matter how singular the theta block is.
# ---------------------------------------------------------------------------


def _joint_with_singular_theta_block(beta_dim=3, num_updates=2, theta_dim=2, n=50):
    d_total = beta_dim * num_updates + theta_dim
    V = np.eye(d_total) / n
    V[-1, -1] = 0.0  # the last theta contrast is not identified
    return np.eye(d_total), np.eye(d_total), V, beta_dim, theta_dim, n


def test_bread_stability_rank_gate_fires_on_a_singular_theta_block_with_healthy_betas():
    B, M, V, beta_dim, theta_dim, n = _joint_with_singular_theta_block()
    config = d.DiagnosticConfig()
    L, _ = d.default_contrast_matrix(B.shape[0] - theta_dim, theta_dim, config)

    result = d.check_bread_stability(B, M, beta_dim, theta_dim, n, V, config, L=L)

    assert result.status == CheckStatuses.FAILED
    assert result.metrics["target_covariance_dim"] == theta_dim
    assert result.metrics["target_covariance_rank_estimate"] < theta_dim
    assert len(result.metrics["target_covariance_eigenvalues"]) == theta_dim
    assert any("rank estimate" in w for w in result.warnings)
    # Why the old full-joint computation could never see this: the beta blocks are healthy and
    # full-rank, so the joint spectrum has 7 of 8 nonzero eigenvalues -- comfortably above
    # theta_dim = 2 -- and the gate stayed silent while a target contrast was unidentified.
    joint_eigvals = np.linalg.eigvalsh(V)
    joint_rank = int(
        np.sum(joint_eigvals > config.rank_tolerance * joint_eigvals.max())
    )
    assert joint_rank > theta_dim


def test_bread_stability_rank_gate_is_judged_on_the_supplied_contrast():
    B, M, V, beta_dim, theta_dim, n = _joint_with_singular_theta_block()
    config = d.DiagnosticConfig()

    def single_contrast(index):
        L = np.zeros((1, B.shape[0]))
        L[0, index] = 1.0
        return d.check_bread_stability(B, M, beta_dim, theta_dim, n, V, config, L=L)

    identified = single_contrast(-2)
    assert identified.metrics["target_covariance_dim"] == 1
    assert identified.metrics["target_covariance_rank_estimate"] == 1
    assert identified.status == CheckStatuses.PASSED

    unidentified = single_contrast(-1)
    assert unidentified.metrics["target_covariance_dim"] == 1
    assert unidentified.metrics["target_covariance_rank_estimate"] == 0
    assert unidentified.status == CheckStatuses.FAILED


# ---------------------------------------------------------------------------
# 20. REGRESSION: an unevaluable distortion gate is never a pass. One non-identified target used
#     to disable the SE band for EVERY target (an all-or-nothing np.all(isfinite(ratios))), so a
#     healthy target's SE went uncompared and the check still reported PASSED.
# ---------------------------------------------------------------------------


def _one_dead_target_inputs(seed=17, n=50):
    """Two targets, the second with identically zero per-subject contributions -- hence zero
    sandwich SE, an unevaluable ratio, and a singular target covariance."""
    rng = np.random.default_rng(seed)
    stacks = np.zeros((n, 2))
    stacks[:, 0] = rng.standard_normal(n) * 3.0
    stacks[:, 0] -= stacks[:, 0].mean()
    M_hat = (stacks.T @ stacks) / n
    return stacks, np.eye(2), M_hat / n


def _run_two_target_bootstrap(g_tilde, stacks, B, V_hat, config):
    return d.check_multiplier_bootstrap(
        g_tilde,
        jnp.zeros(2),
        np.zeros(2),
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(2),
        ["live", "dead"],
        V_hat,
        config,
    )


def test_multiplier_bootstrap_is_indeterminate_when_a_target_cannot_be_compared():
    stacks, B, V_hat = _one_dead_target_inputs()
    result = _run_two_target_bootstrap(
        _affine_map(B), stacks, B, V_hat, d.DiagnosticConfig(num_bootstrap_draws=60)
    )
    assert result.status == CheckStatuses.INDETERMINATE
    assert result.metrics["se_ratio_band_unevaluable_targets"] == ["dead"]
    assert any("no identified target variance" in w for w in result.warnings)
    # The healthy target WAS compared and sits inside its band; the verdict is INDETERMINATE
    # because of the target that could not be compared at all, not because of this one.
    band_lo, band_hi = result.metrics["se_ratio_null_band"]
    assert band_lo < result.metrics["se_ratio_by_target"]["live"] < band_hi


def test_multiplier_bootstrap_still_fails_on_a_healthy_target_beside_a_dead_one():
    # The hole the all-or-nothing gate left open: an above-band ratio on the identified target
    # must produce FAILED even though a sibling target has no SE to compare against.
    stacks, B, V_hat = _one_dead_target_inputs()
    w = math.sqrt(float(V_hat[0, 0])) / 1.5

    def g_tilde(eta):
        return jnp.stack([w * jnp.arcsinh(eta[0] / w), eta[1]])

    result = _run_two_target_bootstrap(
        g_tilde,
        stacks,
        B,
        V_hat,
        d.DiagnosticConfig(num_bootstrap_draws=60, nonlinear_solver_max_iterations=200),
    )
    assert result.status == CheckStatuses.FAILED
    assert result.metrics["se_ratio_band_unevaluable_targets"] == ["dead"]
    assert (
        result.metrics["se_ratio_by_target"]["live"]
        > result.metrics["se_ratio_null_band"][1]
    )
    assert any("anticonservative" in w for w in result.warnings)


def test_exact_check_is_indeterminate_when_the_se_ratio_gate_cannot_be_evaluated():
    # A singular target covariance makes the generalized eigenproblem unsolvable (scipy's eigh
    # needs a positive-definite B), so no se_ratios exist to compare against the null band.
    # Nothing was measured, so nothing passed.
    stacks, B, V_hat = _one_dead_target_inputs()
    result = d.check_exact_nonlinear_perturbations(
        _affine_map(B),
        jnp.zeros(2),
        np.zeros(2),
        B,
        d.factor_bread(B),
        stacks,
        stacks.shape[0],
        np.eye(2),
        ["live", "dead"],
        V_hat,
        d.DiagnosticConfig(num_exact_directions=10),
    )
    assert result.status == CheckStatuses.INDETERMINATE
    assert result.metrics["se_ratios_within_tolerance"] is None
    assert all(math.isnan(ratio) for ratio in result.metrics["se_ratios"])
    assert result.metrics["root_failure_fraction"] == 0.0
    assert any("could not be evaluated" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# 21. check_jacobian_drift's memory bound. Its num_directions * len(drift_path_samples) reverse-
#     mode Jacobians of the whole stacked system are the most backward-pass-expensive thing in
#     the suite; chunking them must not change a single reported number.
# ---------------------------------------------------------------------------


def test_jacobian_drift_is_invariant_to_the_row_chunk_size():
    rng = np.random.default_rng(5)
    n, dim = 25, 3
    stacks = rng.standard_normal((n, dim)) * 5.0
    B = np.eye(dim)

    def g_tilde(eta):
        return jnp.asarray(B) @ eta + 0.5 * eta**2

    def rho_at(chunk_size):
        return d.check_jacobian_drift(
            g_tilde,
            jnp.zeros(dim),
            B,
            d.factor_bread(B),
            stacks,
            n,
            d.DiagnosticConfig(jacobian_row_chunk_size=chunk_size),
        ).metrics["rho_by_direction"]

    unchunked = rho_at(None)  # auto: below the auto policy's threshold, so unchunked
    # 0 forces unchunked, 1/2 do not divide out_dim = 3, 3 is exactly out_dim, 8 exceeds it.
    for chunk_size in (0, 1, 2, 3, 8):
        np.testing.assert_allclose(rho_at(chunk_size), unchunked, rtol=1e-7, atol=1e-9)


def test_jacobian_drift_rejects_a_negative_row_chunk_size():
    stacks, B = _drift_inputs()
    with pytest.raises(ValueError, match="non-negative"):
        d.check_jacobian_drift(
            _affine_map(B),
            jnp.zeros(1),
            B,
            d.factor_bread(B),
            stacks,
            stacks.shape[0],
            d.DiagnosticConfig(jacobian_row_chunk_size=-1),
        )


def test_diagnostic_config_resolves_a_missing_chunk_size_from_an_older_pickle():
    # DiagnosticConfig instances are pickled alongside cluster runs (analyze_dataset's
    # --diagnostic_config_pickle path), so a config written before this field existed must still
    # answer for it -- and still round-trip through dataclasses.asdict into tolerances_used.
    stale = pickle.loads(pickle.dumps(d.DiagnosticConfig()))
    stale.__dict__.pop("jacobian_row_chunk_size")
    assert stale.jacobian_row_chunk_size is None
    assert "jacobian_row_chunk_size" in dataclasses.asdict(stale)


# ---------------------------------------------------------------------------
# 9. DiagnosticReport.verdict -- the decision-level summary (certified / conservative /
#    uncertifiable / invalid). Unit tests pin the precedence ladder of _derive_verdict with
#    synthetic CheckResults; suite-level tests confirm the wiring end-to-end on an affine map.
# ---------------------------------------------------------------------------

from lifejacket.constants import DiagnosticVerdicts, VerdictBases  # noqa: E402


def _cr(name, status, metrics=None):
    return d.CheckResult(name=name, status=status, metrics=metrics or {})


def test_derive_verdict_precedence_ladder():
    config = d.DiagnosticConfig()
    passed_local = _cr("local_nonlinearity", CheckStatuses.PASSED)

    # Any FAILED check -> invalid.
    verdict, _ = d._derive_verdict(
        {
            "exploration_and_weights": _cr(
                "exploration_and_weights", CheckStatuses.FAILED
            )
        },
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.INVALID

    # Rank-deficient target covariance -> invalid even with every status clean (the
    # zero-width-CI collapse mode: bread_stability reads indeterminate at worst, but the rank
    # metric is the condition that identified 76/76 collapses in the undercoverage hunt).
    verdict, _ = d._derive_verdict(
        {
            "local_nonlinearity": passed_local,
            "bread_stability": _cr(
                "bread_stability",
                CheckStatuses.INDETERMINATE,
                {"target_covariance_rank_estimate": 1},
            ),
        },
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.INVALID

    # Indeterminate (fragility/censoring) without rank deficiency -> uncertifiable.
    verdict, _ = d._derive_verdict(
        {
            "local_nonlinearity": passed_local,
            "multiplier_bootstrap": _cr(
                "multiplier_bootstrap", CheckStatuses.INDETERMINATE
            ),
        },
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.UNCERTIFIABLE

    # An INDETERMINATE INPUT row (an input prerequisite that never ran, e.g. under
    # suppress_all_data_checks) -> uncertifiable even when every statistical check passed:
    # unvalidated inputs cannot certify. Before the rows were passed into the suite this
    # combination read CERTIFIED, exit 0.
    verdict, _ = d._derive_verdict(
        {"local_nonlinearity": passed_local},
        {
            "first_wave_input_checks": _cr(
                "first_wave_input_checks", CheckStatuses.INDETERMINATE
            )
        },
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.UNCERTIFIABLE

    # A PASSED input row changes nothing.
    verdict, _ = d._derive_verdict(
        {"local_nonlinearity": passed_local},
        {
            "first_wave_input_checks": _cr(
                "first_wave_input_checks", CheckStatuses.PASSED
            )
        },
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.CERTIFIED

    # The screen called for the bootstrap (local not PASSED) and it never ran -> uncertifiable:
    # no verdict layer means no verdict, not a pass.
    verdict, _ = d._derive_verdict(
        {"local_nonlinearity": _cr("local_nonlinearity", CheckStatuses.WARNING)},
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.UNCERTIFIABLE

    # Calibrated conservatism signals -> conservative (bootstrap below-band WARNING here;
    # influence WARNING is the other path).
    verdict, _ = d._derive_verdict(
        {
            "local_nonlinearity": passed_local,
            "multiplier_bootstrap": _cr("multiplier_bootstrap", CheckStatuses.WARNING),
        },
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.CONSERVATIVE

    verdict, _ = d._derive_verdict(
        {
            "local_nonlinearity": passed_local,
            "influence_concentration": _cr(
                "influence_concentration", CheckStatuses.WARNING
            ),
        },
        {},
        False,
        2,
        config,
    )
    assert verdict == DiagnosticVerdicts.CONSERVATIVE

    # Quiet run, no bootstrap needed -> certified via the screen.
    verdict, basis = d._derive_verdict(
        {"local_nonlinearity": passed_local}, {}, False, 2, config
    )
    assert (verdict, basis) == (DiagnosticVerdicts.CERTIFIED, VerdictBases.SCREEN)


def test_suite_verdict_certified_by_bootstrap_on_affine_map():
    rng = np.random.default_rng(31)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n

    report = d.run_diagnostic_suite(
        _affine_map(B),
        jnp.zeros(d_total),
        B,
        M_hat,
        M_hat / n,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(
            num_directions=10,
            multiplier_bootstrap="always",
            num_bootstrap_draws=40,
        ),
    )
    assert report.verdict == DiagnosticVerdicts.CERTIFIED
    assert report.verdict_basis == VerdictBases.BOOTSTRAP
    # classification is untouched by the verdict layer.
    assert report.classification == DiagnosticClassifications.LOCALLY_SUPPORTED


def test_suite_verdict_certified_by_screen_when_bootstrap_not_called_for():
    rng = np.random.default_rng(32)
    d_total, n = 2, 60
    B = np.eye(d_total)
    stacks = rng.standard_normal((n, d_total))
    stacks -= stacks.mean(axis=0)
    M_hat = (stacks.T @ stacks) / n

    report = d.run_diagnostic_suite(
        _affine_map(B),
        jnp.zeros(d_total),
        B,
        M_hat,
        M_hat / n,
        stacks,
        beta_dim=0,
        theta_dim=d_total,
        num_subjects=n,
        config=d.DiagnosticConfig(num_directions=10, multiplier_bootstrap="auto"),
    )
    assert "multiplier_bootstrap" not in report.check_results
    assert report.verdict == DiagnosticVerdicts.CERTIFIED
    assert report.verdict_basis == VerdictBases.SCREEN


# ---------------------------------------------------------------------------
# The two re-solve optimizations that ship ON by default (divergence abort,
# starvation early stop). Both are verdict-preservation claims, and the whole
# point of each is that it changes cost and NOTHING else -- so every test here
# asserts equality against the same run with the optimization disabled, rather
# than asserting any particular status. Adversarial review found these shipped
# with zero coverage; a regression in either is invisible without these.
# ---------------------------------------------------------------------------


def _abort_flag_configs(**overrides):
    """The same config with the abort on and off, so a test can diff the two runs."""
    on = d.DiagnosticConfig(nonlinear_solver_divergence_abort=True, **overrides)
    off = d.DiagnosticConfig(nonlinear_solver_divergence_abort=False, **overrides)
    return on, off


def test_divergence_abort_does_not_reclassify_converging_solves():
    # The abort's ONLY licence is that it fires on solves that were going to fail anyway. If it
    # ever turns a converged trial into a failed one it moves the failure fraction, which moves
    # the status -- so the converged COUNT is the thing to pin, on a map curved enough that a
    # good fraction of solves genuinely struggle.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]

    def g_tilde(eta):
        return jnp.asarray(B) @ (eta + 0.6 * eta**2)

    def run(config):
        return d.check_multiplier_bootstrap(
            g_tilde,
            jnp.zeros(1),
            np.zeros(1),
            d.factor_bread(B),
            stacks,
            n,
            np.eye(1),
            ["theta_0"],
            V_hat,
            config,
        )

    on_config, off_config = _abort_flag_configs(
        num_bootstrap_draws=40, nonlinear_solver_max_iterations=60
    )
    with_abort, without_abort = run(on_config), run(off_config)

    # An abort can only ever turn converged into failed, so an unchanged count is proof of zero
    # false aborts rather than mere evidence of them.
    assert (
        with_abort.metrics["num_converged_trials"]
        == without_abort.metrics["num_converged_trials"]
    )
    assert with_abort.status == without_abort.status
    # And the ensemble statistics the verdict is actually read off are untouched. Keyed by
    # target label, so compare per label rather than as an array.
    with_ratios = with_abort.metrics["se_ratio_by_target"]
    without_ratios = without_abort.metrics["se_ratio_by_target"]
    assert set(with_ratios) == set(without_ratios)
    for label, ratio in with_ratios.items():
        np.testing.assert_allclose(ratio, without_ratios[label])


def test_divergence_abort_is_reported_and_can_be_switched_off():
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]

    def g_tilde(eta):
        return jnp.asarray(B) @ jnp.tanh(eta * 8.0)

    def run(config):
        return d.check_multiplier_bootstrap(
            g_tilde,
            jnp.zeros(1),
            np.zeros(1),
            d.factor_bread(B),
            stacks,
            n,
            np.eye(1),
            ["theta_0"],
            V_hat,
            config,
        )

    on_config, off_config = _abort_flag_configs(
        num_bootstrap_draws=30, nonlinear_solver_max_iterations=50
    )
    with_abort, without_abort = run(on_config), run(off_config)

    # The count is reported either way, and the escape hatch really does disable the mechanism.
    assert "num_divergence_aborted_trials" in with_abort.metrics
    assert without_abort.metrics["num_divergence_aborted_trials"] == 0
    assert with_abort.metrics["num_divergence_aborted_trials"] >= 0


def test_starvation_early_stop_preserves_status_and_reports_truncation():
    # The stop may only fire once the maximum still-attainable converged count has fallen below
    # the three trials any ensemble statistic needs. On a map where nothing converges it should
    # fire, report the truncation honestly, and reach the same status as the untruncated run.
    stacks, B, V_hat = _scalar_bootstrap_inputs()
    n = stacks.shape[0]

    def g_tilde(eta):
        # Wildly oscillatory: the chord solve cannot land anywhere.
        return jnp.asarray(B) @ jnp.sin(eta * 400.0)

    def run(mode):
        return d.check_multiplier_bootstrap(
            g_tilde,
            jnp.zeros(1),
            np.zeros(1),
            d.factor_bread(B),
            stacks,
            n,
            np.eye(1),
            ["theta_0"],
            V_hat,
            d.DiagnosticConfig(
                num_bootstrap_draws=20,
                nonlinear_solver_max_iterations=8,
                perturbation_early_stop=mode,
            ),
        )

    stopped, full = run("starvation"), run("off")

    assert stopped.status == full.status
    assert stopped.metrics["num_planned_trials"] == full.metrics["num_planned_trials"]
    if stopped.metrics["early_stopped"]:
        # A truncated ensemble must never masquerade as a complete one.
        assert stopped.metrics["num_trials"] < stopped.metrics["num_planned_trials"]
        assert stopped.metrics["early_stop_reason"] == d.EARLY_STOP_REASON_STARVATION
        assert stopped.metrics["num_draws_executed"] <= 20
        # It can only ever skip the last MIN_CONVERGED_TRIALS_FOR_ENSEMBLE - 1 trials.
        skipped = stopped.metrics["num_planned_trials"] - stopped.metrics["num_trials"]
        assert skipped <= d.MIN_CONVERGED_TRIALS_FOR_ENSEMBLE - 1


def test_batched_bootstrap_resolves_auto_is_refused():
    # Batching is known to change verdicts on reachable fixtures, so "auto" must never select
    # it -- only an explicit "always" may reach the driver.
    config = d.DiagnosticConfig(batched_bootstrap_resolves="auto")
    assert d._resolve_bootstrap_batch_width(config, num_planned_trials=1000) == 0
    always = d.DiagnosticConfig(batched_bootstrap_resolves="always")
    assert d._resolve_bootstrap_batch_width(always, num_planned_trials=1000) > 0
    off = d.DiagnosticConfig(batched_bootstrap_resolves="off")
    assert d._resolve_bootstrap_batch_width(off, num_planned_trials=1000) == 0


# ---------------------------------------------------------------------------
# 10. format_diagnostic_summary / diagnostics_flagged -- the end-of-run surface: the one block
#     a reader who scrolled straight to the bottom must be able to act on, and the single
#     definition of "flagged" the CLI's exit status and the consent gate both consume.
# ---------------------------------------------------------------------------


def _summary_report(**overrides):
    fields = dict(
        classification=DiagnosticClassifications.FAILED,
        check_results={
            "bread_stability": d.CheckResult(
                name="bread_stability",
                status=CheckStatuses.FAILED,
                warnings=[
                    "Target SEs changed by 99.99% under a numerically negligible "
                    "(1.0e-06 relative) perturbation of B_hat -- this indicates numerical "
                    "fragility distinct from statistical identification.",
                    "Second flag.",
                    "Third flag.",
                ],
            ),
            "influence_concentration": d.CheckResult(
                name="influence_concentration",
                status=CheckStatuses.PASSED,
                criteria=[
                    d.CriterionResult(
                        description="effective sample at least 2.0 subjects",
                        value="worst 5.0 of 20",
                        ok=True,
                        severity="warn",
                    ),
                    d.CriterionResult(
                        description=(
                            "a deliberately very long criterion description that cannot "
                            "possibly fit on a single detail line together with its value, "
                            "so its marker must be re-aligned after wrapping"
                        ),
                        value="largest 39.8%",
                        ok=False,
                        severity="warn",
                    ),
                    d.CriterionResult(
                        description="a criterion its data could not evaluate",
                        value="not evaluable",
                        ok=None,
                    ),
                    d.CriterionResult(
                        description="a measured quantity with no calibrated threshold",
                        value="85% in the worst direction",
                        ok=None,
                        severity="info",
                    ),
                    d.CriterionResult(
                        description="a criterion whose measured value is itself long",
                        value=(
                            "a measured value long enough that wrapping must split it "
                            "across two detail lines"
                        ),
                        ok=True,
                    ),
                    d.CriterionResult(
                        description="a hard criterion",
                        value="0.5",
                        ok=False,
                    ),
                    d.CriterionResult(
                        description="a soft identifiability criterion",
                        value="3 of 4",
                        ok=False,
                        severity="indeterminate",
                    ),
                ],
            ),
        },
        input_check_results={
            "action_probabilities_reconstructed": d.CheckResult(
                name="action_probabilities_reconstructed",
                status=CheckStatuses.PASSED,
            ),
        },
        metrics={},
        tolerances_used={},
        warnings=[],
        monte_carlo_counts={},
        target_labels=[],
        rank_diagnostics={},
        verdict=DiagnosticVerdicts.INVALID,
        verdict_basis="",
    )
    fields.update(overrides)
    return d.DiagnosticReport(**fields)


def test_format_diagnostic_summary_lists_every_check_and_the_verdict():
    summary = d.format_diagnostic_summary(
        _summary_report(),
        pipeline_rows=[
            (
                "joint_bread_condition_number",
                CheckStatuses.FAILED,
                "cond = 1.473e+13 > 1e+12",
            )
        ],
    )
    for expected in (
        "action_probabilities_reconstructed",
        "bread_stability",
        "influence_concentration",
        "joint_bread_condition_number",
        "FAILED",
        "PASSED",
        "messages from the check:",
        # RETARGETED 2026-09-02: previously pinned "(+2 more flags)". Every message is now
        # listed in full -- on a failing run the elided ones were exactly the explanation for a
        # status the gate line could not account for.
        "- Second flag.",
        "- Third flag.",
        "VERDICT: INVALID -- DO NOT REPORT the adjusted sandwich variance",
    ):
        assert expected in summary
    # INVERTED on 2026-09-02: long detail texts are now WRAPPED into the detail column rather
    # than truncated there. These messages end in the specific numbers and target names that
    # say what actually went wrong, so cutting at a fixed width reliably discarded the
    # actionable half and left the boilerplate. Every line still stays terminal-width readable.
    assert all(len(line) <= 130 for line in summary.splitlines())
    # The tail of the fixture's long warning -- the first thing truncation used to eat.
    # Checked against whitespace-normalized text, since wrapping may put a line break anywhere
    # inside the phrase.
    flattened_summary = " ".join(summary.split())
    assert "distinct from statistical identification." in flattened_summary
    # Wrapped remainder is indented to the detail column, so the name/status columns stay
    # scannable instead of continuation text starting at column 0.
    continuation_lines = [
        line
        for line in summary.splitlines()
        if line.strip() and line.startswith(" " * d._SUMMARY_DETAIL_INDENT)
    ]
    assert continuation_lines
    # The rules span the full row rather than a narrower fixed width, so no row hangs past the
    # end of its own box.
    rule_lengths = {
        len(line) for line in summary.splitlines() if set(line) in ({"="}, {"-"})
    }
    assert rule_lengths == {d._SUMMARY_TOTAL_WIDTH}
    assert max(len(line) for line in summary.splitlines()) <= d._SUMMARY_TOTAL_WIDTH
    # The verdict is the last substantive line, so it cannot scroll away above a wall of rows.
    assert "VERDICT:" in summary.splitlines()[-2]


def test_format_diagnostic_summary_includes_a_status_and_verdict_key():
    # The summary is read by people who did not run the suite and will not go looking for
    # CheckStatuses or DiagnosticVerdicts, so every status and verdict it can print is spelled
    # out in the block itself.
    summary = d.format_diagnostic_summary(_summary_report())

    assert "CHECK STATUS KEY" in summary
    assert "VERDICT KEY" in summary
    for status, meaning in d._STATUS_LEGEND.items():
        assert status.upper() in summary
        assert meaning in summary
    # Sourced from _VERDICT_ACTION_PHRASES rather than restated, so the key cannot drift from
    # the action phrase the verdict line itself prints.
    for verdict, action in d._VERDICT_ACTION_PHRASES.items():
        # The DISPLAYED label, not the stored value: NOT_CERTIFIED renders as "NOT CERTIFIED"
        # while the wire value keeps its underscore for pickled reports and downstream
        # comparisons.
        assert d._verdict_label(verdict) in summary
        assert action in summary

    # ORDER, retargeted 2026-09-02: the CHECKS KEY now leads the legend (what the rows are must
    # come before what their columns mean), so the first bread_stability occurrence is its
    # checks-key description, ABOVE the status key. The whole legend still sits above the rows,
    # and the verdict stays the last substantive line.
    assert summary.index("CHECKS KEY") < summary.index("CHECK STATUS KEY")
    assert summary.index("CHECK STATUS KEY") < summary.index("VERDICT KEY")
    # The units note and the docs pointer travel with the checks key.
    assert "fractions of the reported standard error" in summary
    assert "docs/diagnostics.md" in summary
    assert "VERDICT:" in summary.splitlines()[-2]


def test_format_diagnostic_summary_colors_are_opt_in_and_atomic():
    """
    color=True wraps each status/verdict token in its mapped ANSI color; color=False (and the
    non-tty auto default under pytest) emits none. "NOT CERTIFIED" is pinned as ONE orange
    token: a shorter-token-first bug would render an uncolored NOT beside a green CERTIFIED,
    inverting the meaning at a glance.
    """
    report = _summary_report(verdict=DiagnosticVerdicts.NOT_CERTIFIED)
    plain = d.format_diagnostic_summary(report, color=False)
    assert "\x1b[" not in plain
    # The auto default under pytest (captured stdout, not a tty) is also plain.
    assert "\x1b[" not in d.format_diagnostic_summary(report)

    colored = d.format_diagnostic_summary(report, color=True)
    assert "\x1b[32mPASSED\x1b[0m" in colored  # status green
    assert "\x1b[31mFAILED\x1b[0m" in colored  # status red
    assert "\x1b[38;5;141mINDETERMINATE\x1b[0m" in colored  # status purple
    assert "\x1b[38;5;208mNOT CERTIFIED\x1b[0m" in colored  # verdict orange, one unit
    assert "\x1b[31mDO NOT REPORT\x1b[0m" in colored  # the instruction itself, red
    assert "NOT \x1b" not in colored  # never a bare NOT beside a colored CERTIFIED
    # Per-criterion outcome markers are individually colored, keyed to their own ok/severity
    # (the fixture's influence row carries one of each), not to the row's overall status.
    assert "\x1b[32m[ok]\x1b[0m" in colored
    assert "\x1b[31m[fail]\x1b[0m" in colored
    assert "\x1b[38;5;208m[warn]\x1b[0m" in colored
    assert "\x1b[38;5;141m[indeterminate]\x1b[0m" in colored
    assert "\x1b[38;5;141m[not evaluated]\x1b[0m" in colored
    assert "\x1b[90m[no gate]\x1b[0m" in colored
    # Each criterion's measured VALUE is cyan -- on every line it touches when it wraps, so a
    # value split across two lines carries its own start/reset on each (an unterminated color
    # would bleed into the padding and the next line's text).
    assert "\x1b[36mworst 5.0 of 20\x1b[0m" in colored
    colored_lines = colored.splitlines()
    first_half_index = next(
        index
        for index, line in enumerate(colored_lines)
        if "whose measured value is itself long" in line
    )
    assert colored_lines[first_half_index].count("\x1b[36m") == 1
    assert colored_lines[first_half_index].rstrip().endswith("\x1b[0m")
    second_half = colored_lines[first_half_index + 1]
    assert second_half.lstrip().startswith("\x1b[36m")
    assert "detail lines\x1b[0m" in second_half
    # Stripping the codes recovers the plain block exactly -- color is presentation only.
    import re as _re

    assert _re.sub("\x1b\\[[0-9;]*m", "", colored) == plain


def test_format_diagnostic_summary_right_aligns_criterion_markers():
    """
    Each criterion renders as "- description: value" with its outcome marker flush against the
    summary's right edge -- ON THE CRITERION'S LAST LINE even when the text wraps, so the
    markers form one scannable column. Severity maps ok=False to [fail]/[warn]/[indeterminate]
    and ok=None to [not evaluated], independent of the row's overall status.
    """
    summary = d.format_diagnostic_summary(_summary_report())
    lines = summary.splitlines()
    assert "criteria:" in summary

    def line_with(marker, fragment=""):
        (line,) = [line for line in lines if line.endswith(marker) and fragment in line]
        return line

    # Marker-terminated lines all end at the same right edge, with a dot leader running from
    # the value to the marker so the eye can track across the gap.
    import re as _re

    for marker, fragment in [
        ("[ok]", "worst 5.0 of 20"),
        ("[warn]", ""),
        ("[not evaluated]", "not evaluable"),
        ("[fail]", "0.5"),
        ("[indeterminate]", "3 of 4"),
        ("[no gate]", ""),
    ]:
        line = line_with(marker, fragment)
        assert len(line) == d._SUMMARY_TOTAL_WIDTH
        assert _re.search(r" \.{2,} \[", line), line
    # The deliberately over-long criterion wrapped: its description starts on one line and its
    # marker lands on a LATER line (the value's), still flush right.
    start_index = next(
        index
        for index, line in enumerate(lines)
        if "deliberately very long criterion" in line
    )
    assert not lines[start_index].endswith("[warn]")
    marker_index = next(
        index
        for index, line in enumerate(lines[start_index:], start=start_index)
        if line.endswith("[warn]")
    )
    assert marker_index > start_index
    assert "largest 39.8%" in lines[marker_index]

    # The other wrapping fallback: when the value fills its line, the marker cannot fit after
    # it and takes a leader line of its own -- still flush right at the same column.
    own_line = line_with("[no gate]")
    assert own_line.strip().startswith(".")
    assert len(own_line) == d._SUMMARY_TOTAL_WIDTH
    value_line_index = next(
        index
        for index, line in enumerate(lines)
        if "a measured quantity with no calibrated threshold" in line
    )
    assert "85% in the worst direction" in lines[value_line_index]
    assert lines[value_line_index + 1] == own_line


def test_format_diagnostic_summary_with_no_report_renders_did_not_run():
    summary = d.format_diagnostic_summary(None, suite_error="ValueError: boom")
    assert "DID NOT RUN" in summary
    assert "ValueError: boom" in summary
    assert "VERDICT: UNAVAILABLE" in summary


def test_diagnostics_flagged_definition():
    # Verdict-first: only the two do-not-report verdicts flag.
    assert d.diagnostics_flagged(_summary_report(verdict=DiagnosticVerdicts.INVALID))
    assert d.diagnostics_flagged(
        _summary_report(verdict=DiagnosticVerdicts.UNCERTIFIABLE)
    )
    assert not d.diagnostics_flagged(
        _summary_report(verdict=DiagnosticVerdicts.CERTIFIED)
    )
    assert not d.diagnostics_flagged(
        _summary_report(verdict=DiagnosticVerdicts.CONSERVATIVE)
    )
    # Pre-verdict reports (empty string) fall back to classification.
    assert d.diagnostics_flagged(
        _summary_report(verdict="", classification=DiagnosticClassifications.FAILED)
    )
    assert not d.diagnostics_flagged(
        _summary_report(
            verdict="", classification=DiagnosticClassifications.LOCALLY_SUPPORTED
        )
    )
    # No report at all (the suite crashed): an unevaluated run must read as flagged.
    assert d.diagnostics_flagged(None)
