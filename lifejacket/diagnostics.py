from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
import scipy.linalg
from jax import numpy as jnp

from .constants import CheckStatuses, DiagnosticClassifications
from .helper_functions import get_radon_nikodym_weight, matrix_inv_sqrt
from .simulator_calibration import clopper_pearson_upper_bound

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# NOTE ON SCALING CONVENTION: everywhere below, `joint_sandwich_matrix` (and any
# theta-only slice of it) is taken to already equal B_hat^{-1} M_hat B_hat^{-T} / n, i.e. it IS
# Cov(eta_hat) directly (this is what post_deployment_analysis.form_sandwich_from_bread_and_meat
# returns). Contrast standard errors are therefore sqrt(l @ V @ l) with NO further division by
# n. The perturbation sampler below draws u_j ~ N(0, M_hat), s_j = u_j/sqrt(n), delta_j =
# B_hat^{-1} s_j, so that Cov_j(delta_j) == joint_sandwich_matrix exactly: the sandwich-scale
# perturbations are literally simulated draws of eta_hat's own sampling fluctuation. Any "V_L"-
# style generalized eigenvalue or Mahalanobis computation below is therefore also n-factor-free
# (the two matrices in a generalized eigenvalue problem carry the same missing factor of n and
# it cancels).


###############################################################################
# Configuration and report objects
###############################################################################


@dataclasses.dataclass
class DiagnosticConfig:
    """
    Configuration for the layered diagnostic suite in this module. All tolerances are
    engineering tolerances, not theorem-derived critical values, unless a particular deployment
    has calibrated them via lifejacket.simulator_calibration against its own simulator.
    """

    random_seed: int = 0
    num_directions: int = 15
    paired_directions: bool = True
    perturbation_radii: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5)

    # Target selection. If contrast_matrix is None, defaults to the identity on the theta block
    # (one contrast per theta component) when the suite is run.
    contrast_matrix: np.ndarray | None = None
    target_labels: list[str] | None = None

    root_error_tolerance_se: float = 0.01
    nonlinear_correction_tolerance_se: float = 0.10
    se_distortion_tolerance: float = 0.05
    mean_shift_tolerance_se: float = 0.10
    quantile_shift_tolerance_se: float = 0.10
    bad_direction_probability_target: float = 0.01
    confidence_level: float = 0.95

    rank_tolerance: float = 1e-8

    continuation_steps: int = 10
    nonlinear_solver_max_iterations: int = 50
    # The repository deliberately does not enable JAX's float64 mode (see git history around
    # "ADS-138-remove-float64-default": float64 was tried and explicitly reverted, presumably
    # for memory/performance reasons at scale), so g_tilde evaluations are float32 in practice
    # regardless of any float64 casts applied to the *linear algebra* around them. A tolerance
    # tighter than ~1e-5 relative is generally unreachable at float32 precision.
    nonlinear_solver_tolerance: float = 1e-5
    compute_exact_nonlinear_roots: bool = False
    num_exact_directions: int | None = None

    compute_influence_and_overlap_checks: bool = True
    compute_leave_one_out_sensitivity: bool = False
    leave_one_out_top_k: int = 3
    report_subject_identifiers: bool = True

    drift_num_directions: int = 3
    drift_path_samples: tuple[float, ...] = (0.0, 0.5, 1.0)

    exploration_floor: float | None = None
    exploration_ceiling: float | None = None

    finite_difference_num_directions: int = 3
    # Chosen for float32 precision: the rounding-error/truncation-error tradeoff for a central
    # difference is minimized near eps**(1/3) ~ 5e-3 at float32 machine epsilon, not at the much
    # smaller steps that would be appropriate for float64.
    finite_difference_step: float = 1e-3
    # Must be well above float32 machine epsilon (~1.2e-7) to actually perturb anything; 1e-10
    # would silently round away to zero and this check would be a no-op.
    bread_perturbation_relative_scale: float = 1e-6

    g_tilde_chunk_size: int = 1


@dataclasses.dataclass
class CheckResult:
    name: str
    status: str
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)
    message: str = ""


@dataclasses.dataclass
class DiagnosticReport:
    classification: str
    check_results: dict[str, CheckResult]
    metrics: dict[str, Any]
    tolerances_used: dict[str, Any]
    warnings: list[str]
    monte_carlo_counts: dict[str, int]
    target_labels: list[str]
    rank_diagnostics: dict[str, Any]


###############################################################################
# Small numerical primitives shared by every check. B_hat is factored exactly once and never
# explicitly inverted; every downstream solve reuses that factorization.
###############################################################################


def factor_bread(B_hat: np.ndarray):
    """LU-factor the joint bread matrix once, for reuse by every stable solve below."""
    return scipy.linalg.lu_factor(np.asarray(B_hat, dtype=np.float64))


def solve_with_bread(bread_factored, rhs: np.ndarray) -> np.ndarray:
    """Solve B_hat @ x = rhs (rhs may be a vector or a matrix of columns/rows to solve for)."""
    return scipy.linalg.lu_solve(bread_factored, rhs)


def solve_with_bread_transpose(bread_factored, rhs: np.ndarray) -> np.ndarray:
    """Solve B_hat^T @ x = rhs."""
    return scipy.linalg.lu_solve(bread_factored, rhs, trans=1)


def default_contrast_matrix(
    beta_total_dim: int, theta_dim: int, config: DiagnosticConfig
) -> tuple[np.ndarray, list[str]]:
    """
    Builds the target selector L. Defaults to the identity on the theta block (one scalar
    contrast per theta component), which is the inferential target reported elsewhere in the
    package (see adjusted_sandwich_var_estimate in post_deployment_analysis.py).
    """
    if config.contrast_matrix is not None:
        L = np.atleast_2d(np.asarray(config.contrast_matrix, dtype=np.float64))
        labels = config.target_labels or [f"contrast_{i}" for i in range(L.shape[0])]
        return L, list(labels)

    d_total = beta_total_dim + theta_dim
    L = np.zeros((theta_dim, d_total))
    L[:, beta_total_dim:] = np.eye(theta_dim)
    labels = config.target_labels or [f"theta_{i}" for i in range(theta_dim)]
    return L, list(labels)


def standard_errors_for_contrasts(V_hat: np.ndarray, L: np.ndarray) -> np.ndarray:
    """se(l^T eta_hat) for each row l of L, given V_hat == Cov(eta_hat) (see module note)."""
    variances = np.einsum("ij,jk,ik->i", L, np.asarray(V_hat), L)
    return np.sqrt(np.clip(variances, 0.0, None))


def evaluate_g_tilde_batched(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    deltas: np.ndarray,
    chunk_size: int = 1,
) -> np.ndarray:
    """Evaluates g_tilde(eta_hat + delta) for every row of deltas, chunked to bound memory."""
    # Deliberately not requesting jnp.float64 here: the repository does not enable JAX's x64
    # mode (see the note on DiagnosticConfig.nonlinear_solver_tolerance), so such a request
    # would silently truncate back to float32 anyway, with a noisy warning in the meantime.
    deltas = jnp.asarray(deltas)
    eta_hat = jnp.asarray(eta_hat)
    num_rows = deltas.shape[0]
    outputs = []
    for start in range(0, num_rows, max(1, chunk_size)):
        chunk = deltas[start : start + chunk_size]
        outputs.append(jax.vmap(lambda d: g_tilde(eta_hat + d))(chunk))
    return np.asarray(jnp.concatenate(outputs, axis=0))


###############################################################################
# Section 2: implementation and root accuracy
###############################################################################


def check_root_and_implementation(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    B_hat: np.ndarray,
    M_hat: np.ndarray,
    bread_factored,
    L: np.ndarray,
    target_labels: list[str],
    V_hat: np.ndarray,
    config: DiagnosticConfig,
    *,
    legacy_check_callables: Sequence[tuple[str, Callable[[], Any]]] = (),
    rng: np.random.Generator | None = None,
) -> CheckResult:
    """
    Section 2 (hard prerequisite gate). Computes the root-error correction in target standard-
    error units, checks finiteness of g0/B_hat/M_hat, checks the backward residual of the linear
    solve used to compute that correction, spot-checks the automatic derivative B_hat against a
    directional finite difference of g_tilde, and re-runs any supplied legacy input_checks
    functions, converting their hard raises into a failed CheckResult rather than propagating.
    """
    warnings: list[str] = []
    metrics: dict[str, Any] = {}
    status = CheckStatuses.PASSED
    failure_reasons: list[str] = []

    g0 = np.asarray(g0, dtype=np.float64)
    B = np.asarray(B_hat, dtype=np.float64)
    M = np.asarray(M_hat, dtype=np.float64)

    finite_ok = bool(
        np.all(np.isfinite(g0)) and np.all(np.isfinite(B)) and np.all(np.isfinite(M))
    )
    metrics["finite_inputs"] = finite_ok
    if not finite_ok:
        status = CheckStatuses.FAILED
        failure_reasons.append(
            "g_tilde(eta_hat), B_hat, or M_hat contains nonfinite values."
        )

    if finite_ok:
        d_root = solve_with_bread(bread_factored, -g0)
        backward_residual = B @ d_root + g0
        backward_scale = np.linalg.norm(B) * np.linalg.norm(d_root) + 1e-300
        backward_relative_residual = float(
            np.linalg.norm(backward_residual) / backward_scale
        )
        metrics["backward_relative_residual"] = backward_relative_residual

        se_l = standard_errors_for_contrasts(V_hat, L)
        identified = se_l > 0
        # A contrast with (numerically) zero target variance is a weak-identification/rank
        # issue for the bread-stability and local-nonlinearity checks to flag as indeterminate,
        # not a root-solving failure -- excluded here rather than forced to +inf, which would
        # otherwise turn any rank deficiency into a spurious hard FAILED regardless of whether
        # the root residual itself is actually small.
        a_root = np.full(L.shape[0], math.nan)
        a_root[identified] = np.abs((L @ d_root)[identified]) / se_l[identified]
        metrics["a_root_by_target"] = dict(
            zip(target_labels, a_root.tolist(), strict=False)
        )
        metrics["a_root_max"] = (
            float(np.max(a_root[identified])) if np.any(identified) else 0.0
        )
        if not np.all(identified):
            warnings.append(
                "One or more contrasts have (numerically) zero target variance and were "
                "excluded from the root-error check; see bread_stability/local_nonlinearity "
                "for the corresponding rank diagnostics."
            )

        if metrics["a_root_max"] > config.root_error_tolerance_se:
            status = CheckStatuses.FAILED
            failure_reasons.append(
                f"Root correction of {metrics['a_root_max']:.4g} SE exceeds tolerance "
                f"{config.root_error_tolerance_se}."
            )

        rng = rng or np.random.default_rng(config.random_seed)
        d_total = B.shape[0]
        num_fd_directions = min(config.finite_difference_num_directions, d_total) or 1
        h = config.finite_difference_step
        fd_relative_errors = []
        for _ in range(num_fd_directions):
            v = rng.normal(size=d_total)
            v = v / (np.linalg.norm(v) + 1e-300)
            g_plus = np.asarray(g_tilde(jnp.asarray(eta_hat) + h * jnp.asarray(v)))
            g_minus = np.asarray(g_tilde(jnp.asarray(eta_hat) - h * jnp.asarray(v)))
            central_diff = (g_plus - g_minus) / (2 * h)
            analytic = B @ v
            denom = np.linalg.norm(analytic) + 1e-300
            fd_relative_errors.append(
                float(np.linalg.norm(central_diff - analytic) / denom)
            )
        metrics["finite_difference_relative_errors"] = fd_relative_errors
        metrics["finite_difference_max_relative_error"] = (
            float(max(fd_relative_errors)) if fd_relative_errors else 0.0
        )
        # A generous tolerance: this is meant to catch a wrong/broken derivative, not to certify
        # numerical precision at the last bit.
        if metrics["finite_difference_max_relative_error"] > 1e-2:
            warnings.append(
                "Directional finite-difference check disagrees with B_hat by more than 1% in "
                "relative terms for at least one sampled direction."
            )
            if status == CheckStatuses.PASSED:
                status = CheckStatuses.WARNING

    for check_name, check_callable in legacy_check_callables:
        try:
            check_callable()
            metrics[f"legacy_check::{check_name}"] = "passed"
        except Exception as exc:  # noqa: BLE001 - legacy checks raise assorted exception types
            metrics[f"legacy_check::{check_name}"] = f"failed: {exc}"
            status = CheckStatuses.FAILED
            failure_reasons.append(f"Legacy check '{check_name}' failed: {exc}")

    return CheckResult(
        name="root_and_implementation",
        status=status,
        metrics=metrics,
        warnings=warnings,
        message="; ".join(failure_reasons),
    )


###############################################################################
# Sections 3 & 5: r_j, c_j, and the target-standardized nonlinearity diagnostic a_{j,l}
###############################################################################


def sample_perturbation_directions(
    per_subject_stacks: np.ndarray,
    bread_factored,
    num_subjects: int,
    num_directions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Draws u_j ~ N(0, M_hat) via u_j = (W @ stacks)/sqrt(n) with W ~ N(0, I_n) (so that Cov(u_j)
    is the empirical per-subject-stack covariance, i.e. M_hat), then s_j = u_j/sqrt(n) and
    delta_j = B_hat^{-1} s_j at unit radius. Returns (delta_j, s_j), each (num_directions, d).
    """
    key = jax.random.PRNGKey(seed)
    stacks_jax = jnp.asarray(per_subject_stacks)
    W = jax.random.normal(key, shape=(num_directions, num_subjects))
    U = (W @ stacks_jax) / jnp.sqrt(num_subjects)
    S = np.asarray(U / jnp.sqrt(num_subjects), dtype=np.float64)
    delta = solve_with_bread(bread_factored, S.T).T
    return delta, S


def evaluate_taylor_remainder_and_correction(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    B_hat: np.ndarray,
    bread_factored,
    deltas: np.ndarray,
    chunk_size: int = 1,
) -> dict[str, np.ndarray]:
    """
    For each row delta_j of `deltas`, computes:
      R_j = g_tilde(eta_hat + delta_j) - g0 - B_hat @ delta_j
      r_j = ||R_j|| / ||B_hat @ delta_j||               (retained equation-space diagnostic)
      c_j solving B_hat @ c_j = -R_j                     (first nonlinear parameter correction)
      q_j = ||c_j|| / ||delta_j||                        (secondary; no universal threshold)
    """
    B = np.asarray(B_hat, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    g_plus = evaluate_g_tilde_batched(g_tilde, eta_hat, deltas, chunk_size=chunk_size)
    B_delta = (B @ deltas.T).T
    R = g_plus - g0 - B_delta

    r = np.divide(
        np.linalg.norm(R, axis=1),
        np.linalg.norm(B_delta, axis=1),
        out=np.full(R.shape[0], np.inf),
        where=np.linalg.norm(B_delta, axis=1) > 0,
    )

    c = solve_with_bread(bread_factored, -R.T).T
    delta_norms = np.linalg.norm(deltas, axis=1)
    q = np.divide(
        np.linalg.norm(c, axis=1),
        delta_norms,
        out=np.full(c.shape[0], np.inf),
        where=delta_norms > 0,
    )

    return {"R": R, "r": r, "c": c, "q": q, "g_plus": g_plus, "B_delta": B_delta}


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"median": math.nan, "p90": math.nan, "p95": math.nan, "max": math.nan}
    return {
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.9)),
        "p95": float(np.quantile(finite, 0.95)),
        "max": float(np.max(finite)),
    }


def joint_mahalanobis_correction(
    c_matrix: np.ndarray,
    L: np.ndarray,
    V_L: np.ndarray,
    rank_tolerance: float,
) -> dict[str, Any]:
    """
    a_{j,L} = {(L c_j)^T V_L^+ (L c_j)}^{1/2} for each row c_j of c_matrix, on the identified
    subspace of V_L (an eigenvalue-floor pseudoinverse, used only when V_L is rank-deficient
    relative to rank_tolerance). Also reports the effective rank so an unresolved rank
    deficiency (fewer identified directions than L has rows) can be classified separately from
    a genuine measured distortion.
    """
    eigvals, eigvecs = np.linalg.eigh(np.asarray(V_L, dtype=np.float64))
    max_eig = float(eigvals.max()) if eigvals.size else 0.0
    keep = (
        eigvals > rank_tolerance * max_eig
        if max_eig > 0
        else np.zeros_like(eigvals, dtype=bool)
    )
    effective_rank = int(np.sum(keep))
    inv_sqrt_eigvals = np.zeros_like(eigvals)
    inv_sqrt_eigvals[keep] = 1.0 / np.sqrt(eigvals[keep])
    V_L_pinv_sqrt = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

    L_c = c_matrix @ np.asarray(L).T
    values = np.linalg.norm(L_c @ V_L_pinv_sqrt.T, axis=1)
    return {
        "values": values,
        "effective_rank": effective_rank,
        "target_dim": L.shape[0],
        "rank_deficient": effective_rank < L.shape[0],
    }


def check_local_nonlinearity(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    B_hat: np.ndarray,
    bread_factored,
    per_subject_stacks: np.ndarray,
    num_subjects: int,
    L: np.ndarray,
    target_labels: list[str],
    V_hat: np.ndarray,
    config: DiagnosticConfig,
) -> CheckResult:
    """
    Section 3/5: samples perturbation directions at the configured radii (optionally paired
    +/-delta_j), computes r_j/c_j/a_{j,l} at each radius, and reports radius-scaling and +/-
    symmetry as warnings (never as a pass/fail cutoff, per the task write-up).
    """
    se_l = standard_errors_for_contrasts(V_hat, L)
    base_delta, base_s = sample_perturbation_directions(
        per_subject_stacks,
        bread_factored,
        num_subjects,
        config.num_directions,
        config.random_seed,
    )

    signs = [1.0, -1.0] if config.paired_directions else [1.0]
    per_radius: dict[float, dict[str, Any]] = {}
    a_by_target_radius: dict[str, dict[float, np.ndarray]] = {
        label: {} for label in target_labels
    }
    r_by_radius: dict[float, np.ndarray] = {}
    q_by_radius: dict[float, np.ndarray] = {}

    for radius in config.perturbation_radii:
        sign_results = []
        for sign in signs:
            deltas = sign * radius * base_delta
            result = evaluate_taylor_remainder_and_correction(
                g_tilde,
                eta_hat,
                g0,
                B_hat,
                bread_factored,
                deltas,
                config.g_tilde_chunk_size,
            )
            sign_results.append(result)

        r_all = np.concatenate([res["r"] for res in sign_results])
        q_all = np.concatenate([res["q"] for res in sign_results])
        c_all = np.concatenate([res["c"] for res in sign_results], axis=0)
        r_by_radius[radius] = r_all
        q_by_radius[radius] = q_all

        a_all_by_target = {}
        for label, contrast_row, se in zip(target_labels, L, se_l, strict=False):
            a = np.abs(c_all @ contrast_row) / (se if se > 0 else 1.0)
            a_all_by_target[label] = a
            a_by_target_radius[label][radius] = a

        V_L = L @ np.asarray(V_hat) @ L.T
        joint_correction = joint_mahalanobis_correction(
            c_all, L, V_L, config.rank_tolerance
        )

        per_radius[radius] = {
            "r": _quantile_summary(r_all),
            "q": _quantile_summary(q_all),
            "a_by_target": {
                label: _quantile_summary(a) for label, a in a_all_by_target.items()
            },
            "joint_mahalanobis": {
                **_quantile_summary(joint_correction["values"]),
                "effective_rank": joint_correction["effective_rank"],
                "target_dim": joint_correction["target_dim"],
                "rank_deficient": joint_correction["rank_deficient"],
            },
        }
        for label, a in a_all_by_target.items():
            exceed = a > config.nonlinear_correction_tolerance_se
            frac = float(np.mean(exceed)) if a.size else 0.0
            per_radius[radius]["a_by_target"][label]["exceedance_fraction"] = frac
            per_radius[radius]["a_by_target"][label]["exceedance_upper_bound"] = (
                clopper_pearson_upper_bound(
                    int(np.sum(exceed)), a.size, config.confidence_level
                )
                if a.size
                else math.nan
            )

        if config.paired_directions:
            plus_c, minus_c = sign_results[0]["c"], sign_results[1]["c"]
            even_part = 0.5 * (plus_c + minus_c)
            odd_part = 0.5 * (plus_c - minus_c)
            per_radius[radius]["paired_even_norm_median"] = float(
                np.median(np.linalg.norm(even_part, axis=1))
            )
            per_radius[radius]["paired_odd_norm_median"] = float(
                np.median(np.linalg.norm(odd_part, axis=1))
            )

    # Below this floor, r_j/a_{j,l} are dominated by float32 roundoff rather than genuine
    # curvature signal, and their ratio across radii is meaningless noise -- skip the scaling
    # check entirely rather than warn on it (this is exactly the affine-map case, where r_j and
    # a_{j,l} should be ~0 at every radius and there is no power law to observe).
    _scaling_noise_floor = 1e-4

    warnings_list: list[str] = []
    radii_sorted = sorted(config.perturbation_radii)
    if len(radii_sorted) >= 2:
        r_small, r_large = radii_sorted[0], radii_sorted[-1]
        r_med_small = per_radius[r_small]["r"]["median"]
        r_med_large = per_radius[r_large]["r"]["median"]
        if r_med_small > _scaling_noise_floor and r_med_large > 0 and r_large > r_small:
            observed_exponent = math.log(r_med_large / r_med_small) / math.log(
                r_large / r_small
            )
            if not math.isnan(observed_exponent) and abs(observed_exponent - 1.0) > 0.5:
                warnings_list.append(
                    f"r_j does not scale approximately linearly with radius (observed exponent "
                    f"{observed_exponent:.2f}, expected ~1)."
                )
        for label in target_labels:
            a_med_small = per_radius[r_small]["a_by_target"][label]["median"]
            a_med_large = per_radius[r_large]["a_by_target"][label]["median"]
            if (
                a_med_small > _scaling_noise_floor
                and a_med_large > 0
                and r_large > r_small
            ):
                observed_exponent = math.log(a_med_large / a_med_small) / math.log(
                    r_large / r_small
                )
                if (
                    not math.isnan(observed_exponent)
                    and abs(observed_exponent - 2.0) > 1.0
                ):
                    warnings_list.append(
                        f"a_{{j,{label}}} does not scale approximately quadratically with radius "
                        f"(observed exponent {observed_exponent:.2f}, expected ~2)."
                    )

    headline_radius = 1.0 if 1.0 in per_radius else radii_sorted[-1]
    headline_max = max(
        per_radius[headline_radius]["a_by_target"][label]["max"]
        for label in target_labels
    )
    status = CheckStatuses.PASSED
    if (
        not math.isnan(headline_max)
        and headline_max > config.nonlinear_correction_tolerance_se
    ):
        status = CheckStatuses.WARNING
    if warnings_list and status == CheckStatuses.PASSED:
        status = CheckStatuses.WARNING
    if any(
        per_radius[radius]["joint_mahalanobis"]["rank_deficient"]
        for radius in per_radius
    ):
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            "Target covariance is rank-deficient relative to rank_tolerance: the joint "
            "Mahalanobis correction is reported on an identified subspace only."
        )

    return CheckResult(
        name="local_nonlinearity",
        status=status,
        metrics={"per_radius": per_radius, "headline_radius": headline_radius},
        warnings=warnings_list,
        message="",
    )


###############################################################################
# Section 4: exact nonlinear perturbation (continuation / chord-Newton)
###############################################################################


def solve_exact_perturbation(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    bread_factored,
    s_j: np.ndarray,
    delta_linear: np.ndarray,
    config: DiagnosticConfig,
) -> dict[str, Any]:
    """
    Solves g_tilde(eta_hat + delta) - g0 = s_j via continuation in lambda: 0 -> 1, warm-started
    from the linear solution, using a chord (fixed-Jacobian) Newton iteration built on the same
    B_hat factorization used everywhere else in the suite (never a re-differentiated Jacobian --
    that would make this check far more expensive than the rest of the suite combined).
    """
    d = np.array(delta_linear, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    lambdas = np.linspace(0.0, 1.0, config.continuation_steps + 1)[1:]
    total_iterations = 0
    nonfinite_encountered = False
    converged = True
    final_residual_norm = math.nan
    # Scale convergence relative to the FULL perturbation ||s_j||, not the current lambda-scaled
    # target: using ||lam * s_j|| as the denominator would make the "relative" residual blow up
    # spuriously at small lambda, since the absolute floating-point noise floor in g_tilde's
    # output does not shrink proportionally to lambda.
    full_scale = max(float(np.linalg.norm(s_j)), 1e-12)

    for lam in lambdas:
        target = lam * s_j
        step_converged = False
        for _ in range(config.nonlinear_solver_max_iterations):
            g_val = np.asarray(g_tilde(jnp.asarray(eta_hat) + jnp.asarray(d)))
            total_iterations += 1
            if not np.all(np.isfinite(g_val)):
                nonfinite_encountered = True
                break
            residual = g_val - g0 - target
            residual_norm = float(np.linalg.norm(residual))
            final_residual_norm = residual_norm
            if residual_norm / full_scale < config.nonlinear_solver_tolerance:
                step_converged = True
                break
            d = d - solve_with_bread(bread_factored, residual)
        if nonfinite_encountered:
            converged = False
            break
        if not step_converged:
            converged = False
            break

    discrepancy_ratio = float(
        np.linalg.norm(d - delta_linear) / (np.linalg.norm(delta_linear) + 1e-300)
    )
    branch_change_suspected = converged and discrepancy_ratio > 5.0

    return {
        "delta_nl": d,
        "converged": converged,
        "nonfinite_encountered": nonfinite_encountered,
        "final_residual_norm": final_residual_norm,
        "num_iterations": total_iterations,
        "branch_change_suspected": branch_change_suspected,
        "discrepancy_ratio": discrepancy_ratio,
    }


def se_ratios_from_generalized_eigenvalues(
    nonlinear_cov: np.ndarray, linear_cov: np.ndarray
) -> np.ndarray:
    """
    sqrt(lambda_k) for the generalized eigenvalues lambda_k of nonlinear_cov relative to
    linear_cov -- the nonlinear-to-linear standard-error ratios along the target's identified
    directions. `linear_cov` must be positive definite (it is L @ V_hat @ L^T on the identified
    target subspace); scipy.linalg.eigh(A, B) is the standard, numerically well-behaved way to
    solve the generalized eigenvalue problem A v = lambda B v without forming B^{-1} explicitly.
    """
    eigvals = scipy.linalg.eigh(nonlinear_cov, linear_cov, eigvals_only=True)
    return np.sqrt(np.clip(eigvals, 0.0, None))


def check_exact_nonlinear_perturbations(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    B_hat: np.ndarray,
    bread_factored,
    per_subject_stacks: np.ndarray,
    num_subjects: int,
    L: np.ndarray,
    target_labels: list[str],
    V_hat: np.ndarray,
    config: DiagnosticConfig,
) -> CheckResult:
    """
    Section 4/6: solves the exact nonlinear perturbation for each of num_exact_directions
    (paired +/- when configured), reports a^{NL}_{j,l}, generalized-eigenvalue SE-ratios, mean
    shift, quantile shift, and Clopper-Pearson upper bounds on failure/branch-change fractions.
    """
    num_directions = config.num_exact_directions or config.num_directions
    base_delta, base_s = sample_perturbation_directions(
        per_subject_stacks,
        bread_factored,
        num_subjects,
        num_directions,
        config.random_seed + 1,
    )
    signs = [1.0, -1.0] if config.paired_directions else [1.0]

    se_l = standard_errors_for_contrasts(V_hat, L)
    z_linear_rows: list[np.ndarray] = []
    z_nonlinear_rows: list[np.ndarray] = []
    a_nl_by_target: dict[str, list[float]] = {label: [] for label in target_labels}
    convergence_flags: list[bool] = []
    branch_change_flags: list[bool] = []
    nonfinite_flags: list[bool] = []

    for sign in signs:
        for j in range(num_directions):
            delta_lin = sign * base_delta[j]
            s_j = sign * base_s[j]
            solved = solve_exact_perturbation(
                g_tilde, eta_hat, g0, bread_factored, s_j, delta_lin, config
            )
            convergence_flags.append(solved["converged"])
            branch_change_flags.append(solved["branch_change_suspected"])
            nonfinite_flags.append(solved["nonfinite_encountered"])

            z_linear_rows.append(L @ delta_lin)
            z_nonlinear_rows.append(L @ solved["delta_nl"])
            e_j = solved["delta_nl"] - delta_lin
            for label, contrast_row, se in zip(target_labels, L, se_l, strict=False):
                a_nl_by_target[label].append(
                    float(abs(contrast_row @ e_j) / (se if se > 0 else 1.0))
                )

    z_linear = np.array(z_linear_rows)
    z_nonlinear = np.array(z_nonlinear_rows)
    V_L_lin = L @ np.asarray(V_hat) @ L.T
    nonlinear_cov = np.cov(z_nonlinear, rowvar=False)
    nonlinear_cov = np.atleast_2d(nonlinear_cov)

    warnings_list: list[str] = []
    try:
        se_ratios = se_ratios_from_generalized_eigenvalues(nonlinear_cov, V_L_lin)
    except (scipy.linalg.LinAlgError, ValueError) as exc:
        se_ratios = np.full(V_L_lin.shape[0], math.nan)
        warnings_list.append(f"Generalized eigenvalue computation failed: {exc}")

    try:
        b_L = float(
            np.linalg.norm(matrix_inv_sqrt(V_L_lin) @ (np.mean(z_nonlinear, axis=0)))
        )
    except np.linalg.LinAlgError:
        b_L = math.nan

    quantile_shifts: dict[str, dict[str, float]] = {}
    for idx, label in enumerate(target_labels):
        lin_col = z_linear[:, idx]
        nl_col = z_nonlinear[:, idx]
        se = se_l[idx] if se_l[idx] > 0 else 1.0
        q_lo_lin, q_hi_lin = np.quantile(lin_col, [0.025, 0.975])
        q_lo_nl, q_hi_nl = np.quantile(nl_col, [0.025, 0.975])
        quantile_shifts[label] = {
            "lower_shift_se": float((q_lo_nl - q_lo_lin) / se),
            "upper_shift_se": float((q_hi_nl - q_hi_lin) / se),
        }

    num_trials = len(convergence_flags)
    num_root_failures = int(sum(1 for c in convergence_flags if not c))
    num_branch_changes = int(sum(branch_change_flags))
    num_domain_failures = int(sum(nonfinite_flags))

    failure_upper_bound = clopper_pearson_upper_bound(
        num_root_failures, num_trials, config.confidence_level
    )
    branch_change_upper_bound = clopper_pearson_upper_bound(
        num_branch_changes, num_trials, config.confidence_level
    )
    domain_failure_upper_bound = clopper_pearson_upper_bound(
        num_domain_failures, num_trials, config.confidence_level
    )

    status = CheckStatuses.PASSED
    if failure_upper_bound > config.bad_direction_probability_target:
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            f"Nonlinear-root failure upper bound {failure_upper_bound:.4g} exceeds target "
            f"{config.bad_direction_probability_target}."
        )
    se_ratio_ok = (
        bool(np.all((se_ratios >= 0.95) & (se_ratios <= 1.05)))
        if se_ratios.size
        else False
    )
    if not se_ratio_ok:
        status = CheckStatuses.FAILED
    if not math.isnan(b_L) and b_L > config.mean_shift_tolerance_se:
        status = CheckStatuses.FAILED
    for shifts in quantile_shifts.values():
        if (
            abs(shifts["lower_shift_se"]) > config.quantile_shift_tolerance_se
            or abs(shifts["upper_shift_se"]) > config.quantile_shift_tolerance_se
        ):
            status = CheckStatuses.FAILED

    metrics = {
        "a_nl_by_target": {
            label: _quantile_summary(np.array(values))
            for label, values in a_nl_by_target.items()
        },
        "se_ratios": se_ratios.tolist(),
        "se_ratios_within_tolerance": se_ratio_ok,
        "mean_shift_se": b_L,
        "quantile_shifts_se": quantile_shifts,
        "num_trials": num_trials,
        "root_failure_fraction": num_root_failures / num_trials
        if num_trials
        else math.nan,
        "root_failure_upper_bound": failure_upper_bound,
        "branch_change_fraction": num_branch_changes / num_trials
        if num_trials
        else math.nan,
        "branch_change_upper_bound": branch_change_upper_bound,
        "domain_failure_fraction": num_domain_failures / num_trials
        if num_trials
        else math.nan,
        "domain_failure_upper_bound": domain_failure_upper_bound,
    }

    return CheckResult(
        name="exact_nonlinear_perturbation",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        message="",
    )


###############################################################################
# Section 7: Jacobian drift / heuristic contraction bound
###############################################################################


def check_jacobian_drift(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    B_hat: np.ndarray,
    bread_factored,
    per_subject_stacks: np.ndarray,
    num_subjects: int,
    config: DiagnosticConfig,
) -> CheckResult:
    """
    Samples D g_tilde at a small number of points along a small number of perturbation paths and
    reports the largest observed rho_j = ||B_hat^{-1}(D g_tilde(path point) - B_hat)||_op. This is
    explicitly a SAMPLED PATH MAXIMUM, not a certified supremum over a neighborhood -- the repo
    has no interval-arithmetic/Lipschitz machinery to certify a true bound, so the contraction
    bound q_j/(1-rho_j) reported here is a heuristic, not a proof.
    """
    B = np.asarray(B_hat, dtype=np.float64)
    num_directions = min(config.drift_num_directions, config.num_directions) or 1
    base_delta, _ = sample_perturbation_directions(
        per_subject_stacks,
        bread_factored,
        num_subjects,
        num_directions,
        config.random_seed + 2,
    )

    rho_by_direction = []
    for delta in base_delta:
        max_norm = 0.0
        for t in config.drift_path_samples:
            point = jnp.asarray(eta_hat) + t * jnp.asarray(delta)
            jac = np.asarray(jax.jacrev(g_tilde)(point), dtype=np.float64)
            diff = jac - B
            X = solve_with_bread(bread_factored, diff)
            op_norm = float(np.linalg.norm(X, ord=2))
            max_norm = max(max_norm, op_norm)
        rho_by_direction.append(max_norm)

    rho_by_direction = np.array(rho_by_direction)
    warnings_list = [
        "rho_j is a sampled path maximum over a small number of points/directions, not a "
        "certified supremum; no contraction certificate is claimed."
    ]
    status = (
        CheckStatuses.WARNING
        if np.any(rho_by_direction >= 1.0)
        else CheckStatuses.PASSED
    )

    return CheckResult(
        name="jacobian_drift",
        status=status,
        metrics={
            "rho_by_direction": rho_by_direction.tolist(),
            "rho_max": float(np.max(rho_by_direction))
            if rho_by_direction.size
            else math.nan,
        },
        warnings=warnings_list,
        message="",
    )


###############################################################################
# Section 8: bread and numerical-stability diagnostics
###############################################################################


def _theta_only_variance_diag_qr(
    bread: np.ndarray, meat: np.ndarray, theta_dim: int, num_subjects: int
) -> np.ndarray:
    """Diagonal of the theta-only sandwich, via the same QR technique used elsewhere (no explicit inverse)."""
    Q, R = np.linalg.qr(bread.T, mode="reduced")
    Lmat = R.T
    new_meat = scipy.linalg.solve_triangular(
        Lmat, scipy.linalg.solve_triangular(Lmat, meat.T, lower=True).T, lower=True
    )
    sandwich = (Q @ new_meat @ Q.T) / num_subjects
    return np.diag(sandwich)[-theta_dim:]


def check_bread_stability(
    B_hat: np.ndarray,
    M_hat: np.ndarray,
    beta_dim: int,
    theta_dim: int,
    num_subjects: int,
    V_hat: np.ndarray,
    config: DiagnosticConfig,
) -> CheckResult:
    """
    Section 8. Reports per-update diagonal-block and theta-block singular values/condition
    numbers, off-diagonal coupling magnitudes, target-covariance rank/eigenvalues, and the
    sensitivity of target SEs to a numerically negligible perturbation of B_hat. Does not
    hard-code a universal condition-number threshold: conditioning is reported, not judged, here.
    """
    B = np.asarray(B_hat, dtype=np.float64)
    M = np.asarray(M_hat, dtype=np.float64)
    num_updates = (B.shape[0] - theta_dim) // beta_dim if beta_dim else 0

    block_diagnostics = []
    for i in range(num_updates):
        sl = slice(i * beta_dim, (i + 1) * beta_dim)
        block = B[sl, sl]
        svals = np.linalg.svd(block, compute_uv=False)
        cond = float(svals[0] / svals[-1]) if svals[-1] > 0 else math.inf
        block_diagnostics.append(
            {"update": i, "singular_values": svals.tolist(), "condition_number": cond}
        )

    theta_block = B[-theta_dim:, -theta_dim:]
    theta_svals = np.linalg.svd(theta_block, compute_uv=False)
    theta_cond = (
        float(theta_svals[0] / theta_svals[-1]) if theta_svals[-1] > 0 else math.inf
    )

    off_diag_beta_to_theta = (
        float(np.linalg.norm(B[:-theta_dim, -theta_dim:])) if theta_dim else 0.0
    )
    off_diag_theta_to_beta = (
        float(np.linalg.norm(B[-theta_dim:, :-theta_dim])) if theta_dim else 0.0
    )

    full_svals = np.linalg.svd(B, compute_uv=False)
    full_cond = (
        float(full_svals[0] / full_svals[-1]) if full_svals[-1] > 0 else math.inf
    )

    eigvals_V = np.linalg.eigvalsh(np.asarray(V_hat))
    max_eig = float(eigvals_V.max()) if eigvals_V.size else math.nan
    rank_estimate = (
        int(np.sum(eigvals_V > config.rank_tolerance * max_eig)) if max_eig > 0 else 0
    )

    rel_scale = config.bread_perturbation_relative_scale
    perturbation = rel_scale * np.linalg.norm(B) * np.eye(B.shape[0])
    orig_diag = _theta_only_variance_diag_qr(B, M, theta_dim, num_subjects)
    perturbed_diag = _theta_only_variance_diag_qr(
        B + perturbation, M, theta_dim, num_subjects
    )
    orig_se = np.sqrt(np.clip(orig_diag, 0.0, None))
    perturbed_se = np.sqrt(np.clip(perturbed_diag, 0.0, None))
    relative_se_change = np.divide(
        np.abs(perturbed_se - orig_se),
        orig_se,
        out=np.zeros_like(orig_se),
        where=orig_se > 0,
    )
    sensitivity_max = (
        float(np.max(relative_se_change)) if relative_se_change.size else 0.0
    )

    warnings_list = []
    status = CheckStatuses.PASSED
    if rank_estimate < theta_dim:
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            f"Target covariance rank estimate {rank_estimate} < theta_dim {theta_dim}; "
            "identification of some contrasts may be weak."
        )
    if sensitivity_max > config.se_distortion_tolerance:
        warnings_list.append(
            f"Target SEs changed by {sensitivity_max:.2%} under a numerically negligible "
            f"({rel_scale:.1e} relative) perturbation of B_hat -- this indicates numerical "
            "fragility distinct from statistical identification."
        )
        status = (
            CheckStatuses.INDETERMINATE if status == CheckStatuses.PASSED else status
        )

    return CheckResult(
        name="bread_stability",
        status=status,
        metrics={
            "diagonal_block_diagnostics": block_diagnostics,
            "theta_block_condition_number": theta_cond,
            "theta_block_singular_values": theta_svals.tolist(),
            "off_diagonal_beta_to_theta_norm": off_diag_beta_to_theta,
            "off_diagonal_theta_to_beta_norm": off_diag_theta_to_beta,
            "full_bread_condition_number": full_cond,
            "target_covariance_eigenvalues": eigvals_V.tolist(),
            "target_covariance_rank_estimate": rank_estimate,
            "numerical_sensitivity_max_relative_se_change": sensitivity_max,
        },
        warnings=warnings_list,
        message="",
    )


###############################################################################
# Section 9: influence concentration
###############################################################################


def check_influence_concentration(
    per_subject_stacks: np.ndarray,
    bread_factored,
    L: np.ndarray,
    target_labels: list[str],
    subject_ids: Sequence[Any],
    config: DiagnosticConfig,
    *,
    B_hat: np.ndarray | None = None,
    theta_dim: int | None = None,
) -> CheckResult:
    """
    Section 9. xi_ic = -c^T L B_hat^{-1} g_i(eta_hat), computed via a single transpose-solve per
    contrast (w = B_hat^{-T} l) rather than forming B_hat^{-1} explicitly. Reports the largest
    variance share, effective count of equally-influential subjects, and third-moment
    concentration per contrast, plus the top-k most influential subject identifiers. When
    `config.compute_leave_one_out_sensitivity` is set (and B_hat/theta_dim are supplied), also
    runs the one-step leave-one-out theta sensitivity check for the most influential subjects
    (see `leave_one_out_theta_sensitivity`) -- this is a sensitivity analysis, not a bootstrap.
    """
    stacks = np.asarray(per_subject_stacks, dtype=np.float64)
    n = stacks.shape[0]
    top_k = 5

    by_target: dict[str, Any] = {}
    warnings_list: list[str] = []
    status = CheckStatuses.PASSED
    most_influential_indices: list[int] = []

    for label, contrast_row in zip(target_labels, L, strict=False):
        w = solve_with_bread_transpose(bread_factored, contrast_row)
        xi = -(stacks @ w)
        sq = xi**2
        total = float(sq.sum())
        if total <= 0:
            by_target[label] = {"p_max": math.nan, "n_eff": math.nan, "L_c": math.nan}
            continue
        p = sq / total
        p_max = float(p.max())
        n_eff = float(1.0 / np.sum(p**2))
        L_c = float(np.sum(np.abs(xi) ** 3) / total**1.5)

        order = np.argsort(-np.abs(xi))[:top_k]
        most_influential_indices.extend(
            int(i) for i in order[: config.leave_one_out_top_k]
        )
        top_subjects = [
            {
                "subject_id": (
                    subject_ids[idx].item()
                    if hasattr(subject_ids[idx], "item")
                    else subject_ids[idx]
                )
                if config.report_subject_identifiers
                else None,
                "xi": float(xi[idx]),
                "variance_share": float(p[idx]),
            }
            for idx in order
        ]

        by_target[label] = {
            "p_max": p_max,
            "n_eff": n_eff,
            "third_moment_concentration": L_c,
            "top_influential_subjects": top_subjects,
        }
        # n_eff is bounded below by 1 (it can never indicate fewer than "one equally influential
        # subject"), so a purely relative threshold like `n_eff < 0.1*n` can never fire for
        # small n even in the most extreme single-subject-dominance case; p_max supplements it.
        if n_eff < max(2.0, 0.1 * n) or p_max > 0.5:
            warnings_list.append(
                f"For target '{label}', an effective count of only {n_eff:.1f} out of {n} "
                f"subjects drives the estimated variance (largest single share: {p_max:.1%})."
            )
            status = CheckStatuses.WARNING

    metrics: dict[str, Any] = {"by_target": by_target, "num_subjects": n}

    if config.compute_leave_one_out_sensitivity and B_hat is not None and theta_dim:
        theta_block_factored = factor_bread(np.asarray(B_hat)[-theta_dim:, -theta_dim:])
        unique_indices = sorted(set(most_influential_indices))
        metrics["leave_one_out_sensitivity"] = leave_one_out_theta_sensitivity(
            stacks, theta_dim, theta_block_factored, unique_indices, subject_ids
        )

    return CheckResult(
        name="influence_concentration",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        message="",
    )


def leave_one_out_theta_sensitivity(
    per_subject_stacks: np.ndarray,
    theta_dim: int,
    bread_factored_theta_block,
    subject_indices_to_check: Sequence[int],
    subject_ids: Sequence[Any],
) -> list[dict[str, Any]]:
    """
    Optional sensitivity mode (NOT a valid bootstrap of the adaptive deployment -- deleting a
    subject does not replay the policy that would have been run without them). Holds all beta_k
    fixed at their observed values and reports, for each requested subject, the one-step Newton
    shift in theta implied by excluding that subject's contribution to the theta-block
    estimating equation, using the closed-form leave-one-out average available because
    avg_estimating_function_stack is exactly mean(per_subject_stacks, axis=0).

    This is intentionally a single Newton step, not an iterated re-solve to convergence: doing
    better would require re-evaluating every OTHER subject's estimating-function row at a new
    theta, which needs the full estimator machinery (not available generically here) whenever
    that row also depends on other subjects' data through anything besides the average itself.
    The one-step shift is exactly what the local influence function ("exact deletion effect")
    predicts to first order, so it is reported as such rather than as an "exact" re-fit.
    """
    stacks = np.asarray(per_subject_stacks, dtype=np.float64)
    n = stacks.shape[0]
    total_sum = stacks[:, -theta_dim:].sum(axis=0)

    results = []
    for idx in subject_indices_to_check:
        g_i_theta = stacks[idx, -theta_dim:]
        loo_mean_at_theta_hat = (total_sum - g_i_theta) / (n - 1)
        one_step_shift = -solve_with_bread(
            bread_factored_theta_block, loo_mean_at_theta_hat
        )
        results.append(
            {
                "subject_id": (
                    subject_ids[idx].item()
                    if hasattr(subject_ids[idx], "item")
                    else subject_ids[idx]
                ),
                "one_step_theta_shift": one_step_shift.tolist(),
            }
        )
    return results


###############################################################################
# Section 10: exploration and importance-weight diagnostics
###############################################################################


def compute_importance_weights_under_beta(
    action_prob_func: Callable,
    action_prob_func_args_beta_index: int,
    action_prob_func_args: dict[int, dict[Any, tuple[Any, ...]]],
    action_by_decision_time_by_subject_id: dict[Any, dict[int, int]],
    policy_num_by_decision_time_by_subject_id: dict[Any, dict[int, Any]],
    initial_policy_num: Any,
    beta_index_by_policy_num: dict[Any, int],
    perturbed_betas: np.ndarray,
    subject_ids: Sequence[Any],
) -> dict[Any, dict[int, float]]:
    """
    Cumulative importance weight trajectories evaluated under a perturbed beta (e.g. eta_hat +
    delta_j), rather than at eta_hat itself where these diagnostic weights are trivially 1. Uses
    the existing helper_functions.get_radon_nikodym_weight machinery directly.
    """
    weights_by_subject: dict[Any, dict[int, float]] = {}
    for subject_id in subject_ids:
        key = subject_id.item() if hasattr(subject_id, "item") else subject_id
        times = sorted(policy_num_by_decision_time_by_subject_id.get(key, {}).keys())
        cumulative = 1.0
        trajectory: dict[int, float] = {}
        for t in times:
            policy_num = policy_num_by_decision_time_by_subject_id[key][t]
            if (
                policy_num == initial_policy_num
                or policy_num not in beta_index_by_policy_num
            ):
                trajectory[t] = cumulative
                continue
            args = action_prob_func_args[t][key]
            action = action_by_decision_time_by_subject_id[key][t]
            beta_target = perturbed_betas[beta_index_by_policy_num[policy_num]]
            weight = float(
                get_radon_nikodym_weight(
                    beta_target,
                    action_prob_func,
                    action_prob_func_args_beta_index,
                    action,
                    *args,
                )
            )
            cumulative *= weight
            trajectory[t] = cumulative
        weights_by_subject[key] = trajectory
    return weights_by_subject


def check_exploration_and_weights(
    analysis_df,
    active_col_name: str,
    calendar_t_col_name: str,
    action_prob_col_name: str,
    config: DiagnosticConfig,
    *,
    perturbed_weight_trajectories: Sequence[dict[Any, dict[int, float]]] = (),
    pi_and_weight_gradients_by_calendar_t: dict[int, dict[str, dict[Any, Any]]]
    | None = None,
) -> CheckResult:
    """
    Section 10. Reports per-decision-time action-probability extremes/quantiles from the
    recorded data, exceedance of any supplied exploration_floor/exploration_ceiling (a hard
    requirement when the caller supplies the deployment's actual design bounds), cumulative
    importance-weight quantiles and normalized ESS under the supplied sandwich-scale perturbed
    weight trajectories, and policy-score-derivative norm quantiles when gradients are supplied.
    """
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    by_time = active_df.groupby(calendar_t_col_name)[action_prob_col_name]
    min_by_time = by_time.min()
    max_by_time = by_time.max()

    metrics: dict[str, Any] = {
        "action_prob_min_by_time": min_by_time.to_dict(),
        "action_prob_max_by_time": max_by_time.to_dict(),
        "action_prob_global_min": float(active_df[action_prob_col_name].min()),
        "action_prob_global_max": float(active_df[action_prob_col_name].max()),
    }

    warnings_list: list[str] = []
    status = CheckStatuses.PASSED

    all_probs = active_df[action_prob_col_name].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(all_probs)):
        status = CheckStatuses.FAILED
        warnings_list.append("Nonfinite recorded action probabilities.")
    if np.any(all_probs <= 0) or np.any(all_probs >= 1):
        status = CheckStatuses.FAILED
        warnings_list.append(
            "Recorded action probabilities outside the open interval (0, 1)."
        )

    if config.exploration_floor is not None:
        near_floor = float(np.mean(all_probs <= config.exploration_floor))
        metrics["fraction_at_or_near_floor"] = near_floor
        if np.any(all_probs < config.exploration_floor):
            status = CheckStatuses.FAILED
            warnings_list.append(
                "At least one recorded probability violates exploration_floor."
            )
    if config.exploration_ceiling is not None:
        near_ceiling = float(np.mean(all_probs >= config.exploration_ceiling))
        metrics["fraction_at_or_near_ceiling"] = near_ceiling
        if np.any(all_probs > config.exploration_ceiling):
            status = CheckStatuses.FAILED
            warnings_list.append(
                "At least one recorded probability violates exploration_ceiling."
            )

    ess_by_direction = []
    max_cumulative_weight_by_direction = []
    for trajectory in perturbed_weight_trajectories:
        final_weights = np.array(
            [
                subject_traj[max(subject_traj)]
                for subject_traj in trajectory.values()
                if subject_traj
            ]
        )
        if final_weights.size == 0:
            continue
        if not np.all(np.isfinite(final_weights)):
            status = CheckStatuses.FAILED
            warnings_list.append(
                "Nonfinite importance weight encountered under perturbation."
            )
            continue
        num = final_weights.sum() ** 2
        denom = final_weights.size * np.sum(final_weights**2)
        ess_over_n = float(num / denom) if denom > 0 else math.nan
        ess_by_direction.append(ess_over_n)
        max_cumulative_weight_by_direction.append(float(final_weights.max()))

    if ess_by_direction:
        metrics["ess_over_n_by_direction"] = _quantile_summary(
            np.array(ess_by_direction)
        )
        metrics["max_cumulative_weight_by_direction"] = _quantile_summary(
            np.array(max_cumulative_weight_by_direction)
        )

    if pi_and_weight_gradients_by_calendar_t:
        pi_grad_norms = []
        for _, entry in pi_and_weight_gradients_by_calendar_t.items():
            for grad in entry.get("pi_gradients_by_user_id", {}).values():
                pi_grad_norms.append(float(np.linalg.norm(np.asarray(grad))))
        if pi_grad_norms:
            metrics["policy_score_gradient_norm_summary"] = _quantile_summary(
                np.array(pi_grad_norms)
            )

    return CheckResult(
        name="exploration_and_weights",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        message="",
    )


###############################################################################
# Orchestration
###############################################################################


def _combine_classification(check_results: dict[str, CheckResult]) -> str:
    statuses = [result.status for result in check_results.values()]
    if any(s == CheckStatuses.FAILED for s in statuses):
        return DiagnosticClassifications.FAILED
    if any(s == CheckStatuses.INDETERMINATE for s in statuses):
        return DiagnosticClassifications.INDETERMINATE
    return DiagnosticClassifications.LOCALLY_SUPPORTED


def run_diagnostic_suite(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    B_hat: np.ndarray,
    M_hat: np.ndarray,
    joint_sandwich_matrix: np.ndarray,
    per_subject_stacks: np.ndarray,
    beta_dim: int,
    theta_dim: int,
    num_subjects: int,
    config: DiagnosticConfig | None = None,
    *,
    legacy_check_callables: Sequence[tuple[str, Callable[[], Any]]] = (),
    analysis_df=None,
    active_col_name: str | None = None,
    calendar_t_col_name: str | None = None,
    action_prob_col_name: str | None = None,
    action_prob_func: Callable | None = None,
    action_prob_func_args: dict | None = None,
    action_prob_func_args_beta_index: int | None = None,
    action_by_decision_time_by_subject_id: dict | None = None,
    policy_num_by_decision_time_by_subject_id: dict | None = None,
    initial_policy_num: Any = None,
    beta_index_by_policy_num: dict | None = None,
    subject_ids: Sequence[Any] | None = None,
    pi_and_weight_gradients_by_calendar_t: dict | None = None,
) -> DiagnosticReport:
    """
    Runs the full layered diagnostic suite and combines every check into one DiagnosticReport.
    This function alone can never return DiagnosticClassifications.SUPPORTED -- that
    classification is only available from simulator_calibration.calibrate_and_classify after a
    held-out simulator pass, enforced here directly (not merely by convention).
    """
    config = config or DiagnosticConfig()
    warnings_list: list[str] = []
    check_results: dict[str, CheckResult] = {}

    beta_total_dim = (
        beta_dim * ((B_hat.shape[0] - theta_dim) // beta_dim) if beta_dim else 0
    )
    L, target_labels = default_contrast_matrix(beta_total_dim, theta_dim, config)

    B_hat_np = np.asarray(B_hat, dtype=np.float64)
    bread_factored = factor_bread(B_hat_np)
    g0 = np.asarray(g_tilde(jnp.asarray(eta_hat)), dtype=np.float64)
    V_hat = np.asarray(joint_sandwich_matrix, dtype=np.float64)

    check_results["root_and_implementation"] = check_root_and_implementation(
        g_tilde,
        eta_hat,
        g0,
        B_hat_np,
        M_hat,
        bread_factored,
        L,
        target_labels,
        V_hat,
        config,
        legacy_check_callables=legacy_check_callables,
    )

    hard_failed = (
        check_results["root_and_implementation"].status == CheckStatuses.FAILED
    )

    if not hard_failed:
        check_results["local_nonlinearity"] = check_local_nonlinearity(
            g_tilde,
            eta_hat,
            g0,
            B_hat_np,
            bread_factored,
            per_subject_stacks,
            num_subjects,
            L,
            target_labels,
            V_hat,
            config,
        )

        if config.compute_exact_nonlinear_roots:
            check_results["exact_nonlinear_perturbation"] = (
                check_exact_nonlinear_perturbations(
                    g_tilde,
                    eta_hat,
                    g0,
                    B_hat_np,
                    bread_factored,
                    per_subject_stacks,
                    num_subjects,
                    L,
                    target_labels,
                    V_hat,
                    config,
                )
            )
            check_results["jacobian_drift"] = check_jacobian_drift(
                g_tilde,
                eta_hat,
                B_hat_np,
                bread_factored,
                per_subject_stacks,
                num_subjects,
                config,
            )

        check_results["bread_stability"] = check_bread_stability(
            B_hat_np, M_hat, beta_dim, theta_dim, num_subjects, V_hat, config
        )

        if config.compute_influence_and_overlap_checks:
            check_results["influence_concentration"] = check_influence_concentration(
                per_subject_stacks,
                bread_factored,
                L,
                target_labels,
                subject_ids
                if subject_ids is not None
                else list(range(per_subject_stacks.shape[0])),
                config,
                B_hat=B_hat_np,
                theta_dim=theta_dim,
            )

        if (
            analysis_df is not None
            and active_col_name
            and calendar_t_col_name
            and action_prob_col_name
        ):
            perturbed_trajectories = []
            if (
                action_prob_func is not None
                and action_prob_func_args is not None
                and action_prob_func_args_beta_index is not None
                and action_by_decision_time_by_subject_id is not None
                and policy_num_by_decision_time_by_subject_id is not None
                and beta_index_by_policy_num is not None
                and subject_ids is not None
            ):
                num_weight_directions = min(config.num_directions, 5)
                base_delta, _ = sample_perturbation_directions(
                    per_subject_stacks,
                    bread_factored,
                    num_subjects,
                    num_weight_directions,
                    config.random_seed + 3,
                )
                num_updates = (
                    (B_hat_np.shape[0] - theta_dim) // beta_dim if beta_dim else 0
                )
                for delta in base_delta:
                    beta_flat_perturbed = (
                        np.asarray(eta_hat, dtype=np.float64)[:beta_total_dim]
                        + delta[:beta_total_dim]
                    )
                    perturbed_betas = beta_flat_perturbed.reshape(num_updates, beta_dim)
                    perturbed_trajectories.append(
                        compute_importance_weights_under_beta(
                            action_prob_func,
                            action_prob_func_args_beta_index,
                            action_prob_func_args,
                            action_by_decision_time_by_subject_id,
                            policy_num_by_decision_time_by_subject_id,
                            initial_policy_num,
                            beta_index_by_policy_num,
                            perturbed_betas,
                            subject_ids,
                        )
                    )

            check_results["exploration_and_weights"] = check_exploration_and_weights(
                analysis_df,
                active_col_name,
                calendar_t_col_name,
                action_prob_col_name,
                config,
                perturbed_weight_trajectories=perturbed_trajectories,
                pi_and_weight_gradients_by_calendar_t=pi_and_weight_gradients_by_calendar_t,
            )

    for result in check_results.values():
        warnings_list.extend(f"[{result.name}] {w}" for w in result.warnings)

    classification = (
        DiagnosticClassifications.FAILED
        if hard_failed
        else _combine_classification(check_results)
    )

    return DiagnosticReport(
        classification=classification,
        check_results=check_results,
        metrics={name: result.metrics for name, result in check_results.items()},
        tolerances_used=dataclasses.asdict(config),
        warnings=warnings_list,
        monte_carlo_counts={
            "num_directions": config.num_directions,
            "num_exact_directions": config.num_exact_directions
            or config.num_directions,
        },
        target_labels=target_labels,
        rank_diagnostics=check_results.get(
            "bread_stability", CheckResult(name="", status="", metrics={})
        ).metrics,
    )
