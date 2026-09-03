from __future__ import annotations

import dataclasses
import logging
import math
import os
import re
import sys
import textwrap
from collections.abc import Callable, Sequence
from typing import Any

import jax
import numpy as np
import scipy.linalg
from jax import numpy as jnp

from .constants import (
    CheckStatuses,
    DiagnosticClassifications,
    DiagnosticVerdicts,
    VerdictBases,
)
from .helper_functions import (
    clopper_pearson_lower_bound,
    clopper_pearson_upper_bound,
    compute_row_chunked_jacobian,
    get_radon_nikodym_weight,
    matrix_inv_sqrt,
    resolve_jacobian_row_chunk_size,
)

logger = logging.getLogger(__name__)

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
    has calibrated them against its own simulator.
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
    # Unlike every other tolerance in this dataclass, this one is not an engineering guess and
    # does not need simulator calibration: a healthy LU-based solve is backward-stable by
    # construction, so backward_relative_residual sits near float64 machine epsilon (~1e-16)
    # regardless of how ill-conditioned B_hat is -- confirmed empirically across condition
    # numbers spanning twelve orders of magnitude (well-conditioned to near-exactly-singular).
    # This is a smoke detector for a broken solve (e.g. a stale/mismatched bread_factored, or a
    # bug in factor_bread/solve_with_bread), not a statistical judgment call -- 1e-6 is already
    # about ten orders of magnitude looser than anything a healthy solve produces.
    backward_residual_tolerance: float = 1e-6
    nonlinear_correction_tolerance_se: float = 0.10
    se_distortion_tolerance: float = 0.05
    # Not a standalone cutoff: the mean-shift gate of the exact and bootstrap checks compares
    # b_L against max(this, the simulated finite-draw null upper quantile), so this is the
    # absolute practical-significance FLOOR (a displacement under a tenth of a target SE is not
    # worth reporting however detectable it becomes at large J) while the band absorbs the
    # finite-draw noise. See _evaluate_mean_shift_gate.
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

    # Divergence abort for the chord-Newton inner loop of solve_exact_perturbation. A
    # continuation step only ever fails by exhausting nonlinear_solver_max_iterations, so a step
    # that has visibly left the region where the frozen bread models g_tilde still pays the full
    # iteration budget before being counted as a failure. These two clauses stop such a step
    # early; an aborted step is treated EXACTLY as an exhausted one (converged=False, the solve
    # fails), which is what makes the saving verdict-neutral -- see the correctness argument in
    # solve_exact_perturbation's docstring.
    #
    # Both are calibrated against measured margins on CONVERGING continuation steps, since the
    # only way this can change a verdict is by aborting a step that would have converged:
    #   blowup_factor  -- worst r_k / (best r seen so far in the step) ever observed inside a
    #                     step that then converged: 1.9 on this repository's diagnostic fixtures
    #                     (33,351 converging steps) and 36.7 on an independent synthetic sweep of
    #                     logistic/cubic/tanh systems (118,563 converging steps). 1e4 is a ~270x
    #                     margin on the worse of the two. A chord step that overshoots its own
    #                     best by four orders of magnitude has left the region where the frozen
    #                     B_hat models g_tilde at all.
    #   stall_window   -- longest run of consecutive iterations with no new best residual inside
    #                     a step that then converged: 4 here, 20 on the synthetic sweep (a
    #                     genuine plateau that rose for 20 iterations and then collapsed to
    #                     convergence). 40 is a 2x margin on the worse sample. DO NOT lower this
    #                     toward 25 to buy a few percent: that cuts the margin to 1.25x, and a
    #                     false abort silently converts a converged draw into a root failure,
    #                     which moves the failure fraction and can flip a check's status. Set it
    #                     to 0 to disable the stall clause and keep only the blow-up one, whose
    #                     margin is three orders wider.
    #   guard_factor   -- neither clause is evaluated while the step's best relative residual is
    #                     already within guard_factor * nonlinear_solver_tolerance. At float32
    #                     the residual noise floor sits about an order below the tolerance, so a
    #                     step can legitimately bounce inside a few multiples of it for a dozen
    #                     iterations before dipping under; the guard removes that entire
    #                     mechanism for one to two percentage points of saving.
    # A per-step ITERATION CAP was measured and rejected: converging steps routinely use the full
    # 50 (p95 = 34-41), so no cap below the existing budget is safe. So was a rate-extrapolation
    # ("this contraction rate cannot reach tolerance in the iterations left") clause: it had zero
    # false aborts in-sample and three out of sample, on steps whose residual creeps for five
    # iterations and then collapses.
    #
    # NOT a proof, only a measurement -- ~152,000 converging steps with zero false aborts across
    # two independent corpora, and then on real study data rather than synthetic fixtures: the two
    # ADS-142 cells this was most likely to break, run paired with the flag on and off over the
    # same seeds (A_hard jobs 43585039/43585044, B_influence 43585041/43585045; 40 paired seeds,
    # 796 aborts fired). Zero differences in verdict, classification or any check status, and
    # num_converged_trials identical on both arms -- which is the strong form of the claim, since
    # an abort can only ever turn a converged trial into a failed one, so an unchanged count is
    # proof of zero false aborts rather than evidence of them. 18 of the 20 influence-cell seeds
    # land on FAILED-from-a-censored-subset (verdict `invalid`), i.e. the precedence path an
    # unsound abort or early stop would have destroyed is the one the corpus mostly exercises.
    # Set nonlinear_solver_divergence_abort=False to restore the exhaust-the-budget behavior
    # exactly.
    nonlinear_solver_divergence_abort: bool = True
    nonlinear_solver_divergence_blowup_factor: float = 1e4
    nonlinear_solver_divergence_stall_window: int = 40
    nonlinear_solver_divergence_guard_factor: float = 100.0

    # Sequential early stop for the re-solve ensembles of check_exact_nonlinear_perturbations and
    # check_multiplier_bootstrap. "starvation" (default) | "off".
    #
    # "starvation" stops the trial loop as soon as the converged count that is still ATTAINABLE
    # falls below the 3 that every ensemble statistic needs -- i.e. converged_so_far +
    # trials_remaining <= 2. That predicate is provably status-preserving (see the argument at the
    # stop itself) and therefore verdict-preserving, which is why it is on by default: it is pure
    # saving. It is also provably worth very little -- it can never skip more than 2 trials of the
    # planned ensemble (1% at 100 paired draws, 4% at 25) -- and it fires only on runs where
    # essentially nothing converges. It is here for the guarantee and the honest accounting
    # (num_planned_trials / early_stopped / early_stop_reason), not for the speed.
    #
    # The tempting stronger rule -- "stop once the solver is CERTAINLY unhealthy" -- is NOT
    # implemented, because it is not verdict-preserving. Unhealthiness only confines the status to
    # {FAILED, INDETERMINATE}: the FAILED rung deliberately sits ABOVE the unhealthy rung in both
    # checks, so a distortion measured on the optimistically-selected converged subset still
    # counts. In the ADS-142 influence cell every one of 50 cluster runs was certainly unhealthy
    # from the first trial (failure fractions 0.12-0.86) and every one of them nevertheless
    # reported FAILED from its converged subset; stopping on unhealthiness would have turned 50
    # INVALID verdicts into 50 UNCERTIFIABLE ones. Anything unrecognized here is treated as
    # "starvation" rather than raising: the sound stop cannot change an answer, so a typo in this
    # field must not be able to either.
    perturbation_early_stop: str = "starvation"

    # Batched lockstep re-solves for check_multiplier_bootstrap's ensemble (see
    # solve_exact_perturbations_batched). "off" (default) | "auto" | "always".
    #
    # The serial loop spends essentially all of its time inside one g_tilde evaluation per
    # chord-Newton iteration, dispatched one row at a time. The batched driver advances a whole
    # wave of draws' chord iterations together, so each generation of the wave is ONE vmap'd
    # g_tilde call instead of up to bootstrap_batch_width separate ones. Each draw's trajectory is
    # untouched -- the driver only changes which rows share a kernel launch -- so the ensemble is
    # verdict-equivalent. Measured on the real n=100/T=50 pipeline fixture (eta_dim 200, 100 paired
    # draws): identical status, identical warnings, and an identical converged (sign, draw) pattern
    # -- the same 200 of 200 trials converged, not merely the same count.
    #
    # It is NOT bit-identical, and that is why it is OFF by default rather than merely guarded.
    # XLA selects different float32 kernels per batch width, so a vmap'd g_tilde differs from a
    # one-row-at-a-time g_tilde by a couple of float32 ULPs. On that same fixture nine floats moved
    # at all: the SE ratios and bootstrap SEs by 2e-8 to 1.4e-7 relative (float32 eps), and
    # mean_shift_se -- a difference of means, so cancellation-dominated -- by 1.4e-5 relative,
    # which is 2.2e-7 ABSOLUTE against a gate threshold of at least 0.1. That perturbation is far
    # below every tolerance the checks compare against,
    # but it is not nothing: a solve sitting simultaneously at the convergence tolerance AND at its
    # last permitted iteration could in principle flip converged/failed. Nothing in the measured
    # corpora came near that double boundary (the closest non-accepting final residual was 1.4x the
    # tolerance), but "measured, not proved" is the wrong default for a calibration campaign whose
    # answer key is exactly which runs FAIL, so opting in is deliberate. It also means a report's
    # numbers are reproducible only against the same setting of this field --
    # metrics["resolve_batch_width"] records which path actually ran.
    #
    # "auto" additionally applies the break-even guard below; "always" batches whenever there is
    # more than one trial, which is what an equivalence test wants. Anything unrecognized is
    # treated as "off": unlike perturbation_early_stop -- where the unrecognized value falls
    # through to a stop that provably cannot change an answer -- a typo here must not be able to
    # silently move the arithmetic, so it falls back to the reference path.
    #
    # KNOWN UNSOUND -- do not enable without reading this. Adversarial review (2026-09-01)
    # produced a reproducible counterexample in which flipping ONLY this knob moves
    # check_multiplier_bootstrap from `indeterminate` to `failed`, the classification likewise,
    # and the verdict from `not_certified` to `invalid`, on identical data and seed. The cause is
    # NOT the driver -- that was verified to replay the serial chord-Newton arithmetic exactly
    # over 25 adversarial shapes -- but the float32 error budget: the equivalence argument was
    # measured against |g| (~1e-7 relative), while the accept/reject test compares
    # `g_val - g0 - lam*s_j`, a residual four to five orders SMALLER, where the vmap reduction
    # reorder is worth 0.4-2.7%. A solve sitting near the tolerance can therefore flip
    # converged/failed, which moves the failure fraction and with it the status.
    # It is also SLOWER in every timing taken on the repo's own benchmark fixture (25 draws:
    # 4.98 s serial vs 25.27 s batched; 100 draws: 13.84 s vs 16.70 s), because the pinned vmap
    # width pays for padded rows and an XLA compile the measured ~1.7x dispatch win never repays.
    # Fixing it means giving the accept test hysteresis wide enough to swallow the reorder, not
    # tightening the batching. Until then "auto" is refused outright (see
    # _resolve_bootstrap_batch_width) and only an explicit "always" reaches the driver.
    batched_bootstrap_resolves: str = "off"
    # Trials per wave AND the pinned vmap width (one number deliberately: they are the same
    # buffer). Throughput per row is nearly flat from 25 to 200 at eta_dim 200, so this is chosen
    # for memory rather than for speed. Memory is the real ceiling:
    # every batch row carries its own copy of the full O(n*T) stacked intermediate, so peak usage
    # scales with width -- this is the same construct whose unchunked form is recorded in
    # lifejacket/constants.py as having exhausted a 24GB machine. Lower it if a design is wide;
    # raising it past ~100 buys nothing measurable.
    bootstrap_batch_width: int = 50
    # Break-even guard for "auto", as a projected count of live g_tilde row-evaluations
    # (trials x continuation_steps x a measured median of 4 chord iterations per step). Batching
    # buys one XLA compile of the vmap'd g_tilde -- measured 5-6 s, essentially independent of the
    # width -- which is only repaid above a few thousand rows. At the default 100 paired draws the
    # bootstrap projects ~8,000 rows and batches; check_exact_nonlinear_perturbations' default 15
    # paired directions project ~1,200 and would not, which is why the batched driver is wired into
    # the bootstrap only.
    #
    # The guard is deliberately a pure function of the PLAN'S SHAPE and nothing else. The obvious
    # improvement -- time one g_tilde call and decide from measured cost -- would make the choice
    # of arithmetic depend on machine load, i.e. it could make two runs of the same seed on the
    # same data report different last digits and, at a boundary, a different verdict. A slow
    # decision that is deterministic beats a sharp one that is not. The consequence to know about:
    # the row count alone cannot see how expensive a row IS, so a study whose g_tilde is trivially
    # cheap can clear this threshold and still not repay its compile. Raise the threshold, or set
    # batched_bootstrap_resolves to "off", for such a study.
    bootstrap_batch_min_rows: int = 4000

    # All eight fields above take plain immutable defaults deliberately: @dataclass installs those
    # as CLASS attributes, so a DiagnosticConfig pickled before they existed (analyze_dataset's
    # --diagnostic_config_pickle path ships configs alongside cluster runs) still answers for them
    # by class-attribute fallback and still round-trips through dataclasses.asdict into
    # tolerances_used -- exactly as jacobian_row_chunk_size does. A default_factory here would
    # break that, since dataclasses removes those from the class namespace.

    compute_exact_nonlinear_roots: bool = False
    num_exact_directions: int | None = None

    # Frozen-score multiplier bootstrap (check_multiplier_bootstrap): re-solves the estimating
    # equation shifted by s_b = (1/n) sum_i nu_{b,i} g_i(eta_hat) for iid mean-0/variance-1
    # multipliers nu -- the generalized/weighted bootstrap for Z-estimators (Chatterjee & Bose
    # 2005; Praestgaard-Wellner multipliers), with the multiplier-weighted score frozen at the
    # observed root. Policy adaptivity propagates through the stacked eta-equations exactly as it
    # does in the adjusted sandwich itself; the comparison of bootstrap SEs to sandwich SEs
    # therefore isolates linearization/normal-approximation error, holding the stacked model
    # fixed. Its acceptance band is simulated from the exact finite-draw null (see
    # _simulate_perturbation_null_bands), NOT an engineering tolerance -- unlike most tolerances
    # in this dataclass it needs no simulator calibration.
    # "off" | "auto" | "always": "auto" runs the bootstrap only when check_local_nonlinearity's
    # headline a_{j,l} exceeds bootstrap_screen_a_jl_threshold (or that check did not PASS), using
    # the cheap check as a screen for the expensive one. The 0.05 default screen comes from the
    # ADS-142 calibration experiment (docs/adr/0002): against the exact check's a^NL > 0.10
    # answer key on the borderline cells, screening at 0.05 missed ~11% of exceedances while
    # flagging ~54% of replicates; the default tolerance 0.10 missed ~67%.
    # Defaults to "auto" as of 2026-09-02, previously "off". Two things changed the calculus.
    # (a) Cost: the bootstrap's hours-scale reputation predates the fix that builds and jits
    # g_tilde's closure once instead of rebuilding the O(n*T) precompute per call -- 100 paired
    # draws measured 12.7-21.7 s at n=100/T=50, against ~1.8 h projected under the old closure.
    # (b) Consequence: without it, ANY run whose headline a_{j,l} exceeds the 0.05 screen is
    # UNCERTIFIABLE by construction, because _derive_verdict makes the bootstrap the verdict
    # layer once the screen trips. "off" therefore meant such runs could never be certified, no
    # matter how healthy, and nothing in the output said so.
    # "auto" still costs nothing on a quiet run: the screen has to trip first.
    # CAVEAT worth keeping in view (docs/adr/0002, "Pending, and not to be quoted as measured"):
    # the batched path's verdict-equivalence is measured on CLEAN fixtures only. A fragile
    # ensemble with solves near the convergence boundary, where a ~5e-7 perturbation could flip
    # a convergence flag, is unmeasured. Pass "off" to opt out.
    multiplier_bootstrap: str = "auto"
    num_bootstrap_draws: int = 100
    bootstrap_screen_a_jl_threshold: float = 0.05
    # "rademacher" (default) | "mammen" | "gaussian". Mammen's two-point distribution also
    # matches third moments (better for skewed per-subject contributions); "gaussian" matches
    # sample_perturbation_directions' multiplier LAW but not its draws (that sampler draws from
    # jax.random, this one from np.random.default_rng, so a shared seed gives different streams),
    # and is mostly useful for A/B-ing the two distributions.
    bootstrap_multiplier_distribution: str = "rademacher"
    # Monte Carlo sample count for the finite-draw null bands shared by the bootstrap check and
    # check_exact_nonlinear_perturbations' se_ratios gate. The bands replace the former fixed
    # [0.95, 1.05] se_ratios band, which ignored the J-direction sampling noise of the ensemble
    # covariance and consequently failed 100.0% of 6,648 replicates -- including provably
    # near-affine ones -- in the ADS-142 experiment (max-eigenvalue noise alone spans ~[0.81,
    # 1.20] at J=50 under perfect linearity).
    num_null_band_samples: int = 2000

    compute_influence_and_overlap_checks: bool = True
    compute_leave_one_out_sensitivity: bool = False
    leave_one_out_top_k: int = 3
    report_subject_identifiers: bool = True
    # n_eff is bounded below by 1 (it can never indicate fewer than "one equally influential
    # subject"), so a purely relative floor like `influence_n_eff_min_fraction * n` can never fire
    # for small n even in the most extreme single-subject-dominance case -- hence the separate
    # absolute floor. Like every other tolerance in this dataclass (backward_residual_tolerance is
    # the one exception -- see its comment), these are engineering guesses awaiting simulator
    # calibration, not theorem-derived critical values; see docs/adr/0002-diagnostic-threshold-
    # calibration-plan.md.
    influence_n_eff_min_floor: float = 2.0
    influence_n_eff_min_fraction: float = 0.1
    influence_p_max_tolerance: float = 0.5

    drift_num_directions: int = 3
    drift_path_samples: tuple[float, ...] = (0.0, 0.5, 1.0)
    # Memory bound on check_jacobian_drift's reverse-mode Jacobians of the whole stacked system,
    # resolved by the package-wide helper_functions.resolve_jacobian_row_chunk_size policy that
    # also governs the sandwich's own backward pass: None (auto) chunks only once the output
    # dimension is large enough for an unchunked backward pass to be a memory risk, 0 forces the
    # plain unchunked jax.jacrev, a positive int caps how many output rows are pulled back at
    # once. This matters here more than anywhere else in the suite: the check takes
    # drift_num_directions * len(drift_path_samples) full Jacobians, and the unchunked pass is
    # the one that crashed a 24GB machine at out_dim ~1500 (see lifejacket/constants.py).
    jacobian_row_chunk_size: int | None = None

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

    # Rows of `deltas` evaluated per vmap call in evaluate_g_tilde_batched. Was 1, which meant one
    # vmap dispatch per perturbation -- measured at 0.346 s for 30 rows against 0.035 s at 30, and
    # ~19% of the whole non-bootstrap suite (6.54 s -> 5.29 s at cluster scale; the win is already
    # saturated by 30, with 120 no better).
    #
    # Raising this changes the vmap WIDTH, hence the float32 reduction order -- the same class of
    # change that made batched re-solves flip a verdict (see batched_bootstrap_resolves), and the
    # concern is sharper here than it looks, because the Taylor remainder R = g_plus - g0 - B_delta
    # is a near-cancellation, so an absolute perturbation of g_plus is a much larger RELATIVE
    # perturbation of R. So it was validated rather than assumed, exactly as the divergence abort
    # was: bit-identical metrics across 39 local-nonlinearity fixtures (including maps whose
    # g_tilde genuinely reduces over 400 subjects, since a row-wise toy map cannot exhibit the
    # effect at all), and then on the real jitted closure at study scale -- ADS-142 A_hard, 20
    # paired seeds at chunk 1 vs 30, zero differences in verdict, classification or any check
    # status (cluster jobs 43618469/43618470).
    #
    # Memory is the ceiling, as everywhere else in this package: every row in flight carries its
    # own copy of the full O(n*T) stacked intermediate. At the default num_directions the delta
    # matrix is ~30 rows, so 30 is one chunk rather than a large buffer; a wide study that is
    # memory-tight should lower this, the same way jacobian_row_chunk_size exists for the bread.
    g_tilde_chunk_size: int = 30


@dataclasses.dataclass(frozen=True)
class CriterionResult:
    """
    One criterion of one check, with its measured value and its OWN outcome -- so the summary
    can show, per criterion, what was required, what was measured, and whether THAT criterion
    passed, instead of a prose paragraph the reader has to reconcile with the row's status.

    ok: True passed, False fired, None could not be evaluated on this run's data.
    severity: what an ok=False means for the row -- "fail", "warn", or "indeterminate" --
    mirroring the check's own status logic.
    """

    description: str
    value: str
    ok: bool | None
    severity: str = "fail"


@dataclasses.dataclass
class CheckResult:
    name: str
    status: str
    metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    warnings: list[str] = dataclasses.field(default_factory=list)
    message: str = ""
    # Per-criterion outcomes for the summary. Optional and last: reports pickled before this
    # field existed unpickle without it, so every reader goes through getattr(..., "criteria",
    # []) rather than attribute access.
    criteria: list[CriterionResult] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DiagnosticReport:
    classification: str
    check_results: dict[str, CheckResult]
    # Black-and-white input/data-hygiene correctness results (see run_input_checks), kept
    # separate from check_results so a data-wiring failure is never conflated with a genuine
    # statistical finding about the adjusted sandwich itself.
    input_check_results: dict[str, CheckResult]
    metrics: dict[str, Any]
    tolerances_used: dict[str, Any]
    warnings: list[str]
    monte_carlo_counts: dict[str, int]
    target_labels: list[str]
    rank_diagnostics: dict[str, Any]
    # Decision-level summary derived from the check results (see _derive_verdict and the
    # DiagnosticVerdicts docstring): "certified" / "conservative" / "not_certified" /
    # "invalid", plus how a certification was earned ("bootstrap" vs "screen"). Unlike
    # `classification` -- whose vocabulary is preserved for backward compatibility and whose
    # WARNING-blindness is deliberate -- the verdict makes the calibrated conservative tier
    # visible to automated consumers. Defaults keep reports pickled before this field existed
    # loadable (they simply carry empty strings).
    verdict: str = ""
    verdict_basis: str = ""


# One short action phrase per verdict, printed beside it in the final summary so the reader
# does not need DiagnosticVerdicts' docstring open to know what the word means for them.
_VERDICT_ACTION_PHRASES = {
    # Phrased in terms of the ADJUSTED SANDWICH VARIANCE, not "this CI": the variance estimate
    # is what the analysis definitely reports (the printout's own heading is "Adjusted sandwich
    # variance estimate"), and the suite's evidence is specifically about the adjusted
    # sandwich -- the verdict says nothing either way about the classical sandwich (see the
    # note the verdict key carries).
    DiagnosticVerdicts.CERTIFIED: "report the adjusted sandwich variance",
    DiagnosticVerdicts.CONSERVATIVE: (
        "report the adjusted sandwich variance, but it may be inflated"
    ),
    DiagnosticVerdicts.NOT_CERTIFIED: (
        "DO NOT REPORT the adjusted sandwich variance as validated"
    ),
    DiagnosticVerdicts.INVALID: "DO NOT REPORT the adjusted sandwich variance",
}


_ANSI_RESET = "\x1b[0m"
# 256-color codes for orange and purple (the 16-color palette has neither); plain 32/33/31 for
# green/yellow/red. Verdicts: green CERTIFIED, yellow CONSERVATIVE, orange NOT CERTIFIED, red
# INVALID. Statuses: green PASSED, orange WARNING, purple INDETERMINATE, red FAILED. The two
# no-result outcomes (DID NOT RUN, UNAVAILABLE) render red: an unevaluated run must not look
# calmer than a failed one.
_ANSI_COLORS = {
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "orange": "\x1b[38;5;208m",
    "purple": "\x1b[38;5;141m",
    "red": "\x1b[31m",
    "cyan": "\x1b[36m",
}

# Zero-width sentinels bracketing each criterion's measured VALUE in the laid-out text.
# Values are arbitrary strings (numbers, counts, ranges) that can wrap across lines, so a
# token vocabulary cannot find them the way _COLOR_BY_TOKEN finds statuses; instead
# _criterion_lines brackets each value span per line while it still knows where the value
# is, and format_diagnostic_summary either swaps the sentinels for cyan/reset (color) or
# strips them (plain). Either way they are gone from the returned text, and they are
# inserted only AFTER wrapping and marker alignment, so they never shift the layout they
# annotate.
_VALUE_START = "\x00"
_VALUE_END = "\x01"

# Longest token first: the regex alternation matches left-to-right in this order, which is what
# keeps "NOT CERTIFIED" one orange unit instead of an uncolored NOT beside a green CERTIFIED.
_COLOR_BY_TOKEN = {
    "[not evaluated]": "purple",
    "[indeterminate]": "purple",
    "DO NOT REPORT": "red",
    "NOT CERTIFIED": "orange",
    "[FAIL]": "red",
    "[warn]": "orange",
    "[ok]": "green",
    "INDETERMINATE": "purple",
    "CONSERVATIVE": "yellow",
    "DID NOT RUN": "red",
    "UNAVAILABLE": "red",
    "CERTIFIED": "green",
    "WARNING": "orange",
    "PASSED": "green",
    "FAILED": "red",
    "INVALID": "red",
}
_COLOR_TOKEN_PATTERN = re.compile(
    "|".join(re.escape(token) for token in _COLOR_BY_TOKEN)
)


def _apply_summary_colors(text: str) -> str:
    """
    One regex pass over the finished block, AFTER all layout: ANSI codes are visually
    zero-width but count toward len(), so coloring before the padding/wrapping math would
    shift every column they touch. A single pass also means an already-colored token can
    never be re-matched by a shorter token inside it.
    """
    return _COLOR_TOKEN_PATTERN.sub(
        lambda match: (
            _ANSI_COLORS[_COLOR_BY_TOKEN[match.group(0)]] + match.group(0) + _ANSI_RESET
        ),
        text,
    )


def _verdict_label(verdict: str) -> str:
    """
    The verdict as displayed: upper-cased, with underscores as spaces so NOT_CERTIFIED reads as
    "NOT CERTIFIED". The stored value keeps its underscore -- it is a wire value that pickled
    reports and downstream comparisons depend on.
    """
    return verdict.upper().replace("_", " ")


# How a CERTIFIED verdict was reached (DiagnosticReport.verdict_basis). Rendered only when the
# run actually has a basis, since it is meaningless on the other three verdicts.
_BASIS_LEGEND = {
    VerdictBases.BOOTSTRAP: (
        "the bootstrap ran and its standard errors agreed with the sandwich's"
    ),
    VerdictBases.SCREEN: (
        "the run was quiet enough that the bootstrap was never needed"
    ),
}

# Every heading the legend can print. _legend_lines sizes its heading column from this, so a
# renamed or added heading stays aligned automatically.
_STATUS_HEADING = "CHECK STATUS KEY"
_VERDICT_HEADING = "VERDICT KEY"
_BASIS_HEADING = "CERTIFICATION BASIS KEY"
_CHECKS_HEADING = "CHECKS KEY"
# Only the headings that share the inline column size it. A heading longer than the column --
# "CERTIFICATION BASIS KEY" is 23 characters -- is rendered on its own line by _key_block
# instead, so one long heading cannot push every other block's labels 12 columns to the right
# and squeeze their text into extra wrapping.
_INLINE_LEGEND_HEADINGS = (_STATUS_HEADING, _VERDICT_HEADING, _CHECKS_HEADING)

_SUMMARY_NAME_WIDTH = 34
_SUMMARY_STATUS_WIDTH = 15
_SUMMARY_DETAIL_WIDTH = 78
# The column the detail text starts in, and so also the indent of its continuation lines.
_SUMMARY_DETAIL_INDENT = _SUMMARY_NAME_WIDTH + 2 + _SUMMARY_STATUS_WIDTH
# Rules span the full row rather than a narrower fixed 72, which used to leave every long row
# hanging past the end of its own box.
_SUMMARY_TOTAL_WIDTH = _SUMMARY_DETAIL_INDENT + _SUMMARY_DETAIL_WIDTH

# What each status column value means. Spelled out in the summary itself because this block is
# read by people who did not run the suite and will not go looking for CheckStatuses.
_STATUS_LEGEND = {
    CheckStatuses.PASSED: "check ran and found nothing wrong",
    CheckStatuses.WARNING: "check found something that may or may not matter -- read the detail",
    CheckStatuses.INDETERMINATE: (
        "check could not be evaluated -- this run's data gave it nothing it could measure"
    ),
    CheckStatuses.FAILED: "check measured a real problem",
}


# What each row ASKS, in one line, for the summary's check key. Rendered only for the rows a
# given report actually contains. Full detail for every check -- including the reasoning behind
# each tolerance and what it cannot establish -- is in docs/diagnostics.md.
_CHECK_DESCRIPTIONS = {
    "first_wave_input_checks": (
        "Do the supplied data and configuration pass every basic wiring check -- dozens, "
        "from column dtypes to argument indexing to value finiteness?"
    ),
    "action_probabilities_reconstructed": (
        "Do the recorded action probabilities match what your policy function reproduces?"
    ),
    "estimating_functions_sum_to_zero": (
        "Do the recorded parameters solve their estimating equations, at every update and at "
        "inference?"
    ),
    "root_and_implementation": (
        "Does the final estimate solve its estimating equation, and does this package's math "
        "(the derivative, the linear solve) check out?"
    ),
    "local_nonlinearity": (
        "How far off is the straight-line approximation the variance estimate relies on?"
    ),
    "exact_nonlinear_perturbation": (
        "Re-solving exactly instead of approximating: does the answer move?"
    ),
    "multiplier_bootstrap": (
        "Does a resampling second opinion reproduce the standard errors we report? Note: triggered only when an extra strict local_nonlinearity gate is exceeded"
    ),
    "jacobian_drift": (
        "Does the derivative stay close to the one we used, as we move away from the solution?"
    ),
    "bread_stability": (
        "Are the standard errors well-determined, and stable under tiny numerical nudges?"
    ),
    "influence_concentration": (
        "Do enough subjects drive the result, or does a handful of them?"
    ),
    "exploration_and_weights": (
        "Was there real randomization at every decision, and do the weights stay sane?"
    ),
    "joint_bread_condition_number": (
        "Is the matrix that every solve goes through well-conditioned?"
    ),
    "diagnostic suite": "Did the diagnostic suite itself run to completion?",
}


def _wrap_detail_paragraph(paragraph: str) -> list[str]:
    return textwrap.wrap(
        paragraph,
        width=_SUMMARY_DETAIL_WIDTH,
        # A bulleted paragraph keeps its continuation lines inside the bullet, so
        # consecutive bullets stay visually separate.
        subsequent_indent="  " if paragraph.startswith("- ") else "",
        # Never split a token: these details carry identifiers and numeric literals
        # ("joint_bread_condition_number", "1.473e+13") that are unreadable broken in half.
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]


def _criterion_lines(criterion: CriterionResult) -> list[str]:
    """
    One criterion, laid out: "- <requirement>: <measured value>" wrapped, the value bracketed
    in _VALUE_START/_VALUE_END on every line it touches, and the outcome marker padded flush
    to the detail column's right edge ON THE LAST LINE -- wrapping first and padding after is
    the only order that keeps the markers in a straight column when the text spans lines.
    """
    lines = _wrap_detail_paragraph(f"- {criterion.description}: {criterion.value}")
    # Walk the value back from the end of the wrapped text: it occupies the final
    # len(criterion.value) logical characters, minus one space swallowed at each line break
    # that falls inside it (textwrap consumes exactly the single space between words).
    remaining = len(criterion.value)
    span_start_by_line: dict[int, int] = {}
    for index in range(len(lines) - 1, -1, -1):
        if remaining <= 0:
            break
        content_length = len(lines[index]) - (2 if index else 0)
        take = min(remaining, content_length)
        span_start_by_line[index] = len(lines[index]) - take
        remaining -= take + 1  # the +1 is the space consumed at the wrap point
    for index, start in span_start_by_line.items():
        lines[index] = (
            lines[index][:start] + _VALUE_START + lines[index][start:] + _VALUE_END
        )
    marker = _criterion_marker(criterion)
    # Minus the two sentinels just inserted -- unless the value was empty and none were.
    last_plain_length = len(lines[-1]) - (2 if span_start_by_line else 0)
    if last_plain_length + 2 + len(marker) <= _SUMMARY_DETAIL_WIDTH:
        padding = _SUMMARY_DETAIL_WIDTH - last_plain_length - len(marker)
        lines[-1] = lines[-1] + _marker_leader(padding) + marker
    else:
        # No room on the last line: the marker gets its own line, still flush right (the
        # leader runs from the bullet's hanging indent).
        padding = _SUMMARY_DETAIL_WIDTH - 2 - len(marker)
        lines.append("  " + _marker_leader(padding).lstrip() + marker)
    return lines


def _marker_leader(padding: int) -> str:
    """
    The gap between a criterion's value and its right-aligned outcome marker, exactly `padding`
    characters wide: dot leaders (" .... ") so the eye can track from the value to the marker
    across the row. Degenerates to plain spaces when the gap is too narrow for at least two
    dots -- a single dot reads as a stray period, not a leader.
    """
    if padding - 2 >= 2:
        return " " + "." * (padding - 2) + " "
    return " " * padding


def _summary_row(
    name: str, status: str, detail: str | Sequence[str | CriterionResult]
) -> str:
    """
    One row: name, status, then the detail WRAPPED into its column rather than truncated.

    Truncation was the original behavior and it lost the part that mattered -- these details
    end in the specific numbers and target names that say what actually went wrong, so cutting
    at a fixed width reliably discarded the actionable half and left the boilerplate. Long
    details now continue on following lines, indented to the detail column so the name/status
    columns stay scannable.

    detail is either a plain string (split on its own line breaks -- textwrap.wrap treats \n
    as ordinary whitespace, so without the split a second line would fold back onto the
    first) or a sequence mixing strings with CriterionResults, each of which renders via
    _criterion_lines with its value bracketed for cyan and its outcome marker flush right.
    """
    # The +2s guarantee a gutter even when a name or status fills its column exactly.
    head = (
        f"{name[:_SUMMARY_NAME_WIDTH]:<{_SUMMARY_NAME_WIDTH + 2}}"
        f"{status.upper():<{_SUMMARY_STATUS_WIDTH}}"
    )
    paragraphs = detail.split("\n") if isinstance(detail, str) else list(detail)
    paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    if not paragraphs:
        return head.rstrip()
    wrapped: list[str] = []
    for paragraph in paragraphs:
        if isinstance(paragraph, CriterionResult):
            wrapped.extend(_criterion_lines(paragraph))
        else:
            wrapped.extend(_wrap_detail_paragraph(paragraph))
    lines = [head + wrapped[0]]
    lines.extend(" " * _SUMMARY_DETAIL_INDENT + line for line in wrapped[1:])
    # rstrip is safe against the sentinels: neither \x00 nor \x01 is whitespace.
    return "\n".join(line.rstrip() for line in lines)


def _criterion_marker(criterion: CriterionResult) -> str:
    if criterion.ok is None:
        return "[not evaluated]"
    if criterion.ok:
        return "[ok]"
    return {
        "warn": "[warn]",
        "indeterminate": "[indeterminate]",
    }.get(criterion.severity, "[FAIL]")


def _check_result_detail(result: CheckResult) -> list[str | CriterionResult]:
    """
    The detail paragraphs for one row: any prose message first, then the itemized criteria
    (one line per criterion: requirement, measured value, own outcome marker), then EVERY
    message the check reported, labeled as such and bulleted.

    Two earlier behaviors are deliberately gone, both casualties of the same failing-run
    review. `message or warnings[0]` hid every warning the moment a message existed; and the
    "(+N more flags)" elision hid all but the first warning -- on a genuinely failing run the
    hidden ones were exactly the explanation (a check whose listed gates were all satisfied
    still failed, and the reason sat behind "+4 more flags"). The label matters as much as the
    completeness: the criterion lines are the summary's own rendering, while these are
    verbatim findings from inside the check, and a reader has to be able to tell which is
    which -- especially because a check's status can be set by its findings, not only by the
    criteria above them.
    """
    paragraphs = []
    if result.message:
        # A message may carry its own line breaks; each becomes its own paragraph.
        paragraphs.extend(line for line in result.message.split("\n") if line)
    criteria = getattr(result, "criteria", [])
    if criteria:
        paragraphs.append("criteria:")
        paragraphs.extend(criteria)
    if result.warnings:
        paragraphs.append("messages from the check:")
        paragraphs.extend(f"- {warning}" for warning in result.warnings)
    return paragraphs


def _key_lines(
    heading: str, label: str, text: str, heading_width: int, label_width: int
):
    """
    One legend entry -- heading, label, then text wrapped into whatever width is left and
    indented under itself. Wrapped for the same reason the rows are: the longest check name is
    34 characters, so a fixed-width key line would otherwise run past the block's own rules.
    """
    indent = heading_width + label_width
    wrapped = textwrap.wrap(
        text,
        width=max(_SUMMARY_TOTAL_WIDTH - indent, 20),
        break_long_words=False,
        break_on_hyphens=False,
    ) or [""]
    lines = [f"{heading:<{heading_width}}{label:<{label_width}}{wrapped[0]}".rstrip()]
    lines.extend(" " * indent + continued for continued in wrapped[1:])
    return lines


def _key_block(heading, entries, heading_width, label_width):
    """
    One legend block: its heading, then one entry per (label, text) pair.

    A heading that does not fit the inline column is emitted on its own line and the entries
    start beneath it. Without that, a long heading either collides with its first label
    ("CERTIFICATION BASIS KEYBOOTSTRAP") or forces every block to indent to its width.
    """
    lines = []
    inline_heading = heading if len(heading) < heading_width else ""
    if not inline_heading:
        lines.append(heading)
    for index, (label, text) in enumerate(entries):
        lines.extend(
            _key_lines(
                inline_heading if index == 0 else "",
                label,
                text,
                heading_width,
                label_width,
            )
        )
    return lines


def _legend_lines(row_names: Sequence[str] = (), verdict_basis: str = "") -> list[str]:
    """
    The status, verdict and check keys.

    The first two are sourced from CheckStatuses and _VERDICT_ACTION_PHRASES rather than
    restated, so a new status or a reworded verdict action cannot drift out of sync here. The
    check key covers only the rows THIS report contains: the opt-in checks (exact perturbation,
    multiplier bootstrap, jacobian drift) usually do not run, and describing checks that
    produced no row would pad the block with answers to questions nobody asked.
    """
    # Sized from the INLINE headings only; _key_block puts an over-wide heading on its own
    # line. Sizing from every heading instead would fix the collision but indent all four
    # blocks to the longest one.
    heading_width = max(len(heading) for heading in _INLINE_LEGEND_HEADINGS) + 2

    # ORDER: the checks key leads, because it is what a first-time reader needs before any row
    # below makes sense; the status and verdict keys interpret the columns; the basis key only
    # applies when a run was certified.
    lines: list[str] = []
    described = [name for name in row_names if name in _CHECK_DESCRIPTIONS]
    if described:
        lines.extend(
            _key_block(
                _CHECKS_HEADING,
                [(name, _CHECK_DESCRIPTIONS[name]) for name in described],
                heading_width,
                max(len(name) for name in described) + 2,
            )
        )
        lines.append("")
        lines.append(
            f"{'':<{heading_width}}Quantities quoted in SE are fractions of the reported "
            "standard error."
        )
        lines.append(
            f"{'':<{heading_width}}Full detail for every check: docs/diagnostics.md"
        )
        lines.append("")

    lines.extend(
        _key_block(
            _STATUS_HEADING,
            [(status.upper(), meaning) for status, meaning in _STATUS_LEGEND.items()],
            heading_width,
            max(len(status) for status in _STATUS_LEGEND) + 2,
        )
    )

    lines.append("")
    lines.append(
        f"{'':<{heading_width}}A suite row lists every criterion that can set its status, "
        "each with its measured value"
    )
    lines.append(
        f"{'':<{heading_width}}and its own outcome: [ok], [FAIL], [warn] (fires as a "
        "WARNING), [indeterminate], or"
    )
    lines.append(
        f'{"":<{heading_width}}[not evaluated]. The "messages from the check:" bullets are '
        "the check's own findings --"
    )
    lines.append(f"{'':<{heading_width}}the specifics of whichever criterion fired.")

    # Blank line between the keys: they answer different questions (what one row means, what
    # the whole run means, how it was certified, what each row is asking) and run together as
    # one undifferentiated block without it.
    lines.append("")
    lines.extend(
        _key_block(
            _VERDICT_HEADING,
            [
                (_verdict_label(verdict), action)
                for verdict, action in _VERDICT_ACTION_PHRASES.items()
            ],
            heading_width,
            max(len(_verdict_label(v)) for v in _VERDICT_ACTION_PHRASES) + 2,
        )
    )

    lines.append("")
    lines.append(
        f"{'':<{heading_width}}The verdict gates only the adjusted sandwich variance."
    )
    lines.append(
        f"{'':<{heading_width}}The parameter estimate and the classical sandwich are computed "
        "and saved either"
    )
    lines.append(
        f"{'':<{heading_width}}way -- but a flagged run does NOT mean the classical sandwich "
        "is accurate instead."
    )

    if verdict_basis:
        lines.append("")
        lines.extend(
            _key_block(
                _BASIS_HEADING,
                [(basis.upper(), meaning) for basis, meaning in _BASIS_LEGEND.items()],
                heading_width,
                max(len(basis) for basis in _BASIS_LEGEND) + 2,
            )
        )

    return lines


def _verdict_next_step_lines(report: DiagnosticReport | None) -> list[str]:
    """
    The actionable next step, when there is one the summary can name.

    Exists because UNCERTIFIABLE has two very different causes that rendered identically: an
    unevaluable check (nothing the reader can do without changing the data), and the a_{j,l}
    screen calling for the multiplier bootstrap when it was not run -- which is entirely
    fixable by re-running with it on. A reader of the old block could not tell which they had,
    and nothing anywhere told them certification was still available.

    Rendered AFTER the verdict rather than folded into it so the verdict line stays one
    scannable sentence.
    """
    if report is None or report.verdict != DiagnosticVerdicts.NOT_CERTIFIED:
        return []
    if "multiplier_bootstrap" in report.check_results:
        return []
    local = report.check_results.get("local_nonlinearity")
    if local is None:
        return []
    headline = _local_nonlinearity_headline_max(local.metrics)
    # tolerances_used is dataclasses.asdict(config), so the run's own settings are on the
    # report -- no need to thread the config in, and a re-read report stays self-describing.
    screen = report.tolerances_used.get("bootstrap_screen_a_jl_threshold")
    mode = report.tolerances_used.get("multiplier_bootstrap")
    if screen is None or math.isnan(headline) or headline <= screen:
        return []
    text = (
        f"NEXT STEP: this is fixable. The linearization error ({headline:.3g} SE) is above "
        f"the {screen:g} SE screen, so certifying this run requires the multiplier "
        f"bootstrap's second opinion -- and it did not run "
        f"(multiplier_bootstrap={mode!r}). Re-run with "
        f'multiplier_bootstrap="auto"; if the bootstrap reproduces the reported standard '
        f"errors, the verdict becomes CERTIFIED."
    )
    wrapped = textwrap.wrap(
        text,
        width=_SUMMARY_TOTAL_WIDTH,
        subsequent_indent=" " * len("NEXT STEP: "),
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped


def diagnostics_flagged(report: DiagnosticReport | None) -> bool:
    """
    Single definition of "did the diagnostics flag this run" for job-level consequences (the
    final summary's tone, gating the estimate printout behind consent, and the CLI's nonzero
    exit): True exactly when the verdict says the CI should not be reported (UNCERTIFIABLE or
    INVALID -- see DiagnosticVerdicts), falling back to classification == failed for reports
    pickled before the verdict field existed, and True when there is no report at all (the
    suite was requested but did not produce one): an unevaluated run must not look like a
    passing one to automation.
    """
    if report is None:
        return True
    if report.verdict:
        return report.verdict in (
            DiagnosticVerdicts.NOT_CERTIFIED,
            DiagnosticVerdicts.INVALID,
        )
    return report.classification == DiagnosticClassifications.FAILED


def format_diagnostic_summary(
    report: DiagnosticReport | None,
    pipeline_rows: Sequence[
        tuple[str, str, str | Sequence[str | CriterionResult]]
    ] = (),
    suite_error: str = "",
    color: bool | None = None,
) -> str:
    """
    Renders the end-of-run diagnostic summary: one status row per input check and per suite
    check, then the caller's pipeline-level rows (quantities computed outside the suite, e.g.
    the bread condition number and the local linearization ratio, as (name, status, detail)
    triples), then the decision-level verdict. This is the one block a reader who scrolled
    straight to the bottom must be able to act on -- the estimates print AFTER it, so a failed
    run cannot end with a wall of plausible-looking numbers.

    `report` None means the suite was requested but produced no report (crashed, in
    `suite_error`); that renders as an explicit DID NOT RUN row and an UNAVAILABLE verdict,
    which diagnostics_flagged treats as flagged.
    """
    # Row names are collected first so the check key describes exactly the rows that follow.
    row_names = (
        ["diagnostic suite"]
        if report is None
        else [*report.input_check_results, *report.check_results]
    )
    row_names += [name for name, _, _ in pipeline_rows]
    lines = [
        "=" * _SUMMARY_TOTAL_WIDTH,
        "DIAGNOSTIC SUMMARY",
        "-" * _SUMMARY_TOTAL_WIDTH,
        # The key goes ABOVE the rows, not below: the verdict has to stay the last substantive
        # line (a reader who scrolls to the bottom must land on the decision), so the only
        # other place for it would be wedged between the rows and the verdict.
        *_legend_lines(row_names, "" if report is None else report.verdict_basis),
        "-" * _SUMMARY_TOTAL_WIDTH,
    ]
    if report is None:
        detail = f"error: {suite_error}" if suite_error else ""
        lines.append(_summary_row("diagnostic suite", "did not run", detail))
        verdict_line = (
            "VERDICT: UNAVAILABLE -- the diagnostic suite produced no report; "
            "treat this run as unvalidated"
        )
    else:
        for name, result in report.input_check_results.items():
            lines.append(
                _summary_row(name, result.status, _check_result_detail(result))
            )
        for name, result in report.check_results.items():
            lines.append(
                _summary_row(name, result.status, _check_result_detail(result))
            )
        verdict = report.verdict or report.classification
        action = _VERDICT_ACTION_PHRASES.get(verdict, "")
        basis = f" (basis: {report.verdict_basis})" if report.verdict_basis else ""
        verdict_line = f"VERDICT: {_verdict_label(verdict)}{basis}"
        if action:
            verdict_line += f" -- {action}"
    for name, status, detail in pipeline_rows:
        lines.append(_summary_row(name, status, detail))
    lines.append("-" * _SUMMARY_TOTAL_WIDTH)
    lines.append(verdict_line)
    lines.extend(_verdict_next_step_lines(report))
    lines.append("=" * _SUMMARY_TOTAL_WIDTH)
    text = "\n".join(lines)
    if color is None:
        # Auto: color only a real terminal, and honor the NO_COLOR convention. Captured
        # output -- pytest, subprocess pipes, log files -- stays plain, so nothing that
        # greps or pins this block ever sees an escape code it did not ask for.
        color = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    if not color:
        return text.replace(_VALUE_START, "").replace(_VALUE_END, "")
    # Tokens first (none can straddle a sentinel), then the value sentinels become cyan.
    text = _apply_summary_colors(text)
    return text.replace(_VALUE_START, _ANSI_COLORS["cyan"]).replace(
        _VALUE_END, _ANSI_RESET
    )


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
    # Build the vmap ONCE, outside the loop. Constructing it inside gave every chunk a fresh
    # lambda, which jax cannot match against its trace cache, so each chunk paid a full retrace
    # of g_tilde -- ~20% of this function's cost at the default chunk_size of 1, where there is
    # one chunk per row. This is a pure dispatch win and changes NOTHING about the arithmetic:
    # same chunk widths, same reduction order within each chunk, one trace instead of N. (It is
    # deliberately not the same kind of change as raising chunk_size, which alters the vmap WIDTH
    # and therefore the float32 reduction order -- see the note on g_tilde_chunk_size.)
    evaluate_chunk = jax.vmap(lambda d: g_tilde(eta_hat + d))
    outputs = []
    for start in range(0, num_rows, max(1, chunk_size)):
        outputs.append(evaluate_chunk(deltas[start : start + chunk_size]))
    return np.asarray(jnp.concatenate(outputs, axis=0))


###############################################################################
# Section 1: input/data-hygiene checks -- black-and-white correctness questions about the
# supplied data/functions (does a reconstruction match, does an equation average to zero to a
# loose absolute tolerance), not numeric measurements of how appropriate the adjusted sandwich is
# for this experiment. Kept as their own CheckResults, in DiagnosticReport.input_check_results,
# never folded into check_results -- a data-wiring failure should never be conflated with a
# genuine statistical finding just because both happen to raise/fail.
###############################################################################


def run_input_checks(
    legacy_check_callables: Sequence[tuple[str, Callable[[], Any]]],
) -> dict[str, CheckResult]:
    """
    Re-runs any supplied legacy lifejacket.input_checks functions, converting each one's hard
    raise into its own failed CheckResult rather than propagating -- a non-interactive,
    structured, always-present outcome for callers wiring their own checks into
    run_diagnostic_suite (e.g. an unattended cluster job with nobody to answer an interactive
    "(y/n)" prompt and raise). analyze_dataset itself no longer passes any callables here
    (2026-09-02): its one wired-in check, action-probability reconstruction, is the most
    expensive input check (it evaluates action_prob_func over every active row) and was being
    executed twice per analysis. It already runs in the first-wave input checks, is a hard
    failure with no interactive continue path, and nothing between that run and the suite
    touches its inputs -- so analyze_dataset records the first-wave outcome into
    DiagnosticReport.input_check_results directly instead of re-executing it.
    """
    results: dict[str, CheckResult] = {}
    for check_name, check_callable in legacy_check_callables:
        try:
            check_callable()
            results[check_name] = CheckResult(
                name=check_name, status=CheckStatuses.PASSED
            )
        except Exception as exc:  # noqa: BLE001 - legacy checks raise assorted exception types
            results[check_name] = CheckResult(
                name=check_name, status=CheckStatuses.FAILED, message=str(exc)
            )
    return results


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
    rng: np.random.Generator | None = None,
) -> CheckResult:
    """
    Section 2 (hard prerequisite gate). Computes the root-error correction in target standard-
    error units, checks finiteness of g0/B_hat/M_hat, hard-fails on an anomalously large backward
    residual of the linear solve used to compute that correction (a software-correctness smoke
    detector, not a statistical judgment call -- see the comment on
    DiagnosticConfig.backward_residual_tolerance), and spot-checks the automatic derivative B_hat
    against a directional finite difference of g_tilde. Purely numeric measurements of the
    supplied estimating-function/derivative pair -- see run_input_checks for the separate,
    black-and-white input/data-hygiene checks.
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
        if backward_relative_residual > config.backward_residual_tolerance:
            status = CheckStatuses.FAILED
            failure_reasons.append(
                f"Backward relative residual of {backward_relative_residual:.4g} exceeds "
                f"{config.backward_residual_tolerance:.4g} -- this indicates a broken linear "
                "solve (e.g. a stale/mismatched bread_factored, or a bug in "
                "factor_bread/solve_with_bread), not a statistical finding: a healthy LU solve "
                "is backward-stable regardless of B_hat's conditioning."
            )

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

    # Per-criterion outcomes, each with its own measured value. The metric keys used below are
    # only populated inside the `if finite_ok:` branch, so on a nonfinite g0/B_hat/M_hat run
    # (which hard-fails) the dependent criteria report [not evaluated] rather than KeyError.
    fd_error = metrics.get("finite_difference_max_relative_error", math.nan)
    criteria = [
        CriterionResult(
            description=(
                "the estimating-function value g0, its derivative B_hat, and its "
                "covariance M_hat are all finite"
            ),
            value="yes" if finite_ok else "no",
            ok=bool(finite_ok),
        ),
        CriterionResult(
            description=(
                f"estimate within {config.root_error_tolerance_se:g} SE of exactly solving "
                "its estimating equation (root_error_tolerance_se)"
            ),
            value=f"{metrics['a_root_max']:.3g} SE" if finite_ok else "not evaluated",
            ok=bool(metrics["a_root_max"] <= config.root_error_tolerance_se)
            if finite_ok
            else None,
        ),
        CriterionResult(
            description=(
                "the internal linear algebra is solved cleanly -- leftover solve error "
                f"at most {config.backward_residual_tolerance:g} "
                "(backward_residual_tolerance)"
            ),
            value=f"{metrics['backward_relative_residual']:.1e}"
            if finite_ok
            else "not evaluated",
            ok=bool(
                metrics["backward_relative_residual"]
                <= config.backward_residual_tolerance
            )
            if finite_ok
            else None,
        ),
        CriterionResult(
            description=(
                "our computed derivative agrees with one measured directly by tiny "
                "nudges (finite differences) within 1%"
            ),
            value=f"{fd_error:.2%} worst disagreement"
            if not math.isnan(fd_error)
            else "not evaluated",
            ok=bool(fd_error <= 1e-2) if not math.isnan(fd_error) else None,
            severity="warn",
        ),
    ]
    # The failure prose (if any) stays in message; the measurements now live in the criteria.
    return CheckResult(
        name="root_and_implementation",
        status=status,
        metrics=metrics,
        warnings=warnings,
        message="; ".join(failure_reasons),
        criteria=criteria,
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

    Rows where g_tilde is undefined at eta_hat + delta_j (any nonfinite output) are CENSORED:
    r_j/c_j/q_j come back as NaN, and the returned boolean "finite" mask marks them so the caller
    can report a domain-censoring fraction. Such rows must never reach solve_with_bread: its LU
    solve is check_finite=True, so one nonfinite row would raise on the whole stacked right-hand
    side and take every other direction's measurement -- and, since this runs inside the
    always-on local-nonlinearity check, the entire diagnostic report -- down with it. Probing out
    to 1.5x the sandwich scale routinely leaves the domain of a log/logit/sqrt-shaped estimating
    function, so a censored row is an expected outcome to be reported, not an error.
    """
    B = np.asarray(B_hat, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    g_plus = evaluate_g_tilde_batched(g_tilde, eta_hat, deltas, chunk_size=chunk_size)
    B_delta = (B @ deltas.T).T
    R = g_plus - g0 - B_delta
    finite = np.all(np.isfinite(R), axis=1)

    B_delta_norms = np.linalg.norm(B_delta, axis=1)
    r = np.divide(
        np.linalg.norm(R, axis=1),
        B_delta_norms,
        out=np.full(R.shape[0], np.inf),
        where=B_delta_norms > 0,
    )
    r[~finite] = math.nan

    c = np.full((R.shape[0], B.shape[0]), math.nan)
    if np.any(finite):
        c[finite] = solve_with_bread(bread_factored, -R[finite].T).T
    delta_norms = np.linalg.norm(deltas, axis=1)
    q = np.divide(
        np.linalg.norm(c, axis=1),
        delta_norms,
        out=np.full(c.shape[0], np.inf),
        where=delta_norms > 0,
    )
    q[~finite] = math.nan

    return {
        "R": R,
        "r": r,
        "c": c,
        "q": q,
        "g_plus": g_plus,
        "B_delta": B_delta,
        "finite": finite,
    }


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
    # (censored probes, total probes) per radius; see evaluate_taylor_remainder_and_correction.
    censoring_by_radius: dict[float, tuple[int, int]] = {}

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
        finite_all = np.concatenate([res["finite"] for res in sign_results])
        r_by_radius[radius] = r_all
        q_by_radius[radius] = q_all
        censoring_by_radius[radius] = (
            int(np.sum(~finite_all)),
            int(finite_all.size),
        )

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
            "num_probes": censoring_by_radius[radius][1],
            "num_domain_censored_probes": censoring_by_radius[radius][0],
            "domain_censored_fraction": (
                censoring_by_radius[radius][0] / censoring_by_radius[radius][1]
                if censoring_by_radius[radius][1]
                else math.nan
            ),
        }
        # Exceedance rates are computed over the EVALUABLE probes only: counting a censored
        # probe as a non-exceedance would let domain failures dilute the rate toward zero.
        for label, a in a_all_by_target.items():
            evaluable = a[np.isfinite(a)]
            exceed = evaluable > config.nonlinear_correction_tolerance_se
            frac = float(np.mean(exceed)) if evaluable.size else math.nan
            per_radius[radius]["a_by_target"][label]["exceedance_fraction"] = frac
            per_radius[radius]["a_by_target"][label]["exceedance_upper_bound"] = (
                clopper_pearson_upper_bound(
                    int(np.sum(exceed)), int(evaluable.size), config.confidence_level
                )
                if evaluable.size
                else math.nan
            )

        if config.paired_directions:
            plus_c, minus_c = sign_results[0]["c"], sign_results[1]["c"]
            even_norms = np.linalg.norm(0.5 * (plus_c + minus_c), axis=1)
            odd_norms = np.linalg.norm(0.5 * (plus_c - minus_c), axis=1)
            paired_evaluable = np.isfinite(even_norms) & np.isfinite(odd_norms)
            per_radius[radius]["paired_even_norm_median"] = (
                float(np.median(even_norms[paired_evaluable]))
                if np.any(paired_evaluable)
                else math.nan
            )
            per_radius[radius]["paired_odd_norm_median"] = (
                float(np.median(odd_norms[paired_evaluable]))
                if np.any(paired_evaluable)
                else math.nan
            )

    # Below this floor, r_j/a_{j,l} are dominated by float32 roundoff rather than genuine
    # curvature signal, and their ratio across radii is meaningless noise -- skip the scaling
    # check entirely rather than warn on it (this is exactly the affine-map case, where r_j and
    # a_{j,l} should be ~0 at every radius and there is no power law to observe).
    _scaling_noise_floor = 1e-4
    # Domain-censoring limits at the headline radius, past which the surviving probes are too few
    # or too selected to characterize the neighborhood at all. Three matches the minimum ensemble
    # size the exact/bootstrap checks require before they will compute any ensemble statistic.
    _min_evaluable_probes = 3
    _severe_censored_fraction = 0.5

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
        # This branch -- the check's HEADLINE finding -- used to set WARNING and say nothing:
        # it appended no warning and the CheckResult below is built with message="", so the
        # end-of-run summary rendered a bare "local_nonlinearity WARNING" with an empty detail
        # column and the measured number reachable only by unpickling metrics["per_radius"].
        # A status with no stated reason is not actionable, and it is the one number a reader
        # needs to judge whether the exceedance is marginal or severe.
        # The figure itself is in `message` (printed on every row, pass or not), so this says
        # only what exceeding the gate MEANS -- otherwise a WARNING row prints it twice.
        warnings_list.append(
            "The warn criterion above is a rule of thumb, not a calibrated number; "
            "certification is decided by the separate bootstrap screen."
        )
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

    total_censored = sum(censored for censored, _ in censoring_by_radius.values())
    total_probes = sum(probes for _, probes in censoring_by_radius.values())
    headline_censored, headline_probes = censoring_by_radius[headline_radius]
    headline_evaluable = headline_probes - headline_censored
    # Domain censoring is never allowed to read as clean. The probes that survived are exactly
    # the directions in which g_tilde stayed defined -- the well-behaved ones -- so every summary
    # above is computed on an optimistically selected subset: a measured exceedance on it still
    # stands, but a clean pass on it does not (the same precedence the exact and bootstrap checks
    # apply to their non-converged directions). Once too few probes survive at the headline
    # radius, or most of them are gone, the subset cannot support any verdict at all.
    if total_censored:
        warnings_list.append(
            f"g_tilde was undefined (nonfinite) at {total_censored} of {total_probes} probe "
            f"points ({headline_censored}/{headline_probes} at the headline radius "
            f"{headline_radius}); those directions are censored from every statistic above, "
            "which selects toward the well-behaved part of the neighborhood."
        )
        # Escalation, not just degradation: each censored probe is a MEASURED event -- g_tilde
        # was nonfinite at that point. Partial censoring degrades the sample (a pass on the
        # survivors is untrustworthy, hence INDETERMINATE), but TOTAL censoring at the headline
        # radius is direct evidence: the estimating function is undefined everywhere probed at
        # sampling scale, so the linearization the adjusted sandwich relies on has no domain
        # there. That is a failure the check measured, not one it failed to measure.
        if headline_probes and headline_evaluable == 0:
            status = CheckStatuses.FAILED
            warnings_list.append(
                f"g_tilde was nonfinite at ALL {headline_probes} probe points at the "
                f"headline radius {headline_radius}: the linearization the adjusted "
                "sandwich variance relies on is undefined across the sampling-scale "
                "neighborhood of the estimate -- a measured failure, not an unevaluable "
                "probe set."
            )
        elif (
            headline_evaluable < _min_evaluable_probes
            or headline_censored > _severe_censored_fraction * headline_probes
        ):
            status = CheckStatuses.INDETERMINATE
        elif status == CheckStatuses.PASSED:
            status = CheckStatuses.WARNING

    # Per-criterion outcomes. headline_max is NaN when every probe at the headline radius was
    # domain-censored -- that criterion then reads [not evaluated] rather than showing "nan SE"
    # as if it were a measurement (the censoring message carries that story).
    rank_deficient_any = any(
        per_radius[radius]["joint_mahalanobis"]["rank_deficient"]
        for radius in per_radius
    )
    criteria = [
        CriterionResult(
            description=(
                "the straight-line approximation is off by at most "
                f"{config.nonlinear_correction_tolerance_se:g} SE when probed "
                f"{headline_radius}x a typical sampling fluctuation away "
                "(nonlinear_correction_tolerance_se)"
            ),
            value=f"{headline_max:.3g} SE"
            if not math.isnan(headline_max)
            else "no evaluable probe",
            ok=bool(headline_max <= config.nonlinear_correction_tolerance_se)
            if not math.isnan(headline_max)
            else None,
            severity="warn",
        ),
        CriterionResult(
            description=(
                "no degenerate directions: the reported estimates' covariance has full "
                "rank, so every component is actually probed"
            ),
            value="rank-deficient" if rank_deficient_any else "yes",
            ok=not rank_deficient_any,
            severity="indeterminate",
        ),
        CriterionResult(
            description=(
                "enough probe points stayed well-defined at the headline distance (at "
                f"least {_min_evaluable_probes} evaluable, at most "
                f"{_severe_censored_fraction:.0%} undefined; NONE evaluable is a measured "
                "failure)"
            ),
            value=f"{headline_evaluable} of {headline_probes} evaluable",
            ok=not (
                headline_evaluable < _min_evaluable_probes
                or headline_censored > _severe_censored_fraction * headline_probes
            ),
            # Partial censoring only degrades the evidence (INDETERMINATE); zero evaluable
            # probes is direct evidence the linearization has no domain at sampling scale.
            severity="fail"
            if headline_probes and headline_evaluable == 0
            else "indeterminate",
        ),
    ]
    return CheckResult(
        name="local_nonlinearity",
        status=status,
        criteria=criteria,
        metrics={
            "per_radius": per_radius,
            "headline_radius": headline_radius,
            "num_probes": total_probes,
            "num_domain_censored_probes": total_censored,
            "domain_censored_fraction": (
                total_censored / total_probes if total_probes else math.nan
            ),
        },
        warnings=warnings_list,
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

    DIVERGENCE ABORT (config.nonlinear_solver_divergence_abort, on by default). A continuation
    step can only fail by exhausting nonlinear_solver_max_iterations, so without this a step that
    has demonstrably left the chord method's basin still pays its whole budget. Two clauses stop
    it early -- the iterate blew four orders of magnitude past its own best residual, or it has
    gone stall_window iterations without a new best -- and an aborted step is failed exactly as an
    exhausted one is.

    Why that is verdict-neutral: both callers consume only "converged" (which gates every
    ensemble contribution), "delta_nl" (read only when converged), "nonfinite_encountered" (a
    reported metric -- no status rung reads it) and "branch_change_suspected" (which is
    `converged and ...`, hence False for any failure). So an aborted solve is OBSERVATIONALLY
    IDENTICAL to one that exhausted its budget, and the abort can move a status only by aborting
    a solve that would eventually have converged -- which is what the deliberately wide margins on
    DiagnosticConfig buy, and why that field documents the measured margins rather than a
    rationale.

    One honest caveat, and it is not small on some maps: a solve that would have gone non-finite
    at a LATER iteration is now recorded as a root failure rather than a domain failure, so
    domain_failure_fraction / domain_failure_upper_bound become lower bounds (both are reported,
    neither is gated by any status rung). Measured on the diagnostic fixtures, this erases the
    domain-failure count entirely on POLYNOMIAL maps -- an entire function whose "domain failure"
    was really float64 overflow of a divergent iterate two iterations after the blow-up clause
    fires, which arguably makes the new label the accurate one -- while leaving it untouched on
    bounded maps, where failures stall rather than blow up. The checks report
    num_divergence_aborted_trials next to it so the shift is attributable rather than silent.
    """
    d = np.array(delta_linear, dtype=np.float64)
    g0 = np.asarray(g0, dtype=np.float64)
    lambdas = np.linspace(0.0, 1.0, config.continuation_steps + 1)[1:]
    total_iterations = 0
    nonfinite_encountered = False
    converged = True
    final_residual_norm = math.nan
    aborted = False
    abort_reason = ""
    # Scale convergence relative to the FULL perturbation ||s_j||, not the current lambda-scaled
    # target: using ||lam * s_j|| as the denominator would make the "relative" residual blow up
    # spuriously at small lambda, since the absolute floating-point noise floor in g_tilde's
    # output does not shrink proportionally to lambda.
    full_scale = max(float(np.linalg.norm(s_j)), 1e-12)
    abort_enabled = config.nonlinear_solver_divergence_abort
    blowup_factor = config.nonlinear_solver_divergence_blowup_factor
    stall_window = config.nonlinear_solver_divergence_stall_window
    abort_guard = (
        config.nonlinear_solver_divergence_guard_factor
        * config.nonlinear_solver_tolerance
    )

    for lam in lambdas:
        target = lam * s_j
        step_converged = False
        # The abort's state is PER CONTINUATION STEP: the target moves at every lambda, so the
        # residual legitimately jumps when a step begins, and a best carried over from the
        # previous step would be an unreachable bar that reads as an instant stall.
        best_relative_norm = math.inf
        iterations_since_best = 0
        for _ in range(config.nonlinear_solver_max_iterations):
            g_val = np.asarray(g_tilde(jnp.asarray(eta_hat) + jnp.asarray(d)))
            total_iterations += 1
            if not np.all(np.isfinite(g_val)):
                nonfinite_encountered = True
                break
            residual = g_val - g0 - target
            residual_norm = float(np.linalg.norm(residual))
            final_residual_norm = residual_norm
            relative_norm = residual_norm / full_scale
            if relative_norm < config.nonlinear_solver_tolerance:
                step_converged = True
                break
            # Tested only AFTER the tolerance test, so a step that has just converged can never be
            # aborted, and final_residual_norm is already assigned either way.
            if relative_norm < best_relative_norm:
                best_relative_norm = relative_norm
                iterations_since_best = 0
            else:
                iterations_since_best += 1
                if abort_enabled and best_relative_norm > abort_guard:
                    if relative_norm > blowup_factor * best_relative_norm:
                        aborted, abort_reason = True, "blowup"
                    elif stall_window > 0 and iterations_since_best >= stall_window:
                        aborted, abort_reason = True, "stall"
                    if aborted:
                        break
            d = d - solve_with_bread(bread_factored, residual)
        if nonfinite_encountered:
            converged = False
            break
        # An abort leaves step_converged False, so it falls through here and fails the solve on
        # exactly the same path an exhausted iteration budget takes -- no separate control flow,
        # which is what keeps the two observationally identical.
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
        # Reported so an aborted solve stays distinguishable from one that genuinely exhausted its
        # iteration budget, even though nothing downstream is allowed to treat them differently.
        "aborted": aborted,
        "abort_reason": abort_reason,
        "branch_change_suspected": branch_change_suspected,
        "discrepancy_ratio": discrepancy_ratio,
    }


# Per-trial states of the batched lockstep driver below. FAILED covers both an exhausted iteration
# budget and a divergence abort, exactly as solve_exact_perturbation's single `converged` flag does
# -- the two are deliberately indistinguishable downstream (see its docstring); which one occurred
# is reported separately via "aborted"/"abort_reason".
_TRIAL_RUNNING = 0
_TRIAL_CONVERGED = 1
_TRIAL_FAILED = 2
_TRIAL_NONFINITE = 3

# Median chord-Newton iterations per continuation step on clean data, measured across the
# repository's benchmark fixtures at three scales (per-step min 3 / median 4 / max 6 at
# n=100/T=50). Used only to project a row count for the "auto" break-even guard, never in a
# numerical path: an under-projection costs a serial run, never a wrong answer.
_MEDIAN_ITERATIONS_PER_CONTINUATION_STEP = 4


def _make_batched_g_tilde_evaluator(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    batch_width: int,
) -> Callable[[np.ndarray], Any]:
    """
    Builds the ONE jitted, vmap'd g_tilde the batched driver calls for the life of a check, and
    pays its compile up front on a probe block so the first real wave is not charged for it.

    Hoisting this is not a tidiness preference, it is most of the optimization. jax.jit caches its
    compiled executable on the identity of the wrapped callable plus the argument shape, so a
    FRESH jax.jit(jax.vmap(lambda ...)) object retraces and recompiles from scratch -- measured 4.4
    s even at a width XLA had already compiled for an earlier object. Constructing the evaluator
    per wave, or per chord iteration, therefore trades the entire win away in recompiles; that is
    exactly the mistake evaluate_g_tilde_batched makes today by building its vmap inside the chunk
    loop, which is why that helper is slower than not batching at chunk_size 1.

    For the same reason the width is PINNED: XLA compiles per distinct input shape at 5-6 s a
    shape, essentially independent of the width itself, so compacting the active set down to its
    exact live count each iteration (17 distinct widths on a clean ensemble, 77 on a fragile one)
    would cost 100-470 s of compilation to save a couple of seconds of padded rows. The driver
    compacts by gathering live rows into the low slots of this fixed-width buffer instead.
    """
    eta_j = jnp.asarray(eta_hat)

    @jax.jit
    def evaluate(block):
        return jax.vmap(lambda row: g_tilde(eta_j + row))(block)

    # Probe at the exact call shape the driver uses -- a float64 numpy block handed through
    # jnp.asarray, which truncates to float32 here just as solve_exact_perturbation's own
    # jnp.asarray(d) does (this repository does not enable x64; see nonlinear_solver_tolerance).
    # Anything but that shape/dtype would buy a second compile. Zero rows evaluate g_tilde at
    # eta_hat itself, which is in-domain by construction -- it is the observed root.
    probe = jnp.asarray(
        np.zeros((batch_width, int(np.shape(eta_j)[0])), dtype=np.float64)
    )
    jax.block_until_ready(evaluate(probe))
    return evaluate


def solve_exact_perturbations_batched(
    batch_eval: Callable[[np.ndarray], Any],
    g0: np.ndarray,
    bread_factored,
    s_rows: np.ndarray,
    delta_linear_rows: np.ndarray,
    config: DiagnosticConfig,
    *,
    batch_width: int,
) -> list[dict[str, Any]]:
    """
    Lockstep driver for a wave of perturbation re-solves: solves solve_exact_perturbation's problem
    for every row of (s_rows, delta_linear_rows) at once, returning one result dict per row in the
    SAME ORDER and with the same keys, so a caller's loop body needs no other change.

    WHAT IT CHANGES, AND WHAT IT CANNOT. A draw's chord-Newton trajectory depends only on its own
    iterate, its own continuation index, and the frozen bread factorization that every draw already
    shares (bread_factored is computed once per suite run and passed in -- the serial path was
    never re-factoring per draw). So the sequence of (evaluation point, residual, tolerance test,
    continuation advance, divergence test, chord update) this driver performs for row r is exactly
    the sequence solve_exact_perturbation performs for that row. Only the SCHEDULE differs: which
    rows share a kernel launch, and hence when each row's iteration is executed relative to other
    rows'. converged, nonfinite_encountered, aborted, num_iterations and delta_nl are therefore the
    serial values, and the ensemble the caller assembles from them is the serial ensemble.

    Two things make that an equivalence rather than an identity, both measured and both far below
    every tolerance any status rung compares against:
      * g_tilde under vmap is not bitwise the same program as g_tilde on one row. XLA picks
        different float32 kernels per batch width; measured 9.5e-7 absolute on a max|g| of 3.6,
        ~2.2 float32 ULPs, which propagates to ~1e-7 relative on the SE ratios and ~1e-5 on
        mean_shift_se (a difference of means, so cancellation-dominated). End to end at 100 draws
        this left status, warnings and all 200 per-trial convergence flags identical.
      * the batched residual norm and the batched triangular solve reassociate float64 reductions
        (measured 1.7e-14 on the solve), seven orders below the float32 noise floor above.
    Iteration COUNTS can drift by one on a step whose accepting residual sits a hair under the
    tolerance line (measured 1,990 vs 1,988 over 50 solves); no metric reports them, but do not
    assert equality on num_iterations.

    THE CONTINUATION LADDER ADVANCES PER ROW, not in lockstep across rows: a single batch routinely
    holds rows sitting at different lambdas, each multiplied by its own s_j to form its own target.
    Advancing the ladder in lockstep instead -- waiting for the slowest row at every lambda -- was
    measured at 427 batched calls where per-row needs 259 on a fragile ensemble (69.7% vs 50.0%
    padding waste), and ties on clean data. The extra cost is one (width, dim) elementwise multiply
    per generation, which is free next to a g_tilde evaluation.

    ACTIVE-SET COMPACTION is by index gather into a buffer of PINNED width: terminated rows stop
    consuming batch width (their slots are filled with zeros and their outputs sliced away before
    any test), but the shape handed to XLA never changes, so no iteration can trigger a recompile.
    See _make_batched_g_tilde_evaluator for why compacting to the exact live count instead would
    cost far more than the padding it saves. Padding is safe by construction rather than by
    convention: a zero row evaluates g_tilde at eta_hat, the observed root, so a padded slot can
    neither go non-finite nor manufacture a domain failure even if it were read.
    """
    delta_linear_rows = np.asarray(delta_linear_rows, dtype=np.float64)
    s_rows = np.asarray(s_rows, dtype=np.float64)
    num_rows, dim = delta_linear_rows.shape
    if num_rows > batch_width:
        raise ValueError(
            f"solve_exact_perturbations_batched got {num_rows} rows for a pinned batch width of "
            f"{batch_width}; the caller must split the ensemble into waves of at most that width."
        )
    g0 = np.asarray(g0, dtype=np.float64)
    lambdas = np.linspace(0.0, 1.0, config.continuation_steps + 1)[1:]
    num_steps = int(lambdas.size)
    max_iterations = int(config.nonlinear_solver_max_iterations)

    d_cur = np.array(delta_linear_rows, dtype=np.float64)
    lam_idx = np.zeros(num_rows, dtype=np.int64)
    iters_at_step = np.zeros(num_rows, dtype=np.int64)
    total_iterations = np.zeros(num_rows, dtype=np.int64)
    final_residual_norm = np.full(num_rows, math.nan)
    aborted = np.zeros(num_rows, dtype=bool)
    abort_reasons = [""] * num_rows
    # Per-continuation-step abort state, reset whenever a row advances a lambda -- the target moves
    # there, so the residual legitimately jumps and a best carried across the boundary would read
    # as an instant stall. Same reset points as the serial loop's, which re-initialises them at the
    # top of each `for lam in lambdas`.
    best_relative_norm = np.full(num_rows, math.inf)
    iters_since_best = np.zeros(num_rows, dtype=np.int64)
    # Same convention as the serial solver: scale by the FULL ||s_j||, never the lambda-scaled
    # target, so the relative residual does not blow up spuriously at small lambda.
    full_scale = np.maximum(np.linalg.norm(s_rows, axis=1), 1e-12)

    # Degenerate plans, kept faithful to the serial function rather than to this loop's structure:
    # with no continuation steps its `for lam in lambdas` never runs and every solve is trivially
    # converged at its warm start, and with no iteration budget every step fails before evaluating
    # g_tilde even once.
    if num_steps == 0:
        state = np.full(num_rows, _TRIAL_CONVERGED, dtype=np.int8)
    elif max_iterations <= 0:
        state = np.full(num_rows, _TRIAL_FAILED, dtype=np.int8)
    else:
        state = np.full(num_rows, _TRIAL_RUNNING, dtype=np.int8)

    tolerance = config.nonlinear_solver_tolerance
    abort_enabled = config.nonlinear_solver_divergence_abort
    blowup_factor = config.nonlinear_solver_divergence_blowup_factor
    stall_window = config.nonlinear_solver_divergence_stall_window
    abort_guard = (
        config.nonlinear_solver_divergence_guard_factor
        * config.nonlinear_solver_tolerance
    )

    block = np.zeros((batch_width, dim), dtype=np.float64)
    while True:
        take = np.flatnonzero(state == _TRIAL_RUNNING)
        if take.size == 0:
            break
        # The compaction step: live rows gathered into the low slots, the dead tail zeroed (so a
        # stale iterate from a row that has already terminated is never re-evaluated), the shape
        # left alone.
        block[: take.size] = d_cur[take]
        block[take.size :] = 0.0
        g_vals = np.asarray(batch_eval(jnp.asarray(block)), dtype=np.float64)[
            : take.size
        ]
        total_iterations[take] += 1

        finite = np.all(np.isfinite(g_vals), axis=1)
        # A non-finite evaluation breaks the serial loop BEFORE the chord update, so d_cur must not
        # move for these rows -- their delta_nl is the last in-domain iterate, as it is serially.
        state[take[~finite]] = _TRIAL_NONFINITE
        live = take[finite]
        if live.size == 0:
            continue

        residual = (
            g_vals[finite]
            - g0[None, :]
            - lambdas[lam_idx[live]][:, None] * s_rows[live]
        )
        residual_norm = np.linalg.norm(residual, axis=1)
        # Assigned on every finite iteration including the accepting one, exactly where the serial
        # loop assigns it: before the tolerance test, so a row that converges still reports the
        # residual it converged at, and a row whose very first evaluation is non-finite keeps NaN.
        final_residual_norm[live] = residual_norm
        relative_norm = residual_norm / full_scale[live]

        accepted = relative_norm < tolerance
        advanced = live[accepted]
        # An accepted iteration ends the step BEFORE the chord update (the serial loop breaks), so
        # again d_cur must not move for these rows.
        lam_idx[advanced] += 1
        iters_at_step[advanced] = 0
        best_relative_norm[advanced] = math.inf
        iters_since_best[advanced] = 0
        state[advanced[lam_idx[advanced] >= num_steps]] = _TRIAL_CONVERGED

        stepping = ~accepted
        continuing = live[stepping]
        if continuing.size == 0:
            continue
        continuing_relative = relative_norm[stepping]
        # The divergence test, in the serial function's exact order: a row that set a new best is
        # never tested (its `if` branch has no abort), and a row that did not is tested against the
        # best it failed to beat.
        improved = continuing_relative < best_relative_norm[continuing]
        best_relative_norm[continuing[improved]] = continuing_relative[improved]
        iters_since_best[continuing[improved]] = 0
        iters_since_best[continuing[~improved]] += 1

        updating = np.ones(continuing.size, dtype=bool)
        if abort_enabled:
            testable = ~improved & (best_relative_norm[continuing] > abort_guard)
            blowup = testable & (
                continuing_relative > blowup_factor * best_relative_norm[continuing]
            )
            stalled = (
                testable
                & ~blowup
                & bool(stall_window > 0)
                & (iters_since_best[continuing] >= stall_window)
            )
            for position in np.flatnonzero(blowup | stalled):
                row = int(continuing[position])
                aborted[row] = True
                abort_reasons[row] = "blowup" if blowup[position] else "stall"
                state[row] = _TRIAL_FAILED
            updating = ~(blowup | stalled)

        stepped = continuing[updating]
        if stepped.size == 0:
            continue
        # One triangular solve for the whole wave against the shared frozen factorization -- the
        # chord method's defining property is that this factorization never changes, so sharing it
        # across rows is what the method already assumes rather than an approximation introduced
        # here.
        d_cur[stepped] -= solve_with_bread(
            bread_factored, residual[stepping][updating].T
        ).T
        # Counted AFTER the update, because the serial loop's last permitted pass applies its chord
        # update and only then falls out of range(max_iterations) with the step unconverged.
        iters_at_step[stepped] += 1
        state[stepped[iters_at_step[stepped] >= max_iterations]] = _TRIAL_FAILED

    results: list[dict[str, Any]] = []
    for row in range(num_rows):
        converged = bool(state[row] == _TRIAL_CONVERGED)
        discrepancy_ratio = float(
            np.linalg.norm(d_cur[row] - delta_linear_rows[row])
            / (np.linalg.norm(delta_linear_rows[row]) + 1e-300)
        )
        results.append(
            {
                "delta_nl": d_cur[row],
                "converged": converged,
                "nonfinite_encountered": bool(state[row] == _TRIAL_NONFINITE),
                "final_residual_norm": float(final_residual_norm[row]),
                "num_iterations": int(total_iterations[row]),
                "aborted": bool(aborted[row]),
                "abort_reason": abort_reasons[row],
                "branch_change_suspected": converged and discrepancy_ratio > 5.0,
                "discrepancy_ratio": discrepancy_ratio,
            }
        )
    return results


def _resolve_bootstrap_batch_width(
    config: DiagnosticConfig, num_planned_trials: int
) -> int:
    """
    Pinned batch/wave width for check_multiplier_bootstrap's re-solve ensemble, or 0 for the serial
    reference path. See DiagnosticConfig.batched_bootstrap_resolves.

    "auto" is DELIBERATELY NOT ACCEPTED HERE, and is rejected rather than honored: adversarial
    review found a reproducible fixture on which enabling batching moves the check from
    `indeterminate` to `failed` and the verdict from `not_certified` to `invalid` on identical
    data and seed. Nothing that can silently change a verdict may select itself. The mode is kept
    reachable only as an explicit "always" so the driver stays testable and fixable.
    """
    if config.batched_bootstrap_resolves == "auto":
        logger.warning(
            "batched_bootstrap_resolves='auto' is disabled: batched re-solves are known to "
            "change verdicts on reachable fixtures (see the field's docstring), so they may "
            "never be selected automatically. Re-solving serially."
        )
        return 0
    if config.batched_bootstrap_resolves != "always":
        if config.batched_bootstrap_resolves != "off":
            # Falls back to the reference path rather than raising -- an unrecognized value here is
            # a performance knob, and failing a whole analysis over it would be worse than running
            # it the slow way -- but says so, since silently ignoring the request is exactly how
            # someone concludes the optimization "does nothing".
            logger.warning(
                "Unknown batched_bootstrap_resolves mode %r (expected 'off', 'auto' or "
                "'always'); the bootstrap will re-solve serially.",
                config.batched_bootstrap_resolves,
            )
        return 0
    width = min(int(config.bootstrap_batch_width), num_planned_trials)
    # A width of 1 is not batching at all -- it is the serial schedule paying for a vmap and a
    # compile -- so it falls back to the reference path. Widths of 2 or 3 are reachable only from
    # an ensemble too small to produce any statistic in the first place (both checks need three
    # converged trials), so they are allowed rather than special-cased. Note that XLA selects a
    # different float32 kernel at widths 1-2 than at 8 and above, which is a further reason not to
    # tune this knob down to save memory: prefer fewer, wider waves.
    if width < 2:
        return 0
    if config.batched_bootstrap_resolves == "auto":
        projected_rows = (
            num_planned_trials
            * config.continuation_steps
            * _MEDIAN_ITERATIONS_PER_CONTINUATION_STEP
        )
        if projected_rows < config.bootstrap_batch_min_rows:
            return 0
    return width


# Minimum converged trials any ensemble statistic in this module can be computed from, and the
# first status rung of BOTH re-solve checks: below it they are INDETERMINATE no matter what else
# the ensemble looks like. The sequential early stop below is sound only because it reproduces
# this exact number, so the two must never drift apart.
MIN_CONVERGED_TRIALS_FOR_ENSEMBLE = 3

EARLY_STOP_REASON_STARVATION = "max_attainable_converged_below_minimum"


def _starvation_early_stop_reached(
    config: DiagnosticConfig,
    num_converged: int,
    num_executed: int,
    num_planned: int,
) -> bool:
    """
    The only PROVABLY status-preserving sequential stop for the two re-solve ensembles: fire when
    the converged count that is still attainable has fallen below the minimum any ensemble
    statistic needs, i.e. num_converged + (num_planned - num_executed) <= 2.

    SOUNDNESS. Let c be the converged count so far, k the executed trials, N the planned trials,
    and R = N - k the remaining ones. Every completion of the ensemble -- every assignment of
    "failed" or "converged with row z" to the R trials not yet run -- ends with a converged count
    C_final <= c + R. If c + R < MIN_CONVERGED_TRIALS_FOR_ENSEMBLE then C_final is below that
    minimum for EVERY completion, so no completion can reach the gate or pass rungs: the status
    is decided by the two rungs above them -- FAILED when the failure rate's Clopper-Pearson
    lower bound clears bad_direction_probability_target (confirmed fragility), INDETERMINATE
    otherwise. The stop can only fire with c <= 2, i.e. with at least k - 2 failures among the
    k executed trials, and R <= 2 remaining; for any plan of N >= 9 trials that pins the lower
    bound far above any practical target for the truncated ensemble AND every completion of it
    (lb(k-2, k) and lb(k-2, k+2) are both >> 0.01 once k >= 7), so the status -- and with it
    classification, verdict and verdict_basis, which are pure functions of the statuses -- is
    exactly what the full plan would have produced. Only a toy plan (N < 9, reachable only by
    overriding num_perturbation_directions or num_bootstrap_draws far below their defaults) can
    leave the two sides of the bound comparison in different rungs, and even there both rungs
    sit in the do-not-report family (FAILED -> INVALID, INDETERMINATE -> NOT CERTIFIED), with
    truncation only ever softening FAILED to INDETERMINATE, never inventing a failure. The
    re-solves skipped were therefore pure cost.

    MAXIMALITY -- why nothing more aggressive is offered. Suppose instead c < 3 <= c + R and some
    target has a nonzero sandwich SE. Two completions give two different statuses: if every
    remaining trial fails the status is INDETERMINATE, while if 3 - c of them converge with
    arbitrarily large rows, the per-target SD ratio exceeds any finite null band and the status is
    FAILED. So no rule that cannot see the future outcomes may stop there. In particular
    "the solver is certainly unhealthy" is NOT a stopping condition: unhealthiness only confines
    the status to {FAILED, INDETERMINATE}, because both checks deliberately rank the FAILED rung
    ABOVE the unhealthy rung so that a distortion measured on the optimistically-selected
    converged subset still counts (see the precedence comments at each status block). And no
    a-priori bound on a converged row exists to appeal to instead -- a converged solve satisfies
    only a residual tolerance, which is exactly why these checks carry a branch_change_suspected
    concept at all.

    Consequently the stop can never skip more than MIN_CONVERGED_TRIALS_FOR_ENSEMBLE - 1 = 2 of
    the N trials, and only on ensembles that were headed for INDETERMINATE anyway. It is on by
    default because it is free and provably invisible, not because it is fast.

    The `num_executed >= 1` guard keeps a degenerate plan (N <= 2, i.e. a single unpaired
    direction) from producing an ensemble with zero executed trials, whose reported failure
    fraction and Clopper-Pearson bounds would be NaN rather than merely wide.
    """
    if config.perturbation_early_stop == "off":
        return False
    return (
        num_executed >= 1
        and num_converged + (num_planned - num_executed)
        < MIN_CONVERGED_TRIALS_FOR_ENSEMBLE
    )


def _resolve_accounting_metrics_and_warnings(
    warnings_list: list[str],
    *,
    trial_noun: str,
    early_stop_reason: str,
    num_planned_trials: int,
    num_trials: int,
    num_converged: int,
    num_divergence_aborts: int,
    num_root_failures: int,
) -> dict[str, Any]:
    """
    The honest-accounting half of the two cost optimizations, shared by both re-solve checks: what
    the run actually did, as metrics plus warnings, so a truncated or abort-shortened ensemble can
    never be mistaken for a complete one.

    num_trials is already the EXECUTED count in both callers (they measure the flag lists, not the
    plan), so root_failure_fraction and every Clopper-Pearson bound are computed on what was really
    run -- wider under an early stop, never narrower. num_planned_trials is what makes the
    truncation legible next to it; DiagnosticReport.monte_carlo_counts carries the same
    distinction at report level.
    """
    if early_stop_reason:
        warnings_list.append(
            f"Stopped after {num_trials} of {num_planned_trials} planned {trial_noun}: at most "
            f"{num_converged + num_planned_trials - num_trials} could have converged, below the "
            f"{MIN_CONVERGED_TRIALS_FOR_ENSEMBLE} required for any ensemble statistic -- no "
            "completion of the ensemble could have passed (the status is FAILED when the "
            "failure rate is statistically confirmed, INDETERMINATE otherwise). Trial counts "
            "and failure fractions below describe the truncated ensemble, not the plan."
        )
    if num_divergence_aborts:
        warnings_list.append(
            f"{num_divergence_aborts} of {num_root_failures} non-converged {trial_noun} were "
            "stopped early by the divergence detector (the iterate blew past its own best "
            "residual, or stalled); they are counted as root failures exactly as an exhausted "
            "iteration budget would be. domain_failure_fraction is therefore a LOWER bound here: "
            "a diverging iterate that would have overflowed to a non-finite g_tilde a few "
            "iterations later is now counted as a root failure instead, which is the more "
            "accurate description of it but is not the number an unaborted run would report."
        )
    return {
        "num_planned_trials": num_planned_trials,
        "early_stopped": bool(early_stop_reason),
        "early_stop_reason": early_stop_reason,
        "num_divergence_aborted_trials": num_divergence_aborts,
    }


def _simulate_perturbation_null_bands(
    converged_pattern: Sequence[tuple[float, int]],
    dim: int,
    confidence_level: float,
    seed: int,
    num_null_samples: int,
) -> dict[str, float]:
    """
    Simulates the exact finite-draw null distribution of the ensemble statistics used by
    check_exact_nonlinear_perturbations and check_multiplier_bootstrap, under H0 "the solved
    perturbations are distributed exactly as their linear predictions": target-space rows are
    sign_t * z_{j_t} with z_j iid N(0, V_L), replicating the observed (sign, direction) pattern
    of converged trials -- antithetic pairing and convergence exclusions included, which is what
    makes these bands honest at any J (the former fixed [0.95, 1.05] band was not; see the
    num_null_band_samples comment on DiagnosticConfig). Every statistic banded here is invariant
    under the whitening z -> V_L^{-1/2} z, so the simulation runs at V_L = I and applies to any
    positive-definite V_L.

    Returns two-sided bands for the two SPREAD statistics -- an envelope for the min/max
    sqrt-generalized-eigenvalue "se_ratios" statistic, and a per-target SD-ratio band
    (Bonferroni-adjusted across the `dim` targets, so the family-wise false-positive rate of the
    per-target gate is 1 - confidence_level) -- plus a one-sided upper quantile for the LOCATION
    statistic ||mean_j(rows)||, which is the same b_L = ||V_L^{-1/2} mean_j(z_j)|| the callers
    compute (only large values of a norm are evidence against H0, hence one-sided). That last
    band is what makes b_L honest at finite J: with unpaired draws mean_j(z_j) is pure sampling
    noise of size sqrt(chi2_dim / J) -- about 0.26 in one dimension at J = 15, which no fixed
    tenth-of-an-SE tolerance can distinguish from real curvature-induced displacement.
    """
    unique_directions = sorted({j for _, j in converged_pattern})
    if len(converged_pattern) < 3 or len(unique_directions) < 2:
        return {
            "se_ratio_lower": math.nan,
            "se_ratio_upper": math.nan,
            "per_target_sd_lower": math.nan,
            "per_target_sd_upper": math.nan,
            "mean_shift_upper": math.nan,
            "num_null_samples": 0,
        }
    direction_index = {j: idx for idx, j in enumerate(unique_directions)}
    signs = np.array([sign for sign, _ in converged_pattern])
    rows_of = np.array([direction_index[j] for _, j in converged_pattern])

    rng = np.random.default_rng(seed)
    min_ratios = np.empty(num_null_samples)
    max_ratios = np.empty(num_null_samples)
    sd_samples = np.empty(num_null_samples)
    mean_norms = np.empty(num_null_samples)
    for k in range(num_null_samples):
        z = rng.standard_normal((len(unique_directions), dim))
        rows = signs[:, None] * z[rows_of]
        cov = np.atleast_2d(np.cov(rows, rowvar=False))
        eigvals = np.linalg.eigvalsh(cov)
        ratios = np.sqrt(np.clip(eigvals, 0.0, None))
        min_ratios[k] = ratios[0]
        max_ratios[k] = ratios[-1]
        sd_samples[k] = float(np.std(rows[:, 0], ddof=1))
        mean_norms[k] = float(np.linalg.norm(rows.mean(axis=0)))

    alpha = 1.0 - confidence_level
    per_target_alpha = alpha / max(dim, 1)
    return {
        "se_ratio_lower": float(np.quantile(min_ratios, alpha / 2)),
        "se_ratio_upper": float(np.quantile(max_ratios, 1 - alpha / 2)),
        "per_target_sd_lower": float(np.quantile(sd_samples, per_target_alpha / 2)),
        "per_target_sd_upper": float(np.quantile(sd_samples, 1 - per_target_alpha / 2)),
        "mean_shift_upper": float(np.quantile(mean_norms, confidence_level)),
        "num_null_samples": num_null_samples,
    }


def _mean_shift_ensemble(
    z_rows: np.ndarray,
    converged_pattern: Sequence[tuple[float, int]],
    paired_directions: bool,
) -> tuple[np.ndarray, list[tuple[float, int]]]:
    """
    The sub-ensemble the LOCATION statistic b_L = ||V_L^{-1/2} mean_j(z_j)|| is computed on:
    every converged row when directions are drawn unpaired, but only the directions that survived
    with BOTH signs when they are drawn antithetically. Returns (rows, pattern) so the caller can
    simulate the matching null band.

    Restricting to complete pairs is what keeps the mean shift a measurement of curvature rather
    than an artifact of convergence/domain censoring. Inside a complete pair the linear parts
    cancel identically -- 0.5(z_+ + z_-) is the even (curvature) part and nothing else -- so
    dropping WHOLE pairs can only remove curvature signal, never manufacture it, which puts this
    statistic on the same footing as the spread gates: censoring the hardest directions biases it
    DOWN, so an exceedance measured on the survivors is trustworthy. A LONE surviving row instead
    contributes its full sampling fluctuation to the mean, and since whether a solve converged is
    correlated with where its root landed, that contribution is selection wearing curvature's
    clothes -- it is exactly what manufactured spurious mean-shift failures out of asymmetrically
    censored ensembles.
    """
    rows = np.asarray(z_rows)
    if not paired_directions:
        return rows, list(converged_pattern)
    signs_by_direction: dict[int, set[float]] = {}
    for sign, direction in converged_pattern:
        signs_by_direction.setdefault(direction, set()).add(sign)
    keep = np.array(
        [len(signs_by_direction[direction]) == 2 for _, direction in converged_pattern],
        dtype=bool,
    )
    pattern = [
        entry
        for entry, keep_it in zip(converged_pattern, keep, strict=False)
        if keep_it
    ]
    return rows[keep] if rows.size else rows, pattern


def _mean_shift_null_upper(
    null_bands: dict[str, float],
    mean_shift_pattern: Sequence[tuple[float, int]],
    converged_pattern: Sequence[tuple[float, int]],
    dim: int,
    seed: int,
    config: DiagnosticConfig,
) -> float:
    """
    The null upper quantile for b_L. When the mean-shift sub-ensemble is the whole converged
    ensemble -- every antithetic pair intact, or unpaired draws -- the band already simulated for
    the spread statistics applies verbatim; only a pair-censored ensemble needs its own (smaller)
    pattern simulated.
    """
    if len(mean_shift_pattern) == len(converged_pattern):
        return null_bands["mean_shift_upper"]
    return _simulate_perturbation_null_bands(
        mean_shift_pattern,
        dim,
        config.confidence_level,
        seed,
        config.num_null_band_samples,
    )["mean_shift_upper"]


def _evaluate_mean_shift_gate(
    b_L: float,
    mean_shift_null_upper: float,
    num_dropped_rows: int,
    solver_unhealthy: bool,
    config: DiagnosticConfig,
    warnings_list: list[str],
) -> tuple[float, bool, bool]:
    """
    The shared mean-shift gate of check_exact_nonlinear_perturbations and
    check_multiplier_bootstrap: returns (threshold, gate_evaluable, gate_failed) for the b_L
    computed on _mean_shift_ensemble's rows, appending an explanatory warning whenever an
    exceedance is observed but deliberately not gated on.

    b_L is compared against max(simulated finite-draw null upper quantile,
    config.mean_shift_tolerance_se). The band is what makes the comparison honest at finite J:
    with unpaired draws b_L is raw sampling noise of size sqrt(chi2_dim / J) -- about 0.26 SE in
    one dimension at the default J = 15, which the former fixed 0.10 comparison read as a failure
    on a provably affine map. config.mean_shift_tolerance_se survives as the absolute
    practical-significance FLOOR (a displacement below a tenth of a target SE is not worth
    reporting however many draws make it detectable), and it is the binding threshold in the
    ordinary antithetic case, where the null band collapses to exactly 0 because every complete
    pair cancels.

    With unpaired draws there is no cancellation to lean on, so censoring CAN manufacture a
    location shift that the null band -- which replicates the observed pattern but still treats
    the rows as an unselected sample -- cannot absorb; there, and only there, an unhealthy solver
    suppresses the gate. The caller turns a suppressed or unevaluable gate into INDETERMINATE,
    never into a pass.
    """
    if math.isnan(mean_shift_null_upper):
        return math.nan, False, False
    threshold = max(mean_shift_null_upper, config.mean_shift_tolerance_se)
    if num_dropped_rows:
        warnings_list.append(
            f"{num_dropped_rows} converged row(s) whose antithetic partner did not converge were "
            "excluded from the mean-shift statistic, which is computed on complete +/- pairs "
            "only; the remaining pairs under-represent the most curved directions."
        )
    evaluable = not math.isnan(b_L) and (
        config.paired_directions or not solver_unhealthy
    )
    if evaluable:
        return threshold, True, bool(b_L > threshold)

    if not math.isnan(b_L) and b_L > threshold:
        warnings_list.append(
            f"Mean shift {b_L:.4g} SE exceeds its null threshold {threshold:.4g}, but the "
            "directions are unpaired and the solver is unhealthy, so the displacement cannot be "
            "separated from the censoring of non-converged directions (there is no antithetic "
            "partner left to cancel the linear part against); the mean-shift gate was not "
            "applied."
        )
    return threshold, False, False


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

    The ensemble may be shorter than the plan: the sequential early stop (see
    _starvation_early_stop_reached) abandons the remaining directions once no completion of the
    ensemble could reach a computable statistic. That is status-preserving by construction, and
    what actually ran is reported as num_planned_trials / num_trials / early_stopped rather than
    left to be inferred from the config. Individual solves may also end early via the divergence
    abort, which is counted as an ordinary root failure -- see solve_exact_perturbation.

    This check re-solves SERIALLY and has no batched path, unlike check_multiplier_bootstrap. That
    is a cost decision, not an oversight: batching buys one XLA compile of the vmap'd g_tilde
    (measured 5-6 s, flat in width) which needs a few thousand live row-evaluations to repay, and
    this check's default ensemble -- 15 paired directions, ~1,200 rows -- is well under it, so
    batching it in isolation measures SLOWER. Keeping it serial also leaves the shared
    solve_exact_perturbation exercised by a caller on every run. If num_exact_directions is ever
    raised into the hundreds this becomes worth revisiting, with its own equivalence campaign: this
    check accumulates more per-trial state than the bootstrap does (z_linear_rows, a_nl_by_target),
    all of it order-sensitive.
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
    num_divergence_aborts = 0

    converged_pattern: list[tuple[float, int]] = []
    num_planned_trials = num_directions * len(signs)
    early_stop_reason = ""

    for sign in signs:
        if early_stop_reason:
            break
        for j in range(num_directions):
            # Evaluated BEFORE this trial is solved, so the counts it reads are exactly the (c, k)
            # of the soundness argument in _starvation_early_stop_reached: every trial from here
            # on is skipped, and no completion of the ensemble could have passed (FAILED when
            # fragility is statistically confirmed, INDETERMINATE otherwise).
            if _starvation_early_stop_reached(
                config,
                len(converged_pattern),
                len(convergence_flags),
                num_planned_trials,
            ):
                early_stop_reason = EARLY_STOP_REASON_STARVATION
                break
            delta_lin = sign * base_delta[j]
            s_j = sign * base_s[j]
            solved = solve_exact_perturbation(
                g_tilde, eta_hat, g0, bread_factored, s_j, delta_lin, config
            )
            convergence_flags.append(solved["converged"])
            branch_change_flags.append(solved["branch_change_suspected"])
            nonfinite_flags.append(solved["nonfinite_encountered"])
            num_divergence_aborts += int(solved["aborted"])

            # Every ensemble statistic below is computed on CONVERGED trials only: a
            # non-converged continuation's last iterate is solver debris, not an "exact"
            # measurement, and including it poisons a^NL/se_ratios/mean-shift with values that
            # can be astronomically large (observed up to ~1e34 SE in the ADS-142 experiment).
            # Non-convergence is still fully accounted for via root_failure_fraction and the
            # status logic below -- exclusion never hides it.
            if not solved["converged"]:
                continue
            converged_pattern.append((sign, j))
            z_linear_rows.append(L @ delta_lin)
            z_nonlinear_rows.append(L @ solved["delta_nl"])
            e_j = solved["delta_nl"] - delta_lin
            for label, contrast_row, se in zip(target_labels, L, se_l, strict=False):
                a_nl_by_target[label].append(
                    float(abs(contrast_row @ e_j) / (se if se > 0 else 1.0))
                )

    num_trials = len(convergence_flags)
    num_converged = len(converged_pattern)
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

    warnings_list: list[str] = []
    accounting_metrics = _resolve_accounting_metrics_and_warnings(
        warnings_list,
        trial_noun="perturbation directions",
        early_stop_reason=early_stop_reason,
        num_planned_trials=num_planned_trials,
        num_trials=num_trials,
        num_converged=num_converged,
        num_divergence_aborts=num_divergence_aborts,
        num_root_failures=num_root_failures,
    )
    V_L_lin = L @ np.asarray(V_hat) @ L.T
    # Health is judged on the OBSERVED failure fraction, not the Clopper-Pearson upper bound:
    # at practical trial counts the upper bound sits above bad_direction_probability_target even
    # with zero observed failures (0/200 -> ~1.5%), which would make a perfectly healthy solver
    # read unhealthy purely for lack of certification power. The upper bound stays reported.
    observed_failure_fraction = num_root_failures / num_trials if num_trials else 1.0
    solver_unhealthy = (
        observed_failure_fraction > config.bad_direction_probability_target
    )
    # The LOWER bound separates "could not evaluate" from "measured a failure": failures are
    # counted over ALL executed trials (no optimistic-subset caveat), so when even the lower
    # confidence bound of the failure rate clears the target, fragility is statistically
    # confirmed and the check FAILS -- e.g. 198/198 failures bounds the rate above ~0.985.
    # A merely thin or truncated ensemble (the bound cannot clear the target) stays
    # INDETERMINATE.
    failure_lower_bound = clopper_pearson_lower_bound(
        num_root_failures, num_trials, config.confidence_level
    )
    solver_confirmed_unhealthy = bool(
        not math.isnan(failure_lower_bound)
        and failure_lower_bound > config.bad_direction_probability_target
    )

    if num_converged >= MIN_CONVERGED_TRIALS_FOR_ENSEMBLE:
        z_linear = np.array(z_linear_rows)
        z_nonlinear = np.array(z_nonlinear_rows)
        nonlinear_cov = np.atleast_2d(np.cov(z_nonlinear, rowvar=False))
        try:
            se_ratios = se_ratios_from_generalized_eigenvalues(nonlinear_cov, V_L_lin)
        except (scipy.linalg.LinAlgError, ValueError) as exc:
            se_ratios = np.full(V_L_lin.shape[0], math.nan)
            warnings_list.append(f"Generalized eigenvalue computation failed: {exc}")
        mean_shift_rows, mean_shift_pattern = _mean_shift_ensemble(
            z_nonlinear, converged_pattern, config.paired_directions
        )
        try:
            b_L = (
                float(
                    np.linalg.norm(
                        matrix_inv_sqrt(V_L_lin) @ (np.mean(mean_shift_rows, axis=0))
                    )
                )
                if len(mean_shift_rows)
                else math.nan
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
    else:
        se_ratios = np.full(V_L_lin.shape[0], math.nan)
        b_L = math.nan
        mean_shift_pattern = list(converged_pattern)
        quantile_shifts = {
            label: {"lower_shift_se": math.nan, "upper_shift_se": math.nan}
            for label in target_labels
        }
        warnings_list.append(
            f"Only {num_converged} of {num_trials} exact perturbation solves converged -- "
            "too few for any ensemble statistic; the check is INDETERMINATE."
        )

    # The se_ratios gate is judged against a simulated finite-J null band for the exact
    # converged (sign, direction) pattern, NOT a fixed tolerance: at any practical number of
    # directions the min/max generalized eigenvalue carries substantial sampling noise under
    # perfect linearity (see the num_null_band_samples comment on DiagnosticConfig).
    null_bands = _simulate_perturbation_null_bands(
        converged_pattern,
        V_L_lin.shape[0],
        config.confidence_level,
        config.random_seed + 5,
        config.num_null_band_samples,
    )
    # The band itself is unavailable only when the converged pattern is too thin to simulate;
    # se_ratios are separately unavailable when the generalized eigenproblem failed (a singular
    # V_L_lin). Either way the gate is UNEVALUABLE, which the status logic below turns into
    # INDETERMINATE -- never into a pass.
    se_ratio_band_available = not math.isnan(
        null_bands["se_ratio_lower"]
    ) and not math.isnan(null_bands["se_ratio_upper"])
    evaluable_se_ratios = se_ratios[np.isfinite(se_ratios)]
    se_ratio_band_evaluable = (
        se_ratios.size > 0
        and evaluable_se_ratios.size == se_ratios.size
        and se_ratio_band_available
    )
    se_ratio_above = bool(
        se_ratio_band_available
        and evaluable_se_ratios.size > 0
        and float(np.max(evaluable_se_ratios)) > null_bands["se_ratio_upper"]
    )
    se_ratio_below = bool(
        se_ratio_band_available
        and evaluable_se_ratios.size > 0
        and float(np.min(evaluable_se_ratios)) < null_bands["se_ratio_lower"]
    )
    se_ratio_ok = (
        (not se_ratio_above and not se_ratio_below) if se_ratio_band_evaluable else None
    )

    mean_shift_null_upper = _mean_shift_null_upper(
        null_bands,
        mean_shift_pattern,
        converged_pattern,
        V_L_lin.shape[0],
        config.random_seed + 6,
        config,
    )
    mean_shift_threshold, mean_shift_gate_evaluable, mean_shift_failed = (
        _evaluate_mean_shift_gate(
            b_L,
            mean_shift_null_upper,
            len(converged_pattern) - len(mean_shift_pattern),
            solver_unhealthy,
            config,
            warnings_list,
        )
    )

    # An ABOVE-band se_ratio on the converged subset is trustworthy under censoring (the
    # excluded directions are the hardest, so exclusion biases the spread DOWN); a BELOW-band
    # ratio on a censored subset is exactly what that censoring produces on its own, so it only
    # counts as a failure when the solver is healthy.
    gates_failed = se_ratio_above or (se_ratio_below and not solver_unhealthy)
    if mean_shift_failed:
        gates_failed = True
    for shifts in quantile_shifts.values():
        if (
            not math.isnan(shifts["lower_shift_se"])
            and not math.isnan(shifts["upper_shift_se"])
            and (
                abs(shifts["lower_shift_se"]) > config.quantile_shift_tolerance_se
                or abs(shifts["upper_shift_se"]) > config.quantile_shift_tolerance_se
            )
        ):
            gates_failed = True

    # Status precedence: statistically confirmed fragility FAILS outright (measured on all
    # executed trials, so no optimistic-subset caveat applies and it needs no ensemble
    # statistics at all); then a FAILED verdict from the converged subset stands even when the
    # solver is unhealthy (exclusion of the non-converged -- typically hardest -- directions
    # biases the subset OPTIMISTIC, so a failure on it is trustworthy); a PASS on that same
    # optimistic subset is not, hence INDETERMINATE.
    if solver_confirmed_unhealthy:
        status = CheckStatuses.FAILED
        warnings_list.append(
            f"Re-solve failure fraction {observed_failure_fraction:.4g} "
            f"({num_root_failures}/{num_trials}) has lower confidence bound "
            f"{failure_lower_bound:.4g}, above bad_direction_probability_target="
            f"{config.bad_direction_probability_target:g}: the estimating equation "
            "measurably cannot be re-solved at perturbation scale. This is a confirmed "
            "fragility of the equation, not an unevaluable ensemble."
        )
    elif num_converged == 0 or num_converged < MIN_CONVERGED_TRIALS_FOR_ENSEMBLE:
        status = CheckStatuses.INDETERMINATE
    elif gates_failed:
        status = CheckStatuses.FAILED
        if solver_unhealthy:
            warnings_list.append(
                "Distortion gates failed on the converged subset only; with "
                f"{num_root_failures}/{num_trials} non-converged directions excluded the "
                "subset is optimistically selected, so the failure is trustworthy."
            )
    elif solver_unhealthy:
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            f"Nonlinear-root failure fraction {observed_failure_fraction:.4g} exceeds target "
            f"{config.bad_direction_probability_target}; gates passed on the converged subset "
            "but that subset is optimistically selected."
        )
    elif not (se_ratio_band_evaluable and mean_shift_gate_evaluable):
        # An unevaluable gate is not a pass -- the same convention this check already applies to
        # an ensemble too small for any statistic at all.
        unevaluable = [
            name
            for name, ok in (
                ("se_ratios", se_ratio_band_evaluable),
                ("mean_shift", mean_shift_gate_evaluable),
            )
            if not ok
        ]
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            f"Distortion gate(s) {', '.join(unevaluable)} could not be evaluated on this "
            "ensemble (typically a singular target covariance), so no comparison against their "
            "null bands was performed; the check is INDETERMINATE rather than passed."
        )
    else:
        status = CheckStatuses.PASSED

    metrics = {
        "a_nl_by_target": {
            label: _quantile_summary(np.array(values))
            for label, values in a_nl_by_target.items()
        },
        "se_ratios": se_ratios.tolist(),
        "se_ratios_within_tolerance": se_ratio_ok,
        "se_ratio_null_band": [
            null_bands["se_ratio_lower"],
            null_bands["se_ratio_upper"],
        ],
        "mean_shift_se": b_L,
        "mean_shift_null_upper": mean_shift_null_upper,
        "mean_shift_threshold": mean_shift_threshold,
        "mean_shift_gate_evaluable": mean_shift_gate_evaluable,
        "mean_shift_num_rows": len(mean_shift_pattern),
        "quantile_shifts_se": quantile_shifts,
        "num_trials": num_trials,
        "num_converged_trials": num_converged,
        **accounting_metrics,
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

    # Per-criterion outcomes. Guards mirror the status logic: an empty evaluable_se_ratios or a
    # NaN mean-shift gate reads [not evaluated], never a NaN posing as a measurement; a
    # below-band SE ratio while the solver is unhealthy does not fail (the excluded directions
    # bias the spread down), so that criterion is excused rather than marked [FAIL].
    quantile_shift_values = [
        abs(shifts[key])
        for shifts in quantile_shifts.values()
        for key in ("lower_shift_se", "upper_shift_se")
        if not math.isnan(shifts[key])
    ]
    max_quantile_shift = (
        max(quantile_shift_values) if quantile_shift_values else math.nan
    )
    se_ratio_excused = se_ratio_below and solver_unhealthy
    if se_ratio_band_evaluable:
        se_ratio_value = (
            f"{float(np.min(evaluable_se_ratios)):.3g} to "
            f"{float(np.max(evaluable_se_ratios)):.3g}"
        )
        if se_ratio_excused and not se_ratio_above:
            se_ratio_value += (
                " (below the band, but excused: failed re-solves bias it down)"
            )
    else:
        se_ratio_value = "not evaluable on this ensemble"
    criteria = [
        CriterionResult(
            description=(
                f"at least {MIN_CONVERGED_TRIALS_FOR_ENSEMBLE} of the re-solves converged"
            ),
            value=f"{num_converged} of {num_trials}",
            ok=bool(num_converged >= MIN_CONVERGED_TRIALS_FOR_ENSEMBLE),
            severity="indeterminate",
        ),
        CriterionResult(
            description=(
                f"re-solve failure rate at most "
                f"{config.bad_direction_probability_target:g} "
                "(bad_direction_probability_target)"
            ),
            value=f"{observed_failure_fraction:.4g} (lower confidence bound "
            f"{failure_lower_bound:.3g})"
            if solver_confirmed_unhealthy
            else f"{observed_failure_fraction:.4g}",
            ok=not solver_unhealthy,
            # Exceeding the target is INDETERMINATE while it could still be bad luck or a
            # solver budget issue, but FAILS once the rate's lower confidence bound clears
            # the target: at that point the fragility is measured, not merely unresolved.
            severity="fail" if solver_confirmed_unhealthy else "indeterminate",
        ),
        CriterionResult(
            description=(
                "every re-solved/linear SE ratio inside "
                f"{null_bands['se_ratio_lower']:.3g}-{null_bands['se_ratio_upper']:.3g}, "
                "the range expected if nothing were wrong"
            ),
            value=se_ratio_value,
            ok=(
                not (se_ratio_above or (se_ratio_below and not solver_unhealthy))
                if se_ratio_band_evaluable
                else None
            ),
        ),
        CriterionResult(
            description=(
                f"the re-solved estimates' average shift at most "
                f"{mean_shift_threshold:.3g} SE (the largest expected if nothing "
                "were wrong)"
            )
            if not math.isnan(mean_shift_threshold)
            else (
                "the re-solved estimates' average shift within the largest expected "
                "if nothing were wrong"
            ),
            value=f"{b_L:.3g} SE"
            if mean_shift_gate_evaluable
            else "not evaluable on this ensemble",
            ok=(not mean_shift_failed) if mean_shift_gate_evaluable else None,
        ),
        CriterionResult(
            description=(
                "the ensemble's 2.5% and 97.5% points each shift at most "
                f"{config.quantile_shift_tolerance_se:g} SE (quantile_shift_tolerance_se)"
            ),
            value=f"worst {max_quantile_shift:.3g} SE"
            if not math.isnan(max_quantile_shift)
            else "not evaluated",
            ok=bool(max_quantile_shift <= config.quantile_shift_tolerance_se)
            if not math.isnan(max_quantile_shift)
            else None,
        ),
    ]
    return CheckResult(
        name="exact_nonlinear_perturbation",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        criteria=criteria,
    )


###############################################################################
# Section 4b: frozen-score multiplier bootstrap. Reuses the continuation machinery above, but
# with empirically-weighted score draws instead of Gaussian direction probes, and with a
# self-calibrating verdict: bootstrap-vs-sandwich SE ratios judged against their own simulated
# finite-draw null band. See the multiplier_bootstrap comment on DiagnosticConfig.
###############################################################################


def sample_multiplier_perturbations(
    per_subject_stacks: np.ndarray,
    bread_factored,
    num_subjects: int,
    num_draws: int,
    seed: int,
    distribution: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Frozen-score multiplier-bootstrap draws: s_b = (1/n) sum_i nu_{b,i} g_i(eta_hat) with iid
    mean-0/variance-1 multipliers nu_{b,i} -- exactly sample_perturbation_directions'
    construction ((W @ stacks)/n) with the Gaussian W replaced by empirical multipliers. Same
    covariance M_hat/n; higher moments now inherited from the realized per-subject stacks
    instead of imposed as Gaussian. Freezing the multiplier-weighted score at eta_hat (rather
    than re-evaluating per-subject contributions at each candidate root) differs from the
    textbook weighted bootstrap by a term of the same order as the bootstrap's own error, and is
    what lets this run on per_subject_stacks alone. delta_lin = B^{-1} s as everywhere else.
    """
    rng = np.random.default_rng(seed)
    if distribution == "rademacher":
        nu = rng.choice(np.array([-1.0, 1.0]), size=(num_draws, num_subjects))
    elif distribution == "mammen":
        # Mammen (1993) two-point distribution: mean 0, variance 1, third moment 1.
        sqrt5 = math.sqrt(5.0)
        prob_low = (sqrt5 + 1.0) / (2.0 * sqrt5)
        low, high = (1.0 - sqrt5) / 2.0, (1.0 + sqrt5) / 2.0
        nu = np.where(rng.random(size=(num_draws, num_subjects)) < prob_low, low, high)
    elif distribution == "gaussian":
        nu = rng.standard_normal((num_draws, num_subjects))
    else:
        raise ValueError(
            f"Unknown bootstrap_multiplier_distribution: {distribution!r} "
            "(expected 'rademacher', 'mammen', or 'gaussian')."
        )
    stacks = np.asarray(per_subject_stacks, dtype=np.float64)
    S = (nu @ stacks) / num_subjects
    delta = solve_with_bread(bread_factored, S.T).T
    return delta, S


def check_multiplier_bootstrap(
    g_tilde: Callable[[jnp.ndarray], jnp.ndarray],
    eta_hat: jnp.ndarray,
    g0: np.ndarray,
    bread_factored,
    per_subject_stacks: np.ndarray,
    num_subjects: int,
    L: np.ndarray,
    target_labels: list[str],
    V_hat: np.ndarray,
    config: DiagnosticConfig,
) -> CheckResult:
    """
    Solves the estimating equation under num_bootstrap_draws frozen-score multiplier
    perturbations (paired +/- when configured) and compares the resulting bootstrap SE per
    target against the adjusted sandwich's SE. Per-target ratios outside a simulated
    finite-draw null band mean the nonlinearly-solved sampling distribution disagrees with the
    sandwich's linearization: wider -> FAILED (the sandwich SE is understated -- the
    anticonservative direction), narrower -> WARNING (overstated -- the conservative direction,
    the blow-up phenomenon of docs/adr/0002). A high non-convergence rate across draws is
    itself evidence of solver-level fragility under resampling-scale perturbations ->
    INDETERMINATE.

    The ensemble may be shorter than the plan: the sequential early stop (see
    _starvation_early_stop_reached) abandons the remaining draws once no completion of the
    ensemble could reach a computable bootstrap SE. That is status-preserving by construction, and
    what actually ran is reported as num_planned_trials / num_trials / num_draws_executed /
    early_stopped rather than left to be inferred from the config. Individual re-solves may also
    end early via the divergence abort, which is counted as an ordinary root failure -- see
    solve_exact_perturbation.

    The ensemble is re-solved one trial at a time by default. With
    config.batched_bootstrap_resolves enabled it is re-solved in waves that advance their
    chord-Newton iterations together (solve_exact_perturbations_batched), which replaces up to
    bootstrap_batch_width separate g_tilde calls per iteration with one vmap'd call. Each draw's
    trajectory is unchanged, so the ensemble and its status are the same; the arithmetic is not
    bitwise the same, and metrics["resolve_batch_width"] records which path ran.
    """
    num_draws = config.num_bootstrap_draws
    base_delta, base_s = sample_multiplier_perturbations(
        per_subject_stacks,
        bread_factored,
        num_subjects,
        num_draws,
        config.random_seed + 7,
        config.bootstrap_multiplier_distribution,
    )
    signs = [1.0, -1.0] if config.paired_directions else [1.0]

    se_l = standard_errors_for_contrasts(V_hat, L)
    z_rows: list[np.ndarray] = []
    converged_pattern: list[tuple[float, int]] = []
    convergence_flags: list[bool] = []
    nonfinite_flags: list[bool] = []
    num_divergence_aborts = 0
    num_batched_trials = 0
    num_planned_trials = num_draws * len(signs)
    early_stop_reason = ""

    # Trials in ENUMERATION ORDER (sign-major, draw-minor), materialized once so the wave loop
    # below can slice it. This order is load-bearing well past readability: np.std/np.cov over the
    # converged rows, _mean_shift_ensemble's pair matching and _simulate_perturbation_null_bands'
    # reconstruction from converged_pattern all consume the rows in the order they were appended,
    # so results must be consumed in plan order even when they were COMPUTED together. That is why
    # the batched driver returns a list positionally matched to its input rows rather than, say, a
    # dict keyed by completion.
    trial_plan = [(sign, j) for sign in signs for j in range(num_draws)]
    resolve_batch_width = _resolve_bootstrap_batch_width(config, num_planned_trials)
    batch_eval = None
    if resolve_batch_width:
        try:
            batch_eval = _make_batched_g_tilde_evaluator(
                g_tilde, eta_hat, resolve_batch_width
            )
        except Exception:
            # Not every g_tilde survives jit+vmap (a closure with host-side control flow on its
            # argument, a callback, a shape the tracer cannot handle). The check must still
            # produce a result -- a bootstrap that does not run is an UNCERTIFIABLE verdict, i.e.
            # a materially different report -- so the serial reference path is always reachable.
            logger.warning(
                "Batched bootstrap re-solves are enabled but g_tilde could not be compiled at "
                "batch width %d; falling back to the serial reference path.",
                resolve_batch_width,
                exc_info=True,
            )
            resolve_batch_width = 0
    # Wave = the pinned batch width, so a wave is exactly one buffer's worth of trials, and 1 when
    # serial -- which makes the loop below the original per-trial loop line for line, including
    # where the early stop is evaluated.
    wave_size = resolve_batch_width or 1

    for wave_start in range(0, num_planned_trials, wave_size):
        # Evaluated BEFORE the next wave is re-solved, so the counts it reads are exactly the
        # (c, k) of the soundness argument in _starvation_early_stop_reached: every trial from
        # here on is skipped, and no completion of the ensemble could have passed (FAILED
        # when fragility is statistically confirmed, INDETERMINATE otherwise).
        # Serially this is per trial and identical to the pre-batching behaviour. Batched, it can
        # only be tested at wave boundaries, where the remaining count is a multiple of the wave
        # width -- so with any wave wider than 2 the predicate is effectively unreachable and the
        # stop simply does not fire. That costs nothing to speak of (the stop can never skip more
        # than 2 trials of the plan) and stays honest either way: the metrics report what actually
        # ran, and under batching that is the whole plan.
        if _starvation_early_stop_reached(
            config,
            len(converged_pattern),
            len(convergence_flags),
            num_planned_trials,
        ):
            early_stop_reason = EARLY_STOP_REASON_STARVATION
            break
        wave = trial_plan[wave_start : wave_start + wave_size]
        s_wave = np.array([sign * base_s[j] for sign, j in wave])
        delta_wave = np.array([sign * base_delta[j] for sign, j in wave])
        solved_wave = None
        if batch_eval is not None:
            try:
                solved_wave = solve_exact_perturbations_batched(
                    batch_eval,
                    g0,
                    bread_factored,
                    s_wave,
                    delta_wave,
                    config,
                    batch_width=resolve_batch_width,
                )
            except Exception:
                logger.warning(
                    "Batched bootstrap re-solves failed after %d of %d trials; the remaining "
                    "trials use the serial reference path, so this ensemble mixes the two "
                    "(verdict-equivalent, but not numerically homogeneous).",
                    len(convergence_flags),
                    num_planned_trials,
                    exc_info=True,
                )
                batch_eval = None
            else:
                num_batched_trials += len(wave)
        if solved_wave is None:
            solved_wave = [
                solve_exact_perturbation(
                    g_tilde,
                    eta_hat,
                    g0,
                    bread_factored,
                    s_wave[index],
                    delta_wave[index],
                    config,
                )
                for index in range(len(wave))
            ]
        for (sign, j), solved in zip(wave, solved_wave, strict=True):
            convergence_flags.append(solved["converged"])
            nonfinite_flags.append(solved["nonfinite_encountered"])
            num_divergence_aborts += int(solved["aborted"])
            if not solved["converged"]:
                continue
            converged_pattern.append((sign, j))
            z_rows.append(L @ solved["delta_nl"])

    num_trials = len(convergence_flags)
    num_converged = len(converged_pattern)
    num_root_failures = num_trials - num_converged
    num_domain_failures = int(sum(nonfinite_flags))
    failure_upper_bound = clopper_pearson_upper_bound(
        num_root_failures, num_trials, config.confidence_level
    )
    domain_failure_upper_bound = clopper_pearson_upper_bound(
        num_domain_failures, num_trials, config.confidence_level
    )
    # Observed fraction for the unhealthy gate, and the lower confidence bound for the
    # confirmed-fragility gate -- same rationale as in check_exact_nonlinear_perturbations.
    observed_failure_fraction = num_root_failures / num_trials if num_trials else 1.0
    solver_unhealthy = (
        observed_failure_fraction > config.bad_direction_probability_target
    )
    failure_lower_bound = clopper_pearson_lower_bound(
        num_root_failures, num_trials, config.confidence_level
    )
    solver_confirmed_unhealthy = bool(
        not math.isnan(failure_lower_bound)
        and failure_lower_bound > config.bad_direction_probability_target
    )

    warnings_list: list[str] = []
    accounting_metrics = _resolve_accounting_metrics_and_warnings(
        warnings_list,
        trial_noun="re-solves",
        early_stop_reason=early_stop_reason,
        num_planned_trials=num_planned_trials,
        num_trials=num_trials,
        num_converged=num_converged,
        num_divergence_aborts=num_divergence_aborts,
        num_root_failures=num_root_failures,
    )
    if resolve_batch_width and num_batched_trials < num_trials:
        warnings_list.append(
            f"Batched re-solves stopped after {num_batched_trials} of {num_trials} executed "
            "trials and the rest fell back to the serial solver (see the logged exception): this "
            "ensemble mixes the two arithmetics. They agree to well within every tolerance below, "
            "but the run is not reproducible against either path alone."
        )
    V_L = L @ np.asarray(V_hat) @ L.T

    if num_converged >= MIN_CONVERGED_TRIALS_FOR_ENSEMBLE:
        z = np.array(z_rows)
        bootstrap_se = np.std(z, axis=0, ddof=1)
        se_ratio_by_target = {
            label: float(bootstrap_se[idx] / se_l[idx]) if se_l[idx] > 0 else math.nan
            for idx, label in enumerate(target_labels)
        }
        mean_shift_rows, mean_shift_pattern = _mean_shift_ensemble(
            z, converged_pattern, config.paired_directions
        )
        try:
            b_L = (
                float(
                    np.linalg.norm(
                        matrix_inv_sqrt(V_L) @ np.mean(mean_shift_rows, axis=0)
                    )
                )
                if len(mean_shift_rows)
                else math.nan
            )
        except np.linalg.LinAlgError:
            b_L = math.nan
    else:
        bootstrap_se = np.full(len(target_labels), math.nan)
        se_ratio_by_target = {label: math.nan for label in target_labels}
        b_L = math.nan
        mean_shift_pattern = list(converged_pattern)
        warnings_list.append(
            f"Only {num_converged} of {num_trials} bootstrap re-solves converged -- too few "
            "for a bootstrap SE; the check is INDETERMINATE."
        )

    null_bands = _simulate_perturbation_null_bands(
        converged_pattern,
        L.shape[0],
        config.confidence_level,
        config.random_seed + 11,
        config.num_null_band_samples,
    )
    band_lo = null_bands["per_target_sd_lower"]
    band_hi = null_bands["per_target_sd_upper"]

    ratios = np.array(
        [se_ratio_by_target[label] for label in target_labels], dtype=np.float64
    )
    # Per-target evaluation over the FINITE ratios only. A single non-identified target (se_l == 0
    # -> NaN ratio) must not disable the band gate for every other target, which is what an
    # all-or-nothing np.all(np.isfinite(ratios)) did: it let every healthy target's SE go
    # uncompared and still reported PASSED.
    band_available = not math.isnan(band_lo) and not math.isnan(band_hi)
    evaluable_ratios = ratios[np.isfinite(ratios)]
    any_above = bool(
        band_available and evaluable_ratios.size and np.any(evaluable_ratios > band_hi)
    )
    any_below = bool(
        band_available and evaluable_ratios.size and np.any(evaluable_ratios < band_lo)
    )
    unevaluable_targets = [
        label
        for label, ratio in zip(target_labels, ratios, strict=False)
        if not np.isfinite(ratio)
    ]
    band_evaluable = (
        band_available and not unevaluable_targets and len(target_labels) > 0
    )
    mean_shift_null_upper = _mean_shift_null_upper(
        null_bands,
        mean_shift_pattern,
        converged_pattern,
        L.shape[0],
        config.random_seed + 13,
        config,
    )
    mean_shift_threshold, mean_shift_gate_evaluable, mean_shift_failed = (
        _evaluate_mean_shift_gate(
            b_L,
            mean_shift_null_upper,
            len(converged_pattern) - len(mean_shift_pattern),
            solver_unhealthy,
            config,
            warnings_list,
        )
    )

    narrower_than_band_message = (
        "Bootstrap SEs fall below the sandwich SEs beyond the simulated null band: the "
        "sandwich SE is likely overstated (conservative) on this dataset."
    )

    # Same precedence rationale as check_exact_nonlinear_perturbations: statistically
    # confirmed fragility FAILS outright (measured on all executed re-solves, with no
    # optimistic-subset caveat); after that, exclusions select the easy draws, so a FAILED
    # verdict from the converged subset stands, but a PASS does not.
    if solver_confirmed_unhealthy:
        status = CheckStatuses.FAILED
        warnings_list.append(
            f"Bootstrap re-solve failure fraction {observed_failure_fraction:.4g} "
            f"({num_root_failures}/{num_trials}) has lower confidence bound "
            f"{failure_lower_bound:.4g}, above bad_direction_probability_target="
            f"{config.bad_direction_probability_target:g}: the estimating equation "
            "measurably cannot be re-solved under resampling-scale perturbations. This is "
            "a confirmed fragility of the equation, not an unevaluable ensemble."
        )
    elif num_converged < MIN_CONVERGED_TRIALS_FOR_ENSEMBLE:
        status = CheckStatuses.INDETERMINATE
    elif any_above or mean_shift_failed:
        status = CheckStatuses.FAILED
        if any_above:
            warnings_list.append(
                "Bootstrap SEs exceed the sandwich SEs beyond the simulated null band: the "
                "sandwich SE is likely understated (anticonservative) on this dataset."
            )
        if mean_shift_failed:
            warnings_list.append(
                f"Bootstrap mean shift {b_L:.4g} SE exceeds its null threshold "
                f"{mean_shift_threshold:.4g} (the simulated finite-draw null band, floored at "
                f"mean_shift_tolerance_se={config.mean_shift_tolerance_se}): the resampled "
                "roots are systematically displaced (curvature-induced bias)."
            )
    elif solver_unhealthy:
        status = CheckStatuses.INDETERMINATE
        warnings_list.append(
            f"Bootstrap re-solve failure fraction {observed_failure_fraction:.4g} exceeds "
            f"target {config.bad_direction_probability_target}: the estimating equation is "
            "fragile under resampling-scale perturbations, and the converged subset is "
            "optimistically selected."
        )
    elif not (band_evaluable and mean_shift_gate_evaluable):
        # An unevaluable gate is not a pass: with no SE comparison performed for some target (or
        # no evaluable mean-shift comparison at all) there is nothing left to have passed.
        status = CheckStatuses.INDETERMINATE
        if unevaluable_targets or not band_available:
            warnings_list.append(
                "The bootstrap-vs-sandwich SE band could not be evaluated for "
                + (
                    f"target(s) {', '.join(unevaluable_targets)} (no identified target "
                    "variance to compare against)"
                    if unevaluable_targets
                    else "any target (the null band itself is unavailable)"
                )
                + "; no SE comparison was performed for those targets, so the check is "
                "INDETERMINATE rather than passed."
            )
        if not mean_shift_gate_evaluable:
            warnings_list.append(
                "The bootstrap mean-shift gate could not be evaluated on this ensemble, so the "
                "check is INDETERMINATE rather than passed."
            )
        if any_below:
            warnings_list.append(narrower_than_band_message)
    elif any_below:
        status = CheckStatuses.WARNING
        warnings_list.append(narrower_than_band_message)
    else:
        status = CheckStatuses.PASSED

    metrics = {
        "bootstrap_se_by_target": {
            label: float(bootstrap_se[idx]) for idx, label in enumerate(target_labels)
        },
        "sandwich_se_by_target": {
            label: float(se_l[idx]) for idx, label in enumerate(target_labels)
        },
        "se_ratio_by_target": se_ratio_by_target,
        "se_ratio_null_band": [band_lo, band_hi],
        "se_ratio_band_unevaluable_targets": unevaluable_targets,
        "mean_shift_se": b_L,
        "mean_shift_null_upper": mean_shift_null_upper,
        "mean_shift_threshold": mean_shift_threshold,
        "mean_shift_gate_evaluable": mean_shift_gate_evaluable,
        "mean_shift_num_rows": len(mean_shift_pattern),
        "num_draws": num_draws,
        # Distinct multiplier draws actually re-solved. Under the sign-major trial order an early
        # stop truncates the SECOND sign block first, so this equals num_draws unless the stop
        # landed inside the first block.
        "num_draws_executed": min(num_trials, num_draws),
        "num_trials": num_trials,
        "num_converged_trials": num_converged,
        # Which re-solve path produced this ensemble: 0 for the serial reference solver, otherwise
        # the pinned lockstep batch width. Reported because the two are verdict-equivalent but not
        # bit-identical (see DiagnosticConfig.batched_bootstrap_resolves), so reproducing a run's
        # last digits means reproducing this number too.
        "resolve_batch_width": resolve_batch_width,
        **accounting_metrics,
        "root_failure_fraction": num_root_failures / num_trials
        if num_trials
        else math.nan,
        "root_failure_upper_bound": failure_upper_bound,
        "domain_failure_fraction": num_domain_failures / num_trials
        if num_trials
        else math.nan,
        "domain_failure_upper_bound": domain_failure_upper_bound,
        "multiplier_distribution": config.bootstrap_multiplier_distribution,
    }

    # Per-criterion outcomes, guarded the same way the status logic is: evaluable_ratios is
    # empty when every target is unidentified or too few trials converged (np.argmax on an
    # empty array RAISES rather than returning NaN), and b_L / mean_shift_threshold are NaN
    # whenever the ensemble was too thin to simulate a null band -- those criteria then read
    # [not evaluated] instead of showing NaN as if it were a measurement.
    ratio_gates_evaluable = bool(evaluable_ratios.size) and band_evaluable
    if ratio_gates_evaluable:
        worst_ratio = float(evaluable_ratios[np.argmax(np.abs(evaluable_ratios - 1.0))])
        ratio_value = f"worst {worst_ratio:.3g}"
    else:
        ratio_value = "not evaluable on this ensemble"
    criteria = [
        CriterionResult(
            description=(
                f"at least {MIN_CONVERGED_TRIALS_FOR_ENSEMBLE} of the bootstrap re-solves "
                "converged"
            ),
            value=f"{num_converged} of {num_trials}",
            ok=bool(num_converged >= MIN_CONVERGED_TRIALS_FOR_ENSEMBLE),
            severity="indeterminate",
        ),
        CriterionResult(
            description=(
                f"re-solve failure rate at most "
                f"{config.bad_direction_probability_target:g} "
                "(bad_direction_probability_target)"
            ),
            value=f"{observed_failure_fraction:.4g} (lower confidence bound "
            f"{failure_lower_bound:.3g})"
            if solver_confirmed_unhealthy
            else f"{observed_failure_fraction:.4g}",
            ok=not solver_unhealthy,
            # Same escalation as the exact check: INDETERMINATE while the exceedance could
            # be bad luck, FAILED once the lower confidence bound confirms it.
            severity="fail" if solver_confirmed_unhealthy else "indeterminate",
        ),
        CriterionResult(
            description=(
                f"no bootstrap/sandwich SE ratio above {band_hi:.3g}, the top of the range "
                "expected if nothing were wrong (above it, the sandwich SE is likely "
                "understated)"
            )
            if not math.isnan(band_hi)
            else (
                "no bootstrap/sandwich SE ratio above the top of the range expected if "
                "nothing were wrong"
            ),
            value=ratio_value,
            ok=(not any_above) if ratio_gates_evaluable else None,
        ),
        CriterionResult(
            description=(
                f"no bootstrap/sandwich SE ratio below {band_lo:.3g}, the bottom of that "
                "range (below it, the sandwich SE is likely conservative)"
            )
            if not math.isnan(band_lo)
            else (
                "no bootstrap/sandwich SE ratio below the bottom of the range expected if "
                "nothing were wrong"
            ),
            value=ratio_value,
            ok=(not any_below) if ratio_gates_evaluable else None,
            severity="warn",
        ),
        CriterionResult(
            description=(
                f"the re-solved estimates' average shift at most "
                f"{mean_shift_threshold:.3g} SE (the largest expected if nothing "
                "were wrong)"
            )
            if not math.isnan(mean_shift_threshold)
            else (
                "the re-solved estimates' average shift within the largest expected "
                "if nothing were wrong"
            ),
            value=f"{b_L:.3g} SE"
            if mean_shift_gate_evaluable
            else "not evaluable on this ensemble",
            ok=(not mean_shift_failed) if mean_shift_gate_evaluable else None,
        ),
    ]
    return CheckResult(
        name="multiplier_bootstrap",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        criteria=criteria,
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

    # This check takes num_directions * len(drift_path_samples) reverse-mode Jacobians of the
    # WHOLE stacked system -- the single most backward-pass-expensive thing in the suite, and
    # exactly the pass that is known to exhaust memory at real study scale. Resolve the chunk
    # size once (rather than per Jacobian) so the auto heuristic logs its decision once.
    jacobian_row_chunk_size = resolve_jacobian_row_chunk_size(
        config.jacobian_row_chunk_size, int(B.shape[0])
    )

    rho_by_direction = []
    for delta in base_delta:
        max_norm = 0.0
        for t in config.drift_path_samples:
            point = jnp.asarray(eta_hat) + t * jnp.asarray(delta)
            jac = np.asarray(
                compute_row_chunked_jacobian(
                    g_tilde, point, chunk_size=jacobian_row_chunk_size
                ),
                dtype=np.float64,
            )
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
        # The "sampled maximum, not a certified supremum" qualifier lives both in the standing
        # warning and in the criterion description: it is the single most important thing about
        # the number, and this check never fails or blocks on its own.
        criteria=[
            CriterionResult(
                description=(
                    "the estimate's derivative matrix drifts by less than 1 along every "
                    f"sampled path (a sampled maximum over {rho_by_direction.size} "
                    "direction(s), not a guaranteed worst case; this check never fails on "
                    "its own)"
                ),
                value=f"worst {float(np.max(rho_by_direction)):.3g}"
                if rho_by_direction.size
                else "no sampled directions",
                ok=bool(not np.any(rho_by_direction >= 1.0))
                if rho_by_direction.size
                else None,
                severity="warn",
            )
        ],
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
    *,
    L: np.ndarray | None = None,
) -> CheckResult:
    """
    Section 8. Reports per-update diagonal-block and theta-block singular values/condition
    numbers, off-diagonal coupling magnitudes, target-covariance rank/eigenvalues, and the
    sensitivity of target SEs to a numerically negligible perturbation of B_hat. Does not
    hard-code a universal condition-number threshold: conditioning is reported, not judged, here.

    `L` is the target selector (the suite always supplies it); the target covariance whose rank
    is judged is L @ V_hat @ L^T, falling back to the theta block of V_hat when no selector is
    given. It is emphatically NOT the full joint V_hat: the joint sandwich's rank is generically
    at least theta_dim even when the theta target block is exactly singular, because the healthy
    beta blocks supply that rank on their own -- judging identification on it made this gate
    unfireable at any real study scale (beta_total ~4000 vs. theta_dim ~5) and reported
    beta-block eigenvalues, in beta's units, under a "target covariance" name.
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

    V = np.asarray(V_hat, dtype=np.float64)
    target_covariance = (
        np.atleast_2d(
            np.asarray(L, dtype=np.float64) @ V @ np.asarray(L, dtype=np.float64).T
        )
        if L is not None
        else V[-theta_dim:, -theta_dim:]
    )
    target_dim = target_covariance.shape[0]
    eigvals_target = np.linalg.eigvalsh(target_covariance)
    max_eig = float(eigvals_target.max()) if eigvals_target.size else math.nan
    rank_estimate = (
        int(np.sum(eigvals_target > config.rank_tolerance * max_eig))
        if max_eig > 0
        else 0
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
    if rank_estimate < target_dim:
        # FAILED, not INDETERMINATE: in the ADS-142 undercoverage hunt this condition
        # identified 76/76 of the zero-width-interval collapses, and _derive_verdict already
        # treats it as INVALID-grade evidence -- the row's status now says the same thing the
        # verdict does, instead of reading "could not be evaluated" while silently driving
        # the worst verdict.
        status = CheckStatuses.FAILED
        warnings_list.append(
            f"Target covariance rank estimate {rank_estimate} < target dimension "
            f"{target_dim}: some reported components are unidentified. In calibration "
            "experiments this condition marked every catastrophic variance collapse, so it "
            "is a measured failure, not a weak-identification caveat."
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
            # Eigenvalues of the TARGET covariance (L V_hat L^T, or the theta block of V_hat when
            # no selector was supplied) -- not of the joint sandwich, whose beta blocks dominate
            # the spectrum and carry different units.
            "target_covariance_eigenvalues": eigvals_target.tolist(),
            "target_covariance_rank_estimate": rank_estimate,
            "target_covariance_dim": target_dim,
            "numerical_sensitivity_max_relative_se_change": sensitivity_max,
        },
        warnings=warnings_list,
        # Reports the SE sensitivity and the identification rank, deliberately NOT the bread
        # condition number: that same figure already has its own pipeline row
        # (joint_bread_condition_number), and repeating it here would read as two findings.
        # NaN-guarded because a nonfinite perturbed SE propagates into sensitivity_max, and
        # `nan > tolerance` is False -- a NaN reads [not evaluated], not a passing "nan%".
        criteria=[
            CriterionResult(
                description=f"all {target_dim} estimate components identified",
                value=f"{rank_estimate} of {target_dim}",
                ok=bool(rank_estimate >= target_dim),
            ),
            CriterionResult(
                description=(
                    f"SE moves at most {config.se_distortion_tolerance:.0%} under a "
                    f"numerically negligible {rel_scale:.0e} nudge to the bread matrix "
                    "(se_distortion_tolerance; on its own this criterion never fails the "
                    "check)"
                ),
                value=f"{sensitivity_max:.2%}"
                if not math.isnan(sensitivity_max)
                else "not evaluable (nonfinite perturbed SE)",
                ok=bool(sensitivity_max <= config.se_distortion_tolerance)
                if not math.isnan(sensitivity_max)
                else None,
                severity="indeterminate",
            ),
        ],
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
        n_eff_threshold = max(
            config.influence_n_eff_min_floor, config.influence_n_eff_min_fraction * n
        )
        if n_eff < n_eff_threshold or p_max > config.influence_p_max_tolerance:
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

    # Derived from by_target, NOT from the loop locals n_eff/p_max/L_c: those hold the LAST
    # target's values at this point, or a stale earlier target's if the last one hit the
    # degenerate `continue`, and are unbound entirely when there are no targets.
    evaluable_n_eff = [
        entry["n_eff"] for entry in by_target.values() if not math.isnan(entry["n_eff"])
    ]
    evaluable_p_max = [
        entry["p_max"] for entry in by_target.values() if not math.isnan(entry["p_max"])
    ]
    n_eff_gate = max(
        config.influence_n_eff_min_floor, config.influence_n_eff_min_fraction * n
    )
    # Both criteria are severity="warn": this check never fails or blocks on its own.
    return CheckResult(
        name="influence_concentration",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        criteria=[
            CriterionResult(
                description=(
                    f"an effective sample of at least {n_eff_gate:.1f} subjects behind "
                    "every reported quantity (the larger of "
                    f"{config.influence_n_eff_min_floor:g} subjects and "
                    f"{config.influence_n_eff_min_fraction:.0%} of the {n}; this check "
                    "never fails on its own)"
                ),
                value=f"worst {min(evaluable_n_eff):.1f} of {n}"
                if evaluable_n_eff
                else "not evaluable (no variance to attribute)",
                ok=bool(min(evaluable_n_eff) >= n_eff_gate)
                if evaluable_n_eff
                else None,
                severity="warn",
            ),
            CriterionResult(
                description=(
                    "no single subject accounts for more than "
                    f"{config.influence_p_max_tolerance:.0%} of any reported quantity's "
                    "variance (influence_p_max_tolerance)"
                ),
                value=f"largest {max(evaluable_p_max):.1%}"
                if evaluable_p_max
                else "not evaluable (no variance to attribute)",
                ok=bool(max(evaluable_p_max) <= config.influence_p_max_tolerance)
                if evaluable_p_max
                else None,
                severity="warn",
            ),
        ],
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
    num_nonfinite_weight_directions = 0
    num_weight_directions = 0
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
        num_weight_directions += 1
        if not np.all(np.isfinite(final_weights)):
            status = CheckStatuses.FAILED
            num_nonfinite_weight_directions += 1
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

    # Per-criterion outcomes. On an empty active set pandas' min()/max() are NaN while the
    # finiteness and (0,1) tests both vacuously hold, so every probability criterion reads
    # [not evaluated] rather than "[nan, nan]" posing as a passing measurement. The floor and
    # ceiling bounds are Optional and default to None: each appears only when the caller
    # actually supplied that bound.
    have_rows = bool(all_probs.size)
    probs_finite = bool(np.all(np.isfinite(all_probs)))
    # NOT redundant with the interval criterion below it: a NaN passes `<= 0` and `>= 1`
    # vacuously (NaN comparisons are always false), so only this criterion catches it -- and
    # the displayed [min, max] range cannot reveal one either, because pandas' min()/max()
    # skip NaN. Hence this criterion reports a finiteness COUNT, not the range.
    num_nonfinite_probs = int(np.sum(~np.isfinite(all_probs)))
    prob_range_value = (
        f"[{metrics['action_prob_global_min']:.4g}, "
        f"{metrics['action_prob_global_max']:.4g}] over {all_probs.size} active rows"
        if have_rows
        else "no active rows"
    )
    criteria = [
        CriterionResult(
            description=(
                "every recorded action probability finite (a NaN would pass the interval "
                "test below silently)"
            ),
            value=(
                f"all {all_probs.size} finite"
                if probs_finite
                else f"{num_nonfinite_probs} of {all_probs.size} nonfinite"
            )
            if have_rows
            else "no active rows",
            ok=probs_finite if have_rows else None,
        ),
        CriterionResult(
            description="every recorded action probability strictly inside (0, 1)",
            value=prob_range_value,
            ok=bool(not (np.any(all_probs <= 0) or np.any(all_probs >= 1)))
            if have_rows
            else None,
        ),
    ]
    if config.exploration_floor is not None:
        criteria.append(
            CriterionResult(
                description=(
                    "every recorded action probability at or above the deployment's "
                    f"exploration_floor {config.exploration_floor:g}"
                ),
                value=f"min {metrics['action_prob_global_min']:.4g}"
                if have_rows
                else "no active rows",
                ok=bool(not np.any(all_probs < config.exploration_floor))
                if have_rows
                else None,
            )
        )
    if config.exploration_ceiling is not None:
        criteria.append(
            CriterionResult(
                description=(
                    "every recorded action probability at or below the deployment's "
                    f"exploration_ceiling {config.exploration_ceiling:g}"
                ),
                value=f"max {metrics['action_prob_global_max']:.4g}"
                if have_rows
                else "no active rows",
                ok=bool(not np.any(all_probs > config.exploration_ceiling))
                if have_rows
                else None,
            )
        )
    criteria.append(
        CriterionResult(
            description=(
                "the importance weights stay finite when the policy parameters are "
                "nudged a typical sampling fluctuation"
            ),
            value=f"{num_nonfinite_weight_directions} of {num_weight_directions} "
            "nudge direction(s) produced a nonfinite weight"
            if num_weight_directions
            else "not evaluated on this run",
            ok=(num_nonfinite_weight_directions == 0)
            if num_weight_directions
            else None,
        )
    )
    # The ESS figure is informational, not a criterion: no threshold for it has been
    # calibrated, so it is reported without a gate and can never set the status.
    finite_ess = [value for value in ess_by_direction if math.isfinite(value)]
    return CheckResult(
        name="exploration_and_weights",
        status=status,
        metrics=metrics,
        warnings=warnings_list,
        message=(
            f"worst-direction importance weights keep {min(finite_ess):.0%} of the "
            "effective sample (reported without a gate)"
            if finite_ess
            else ""
        ),
        criteria=criteria,
    )


###############################################################################
# Orchestration
###############################################################################


def _local_nonlinearity_headline_max(metrics: dict[str, Any]) -> float:
    """The headline-radius max a_{j,l} over targets, from check_local_nonlinearity's metrics."""
    headline_radius = metrics.get("headline_radius")
    per_radius = metrics.get("per_radius", {})
    if headline_radius not in per_radius:
        return math.nan
    a_by_target = per_radius[headline_radius].get("a_by_target", {})
    maxima = [summary.get("max", math.nan) for summary in a_by_target.values()]
    finite = [m for m in maxima if not math.isnan(m)]
    return max(finite) if finite else math.nan


def _combine_classification(check_results: dict[str, CheckResult]) -> str:
    statuses = [result.status for result in check_results.values()]
    if any(s == CheckStatuses.FAILED for s in statuses):
        return DiagnosticClassifications.FAILED
    if any(s == CheckStatuses.INDETERMINATE for s in statuses):
        return DiagnosticClassifications.INDETERMINATE
    return DiagnosticClassifications.LOCALLY_SUPPORTED


def _derive_verdict(
    check_results: dict[str, CheckResult],
    input_check_results: dict[str, CheckResult],
    hard_failed: bool,
    theta_dim: int,
    config: DiagnosticConfig,
) -> tuple[str, str]:
    """
    Maps the check results onto the decision-level verdict (see DiagnosticVerdicts) -- the
    single-run trust protocol the ADS-142 calibration experiments support, made machine-readable.
    Deliberately additive: `classification` and every check's own status are unchanged; this is
    a pure function of them (plus two metrics reads).

    Precedence, and the evidence behind each rung:

    1. INVALID -- any hard/measured failure, or a rank-deficient target covariance
       (`bread_stability`'s `target_covariance_rank_estimate < theta_dim`). The rank condition
       is pulled up to INVALID rather than left in UNCERTIFIABLE because it identified 76/76 of
       the zero-width-interval collapses in the undercoverage hunt -- runs whose CIs miss with
       certainty. (check_bread_stability now also FAILS on it directly, so any_failed usually
       covers this; the explicit metrics read stays as defense for reports pickled by older
       check versions, whose row says INDETERMINATE.)
    2. UNCERTIFIABLE -- any indeterminate check (an unconfirmed re-solve exceedance, a
       partially censored ensemble, unevaluable gates; CONFIRMED fragility and TOTAL censoring
       now fail instead), any indeterminate INPUT row (an input prerequisite that never ran,
       e.g. under suppress_all_data_checks -- unvalidated inputs cannot certify), or the
       a_{j,l} screen called for the multiplier bootstrap and it did
       not run (`multiplier_bootstrap="off"`, or the report predates the check): the calibrated
       division of labor makes the bootstrap the verdict layer once the screen trips, so its
       absence there means no verdict was reached, not a pass.
    3. CONSERVATIVE -- nothing failed or unresolved, but a calibrated conservatism signal fired:
       the bootstrap's below-band WARNING (sandwich SE overstated), or
       `influence_concentration`'s WARNING (its n_eff floor has measured precision 0.73 /
       recall 0.59 for a >2x-inflated variance, and the inflation it predicts is conservative).
    4. CERTIFIED -- everything else, with `verdict_basis` recording how: "bootstrap" (the
       linearization-free SE comparison ran and passed) or "screen" (the run was quiet enough
       that the screen never called for it -- every screen-quiet design in the calibration grids
       was well-calibrated). On the clean validation cells the bootstrap's SEs tracked the
       replicate-pool truth to ratios of 0.96-1.01; on the influence cell -- the design built to
       falsify the no-policy-replay concern -- the per-coordinate ratios were 1.05 / 1.32 / 0.99
       / 1.31, i.e. no systematic undershoot. See docs/adr/0002's round-2 validation results.

    Warnings outside the calibrated set (scaling exponents, jacobian-drift rho, exploration
    leading indicators) do not move the verdict: they are reported for reading, and the
    experiments gave them no operating characteristics to gate on.
    """
    any_failed = (
        hard_failed
        or any(r.status == CheckStatuses.FAILED for r in check_results.values())
        or any(r.status == CheckStatuses.FAILED for r in input_check_results.values())
    )
    bread = check_results.get("bread_stability")
    rank_estimate = (
        bread.metrics.get("target_covariance_rank_estimate") if bread else None
    )
    rank_deficient = rank_estimate is not None and rank_estimate < theta_dim
    if any_failed or rank_deficient:
        return DiagnosticVerdicts.INVALID, ""

    if any(
        r.status == CheckStatuses.INDETERMINATE for r in check_results.values()
    ) or any(
        r.status == CheckStatuses.INDETERMINATE for r in input_check_results.values()
    ):
        # Input rows too: an INDETERMINATE one means an input prerequisite did not run
        # (suppress_all_data_checks) or could not conclude, and unvalidated inputs cannot
        # certify -- without this, suppressing the first wave left quiet runs CERTIFIED.
        return DiagnosticVerdicts.NOT_CERTIFIED, ""

    bootstrap = check_results.get("multiplier_bootstrap")
    local = check_results.get("local_nonlinearity")
    headline = _local_nonlinearity_headline_max(local.metrics) if local else math.nan
    screen_quiet = (
        local is not None
        and local.status == CheckStatuses.PASSED
        and (math.isnan(headline) or headline <= config.bootstrap_screen_a_jl_threshold)
    )
    if bootstrap is None and not screen_quiet:
        return DiagnosticVerdicts.NOT_CERTIFIED, ""

    influence = check_results.get("influence_concentration")
    conservative = (
        bootstrap is not None and bootstrap.status == CheckStatuses.WARNING
    ) or (influence is not None and influence.status == CheckStatuses.WARNING)
    if conservative:
        return DiagnosticVerdicts.CONSERVATIVE, ""

    basis = (
        VerdictBases.BOOTSTRAP
        if bootstrap is not None and bootstrap.status == CheckStatuses.PASSED
        else VerdictBases.SCREEN
    )
    return DiagnosticVerdicts.CERTIFIED, basis


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
    extra_input_check_results: dict[str, CheckResult] | None = None,
) -> DiagnosticReport:
    """
    Runs the full layered diagnostic suite and combines every check into one DiagnosticReport.
    The decision-level summary this pairs with is DiagnosticVerdicts (see _derive_verdict);
    `classification` itself stays deliberately WARNING-blind.
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

    input_check_results = run_input_checks(legacy_check_callables)
    # Pipeline-level input rows (the first wave, the reconstruction check, the sum-to-zero
    # check) are merged BEFORE hard_failed and the verdict are derived, so they shape both.
    # They used to be grafted onto the finished report instead, which let
    # suppress_all_data_checks leave a quiet run CERTIFIED while its own report said the
    # input prerequisites never ran.
    input_check_results.update(extra_input_check_results or {})

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
    )

    hard_failed = check_results[
        "root_and_implementation"
    ].status == CheckStatuses.FAILED or any(
        result.status == CheckStatuses.FAILED for result in input_check_results.values()
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

        if config.multiplier_bootstrap not in ("off", "auto", "always"):
            raise ValueError(
                f"Unknown multiplier_bootstrap mode: {config.multiplier_bootstrap!r} "
                "(expected 'off', 'auto', or 'always')."
            )
        run_bootstrap = config.multiplier_bootstrap == "always"
        if config.multiplier_bootstrap == "auto":
            # The cheap a_{j,l} check screens for the expensive bootstrap: trigger on a headline
            # exceedance of the screen threshold, or on any non-PASSED local-nonlinearity status
            # (a rank-deficient or otherwise unusual local check is exactly when a
            # linearization-free second opinion is worth its cost).
            local_result = check_results["local_nonlinearity"]
            headline = _local_nonlinearity_headline_max(local_result.metrics)
            run_bootstrap = (
                not math.isnan(headline)
                and headline > config.bootstrap_screen_a_jl_threshold
            ) or local_result.status != CheckStatuses.PASSED
        if run_bootstrap:
            check_results["multiplier_bootstrap"] = check_multiplier_bootstrap(
                g_tilde,
                eta_hat,
                g0,
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
            B_hat_np, M_hat, beta_dim, theta_dim, num_subjects, V_hat, config, L=L
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

    for result in list(check_results.values()) + list(input_check_results.values()):
        warnings_list.extend(f"[{result.name}] {w}" for w in result.warnings)

    classification = (
        DiagnosticClassifications.FAILED
        if hard_failed
        else _combine_classification(check_results)
    )
    verdict, verdict_basis = _derive_verdict(
        check_results, input_check_results, hard_failed, theta_dim, config
    )

    # These are the PLANNED Monte Carlo sizes -- what the config asked for. The re-solve checks can
    # legitimately execute fewer trials than planned (the sequential early stop of
    # _starvation_early_stop_reached), so the executed count is reported alongside rather than
    # letting the plan stand in for it; per-check metrics carry the same distinction as
    # num_planned_trials / num_trials / early_stopped.
    monte_carlo_counts = {
        "num_directions": config.num_directions,
        "num_exact_directions": config.num_exact_directions or config.num_directions,
        "num_bootstrap_draws": config.num_bootstrap_draws,
    }
    bootstrap_result = check_results.get("multiplier_bootstrap")
    if bootstrap_result is not None:
        monte_carlo_counts["num_bootstrap_draws_executed"] = int(
            bootstrap_result.metrics.get(
                "num_draws_executed", config.num_bootstrap_draws
            )
        )

    return DiagnosticReport(
        classification=classification,
        check_results=check_results,
        input_check_results=input_check_results,
        metrics={name: result.metrics for name, result in check_results.items()},
        tolerances_used=dataclasses.asdict(config),
        warnings=warnings_list,
        monte_carlo_counts=monte_carlo_counts,
        target_labels=target_labels,
        rank_diagnostics=check_results.get(
            "bread_stability", CheckResult(name="", status="", metrics={})
        ).metrics,
        verdict=verdict,
        verdict_basis=verdict_basis,
    )
