# Diagnostic suite for the adjusted sandwich

`lifejacket.diagnostics` implements a layered set of checks for the adjusted-sandwich
inference produced by `post_deployment_analysis.analyze_dataset`. It runs by default
(`run_diagnostics` defaults to `True`; pass `run_diagnostics=False` / `--run_diagnostics=False`
to disable it), does not change the adjusted sandwich estimator itself, and never automatically
falls back to the classical sandwich when a check fails: failure of these checks means the
current adjusted-Wald analysis lacks support, not that the classical sandwich is valid instead.
The default `DiagnosticConfig()` only runs the cheap checks below; the expensive checks stay
opt-in regardless of `run_diagnostics`, each behind its own switch -- the exact-nonlinear-
perturbation and Jacobian-drift checks behind `compute_exact_nonlinear_roots`, and the
frozen-score multiplier bootstrap (section 3b) behind `multiplier_bootstrap`
(`"off"`/`"auto"`/`"always"`; `"auto"` triggers it off the cheap `a_{j,l}` screen).

The suite is organized around one idea: the adjusted sandwich can fail for different reasons,
and a single number cannot certify all of them.

```
correct inputs -> adequate exploration -> stable moments and bread -> linearization -> CLT/normal approximation
```

The `r_j`/`q_j`/`a_{j,l}` family of diagnostics (sections 2 below) examines mainly the
linearization link. They cannot certify the others -- which is why the suite also reports
exploration/importance-weight diagnostics, bread-stability diagnostics, and influence
concentration, why an exact nonlinear perturbation check exists as a measurement-grade (and more
expensive) alternative, and why the multiplier bootstrap (section 3b) exists as the per-dataset
verdict layer on the reported SEs themselves.

## What each check tests, and what it cannot test

| Check (module function) | What it targets | What it cannot rule out |
| --- | --- | --- |
| `run_input_checks` (`DiagnosticReport.input_check_results`) | Black-and-white data/wiring correctness (e.g. can the recorded action probabilities be reconstructed from the supplied function) -- not a statistical measurement | Anything about the adjusted sandwich's own statistical adequacy; this is deliberately not what it's for |
| `check_root_and_implementation` | An unsolved estimating equation, broken/inconsistent derivatives, non-finite inputs, a broken linear solve | Anything downstream of a correctly-solved root: overlap, bread conditioning, linearization, the CLT step |
| `check_local_nonlinearity` (`r_j`, `c_j`, `a_{j,l}`) | Whether the first-order Taylor expansion of the estimating equation is adequate at the estimation scale, for the *reported contrasts specifically* | Whether the sampling distribution is actually close to normal (that's `check_influence_concentration`); overlap/positivity; whether *un-sampled* directions behave differently |
| `check_exact_nonlinear_perturbations` | The same question as above, but exactly (via continuation) rather than via a one-step Taylor correction, plus the resulting distortion to the target's covariance, mean, and tail quantiles | Anything not captured by the sampled perturbation directions; with few directions, only a wide bound on the "bad direction" probability is achievable |
| `check_multiplier_bootstrap` | Whether the sandwich SEs match a linearization-free, empirically-weighted (generalized bootstrap) estimate of the sampling distribution from this one dataset -- verdict judged against its own simulated finite-draw null band, no engineering tolerance | Derivative-level misspecification of the stacked representation: an update rule that matches the recorded trajectory at value level (so the sum-to-zero and reconstruction input checks pass) but responds differently to perturbed data, e.g. a constraint inactive on the realized path -- shared blind spot with the adjusted sandwich |
| `check_jacobian_drift` (`rho_j`) | How much the Jacobian changes along a sampled path -- a heuristic input to a contraction-style error bound | Nothing rigorously: this is a *sampled path maximum*, never a certified supremum. `rho_j < 1` does not prove the nonlinear correction is small; `rho_j >= 1` does not by itself prove failure |
| `check_bread_stability` | Numerical conditioning of the bread matrix and its blocks, and the sensitivity of target SEs to numerically negligible perturbations | Statistical identification as distinct from numerical conditioning (a well-conditioned-looking bread can still reflect weak identification if the *meat* is what's driving the standard error up) |
| `check_influence_concentration` (`p_max`, `n_eff`, third-moment) | Whether the estimator's fluctuation is built from many small contributions (the premise the CLT approximation needs), or is dominated by a handful of subjects | It does not itself validate normality -- it only flags a specific, common way that a CLT approximation can fail |
| `check_exploration_and_weights` | Positivity/overlap, importance-weight concentration (ESS), and policy-score-derivative magnitude, evaluated at the estimate *and* under sandwich-scale perturbations | Whether the *supplied* exploration bounds are the deployment's real design bounds -- those must be supplied by the caller via `exploration_floor`/`exploration_ceiling` to be enforced as hard requirements |


## The checks, in detail

Every check below takes the same core inputs, computed once in `run_diagnostic_suite`:
`g_tilde` (a closure evaluating the average per-subject weighted estimating-function stack at a
given `eta`), `eta_hat`, `g0 = g_tilde(eta_hat)`, `B_hat` (the joint bread, LU-factored once via
`factor_bread` and reused everywhere via `solve_with_bread`/`solve_with_bread_transpose` -- the
bread is never explicitly inverted), `M_hat` (the joint meat), `V_hat` (= `joint_sandwich_matrix`,
which already equals `Cov(eta_hat)` -- see the scaling-convention note at the top of
`diagnostics.py`), and `per_subject_stacks` (`g_i(eta_hat)` for every subject `i`). A target
selector `L` (rows are contrasts `l`) and matching `target_labels` are built once by
`default_contrast_matrix`: if `config.contrast_matrix` is `None`, `L` defaults to the identity on
the `theta` block, i.e. one contrast per component of `theta`, labeled `theta_0`, `theta_1`, ...

### 0. Input-check results -- `run_input_checks` (not one of the numbered statistical checks)

Kept deliberately separate from every check below, in `DiagnosticReport.input_check_results`
(a `dict[str, CheckResult]`, distinct from `check_results`) rather than folded into any of them.
The distinction: these are black-and-white correctness questions about the supplied *data and
functions* (does a reconstruction match, is a shape/index consistent) -- not numeric measurements
of how appropriate the adjusted sandwich is for this experiment, which is what every numbered
check below actually does. Conflating the two would mean a data-wiring bug and a genuine
statistical finding could look identical in the report.

`run_input_checks` takes `(name, callable)` pairs (the same `legacy_check_callables` argument
`run_diagnostic_suite` has always accepted) and invokes each; an exception from any of them is
caught and turned into its own `failed` `CheckResult` (named after that check, with the exception
message) rather than propagating. As of 2026-09-02, `post_deployment_analysis.analyze_dataset`
no longer wires `require_action_probabilities_in_analysis_df_can_be_reconstructed` in here --
that check evaluates the action-probability function over every active row (the most expensive
input check) and was being executed twice per analysis. It already runs in the first-wave input
checks near the start of `analyze_dataset`, it is a hard failure with no interactive continue
path, and nothing between that run and the suite touches `analysis_df` or
`action_prob_func_args`, so reaching the suite at all proves it passed. `analyze_dataset`
therefore records the outcome into the report directly under the same
`action_probabilities_reconstructed` name: `passed` (with a provenance message) when data checks
ran, and an `indeterminate` entry with the message `"Not run: suppress_all_data_checks=True."`
when they were suppressed -- a suppressed check and a passing one never look alike in the
report. Consumers of `input_check_results` should therefore treat `indeterminate` as "not
run". The sum-to-zero check is deliberately **not** also wired in here, because
the main `analyze_dataset` pipeline already runs it (interactively, and under the same
`suppress_all_data_checks` gate) before the suite starts -- and, as of the ADS-142 follow-ups, in
SE-standardized form:
`input_checks.require_estimating_functions_sum_to_zero_se_standardized` judges each component
of the residual (the subject-mean estimating function) against its own standard error, taken
directly from the per-subject values that were averaged
(`a_j = |mean_i psi_ij| / (rms_i psi_ij / sqrt(n))`, soft `0.01` / hard `0.1`, with per-update
attribution). It is both a portable replacement for the legacy raw-units version (whose absolute
tolerances were reward-scale-dependent and false-alarmed on healthy high-noise runs -- see
`docs/adr/0002`) and a genuine value-level test of the stacked model of the algorithm: the
recorded `eta_k` must actually root the claimed update-`k` equations on the realized data. An
earlier incarnation standardized by the displacement `|(B_hat^{-1} r)_j| / SE_j` instead; that
form inherited the bread/sandwich degeneracies (masked under meat-driven SE blow-ups, silent or
astronomically wrong on collapsed SEs) and was replaced 2026-09-02 -- the current statistic
consults neither matrix, is bounded by `sqrt(n)`, and treats a component at or below the
stack's numerical noise floor as trivially rooted rather than excluded (provably safe:
`|r_j| <= s_j` bounds any skipped residual by the floor itself). The legacy
`require_estimating_functions_sum_to_zero` remains only for backward compatibility.

Any `failed` entry in `input_check_results` short-circuits the rest of the suite exactly like a
`failed` `root_and_implementation` does (`run_diagnostic_suite` sets `hard_failed = True` either
way) and forces `classification == "failed"` -- but `check_root_and_implementation`'s own `status`
is unaffected by it either way, so a data-hygiene failure is never visible as if it were a
statistical one.

### 1. Root and implementation accuracy -- `check_root_and_implementation`

Always runs first; a `failed` result here short-circuits every other check (`run_diagnostic_suite`
sets `hard_failed = True` and skips straight to building the report).

1. **Finiteness.** `metrics["finite_inputs"]` is `True` only if `g0`, `B_hat`, and `M_hat` are all
   finite. If not, the check fails immediately with no further computation.
2. **Root correction in SE units.** Solves `B_hat @ d_root = -g0` (one stable solve). For each
   contrast `l` with `se(l^T eta_hat) = sqrt(l @ V_hat @ l) > 0`, reports
   `a_root_by_target[label] = |l @ d_root| / se(l^T eta_hat)`, and `a_root_max` as the max over
   identified contrasts. **Hard-fails** if `a_root_max > config.root_error_tolerance_se`
   (default `0.01`). Contrasts with (numerically) zero target variance are excluded from this
   check and reported as a warning instead -- that is a rank/identification issue for
   `check_bread_stability`/`check_local_nonlinearity` to flag, not a root-solving failure.
3. **Backward solve residual.** `metrics["backward_relative_residual"] = ||B_hat @ d_root + g0|| /
   (||B_hat|| * ||d_root|| + eps)` -- a standard numerical-linear-algebra sanity check on the
   solve itself. **Hard-fails** if it exceeds `config.backward_residual_tolerance` (default
   `1e-6`). Unlike every other tolerance in `DiagnosticConfig`, this one is not an engineering
   guess needing simulator calibration: a healthy LU-based solve is backward-stable by
   construction, so this residual sits near float64 machine epsilon (~1e-16) regardless of how
   ill-conditioned `B_hat` is (confirmed empirically across condition numbers spanning twelve
   orders of magnitude) -- `1e-6` is already ~10 orders of magnitude looser than anything a
   healthy solve produces, so this is a smoke detector for a broken solve (e.g. a stale/mismatched
   `bread_factored`), not a statement about statistical adequacy.
4. **Directional finite-difference check.** For `config.finite_difference_num_directions`
   (default `3`) random unit directions `v`, compares the central finite difference
   `(g_tilde(eta_hat + h v) - g_tilde(eta_hat - h v)) / (2h)` (with
   `h = config.finite_difference_step`, default `1e-3`, chosen for float32 precision -- see the
   comment on `DiagnosticConfig.finite_difference_step`) against the analytic `B_hat @ v`. Reports
   `finite_difference_relative_errors` (one per direction) and their max; a max relative error
   above `1%` is a **warning** (catches a broken/mismatched derivative, not meant to certify
   numerical precision at the last bit).
Legacy `input_checks` re-verification (was previously folded into this check's own status) now
lives separately -- see "0. Input-check results" below.

### 2. Local nonlinearity -- `check_local_nonlinearity` (`r_j`, `c_j`, `a_{j,l}`, joint Mahalanobis)

Samples `config.num_directions` (default `15`) perturbation directions via
`sample_perturbation_directions`: draws `u_j ~ N(0, M_hat)` as `u_j = (W @ per_subject_stacks) /
sqrt(n)` with `W ~ N(0, I_n)`, sets `s_j = u_j / sqrt(n)`, and solves `delta_j = B_hat^{-1} s_j`
(one stable solve per direction, at unit radius). Because of how `V_hat` is scaled,
`Cov_j(delta_j) == V_hat` exactly -- these are literally simulated draws of `eta_hat`'s own
sampling fluctuation, not arbitrary probes.

For every configured radius in `config.perturbation_radii` (default `(0.25, 0.5, 1.0, 1.5)`), and
for both signs of each direction when `config.paired_directions` (default `True`), computes via
`evaluate_taylor_remainder_and_correction`:

- `R_j = g_tilde(eta_hat + s*delta_j) - g0 - B_hat @ (s*delta_j)` (the Taylor remainder);
- `r_j = ||R_j|| / ||B_hat @ (s*delta_j)||` (retained, equation-space, no universal threshold --
  see below);
- `c_j` solving `B_hat @ c_j = -R_j` (the first nonlinear parameter correction);
- `q_j = ||c_j|| / ||s*delta_j||` (secondary, parameter-space, no universal threshold);
- `a_{j,l} = |l @ c_j| / se(l^T eta_hat)` for every contrast -- **the headline metric**.

**Domain censoring.** Probing out to `1.5x` the sandwich scale routinely leaves the domain of a
log/logit/sqrt-shaped estimating function, so `g_tilde` returning a nonfinite value at
`eta_hat + s*delta_j` is an expected outcome, not an error. Such a probe is **censored**:
`evaluate_taylor_remainder_and_correction` returns a boolean `finite` mask alongside `R`/`r`/`c`/
`q` (censored rows of the latter three are `NaN`) and never routes a nonfinite row into
`solve_with_bread`, whose `check_finite=True` LU solve would otherwise raise on the whole stacked
right-hand side and take the entire diagnostic report down with it. `check_local_nonlinearity`
reports `num_probes`/`num_domain_censored_probes`/`domain_censored_fraction` both per radius and
at the top level of its metrics, and every summary above is computed on the evaluable probes only
(including the paired even/odd norms, which use directions finite in both signs).

For each radius and target, `per_radius[radius]["a_by_target"][label]` reports `median`, `p90`,
`p95`, `max`, the fraction of **evaluable** probes with `a_{j,l} >
config.nonlinear_correction_tolerance_se` (default `0.10`; `NaN` when a radius has no evaluable
probe, rather than a spurious `0.0` -- counting censored probes as non-exceedances would let
domain failures dilute the rate toward zero), and a Clopper-Pearson upper bound on that exceedance
fraction over the same evaluable count. `per_radius[
radius]["r"]`/`["q"]` report the same quantile summary for `r_j`/`q_j`. When paired directions are
on, `paired_even_norm_median`/`paired_odd_norm_median` report the median norm of `(c_j^+ +
c_j^-)/2` and `(c_j^+ - c_j^-)/2` -- the even (quadratic-like) and odd (linear-like) parts of the
correction.

Also computes, per radius, `joint_mahalanobis` via `joint_mahalanobis_correction`: for a
vector target, `a_{j,L} = ||V_L^{+1/2} (L c_j)||` using an eigenvalue-floor pseudoinverse of
`V_L = L @ V_hat @ L^T` (only where `V_L`'s eigenvalues fall below `config.rank_tolerance` times
its max eigenvalue), reporting the same quantile summary plus `effective_rank`, `target_dim`, and
`rank_deficient`.

**Status logic:** `warning` if the headline `a_by_target` max at the `1.0`-radius (or the largest
configured radius, if `1.0` isn't present) exceeds `config.nonlinear_correction_tolerance_se`, or
if a radius-scaling warning fires (see below), or if any radius's target covariance is
rank-deficient, in which case the status is escalated to `indeterminate` instead. Any domain
censoring at all attaches a warning naming the censored counts and downgrades a `passed` to
`warning`: the surviving probes are exactly the directions in which `g_tilde` stayed defined, so
every statistic above is computed on an optimistically selected subset -- a measured exceedance on
it still stands, but a clean pass on it does not (the same precedence the exact and bootstrap
checks apply to their non-converged directions). Censoring escalates to `indeterminate` once the
headline radius has fewer than 3 evaluable probes or more than half of its probes censored, at
which point the surviving subset cannot support any verdict -- and to `failed` when NO probe at
the headline radius survives: each censored probe is a measured event (`g_tilde` was nonfinite at
that point), so total censoring at sampling scale is direct evidence that the linearization the
adjusted sandwich relies on has no domain around the estimate, not an absence of evidence.
`passed` otherwise. The **size of the measured `a_{j,l}` never hard-fails on its own** -- see
"Why `r_j`/raw `q_j` have no universal threshold" below for why; total censoring is the one
`failed` this check can produce, and it is a statement about `g_tilde`'s domain, not about any
nonlinearity magnitude.

**Radius-scaling/pairing warnings** (informational, not pass/fail): comparing the smallest and
largest configured radii, a warning fires if the observed exponent of `r_j`'s median vs. radius
deviates from `1` by more than `0.5`, or if `a_{j,l}`'s deviates from `2` by more than `1` --
skipped below `1e-4` (`_scaling_noise_floor`), where the ratio is dominated by float32 roundoff
rather than genuine curvature and a "scaling exponent" is meaningless noise (this is exactly what
an affine estimating map looks like).

### 3. Exact nonlinear perturbation -- `check_exact_nonlinear_perturbations`

Opt-in (`config.compute_exact_nonlinear_roots`, default `False` -- it costs one root-solve per
direction, so it's meant to be turned on deliberately, not run by default). Uses
`config.num_exact_directions` directions (falls back to `config.num_directions` if unset).

For each direction/sign, `solve_exact_perturbation` solves `g_tilde(eta_hat + delta) - g0 = s_j`
exactly via continuation: steps `lambda` over `config.continuation_steps` (default `10`) values
from `0` to `1`, and at each step iterates the chord-Newton update
`d <- d - B_hat^{-1}(g_tilde(eta_hat + d) - g0 - lambda*s_j)` (warm-started from the previous
step's `d`, starting the whole path from the linear solution `delta_j`) for up to
`config.nonlinear_solver_max_iterations` (default `50`) iterations or until the residual, relative
to `||s_j||`, is below `config.nonlinear_solver_tolerance` (default `1e-5` -- see the comment on
that field for why it isn't tighter). It never re-differentiates `g_tilde`; the Jacobian is always
the same `B_hat` factorization, which is what makes hundreds of directions computationally
tractable. Reports per direction: `converged`, `nonfinite_encountered` (left the valid domain),
`final_residual_norm`, `num_iterations`, `aborted`/`abort_reason` (the divergence abort below),
and `branch_change_suspected` (converged, but
`||delta^{NL} - delta^{L}|| / ||delta^{L}|| > 5`, a heuristic flag for "this looks like a
different root," not a proof).

A continuation step can otherwise only fail by exhausting its whole iteration budget, so a step
that has demonstrably left the chord method's basin still pays for all 50 iterations before being
counted as a failure. `config.nonlinear_solver_divergence_abort` (default `True`) stops such a
step early on either of two clauses, both evaluated per continuation step and only while that
step's best relative residual is still above
`config.nonlinear_solver_divergence_guard_factor * config.nonlinear_solver_tolerance`: the
iterate blew past its own best relative residual by more than
`config.nonlinear_solver_divergence_blowup_factor`, or
`config.nonlinear_solver_divergence_stall_window` iterations passed with no new best. An aborted
step is failed on exactly the same control-flow path an exhausted budget takes, so an aborted
solve is *observationally identical* to an exhausted one for both callers -- see "What the
re-solve checks cost" below for the defaults, the measured margins behind them, and the one
metric the abort does move (`domain_failure_fraction`).

Aggregated across **converged** directions only (a non-converged continuation's last iterate is
solver debris, not an "exact" measurement -- the ADS-142 calibration experiment observed such
debris reaching `~1e34` SE when it was still allowed into these statistics; non-convergence
remains fully visible via `root_failure_fraction` and the status logic below):

- `a_nl_by_target[label]`: quantile summary of `a^{NL}_{j,l} = |l @ e_j| / se(l^T eta_hat)` where
  `e_j = delta_j^{NL} - delta_j^{L}`. It measures the same construct as the cheap `a_{j,l}`
  without the one-step-Taylor approximation. Note what ADS-142 measured about both: they rank
  *scenarios* (deployments/designs) by nonlinearity essentially identically, but their
  per-replicate agreement is weak (within-cell Spearman 0.17-0.35 on per-direction maxima), and
  neither predicted per-replicate variance accuracy in that experiment -- so treat `a^{NL}` as
  corroborating `a_{j,l}` at scenario level, not as a per-dataset verdict that "supersedes" it.
- `se_ratios`: `sqrt(lambda_k)` for the generalized eigenvalues of `Cov_j(L delta_j^{NL})` relative
  to `V_L = L @ V_hat @ L^T` (via `se_ratios_from_generalized_eigenvalues`, i.e.
  `scipy.linalg.eigh(nonlinear_cov, V_L)`) -- the nonlinear-to-linear standard-error ratio along
  each identified target direction. `se_ratios_within_tolerance` is judged against
  `se_ratio_null_band`, a **simulated finite-J null band** (`_simulate_perturbation_null_bands`):
  the exact distribution of the min/max ratio under perfect linearity for the observed
  (sign, direction, converged) pattern, at `config.confidence_level`. A fixed band is not usable
  here: at any practical direction count the extreme generalized eigenvalues of a J-draw sample
  covariance carry large sampling noise under perfect linearity (the former fixed `[0.95, 1.05]`
  band failed 100.0% of 6,648 ADS-142 replicates, provably-near-affine cells included).
- `mean_shift_se`: `||V_L^{-1/2} @ mean_j(L @ delta_j^{NL})||` (via `matrix_inv_sqrt`) -- the
  standardized nonlinear mean shift, computed on the sub-ensemble `_mean_shift_ensemble` returns:
  every converged row when directions are drawn unpaired, but only directions that converged with
  **both** signs when they are drawn antithetically (`mean_shift_num_rows` reports how many rows
  that leaves, and dropped incomplete pairs attach a warning). Inside a complete pair the linear
  parts cancel identically, so dropping whole pairs can only remove curvature signal, never
  manufacture it -- which is what keeps this a measurement of curvature rather than an artifact of
  convergence censoring. It is judged against `mean_shift_threshold = max(mean_shift_null_upper,
  config.mean_shift_tolerance_se)`: the simulated finite-draw null upper quantile of
  `||mean_j(rows)||` for the observed converged pattern, floored at the configured tolerance. The
  band is what makes the comparison honest at finite `J` -- with unpaired draws the statistic is
  raw sampling noise of size `sqrt(chi2_dim / J)` (~`0.26` in one dimension at `J = 15`), which
  the former fixed `0.10` comparison read as a failure on a provably affine map; under intact
  antithetic pairing the null band collapses to exactly `0` and
  `config.mean_shift_tolerance_se` is the binding threshold. `mean_shift_gate_evaluable` records
  whether the comparison could be made at all: with unpaired draws there is no cancellation to
  lean on, so an unhealthy solver suppresses the gate (an observed exceedance is reported as a
  warning instead), and the status logic turns that into `indeterminate` rather than a pass.
- `quantile_shifts_se[label]`: the 2.5%/97.5% quantiles of the nonlinear draws minus the linear
  draws, in SE units, for each scalar contrast -- also exactly `0` per direction for a linear map.
- `root_failure_fraction`/`branch_change_fraction`/`domain_failure_fraction` and their
  Clopper-Pearson upper bounds (`root_failure_upper_bound`, etc., at `config.confidence_level`,
  default `0.95`), plus `num_converged_trials`.
- Ensemble accounting: `num_trials` is the number of directions/signs **actually executed**,
  `num_planned_trials` is what the config asked for, and `early_stopped`/`early_stop_reason` say
  whether the two differ because the sequential early stop fired. `num_divergence_aborted_trials`
  counts how many of the non-converged solves were stopped by the divergence abort rather than by
  exhausting their iteration budget. All four are described under "What the re-solve checks cost"
  below; every fraction above is computed against `num_trials`, i.e. against what was run.

**Status logic** (precedence matters). First, **confirmed fragility fails outright**: when the
failure fraction's one-sided Clopper-Pearson *lower* bound (`clopper_pearson_lower_bound`, at
`config.confidence_level`) exceeds `config.bad_direction_probability_target`, the equation
measurably cannot be re-solved at perturbation scale -- counted over ALL executed trials, so no
optimistic-subset caveat applies, and no ensemble statistic is needed. This is the rung that
separates "could not evaluate" from "measured a collapse" (0 of 198 converging bounds the rate
above ~0.985). Below that rung, solver health is judged on the **observed** failure fraction vs.
the same target -- not the Clopper-Pearson *upper* bound, which exceeds any small target at
practical trial counts even with zero observed failures: `indeterminate` if fewer than 3 solves
converged. Otherwise `failed` if a distortion gate fires
on the converged subset -- `se_ratios` **above** the null band, `mean_shift_se` above its banded
threshold `max(mean_shift_null_upper, config.mean_shift_tolerance_se)` (the tolerance defaults to
`0.10` and acts as the floor, not as the whole comparison), or any quantile shift beyond
`config.quantile_shift_tolerance_se` (default `0.10`); a below-band `se_ratios` also fails, but
only when the solver is healthy, because convergence-censoring itself compresses the converged
subset (the excluded directions are the hardest ones), making below-band readings on a censored
ensemble an artifact. A `failed` from the converged subset stands even under heavy censoring --
the same censoring argument makes it *optimistically* selected, so a failure on it is
trustworthy, while a pass on it is not: gates passing with an unhealthy solver reads
`indeterminate`. An **unevaluable** gate is likewise never a pass: if the `se_ratios` comparison
could not be made for every target (a singular `V_L` makes the generalized eigenproblem fail, or
the converged pattern is too thin to simulate a band) or the mean-shift gate was suppressed or
unavailable, the status is `indeterminate` with a warning naming which gate(s) went unevaluated.
`passed` otherwise.

### 3b. Frozen-score multiplier bootstrap -- `check_multiplier_bootstrap`

`"auto"` by default as of 2026-09-02 (previously `"off"`); `"off"` disables it, `"always"` runs
it unconditionally, and `"auto"` uses the cheap `a_{j,l}` check as a screen -- the bootstrap runs only when
`check_local_nonlinearity`'s headline max exceeds `config.bootstrap_screen_a_jl_threshold`
(default `0.05`, an ADS-142-calibrated screen: against the exact check's `a^NL > 0.10` key on
the borderline cells it missed ~11% of exceedances while triggering on ~54% of replicates) or
that check did not `pass` outright.

The default moved to `"auto"` because `"off"` made certification unreachable for any run above
the screen: `_derive_verdict` makes the bootstrap the verdict layer once the screen trips, so a
screened-in run with no bootstrap result is UNCERTIFIABLE by construction however healthy it is.
Cost is no longer the objection it was -- the hours-scale figures predate the `g_tilde` closure
fix, and 100 paired draws now measure `12.7-21.7 s` at `n=100`/`T=50` (see 3c) -- and `"auto"`
costs nothing at all on a run whose screen stays quiet. The caveat in `docs/adr/0002` still
stands: the batched path's verdict-equivalence is measured on clean fixtures only, and a fragile
ensemble with solves near the convergence boundary is unmeasured.

This is the suite's answer to "certify this one dataset without a simulator." It draws
`config.num_bootstrap_draws` (default `100`, paired `+/-`) multiplier vectors `nu_b` with iid
mean-0/variance-1 entries (`config.bootstrap_multiplier_distribution`: `"rademacher"` default,
`"mammen"` to match third moments, `"gaussian"` to match the direction sampler's own multiplier
distribution -- the same law, not the same draws),
forms the frozen-score bootstrap perturbation `s_b = (1/n) sum_i nu_{b,i} g_i(eta_hat)` from
`per_subject_stacks`, and re-solves `g_tilde(eta_hat + delta) - g0 = s_b` with the same
continuation/chord-Newton machinery as the exact check. This is the generalized (weighted)
bootstrap for Z-estimators -- Chatterjee & Bose (2005), Praestgaard-Wellner multipliers -- with
the multiplier-weighted score frozen at the observed root (a second-order approximation of the
same size as the bootstrap's own error, and what lets it run from `per_subject_stacks` alone).

Two things make it more than a rerun of the sandwich: the re-solve is **nonlinear** (no
linearization to trust), and the draw is **empirical** (higher moments come from the realized
per-subject contributions). Policy adaptivity needs no replay for the same reason the adjusted
sandwich needs none: the algorithm's updates are estimating equations *inside* the stacked
system, with action probabilities reconstructed as functions of `eta`, so a multiplier
perturbation propagates through the eta-equations into every subject's contribution exactly
along the path the sandwich's cross-derivative bread terms linearize. On misspecification of
the stacked representation itself, the division of labor is: **value-level** fidelity is
testable and tested by the input checks (each recorded `eta_k` must root the claimed update
equations on the realized data -- the sum-to-zero check per update -- and the claimed policy
mapping must reproduce the recorded action probabilities); the residual blind spot shared by
the sandwich and this bootstrap is **derivative-level** fidelity -- an update rule that agrees
with the model at the realized trajectory but responds differently to perturbed data (e.g. a
clip or projection that was inactive on the realized path) passes the value checks while the
bread's cross-terms are computed under the wrong response model. This check isolates
linearization/normal-approximation error *given* that response model.

Reported per target: `bootstrap_se_by_target` (SD of `l @ delta^{NL}_b` over converged draws),
`sandwich_se_by_target`, and their ratio `se_ratio_by_target`, judged against
`se_ratio_null_band` -- the same simulated finite-draw null band construction as in the exact
check (per-target SD ratios, Bonferroni-adjusted across targets), so the verdict is
self-calibrating: no engineering tolerance, no simulator. The band comparison is made **per
target, over the finite ratios**: a target with no identified variance (`se_l == 0`, hence a
`NaN` ratio) is listed in `se_ratio_band_unevaluable_targets` and disables the *pass*, but it no
longer disables the comparison for its healthy siblings -- an above-band ratio on an identified
target still fires. `mean_shift_se` is the same statistic, on the same complete-pairs
sub-ensemble and against the same banded-with-floor threshold, as in the exact check above.

Ensemble accounting is the same as the exact check's -- `num_trials` (executed) against
`num_planned_trials`, `early_stopped`/`early_stop_reason`, `num_divergence_aborted_trials` --
plus two of its own: `num_draws_executed` (distinct multiplier draws actually re-solved, against
the planned `num_draws`) and `resolve_batch_width` (`0` when the serial reference solver produced
the whole ensemble, otherwise the pinned lockstep batch width the batched driver ran at -- if it
fell back to the serial solver partway, this still names that width and a warning says so; see
section 3c).
`DiagnosticReport.monte_carlo_counts` carries `num_bootstrap_draws_executed` next to the planned
`num_bootstrap_draws` for the same reason, whenever this check ran at all.

**Status logic:** same confirmed-fragility rung first as the exact check's -- `failed` when the
failure fraction's Clopper-Pearson lower bound clears `config.bad_direction_probability_target`
(fragility under resampling-scale perturbations statistically confirmed on all executed
re-solves). Then `indeterminate` if fewer than 3 re-solves converged. `failed` if any evaluable
target's ratio exceeds the band's upper edge (the bootstrap distribution is wider than the
sandwich claims -- the **anticonservative** direction) or `mean_shift_se` exceeds its banded
threshold `max(mean_shift_null_upper, config.mean_shift_tolerance_se)` (systematic
curvature-induced displacement of the resampled roots). Else `indeterminate` if the observed
re-solve failure fraction exceeds the target without the lower bound confirming it -- fragility
suspected but not measured, and the converged subset is optimistically selected. Else `indeterminate` if a gate could not be fully evaluated -- any target in
`se_ratio_band_unevaluable_targets`, an unavailable null band, or a suppressed/unavailable
mean-shift gate -- because with no SE comparison performed for some target there is nothing left
to have passed (a below-band ratio elsewhere is still reported as a warning in that case). Else
`warning` if any ratio falls below the band (sandwich SE overstated -- the **conservative**
blow-up direction ADS-142 found to be the common failure mode). `passed` otherwise.

### 3c. What the re-solve checks cost, and the knobs that change it

Both re-solve checks have the same cost model: (trials) x (chord-Newton iterations per solve) x
(one `g_tilde` evaluation plus one triangular solve against the already-factored bread). The
`g_tilde` evaluation is essentially the whole of it -- 94-97% of the bootstrap's wall time at the
scale below -- so everything here is about how many `g_tilde` evaluations happen and how they are
dispatched.

**Read the numbers below with two caveats.** They were measured **locally** (Apple-silicon laptop
CPU, JAX CPU backend, float32 -- this repository deliberately does not enable x64), not on the
FASRC cluster, so absolute seconds could move by roughly a factor of two; the *ratios* are the
transferable part. And **any hours-scale figure you may remember for this check is stale**: it
predates the fix that builds `post_deployment_analysis`'s diagnostics `g_tilde` closure once and
jits it instead of rebuilding the whole `O(n*T)` structural precompute on every call. Measured
against a faithful in-process reconstruction of the pre-fix closure at `n=100`/`T=50`, one
evaluation cost `797 ms` then and `1.4-2.6 ms` now, so the same 8,027 evaluations a 100-draw
bootstrap performs project to about `1.8 h` under the old closure. Nothing else is needed to
explain the old runtimes.

Measured at the pipeline's own `n=100`, `T=50`, `eta_dim=200` clean fixture, with the bootstrap
forced on and everything else at its default:

| What | Measured |
| --- | --- |
| `check_multiplier_bootstrap`, 100 paired draws (200 trials, 8,027 `g_tilde` evaluations) | `12.7 s` and `21.7 s` in two independent local harnesses -- same evaluation count, so the spread is machine load, not different work |
| `check_multiplier_bootstrap`, 25 paired draws (50 trials, 1,988 evaluations) | `3.5 s` / `5.4 s` in the same two harnesses |
| `run_diagnostic_suite` with the bootstrap off, same fixture | `6.6 s` |
| Chord-Newton iterations per converged solve, clean data | median `39`, range `31-51` |

Synthetically fragile ensembles at the same scale cost *more* than clean ones, not less, and the
peak is at partial fragility: 8,027 evaluations at a 0% failure fraction, ~24,700 at 32%, ~21,600
at 97.5%, ~12,200 at 100%. A cell where everything fails is the cheap one -- a failing solve exits
the continuation at the first step that exhausts its budget, so it does not pay the
`continuation_steps x nonlinear_solver_max_iterations` ceiling. **Cluster re-measurement of the
fragile cells on the current code is still pending** (`docs/adr/0002`).

**The knobs**, all `DiagnosticConfig` fields, all with plain immutable defaults so a config
pickled before they existed still resolves them:

| Field | Default | What it trades off |
| --- | --- | --- |
| `nonlinear_solver_divergence_abort` | `True` | Stops a continuation step that has visibly left the chord method's basin instead of paying its whole iteration budget. Saves nothing on healthy solves and 20-66% of a check's evaluations (max 96% observed) where nearly every solve fails; the cost is that the two clauses below are *measured*, not proved, so a false abort would silently turn a converged draw into a root failure. Set `False` to restore exhaust-the-budget behavior exactly. |
| `nonlinear_solver_divergence_blowup_factor` | `1e4` | Abort when the relative residual exceeds this multiple of the best seen so far in the same continuation step. Worst value ever observed inside a step that then *converged*: `1.9` on this repository's fixtures (33,351 converging steps) and `36.7` on an independent synthetic sweep (118,563 steps) -- a ~270x margin on the worse sample. |
| `nonlinear_solver_divergence_stall_window` | `40` | Abort after this many iterations with no new best residual. Longest such run inside a step that then converged: `4` here, `20` on the synthetic sweep -- a 2x margin. Lowering it toward 25 buys a few percent and cuts the margin to 1.25x; `0` disables this clause and keeps only the blow-up one. |
| `nonlinear_solver_divergence_guard_factor` | `100.0` | Neither clause is evaluated while the step's best relative residual is already inside this multiple of `nonlinear_solver_tolerance`, where float32 noise makes a legitimate bounce look like a stall. |
| `perturbation_early_stop` | `"starvation"` | `"starvation"` abandons the remaining trials once the still-attainable converged count has fallen below the 3 every ensemble statistic needs; `"off"` always runs the full plan. Provably status-preserving, and provably worth almost nothing (see below). |
| `batched_bootstrap_resolves` | `"off"` | `"auto"`/`"always"` re-solve the bootstrap ensemble in lockstep waves, one vmap'd `g_tilde` call per generation instead of one per row. Measured ~1.5-1.75x end to end at 100 draws (~2.6x excluding a one-time ~5-6 s XLA compile), verdict-equivalent but **not bit-identical** -- which is why it is opt-in; see below. |
| `bootstrap_batch_width` | `50` | Trials per wave and the pinned vmap width (deliberately one number: they are the same buffer). Throughput per row is nearly flat from 25 to 200 here, so choose this for memory -- every batch row carries its own copy of the full stacked intermediate. |
| `bootstrap_batch_min_rows` | `4000` | Break-even guard for `"auto"`, in projected live row-evaluations (`trials x continuation_steps x 4`). Batching buys one XLA compile that only repays above a few thousand rows. The guard reads the plan's shape and nothing else, deliberately: deciding from a *timed* `g_tilde` call would make the arithmetic depend on machine load, and two runs of the same seed could then report different last digits. |

**Batched re-solves are verdict-equivalent, not bit-identical.** A draw's chord-Newton trajectory
depends only on its own iterate, its own continuation index, and the frozen bread factorization
every draw already shares, so the lockstep driver changes only *which rows share a kernel launch*.
What it cannot preserve is the last few bits: XLA selects different float32 kernels per batch
width (measured `9.5e-7` absolute on a `max|g|` of `3.6`, ~2.2 float32 ULPs), which propagates to
~`1e-7` relative on the SE ratios and ~`1e-5` on `mean_shift_se`, a difference of means and so
cancellation-dominated. End to end at 100 draws on the clean fixture this left `status`, the
warnings, and all 200 per-trial convergence flags identical -- the same trials converged, not
merely the same number of them. A solve sitting simultaneously at the convergence tolerance *and*
at its last permitted iteration could in principle flip; nothing in the measured corpora came near
that (the closest non-accepting final residual was 1.4x the tolerance). "Measured, not proved" is
the wrong default for a calibration campaign whose answer key is exactly which runs fail, hence
`"off"`. `metrics["resolve_batch_width"]` records which path actually ran, because reproducing a
run's last digits means reproducing that number too.

Only the bootstrap has the batched driver wired in; `check_exact_nonlinear_perturbations` always
re-solves serially. At its default 15 paired directions it projects ~1,200 live rows, which is on
the wrong side of the break-even above -- batching it in isolation would make it slower, not
faster. And the batched path is never load-bearing for whether a result exists at all: not every
`g_tilde` survives `jit`+`vmap` (host-side control flow on the argument, a callback, a shape the
tracer refuses), so a failure to compile at the pinned width logs a warning and falls back to the
serial solver with `resolve_batch_width` reported as `0` -- a bootstrap that did not run would be
a `not_certified` verdict, i.e. a materially different report, which is not an acceptable outcome
for a performance knob. A failure *after* some waves have already been re-solved cannot be undone
that way: the remaining waves fall back, `resolve_batch_width` still names the width that ran, and
the check attaches a warning that the ensemble mixes the two arithmetics -- they agree well inside
every tolerance the check compares against, but such a run reproduces against neither path alone.

**A truncated ensemble is deliberate, and it is not a failure.** When `perturbation_early_stop`
fires you will see `num_trials < num_planned_trials`, `early_stopped: True`,
`early_stop_reason: "max_attainable_converged_below_minimum"`, and a warning naming the
truncation; the bootstrap additionally reports `num_draws_executed < num_draws`, and
`DiagnosticReport.monte_carlo_counts` carries `num_bootstrap_draws_executed` next to the planned
`num_bootstrap_draws`. Every fraction and Clopper-Pearson bound in the check is computed against
what actually ran, so they get *wider*, never narrower. The stop is status-preserving by
construction: it fires only when `converged_so_far + trials_remaining < 3`, at which point no
completion of the ensemble can reach the gate or pass rungs -- the status is decided between
`failed` (confirmed fragility: the failure rate's lower bound clears the target) and
`indeterminate`, and for any plan of 9+ trials that comparison lands the same way for the
truncated ensemble and every completion of it, so the reported `status` -- and therefore
`classification` and `verdict`, which are pure functions of the check statuses -- is what the
full plan would have produced (a toy plan below ~9 trials can at worst soften a would-be
`failed` to `indeterminate`, both in the do-not-report family). Because it requires at most 2
trials to remain, it can never skip more than 2, i.e. ~1% of a 100-paired-draw plan and 4% of a
25-draw one; it exists for the guarantee and the honest accounting, not for the speed.

The two optimizations do not compose, and the direction of that is worth knowing before you go
looking for a truncation that never happens: batched re-solves can only test the predicate at a
wave boundary, where the remaining count is a multiple of the wave width, so at any width above
2 "at most 2 trials remain" is unreachable and the stop simply never fires. With
`batched_bootstrap_resolves` on you should expect `num_trials == num_planned_trials` always. That
costs at most the 2 trials above, and the accounting stays honest either way -- the metrics report
what actually ran, which under batching is the whole plan.

The tempting stronger rule -- "stop once fragility is *confirmed*" -- is **not** implemented,
because confirmation at a truncation point does not survive every completion: the lower bound
falls as remaining trials converge (`lb(x, n)` shrinks in `n` at fixed `x`), so a rate confirmed
above the target mid-ensemble can end the full plan below it. Mere unhealthiness (observed
fraction above the target, unconfirmed) is weaker still: it only confines the status to
`{failed, indeterminate}`, and the distortion-gate `failed` rung deliberately sits above the
unhealthy rung in both checks so a distortion measured on the optimistically-censored converged
subset still counts. In the ADS-142 influence cell every one of 50 cluster runs had failure
fractions of `0.12-0.86` -- rates whose lower bounds clear the default target, so under the
current rungs they report `failed` as confirmed fragility (they previously reported the same
`failed` from their converged subsets' distortion gates; the verdict is `invalid` either way).

**One metric the divergence abort does move.** A solve that would have gone non-finite at a *later*
iteration is now recorded as a root failure rather than a domain failure, so
`domain_failure_fraction`/`domain_failure_upper_bound` become lower bounds. No status rung reads
either, and `num_divergence_aborted_trials` sits next to them so the shift is attributable rather
than silent -- but any analysis that pools those columns across this change will see a
discontinuity. Measured on the diagnostic fixtures, the effect is confined to polynomial maps
(whose "domain failure" was really float64 overflow of an already-divergent iterate two iterations
after the blow-up clause fires); it is untouched on bounded maps, where failures stall rather than
blow up.

### 4. Jacobian drift -- `check_jacobian_drift` (`rho_j`)

Also gated on `config.compute_exact_nonlinear_roots` (it shares the same "this is expensive,
opt-in" framing). Uses a *separate, smaller* sample of `min(config.drift_num_directions, config.
num_directions)` directions (`config.drift_num_directions` defaults to `3` -- re-differentiating
`g_tilde` is far more expensive than evaluating it, so this stays small even when `num_exact_
directions` is cranked up for the check above). For each direction and each `t` in `config.
drift_path_samples` (default `(0.0, 0.5, 1.0)`), computes the true Jacobian `D g_tilde(eta_hat + t
* delta_j)` via `helper_functions.compute_row_chunked_jacobian` and the operator norm of
`B_hat^{-1} (D g_tilde(...) - B_hat)` (via a stable solve of a matrix right-hand side, never an
explicit inverse). `rho_by_direction[j]` is the max of that operator norm over the sampled `t`;
`rho_max` is the max over directions.

These are reverse-mode Jacobians of the *whole* stacked system -- the most backward-pass-expensive
thing in the suite, and the pass that is known to exhaust memory at real study scale -- so their
peak memory is bounded by the same package-wide policy as the sandwich's own backward pass:
`helper_functions.resolve_jacobian_row_chunk_size(config.jacobian_row_chunk_size, out_dim)`,
resolved once before the direction loop (so the auto heuristic logs its decision once rather than
nine times) and passed straight through as the chunk size. `config.jacobian_row_chunk_size`
defaults to `None` (auto: unchunked at `out_dim <= 512`, above which the chunk size targets
`chunk_size * out_dim <= 65536` and is capped at 64 -- see the empirical calibration in
`lifejacket/constants.py`); `0` forces the plain
unchunked `jax.jacrev`, and a positive int caps how many output rows are pulled back at once.
Chunking changes nothing numerically.

**This is explicitly a sampled path maximum, not a certified supremum** -- the module always
attaches a warning saying so, and the status is `warning` if `rho_max >= 1`, `passed` otherwise.
A verified contraction bound `||delta^{NL} - delta^L|| / ||delta^L|| <= q_j / (1 - rho_j)` is
mathematically available when `rho_j < 1` is a genuine neighborhood bound, but this package has no
interval-arithmetic/Lipschitz machinery to certify that `rho_j` bounds the *whole* neighborhood
rather than just the sampled points, so no such certified bound is ever reported here -- only the
sampled `rho_j` values themselves.

### 5. Bread and numerical stability -- `check_bread_stability`

Always runs (unless the root check hard-failed). Purely numerical/linear-algebraic, no calls to
`g_tilde`:

- `diagonal_block_diagnostics`: for each of the `(B_hat.shape[0] - theta_dim) // beta_dim`
  per-update beta blocks, its singular values and condition number.
- `theta_block_condition_number`/`theta_block_singular_values`: same, for the bottom-right
  theta-theta block.
- `off_diagonal_beta_to_theta_norm`/`off_diagonal_theta_to_beta_norm`: Frobenius norms of the two
  off-diagonal coupling blocks.
- `full_bread_condition_number`: condition number of the whole `B_hat`.
- `target_covariance_eigenvalues`/`target_covariance_rank_estimate`/`target_covariance_dim`:
  eigenvalues of the **target** covariance `L @ V_hat @ L^T` (falling back to `V_hat`'s theta
  block when the caller supplies no selector; `run_diagnostic_suite` always supplies one), how
  many exceed `config.rank_tolerance * max_eigenvalue` (default `1e-8`), and that matrix's
  dimension. Emphatically not the full joint `V_hat`: the joint sandwich's rank is generically at
  least `theta_dim` even when the theta target block is exactly singular, because the healthy beta
  blocks supply that rank on their own -- judging identification on it made the gate below
  unfireable at any real study scale (`beta_total ~4000` vs. `theta_dim ~5`) and reported
  beta-block eigenvalues, in beta's units, under a "target covariance" name.
- `numerical_sensitivity_max_relative_se_change`: `B_hat` is perturbed by a relative
  `config.bread_perturbation_relative_scale` (default `1e-6` -- chosen to sit well above float32
  machine epsilon, `~1.2e-7`, so the perturbation is not itself silently rounded away) times
  `||B_hat|| * I`, the theta-only SEs are recomputed via the same QR technique used elsewhere
  (`_theta_only_variance_diag_qr`, no explicit inverse), and the maximum relative change in any
  theta SE is reported. This measures numerical fragility specifically, as distinct from
  statistical identification.

**No universal condition-number threshold is hard-coded anywhere in this function** -- every
number above is reported as a metric, and only two things drive the status: `failed` if
`target_covariance_rank_estimate < target_covariance_dim` (unidentified reported components),
and otherwise `indeterminate` if
`numerical_sensitivity_max_relative_se_change > config.se_distortion_tolerance`
(default `0.05`) -- i.e. the reported SEs are themselves numerically fragile at a scale well below
real precision loss. `passed` otherwise.

**Why the two gates land on different tiers:** rank deficiency `failed` because it is the
suite's single best-measured predictor of catastrophe -- it identified 76/76 of the
zero-width-interval collapses in the ADS-142 undercoverage hunt, and `_derive_verdict` was
already treating it as INVALID-grade evidence (the row now says the same thing the verdict
does). The sensitivity gate stays capped at `indeterminate` because the same experiments
measured the opposite for it: a second, higher sensitivity threshold does NOT predict actual
per-replicate variance failure -- in the cell where variance blow-ups occurred,
`sensitivity > 0.20` caught only ~5% of >2x-inflated replicates (vs. `n_eff`'s ~59% at its
default floor), and the blow-ups it does catch are *conservative* (empirical coverage never fell
below ~0.94 anywhere in the 9,883-run grid). `indeterminate` is the calibrated ceiling for the
sensitivity gate, not a placeholder.

### 6. Influence concentration -- `check_influence_concentration`

Runs when `config.compute_influence_and_overlap_checks` (default `True`). For each contrast `l`,
solves `w = B_hat^{-T} l` once (`solve_with_bread_transpose`, no explicit inverse) and forms the
per-subject linearized influence `xi_i = -(per_subject_stacks @ w)_i` for every subject `i` in one
vectorized dot product. From `xi`, reports per target label in `by_target[label]`:

- `p_max`: the largest single-subject share of `sum(xi_i^2)`;
- `n_eff = 1 / sum(p_i^2)` (bounded below by `1`, by construction -- see the code comment on why
  the warning threshold below is `max(config.influence_n_eff_min_floor,
  config.influence_n_eff_min_fraction * n)` rather than a pure fraction of `n`, which can never
  fire for small `n`);
- `third_moment_concentration = sum(|xi_i|^3) / sum(xi_i^2)^1.5`;
- `top_influential_subjects`: the top-5 subjects by `|xi_i|`, each with its `xi` value and
  variance share, and its `subject_id` (from the `subject_ids` passed to `run_diagnostic_suite`)
  when `config.report_subject_identifiers` (default `True`).

**Status logic:** `warning` for any target with `n_eff < max(config.influence_n_eff_min_floor,
config.influence_n_eff_min_fraction * n)` (defaults `2.0`/`0.1`) or `p_max >
config.influence_p_max_tolerance` (default `0.5`). This check never hard-fails -- concentrated
influence is evidence against the CLT premise, not proof that the estimator is wrong.

These defaults now carry real calibration data (the ADS-142 experiment,
`docs/adr/0002-diagnostic-threshold-calibration-plan.md`): `n_eff` was the strongest
per-replicate predictor of variance blow-up the suite has (within-cell Spearman `-0.56` against
per-replicate SE inflation in the influence-stressed cell) -- at the default floor (`0.1 * n`,
i.e. `n_eff < 10` at `n = 100`) it flagged >2x-inflated variance estimates with precision 0.73
and recall 0.59 across 993 replicates. `p_max` at its `0.5` default fired on ~2% of replicates
with recall 0.04 -- essentially inert, and largely redundant with `n_eff` when tightened; treat
its default as a last-resort tripwire rather than a calibrated gate. The blow-ups these
thresholds catch were uniformly *conservative* in that experiment (over-wide CIs, coverage
intact), which is why `warning` rather than a gating status remains the calibrated answer.

**Optional leave-one-out sensitivity** (`config.compute_leave_one_out_sensitivity`, default
`False`): for the union of the top `config.leave_one_out_top_k` (default `3`) most influential
subjects across all targets, `leave_one_out_theta_sensitivity` reports a **one-step Newton**
theta shift implied by excluding that subject, holding every `beta_k` fixed at its observed value
(this deliberately never replays the adaptive policy -- deleting a subject does not tell you what
the policy would have done without them, so this is a sensitivity analysis, not a valid
bootstrap). It uses the closed-form leave-one-out average
`(n * mean(stacks) - stack_i) / (n - 1)`, restricted to the theta block, and a single chord-Newton
step against the theta-theta block of `B_hat` -- not an iterated re-fit, since that would require
re-evaluating every other subject's row at a new theta, which needs the full estimator machinery.

### 7. Exploration and importance weights -- `check_exploration_and_weights`

Runs when `analysis_df`/`active_col_name`/`calendar_t_col_name`/`action_prob_col_name` are
supplied (they come from `post_deployment_analysis.analyze_dataset`'s own arguments when
`run_diagnostics=True`; a caller driving `run_diagnostic_suite` with only a toy `g_tilde` simply
skips this check).

- `action_prob_min_by_time`/`action_prob_max_by_time`/`action_prob_global_min`/
  `action_prob_global_max`: straight from the recorded `analysis_df`, via pandas groupby --
  no autodiff needed for this part.
- **Hard fails** if any recorded probability is nonfinite, or falls outside the open interval
  `(0, 1)`. This is where positivity/overlap is actually enforced -- the legacy
  `input_checks.require_action_probabilities_in_range_0_to_1` was a deliberate no-op for backward
  compatibility (some legitimate near-deterministic policies produce recorded probabilities of
  exactly `0.0`/`1.0` after float rounding) and has since been removed from `input_checks.py`
  entirely now that this check does the job properly.
- If `config.exploration_floor`/`config.exploration_ceiling` are supplied (they are not
  recoverable from the data alone -- the caller must know the deployment's actual design bounds),
  `fraction_at_or_near_floor`/`_ceiling` are reported, and any violation is a **hard fail**.
- **Importance weights under perturbation**: `run_diagnostic_suite` draws up to `min(config.
  num_directions, 5)` sandwich-scale perturbations of the beta block of `eta_hat` and, for each,
  calls `compute_importance_weights_under_beta` (built on `helper_functions.
  get_radon_nikodym_weight`) to get each subject's cumulative importance-weight trajectory under
  that perturbed beta -- evaluated away from `eta_hat`, where these diagnostic weights would be
  trivially `1`. From the final cumulative weight per subject per direction,
  `ess_over_n_by_direction` reports the quantile summary of normalized ESS
  `(sum W_i)^2 / (n * sum W_i^2)` across directions, and `max_cumulative_weight_by_direction`
  the corresponding max-weight summary. A nonfinite weight under perturbation is a **hard fail**.
- **Policy-score-derivative norms**: if the caller supplies `pi_and_weight_gradients_by_calendar_t`
  (e.g. from `calculate_derivatives.calculate_pi_and_weight_gradients`),
  `policy_score_gradient_norm_summary` reports the quantile summary of `||d(pi)/d(beta)||` across
  all subjects and decision times.

Everything else in this check is a reported metric, not a hard requirement, since the "right"
threshold for weight concentration/ESS is deployment-specific and should be simulator-calibrated.

## Why `r_j` and raw `q_j` have no universal threshold

`r_j = ||R_j|| / ||B_hat @ delta_j||` and `q_j = ||c_j|| / ||delta_j||` are ratios taken in
*equation space* and *raw parameter space* respectively. Both depend on how the estimating
equation and the parameters happen to be scaled -- rescale the equation by any invertible `D`
and `r_j` changes even though the actual estimator and its sandwich covariance are unchanged;
reparametrize the parameters by any invertible `P` and `q_j` changes the same way (see
`tests/unit_tests/test_diagnostics.py` for both invariances verified directly). Neither is
wrong to report, but neither can carry a universal pass/fail cutoff, because "0.1" means
something different depending on units that have nothing to do with the statistics.

## Why the target-standardized `a_{j,l}` is the main cheap diagnostic

`a_{j,l} = |l^T c_j| / se(l^T eta_hat)` measures the first nonlinear correction in units of the
*actual reported* standard error for the *actual reported* contrast. It is invariant to both of
the rescalings above (also verified directly in the test suite), because both the numerator and
`se(l^T eta_hat)` transform the same way and the transformation cancels. That is what makes
`0.10 standard errors` a meaningful, portable engineering tolerance in a way that a raw `r_j` or
`q_j` cutoff cannot be -- though it is still an engineering tolerance, not a value derived from
a theorem, unless it has been calibrated against a simulator for this specific deployment family.

## Why the exact nonlinear perturbation check is stronger -- and what "stronger" does not mean

`check_local_nonlinearity` uses one Newton-style correction (`c_j`) as an estimate of how far the
true nonlinear root of the perturbed equation is from the linear guess. `check_exact_nonlinear_
perturbations` instead solves the perturbed equation via continuation, so it does not rely on a
single first-order correction being adequate -- it also detects genuinely different behavior
(non-convergence, apparent branch changes, departures from the parameter domain) that a one-step
correction cannot see by construction. It costs one root-solve per direction (via `continuation_
steps` chord-Newton steps against a fixed bread factorization -- never a re-differentiated
Jacobian, which is what keeps hundreds of directions tractable), so `compute_exact_nonlinear_
roots` defaults to `False` and is meant to be turned on when the cheap diagnostic is ambiguous or
when a rigorous bad-direction-probability bound is wanted (`num_exact_directions` should be at
least ~300 to bound that probability below ~1% with zero observed failures, per the
Clopper-Pearson calculation in `lifejacket.helper_functions.clopper_pearson_upper_bound`).

"Stronger" is a statement about *measurement* (no Taylor step), not about *verdicts*. The
ADS-142 experiment measured both checks against ~1,660-replicate empirical truth per scenario
(6,648 completed Track A replicates across four cells; see `docs/adr/0002`) and found: the two agree closely on scenario-level severity ranking; both are weak
per-replicate predictors of actual variance accuracy; and in severely nonlinear regimes the
continuation solver itself stops converging (median replicate in the hardest cell: ~all
directions failed), at which point this check's honest output is its failure fraction, not its
distortion statistics. When a per-dataset verdict about the sandwich's SEs is the goal, prefer
`check_multiplier_bootstrap` (section 3b), which solves the same kind of nonlinear system but
compares against a self-calibrating null band -- and use `a_{j,l}`/this check for what they
measure well: locating *where* (which deployments, which designs) nonlinearity lives.

## Why a passing local diagnostic does not establish the CLT step

Every check above except `check_influence_concentration` and simulator calibration is a
*local* statement about the estimating equation near `eta_hat`: it says the linearization is
adequate, or that the bread is well-conditioned, at the scale of the observed perturbations. None
of that implies that `sqrt(n)(eta_hat - eta*)` is actually close to normal -- that additionally
requires the estimator's fluctuation to be built from many relatively small contributions
(the premise `check_influence_concentration` checks, imperfectly: a low `p_max`/high `n_eff` is
necessary-flavored evidence, not a formal Berry-Esseen bound), and it requires the local
linearization to hold not just at `eta_hat` but in a neighborhood large enough to contain the
estimator's actual sampling fluctuation. `run_diagnostic_suite` therefore returns at most
`locally_supported`, never `supported`, on its own.

## Why diagnostic failure does not validate the standard sandwich as a fallback

A failed or indeterminate diagnostic means the *adjusted* sandwich lacks support here. It says
nothing about the *classical* sandwich, which relies on a different (and, under a genuinely
adaptive/pooling design, generally false) assumption that the deployment's policy did not depend
on past data. Silently reverting to the classical sandwich on diagnostic failure would trade one
unverified claim for a claim this package has no reason to think is better-founded. This is why
`run_diagnostic_suite` only ever reports a classification for the
*adjusted* analysis, and why the classification vocabulary is
`locally_supported` / `failed` / `indeterminate` rather than a "use this sandwich instead" flag.

## Classification

- `locally_supported`: the observed-data checks pass, but no end-to-end simulator calibration is
  available.
- `failed`: a hard prerequisite (an input-check result, root/implementation, out-of-range
  probabilities, non-finite values) or a measured collapse or distortion -- a rank-deficient
  target covariance (unidentified reported components), total domain censoring of the
  local-nonlinearity probes (`g_tilde` nonfinite at every sampling-scale probe), a re-solve
  failure rate statistically confirmed above `bad_direction_probability_target` (either
  re-solve check), SE-ratio/mean-shift/quantile-shift beyond its null band or tolerance in the
  exact nonlinear check, or bootstrap SEs above their null band (sandwich SE understated) in
  `check_multiplier_bootstrap`.
- `indeterminate`: numerically fragile SEs, unstable solves, insufficient perturbation
  directions, a failure rate above target but below statistical confirmation, a distortion gate
  that could not be evaluated at all, severe (but not total) domain censoring of the
  local-nonlinearity probes, or inadequate simulator coverage prevent a conclusion either way.

## Verdict -- the decision-level summary (`DiagnosticReport.verdict`)

`classification`'s vocabulary is preserved for backward compatibility, and its deliberate
WARNING-blindness (section 4.4 of the tutorial) means an automated consumer reading only
`classification` cannot see the suite's most common non-clean finding -- calibrated
*conservatism*. `report.verdict` is the additive fix: a pure function of the check results
(`_derive_verdict`) that answers "can I report the adjusted sandwich variance?" in
four values, each carrying the
operating characteristics the ADS-142 experiments measured for it (see the
`DiagnosticVerdicts` docstring in `constants.py` for the full statement):

- `invalid` -- a hard/measured failure, or a rank-deficient target covariance (pulled up from
  `indeterminate` because that condition identified 76/76 zero-width-CI collapses in the
  undercoverage hunt). Do not report.
- `not_certified` (renamed from `uncertifiable` on 2026-09-02: the verdict means "not
  established yet", not "impossible to establish" -- its most common cause is fixed by
  re-running with the multiplier bootstrap on) -- something unresolved: re-solve fragility, censored ensembles, unevaluable
  gates, or the `a_{j,l}` screen called for the bootstrap and it did not run. Empirically,
  every genuinely miscalibrated design landed here or in `invalid` rather than falsely
  certifying. Do not report as validated.
- `conservative` -- clean except that a calibrated conservatism signal fired (bootstrap
  below-band, or the influence `n_eff` floor): direction trustworthy, width likely inflated.
  The ADR 0003 percentile refit bootstrap is the interval-level remedy.
- `certified` -- everything gated passed. `verdict_basis` records how: `"bootstrap"` (the
  SE comparison ran and passed) or `"screen"` (a quiet run that never called for it).

Uncalibrated warnings (radius-scaling exponents, `jacobian_drift`'s `rho`, exploration leading
indicators) are reported but never move the verdict -- the experiments gave them no operating
characteristics to gate on.

## The end-of-run summary, and what "flagged" does to the job

When `run_diagnostics=True`, the analysis ends with a printed **diagnostic summary**: one
status row per input check and per suite check (`diagnostics.format_diagnostic_summary`),
one pipeline-level row computed outside the suite (below), and the verdict -- printed
*before* the parameter/variance estimates so a failed run cannot end in a wall of
plausible-looking numbers.

The pipeline-level row:

- **`joint_bread_condition_number`** -- fails above
  `post_deployment_analysis.EXTREME_CONDITION_NUMBER_THRESHOLD` (`1e12`). Condition-number
  thresholds are fraught in general (cond changes under diagonal rescaling, so moderate values
  can be a units artifact), but the argument runs out at the compute precision's wall: the
  matrices come from float32 evaluations (~7 significant digits), so beyond ~`1/eps32 ~ 1e7` a
  solve against the bread retains no trustworthy digits in *any* scaling. `1e12` sits five
  orders past that wall (and is the threshold the diagnostic solves already use to decide the
  bread needs a ridge), so this row fires only for the numerically hopeless.

(The standalone "local linearization error ratio" that used to print alongside these was
removed 2026-09-02: it was exactly the suite's equation-space `r_j` -- same formula, same
O(1/sqrt(n)) covariance-aligned draws -- computed by a pre-suite code path with strictly worse
machinery (one radius, no nonfinite censoring, no `q_j`/`a_{j,l}` companions, no role in the
verdict) at ~25s per run. `local_nonlinearity` is its calibrated replacement; old
`debug_pieces.pkl` files still carry its `local_linearization_error_ratio_*` keys.)

A run is **flagged** (`diagnostics.diagnostics_flagged`, plus the extreme-condition row) when
the verdict is `not_certified`/`invalid`, when the suite was requested but produced no report
(a crashed suite must not look like a passing one), or when the condition gate fires. A
flagged run:

- still completes -- every output file (`analysis.pkl`, `debug_pieces.pkl`,
  `diagnostic_report.pkl`) is written as usual;
- interactively asks for consent before *printing* the estimates (suppressed along with the
  other interactive checks by `--suppress_interactive_data_checks`; declining only skips the
  printout);
- makes the CLI exit with **status 3** -- distinct from generic errors (1) and usage errors
  (2) -- unless `--fail_on_flagged_diagnostics=False` (for calibration sweeps that
  intentionally probe pathological regimes and read `diagnostic_report.pkl` instead). Library
  callers of `analyze_dataset` get `diagnostics_flagged` / `diagnostic_verdict` /
  `diagnostic_classification` on the returned dict and decide for themselves.
