# Diagnostic suite for the adjusted sandwich

`lifejacket.diagnostics` implements a layered set of checks for the adjusted-sandwich
inference produced by `post_deployment_analysis.analyze_dataset`. It is opt-in
(`run_diagnostics=True` / `--run_diagnostics=True`), does not change the adjusted sandwich
estimator itself, and never automatically falls back to the classical sandwich when a check
fails: failure of these checks means the current adjusted-Wald analysis lacks support, not that
the classical sandwich is valid instead.

The suite is organized around one idea: the adjusted sandwich can fail for different reasons,
and a single number cannot certify all of them.

```
correct inputs -> adequate exploration -> stable moments and bread -> linearization -> CLT/normal approximation
```

The `r_j`/`q_j`/`a_{j,l}` family of diagnostics (sections 2 below) examines mainly the
linearization link. They cannot certify the others -- which is why the suite also reports
exploration/importance-weight diagnostics, bread-stability diagnostics, and influence
concentration, and why an exact nonlinear perturbation check and simulator calibration exist as
stronger (and more expensive) alternatives.

## What each check tests, and what it cannot test

| Check (module function) | What it targets | What it cannot rule out |
| --- | --- | --- |
| `check_root_and_implementation` | Wrong supplied functions, an unsolved estimating equation, broken/inconsistent derivatives, non-finite inputs | Anything downstream of a correctly-solved root: overlap, bread conditioning, linearization, the CLT step |
| `check_local_nonlinearity` (`r_j`, `c_j`, `a_{j,l}`) | Whether the first-order Taylor expansion of the estimating equation is adequate at the estimation scale, for the *reported contrasts specifically* | Whether the sampling distribution is actually close to normal (that's the influence-concentration and simulator-calibration checks); overlap/positivity; whether *un-sampled* directions behave differently |
| `check_exact_nonlinear_perturbations` | The same question as above, but exactly (via continuation) rather than via a one-step Taylor correction, plus the resulting distortion to the target's covariance, mean, and tail quantiles | Anything not captured by the sampled perturbation directions; with few directions, only a wide bound on the "bad direction" probability is achievable |
| `check_jacobian_drift` (`rho_j`) | How much the Jacobian changes along a sampled path -- a heuristic input to a contraction-style error bound | Nothing rigorously: this is a *sampled path maximum*, never a certified supremum. `rho_j < 1` does not prove the nonlinear correction is small; `rho_j >= 1` does not by itself prove failure |
| `check_bread_stability` | Numerical conditioning of the bread matrix and its blocks, and the sensitivity of target SEs to numerically negligible perturbations | Statistical identification as distinct from numerical conditioning (a well-conditioned-looking bread can still reflect weak identification if the *meat* is what's driving the standard error up) |
| `check_influence_concentration` (`p_max`, `n_eff`, third-moment) | Whether the estimator's fluctuation is built from many small contributions (the premise the CLT approximation needs), or is dominated by a handful of subjects | It does not itself validate normality -- it only flags a specific, common way that a CLT approximation can fail |
| `check_exploration_and_weights` | Positivity/overlap, importance-weight concentration (ESS), and policy-score-derivative magnitude, evaluated at the estimate *and* under sandwich-scale perturbations | Whether the *supplied* exploration bounds are the deployment's real design bounds -- those must be supplied by the caller via `exploration_floor`/`exploration_ceiling` to be enforced as hard requirements |
| `lifejacket.simulator_calibration.calibrate_and_classify` | End-to-end behavior (bias, coverage, tail imbalance, diagnostic accuracy) under a caller-supplied simulator | Anything outside the simulated family: a calibrated claim is only ever "within this simulator family," never a universal guarantee |

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
   solve itself, reported but not currently gated on a threshold.
4. **Directional finite-difference check.** For `config.finite_difference_num_directions`
   (default `3`) random unit directions `v`, compares the central finite difference
   `(g_tilde(eta_hat + h v) - g_tilde(eta_hat - h v)) / (2h)` (with
   `h = config.finite_difference_step`, default `1e-3`, chosen for float32 precision -- see the
   comment on `DiagnosticConfig.finite_difference_step`) against the analytic `B_hat @ v`. Reports
   `finite_difference_relative_errors` (one per direction) and their max; a max relative error
   above `1%` is a **warning** (catches a broken/mismatched derivative, not meant to certify
   numerical precision at the last bit).
5. **Legacy checks.** Any `(name, callable)` pairs passed in via `legacy_check_callables` are
   invoked; an exception from any of them is caught and turned into a hard failure of this check
   (with the exception message recorded under `metrics["legacy_check::<name>"]`) rather than
   propagating. `post_deployment_analysis.analyze_dataset` wires in
   `input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed` and
   `input_checks.require_estimating_functions_sum_to_zero` here, so the diagnostic suite
   re-verifies (cheaply) that the supplied action-probability function reproduces the recorded
   probabilities and that the update/inferential estimating equations actually average to zero,
   in addition to whatever the main analysis pipeline already checked.

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

For each radius and target, `per_radius[radius]["a_by_target"][label]` reports `median`, `p90`,
`p95`, `max`, the fraction of directions with `a_{j,l} > config.nonlinear_correction_tolerance_se`
(default `0.10`), and a Clopper-Pearson upper bound on that exceedance fraction. `per_radius[
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
rank-deficient, in which case the status is escalated to `indeterminate` instead. `passed`
otherwise. This check **never** hard-fails on its own -- see "Why `r_j`/raw `q_j` have no
universal threshold" below for why.

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
`final_residual_norm`, `num_iterations`, and `branch_change_suspected` (converged, but
`||delta^{NL} - delta^{L}|| / ||delta^{L}|| > 5`, a heuristic flag for "this looks like a
different root," not a proof).

Aggregated across all directions:

- `a_nl_by_target[label]`: quantile summary of `a^{NL}_{j,l} = |l @ e_j| / se(l^T eta_hat)` where
  `e_j = delta_j^{NL} - delta_j^{L}` -- supersedes the cheap `a_{j,l}` when available.
- `se_ratios`: `sqrt(lambda_k)` for the generalized eigenvalues of `Cov_j(L delta_j^{NL})` relative
  to `V_L = L @ V_hat @ L^T` (via `se_ratios_from_generalized_eigenvalues`, i.e.
  `scipy.linalg.eigh(nonlinear_cov, V_L)`) -- the nonlinear-to-linear standard-error ratio along
  each identified target direction. `se_ratios_within_tolerance` is `True` only if every ratio
  falls in `[0.95, 1.05]`.
- `mean_shift_se`: `||V_L^{-1/2} @ mean_j(L @ delta_j^{NL})||` (via `matrix_inv_sqrt`) -- the
  standardized nonlinear mean shift.
- `quantile_shifts_se[label]`: the 2.5%/97.5% quantiles of the nonlinear draws minus the linear
  draws, in SE units, for each scalar contrast.
- `root_failure_fraction`/`branch_change_fraction`/`domain_failure_fraction` and their
  Clopper-Pearson upper bounds (`root_failure_upper_bound`, etc., at `config.confidence_level`,
  default `0.95`).

**Status logic:** `indeterminate` if `root_failure_upper_bound > config.
bad_direction_probability_target` (default `0.01` -- with few directions this is very easy to
trip: even zero observed failures in 30 draws only bounds the failure probability below ~9%, so
turning this check on with a small `num_exact_directions` will often read `indeterminate` on that
basis alone, which is the honest answer, not a bug -- use `num_exact_directions >= ~300` for the
`~1%` bound the task write-up references). `failed` if `se_ratios_within_tolerance` is `False`, or
`mean_shift_se` exceeds `config.mean_shift_tolerance_se` (default `0.10`), or any quantile shift
exceeds `config.quantile_shift_tolerance_se` (default `0.10`) in absolute value. `passed`
otherwise.

### 4. Jacobian drift -- `check_jacobian_drift` (`rho_j`)

Also gated on `config.compute_exact_nonlinear_roots` (it shares the same "this is expensive,
opt-in" framing). Uses a *separate, smaller* sample of `min(config.drift_num_directions, config.
num_directions)` directions (`config.drift_num_directions` defaults to `3` -- re-differentiating
`g_tilde` is far more expensive than evaluating it, so this stays small even when `num_exact_
directions` is cranked up for the check above). For each direction and each `t` in `config.
drift_path_samples` (default `(0.0, 0.5, 1.0)`), computes the true Jacobian `D g_tilde(eta_hat + t
* delta_j)` via `jax.jacrev` and the operator norm of `B_hat^{-1} (D g_tilde(...) - B_hat)` (via a
stable solve of a matrix right-hand side, never an explicit inverse). `rho_by_direction[j]` is the
max of that operator norm over the sampled `t`; `rho_max` is the max over directions.

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
- `target_covariance_eigenvalues`/`target_covariance_rank_estimate`: eigenvalues of `V_hat` and
  how many exceed `config.rank_tolerance * max_eigenvalue` (default `1e-8`).
- `numerical_sensitivity_max_relative_se_change`: `B_hat` is perturbed by a relative
  `config.bread_perturbation_relative_scale` (default `1e-6` -- chosen to sit well above float32
  machine epsilon, `~1.2e-7`, so the perturbation is not itself silently rounded away) times
  `||B_hat|| * I`, the theta-only SEs are recomputed via the same QR technique used elsewhere
  (`_theta_only_variance_diag_qr`, no explicit inverse), and the maximum relative change in any
  theta SE is reported. This measures numerical fragility specifically, as distinct from
  statistical identification.

**No universal condition-number threshold is hard-coded anywhere in this function** -- every
number above is reported as a metric, and only two things drive the status: `indeterminate` if
`target_covariance_rank_estimate < theta_dim` (weak/degenerate identification), and (further)
`indeterminate` if `numerical_sensitivity_max_relative_se_change > config.se_distortion_tolerance`
(default `0.05`) -- i.e. the reported SEs are themselves numerically fragile at a scale well below
real precision loss. `passed` otherwise.

### 6. Influence concentration -- `check_influence_concentration`

Runs when `config.compute_influence_and_overlap_checks` (default `True`). For each contrast `l`,
solves `w = B_hat^{-T} l` once (`solve_with_bread_transpose`, no explicit inverse) and forms the
per-subject linearized influence `xi_i = -(per_subject_stacks @ w)_i` for every subject `i` in one
vectorized dot product. From `xi`, reports per target label in `by_target[label]`:

- `p_max`: the largest single-subject share of `sum(xi_i^2)`;
- `n_eff = 1 / sum(p_i^2)` (bounded below by `1`, by construction -- see the code comment on why
  the warning threshold below is `max(2.0, 0.1*n)` rather than a pure `0.1*n`, which can never
  fire for small `n`);
- `third_moment_concentration = sum(|xi_i|^3) / sum(xi_i^2)^1.5`;
- `top_influential_subjects`: the top-5 subjects by `|xi_i|`, each with its `xi` value and
  variance share, and its `subject_id` (from the `subject_ids` passed to `run_diagnostic_suite`)
  when `config.report_subject_identifiers` (default `True`).

**Status logic:** `warning` for any target with `n_eff < max(2.0, 0.1*n)` or `p_max > 0.5`.
This check never hard-fails -- concentrated influence is evidence against the CLT premise, not
proof that the estimator is wrong.

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
  `(0, 1)`. (This is where positivity/overlap is actually enforced for the new suite; the
  legacy `input_checks.require_action_probabilities_in_range_0_to_1` deliberately stays a no-op
  in the always-on pipeline for backward compatibility -- see the comment on that function.)
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

### 8. Simulator calibration -- `lifejacket.simulator_calibration.calibrate_and_classify`

See the dedicated section below.

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

## Why the exact nonlinear perturbation check is stronger

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
Clopper-Pearson calculation in `lifejacket.simulator_calibration.clopper_pearson_upper_bound`).

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
`run_diagnostic_suite`/`calibrate_and_classify` only ever report a classification for the
*adjusted* analysis, and why the classification vocabulary is `supported` /
`locally_supported` / `failed` / `indeterminate` rather than a "use this sandwich instead" flag.

## Classification

- `supported`: prerequisite and local checks pass, *and* `lifejacket.simulator_calibration.
  calibrate_and_classify` has certified a low `P(inferential failure | diagnostics pass)` on a
  held-out simulator sweep for this deployment family. `run_diagnostic_suite` alone can never
  return this.
- `locally_supported`: the observed-data checks pass, but no end-to-end simulator calibration is
  available.
- `failed`: a hard prerequisite (root/implementation, out-of-range probabilities, non-finite
  values) or a material measured distortion (SE-ratio/mean-shift/quantile-shift outside
  tolerance in the exact nonlinear check) failed.
- `indeterminate`: weak identification, rank-deficient target covariance, unstable solves,
  insufficient perturbation directions, or inadequate simulator coverage prevent a conclusion
  either way.

## Simulator calibration

`lifejacket.simulator_calibration.calibrate_and_classify` is a simulator-agnostic interface: it
does not assume or invent any particular deployment model. Given a `replay_fn(seed) ->
DeploymentReplay` that replays recruitment, outcome generation, policy updates, action
selection, estimation, and the adjusted sandwich calculation for one simulated deployment, it
runs the diagnostic suite on training and held-out seeds, and among held-out replays whose
diagnostics pass, computes `P(inferential failure | diagnostics pass)` (via a caller-supplied
`failure_predicate`) and its one-sided Clopper-Pearson upper confidence bound. Only when that
bound is below the configured `risk_tolerance` does it return `supported`; any such claim is
scoped to "within this simulator family," never a universal guarantee. `tests/unit_tests/
test_simulator_calibration.py` exercises `calibrate_and_classify` directly against small
in-process toy replays (affine estimating maps), which is the fastest way to validate the
calibration/classification logic itself in isolation from a full RL simulator.

This repository does contain a full deployment simulator
(`tests/simulators_and_runners/rl_study_simulation.py`, driven by `run_local_synthetic.sh`), and
`post_deployment_analysis.analyze_dataset` accepts `run_diagnostics`/`diagnostic_config` so the
suite can be run against real simulated deployments end-to-end -- see
`tests/integration_tests/test_RL_diagnostics_smoke`, which runs the simulator, the estimator,
and the diagnostic suite together via the CLI and checks that `diagnostic_report.pkl` comes out
sane. Wiring that same simulator directly into `calibrate_and_classify`'s in-process `replay_fn`
contract would require exposing the estimator's internal bread/meat/`g_tilde` pieces from
`analyze_dataset` for reuse (rather than only its pickled outputs), which this change
deliberately does not do in order to avoid touching the core estimation pipeline more than
necessary; a deployment wanting the full `calibrate_and_classify` treatment against this
repository's simulator can build a `replay_fn` the same way `analyze_dataset` does internally,
using the already-public helpers in `post_deployment_analysis`
(`construct_beta_index_by_policy_num_map`, `construct_classical_and_adjusted_sandwiches`, etc.).
