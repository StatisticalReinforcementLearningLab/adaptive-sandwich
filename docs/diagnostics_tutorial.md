# Diagnostics tutorial: reading and acting on the diagnostic suite

`lifejacket.diagnostics` runs by default on every `analyze_dataset`/`lifejacket analyze` call
(`run_diagnostics=True`), producing a `DiagnosticReport` written to `diagnostic_report.pkl` next
to `analysis.pkl`. `docs/diagnostics.md` is the full technical reference (every formula, every
default tolerance, what each check can and can't rule out); this is the practical companion --
what to actually look at, in what order, and what to do about what you find.

## 1. The one field to look at first: `classification`

```python
import pickle
report = pickle.load(open("diagnostic_report.pkl", "rb"))
report.classification
```

Four possible values:

- **`failed`** -- a hard prerequisite broke: an `input_check_results` entry failed (section 4.1),
  `root_and_implementation` broke, out-of-range action probabilities, non-finite values, or a
  material distortion was *measured* -- reachable only via the opt-in checks: the exact-nonlinear
  check's gates, or `check_multiplier_bootstrap` finding bootstrap SEs above their null band
  (the sandwich SE understated -- the anticonservative direction). Stop and look at
  `check_results`/`input_check_results` before trusting anything else in the report.
- **`indeterminate`** -- weak/degenerate identification, an unstable solve, or (for the opt-in
  re-solving checks) continuation/bootstrap re-solves that actually failed to converge -- which
  is itself a fragility finding about the estimating equation, not merely missing information
  (see section 4.6). Not proof of wrongness, but not resolvable from this dataset: in the
  ADS-142 experiments, every genuinely miscalibrated design landed here rather than falsely
  passing.
- **`locally_supported`** -- everything on-by-default passed. This is the ceiling
  `run_diagnostic_suite` can reach on its own, **always** -- read the next paragraph before
  treating it as "safe."
- **`supported`** -- only ever returned by `simulator_calibration.calibrate_and_classify` (section
  10 below -- a separate, multi-run experiment, not one of `run_diagnostic_suite`'s own checks),
  after a held-out simulator sweep certified a low failure rate. If you're only calling
  `analyze_dataset`, you will never see this.

**`locally_supported` is not "everything is fine," it's "nothing checkable-without-a-simulator
looks broken."** It says the linearization holds *near* `eta_hat` and the observed-data checks
pass -- it does not say `sqrt(n)(eta_hat - eta*)` is actually close to normal (that needs
`check_influence_concentration`'s premise to hold *and* simulator calibration, neither of which a
single `locally_supported` guarantees). And **a `failed`/`indeterminate` result never means "fall
back to the classical sandwich instead"** -- the classical sandwich relies on a different
assumption (the policy didn't depend on past data) that a genuinely adaptive deployment usually
violates too. A failed diagnostic means the adjusted analysis lacks support *here*; it says nothing
about whether the classical one would have been better-founded.

**Two separate dicts, on purpose.** `DiagnosticReport` has both `check_results` (the numbered
statistical checks below) and `input_check_results` (black-and-white data/wiring correctness --
section 4.1). A failure in either one forces `classification == "failed"`, but they're never
merged into one status: a data-hygiene bug (e.g. action probabilities that don't reconstruct) and
a genuine statistical finding (e.g. the root wasn't actually solved) can look identical from
`classification` alone, but you can always tell them apart by which dict the failing entry is in.

## 2. Cheat sheet: report says X, look at Y

| You see | Look at | Likely means |
| --- | --- | --- |
| `classification == "failed"`, an entry in `input_check_results` is the culprit | That entry's `.message` | A data/wiring bug (e.g. action probabilities that can't be reconstructed from the supplied function) -- section 4.1. Not a statistical finding at all; fix the data/wiring before looking at anything else. |
| `classification == "failed"`, `root_and_implementation` is the culprit | `metrics["a_root_max"]`, `metrics["backward_relative_residual"]` | The estimating equation wasn't actually solved, the derivative doesn't match the function, or (if `backward_relative_residual` is the one that's large) the *linear solve itself* is broken -- see section 5's case E. All are implementation bugs, not statistics questions -- fix before looking at anything else. |
| `classification == "failed"`, `exploration_and_weights` is the culprit | `action_prob_global_min`/`_max`, `fraction_at_or_near_floor`/`_ceiling` | A recorded action probability left `(0, 1)`, or violated a design bound you supplied. A real positivity/overlap violation, not a numerical artifact. |
| `classification == "indeterminate"`, `bread_stability` is the culprit | `target_covariance_rank_estimate` vs. `theta_dim`, `numerical_sensitivity_max_relative_se_change` | Either the target is weakly identified (rank-deficient `V_hat`), or the reported SEs themselves move under a numerically negligible perturbation of the bread -- a fragility problem distinct from identification. |
| `classification == "locally_supported"`, but `warnings` is non-empty | Which check's name prefixes each warning string | `local_nonlinearity` and `influence_concentration` warnings never change the classification (see section 3, and section 4.4 for influence_concentration specifically -- it has *no* path to anything but `PASSED`/`WARNING` at all) -- read them yourself, they're the two checks most likely to have something real to say even when nothing "failed." |
| You turned on the expensive checks and got `indeterminate` from `exact_nonlinear_perturbation` | `root_failure_fraction`/`num_converged_trials` vs. `num_trials` | Some continuation solves actually failed (the status is driven by the *observed* failure fraction -- zero observed failures no longer reads `indeterminate` at small direction counts). The distortion statistics you do get were computed on the converged subset only, which is optimistically selected -- treat them as a lower bound on trouble. `root_failure_upper_bound` (Clopper-Pearson) is still reported if you want the certified bound; ~300+ directions for a ~1% one. |
| `multiplier_bootstrap` is `failed` | `se_ratio_by_target` vs. `se_ratio_null_band`, `mean_shift_se` | Bootstrap SEs sit above the null band (the sandwich SE is likely *understated* on this dataset -- the dangerous direction), or the resampled roots are systematically displaced (curvature bias). Do not report the interval as-is. |
| `multiplier_bootstrap` is `warning` | `se_ratio_by_target` below the band | Bootstrap SEs sit *below* the sandwich's: the sandwich is likely overstating uncertainty (the conservative blow-up direction ADS-142 found to be the common failure mode). The interval's direction is trustworthy; its width is inflated. |
| `multiplier_bootstrap` is `indeterminate` | `root_failure_fraction` | The equation would not re-solve under resampling-scale perturbations -- fragility is the finding. In ADS-142, every genuinely anticonservative design landed here; treat as "cannot certify," not "probably fine." |

## 3. `a_{j,l}` vs. `r_j`/`q_j` -- the headline metric and its raw ingredients

`check_local_nonlinearity` reports three related numbers per sampled perturbation direction, and
they are not interchangeable:

- `r_j = ||R_j|| / ||B_hat @ delta_j||` -- an **equation-space** ratio (how big is the Taylor
  remainder relative to the linear term, in whatever units the estimating equation happens to be
  written in).
- `q_j = ||c_j|| / ||delta_j||` -- a **raw parameter-space** ratio (how big is the resulting
  parameter correction relative to the perturbation, in whatever units the parameters happen to be
  written in).
- `a_{j,l} = |l^T c_j| / se(l^T eta_hat)` -- the nonlinear correction to *one specific reported
  contrast*, in units of *that contrast's own standard error*.

**Only `a_{j,l}` gets a default tolerance (`nonlinear_correction_tolerance_se = 0.10`), and that's
deliberate, not an oversight.** `r_j` and `q_j` are real ratios (you could call them percentages),
but "10%" only means the same thing across two different, statistically equivalent
implementations if both the numerator and denominator are pinned to something that can't drift
between them. `r_j`/`q_j` aren't: rescale the estimating equation by any invertible `D` (e.g. one
component measured in dollars vs. cents), and the root, `theta_hat`, and every reported standard
error are all *provably unchanged* -- but `r_j` changes, because `D` stretches the remainder and
the linear term by different amounts unless `D` happens to be a pure rotation. Reparametrize the
parameters by any invertible `P` (e.g. reporting a rate per day instead of per hour) and the same
thing happens to `q_j`. `a_{j,l}` is immune to both: the same `l`/`se` transformation that a
reparametrization forces on the numerator forces on the denominator too, and it cancels exactly.
`tests/unit_tests/test_diagnostics.py::test_equation_rescaling_changes_r_but_not_c_or_a` and
`::test_parameter_rescaling_changes_q_but_not_a` demonstrate this directly with a `diag([7.0,
0.1])` equation rescaling and a `diag([5.0, 0.2])` reparametrization, respectively -- both move
`r_j`/`q_j` while leaving `c_j` (and hence `a_{j,l}`) bit-for-bit identical.

**So should you ignore `r_j`/`q_j`? No -- report them, just never gate on an absolute cutoff for
them (the code already doesn't).** They're still useful for three things:

1. **The radius-scaling sanity check already uses `r_j` productively, and safely.** It compares
   `r_j` at a small configured radius to `r_j` at a large one, and *that ratio* is invariant to
   equation rescaling (the same arbitrary `D` applies at both radii, so it cancels out of the
   ratio-of-ratios even though the absolute value doesn't). If `r_j` doesn't scale roughly linearly
   with radius (or `a_{j,l}` roughly quadratically), that's a real, portable warning -- often a
   sign of a bug rather than of a genuinely poorly-scaled problem.
2. **They localize *why* `a_{j,l}` failed, when it does.** A large `r_j` means the equation itself
   is very nonlinear everywhere. A small `r_j` with a large `a_{j,l}` instead means the equation is
   fine but the bread is amplifying a small remainder into a large effect specifically along your
   reported contrast (a bread-stability question, not a linearization one) -- pair it with
   `check_bread_stability`'s condition numbers to tell the two apart.
3. **They're internally comparable run-over-run for one fixed pipeline.** If you always write your
   `g_tilde` the same way, `r_j`/`q_j` are consistent from one analysis to the next even though
   they aren't portable to someone else's differently-scaled implementation of the same model --
   useful for tracking drift within a single deployment's own codebase.

## 4. What the rest of the suite is actually for

The technical spec (`docs/diagnostics.md`) tells you the formula for each check; here's the
practical "why would I care" for each one, in the order they run. (`lifejacket.simulator_calibration
.calibrate_and_classify` is deliberately not one of these -- see section 10.)

### 4.1 `run_input_checks` -- black-and-white correctness, not a statistical measurement

**Mechanics.** Lives in its own `DiagnosticReport.input_check_results` dict, never folded into
`check_results`. Takes `(name, callable)` pairs -- the same legacy `lifejacket.input_checks`
functions the main `analyze_dataset` pipeline already runs -- and converts each one's hard raise
into its own `failed` `CheckResult` rather than propagating, giving an otherwise-interactive check
(one that can pop a `(y/n)` confirmation prompt) a non-interactive, structured, always-present
outcome instead. `analyze_dataset` wires in
`input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed` here.

**Why it's separate from every numbered check below.** These are questions about whether the
*supplied data and functions* are wired together correctly -- not numeric measurements of how
appropriate the adjusted sandwich is for this experiment, which is what every check from 4.2
onward actually does. A `failed` entry here still short-circuits the rest of the suite exactly
like a `failed` `root_and_implementation` does, and still forces `classification == "failed"` --
but `root_and_implementation`'s own `status` is never affected by it, so a data-hygiene bug can
never masquerade as (or hide behind) a statistical finding. See section 6 for a reproducible
example.

**The sum-to-zero check isn't wired in here, on purpose -- and it's now SE-standardized.** The
main `analyze_dataset` pipeline runs
`require_estimating_functions_sum_to_zero_se_standardized` unconditionally (interactively): the
residual is judged by the displacement it induces on every stacked estimate in units of that
estimate's own SE (`a_j = |(B_hat^{-1} r)_j| / SE_j`, soft 0.01 / hard 0.1), replacing the
legacy raw-units version whose absolute tolerances were reward-scale-dependent and documented to
false-alarm on healthy high-noise runs (docs/adr/0002). This is `a_root_max`'s construction
extended beyond the theta targets to every update block, with per-update attribution -- so it is
a genuine value-level test of the stacked model of the algorithm (do the recorded parameters
actually root the claimed update equations on the real data?), not just a rerun of the root
check. It isn't duplicated into this dict because the pipeline already hard-runs it before the
suite ever starts.

### 4.2 `check_root_and_implementation` -- is the thing I built actually correct?

This is a correctness gate, not a statistics question. It answers: did the estimating equation
actually get solved, are the supplied `g_tilde`/derivative mutually consistent, are the inputs
finite, and did the linear solve itself actually run correctly (`backward_residual_tolerance` --
a hardware-scale smoke detector, not a calibrated statistical tolerance; see section 5's case E)?
Care about this before anything else in the report, because every other check assumes this one
passed -- a `failed` status here short-circuits the whole suite for exactly that reason. In
practice this is the check most likely to catch "I wired the wrong function/index into
`analyze_dataset`" during development, well before you'd otherwise notice from a wrong-looking
number downstream.

### 4.3 `check_bread_stability` -- can I trust these standard errors as *numbers*, independent of
what they mean statistically?

Reports condition numbers (per-update blocks, the full bread, the theta block) and, more
importantly, how much the reported standard errors move under a perturbation of the bread that's
numerically negligible (`1e-6` relative, far below anything a real re-estimate would produce).
Care about this when standing up a new deployment (a redesigned feature set, a new update
cadence) or scaling to a much larger study than you've tested at -- it catches "these SEs are
numerically fragile" as a distinct failure mode from "these SEs reflect weak identification,"
which matters because the fix is different (a numerics bug vs. an actual design problem).

### 4.4 `check_influence_concentration` -- is my reported CI actually about the whole sample, or
about a handful of people?

**Mechanics.** For each reported contrast `l`, it solves `w = B_hat^{-T} @ l` once (a single
transpose-solve, never an explicit inverse) and forms the per-subject linearized influence
`xi_i = -(per_subject_stacks @ w)_i` for every subject `i` in one vectorized dot product -- this
is the first-order sensitivity of the reported contrast to subject `i`'s own data. From
`xi`, it reports, per target:

- `p_max` -- the largest single subject's share of `sum(xi_i^2)` (i.e. of the total variance the
  contrast's estimator accumulates across subjects).
- `n_eff = 1 / sum(p_i^2)` -- the "effective number of equally-influential subjects." Bounded
  below by `1` by construction, which is why the warning threshold is
  `max(config.influence_n_eff_min_floor, config.influence_n_eff_min_fraction * n)` (defaults
  `2.0`/`0.1`) rather than a bare fraction of `n` (which could never fire for small `n`).
- `third_moment_concentration = sum(|xi_i|^3) / sum(xi_i^2)^1.5` -- a skewness-flavored measure of
  how lopsided the contributions are.
- `top_influential_subjects` -- the top-5 subjects by `|xi_i|`, each with its `xi` value, variance
  share, and (if `config.report_subject_identifiers`, default `True`) its actual `subject_id`.

**Why you care.** If `n_eff` for your headline contrast comes back as, say, 3 out of 200 subjects,
your standard error is not really summarizing 200 people's worth of information -- it's dominated
by 3. That's evidence against the CLT approximation the whole adjusted-sandwich machinery leans
on, and it's the single most *actionable* finding in the suite for a study team deciding whether to
trust a result: `top_influential_subjects` names exactly who. Turn on
`compute_leave_one_out_sensitivity` when you want to go one step further and see how much
`theta_hat` itself would move if one of those subjects were excluded (a one-step sensitivity
analysis, explicitly not a valid bootstrap of the adaptive deployment).

**How it's gated -- and the important caveat.** `status` is set to `WARNING` for any target with
`n_eff` below its threshold or `p_max` above its tolerance (`influence_n_eff_min_floor`/
`influence_n_eff_min_fraction`/`influence_p_max_tolerance`, all overridable `DiagnosticConfig`
fields -- see the FAQ); otherwise `PASSED`. **That's the only status logic in this check -- there
is no code path to `INDETERMINATE` or `FAILED` anywhere in it, and overriding the thresholds
changes *when* `WARNING` fires, not what it does once it does.** Combined with
`_combine_classification` only ever looking at `FAILED`/`INDETERMINATE` (section 3, on `a_{j,l}`),
this means **`check_influence_concentration` can never move `classification` away from
`locally_supported`, no matter how concentrated the influence is or how strict you set the
thresholds** -- not via any escalation path at all, which is actually a stronger statement than the
one that applies to `a_{j,l}` (that check at least has an unrelated rank-deficiency condition that
can reach `INDETERMINATE`). A `p_max` of `0.99` and an `n_eff` of `1.01` will read
`locally_supported` with a warning attached, same as a perfectly healthy result would with no
warning. **You have to read `warnings`/`by_target` yourself** -- this is deliberate (the docstring
is explicit that "concentrated influence is evidence against the CLT premise, not proof that the
estimator is wrong"), but it means an automated pipeline that only checks `report.classification`
will silently miss this entirely. The ADS-142 calibration (docs/adr/0002, results section)
settled the gate question with data: warning-level is *correct* -- the blow-ups `n_eff` predicts
are conservative (over-wide CIs, coverage intact), so a hard gate would block valid inference.
Read the warning with its measured meaning: at the default floor, precision 0.73 / recall 0.59
for a >2x-inflated variance estimate. `p_max` at its 0.5 default is essentially inert (fired on
~2% of stressed replicates, recall 0.04) -- treat it as a last-resort tripwire, not a calibrated
gauge.

### 4.5 `check_exploration_and_weights` -- did the policy actually explore enough, and is that
about to blow up my variance?

**Mechanics.** Two different jobs in one check:

- **A hard requirement, always checked:** every recorded action probability must be finite and
  strictly inside `(0, 1)` -- either violation is a hard `failed`. If you additionally supply the
  deployment's actual design bounds (`config.exploration_floor`/`exploration_ceiling` -- these are
  not recoverable from the data itself, they have to come from you), any recorded probability
  outside them is *also* a hard `failed`. This is where positivity/overlap is actually enforced.
- **Leading indicators, reported only, no threshold at all:** `ess_over_n_by_direction`/
  `max_cumulative_weight_by_direction` (importance-weight concentration under
  `num_directions`-many sandwich-scale perturbations of beta) and
  `policy_score_gradient_norm_summary` (if gradients are supplied). There is deliberately no cutoff
  on these -- the "right" ESS threshold is as deployment-specific as `a_{j,l}`'s tolerance, and
  isn't calibrated by this package at all. Watch them the same way you'd watch `r_j`/`q_j`
  (section 3): informative, not gating.

**How it's gated.** This is one of only two checks that can independently reach `failed` on its
own from a *default* run (the other being `root_and_implementation`) -- `local_nonlinearity`,
`bread_stability`, `influence_concentration`, and `jacobian_drift` structurally cannot.

**Is this redundant with `input_checks`'s existing action-probability checks? No -- the opposite.**
`input_checks.require_action_probabilities_in_range_0_to_1` *used to be* a **deliberate no-op**: it
computed `analysis_df[action_prob_col_name].between(0, 1, inclusive="neither").all()` and never
asserted on the result. Its own comment explained why: some legitimate near-deterministic policies
produce recorded probabilities of exactly `0.0`/`1.0` after floating-point rounding, and turning
this into a hard assertion in the always-on input-check path would have been a
backward-incompatible break -- so the comment said outright that `check_exploration_and_weights`
"is the appropriate place to enforce this positivity/overlap requirement" instead. So the two
were never duplicated logic doing the same job twice; the legacy one was intentionally left
toothless once this one existed to actually do the job, and has since been removed from
`input_checks.py` entirely.

### 4.6 `check_multiplier_bootstrap` / `check_exact_nonlinear_perturbations` /
`check_jacobian_drift` -- "are you sure," the expensive way

All opt-in, because each costs real root-solves (or a real re-differentiation) per draw or
direction -- but they have *separate* switches: the bootstrap has its own
`multiplier_bootstrap` mode (`"off"`/`"auto"`/`"always"`; `"auto"` triggers off the cheap
`a_{j,l}` screen), while the exact/Jacobian pair share `compute_exact_nonlinear_roots=True`.
When you want a per-dataset verdict on the reported SEs, reach for the **bootstrap first** --
that's the calibrated recommendation from ADS-142, where `a_{j,l}` and the exact check ranked
*designs* by severity well but neither predicted per-dataset error-bar accuracy, while the
bootstrap matched known truth within ~4% on clean cells and refused to certify every genuinely
miscalibrated design. The exact check remains the right instrument for *measuring* nonlinearity
itself (research/design questions rather than certification).

**`check_exact_nonlinear_perturbations`** solves the perturbed equation exactly via continuation
(`solve_exact_perturbation`: chord-Newton against the *same* fixed `B_hat` factorization, never
re-differentiated, which is what makes hundreds of directions tractable) instead of a one-step
Taylor correction, then reports `se_ratios` (generalized-eigenvalue nonlinear-to-linear SE ratios),
`mean_shift_se`, `quantile_shifts_se`, and Clopper-Pearson upper bounds on root-failure/branch-change/
domain-failure fractions. Ensemble statistics are computed on **converged** solves only, and the
`se_ratios` verdict is judged against a **simulated finite-J null band** for the observed
converged pattern, not a fixed band -- the former hardcoded `[0.95, 1.05]` band ignored the
finite-direction sampling noise of the ensemble covariance and failed 100% of the ADS-142
experiment's 6,648 replicates, near-affine cells included (`docs/adr/0002`, results section).
Three-tier status logic: `failed` if a distortion gate fires on the converged subset
(above-band `se_ratios`; `mean_shift_se`/either quantile shift beyond `mean_shift_tolerance_se`/
`quantile_shift_tolerance_se`, both `0.10` by default; below-band `se_ratios` counts only when
the solver is healthy, since convergence-censoring itself compresses the ensemble);
`indeterminate` if too few solves converged, or gates pass but the observed failure fraction
exceeds `config.bad_direction_probability_target` (a pass on the optimistically-censored subset
is unearned; a failure on it is trustworthy); `passed` otherwise.

**`check_multiplier_bootstrap`** (see `docs/diagnostics.md` section 3b) reuses the same
continuation machinery to re-solve the equation under frozen-score multiplier-bootstrap draws
and compares bootstrap SEs against the sandwich SEs, judged against their own simulated null
band -- the per-dataset, no-simulator verdict. `multiplier_bootstrap="auto"` runs it only when
the cheap `a_{j,l}` screen (`bootstrap_screen_a_jl_threshold`, default `0.05`) trips, which is
the calibrated division of labor: `a_{j,l}` locates suspicious datasets cheaply, the bootstrap
delivers the verdict.

**`check_jacobian_drift`** samples the true Jacobian `D g_tilde` at a few points along a few
perturbation paths and reports `rho_max = max_j rho_j` where
`rho_j = ||B_hat^{-1}(D g_tilde(path point) - B_hat)||_op`. Status is `warning` if any
`rho_j >= 1.0`, else `passed` -- **and unlike almost every other cutoff in this module, `1.0` here
is not an engineering guess awaiting calibration: it's the literal threshold for a contraction
mapping.** `rho_j < 1` is what a contraction-style error bound `||delta^NL - delta^L||/||delta^L||
<= q_j/(1-rho_j)` needs to even be meaningful -- so `1.0` is a mathematical constant here, the same
category as `backward_residual_tolerance` (section 5, case E), not a simulator-calibration
candidate. That said, `check_jacobian_drift` always attaches a warning that `rho_j` is a *sampled
path maximum*, never a certified supremum -- `rho_max < 1` doesn't prove the contraction bound
actually holds over the whole relevant neighborhood, only over the points it happened to sample.
This check can also never reach `INDETERMINATE`/`FAILED` -- same structural ceiling as
`check_influence_concentration` (section 4.4), just gated behind the expensive opt-in rather than
being on by default.

## 5. Reproducing `check_root_and_implementation`'s failure modes

`check_root_and_implementation` isn't one check, it's five sub-checks bundled into one
`CheckResult` (section 4.2), and they don't all behave the same way -- most hard-fail, one only
warns, and one (D) never even reaches a failure. Rather than wait for a real dataset to happen to
trip one, each is reliably reproducible with a tiny hand-built `g_tilde` -- the same style
`tests/unit_tests/test_diagnostics.py` already uses, calling `check_root_and_implementation`
directly rather than going through `analyze_dataset`. All five snippets below were run against
this repo to confirm the exact `status`/`message` shown. (Data/wiring correctness -- the legacy
`input_checks` re-verification this check used to fold in -- is no longer part of it at all; see
section 6.)

**A. Non-finite inputs -> hard `failed`.** Any of `g0`/`B_hat`/`M_hat` containing a NaN/Inf short-
circuits everything else:

```python
import numpy as np, jax.numpy as jnp
from lifejacket import diagnostics as d

B_hat = M_hat = V_hat = np.eye(2)
eta_hat = jnp.zeros(2)
g0 = np.array([np.nan, 0.0])  # g_tilde(eta_hat) would also need to actually return this
result = d.check_root_and_implementation(
    lambda eta: jnp.asarray([np.nan, 0.0]), eta_hat, g0, B_hat, M_hat,
    d.factor_bread(B_hat), np.eye(2), ["theta_0", "theta_1"], V_hat, d.DiagnosticConfig(),
)
print(result.status, result.message)
# failed  g_tilde(eta_hat), B_hat, or M_hat contains nonfinite values.
```

**B. Root correction exceeds `root_error_tolerance_se` (default `0.01`) -> hard `failed`.** Give it
an `eta_hat` that isn't actually a root -- here `g_tilde(eta) = eta - [1, 0]`, evaluated at
`eta_hat = 0`, so `g0 = [-1, 0]` and the implied correction is 10 standard errors away for a
target with `se = 0.1`:

```python
B_true = np.eye(2)
result = d.check_root_and_implementation(
    lambda eta: B_true @ eta - jnp.array([1.0, 0.0]), jnp.zeros(2), np.array([-1.0, 0.0]),
    B_true, M_hat, d.factor_bread(B_true), np.eye(2), ["theta_0", "theta_1"],
    np.eye(2) * 0.01,  # se = 0.1 per contrast
    d.DiagnosticConfig(),
)
print(result.status, result.message, result.metrics["a_root_max"])
# failed  Root correction of 10 SE exceeds tolerance 0.01.  10.0
```

**C. Finite-difference disagrees with `B_hat` by >1% -> `warning`, not `failed`.** Pass a `B_hat`
that doesn't actually match `g_tilde`'s real derivative (here `g_tilde` is really `I @ eta`, but
the supplied "analytic" Jacobian is `diag([1, 2])`) -- note `g0` is still exactly `0`, so this
isolates the finite-difference warning from case B:

```python
B_wrong = np.array([[1.0, 0.0], [0.0, 2.0]])
result = d.check_root_and_implementation(
    lambda eta: B_true @ eta, jnp.zeros(2), np.zeros(2), B_wrong, M_hat,
    d.factor_bread(B_wrong), np.eye(2), ["theta_0", "theta_1"], np.eye(2), d.DiagnosticConfig(),
)
print(result.status, result.warnings, result.metrics["finite_difference_max_relative_error"])
# warning  ['Directional finite-difference check disagrees with B_hat by more than 1% ...']  0.451...
```

**D. A zero-target-variance contrast -> stays `passed`, but with a warning.** Not a failure mode by
itself -- if `se(l^T eta_hat) == 0` for some contrast (here the second diagonal entry of `V_hat` is
exactly `0`), that contrast is excluded from the root-error check rather than forced to `+inf`:

```python
result = d.check_root_and_implementation(
    lambda eta: B_true @ eta, jnp.zeros(2), np.zeros(2), B_true, M_hat,
    d.factor_bread(B_true), np.eye(2), ["theta_0", "theta_1"], np.diag([1.0, 0.0]),
    d.DiagnosticConfig(),
)
print(result.status, result.warnings)
# passed  ['One or more contrasts have (numerically) zero target variance and were excluded ...']
```

**E. Backward relative residual exceeds `backward_residual_tolerance` (default `1e-6`) -> hard
`failed`.** Unlike every other tolerance in this check, this one isn't a statistical judgment call
that needs simulator calibration -- a healthy LU solve is backward-stable *regardless* of how
ill-conditioned `B_hat` is (confirmed empirically: this residual stayed pinned near float64
machine epsilon, ~`1e-16`, across condition numbers spanning twelve orders of magnitude, from
well-conditioned to nearly-exactly-singular). So `1e-6` is a smoke detector for something actually
broken in the solve plumbing, not a "how nonlinear is too nonlinear" question. Reproduced here with
a stale/mismatched `bread_factored` -- the realistic bug this exists to catch (e.g. forgetting to
re-factor `B_hat` after it changes) -- rather than a badly-conditioned `B_hat`, since conditioning
alone won't trip it:

```python
B_true = np.diag([2.0, 3.0])
B_wrong = np.diag([2.0, 30.0])  # factored and used to solve, instead of B_true
result = d.check_root_and_implementation(
    lambda eta: B_true @ eta - jnp.array([0.02, 0.03]), jnp.zeros(2), np.array([-0.02, -0.03]),
    B_true, M_hat, d.factor_bread(B_wrong),  # <-- factored from the WRONG matrix
    np.eye(2), ["theta_0", "theta_1"], np.eye(2) * 1e6,  # inflated so a_root_max stays clean
    d.DiagnosticConfig(),
)
print(result.status, result.metrics["backward_relative_residual"])
# failed  0.745...   (vs. the ~1e-16 you'd see with a correctly-factored B_hat)
```

## 6. Reproducing an `input_check_results` failure

Section 4.1's mechanism, `run_input_checks`, is a plain function you can call directly -- no need
for a real dataset:

```python
from lifejacket import diagnostics as d

def failing_check():
    raise ValueError("could not reconstruct action probabilities")

results = d.run_input_checks([("action_probabilities_reconstructed", failing_check)])
print(results["action_probabilities_reconstructed"].status)
print(results["action_probabilities_reconstructed"].message)
# failed  could not reconstruct action probabilities
```

At the `run_diagnostic_suite` level, this shows up under `report.input_check_results`, *not*
`report.check_results["root_and_implementation"]`, and still forces
`classification == "failed"` and short-circuits the rest of the suite --
`tests/unit_tests/test_diagnostics.py::test_run_diagnostic_suite_hard_fails_on_input_check_failure_but_keeps_it_separate`
demonstrates both halves of that claim on a well-behaved affine map, where `root_and_implementation`
itself reads `passed` even though `classification` is `failed`.

## 7. Running it (Python / CLI)

**CLI** (`lifejacket analyze`, already the default):

```bash
lifejacket analyze \
  ... \
  --run_diagnostics=True \
  --diagnostic_config_pickle="my_config.pkl"   # optional; omit for DiagnosticConfig() defaults
```

**Python** (`post_deployment_analysis.analyze_dataset`):

```python
from lifejacket import diagnostics

config = diagnostics.DiagnosticConfig(
    nonlinear_correction_tolerance_se=0.10,   # the one tolerance with a portable meaning -- see section 3
    multiplier_bootstrap="auto",              # recommended: bootstrap verdict when the a_jl screen trips (section 4.6)
    contrast_matrix=my_L,                     # optional: defaults to the identity on theta otherwise
    target_labels=["marginal_effect"],        # must match contrast_matrix's row count if supplied
)
result = analyze_dataset(..., run_diagnostics=True, diagnostic_config=config)
```

Custom `contrast_matrix`/`target_labels` are how you point every SE-standardized check
(`a_{j,l}`, the exact check's `se_ratios`/`mean_shift_se`/`quantile_shifts_se`,
`check_influence_concentration`) at the specific quantity you actually report -- e.g. a marginal
treatment effect contrast rather than the raw `theta` components -- rather than accepting the
identity-on-`theta` default.

## 8. FAQ

**Q: My classification is `locally_supported` with zero warnings. Am I done?**
A: You've cleared everything checkable from one dataset without a simulator. If this result is
going into a paper or a real decision, that's the point to run the multiplier bootstrap
(`multiplier_bootstrap="always"`, or `"auto"` to let the `a_{j,l}` screen decide) rather than
stop here -- section 4.6. In the ADS-142 calibration, designs that read clean on the cheap
checks were uniformly well-calibrated, so a quiet report plus a passing bootstrap is as strong
as single-run evidence gets.

**Q: Can I just lower `nonlinear_correction_tolerance_se` to make warnings go away?**
A: You can, but know what the number can and can't do: the ADS-142 calibration
(`docs/adr/0002-diagnostic-threshold-calibration-plan.md`, results section) found `a_{j,l}`
ranks *designs* by nonlinearity almost perfectly but predicts *per-dataset* error-bar accuracy
weakly at any threshold (the ROC is intrinsically shallow). Its validated role is as the screen
that triggers the bootstrap (`bootstrap_screen_a_jl_threshold=0.05`), not as a verdict whose
threshold is worth hand-tuning.

**Q: Why does `check_influence_concentration`'s `n_eff` warning threshold look like
`max(influence_n_eff_min_floor, influence_n_eff_min_fraction * n)` instead of just a fraction of
`n`?**
A: `n_eff` is bounded below by `1` by construction (it can never indicate fewer than "one equally
influential subject"), so a pure fraction of `n` could never fire for small `n` even in the most
extreme single-subject-dominance case -- the floor exists so the warning still has teeth at small
sample sizes.

**Q: Can I tighten `check_influence_concentration`'s thresholds myself?**
A: Yes -- `influence_n_eff_min_floor` (default `2.0`), `influence_n_eff_min_fraction` (default
`0.1`), and `influence_p_max_tolerance` (default `0.5`) are all overridable `DiagnosticConfig`
fields. But remember section 4.4's caveat: tightening them changes when `WARNING` fires, not
whether it can ever gate `classification` -- it can't, by design, and the ADS-142 calibration
answered the "should it?" question with a measured no (the failures it predicts are
conservative, so a hard gate would block valid inference).

**Q: A check says `indeterminate`. Is that better or worse than `failed`?**
A: Neither -- they mean different things. `failed` means something concrete and measured broke.
`indeterminate` means the suite couldn't resolve the question either way (rank deficiency, an
unstable solve, or too few directions) -- usually the actionable response is "get more
information" (more directions, a better-conditioned design), not "this is definitely wrong."

**Q: Why is `input_check_results` a separate dict instead of just another entry in
`check_results`?**
A: So a data-hygiene bug can never be mistaken for a statistical finding just by looking at
`classification` or scanning `check_results`. See section 4.1.

## 9. Where these tolerances come from

Most tolerances in `DiagnosticConfig` began as documented engineering guesses; the ADS-142
cluster experiments (`docs/adr/0002-diagnostic-threshold-calibration-plan.md`, results and
correction sections -- ~27,000 simulated deployments) have since put real operating
characteristics behind the on-by-default ones. The short version: `n_eff`'s default floor has
measured precision 0.73 / recall 0.59 for 2x-inflated variance (kept warning-level on purpose:
the failures it predicts are conservative); `p_max`'s 0.5 default is essentially inert;
`a_{j,l}`'s 0.10 is a design-severity gauge whose calibrated per-dataset role is the 0.05
bootstrap screen; `check_bread_stability` deliberately keeps `indeterminate` as its ceiling
(a `failed` tier was measured and rejected); the exact check's and bootstrap's ensemble gates
are simulated exact nulls, not tolerances at all; and `backward_residual_tolerance` and the
sum-to-zero displacement gates are correctness smoke detectors pinned to machine behavior. Read
the ADR before changing a default -- it records not just the values but the evidence and the
decisions taken on it.

## 10. Going further: validating against a simulator (not one of the checks above)

**`lifejacket.simulator_calibration.calibrate_and_classify` is not a single-run check, and isn't
listed as one in section 4 on purpose.** Every check above runs once, on one dataset, inside one
`analyze_dataset`/`run_diagnostic_suite` call. Actually validating a diagnostic's threshold --
rather than trusting an engineering guess -- requires running the estimator many times against a
simulator with a known ground truth, which is a different kind of activity: a standalone
experiment you design and run, not a flag you pass to `analyze_dataset`.

**The interface.** `calibrate_and_classify` is simulator-agnostic: it does not assume or invent any
particular deployment model. You supply a `replay_fn(seed) -> DeploymentReplay` that replays
recruitment, outcome generation, policy updates, action selection, estimation, and the adjusted
sandwich calculation for one simulated deployment; it runs the diagnostic suite on every seed in
`train_seeds` and `holdout_seeds`, and among *held-out* replays whose diagnostics pass, computes
`P(inferential failure | diagnostics pass)` (via a caller-supplied `failure_predicate`) and its
one-sided Clopper-Pearson upper confidence bound. Only when that bound is below the configured
`risk_tolerance` does it return `supported`; any such claim is scoped to "within this simulator
family," never a universal guarantee.

**Its "thresholds" live at a different layer entirely.** `risk_tolerance` (default `0.05`) and
`confidence_level` (default `0.95`) are plain arguments to `calibrate_and_classify` itself, not
`DiagnosticConfig` fields -- a different call surface for a different situation. The failure
definition is even more caller-controlled: you supply `failure_predicate`, and the shipped
`default_failure_predicate` is explicitly "a minimal, non-authoritative example" with its *own*
three parameters (`se_distortion_tolerance=0.05`/`coverage_tolerance=0.05`/`nominal_coverage=0.95`).
**Watch out for the name collision**: `default_failure_predicate`'s `se_distortion_tolerance`
shares a name (and, coincidentally, a default value) with `DiagnosticConfig.se_distortion_tolerance`
(`check_bread_stability`'s tolerance) but is a completely independent parameter of a completely
different function -- overriding one never touches the other.

**How to actually run one, for your own deployment family:**

1. **Build `replay_fn`.** For each `seed`, run your simulator end-to-end and package a
   `DeploymentReplay`: `diagnostic_kwargs` (unpacked directly into `run_diagnostic_suite`),
   `theta_hat`, `theta_variance_estimate`, and `ground_truth_theta` if your simulator can supply
   one (not every simulator can -- an adaptive design's estimand isn't always a fixed population
   constant; a common approach is a cross-replicate Monte Carlo mean at the same `n`/`T` as the
   test seeds, treated as its own design decision, not something `calibrate_and_classify` picks
   for you).
2. **Split `train_seeds`/`holdout_seeds`, and don't skip this.** `train_seeds` are available for
   your own threshold-tuning but otherwise unused by the function; the reported bound is computed
   *only* from `holdout_seeds`. Reusing training seeds for that bound would be circular.
3. **Write your own `failure_predicate`** matched to what "inferential failure" actually means for
   your deployment -- don't just keep the shipped default, it's explicitly a placeholder.
4. **Call it and read `CalibrationResult`**: `classification`, `conditional_failure_rate_upper_bound`,
   `per_replay_records` (per-seed detail for your own further analysis).

**Honest scope.** This is inherently a multi-run, often computationally significant undertaking --
each replay re-runs your full simulator, estimator, and diagnostic suite.
`tests/unit_tests/test_simulator_calibration.py` exercises `calibrate_and_classify` against small
in-process toy replays (affine estimating maps) -- the fastest way to validate the
calibration/classification logic itself, but not a stand-in for a real experiment at real scale.
`tests/simulators_and_runners/rl_study_simulation.py` (driven by `run_local_synthetic.sh`) is this
repository's own full deployment simulator, already wired through `analyze_dataset`'s
`run_diagnostics`/`diagnostic_config` for testing `run_diagnostic_suite` end-to-end (see
`tests/integration_tests/test_RL_diagnostics_smoke`); wiring it directly into
`calibrate_and_classify`'s `replay_fn` contract is a separate, not-yet-done step (see
`docs/diagnostics.md`'s closing section for what that would take). For a real, cluster-scale
worked example of exactly this kind of experiment -- validating this package's own default
tolerances, not a specific deployment's -- see
`docs/adr/0002-diagnostic-threshold-calibration-plan.md`; its scenario-grid, ground-truth, and
SLURM-mechanics design generalizes directly to validating your own deployment family.
