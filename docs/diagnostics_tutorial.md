# Diagnostics tutorial: reading and acting on the diagnostic suite

`lifejacket.diagnostics` runs by default on every `analyze_dataset`/`lifejacket analyze` call
(`run_diagnostics=True`), producing a `DiagnosticReport` written to `diagnostic_report.pkl` next
to `analysis.pkl`. `docs/diagnostics.md` is the full technical reference (every formula, every
default tolerance, what each check can and can't rule out); this is the practical companion --
what to actually look at, in what order, and what to do about what you find.

## 1. The two fields to look at first: `verdict`, then `classification`

```python
import pickle
report = pickle.load(open("diagnostic_report.pkl", "rb"))
report.verdict, report.verdict_basis, report.classification
```

**`report.verdict` is the decision-level answer to "can I report the adjusted sandwich
variance?"** in four values --
`certified` (report it; `verdict_basis` says whether the multiplier bootstrap verified the SEs
or the run was quiet enough that the calibrated screen never called for it), `conservative`
(report it, width likely inflated -- the one finding `classification` structurally cannot
surface, because warnings never move it), `uncertifiable` (no verdict was reached -- fragility,
censoring, or the screen called for the bootstrap and it didn't run), and `invalid` (something
failed, or the target covariance is rank-deficient -- the zero-width-CI mode). Each value's
meaning is backed by the ADS-142 operating characteristics (see `docs/diagnostics.md`'s
"Verdict" section and the `DiagnosticVerdicts` docstring). Reports pickled before this field
existed carry an empty string -- fall back to reading `classification` plus `warnings` there.

`classification` remains the compatibility-stable field, with four possible values:

- **`failed`** -- a hard prerequisite broke, or a collapse was *measured*: an
  `input_check_results` entry failed (section 4.1), `root_and_implementation` broke,
  out-of-range action probabilities, non-finite values, an unidentified reported component
  (`bread_stability`'s rank gate -- 76/76 of the undercoverage hunt's zero-width collapses),
  `g_tilde` nonfinite at every sampling-scale probe (`local_nonlinearity`'s total-censoring
  gate), a re-solve failure rate whose Clopper-Pearson lower bound clears
  `bad_direction_probability_target` (confirmed fragility, both re-solve checks), or a material
  distortion measured by the opt-in checks: the exact-nonlinear check's gates, or
  `check_multiplier_bootstrap` finding bootstrap SEs above their null band (the sandwich SE
  understated -- the anticonservative direction). Stop and look at
  `check_results`/`input_check_results` before trusting anything else in the report.
- **`indeterminate`** -- an unstable solve, `g_tilde` undefined at too many (but not all) of
  the local-nonlinearity probe points (section 3), a re-solve failure fraction above the target
  without enough trials to statistically confirm it (a confirmed rate is `failed` -- see
  section 4.6), or a gate the check could not evaluate at all, which by design reads
  `indeterminate` rather than `passed`. Not proof of wrongness, but not resolvable from this
  dataset. Treat it as "cannot certify," not "probably fine": the status logic is deliberately
  built so that a miscalibrated design cannot slip through as a pass on an optimistically
  selected subset. That is a design property, not yet a measured one -- the bootstrap-validation
  experiment that would put numbers behind it is still in flight (`docs/adr/0002`, "Follow-up
  experiments").
- **`locally_supported`** -- everything on-by-default passed. This is the ceiling
  `run_diagnostic_suite` can reach on its own, **always** -- read the next paragraph before
  treating it as "safe."

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
| `classification == "failed"`, `bread_stability` is the culprit | `target_covariance_rank_estimate` vs. `target_covariance_dim` | The target is unidentified: a rank-deficient *target* covariance `L @ V_hat @ L^T` (not the full joint `V_hat` -- which is why this gate can actually fire at real study scale). The best-measured predictor of catastrophic variance collapse in the ADS-142 hunt (76/76 zero-width-interval runs), so it fails outright and drives an `invalid` verdict. |
| `classification == "indeterminate"`, `bread_stability` is the culprit | `numerical_sensitivity_max_relative_se_change` | The reported SEs move under a numerically negligible perturbation of the bread -- numerical fragility distinct from identification (the rank gate, which *fails*, covers that). |
| `classification == "locally_supported"`, but `warnings` is non-empty | Which check's name prefixes each warning string | `local_nonlinearity` and `influence_concentration` warnings never change the classification (see section 3, and section 4.4 for influence_concentration specifically -- it has *no* path to anything but `PASSED`/`WARNING` at all) -- read them yourself, they're the two checks most likely to have something real to say even when nothing "failed." |
| You turned on the expensive checks and got `indeterminate` from `exact_nonlinear_perturbation` | `root_failure_fraction`/`num_converged_trials` vs. `num_trials`, then `mean_shift_gate_evaluable` and the warnings | Either some continuation solves failed but not enough to statistically confirm a rate above target (a confirmed rate -- Clopper-Pearson lower bound over `bad_direction_probability_target` -- reads `failed`, not `indeterminate`; the unhealthy gate below that is driven by the *observed* fraction, so zero observed failures no longer reads `indeterminate` at small direction counts), or a distortion gate could not be evaluated at all (typically a singular target covariance), which is deliberately not a pass. The distortion statistics you do get were computed on the converged subset only, which is optimistically selected -- treat them as a lower bound on trouble. `root_failure_upper_bound` (Clopper-Pearson) is still reported if you want the certified bound; ~300+ directions for a ~1% one. |
| `multiplier_bootstrap` is `failed` | `root_failure_fraction` and its lower bound first, then `se_ratio_by_target` vs. `se_ratio_null_band`, `mean_shift_se` vs. `mean_shift_threshold` | Confirmed fragility (the re-solve failure rate's Clopper-Pearson lower bound clears `bad_direction_probability_target` -- the equation measurably cannot be re-solved under resampling-scale perturbations), bootstrap SEs above the null band for at least one identified target (the sandwich SE likely *understated* -- the dangerous direction), or systematically displaced resampled roots (curvature bias). Do not report the interval as-is. |
| `multiplier_bootstrap` is `warning` | `se_ratio_by_target` below the band | Bootstrap SEs sit *below* the sandwich's: the sandwich is likely overstating uncertainty (the conservative blow-up direction ADS-142 found to be the common failure mode). The interval's direction is trustworthy; its width is inflated. |
| `multiplier_bootstrap` is `indeterminate` | `root_failure_fraction`, `se_ratio_band_unevaluable_targets`, `mean_shift_gate_evaluable` | Either re-solves failed at a rate above target but below statistical confirmation (a confirmed rate reads `failed`), or a gate could not be fully evaluated -- e.g. a target with no identified sandwich SE to compare against, so no SE comparison was performed for it. Both read as "cannot certify," not "probably fine": the check is built never to pass on an unevaluated or optimistically censored comparison. Whether that design property translates into catching real anticonservative designs is what the in-flight bootstrap-validation experiment (`docs/adr/0002`, "Follow-up experiments") is meant to measure -- wave 2 itself contained no anticonservative cells to test it on. |
| A re-solve check ran fewer trials than you configured (`num_trials` < `num_planned_trials`, or `monte_carlo_counts["num_bootstrap_draws_executed"]` < `num_bootstrap_draws`) | `early_stopped`, `early_stop_reason`, and the warning naming the truncation | The sequential early stop fired: too few trials could still converge to compute any ensemble statistic, so no completion could have passed -- the status is `failed` when the failure rate is statistically confirmed, `indeterminate` otherwise. Deliberate and (for real-sized plans) status-preserving, not a crash or a timeout -- section 4.6. The reported fractions and Clopper-Pearson bounds are computed on what actually ran, so they are wider, never narrower. |

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

**Check `domain_censored_fraction` before reading any of the three.** The check probes out to
`1.5x` the sandwich scale, which routinely leaves the domain of a log/logit/sqrt-shaped estimating
function. A probe where `g_tilde` returns a nonfinite value is *censored* rather than fatal
(before, one such probe raised out of the always-on check and you got no `diagnostic_report.pkl`
at all), and `check_local_nonlinearity` reports `num_probes`/`num_domain_censored_probes`/
`domain_censored_fraction` at the top level of its metrics and per radius. Every quantile,
exceedance fraction, and Clopper-Pearson bound above is then computed over the surviving probes
only -- which are exactly the directions where the map stayed well-behaved, so a clean-looking
summary on a censored ensemble is optimistically selected. Any censoring at all attaches a warning
and downgrades a `passed` to `warning`; the check goes `indeterminate` once the headline radius
has fewer than 3 surviving probes or more than half of them censored, and `failed` when NONE
survive -- total censoring is a measured statement that the linearization has no domain at
sampling scale, not a gap in the evidence. So these are the further ways `local_nonlinearity`
can reach a gating status (alongside the joint-Mahalanobis rank-deficiency condition) -- the
size of `a_{j,l}` itself still never gates.

## 4. What the rest of the suite is actually for

The technical spec (`docs/diagnostics.md`) tells you the formula for each check; here's the
practical "why would I care" for each one, in the order they run.

### 4.1 `run_input_checks` -- black-and-white correctness, not a statistical measurement

**Mechanics.** Lives in its own `DiagnosticReport.input_check_results` dict, never folded into
`check_results`. Takes `(name, callable)` pairs -- the same legacy `lifejacket.input_checks`
functions the main `analyze_dataset` pipeline already runs -- and converts each one's hard raise
into its own `failed` `CheckResult` rather than propagating, giving an otherwise-interactive check
(one that can pop a `(y/n)` confirmation prompt) a non-interactive, structured outcome instead.
`analyze_dataset` no longer re-executes any check here (as of 2026-09-02): the one check it
used to wire in, `require_action_probabilities_in_analysis_df_can_be_reconstructed`, is the most
expensive input check (it evaluates the action-probability function over every active row), and
it already runs -- as a hard failure, with no interactive continue path -- in the first-wave
input checks near the start of the pipeline, on inputs nothing later touches. Reaching the suite
proves it passed, so `analyze_dataset` records that outcome directly, as a `passed` entry
carrying the measured agreement under `action_probabilities_reconstructed`. Setting
`suppress_all_data_checks=True` does not produce a suppressed variant of this row: it turns the
whole diagnostic suite off (the suite is data checking too, and its verdict could not exceed
`not_certified` with the input checks unrun), so such a run writes no `diagnostic_report.pkl`
and is not flagged.
**Still, `input_check_results` is not a passed/failed dict**: read an `indeterminate` entry
there as "not run," not as a finding -- and note that such an entry caps the verdict at
`not_certified`, because unvalidated inputs cannot certify.

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
`require_estimating_functions_sum_to_zero_se_standardized` (interactively, and like every other
input check only when `suppress_all_data_checks=False`): each component of the residual (the
subject-mean estimating function) is judged against its own standard error, taken directly from
the per-subject values that were averaged
(`a_j = |mean_i psi_ij| / (rms_i psi_ij / sqrt(n))`, soft 0.01 / hard 0.1), replacing the
legacy raw-units version whose absolute tolerances were reward-scale-dependent and documented to
false-alarm on healthy high-noise runs (docs/adr/0002; an intermediate bread/sandwich
displacement form was itself replaced 2026-09-02 because it inherited the sandwich's
degeneracies and went blind exactly in the blow-up regimes). It carries per-update attribution,
so it is a genuine value-level test of the stacked model of the algorithm (do the recorded
parameters actually root the claimed update equations on the real data?), not just a rerun of
the root check. It isn't duplicated into this dict because the pipeline already runs it before
the suite ever starts.

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
one that applies to `a_{j,l}` (that check at least has two conditions unrelated to the size of
`a_{j,l}` -- rank deficiency and severe domain censoring -- that can reach `INDETERMINATE`).
A `p_max` of `0.99` and an `n_eff` of `1.01` will read
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
  `min(config.num_directions, 5)`-many sandwich-scale perturbations of beta -- the cap is in
  `run_diagnostic_suite`, which draws these separately from the local-nonlinearity directions) and
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
`a_{j,l}` screen or a non-`pass` local-nonlinearity check), while the exact/Jacobian pair share
`compute_exact_nonlinear_roots=True`.
When you want a per-dataset verdict on the reported SEs, reach for the **bootstrap first**. That
recommendation comes from the *design* rationale ADS-142 recorded (`docs/adr/0002`, results
section and the decisions taken on them): `a_{j,l}` ranked *designs* by severity essentially
perfectly while replicating poorly within a cell, and the exact check's `a^NL` predicted
per-replicate variance accuracy nowhere (`|rho| <= 0.19`, and `<= 0.06` where the continuation
solver was healthy) -- so neither is a per-dataset verdict. That is why the bootstrap was written:
it re-solves nonlinearly and judges
against a self-calibrating null band rather than an engineering tolerance, so it is the only
instrument in the suite built to answer "is *this* dataset's SE right." Its empirical validation
is still pending: the `ads142_bootstrap_validation.sh` run against five known-truth wave-2 cells
is listed as in flight, with expected-per-cell outcomes only, not measured ones. The exact check
remains the right instrument for *measuring* nonlinearity itself (research/design questions rather
than certification).

**`check_exact_nonlinear_perturbations`** solves the perturbed equation exactly via continuation
(`solve_exact_perturbation`: chord-Newton against the *same* fixed `B_hat` factorization, never
re-differentiated, which is what makes hundreds of directions tractable; each continuation step
runs to its iteration budget, its residual tolerance, or the divergence abort described at the end
of this section) instead of a one-step
Taylor correction, then reports `se_ratios` (generalized-eigenvalue nonlinear-to-linear SE ratios),
`mean_shift_se`, `quantile_shifts_se`, and Clopper-Pearson upper bounds on root-failure/branch-change/
domain-failure fractions. Ensemble statistics are computed on **converged** solves only, and the
`se_ratios` verdict is judged against a **simulated finite-J null band** for the observed
converged pattern, not a fixed band -- the former hardcoded `[0.95, 1.05]` band ignored the
finite-direction sampling noise of the ensemble covariance and failed 100% of the ADS-142
experiment's 6,648 replicates, near-affine cells included (`docs/adr/0002`, results section).
Three-tier status logic: `failed` if a distortion gate fires on the converged subset
(above-band `se_ratios`; `mean_shift_se` above `mean_shift_threshold`; either quantile shift
beyond `quantile_shift_tolerance_se`, `0.10` by default; below-band `se_ratios` counts only when
the solver is healthy, since convergence-censoring itself compresses the ensemble);
`indeterminate` if too few solves converged, or gates pass but the observed failure fraction
exceeds `config.bad_direction_probability_target` (a pass on the optimistically-censored subset
is unearned; a failure on it is trustworthy), or a gate could not be evaluated at all -- an
unevaluable `se_ratios` comparison (singular target covariance) or mean-shift gate is
`indeterminate`, never `passed`; `passed` otherwise.

`mean_shift_se` is no longer compared against the fixed `mean_shift_tolerance_se`. It is measured
on complete `+/-` pairs only and compared against `mean_shift_threshold =
max(mean_shift_null_upper, config.mean_shift_tolerance_se)` -- the simulated finite-draw null
quantile of the same statistic, floored at the configured tolerance. Under intact antithetic
pairing the null band is exactly `0` and the `0.10` floor is what binds (so nothing changes for
the ordinary case); with unpaired draws the band is what stops raw sampling noise -- about `0.26`
SE in one dimension at the default `J = 15` -- from reading as curvature. `mean_shift_null_upper`,
`mean_shift_threshold`, `mean_shift_gate_evaluable`, and `mean_shift_num_rows` are all reported so
you can see which of the two bound.

**`check_multiplier_bootstrap`** (see `docs/diagnostics.md` section 3b) reuses the same
continuation machinery to re-solve the equation under frozen-score multiplier-bootstrap draws
and compares bootstrap SEs against the sandwich SEs, judged against their own simulated null
band -- the per-dataset, no-simulator verdict. The band is applied **per target, over the targets
whose ratio is finite**: an unidentified target (zero sandwich SE) is listed in
`se_ratio_band_unevaluable_targets` and blocks a `passed`, but no longer silences the comparison
for its identified siblings. It shares the exact check's banded mean-shift gate verbatim.
`multiplier_bootstrap="auto"` runs it only when
the cheap `a_{j,l}` screen (`bootstrap_screen_a_jl_threshold`, default `0.05`) trips **or**
`check_local_nonlinearity` did not `pass` outright, which is the calibrated division of labor:
the cheap checks locate suspicious datasets, the bootstrap delivers the verdict.

**What the bootstrap actually costs now, and why you may have heard otherwise.** If you remember
this check turning a five-minute analysis into a multi-hour one, that memory predates a fix to how
`analyze_dataset` builds the `g_tilde` closure it hands the suite: it used to rebuild the entire
`O(n*T)` structural precompute on *every* call, and the bootstrap makes thousands of them. The
closure is now built once and jitted. Measured locally at the pipeline's own `n=100`/`T=50`
fixture (`eta_dim=200`), one evaluation costs `1.4-2.6 ms` against `797 ms` for a faithful
reconstruction of the old closure, and the whole 100-paired-draw check -- 8,027 evaluations --
runs in `12.7-21.7 s` (two independent local harnesses; the spread is machine load, the evaluation
count is identical) against a projected `~1.8 h` under the old closure. Twenty-five draws is
`3.5-5.4 s`; the same suite with the bootstrap off is `6.6 s`. These are **laptop** numbers (JAX
CPU backend, float32), not FASRC cluster numbers -- expect absolute seconds to move by roughly a
factor of two, and see `docs/diagnostics.md` section 3c for the knobs, the divergence abort's
measured margins, and what is still pending re-measurement on the cluster. Practical consequence:
`multiplier_bootstrap="always"` is no longer a decision you need to schedule around at this scale.

**Two things stop a re-solve early, and both report themselves.** The chord-Newton loop aborts a
continuation step that has visibly left the method's basin -- the iterate blew four orders of
magnitude past its own best residual, or went 40 iterations without a new best
(`nonlinear_solver_divergence_abort`, on by default). An aborted solve is counted as an ordinary
root failure, exactly as one that exhausted its iteration budget is, and
`num_divergence_aborted_trials` says how many there were. It saves nothing on healthy solves; it
is worth 20-66% of a check's evaluations only in cells where nearly everything fails. The
thresholds are wide because the only way this can change an answer is by aborting a solve that
would have converged: zero false aborts across ~152,000 converging continuation steps in two
independent corpora, with a ~270x margin on the blow-up clause and 2x on the stall clause. That is
a measurement, not a proof -- if you are staring at a failure fraction you cannot explain, setting
`nonlinear_solver_divergence_abort=False` restores exhaust-the-budget behavior exactly and is the
first thing to toggle.

Separately, `perturbation_early_stop="starvation"` (the default) abandons the remaining trials
once too few could still converge to compute any ensemble statistic. **If you see `num_trials`
below `num_planned_trials`, that is deliberate, not a crash or a timeout.** Look for
`early_stopped: True` and `early_stop_reason: "max_attainable_converged_below_minimum"`, plus a
warning naming the truncation; the bootstrap also reports `num_draws_executed`, and
`monte_carlo_counts["num_bootstrap_draws_executed"]` sits next to the planned
`num_bootstrap_draws`. It fires only when `converged_so_far + trials_remaining < 3`, at which
point the first status rung (`indeterminate` below 3 converged solves) is already settled for
every possible completion -- so the status, and therefore the `verdict`, is exactly what running
the full plan would have produced. Every failure fraction and Clopper-Pearson bound is computed
against what actually ran, so they are *wider* than a full plan's, never narrower. It can never
skip more than 2 trials, so it is there for the guarantee and the honest counts, not for speed;
set it to `"off"` if you would rather always execute the whole plan.

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
being on by default. One operational knob worth knowing about: these are full reverse-mode
Jacobians of the whole stacked system (`drift_num_directions * len(drift_path_samples)` of them),
the single most backward-pass-expensive thing in the suite and the pass most likely to exhaust
memory at real study scale, so `DiagnosticConfig.jacobian_row_chunk_size` bounds it with the same
auto policy as the sandwich's own backward pass -- `None` (the default) chunks only once the
output dimension is large enough for it to matter, `0` forces the unchunked `jax.jacrev`, and a
positive int caps the output rows pulled back at once. It changes memory, never numbers.

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
    multiplier_bootstrap="auto",              # recommended: bootstrap verdict when a cheap check flags the run (section 4.6)
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
(`multiplier_bootstrap="always"`, or `"auto"` to let the cheap screens decide) rather than
stop here -- section 4.6. A quiet report plus a passing bootstrap is the strongest thing this
package can say from a single run, but note what is behind each half: the cheap checks' operating
characteristics were measured in ADS-142, while the bootstrap's own validation run is still in
flight (`docs/adr/0002`, "Follow-up experiments"), so its passing verdict currently rests on its
construction -- a self-calibrating null band against a nonlinear re-solve -- rather than on
measured accuracy.

**Q: Isn't the multiplier bootstrap too slow to just turn on?**
A: Not any more, and if you were told otherwise the advice predates the fix that stopped
`analyze_dataset` rebuilding its whole `g_tilde` precompute on every call. Measured locally at
`n=100`/`T=50`, the default 100-paired-draw check is `12.7-21.7 s` next to a `6.6 s` suite; the
same work projects to `~1.8 h` under the old closure. See section 4.6 for the caveats (laptop, not
cluster; fragile-cell re-measurement on the cluster still pending) and `docs/diagnostics.md`
section 3c for the knobs.

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
A: Neither -- they mean different things. `failed` means something concrete and measured broke
(including measured collapses: rank deficiency, total probe censoring, a statistically
confirmed re-solve failure rate). `indeterminate` means the suite couldn't resolve the question
either way (an unstable solve, too few directions, an exceedance not yet statistically
confirmed) -- usually the actionable response is "get more information" (more directions, a
better-conditioned design), not "this is definitely wrong."

**Q: Why is `input_check_results` a separate dict instead of just another entry in
`check_results`?**
A: So a data-hygiene bug can never be mistaken for a statistical finding just by looking at
`classification` or scanning `check_results`. See section 4.1.

## 9. Where these tolerances come from

Most tolerances in `DiagnosticConfig` began as documented engineering guesses; the ADS-142
cluster experiments (`docs/adr/0002-diagnostic-threshold-calibration-plan.md`, results and
correction sections -- 9,883 aggregated runs in the wave-2 grid, with an undercoverage hunt and a
bootstrap-validation run still in flight) have since put real operating
characteristics behind the on-by-default ones. The short version: `n_eff`'s default floor has
measured precision 0.73 / recall 0.59 for 2x-inflated variance (kept warning-level on purpose:
the failures it predicts are conservative); `p_max`'s 0.5 default is essentially inert;
`a_{j,l}`'s 0.10 is a design-severity gauge whose calibrated per-dataset role is the 0.05
bootstrap screen; `check_bread_stability` deliberately keeps `indeterminate` as its ceiling
(a `failed` tier was measured and rejected); the exact check's and bootstrap's ensemble gates
are simulated exact nulls rather than tolerances (`mean_shift_tolerance_se` survives only as the
practical-significance floor under that check's null band); and `backward_residual_tolerance` and
the sum-to-zero displacement gates are correctness smoke detectors pinned to machine behavior.
Read the ADR before changing a default -- it records not just the values but the evidence and the
decisions taken on it, and it is explicit about which of them are measured and which are still
awaiting their experiment.

