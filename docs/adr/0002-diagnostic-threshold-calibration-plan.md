# 0002. Cluster experiment plan for calibrating enforced diagnostic thresholds

- Status: Proposed
- Date: 2026-08-27
- Ticket: ADS-142

## Context

`lifejacket.diagnostics` (see `docs/diagnostics.md`) ships a layered suite of checks, most of
which now run by default (`b28200d` turned on all but the most expensive). Every numeric
tolerance in `DiagnosticConfig` (`nonlinear_correction_tolerance_se=0.10`,
`root_error_tolerance_se=0.01`, `se_distortion_tolerance=0.05`,
`influence_n_eff_min_floor`/`influence_n_eff_min_fraction`/`influence_p_max_tolerance` in
`check_influence_concentration` (recently promoted from hardcoded literals to overridable fields,
same defaults), etc.) is documented as an
"engineering tolerance, not a theorem-derived critical value, unless a particular deployment has
calibrated them via `lifejacket.simulator_calibration`." None of them have actually been
calibrated yet. This plan designs a cluster experiment, using `adjusted-sandwich-user`'s existing
SLURM infrastructure, to do that — and separately, to decide which checks are strong enough to
actually *gate* the classification (vs. only surface a warning) once calibrated.

**Scope for this round:** `a_{j,l}` (`check_local_nonlinearity`), `n_eff`/`p_max`
(`check_influence_concentration`), and `se_distortion_tolerance`/`rank_tolerance`
(`check_bread_stability`) — the three checks that are both on-by-default and have a numeric
tolerance a caller could plausibly want enforced. `check_root_and_implementation` is excluded
(it's a correctness/solve-residual check, not a statistical calibration question).
`check_exact_nonlinear_perturbations`/`check_jacobian_drift` stay opt-in/expensive and are used
here only as an *instrument* (Track A below), not as a calibration target themselves.

**A note on `_combine_classification`:** today, a `WARNING` status (which is all `a_{j,l}` and
`check_influence_concentration` can ever produce) never changes the returned classification away
from `locally_supported` — only `FAILED`/`INDETERMINATE` do. So calibrating these tolerances is
necessary but not sufficient for "enforcing" them; actually gating on them is a follow-up code
change to `diagnostics.py`'s status logic, to be made *after* this experiment produces numbers,
not before. This plan only produces the calibration data.

## Two independent validation tracks

These three checks split into two genuinely different validation problems, and conflating them
would make the design worse.

### Track A — `a_{j,l}` vs. the exact nonlinear check (no ground truth needed)

`check_exact_nonlinear_perturbations` measures the *same thing* `a_{j,l}` approximates
(SE-ratio distortion, mean shift, quantile shift) but exactly, via continuation, rather than via a
one-step Taylor correction — `docs/diagnostics.md` already calls it out as the thing that
"supersedes the cheap `a_{j,l}` when available." That means Track A needs **no repeated-sampling
ground truth and no notion of a true theta\***: on a single simulated dataset, run both checks and
ask whether the cheap check's pass/fail at a candidate threshold agrees with what the expensive
check says is actually true.

Mechanics: one array task = one simulated dataset (one seed), diagnostics run with
`compute_exact_nonlinear_roots=True` and `num_exact_directions>=300` (the value
`docs/diagnostics.md` says is needed for the exact check's own ~1% Clopper-Pearson bound). From
the resulting `diagnostic_report.pkl`, pull `local_nonlinearity`'s per-radius `a_by_target`
quantiles and `exact_nonlinear_perturbation`'s `se_ratios`/`mean_shift_se`/`quantile_shifts_se`.
Define the exact check's own "material distortion" condition (`se_ratios` outside `[0.95, 1.05]`,
or `mean_shift_se`/`quantile_shift` above `0.10`) as the answer key, and, offline, sweep candidate
`nonlinear_correction_tolerance_se` values against it to get a false-negative-rate curve (cheap
check says fine, exact check says it isn't) with a Clopper-Pearson upper bound at each candidate
threshold, reusing `simulator_calibration.clopper_pearson_upper_bound` directly.

This needs real variation in *how nonlinear* the estimating equation actually is across scenario
cells — a grid where every cell is comfortably linear can't calibrate anything.

### Track B — influence-concentration / bread-stability vs. coverage (needs ground truth)

`n_eff`/`p_max` and `se_distortion_tolerance` are claims about the actual sampling distribution
across repeated realizations of the same design, not about one dataset — Track A's trick doesn't
apply. This needs real coverage: many replicate seeds per scenario cell, a definition of ground
truth theta\* for that cell, and per-replicate coverage indicators correlated against each
replicate's `n_eff`/`p_max`/`se_distortion` values, swept the same ROC-style way as Track A.

**Added scope: does `bread_stability`'s existing `indeterminate` deserve a `failed` tier?**
`check_bread_stability` can only ever return `passed` or `indeterminate` today — it has no path to
`failed` at all (`numerical_sensitivity_max_relative_se_change > se_distortion_tolerance` sets
`indeterminate`, never anything stronger). Whether a *more severe* SE-sensitivity value should
instead mean "this is actually wrong" rather than merely "we can't tell" is exactly a coverage
question: sweep a second, higher candidate threshold on `numerical_sensitivity_max_relative_se_change`
against the same per-replicate coverage indicators used for `n_eff`/`p_max`, and check whether
crossing it actually predicts coverage failure (as opposed to just "very fragile but still fine").
If it does, that's the calibrated severity level to gate `failed` on; if it doesn't, the current
`indeterminate`-only ceiling is already the right answer and shouldn't be hardened further.

**Ground truth (per your steer):** use both of the following, and treat their agreement as a
go/no-go gate before trusting either:

1. **Cross-replicate Monte Carlo mean**, at the *same* `n`/`T` as the test cell — run a large pool
   of independent seeds, average `theta_hat` across the pool. Simplest version uses the same pool
   for both the mean and the per-replicate coverage test (standard simulation-study practice,
   slightly optimistic if the pool is small); if you want strict separation instead, split the
   pool in half.
2. **Large-`n`, same-`T` surrogate** — a separate run (or small MC-averaged handful) at much larger
   `n`, *same* `T` — per your note that the asymptotic theory is fixed-`T`, `n -> infinity`, `T`
   should not be increased alongside `n`.
3. **Consistency check (not optional):** compare (1) and (2) for at least one scenario cell before
   trusting either downstream. You've already observed the result sometimes depends on `n`, which
   is exactly the thing that would break using either as "the" ground truth silently. If the two
   estimates disagree by more than the test-cell's own sampling SE, that's a finding to report and
   investigate — e.g. whether this deployment family's theta\* is genuinely `n`-dependent in a way
   the fixed-`T` asymptotics don't capture — not something to average over or paper past.

## Scenario grid

Both tracks reuse `rl_study_simulation.py`'s existing misspecification knobs (already exposed by
`adjusted-sandwich-user`'s `run_and_analysis_parallel_synthetic.sh` and already used for a
similar "stress the estimator" sweep in `synthetic_blowup_cases.sh`):

| Axis | Knob | Values to span |
| --- | --- | --- |
| Scale | `--n` | e.g. 50, 100, 500, 1000 (Track A); a couple of representative values (Track B, cost-limited) |
| Adaptivity / nonlinearity | `--steepness` | 0.5, 1.0, 5.0 |
| Structural misspecification | `--synthetic_mode` | `no_effect`, `delayed_1_action_dosage`, `delayed_5_action_dosage`, `delayed_1_dosage_paper` (true effect enters somewhere the fitted model doesn't expect) |
| Noise scale | `--synthetic_reward_noise_var` | 1, 10, 100 |
| Regularization / adaptivity damping | `--lambda_` | default, and a "huge regularization" cell (500.0), mirroring `synthetic_blowup_cases.sh`'s own framing of that as a deliberate adjusted-sandwich stress test |
| Noise structure | `--err_corr` | `time_corr`, `independent` |

Track A: full or near-full grid x ~20-50 seeds per cell (seeds don't need to share a ground truth,
each is an independent (cheap-check, exact-check) pair for the ROC curve — more is better, cost
permitting).

Track B: a **small** number of representative cells (e.g. 2-4 combinations spanning "easy" and
"hard"), each needing a much bigger seed pool (see resourcing below) — do not run this at the full
Track A grid's cardinality, the replicate multiplier makes that prohibitive.

## Cluster mechanics (`adjusted-sandwich-user`)

- Base job: `run_and_analysis_parallel_synthetic.sh` (already the right simulator — has all the
  misspecification knobs above, already passes `--run_diagnostics`/`--diagnostic_config_pickle`
  through to `lifejacket analyze`). One array task = one seed, `SLURM_ARRAY_TASK_ID` already used
  as `--parallel_task_index`.
- Grid submission: follow `paper_simulation_cases.sh`/`synthetic_blowup_cases.sh`'s existing
  pattern — a new driver script with one `sbatch --array=1-K ...` line per scenario cell, K =
  seeds-per-cell (small for Track A, large for Track B).
- Two new pickled `DiagnosticConfig`s needed, generated by a small one-off script and passed via
  each cell's `--diagnostic_config_pickle`:
  - Track A config: `compute_exact_nonlinear_roots=True`, `num_exact_directions=300`,
    `paired_directions=True`.
  - Track B config: defaults are already right (`compute_influence_and_overlap_checks=True` is
    already the default; `compute_exact_nonlinear_roots` stays `False` — Track B doesn't need it
    and it would make each of the many replicates far more expensive for no purpose).
- Output convention: reuse the existing
  `/n/netscratch/murphy_lab/Lab/nclosser/adjusted_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/`
  layout, so `diagnostic_report.pkl`/`analysis.pkl` land where they already do.

**Blocking prerequisite, found while investigating this:** all four `run_and_analysis_parallel_*.sh`
scripts still do `pip install -r requirements.txt`, but that file was deleted from
`adjusted-sandwich-user` in the local uv migration (`461b3ab`) and never replaced in the cluster
scripts — they will fail on a fresh `~/venv` right now, independent of anything in this plan. This
is the same "convert `adjusted-sandwich-user` to uv too" item already flagged as deferred
(`[[project_uv_migration]]`), except it's no longer a someday-cleanup: it now blocks submitting
*any* new array, including this one. Needs a decision on how to fix it before this experiment runs
(regenerate a frozen `requirements.txt` via `uv export` as a quick unblock, vs. actually switching
the cluster scripts to `uv` — the latter needs the "can the cluster module setup run `uv`" check
that memory item already flagged as not yet investigated).

## Aggregation

`collect_existing_analyses.py` (the existing aggregator) never reads `diagnostic_report.pkl` at
all today — it only rolls up `analysis.pkl`/`debug_pieces.pkl` into terminal plots, no
CSV/DataFrame export. Needed for this experiment: a new small aggregator (can reuse its
`glob.glob(input_glob)` + per-file load-and-discard skeleton) that, per array task, pulls the
scenario parameters (from the output-folder-name convention) plus the specific fields needed for
the offline threshold sweep — `local_nonlinearity`'s per-radius `a_by_target` quantiles,
`exact_nonlinear_perturbation`'s `se_ratios`/`mean_shift_se`/`quantile_shifts_se` (Track A),
`influence_concentration`'s `n_eff`/`p_max` and `bread_stability`'s
`numerical_sensitivity_max_relative_se_change`/`target_covariance_rank_estimate` (Track B), plus
`theta_est`/`theta_variance_estimate` — into one tidy CSV/DataFrame. The actual threshold-sweep /
ROC-curve analysis happens offline, locally, against that CSV — not on the cluster.

## Resourcing — pilot first

Track A's cost per cell scales with `n` and `num_exact_directions` (300 directions x up to 50
chord-Newton iterations x `continuation_steps=10`, each iteration one `g_tilde` evaluation).
Track B's cost per *cell* is cheap per replicate but multiplied by however many replicates the MC
ground truth needs. Given this session's own recent history of a real OOM at oralytics scale
(see `[[project_ads139_performance_audit]]`), the plan is: run a small pilot (a handful of Track A
cells at `num_exact_directions=50`, one Track B cell at ~100 replicates) first, measure actual
walltime/memory, and size the full array's `--mem`/`-t` from that — not from guessing.

## Open decisions still needed from you

1. **The requirements.txt blocker** — quick unblock (regenerate a frozen file) vs. investigate
   switching the cluster scripts to `uv` now.
2. **Track B's representative cells** — which 2-4 scenario combinations count as the
   must-calibrate "easy"/"hard" cases (I've drafted candidates above from the existing
   `synthetic_blowup_cases.sh` framing, but you know which real deployments this needs to speak
   to).
3. **Replicate pool size for Track B's MC-mean truth**, and whether you want the same-pool
   (simpler, slightly optimistic) or split-pool (stricter) version.
4. Whether to run the pilot now, or go straight to writing the full driver scripts /
   `DiagnosticConfig` pickles / aggregator first and size the array afterward.

## Follow-up decisions this experiment is meant to unblock (not decided yet)

Two concrete status-logic changes to `diagnostics.py` are deliberately **not** being made now —
they're the reason this experiment exists, not a parallel task:

1. **Should `a_{j,l}` exceeding a validated threshold actually gate classification, not just
   warn?** Today it can only ever produce `WARNING` (see the `_combine_classification` note
   above), which never moves `classification` away from `locally_supported` — the only thing that
   currently escalates `local_nonlinearity` to something that gates (`indeterminate`) is the
   unrelated joint-Mahalanobis rank-deficiency condition, not the size of `a_{j,l}` itself. Once
   Track A produces a calibrated `nonlinear_correction_tolerance_se`, wiring an exceedance of it
   into a real gating status is the natural next step.
2. **Should `bread_stability`'s fragile-SE `indeterminate` escalate to `failed` past a second,
   higher threshold?** See "Added scope" under Track B above — this is now an explicit part of
   the coverage analysis, not just a hypothetical.

Both are deliberately deferred until this experiment has numbers, so neither threshold gets
hand-picked the same way we're trying to avoid for `a_{j,l}` in the first place.

## Not in scope here

- `check_exploration_and_weights`'s floor/ceiling — those are supplied by the caller from the
  deployment's real design bounds, not something a simulator sweep calibrates.
- `check_root_and_implementation` — a correctness check, not a statistical calibration target.
  Its `backward_relative_residual` sub-check *did* get a hard gate
  (`backward_residual_tolerance`, default `1e-6`), but outside this experiment entirely and for a
  different reason: it's pinned to float64 machine epsilon regardless of `B_hat`'s conditioning
  (confirmed empirically across condition numbers spanning twelve orders of magnitude), so it's a
  software-correctness smoke detector, not an engineering tolerance needing simulator data — see
  `docs/diagnostics_tutorial.md` section 5, case F.
