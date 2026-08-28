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
- Two pickled `DiagnosticConfig`s, generated by `make_diagnostic_configs.py` and passed via each
  cell's `--diagnostic_config_pickle`:
  - Track A config: `compute_exact_nonlinear_roots=True`, `num_exact_directions=50` (not the
    `>=300` that would give the exact check's own tightest ~1% Clopper-Pearson bound on any
    *individual* replicate — deliberately looser here, since this run's informativeness comes
    from the volume of independent (cheap-check, exact-check) pairs pooled across ~6800
    replicates, not from any single replicate's own bound being maximally tight; revisit if the
    pooled analysis turns out to need tighter per-replicate bounds after all).
  - Track B config: defaults (`compute_exact_nonlinear_roots` stays `False` — Track B doesn't
    need it and it would make each of the many replicates far more expensive for no purpose).
- Output convention: reuse the existing
  `/n/netscratch/murphy_lab/Lab/nclosser/adjusted_sandwich_simulation_results/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}/`
  layout, so `diagnostic_report.pkl`/`analysis.pkl` land where they already do.
- Driver scripts: `ads142_track_a.sh` (4 cells x 1700 seeds = 6800 tasks) and `ads142_track_b.sh`
  (3 cells x (1000 @ n=100 + 30 @ n=10000) = 3090 tasks), both in `adjusted-sandwich-user`. Sized
  together to total 9890 — under a hard 10k-submitted-at-once ceiling on this account.

**Blocking prerequisite, found while investigating this — resolved, but not the way this section
originally proposed.** All four `run_and_analysis_parallel_*.sh` scripts did `pip install -r
requirements.txt`, but that file had been deleted from `adjusted-sandwich-user` in the local uv
migration (`461b3ab`) and never replaced in the cluster scripts. Rather than switching the cluster
itself to `uv` (which would have needed confirming the cluster's module setup can actually run it
— never resolved), the fix followed an already-proven pattern from another project
(`cosmobench_sur_lab`): a real, checked-in `requirements.txt` with `-e ../adaptive-sandwich` (pip's
native local-editable-install syntax, not a uv construct) plus the same pins already in
`pyproject.toml`, so the existing plain pip/venv cluster flow works completely unmodified. This
means `adaptive-sandwich` itself now also needs to be deployed to the cluster, as a sibling
directory — a new `deploy.sh` was added there (mirroring `adjusted-sandwich-user/deploy.sh`'s
exact rsync pattern) targeting `/n/home08/nclosser/adaptive-sandwich/`. Confirmed working via a
real small end-to-end job. One consequence worth flagging: `deploy.sh` rsyncs whatever's on disk,
uncommitted changes included — no `git push`/merge was needed to get today's changes onto the
cluster, but any further local edits need a re-deploy before they reach it.

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

## Analysis-plan revisions (2026-08-28, after the first submission wave)

A critical re-review of the design after the first wave surfaced one genuine flaw and several
analysis improvements. None require re-running anything — the data being collected already
supports the corrected analysis — but the analysis must follow this section, not the original
Track A/B text above, where they conflict.

1. **Track A's answer key is `a^NL` (per-direction exact-vs-linear root displacement), NOT the
   `se_ratios` band, and NOT `report.classification`.** Confirmed by direct null simulation: under
   a *perfectly linear* estimating map, the exact check's `se_ratios` are sqrt generalized
   eigenvalues of a sample covariance built from `num_exact_directions` antithetic pairs —
   effective d.o.f. J, sqrt-eigenvalue noise ~ `0.5*sqrt(2/J)` per eigenvalue, maxed over
   `dim(theta)` eigenvalues. Measured null failure rates of the `[0.95, 1.05]` gate: **100% at
   J=50** (the experiment's setting), 99.9% at 150, 98.6% at 300, still 55% at 1000. So nearly
   every Track A replicate will read `failed` on `se_ratios` regardless of actual linearity —
   pure label noise. `a^NL` has no such problem: it compares each direction's continuation-solved
   root against its own linear guess, paired, and is ~0 under linearity (confirmed ~1e-8 on the
   affine test fixture whose `se_ratios` "failed" at 0.47). The aggregator now extracts
   `a_nl_{median,p95,max}` per target; the threshold sweep for
   `nonlinear_correction_tolerance_se` runs `a_{j,l}` against `a^NL` exceedance.
   `mean_shift_se`/`quantile_shifts` remain secondary keys (they average over directions, so
   their noise shrinks with J, unlike the eigenvalue extremes).
2. **Follow-up code fix (post-experiment, deliberately not mid-flight):** the `se_ratios`
   `[0.95, 1.05]` hard-fail gate in `check_exact_nonlinear_perturbations` is miscalibrated for
   any practical `num_exact_directions` and should be replaced with J- and dim-dependent null
   quantiles (simulate the Wishart null, as above) or dropped in favor of the mean-shift/
   quantile-shift gates. Not changed now to avoid forking the running experiment's config.
3. **Track B's primary outcome is continuous variance accuracy, not binary coverage.** Per
   replicate: `log(adjusted_var_jj / empirical_var_jj)` where the empirical variance comes from
   the cell's own replicate pool. Binary 95%-coverage failures are ~Bernoulli(0.05) under the
   null (~50 events per 1000 replicates — weak power), and the blow-up phenomenon the thresholds
   exist to catch is exactly an extreme-variance-estimate phenomenon that coverage alone provably
   obscures (a paper-documented failure mode: 100% coverage with estimates 1000x too large).
   Coverage stays as the secondary outcome.
4. **Within-cell correlation analyses are required, not optional.** The influence-stress cell
   (recruit_n=10) also stresses the linearization and bucket structure, so cross-cell contrasts
   confound mechanisms; the calibration question ("does n_eff predict failure?") must be answered
   within cells across replicates, with cross-cell as supporting evidence only.
5. **Informative censoring from timeouts.** Timeouts correlate with nonlinearity (harder
   replicates take more chord-Newton iterations), so analyzing only completed tasks censors
   exactly the most-nonlinear tail — biasing the calibration optimistic. Per-cell completion
   rates must be reported alongside results; residual non-completions at the raised 8h limit are
   findings, not omissions.
6. **Null-calibration byproduct:** the easy cell's `se_ratios` distribution across ~1700
   replicates doubles as an empirical measurement of the exact check's false-positive rate at
   J=50, validating (or correcting) the Wishart prediction above with real pipeline data.
7. **n=10000 contingency:** if the memory probe shows the surrogate is impractical even at high
   `--mem`, fall back to n=2000 x 100 replicates (comparable total subject-count for the truth
   estimate, already within known-safe memory territory).
8. **n=10000 probe results and consequent large-n design (probe job 42575297, 96G, TIMEOUT at
   1h):** MaxRSS ~27 GiB on both tasks (so `--mem=48G` is right, not 16G or 96G); simulation
   ~20 min + analysis ~10 min, `analysis.pkl` written by ~40 min. The timeout happened inside
   the diagnostic suite: the action-probability reconstruction input-check alone ran **>28
   minutes without finishing** at n=10000 -- a new, standalone scalability finding about the
   default-on suite (never profiled at this scale; the n=100 cells are unaffected).
   Consequences, all applied: (a) the large-n surrogate cells run `--run_diagnostics=0` -- they
   exist only for the truth cross-check's mean theta, their diagnostics are never used by the
   analysis plan, and skipping the suite removes the walltime hazard entirely (aggregator
   updated to emit theta/se-only rows for report-less runs, `has_diagnostic_report` column
   added); (b) two bugs found in the influence-stress large-n cell as originally scripted:
   `recruit_n=10` copied verbatim from the n=100 cell would recruit only ~500 of 10000 subjects
   within T=50 -- the surrogate must scale recruitment proportionally (`recruit_n=1000`, same
   10 waves of 10%) -- and that staggering produces ~490 shape-buckets, the exact trigger for
   the local-linearization diagnostic's JIT-compile blowup (24.7 min at n=100), which runs
   regardless of `--run_diagnostics`; so that cell is probe-first (2 tasks @ 96G/4h,
   `ads142_track_b_large_n_influence_probe.sh`) before its 30-seed submission.

## Resourcing — decided

Track A's cost per cell scales with `n` and `num_exact_directions` (50 directions x up to 50
chord-Newton iterations x `continuation_steps=10`, each iteration one `g_tilde` evaluation).
Track B's per-cell cost is cheap per replicate but multiplied by however many replicates the MC
ground truth needs. Given this session's own recent history of a real OOM at oralytics scale
(see `[[project_ads139_performance_audit]]`), a *separate* smaller pilot before the real run was
considered — but a SLURM array reports timing/memory for its earliest-completing tasks well
before the whole array finishes, so `ads142_track_a.sh`/`ads142_track_b.sh` (the real runs) double
as their own pilot data; no separate smaller submission first. `--mem`/`-t` for Track A
(`20G`/`0-2:00`) are still unvalidated guesses at the time of submission — worth confirming against
the earliest-completing tasks before assuming the rest of the 6800-task array will behave the
same way. Track B's large-`n`=10000 cells (`16G`/`0-1:00`) are grounded in real prior profiling of
this exact simulator at that scale (`[[project_ads139_performance_audit]]`: `construct_classical_
and_adjusted_sandwiches` ~49s, `local_linearization_diagnostic` ~139s, no OOM); its `n`=100 cells
(`8G`/`0-0:30`) match the already-validated "medium" benchmark scale.

## Decisions made (previously "Open decisions still needed from you")

1. **The requirements.txt blocker** — resolved via the pip-editable-sibling-repo approach
   described above, not a `uv` conversion.
2. **Track B's representative cells** — 3, not the drafted 2-4: a control (`no_effect`,
   `steepness=0.5`, well-identified, should mostly `pass` — true-negative examples for the
   ROC-style curve), an influence-concentration stress cell (`delayed_1_action_dosage`,
   `steepness=5.0`, **`recruit_n=10`** — a small early-recruitment batch disproportionately
   shapes the whole policy trajectory, the mechanism most likely to actually produce low
   `n_eff`/high `p_max`), and a bread-conditioning stress cell (reusing Track A's
   `lambda_=500.0` "huge regularization" cell).
3. **Replicate pool size / same-pool vs. split-pool** — same-pool, explicitly not split: the
   same-pool optimism bias for any one replicate is `O(1/N)` (its own weight in the pool mean),
   while the pool mean's own Monte Carlo uncertainty is `O(1/sqrt(N))` — for `N` large enough to
   make the ground truth trustworthy at all, the bias is already dominated by uncertainty the
   design has to live with anyway, so split-pool would spend replicates fixing a second-order
   problem. Sizes: **1000** replicates/cell at test-`n`=100 (serves as both the truth pool and
   the coverage-test pool), **30** replicates/cell at the `n`=10000 surrogate (a single run at
   that scale already has ~10x tighter sampling noise than at `n`=100, so far fewer replicates
   give an equally tight cross-check mean — the lever here is `n`, not replicate count).
4. **Pilot vs. go straight to the real run** — went straight to the real run; see "Resourcing"
   above for why a separate pilot wasn't needed after all.

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
