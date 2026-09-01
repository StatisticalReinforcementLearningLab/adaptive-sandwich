# 0002. Cluster experiment plan for calibrating enforced diagnostic thresholds

- Status: Wave 2 complete; results and consequent decisions recorded below (2026-08-29);
  follow-up experiments (undercoverage hunt, bootstrap validation) in flight
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
threshold, reusing `clopper_pearson_upper_bound` directly (it lived in `simulator_calibration`
when this was written; that module was removed on 2026-09-01 and the helper now lives in
`lifejacket.helper_functions`).

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
  `docs/diagnostics_tutorial.md` section 5, case E.

## Results (2026-08-29, wave 2 essentially complete)

Wave 2 (jobs 42600365/375/437/439 = Track A easy/medium/hard/hard-regularized,
42600440/442/443 = Track B control/influence/bread at n=100, 42612539/40 + 42616376 = the
n=10000 truth surrogates) delivered 9,883 aggregated runs (`aggregate_ads142_results.py`;
snapshot CSVs `ads142_results*.csv` in `adjusted-sandwich-user` on the cluster). The only
losses: 7 deterministic failures in the influence cell (below), one hard-cell timeout at 8h,
and a residual handful of stragglers immaterial to any conclusion. Analysis followed the
2026-08-28 revisions section (a^NL answer key, continuous variance accuracy, within-cell
correlations).

### Track A

1. **The `se_ratios` `[0.95, 1.05]` gate failed 6,648/6,648 replicates** -- including both
   effectively-linear cells -- empirically confirming the Wishart-null prediction of revision
   #1 at full scale. Every `classification` from these runs reads `failed` and is unusable as
   a label, as anticipated.
2. **The grid came out bimodal.** The hard-regularized cell turned out to be the *most linear*
   cell (`a^NL` median 0.046, 3.6% exceedance at 0.10): huge `lambda_` *damps* adaptivity and
   *improves* bread conditioning (condition number ~56 vs. the control cell's ~1,500), so the
   intended "bread stress" never materialized anywhere in the grid -- the influence cell is the
   only true conditioning stress (up to ~1e14). Medium/hard cells saturate (100% exceedance on
   both checks; `a^NL` medians ~92 and ~4e14, the latter dominated by solver debris -- see
   finding 4).
3. **`a_{j,l}` ranks scenarios essentially perfectly and replicates poorly.** Cell-median
   ordering matches `a^NL` and true variance accuracy exactly; within-cell Spearman vs. `a^NL`
   per-replicate maxima is only 0.17-0.35. Threshold sweep on the borderline cells (n=3,400,
   answer key `a^NL > 0.10`, base rate 25%): the default 0.10 misses 67% of exceedances;
   0.05 misses 11% while flagging 54%; 0.03 misses <1% while flagging 82%. There is no good
   per-replicate operating point -- the ROC is intrinsically shallow.
4. **The exact check's continuation solver collapses exactly where the check would matter**
   (median replicate in the hard cell: ~all 100 directions non-converged; medium: ~25%), and
   the implementation fed non-converged last iterates into every ensemble statistic, which is
   where the 1e14-1e34 "a^NL" values came from. Additionally, per-replicate `a^NL` predicted
   actual per-replicate variance accuracy nowhere (|rho| <= 0.19; <= 0.06 in the cells where
   the solver was healthy). Both checks are scenario-level instruments, not per-dataset
   verdicts; `docs/diagnostics.md`'s "supersedes the cheap check" claim was wrong and has been
   revised to "corroborates at scenario level".

### Track B

5. **The mandatory ground-truth consistency gate produced its anticipated failure, in the
   bread cell**: pool mean vs. n=10000 surrogate disagree at |z| up to 33, because a *fixed*
   `lambda_=500` regularizes ~100x less at n=10000 than at n=100 -- theta* is genuinely
   n-dependent under fixed-lambda regularization, exactly the scenario the plan said to report
   rather than average over. Control passed cleanly (|z| <= 1.4); influence marginal
   (|z| <= 2.9, expected: its recruit_n mechanism cannot be scale-matched). Consequence: the
   bread and influence cells use pool-mean truth; the surrogate is valid only for control.
6. **Nowhere in the 9,883-run grid did inference become anticonservative.** Empirical coverage
   never fell below ~0.94 in any cell. Where things "fail", they fail conservative: the
   influence cell reproduced the paper's variance blow-up (median 1.6-2.2x overestimation, p95
   ~13x, 47% of replicates >2x inflated, coverage 0.94-0.98); the hard Track A cell reached
   ~70x median overestimation with ~0.99 coverage. This has a sharp implication: **no
   anticonservativeness-detecting threshold can be calibrated from this grid at all -- it
   contains zero positive examples.**
7. **`n_eff` is the strongest per-replicate signal the suite has** (within-influence-cell
   Spearman -0.56 vs. per-replicate SE inflation; p_max +0.49; bread sensitivity +0.36). At the
   default floor (`0.1*n` = 10): precision 0.73 / recall 0.59 against >2x inflation. `n_eff<5`:
   precision 0.84 / recall 0.18. `p_max>0.5` (default): fires on 2% of replicates, recall 0.04
   -- effectively inert. Bread sensitivity at a hypothetical `failed` tier (>0.20): recall 0.05.
8. **The 7 influence-cell failures are a standalone finding**: seeds where the first policy
   update (fit on ~10 subjects x 1 decision at steepness 5.0) yields NaN action probabilities,
   crashing `rl_study_simulation.py` at decision time 2 -- the algorithm itself can emit NaN
   probabilities under extreme adaptivity, before any lifejacket code runs. Deterministic per
   seed; rerunning is pointless without an algorithm-side fix. 0.7% informative censoring of
   exactly the most extreme replicates.

### Decisions taken on these results (code landed with this update)

- `check_exact_nonlinear_perturbations`: ensemble statistics now computed on **converged solves
  only**; the fixed `se_ratios` band replaced by a **simulated finite-J null band** for the
  observed converged pattern (`_simulate_perturbation_null_bands`); solver health judged on the
  observed failure fraction (the Clopper-Pearson bound made zero-failure runs read unhealthy at
  practical J); status precedence reworked so censored ensembles can FAIL trustworthily (the
  censoring is optimistic) but can only PASS into `indeterminate`.
- **New `check_multiplier_bootstrap`** (frozen-score generalized bootstrap for Z-estimators,
  reusing the continuation machinery; `docs/diagnostics.md` section 3b): the per-dataset,
  no-simulator verdict this experiment showed `a_{j,l}`/`a^NL` cannot provide. Verdict =
  per-target bootstrap-vs-sandwich SE ratios against their own simulated null band
  (self-calibrating, no engineering tolerance); above-band -> `failed` (anticonservative
  direction), re-solve fragility -> `indeterminate`, below-band -> `warning` (the conservative
  blow-up direction). `multiplier_bootstrap="auto"` uses `a_{j,l}` as the screen
  (`bootstrap_screen_a_jl_threshold=0.05`, the measured 11%-miss / 54%-flag point) -- the
  calibrated role finding 3 leaves for the cheap check.
  On the "no policy replay" concern: the stacked system *is* the replay to first order --
  multiplier perturbations propagate through the eta-equations into reconstructed action
  probabilities along exactly the path the adjusted sandwich's cross-derivative bread terms
  linearize. On representation misspecification (correction 2026-08-31, after a fair
  pushback): the *value-level* fidelity of the stacked model IS detectable from one run, and
  the input checks detect it -- each recorded eta_k must root the claimed update equations on
  the realized data (sum-to-zero per update) and the claimed policy mapping must reproduce the
  recorded action probabilities. The residual blind spot shared by sandwich and bootstrap is
  *derivative-level* fidelity: a rule that agrees at the realized trajectory but responds
  differently to perturbed data (e.g. a clip/projection inactive on the realized path) passes
  both value checks with a wrong response model. Two consequences for the trust protocol:
  (a) the sum-to-zero check needs an SE-standardized tolerance (its absolute atol=0.01 is the
  documented non-portable one that crashed wave 1's hard cell and forced
  --suppress_all_data_checks in every calibration run -- meaning the calibration evidence was
  gathered with these checks off, harmless here since the representations were correct by
  construction, but in production they are Layer 0 and must be on). DONE 2026-08-31:
  `require_estimating_functions_sum_to_zero_se_standardized` measures the residual by the
  SE-standardized displacement it induces (a_j = |(B^-1 r)_j| / SE_j over ALL stacked
  components -- a_root's construction extended beyond the theta targets, with per-update
  attribution; soft 0.01 SE / hard 0.1 SE), and `analyze_dataset` now calls it in place of the
  raw-units version. Field-validated on wave-2 debug_pieces across reward-noise variances
  1/10/100: the raw max residual grew 2e-5 -> 2e-4 -> ~5e-3 (crossing the legacy 5e-4 gate on
  healthy runs), while the standardized statistic stayed flat at 4e-6..8e-5 -- two-plus orders
  of magnitude inside tolerance at every scale; (b) future work: an
  "update replay" check closing the derivative gap to first order -- perturb update-k's input
  data slightly, re-run the actual algorithm update code, compare the realized eta shift
  against the stacked model's derivative prediction.
  The frozen-score approximation differs from the textbook weighted bootstrap at the same order
  as the bootstrap's own error. Empirical falsification designed into the validation run below:
  if the approximation broke at this scale it would undershoot empirical variance in the
  influence cell specifically (where realized-eta randomness is largest).
- `a_{j,l}` deliberately does **not** gain a gating status at any threshold (deferred decision
  #1 from this ADR: answered no on the evidence of finding 3).
- `bread_stability` keeps `indeterminate` as its ceiling (deferred decision #2: answered no --
  finding 7's recall 0.05, and the blow-ups are conservative).
- `n_eff`/`p_max` defaults stay, now documented with their measured operating characteristics
  in `docs/diagnostics.md`.
- `se_distortion_tolerance` was not re-tuned: in the only cell with real variance failure its
  signal is dominated by `n_eff`, and its false-fire cost elsewhere is nil (sensitivities ~1e-4
  in healthy cells).

### Which of this plan's thresholds are still thresholds (2026-08-31)

Two of the quantities this plan named as calibration targets have since stopped being fixed
tolerances, so the corresponding parts of the Track A/B text above and of the Aggregation section
describe fields whose meaning has moved. Recorded here rather than edited into the historical
plan:

- **`mean_shift_se` is no longer judged against a fixed `0.10`.** Track A's answer key (above) and
  revision #1's "secondary keys" both treat `mean_shift_se > 0.10` as a distortion condition. The
  check now compares it against `max(simulated finite-draw null upper quantile,
  config.mean_shift_tolerance_se)` and computes it on complete antithetic pairs only, so
  `mean_shift_tolerance_se` is a practical-significance FLOOR under a self-calibrating band — the
  same treatment `se_ratios` got from finding 1, and for the same reason: at the experiment's own
  `J`, an unpaired `||mean_j(rows)||` is sampling noise of order `sqrt(chi2_dim / J)`, well above
  `0.10`. It therefore joins `se_ratios` in the "not a calibration target" column. (Under intact
  pairing the band is exactly `0` and the floor binds, so the answer key's behavior on the
  paired wave-2 configs is unchanged.)
- **`target_covariance_rank_estimate` now measures the target block, not the joint sandwich.**
  The Aggregation section lists it as a Track B field; `check_bread_stability` now computes it
  (and the new `target_covariance_dim`, plus `target_covariance_eigenvalues`) on `L @ V_hat @ L^T`
  — the theta block by default — instead of on the full joint `V_hat`, whose healthy beta blocks
  kept the rank gate unfireable at any real study scale. Wave-2 rows collected under the old
  meaning are joint-sandwich ranks; re-derive rather than pool across the change.

Neither change affects a decision recorded above: `se_distortion_tolerance` and the `n_eff`/
`p_max` defaults, the only thresholds this experiment actually settled, are untouched.

### Follow-up experiments (submitted from `adjusted-sandwich-user`)

1. **Undercoverage hunt** (`ads142_undercoverage_hunt.sh`): 5 cells x 1000 seeds deliberately
   aimed at anticonservativeness via the levers wave 2 never exercised (n=25; steepness 10 with
   0.01/0.99 clipping = importance-weight tails; `delayed_1_dosage_paper` misspecification;
   staggered recruitment x weight tails at n=50), plus one intermediate-nonlinearity Track A
   cell (1700 seeds) to fill the empty `a^NL` in (0.1, 100) region of finding 3. Pool-mean truth
   only. Either outcome is a deliverable: positive cells calibrate the thresholds that actually
   matter; a failed hunt is a robustness statement about the adjusted sandwich.
2. **Bootstrap validation** (`ads142_bootstrap_validation.sh`): 5 known-truth wave-2 cells x 50
   seeds (same seed range = identical datasets), `multiplier_bootstrap="always"`. Expected per
   cell if the method is sound: easy/control PASS with ratios ~1; medium WARNING-low (~0.82,
   the known 1.5x conservatism); hard INDETERMINATE via re-solve fragility; influence
   below-sandwich ratios tracking the known 1.6-2.2x overestimation -- the last being the
   designed empirical test of the no-replay concern.

### Round-2 results (2026-09-01): both follow-ups complete

321 runs across 9 jobs (`ads142_validation_final.csv`). Two findings, the second of which
supersedes an earlier reading of the partial data.

**1. The no-policy-replay concern is refuted.** The influence cell at 25 draws (job 43197763)
against its own 993-replicate pool (job 42600442, identical design: `delayed_1_action_dosage`,
steepness 5.0, n=100, T=50, clips 0.1/0.9) gives bootstrap-SE-median / pool-truth-SD of
**1.05, 1.32, 0.99, 1.31** across theta_0..3 -- no systematic undershoot, which is the opposite
of what a missing policy replay predicts. On the clean cells the same ratio is 0.96-1.01. The
bootstrap is also far better behaved in the tail than the estimator it audits: its p90/median
is ~1.4x, while the sandwich's median SE of 0.047 sits against a **maximum of 1.1e4**. Below-band
flags track genuine over-inflation (flagged runs' sandwich/truth median 2.09, IQR 1.76-2.69, vs
1.59 unflagged); the 7 above-band trips track nothing real (their sandwich inflation is *lower*
than average), i.e. small-J band noise, consistent with this grid containing no anticonservative
cell.

**2. Most FAILED verdicts in the fragile cells were artifacts of a defective gate, not findings.**
The 2026-08-31 code review confirmed that the mean-shift statistic `b_L` was compared against a
fixed 0.10 with no finite-draw null band, and that its antithetic pairing -- the only thing
suppressing pure sampling noise -- is destroyed by asymmetric convergence censoring. Decomposing
what actually drove each verdict: **24 of the influence cell's 50 failures and 16 of the medium
cell's 34 were mean-shift-only, with no band violation at all.** Clean cells never misfire
(mean-shift median 0.012, max 0.031, all 200 draws converged, pairing intact). Every one of the
50 influence runs has a solver failure fraction above 10% (median 0.56, range 0.12-0.86), so
once solver-unhealthy preempts the location gate that cell reads INDETERMINATE / cannot-certify
rather than FAILED -- matching the hard cell and U5, both at 100% solver failure.

Per-cell verdicts as recorded *before* the fix, for the record: clean cells 45/50 and 43/47
passed; medium 34/36 failed; hard 34/39 indeterminate; influence 50/50 failed; U5 36/42
indeterminate. The fix landed in `check_multiplier_bootstrap` is the simulated null band on
`b_L` plus solver-health preemption -- **not** the coverage-derived ~0.5 SE retune this ADR
speculated about earlier, which addressed the symptom rather than the missing band.

## Addendum (2026-09-01): decision-level verdict field

`DiagnosticReport` gains an additive `verdict` (`certified` / `conservative` / `uncertifiable`
/ `invalid`) plus `verdict_basis` (`bootstrap` / `screen`) -- the single-run trust protocol
these experiments calibrated, made machine-readable (`_derive_verdict` in `diagnostics.py`;
vocabulary and evidence in `constants.DiagnosticVerdicts`). `classification` is unchanged and
WARNING-blind by design; the verdict makes the calibrated conservative tier and the
rank-deficiency collapse mode (pulled up to `invalid`) visible to automated consumers.
`aggregate_ads142_results.py` extracts both columns, tolerant of pre-field reports.

## Addendum (2026-09-01): re-solve performance work

Recorded here because it touches the instrument this ADR's answer key is measured with, and
because the runtime figures that motivated it are no longer current.

**The premise moved before the work started.** The multi-hour bootstrap runs behind the ADS-142
cluster jobs (whole-task medians of 3.3-3.6 h on the clean cells, 6.4-11.5 h on the fragile ones,
against a 5 min no-bootstrap control) predate the fix that builds `post_deployment_analysis`'s
diagnostics `g_tilde` closure once and jits it, instead of rebuilding the whole `O(n*T)`
structural precompute on every call. Measured in-process against a faithful reconstruction of the
pre-fix closure at `n=100`/`T=50`/`eta_dim=200`: `797 ms` per evaluation then, `1.4-2.6 ms` now.
A 100-paired-draw check makes 8,027 evaluations, so the old closure projects `~1.8 h` for the
check alone against the cluster's `~3.5 h` marginal for that cell -- close enough, at ~2x for a
slower cluster core, that nothing else is needed to explain those rows. **Those per-cell hours
are history; do not quote them as current cost.** (Separately, and unaffected by any of this: the
n=10000 walltime finding in "Analysis-plan revisions", item 8, is about the input-check and
main-analysis passes, not the bootstrap.)

**Measured cost structure on the current code** (local: Apple-silicon laptop CPU, JAX CPU backend,
float32; ratios transfer, absolute seconds may move ~2x on the cluster):

- 100 paired draws (200 trials, 8,027 evaluations): `12.7 s` and `21.7 s` in two independent
  harnesses -- identical evaluation counts, so the spread is machine load. 25 draws: `3.5-5.4 s`.
  The suite with the bootstrap off: `6.6 s`.
- `g_tilde` is 94-97% of the check; the cost model is exactly
  `(trials) x (chord iterations) x (one g_tilde evaluation + one triangular solve)`.
- Two premises in the original brief were wrong and are recorded so they are not re-derived: a
  failing solve does **not** burn the `continuation_steps x max_iterations` ceiling (the
  continuation exits at the first step that exhausts its budget; no observed solve exceeded 336
  iterations), and cost peaks at *partial* fragility, not maximum -- 8,027 evaluations at a 0%
  failure fraction, ~24,700 at 32%, ~21,600 at 97.5%, ~12,200 at 100%.

**What landed.** Three changes, all in `lifejacket/diagnostics.py`:

1. *Divergence abort* (`nonlinear_solver_divergence_abort=True`, with `blowup_factor=1e4`,
   `stall_window=40`, `guard_factor=100.0`). Aborts a continuation step that blew past its own
   best relative residual or stalled; an aborted solve is failed on the same control-flow path an
   exhausted budget takes, so it is observationally identical to one downstream. Calibrated on
   converging steps, since a false abort is the only way it can move an answer: zero false aborts
   across ~152,000 converging steps in two independent corpora (33,351 on this repository's
   fixtures, 118,563 synthetic), worst observed blow-up ratio `1.9`/`36.7` and worst stall run
   `4`/`20`, i.e. ~270x and 2x margins. Measured saving, as a fraction of a check's `g_tilde`
   evaluations: 0% below a 0.6 failure fraction, 0.1% from 0.6 to 0.95, and 20-66% median
   (max 96%) at or above 0.95 -- i.e. it does nothing for the influence cell and everything it
   does is in the hard/U5 regime. A
   per-step iteration cap and a rate-extrapolation clause were both built, measured, and rejected
   (the cap reclassifies converging steps; the rate clause had zero false aborts in-sample and
   three out of sample).
2. *Sequential early stop* (`perturbation_early_stop="starvation"`). Stops when
   `converged_so_far + trials_remaining < 3`, the point at which the first status rung is settled
   for every completion. Provably status-preserving, and provably capped at 2 skipped trials
   (~1% at 100 paired draws, 4% at 25). On the 292 field runs with a bootstrap result it would
   have fired on 70, all in the hard and U5 cells, for a mean saving of 0.61% of trials. It does
   not compose with (3): batching can only test the predicate at a wave boundary, so at any wave
   width above 2 the stop never fires. Irrelevant for this experiment's own runs, which should
   leave batching off for the reason in (3) and therefore keep the stop live.
3. *Batched lockstep re-solves* (`batched_bootstrap_resolves="off"` by default, with
   `bootstrap_batch_width=50`, `bootstrap_batch_min_rows=4000`). ~1.5-1.75x end to end at 100
   draws, ~2.6x excluding a one-time ~5-6 s XLA compile. Verdict-equivalent but not bit-identical
   (XLA picks different float32 kernels per batch width: ~2.2 ULPs on `g_tilde`, ~1e-7 relative on
   the SE ratios, ~1e-5 on the cancellation-dominated `mean_shift_se`), which is why it is opt-in
   rather than merely guarded -- **this experiment's answer key is exactly the FAILED /
   INDETERMINATE distinction, so calibration and validation runs should leave it off.**
   End to end at 100 draws on the clean fixture it left `status`, the warnings and all 200
   per-trial convergence flags identical.

**Why the ADS-142 answer key is preserved.** `_derive_verdict` is a pure function of the check
statuses (plus `bread_stability`'s target rank and `local_nonlinearity`'s headline `a_{j,l}`), so
anything that preserves per-check status preserves `verdict`/`verdict_basis`. The early stop
preserves status by construction; the abort is status-preserving iff it never aborts a solve that
would have converged, which is what the margins above buy. The stronger stop the brief proposed --
"stop once the solver is certainly unhealthy" -- was **rejected on this ADR's own data**:
unhealthiness only confines the status to `{FAILED, INDETERMINATE}`, because the FAILED rung sits
above the unhealthy rung so that a distortion measured on the optimistically-censored converged
subset still counts. All 50 influence-cell runs were certainly unhealthy from the first trial
(failure fractions 0.12-0.86) and all 50 nevertheless reported FAILED from that subset; across the
292 runs the rule would have relabelled 119 (41%) from FAILED to INDETERMINATE, i.e. `invalid` to
`uncertifiable`.

**Honest reporting.** Both checks now report `num_planned_trials`, `early_stopped`,
`early_stop_reason` and `num_divergence_aborted_trials`; `num_trials` is the executed count. The
bootstrap adds `num_draws_executed` and `resolve_batch_width`, and
`DiagnosticReport.monte_carlo_counts` adds `num_bootstrap_draws_executed`. One reported metric
genuinely moves: a solve that would have gone non-finite *later* is now a root failure rather than
a domain failure, so `domain_failure_fraction`/`_upper_bound` are lower bounds when
`num_divergence_aborted_trials` is nonzero. No status rung reads either, but do not pool those
columns across this change.

**Pending, and not to be quoted as measured:**

- Every number above is local. No cluster re-baseline of a clean and a fragile cell on the current
  code has been run; extrapolating the historical per-cell hours by the measured ~500x collapse
  (3.5 h -> ~25 s, 11.4 h -> ~82 s) assumes those cells' cost was likewise ~all `g_tilde` rebuild,
  which is verified only on the clean cell.
- No end-to-end run on a real ADS-142 fragile cell. The influence-cell fixture could not be pushed
  through `analyze_dataset` locally -- it stalled on XLA compilation of
  `compute_local_linearization_error_ratio`'s jit, on the *main analysis* path, not the
  diagnostics one. At staggered/many-policy shapes some residual cluster wall clock may be
  one-time compilation that no bootstrap change can touch; attribute it by measurement before
  assuming otherwise.
- The batched path's verdict-equivalence is measured on clean fixtures. The case that matters -- a
  fragile ensemble with solves near the convergence boundary, where a ~5e-7 perturbation could
  flip a convergence flag -- is unmeasured. That is the second reason it is off by default.
- The abort's savings figures come from synthetic sweeps and this repository's fixtures; the
  failure-*mode* mix (blow-up caught immediately vs. slow creep never caught) is what drives them
  and could differ on the real stacked estimating function.
- Not taken: `DiagnosticConfig.g_tilde_chunk_size` is still `1`, and
  `evaluate_g_tilde_batched` still constructs its `jax.vmap` closure inside the chunk loop, so it
  retraces per chunk. Measured 10-42x on that helper alone and 19% off the non-bootstrap suite at
  `n=100`/`T=50` (`6.54 s` -> `5.29 s`), with classification, verdict and `local_nonlinearity`
  status unchanged. It is orthogonal to all three changes above and is the cheapest remaining win;
  it needs a test pinning `check_local_nonlinearity`'s status and the report's
  `verdict`/`verdict_basis` across chunk sizes, since float32 reassociation moves `a_{j,l}` by
  ~1e-6 relative and the `0.05` screen is what `verdict_basis` reads.
