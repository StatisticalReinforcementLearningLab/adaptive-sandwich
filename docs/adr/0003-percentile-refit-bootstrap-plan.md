# ADR 0003: Percentile refit bootstrap for adaptive-sandwich confidence intervals

Status: implemented (2026-08-31 — `analyze_dataset(..., percentile_bootstrap_draws=...)` /
`--percentile_bootstrap_draws`; acceptance gate §4 passed: weights=1 identity holds, the
lifejacket refit matched `two_stage_refit_bootstrap.py`'s closed forms per-draw to median
relative difference 2.4e-8 / max 1.2e-4 on a shared-seed n=200 run (300/300 draws converged),
and the negative control — RN weights omitted — disagreed by a median 8e-3, ~3e5x larger,
confirming the comparison detects the §2 failure mode. Implementation notes: multiplicity
generation is pinned to `np.random.default_rng(seed).poisson(1.0, size=(draws, n))` in
subject-ids order so references can reproduce draws from the seed; the per-draw Newton
re-differentiates at every iterate (a chord solve on the original bread was rejected — the
draws that matter move where the Jacobian changes); non-converging draws are dropped and
counted with a reason taxonomy (`singular_jacobian` at small n is usually a draw that zeroed
out an early recruitment wave, leaving that update block genuinely unidentified); the
structural precompute is built once and the (stack, Jacobian) evaluation is jax.jit-compiled,
mirroring the local-linearization diagnostic's construction — n=200 x 300 draws runs in
seconds. §3's coverage benchmark has now RUN, end-to-end through `lifejacket analyze` rather
than through the reference implementation — see "Coverage benchmark result" below; it remains a
documented benchmark, not CI.)

This document is a self-contained implementation brief. The empirical work
behind it lives in the `adjusted-sandwich-user` repo (two-stage
known-projection experiment; see `TWO_STAGE_COVERAGE_RESULTS.md` there, and
the reference implementation `two_stage_refit_bootstrap.py`). Everything
below was validated against lifejacket's own stored outputs on 1000-replication
grids.

## Context

The adjusted sandwich variance estimate is well calibrated *on average*
(mean adjusted SE / empirical SD was 0.99–1.04 across a large grid), and the
Wald interval `theta_hat ± 1.96·SE_adj` reaches nominal 95% coverage as
n grows. But at moderate n (100–1000) on designs where the policy's
sensitivity to beta varies — e.g. any clipped/saturating policy — the Wald
interval undercovers by 2–6 points. Decomposition on 1000-rep grids showed
the shortfall is **not** bias and **not** non-normality of `theta_hat`
(substituting a perfect SE restores exactly 0.95). It is the SE itself, in
two parts:

1. **Noise**: the SE varies across realizations; coverage of a Wald interval
   is concave in the SE, so noise alone costs
   `~ 0.95 − 1.96³·φ(1.96)·Var(SE/sd) ≈ 0.95 − 0.44·Var(SE/sd)` (Jensen).
2. **Anti-correlation with the error**: a realization misses when `beta_hat`
   overshoots into the clip's saturating region — which is exactly where
   `s'(beta_hat)` collapses and shrinks the *estimated* adaptivity
   correction. Large error coincides with small SE by mechanism, not chance.

Because both effects are functions of the same realized sample, every
**symmetric, within-sample** correction we tested fails or under-delivers:

| approach | result (config A, n=250; Wald baseline 0.926) |
| --- | --- |
| t critical value, oracle Satterthwaite df | 0.954 — but the df needs the across-replication spread, unavailable in a single run; under-corrects where dependence dominates |
| multiplier bootstrap on the stored stacks, bread held fixed | no correction at all (crit ≈ 1.96); two-thirds of SE variability lives in the bread — its propagation entry has 51–87% relative sd because it is ∝ `s'(beta_hat)` |
| full refit bootstrap, **studentized** | 0.927 — per-realization critical values are themselves anti-correlated with the misses (miss reps got mean crit 1.99 vs 2.15 for covering reps) |
| full refit bootstrap, **percentile** | **0.951** |

The percentile interval works because it is **asymmetric**: in an
overshoot realization, bootstrap draws that pull `beta_hat*` back toward
`beta*` traverse the steep part of `theta*(s(b))`, stretching the interval
toward the truth in exactly the miss direction. No interval of the form
`theta_hat ± c·SE` can encode that.

Full validated coverage table (1000 reps per cell, two-stage experiment):

| case | Wald | percentile refit bootstrap |
| --- | --- | --- |
| config A n=100 adaptive | 0.9290 | 0.9440 [0.928, 0.957] |
| config A n=250 adaptive | 0.9260 | 0.9510 [0.936, 0.963] |
| config A n=500 adaptive | 0.9090 | 0.9560 [0.941, 0.967] |
| config A n=1000 adaptive | 0.9200 | 0.9370 [0.920, 0.950] |
| config A n=2000 adaptive | 0.9360 | 0.9480 [0.932, 0.960] |
| order-8 clip s1=1 n=250 | 0.9170 | 0.9370 [0.920, 0.950] |
| order-8 clip s1=2 n=250 | 0.8980 | 0.9410 [0.925, 0.954] |
| fixed-policy control n=250 | 0.9530 | 0.9510 |
| fixed-policy control n=1000 | 0.9430 | 0.9460 |

Controls unchanged: the method adds no spurious width when the adaptivity
correction is zero.

## Decision

A **refit percentile bootstrap** over the same RN-weighted joint estimating
system the adjusted sandwich is already built on.

Per bootstrap draw `b = 1..B`:

1. Draw per-subject multiplicities `m_i ~ Poisson(1)` (i.i.d. across
   subjects; Poisson approximates multinomial resampling and vectorizes
   cleanly — either is fine).
2. **Re-solve the full weighted joint system** for
   `(beta*_1, …, beta*_K, theta*)`:
   `0 = (1/n) Σ_i m_i · stack_i(beta_1, …, beta_K, theta)`,
   where `stack_i` is EXACTLY the per-subject RN-weighted estimating function
   stack that `_reference_single_subject_weighted_estimating_function_stacker`
   defines (algorithm components with their Radon–Nikodym weight products,
   inference component with its weight product).
3. Record `theta*_b`.

Report the interval `[quantile(theta*, α/2), quantile(theta*, 1−α/2)]`
(default α = 0.05).

Note what is NOT needed per draw: no bread, no meat, no per-draw sandwich.
The percentile interval only needs the re-solved point estimates, which makes
it much cheaper than the (ineffective) studentized variant.

### The one thing that must not be gotten wrong: the RN weights

At the ORIGINAL solution the RN weights are identically 1 (numerator and
denominator probabilities coincide), so an implementation that forgets them
still passes every point-estimate check at unit multiplicities. They only
bite when the re-solved `beta*` differs from the recorded `beta_hat` — which
is every bootstrap draw. The weights are what make the inference estimating
function mean-zero as beta varies over adaptively collected data; they are
the same mechanism that puts the score term
`psi_theta · p'(A−p)/(p(1−p))` into the joint bread's cross entry (verified:
naive cross-derivative + score term reproduces the stored
`joint_bread_matrix[1,0]` to 1e-6 on every replication checked).
**A refit that re-solves the naive unweighted equations is wrong on adaptive
data and will look right in every weights=1 test.** Include a regression test
that solves at multiplicities forcing `beta* ≠ beta_hat` and checks against
an independent implementation (see §4).

Fixed-policy arms (action probs carry no beta dependence): the weights are
identically 1 for all beta and the refit reduces to plain resampling of the
inference equation. This must fall out naturally rather than be special-cased
incorrectly — the stacker's structure already handles it since d(pi)/d(beta)=0.

### Suggested implementation route

The machinery already exists: `_avg_stack_fn(flattened_x)` (built inside
`get_avg_weighted_estimating_function_stacks_and_aux_values` /
`post_deployment_analysis.py`) evaluates the average weighted stack as a
function of the flattened betas-and-theta vector. The refit is a root-find of
that same function with per-subject multiplicities:

- Add an optional per-subject weight vector to the stack averaging (multiply
  subject i's stack by `m_i` before averaging). The per-subject stacks are
  already computed individually, so this is a weighted mean instead of a mean.
- Newton's method from the original solution `(beta_hats, theta_hat)`:
  2–5 iterations with the Jacobian from the existing vjp/jacrev machinery
  (the same code path that already produces the joint bread — but now at the
  current iterate and with multiplicities). Fall back to more iterations or
  damping on non-convergence; drop and count draws that fail.
- Draws are embarrassingly parallel and shapes are fixed, so `jax.vmap` over
  the multiplicity matrix `(B, n)` with a fixed Newton iteration count is the
  natural fast path; a plain Python loop over draws is an acceptable first
  version behind the flag.

### API

- `analyze_dataset(..., percentile_bootstrap_draws: int = 0,
  percentile_bootstrap_alpha: float = 0.05, percentile_bootstrap_seed: int | None)`
  — 0 means off (default; no behavior change for existing users).
- CLI: `--percentile_bootstrap_draws`, `--percentile_bootstrap_alpha`,
  `--percentile_bootstrap_seed` on `lifejacket analyze`. The seed is
  `click.IntRange(min=0)`: `np.random.default_rng` raises on a negative seed,
  and rejecting a typo'd command line at the boundary is worth more than the
  hours of cluster compute that would otherwise precede the failure.
- Outputs added to the analysis dict when enabled:
  `percentile_bootstrap_ci` (theta_dim × 2), `bootstrap_num_draws`,
  `bootstrap_num_failed_draws`, `bootstrap_failure_reasons` (the reason
  taxonomy, counted), and `bootstrap_error`; the raw `theta*` draws go to the
  debug-pieces output as `percentile_bootstrap_theta_draws`.
- Log a warning if more than ~2% of draws fail to converge or produce
  degenerate systems.
- **The bootstrap is best-effort, like the diagnostic suite.** The adjusted
  sandwich is fully computed before it runs but nothing has been written yet,
  so an unguarded failure here would discard the whole analysis over an added
  interval. Instead the block is guarded: the failure is logged, the interval
  comes back all-NaN with `bootstrap_num_failed_draws == bootstrap_num_draws`,
  and `bootstrap_error` carries a `"Type: message"` string (it is `None` on a
  bootstrap that ran to completion, however many individual draws failed).
  Downstream code can therefore distinguish "the bootstrap failed" from "the
  bootstrap was not requested" — the keys are absent entirely in the latter
  case — and must not read the NaN interval as a finding.
- The per-iterate Newton Jacobian is memory-bounded by the package-wide
  chunking policy: `analyze_dataset` resolves its `jacobian_row_chunk_size`
  once against the flattened solution's dimension and threads it into both the
  jitted `(stack, Jacobian)` fast path and `_newton_refit`'s eager fallback,
  which build their Jacobians with
  `helper_functions.compute_row_chunked_jacobian` rather than a bare
  `jax.jacrev`/`vmap(pullback)`. The fast path is probed once up front; if that
  graph will not trace or compile, the bootstrap logs a downgrade and runs the
  eager path on the identical draws instead of failing.

### Numerical guardrails

- RN weights can explode when a recorded probability is near 0/1 and the
  re-solved beta moves far: compute weight products in log space, and drop
  (count) draws with non-finite stacks.
- Degenerate inference denominators (e.g. all `m_i` mass on subjects with
  tiny `(A−p*)²`): detect via Jacobian conditioning at the solution; drop and
  count.
- Multivariate theta: everything above is dimension-agnostic; percentile
  intervals are per-coordinate.

### Validation plan (the acceptance gate)

1. **Weights=1 identity**: with all multiplicities 1, the Newton solve must
   return the original `(beta_hats, theta_hat)` immediately (the avg stack is
   already ~0 there — this is the existing fidelity check).
2. **Independent reference**: `adjusted-sandwich-user/two_stage_refit_bootstrap.py`
   implements the identical weighted system in closed form for the two-stage
   design. With the same multiplicity draws (same seed / shared fixture), the
   lifejacket refit's `(beta*, theta*)` must match it to numerical precision
   on stored two-stage replications (any `analysis.pkl`/`study_df.pkl` pair
   from the jobs listed in TWO_STAGE_COVERAGE_RESULTS.md's manifest). This
   catches the missing-RN-weights failure mode described in §2.
3. **Coverage benchmark** (slow test or documented benchmark, not CI): on the
   stored config A n=250 replications, percentile coverage ≈ 0.95 (0.9510
   observed) vs Wald 0.9260; on the n=250 control, unchanged (0.9510 vs
   0.9530).

### Coverage benchmark result (2026-09-01)

Step 3 above has now run through lifejacket's own `analyze_dataset`/`lifejacket analyze`
path, rather than through `two_stage_refit_bootstrap.py`. This is a different claim from
step 2: step 2 showed the two implementations agree draw-for-draw on a shared fixture
(2.4e-8 median relative difference), while this shows the shipped code delivers the
coverage over 1000 fresh replications per arm. Config A, soft clip, n=250, 300 draws per
replication, exact target theta* = 3.3783359435623384; jobs 43487413 (adaptive) and
43487414 (fixed-policy control), 1000/1000 analyses readable in both.

| arm | Wald | percentile refit bootstrap | mean Wald width | mean percentile width |
| --- | --- | --- | --- | --- |
| adaptive n=250 | 0.9260 [0.910, 0.942] | **0.9460** [0.932, 0.960] | 1.6176 | 1.4717 |
| fixed-policy control n=250 | 0.9530 [0.940, 0.966] | 0.9490 [0.935, 0.963] | 1.2249 | 1.2095 |

Reading:
- Both Wald figures reproduce this ADR's reference table EXACTLY (0.9260 and 0.9530),
  which is the evidence that the benchmark reproduces the intended design rather than
  some neighbouring configuration.
- The adaptive arm's percentile coverage, 0.9460, sits below the 0.9510 recorded above,
  but that reference came from a different 1000-replication draw through the reference
  implementation; at 1000 reps the Monte Carlo standard error is ~0.007, the two
  intervals overlap heavily, and the difference is not evidence of disagreement.
- **The percentile interval is NARROWER than Wald on the adaptive arm (1.4717 vs 1.6176,
  ~9% narrower) while covering better.** The undercoverage was never a width problem —
  it is the miss-direction geometry described in §1, and the asymmetric interval fixes it
  without buying coverage with width.
- The control is the negative control and behaves: coverage unchanged and width within
  1.3% of Wald, so the method adds no spurious width where the adaptive correction is zero.
- Robustness: **zero failed refits** across 2000 replications x 300 draws = 600,000 Newton
  refits. The failure taxonomy exists for small-n draws that zero an early recruitment
  wave; nothing in this grid hit it.

## Consequences

- Default behavior is unchanged (`percentile_bootstrap_draws=0`); enabling it
  costs ~B x (a few Newton steps of the average-stack evaluation), and draws
  vmap cleanly since shapes are fixed.
- Reported intervals become asymmetric around `theta_hat` when enabled;
  downstream collection/plotting that reconstructs intervals as
  `theta_hat +/- c*SE` must instead read the new `percentile_bootstrap_ci`.
- The adjusted sandwich variance remains the reported variance estimate (it
  is well calibrated on average); this ADR changes the *interval*, not the
  variance.

### Explicitly out of scope / do not "improve" into

- **Studentized (percentile-t) intervals**: tested; they do not help here
  (see §1 table) despite being the textbook higher-order choice — the
  per-realization critical value inherits the same anti-correlation. Fine to
  expose the T* draws for diagnostics; do not make studentized the default.
- **A multiplicative SE inflation factor**: the estimable version of this is
  the studentized bootstrap (fails); the constant version needs oracle
  knowledge and is config-dependent (needed 1.13–1.29× across our configs).
- **BCa/basic intervals**: the basic ("reverse percentile") interval reflects
  the skew the wrong way for this geometry; BCa untested — a possible later
  refinement, not part of this ADR.
