# 0001. Performance plan for the adjusted sandwich computation

- Status: Accepted (Steps 0-2 and three correctness bugs below are done; Step 4 is done -- see
  "Step 4: padded + masked jax.vmap batching" below; Step 3 has been attempted twice -- once
  before Step 4 (reverted in full) and once after (partially kept: the local-linearization
  diagnostic's forward-pass closure is now jitted, a clear win; jitting the main differentiated
  call was tried and reverted again, a clear regression even post-Step-4 -- see "Step 3, second
  attempt: jit after Step 4" below); Step 5 remains proposed; Step 7 is done -- the
  small-sample-correction feature was removed outright, not optimized, once profiling
  identified it as the dominant real-scale memory cost -- see "Step 7" below)
- Date: 2026-08-26
- Ticket: ADS-139

## Context

`analyze_dataset` (`lifejacket/post_deployment_analysis.py`) is slow enough
to be a real obstacle to iterating on this library, and it gets slower
faster than users expect as a study grows. A live benchmark on a *tiny*
synthetic fixture (20 subjects, 6 decision times -- see
`tests/benchmarks/`) already took 20 seconds; a real integration test at
100 subjects / 10 decision times takes 2-3 minutes per run.

### Root cause

The estimating-function stack that gets differentiated to build the joint
bread matrix is constructed via **plain Python loops, not `jax.vmap`**, and
the whole thing is **never wrapped in `jax.jit`**:

- `construct_classical_and_adjusted_sandwiches` calls
  `jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values, has_aux=True)`
  with nothing jitted.
- Inside that function, `get_avg_weighted_estimating_function_stacks_and_aux_values`
  loops over every subject in a Python list comprehension (not `vmap`).
- Inside *that*, `single_subject_weighted_estimating_function_stacker` loops
  over every policy update in a second Python list comprehension -- the
  code's own comment says vmapping this axis "would be tricky due to
  different shapes across updates."

So `jax.jacrev` has to trace and reverse-differentiate a graph that's been
unrolled by Python into `O(subjects x updates)` individually-dispatched JAX
ops, instead of a handful of batched, compiled ones. This exact pattern is
duplicated with minor variations in `deployment_conditioning_monitor.py` and
`get_datum_for_blowup_supervised_learning.py`.

The reason it isn't already `vmap`'d over subjects: each subject's data is
represented as a *variable-length* sequence (an empty-tuple `()` sentinel
marks decision times / updates a subject wasn't active for, and per-subject
code filters those out before building tensors). Subjects and updates
actually **do** share a common global time/update axis -- only *validity*
varies per subject -- so this is fixable with padding + masking, but it's a
real architectural change, not a one-line fix.

### Empirical confirmation (Step 0)

`tests/benchmarks/test_analyze_dataset_benchmark` instruments
`analyze_dataset`'s major phases (via `log_phase_duration` in
`lifejacket/helper_functions.py`) and reports, at both a `"small"` fixture
(n=20, T=6) and a `"medium"` one (n=100, T=10, matching
`tests/integration_tests`' scale):

```
Total wall-clock time: ~17-20s (small)
    ~74-80%  local_linearization_diagnostic
    ~20-26%  construct_classical_and_adjusted_sandwiches (the jax.jacrev hot path)
    < 3%     everything else (input checks, small-sample correction, stabilization, ...)
```

This was a genuine surprise relative to the initial hypothesis: the
always-on "local linearization validity" diagnostic in `analyze_dataset`
(wrapped in a `try/except` that silently swallows failures) re-runs the
*entire* un-jitted forward pass 15 times to validate linearity, and at this
scale costs **more** than the main `jax.jacrev` call it's meant to sanity
check. At medium scale the same diagnostic costs even more in absolute
terms (measured 113-142s), and still dominates the total -- run the
benchmark at whatever scale you're optimizing for before assuming which
piece dominates; don't assume the small-scale proportions hold at large
scale, or vice versa (see Step 3 below for a case where scale changed the
conclusion entirely).

## Decision

Work through the following stages, in order, each gated by the benchmark
suite (numerical regression + timing) and the existing test suite:

| # | Step | Status |
|---|---|---|
| 0 | Build a phase-timing + golden-output regression benchmark (`tests/benchmarks/`, two scales) before optimizing anything | **Done** |
| - | Fix correctness bugs found while investigating (below) | **Done** |
| 1 | Vectorize the eager per-row/per-subject checks in `input_checks.py` (they run by default in real usage, not just diagnostics) | **Done** |
| 2 | Extract the (currently byte-identical) triplicated Radon-Nikodym weight-computation block into one shared function | **Done** |
| 3 | Wrap the `jax.jacrev` hot path in `jax.jit` | **Partially done.** The local-linearization diagnostic's forward-pass closure is jitted (real win, both scales). Jitting the main differentiated call was tried again post-Step-4 and reverted again (regression at medium scale). See "Step 3, second attempt" below. |
| 4 | Convert the ragged per-subject/per-update Python loops to padded + masked `jax.vmap`/`jax.lax.scan` | **Done** -- see "Step 4: padded + masked `jax.vmap` batching" below |
| 5 | Exploit block-lower-triangular Jacobian sparsity (roughly half the joint bread matrix is analytically zero) | Deferred -- see below |

Steps 1-2 turned out to have a real but small effect at the scales
benchmarked (Step 1's `input_checks` phase measured ~0.3-1.1s out of a
~17-20s small-scale total; Step 2 is a pure, value-preserving refactor with
no expected timing effect) -- they're correct, low-risk, and worth keeping,
but neither was the dominant cost. Step 3, expected going in to be "the
single highest-leverage, lowest-redesign-risk change," turned out to be a
severe regression once actually measured (see below) -- a reminder that this
plan's own Step 0 rationale (measure before extrapolating) applies to each
step's *design*, not just the starting diagnosis. Step 4 (real vectorization)
is the deepest architectural change and the only way to remove the
`O(subjects)` factor entirely; it must use a cumulative-product-with-reset
construction for the Radon-Nikodym weight window (never a division of prefix
products -- a clipped weight of exactly 0 earlier in time would produce a
silent `0/0` unrelated to the window in question), and must sanitize
*inputs* before calling user-supplied functions on padding, not just mask
*outputs* -- masking output values does not stop a singularity in the
function's *own* internals (e.g. `1/x` at a padded `x=0`) from NaN-poisoning
the whole gradient during backprop, since `0 * NaN = NaN`.

### Step 3 attempted and reverted: `jax.jit` compile time regression

Implemented as designed: a closure baked in every static (non-differentiated)
argument to `get_avg_weighted_estimating_function_stacks_and_aux_values`
(callables, type tags, indices, `policy_num_by_decision_time_by_subject_id`),
leaving only the dynamic array-valued pytree arguments as call-time
parameters, then wrapped `jax.jit(jax.jacrev(closure, has_aux=True))` around
it (jit-outside-jacrev, matching the existing `jax.jit(jax.vmap(jax.jacrev(...)))`
precedent in `calculate_derivatives.py`); the `.tolist()`/`int()`
concretization breaks in `arg_threading_helpers.py` (`thread_update_func_args`,
`thread_inference_func_args`) were fixed with an optional precomputed-indices
parameter so they're safe to call from inside a trace; the
`require_threaded_*_equivalent` input checks (which call
`np.testing.assert_allclose` and therefore cannot run on a traced value) were
hoisted to run once, eagerly, before constructing the jit closure. All of
this was numerically correct -- the golden-fixture comparison passed.

**But it made `analyze_dataset` far slower, not faster, and the regression
gets worse with scale, not better:**

| Scale | Eager (`jax.jacrev` alone, Steps 1-2) | `jax.jit(jax.jacrev(...))` (Step 3) |
|---|---|---|
| small (n=20, T=6) | ~5-6s | ~77-93s (compile-dominated) |
| medium (n=100, T=10) | ~22-25s | killed after 37+ minutes of CPU time, still compiling |

This is the opposite of what JIT is supposed to do, and the reason is
specific to this codebase's structure: `get_avg_weighted_estimating_function_stacks_and_aux_values`
builds its computation graph via a **fully Python-unrolled** per-subject list
comprehension (and a second, nested one per policy update inside it) rather
than `jax.vmap`. `jax.jit` has to compile whatever graph it's given; a
graph unrolled to thousands of individually-scheduled small ops with no
shared/batched structure gives XLA's optimization passes (scheduling, buffer
assignment, CSE) nothing to exploit, and their cost appears to scale
noticeably worse than linearly in the number of unrolled subject iterations
here (5x the subjects took the compile from ~80s to well past 37 minutes).
Since this compile happens fresh on every `analyze_dataset` call (a new
closure each call means no cross-call compilation cache hit), there's no
realistic usage pattern in which the one-time compile cost is amortized away
for this specific unrolled shape.

**Conclusion: Step 3 only becomes viable once Step 4 has replaced the
per-subject/per-update Python loops with real `jax.vmap` batching**, which
would give XLA a small number of batched, compiled-once kernels instead of a
massive bespoke unrolled graph -- shrinking compile time by construction, not
just hoping XLA optimizes an unrolled graph well. Attempting Step 3 before
Step 4 is not "a smaller, safer first step" toward the same goal; it's a
different bet that failed. Re-attempt Step 3 (or fold it into Step 4's
design directly) only after Step 4 lands, and re-measure at both benchmark
scales before considering it done, not just at whichever scale is fastest to
iterate on.

**What was kept from the attempt** (see "Bugs found" below for the
independent one): the revert removed the closure-building and
check-hoisting scaffolding in `post_deployment_analysis.py` and the
optional precomputed-indices parameters in `arg_threading_helpers.py`
entirely, rather than leaving them in place unused -- once `jax.jit` isn't
applied, `jax.jacrev`'s own forward pass already runs the checks exactly
once (same as the hoisted version would), so the extra machinery had no
remaining purpose and would just have been unexercised complexity.

### Step 4: padded + masked `jax.vmap` batching (done)

Implemented in `lifejacket/batched_weighted_estimating_function_stack.py`
(new module) and wired into
`get_avg_weighted_estimating_function_stacks_and_aux_values`
(`post_deployment_analysis.py`), replacing that function's O(subjects) and
O(subjects x updates) Python loops with:

- A one-time, plain-numpy structural precompute
  (`build_action_prob_layer_precompute`, `build_update_layer_precompute`,
  `build_inference_layer_precompute`) that converts the ragged
  action-prob-args / update-args / inference-args into fixed-size `(N, T,
  ...)` padded tensors plus boolean validity masks. Every invalid cell is
  **self-padded**: filled with that same subject's own real data from their
  nearest active time (forward/backward-filled), never a fabricated
  constant.
- A single `jax.vmap` call spanning every subject and every global decision
  time at once for the Radon-Nikodym weight computation
  (`compute_action_prob_layer_outputs`), replacing what used to be one
  small `jax.vmap` dispatched per subject.
- A **cumulative-product-with-reset** construction for the weight-window
  product (`compute_windowed_weight_products`): invalid/out-of-window
  positions are reset to the multiplicative identity `1.0` before one
  `jnp.cumprod` runs, never a division of prefix products (which would risk
  a spurious `0/0` from an unrelated clipped weight earlier in time).
- Shape-bucketed `jax.vmap` (reusing `calculate_derivatives.group_user_args_by_shape`,
  the same machinery `input_checks.py`'s equivalence checks already used
  for this identical problem) for the algorithm and inference estimating
  functions, since subjects at a given update do **not** all share
  argument shape in general (incremental recruitment means two subjects
  present at the same update can have different accumulated-row counts,
  and `alg_update_func`s here reduce over that row axis -- padding it would
  silently change the answer, so bucketing by shape, not padding, is used
  for this axis).
- `previous_post_update_betas` needed no special-case handling: traced
  directly, this argument's *content* is never actually read by
  `thread_update_func_args` (only its length is, to slice `betas`), and
  every `alg_update_func` this repo ships supplies the identical value to
  every subject at a given update -- so a plain broadcast is correct. This
  is checked, not assumed: `compute_batched_algorithm_component` raises a
  clear `ValueError` if two subjects in the same shape bucket ever have
  non-matching previous-betas content.
- The intra-window-gap invariant the original code's `active_index`
  bookkeeping implicitly relied on (a subject, once active, stays active
  with no drop-out/re-entry) is now an explicit, loud `ValueError` --
  `assert_no_intra_window_gaps` in the new module (for the batched path)
  and a matching guard added directly to the shared
  `compute_subject_radon_nikodym_weights` (`helper_functions.py`, protecting
  its other two callers too -- see "Known follow-ups" below).

**Scope: this pass touches only `post_deployment_analysis.py`.**
`deployment_conditioning_monitor.py` and
`get_datum_for_blowup_supervised_learning.py` still use the original,
un-batched per-subject implementation (via the same shared
`compute_subject_radon_nikodym_weights`). This was a deliberate choice, not
an oversight: `post_deployment_analysis.py` is the only one of the three
`tests/benchmarks` actually measures, the three files have genuinely
different shapes (`only_latest_block` filtering, RL-only vs. inference-only,
differing return-tuple arity), and landing a brand-new vectorization
pattern once, validated thoroughly, was judged lower total risk than
repeating it three times in one pass -- especially as the highest-risk step
in this plan, immediately following Step 3's regression. Porting the same
pattern to the other two files is the natural next step if their
performance ever becomes a concern; the shape-bucketed-vmap helpers here
are not `post_deployment_analysis.py`-specific and should be close to
directly reusable.

**Result, both benchmark scales** (`tests/benchmarks`, re-measured per this
ADR's own Step 3 postmortem warning not to trust one scale):

| Scale | `jax.jacrev(...)` before Step 4 | after Step 4 |
|---|---|---|
| small (n=20, T=6) | ~5-6s | ~3.3s |
| medium (n=100, T=10) | ~22-25s | ~8-12s |

A real, meaningful win at both scales, and -- per the whole point of this
step -- it should now give `jax.jit` a small number of batched, compile-once
kernels instead of a massive unrolled graph. Re-attempting Step 3 on top of
this is the natural next step, not yet done.

**Numerical fixtures needed regenerating, and this was verified, not
assumed.** Batching the weight-window product changes the floating-point
*order of operations* relative to the original per-subject `jnp.prod`
(different summation/multiplication order under `jax.vmap`'s SIMD
execution), producing float32 reassociation noise on the order of `1e-9` to
`1.5e-5` absolute in the joint bread matrix and downstream sandwich
matrices -- exceeding the existing `rtol=1e-6`/`atol=1e-5`/`atol=1e-8`
tolerances in `tests/benchmarks` and all four `tests/integration_tests`
fixtures. Before regenerating anything, this was confirmed to be noise, not
a bug: `theta_est` and `classical_sandwich_var_estimate` (quantities the
batched weight computation does not touch) matched the pre-Step-4 golden
values **bit-for-bit** in every case; only `adjusted_sandwich_var_estimate`
and the joint bread matrix (which the batched weight computation *does*
feed) differed, by exactly the reassociation-scale amount; and the
independent closed-form regression test
(`tests/unit_tests/test_sandwich_closed_form_notebook_example.py`, which
derives its expected values from first principles, not from a golden
pickle) passed unchanged. `tests/benchmarks/fixtures/{small,medium}/golden_analysis.pkl`
and all four `tests/integration_tests/*/expected_*.pkl` were then
regenerated from fresh, deterministic (fixed-seed) runs of the new code.
Regenerating the integration fixtures also surfaced two pre-existing,
unrelated legacy key-naming mismatches in `tests/utils.py`'s comparison
(`joint_bread_inverse_matrix` vs. the current `raw_joint_bread_matrix`;
`adaptive_sandwich_var_estimate` vs. the current `adjusted_sandwich_var_estimate`)
-- worked around by preserving the old key names inside the regenerated
fixtures (not by changing the shared `tests/utils.py` comparison code),
keeping this fix scoped to data, not test infrastructure.

**Known follow-ups, deliberately not done in this pass** (found by an
adversarial code review after implementation; the concrete correctness
regressions it found were fixed immediately -- see below -- these are the
ones left as documented, lower-priority gaps):

- The intra-window-gap guard added to `compute_subject_radon_nikodym_weights`
  now also protects `deployment_conditioning_monitor.py` and
  `get_datum_for_blowup_supervised_learning.py` (both call it), but neither
  file's own input-check entry point (`perform_alg_only_input_checks`, or no
  check at all, respectively) validates this invariant *before* reaching it,
  and neither has a test exercising this path. A subject whose active flag
  legitimately goes `1 -> 0 -> 1` during live monitoring (e.g. a temporary
  app outage) will now hard-raise there where it previously silently
  mis-indexed -- a strict improvement (loud failure over silent corruption),
  but undocumented and untested at those two call sites.
- Two independent implementations of the Radon-Nikodym weight-window
  computation now exist: the original, per-subject one in
  `compute_subject_radon_nikodym_weights` (`helper_functions.py`, still used
  by the two un-ported files) and the new batched one in
  `batched_weighted_estimating_function_stack.py`. This partially undoes
  Step 2's consolidation for the one caller Step 4 touches. A future
  numerical edge-case fix to one is not guaranteed to be ported to the
  other; no test directly diffs their *values* against each other on a
  shared fixture (the correctness evidence instead comes from the
  hand-derived expected values in `tests/unit_tests/test_post_deployment_analysis.py`'s
  seven existing per-subject/per-outer-product fixtures, all of which
  independently derive their expected numbers rather than trusting either
  implementation, plus one new direct cross-check --
  `test_batched_and_reference_implementations_agree_per_subject_incremental_recruitment`).
- **Fixed** (post Step-4/Step-3 work, same session): the residual data-check
  block (`require_threaded_algorithm_estimating_function_args_equivalent` /
  `require_threaded_inference_estimating_function_args_equivalent`, run
  whenever `suppress_all_data_checks=False` and action probabilities feed
  the algorithm/inference estimating function) used to re-derive threaded
  args via the original, un-batched per-subject `jax.vmap` dispatches in
  `arg_threading_helpers.py`, then re-run the estimating function again per
  shape bucket on top of that -- comparable in dispatch-count order to the
  entire batched path it was double-checking. Replaced with
  `check_batched_algorithm_estimating_function_args_equivalent` /
  `check_batched_inference_estimating_function_args_equivalent`
  (`batched_weighted_estimating_function_stack.py`), which reuse the
  already-computed `pi_beta_grid`/`action_prob_layer`/`update_layer` instead
  of re-deriving anything via `arg_threading_helpers`, and share the exact
  same bucket-override construction (`_build_algorithm_bucket_overrides` /
  `_build_inference_bucket_overrides`, extracted from
  `compute_batched_algorithm_component` / `compute_batched_inference_outputs`
  for this purpose, closing a duplication gap the earlier adversarial review
  had flagged) as the main computation, so the "threaded" values being
  checked are guaranteed identical to what the main computation actually
  uses. `get_avg_weighted_estimating_function_stacks_and_aux_values` was
  reordered so the batched forward pass (which produces `pi_beta_grid`) runs
  *before* the data-check block, not after. Measured at medium scale
  (n=100): the gap between checks-suppressed and checks-enabled `jax.jacrev`
  time, previously several seconds (checks-enabled always slower), is now
  within this session's measurement noise -- the two are indistinguishable.
  `arg_threading_helpers.thread_action_prob_func_args`/`thread_update_func_args`/`thread_inference_func_args`
  and `input_checks.require_threaded_*_equivalent` are all still imported/used
  elsewhere (`get_datum_for_blowup_supervised_learning.py` and
  `deployment_conditioning_monitor.py` respectively) and were deliberately
  left untouched.

  Writing a unit test for this surfaced a genuine surprise, not a bug:
  `tests/unit_tests/test_post_deployment_analysis.py`'s
  `setup_data_two_loss_functions_use_action_probs_both_sides` fixture
  deliberately uses a different beta in `update_func_args` than in
  `all_post_update_betas` (to test that the shared beta gets substituted in
  for differentiation) -- which is *exactly* the inconsistency this data
  check exists to catch. No existing test had ever run this fixture with
  `suppress_all_data_checks=False`, so this had never been exercised before.
  Confirmed directly that the original, un-batched
  `require_threaded_algorithm_estimating_function_args_equivalent` also
  raises on this exact fixture -- the new check is behaviorally faithful,
  not a new bug -- and added both this as a named regression test
  (`test_batched_algorithm_data_check_detects_the_same_inconsistency_as_the_original`)
  and a separate, hand-built, genuinely self-consistent fixture proving the
  check passes silently on well-formed data
  (`test_check_batched_algorithm_estimating_function_args_equivalent_passes_on_consistent_data`,
  `tests/unit_tests/test_batched_weighted_estimating_function_stack.py`).

  A second adversarial review of this same change (10 parallel angles) found
  and fixed one real, higher-severity bug and two lower-severity gaps, all
  independently confirmed by multiple review angles: (a)
  `check_batched_inference_estimating_function_args_equivalent` used the
  algorithm check's tolerance (`atol=1e-7, rtol=1e-3`) instead of the
  inference check's own, deliberately looser original tolerance
  (`input_checks.require_threaded_inference_estimating_function_args_equivalent`
  uses `rtol=1e-2`, no `atol`) -- an unintentional copy-paste that would have
  made real production data fail this check spuriously. Fixed by extracting
  a shared `_assert_original_and_threaded_bucket_results_agree(estimating_func,
  bucket, threaded_overrides, atol, rtol)` helper used by both check
  functions, each now passing its own correct, original tolerance explicitly
  -- this also closes the duplication gap that let the two drift apart in
  the first place. (b) `_stack_grid_to_tensor`'s "arrays with dimension > 2
  are not supported" guard only inspected the first flattened cell, not
  every cell -- fixed to check all of them. (c)
  `check_batched_inference_estimating_function_args_equivalent` had zero
  test coverage anywhere in the suite (the one fixture exercising both
  checks together is deliberately inconsistent on the *algorithm* side, so
  it raises before the inference check ever runs) -- added
  `test_check_batched_inference_estimating_function_args_equivalent_passes_on_consistent_data`
  (a hand-built, self-consistent, updates-free fixture isolating the
  inference side). A duplicated ~8-line "gather the reconstructed
  action-probability override" block between `_build_algorithm_bucket_overrides`
  and `_build_inference_bucket_overrides` was also extracted into
  `_add_action_prob_override_if_used`.

  Two more findings were investigated and did **not** need a code fix: one
  reviewer flagged that the new checks no longer independently re-derive
  their own subject grouping/bucketing the way the old per-subject checks
  did (they now reuse `update_layer`/`inference_layer`, the same objects the
  main computation uses) -- raised as a loss of independent verification
  power for a bug in the precompute itself, but this is an accepted,
  deliberate tradeoff (re-deriving independent bucketing here would
  reintroduce the exact O(subjects) cost this change removes; the
  precompute's own correctness is independently established by Step 4's
  test suite, including a cross-check against the retained
  `_reference_single_subject_weighted_estimating_function_stacker`).
  Another flagged that dropping the OLD checks (which iterated
  `update_func_args_by_by_subject_id_by_policy_num`'s own keys directly) in
  favor of the new checks (which only walk `beta_index_by_policy_num`'s
  keys via `update_layer`) loses an incidental KeyError guard against an
  unexpected policy_num with real data -- confirmed this exact scenario is
  already caught earlier and independently, under the same
  `suppress_all_data_checks` gate, by
  `input_checks.perform_first_wave_input_checks`'s
  `require_no_policy_numbers_present_in_alg_update_args_but_not_analysis_df`,
  so this is not a net loss of protection.

  One efficiency finding was left as a documented, lower-priority follow-on:
  each check computes the "threaded" bucket value via `jax.vmap`, uses it
  only for the `assert_allclose` comparison, then discards it -- the main
  computation (`compute_batched_algorithm_component`/`compute_batched_inference_outputs`)
  recomputes the identical value moments later for real use. This doubles
  the estimating-function evaluation cost specifically when checks are
  enabled, bounded by O(updates x shape-buckets), not O(subjects) -- a much
  smaller cost than what this change already removed, and not distinguishable
  from noise in this session's measurements, but a real, avoidable
  redundancy if a future profile shows otherwise (thread the check's already-computed
  threaded value through to the main computation instead of recomputing it).
- The one-time structural precompute
  (`build_action_prob_layer_precompute`/`build_update_layer_precompute`/`build_inference_layer_precompute`)
  is rebuilt from scratch on every call to
  `get_avg_weighted_estimating_function_stacks_and_aux_values`. This function
  runs ~16 times per `analyze_dataset` call (once for the real `jax.jacrev`
  call, 15 more for the local-linearization diagnostic's forward passes),
  all using identical structural data -- only the differentiated betas/theta
  values change. **Update (Step 3, second attempt, below): this is now fixed
  for 15 of those 16 calls** -- jitting the diagnostic's forward-pass closure
  means its precompute only actually executes once, at trace/compile time,
  with the (numpy) result baked into the compiled program as a constant for
  all 15 evaluations. It is **still true for the 1 real `jax.jacrev` call**,
  since that call was not jitted (see "Step 3, second attempt" below for why)
  -- hoisting the three precompute objects out to that one remaining caller
  would remove the redundant O(N·T) work there too, but was not done here to
  keep this change's surface area small.

**Correctness regressions found by adversarial review and fixed before
landing:** a safety check meant to catch a malformed `algorithm_estimating_func`/`inference_estimating_func`
output shape (`... does not have a size that is a multiple of the beta
dimension`) had become tautologically true against the new pre-allocated
`(N, U, beta_dim)`/`(N, theta_dim)` output arrays and could never fire --
silently letting `.at[].set()` broadcast a wrong-shaped-but-broadcastable
output across every slot instead of raising; replaced with an explicit
shape check at the point each bucket's output is produced, before the
scatter. `num_previous = prev_raw_list[0].shape[0]` required an ndarray
where the original used `len(...)` (accepting a plain list too); switched
back to `len(...)`. The previous-betas cross-subject consistency check used
bit-exact `np.array_equal`, which could spuriously reject two subjects
whose logically-identical values reached the bucket via different
floating-point paths; loosened to `np.allclose`. `betas[:num_previous]`
could silently clamp to a shorter-than-requested slice instead of raising
if a subject's recorded previous-betas length ever exceeded the number of
available updates; added an explicit length check. `actions_grid` was left
as a bare zero-fill at invalid cells, the one argument tensor not
self-padded like the module's own stated invariant -- fixed to self-pad the
same way as everything else (`action` is masked and non-differentiated
downstream, so this was not an active bug, but it was a real, silent
exception to the documented design).

### Step 3, second attempt: `jax.jit` after Step 4

With Step 4's `jax.vmap` batching in place, Step 3 was re-attempted, split
into two independent pieces since they have very different reuse
characteristics.

**Jitting the local-linearization diagnostic's forward-pass closure: kept, a
clear win.** `_compute_local_linearization_error_ratio` (nested inside
`analyze_dataset`) already had its own closure,
`_eval_avg_stack_jit` -- a per-perturbation forward evaluator, already
hardcoding `suppress_all_data_checks=True` and
`include_auxiliary_outputs=False`, so none of the data-check
concretization hazards from the first Step 3 attempt applied to it. This
closure is called `J=15` times per `analyze_dataset` call (once per
perturbation), so compiling it once and reusing the compiled artifact for
all 15 evaluations is exactly the scenario `jax.jit` is built for.
Decorating it with `@jax.jit` (a one-line change) is what it took:

| Scale | eager (Step 4 only) | jitted |
|---|---|---|
| small (n=20, T=6) | ~2.2-2.9s | ~0.4-1.3s |
| medium (n=100, T=10) | ~7.5-8.9s | ~0.2-3.8s |

(The medium-scale jitted range reflects system-load variance across this
session's many benchmark runs on a shared machine, not code changes --
every measurement in this range was still a clear win over the eager
baseline, confirmed by direct back-to-back A/B comparisons at a fixed point
in time, not by comparing absolute numbers taken minutes apart.) This also
incidentally fixes part of Step 4's "structural precompute rebuilt ~16
times" known follow-up for this specific call path: `build_action_prob_layer_precompute`/`build_update_layer_precompute`/`build_inference_layer_precompute`
only ever touch static, non-differentiated data, so under `jax.jit` they
execute once at trace/compile time and their (numpy) results get baked into
the compiled program as constants, rather than being recomputed in Python
on every one of the 15 calls.

This surfaced one real, latent bug in the Step 4 module, now fixed:
`_stack_grid_to_tensor` (used only by the one-time structural precompute)
called the `jax.numpy`-based `stack_batched_arg_lists_into_tensors`
(`vmap_helpers.py`) and then `np.asarray(...)`'d the result. This worked by
accident in eager mode (a concrete `jnp` array converts fine via
`__array__`), but under `jax.jit` tracing, every `jnp` operation inside the
trace produces an abstract tracer regardless of whether its inputs are
literal constants -- so the subsequent `np.asarray(...)` raised
`TracerArrayConversionError`. Fixed by making `_stack_grid_to_tensor`
strictly `numpy`-only (`np.stack`/`np.asarray`, never `jnp`), matching what
it always should have been given its role. Every other function in the
structural-precompute path was audited for the same class of leak and found
clean.

An adversarial review of this fix caught two behavior changes it introduced
relative to the old `jnp`-based version, both now fixed: (a) it silently
turned a `None`-valued `action_prob_func_args` cell into a `dtype=object`
array (previously a silent float32 NaN-fill), which crashed confusingly
deep inside `jax.vmap` instead -- fixed with an explicit check that raises a
clear `ValueError` naming the offending `(subject, time)` cell, since
`action_prob_func_args` positions (unlike `alg_update_func_args`/`inference_func_args`
override positions) are never legitimately `None` in the first place; (b) it
silently dropped the old code's explicit `TypeError` guard against >2D
argument values -- restored. A third finding (the new numpy path defaults to
float64/int64 where the old `jnp` path defaulted to float32/int32) was
reviewed and left as-is: this repo never sets `jax_enable_x64`, so every one
of these tensors gets silently downcast back to float32/int32 the moment it
crosses into any `jnp` operation downstream, same as before -- the only
actual cost is a temporarily wider *host-memory* footprint for these small
(N, T, ...) precompute tensors before that conversion happens, negligible at
this codebase's scale.

**Jitting the main differentiated call (`construct_classical_and_adjusted_sandwiches`'s
`jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values, has_aux=True)`):
tried, reverted again.** Implemented as a closure baking in all static
config (mirroring the first attempt and the diagnostic's own closure),
`jax.jit(jax.jacrev(closure, has_aux=True))`, applied only when
`suppress_all_data_checks=True` (the data-check block's
`np.testing.assert_allclose`/`.tolist()`/`int()` calls have the same
concretization problem as before and were not addressed this time -- see
"known follow-ups" below). Unlike the diagnostic, this call runs only
**once** per `analyze_dataset` invocation, so there is no
compile-once-reuse-many-times amortization available to offset compile
cost. Measured, reproducibly (two independent runs, nearly identical
numbers each time):

| Scale | eager (Step 4 + diagnostic jit) | jacrev also jitted |
|---|---|---|
| small (n=20, T=6) | ~3.3-3.6s | ~2.2s |
| medium (n=100, T=10) | ~6.2-6.9s | ~12.1-12.3s |

Small scale looked like a modest win in isolation, but **medium scale was a
clear regression, and it also degraded the separate, already-jitted
diagnostic closure running later in the same process** (its own time going
from ~0.45s to ~3.6s, despite that closure itself being completely
unchanged) -- some interaction between two concurrent `jax.jit`
compilations competing for resources within one process, not fully
root-caused, but reproduced identically twice. Net effect at n=100: total
`analyze_dataset` wall-clock roughly doubled (~7.1s -> ~16.3-16.5s). This
is far short of the first attempt's catastrophic (minutes-scale) blowup --
Step 4's batching clearly did fix the "nothing for XLA to compile once"
root cause -- but it is still a real, reproducible net loss at the scale
that matters, so it was reverted. **Do not re-add `jax.jit` to this
specific call without re-measuring at both benchmark scales and checking
for cross-jit interaction effects if any other `jax.jit` call exists in the
same code path.**

### Shape-bucket scaling: `jax.jit`/`jax.vmap` internals research

The per-update shape-bucketing in `compute_batched_algorithm_component` (see
Step 4) exists because subjects don't share argument shape at a given update
in general under incremental/staggered recruitment. A natural worry: how
many distinct buckets could a real trial actually produce, and does
`jax.jit` offer any way to tame that? Researched JAX's own documentation and
maintainer discussions directly (rather than working from priors) to answer
this precisely:

- **Padding + shape-bucketing is JAX's own recommended pattern for ragged
  data**, not a workaround specific to this codebase: "JAX does not
  currently have any facility for computation on ragged arrays, and `vmap`
  is built with homogeneous batches in mind... the best option is to pad all
  batches to the same length, and then use `vmap` on the padded version,"
  and for reducing bucket count specifically: "bucket your data by size...
  and pad to the maximum size of the bucket." This matches the two-level
  structure already in place (pad the time axis fully; bucket-then-exact-shape
  the per-update row axis).
- **`jax.jit`'s compilation cache keys on shape/dtype/pytree-treedef**, and
  a Python `for` loop over shape-buckets, when traced under `jit`, is fully
  unrolled into the compiled graph: "if you need many loop iterations, XLA
  must optimize thousands of unrolled HLO instructions." This is the
  root-cause explanation for the Step 3 second-attempt regression above
  (more subjects -> more likely more distinct buckets -> a bigger single
  compiled program) and means **`jax.jit` would make a "many shape buckets"
  scenario strictly worse, not better** -- the eager per-bucket dispatch
  path already in place is the right one for this axis regardless of jit.
- **Measured, not assumed, how many buckets the existing benchmark fixtures
  actually produce**, via a new always-on diagnostic log line
  ("`Algorithm shape-bucket fan-out: ...`", added at the same point
  `update_layer` is built): small scale (n=20, 6 updates) produces 11 total
  buckets, max 2 in any single update; medium scale (n=100, 13 updates)
  produces 49 total, max 5 in any single update -- both far from the
  pathological one-bucket-per-subject case. These are synthetic fixtures
  with a specific recruitment pattern (an initial cohort + a staggered
  trickle), not necessarily representative of real trials -- the logging is
  there so this can be checked directly against real data before assuming
  either way.
- **If real trials do show large bucket counts**, the only general fix
  found is a real interface change, not a quick one: the row axis being
  bucketed can't be padded the way the time axis was, because
  `alg_update_func`s like `RL_least_squares_loss_regularized` **sum over
  it** -- padding would silently change the answer unless padded rows are
  guaranteed to contribute exactly zero to that sum, which isn't
  determinable in general for an arbitrary user-supplied function without
  adding an explicit mask argument to the estimating-function contract
  (affecting every shipped `alg_update_func`/`inference_func`). Not started;
  this is a scope/risk judgment call, worth pursuing only if the new logging
  shows bucket counts actually approaching subject counts in practice.

**Step 5 is deferred and re-scoped lower priority than initially expected.**
The one existing precedent for exploiting this sparsity in this codebase --
`DeploymentConditioningMonitor`'s "incremental" mode -- turned out to be
broken (see bug #1 below) and its regression tests were skipped, so this
isn't "port a working technique," it's "design and validate one from
scratch" in code where a subtly wrong block assembly produces a
plausible-looking but silently wrong variance estimate. Do this only after
Step 2's consolidation gives one shared, tested implementation to modify
once, and only if the benchmark still shows it matters at realistic update
counts after Steps 3-4.

### Step 6: `combine_updates_into_one_vmap` (per-update loop collapse, done)

Motivation: `compute_batched_algorithm_component`'s outer Python loop over
updates (`U` ~11-31 at oralytics scale) exists only because each update has a
different active-subject set/shape, not because of a real data dependency --
every update's `betas[u]` is already concrete by the time this runs. Two
CPU-only toy experiments (dense, no staggered recruitment) suggested
collapsing this loop plus the per-update shape-bucket loop into ONE combined
`jax.vmap` call (spanning update x subject at once) could reduce peak RSS
and wall-clock time, with the gap *widening*, not narrowing, at larger scale
-- a different, more promising signal than chunking-alone or
JIT-with-explicit-residuals, both of which looked fine on a toy and did not
transfer to the real hot path (see Step 3's second attempt, above).

Implemented as a new opt-in, `combine_updates_into_one_vmap: bool = False`,
(since made the AUTO-resolved default, `bool | None = None` -- see Step 8)
following this file's established index-convention/opt-in pattern
(`lifejacket/batched_weighted_estimating_function_stack.py`,
`lifejacket/post_deployment_analysis.py`). Mechanism: every update's ragged
positions are self-padded (reusing `self_pad_ragged_args_and_build_mask`,
extended with an optional `target_max_length` so a per-update local pad can
be re-padded to one shared global length) up to one length shared by every
update, instead of each update's own local max; a subject invalid at a given
update is self-padded with that subject's OWN real args from their nearest
valid update (`update_fill_index` -- the same forward/backward-fill
construction `ActionProbLayerPrecompute.fill_col_index` already uses along
the time axis, applied here along the update axis instead); the resulting
`(N, U, *shape)` tensors are flattened to `(N*U, *shape)` and evaluated via
one `jax.vmap` call, then reshaped back and gated by the same
`valid_update * rl_weight_products` multiply the original loop already
applies. Requires `alg_update_func_args_mask_index >= 0` (raises `ValueError`
otherwise); requires `alg_update_func_args_previous_betas_index < 0` (raises
`NotImplementedError` otherwise) -- a repo-wide grep found the only two
shipped functions that use `previous_betas_index >= 0`
(`RL_least_squares_loss_regularized_previous_betas_as_args[_hard_clipping].py`)
index into it via a separate `post_update_policy_nums` offset array, never
via `previous_betas.shape[0]`/`len(...)`, so no shipped function is known to
depend on its length -- but the real target function
(`oralytics_RL_estimating_function`, sibling `adjusted-sandwich-user` repo)
has no `previous_betas` argument at all (index always -1), and no golden
fixture in this repo exercises `previous_betas_index >= 0` either, so
supporting that combination was left unbuilt (loud error) rather than
shipped unverified. The inference side (`compute_batched_inference_outputs`)
has no per-update axis at all and was deliberately left out of scope.

**Measured, not assumed, against this repo's own real fixtures** (not
synthetic): both `tests/benchmarks` fixtures' `rl_update_args` already carry
genuinely ragged, staggered-recruitment per-subject/per-update history
lengths (confirmed by direct inspection), so a mask-aware variant of the
shipped `RL_least_squares_loss_regularized` loss
(`RL_least_squares_loss_regularized_masked.py`, algebraically identical on
real data -- the appended mask is the only difference) could be run through
a full `analyze_dataset` call on the real fixture data three ways --
original/unmasked, masked-uncombined, masked-combined -- and compared
directly (see `tests/benchmarks/test_combine_updates_into_one_vmap_benchmark.py`).
All three match the existing golden fixture to float32 noise at both scales.
Isolated-process (to avoid `ru_maxrss`'s monotonic-watermark contamination
across runs in one process) timing/RSS for the `jax.jacrev(...)` forward/backward
split:

| scale (n, T, U) | variant | forward_vjp | backward_vmap_chunks | peak RSS after forward | peak RSS after backward |
|---|---|---|---|---|---|
| small (20, 6, 6) | baseline | 1.98s | 1.10s | 450.7 MB | 570.2 MB |
| small (20, 6, 6) | combined | 1.41s | 0.71s | 400.6 MB | 489.4 MB |
| medium (100, 10, 13) | baseline | 3.91s | 1.97s | 645.1 MB | 838.0 MB |
| medium (100, 10, 13) | combined | 1.51s | 0.78s | 459.8 MB | 579.7 MB |

Unlike the chunking and JIT-with-explicit-residuals attempts above, this
result **transfers positively** at both fixture scales, and the gap widens
with scale exactly as the toy predicted (small: ~29% faster / ~14% lower
peak RSS; medium: ~61% faster / ~31% lower peak RSS) -- despite these
fixtures having a far smaller `beta_dim` (4) than oralytics (135), so this is
not yet a verdict at oralytics scale, only a real (not toy), reproducible,
same-direction signal at both scales this repo's fixtures can exercise.
Real-scale (oralytics) verification is a deliberate follow-up, out of scope
for this step (see task handoff), to be run under the same memory watchdog
used for the original OOM repro.

### Step 7: small-sample correction feature removed (not optimized, done)

A real, watchdog-protected oralytics-scale run (`num_users=50`, `beta_dim=135`,
`out_dim~3130`) showed the per-subject `(num_subjects, out_dim, out_dim)`
outer-product tensor built by `get_avg_weighted_estimating_function_stacks_and_aux_values`
for the joint adjusted meat matrix -- plus its `(num_subjects, theta_dim,
theta_dim)` classical-meat sibling -- as the dominant real-scale memory cost:
the first tensor alone is ~1.96GB at that size, and both were carried through
`jax.vjp`'s residual tape into `perform_desired_small_sample_correction`
(`lifejacket/small_sample_corrections.py`), which existed solely to apply one
of four small-sample corrections (`none`, `Z1theta`, `Z2theta`, `Z3theta`) to
the meat matrices before forming the sandwich.

An initial fix (committed earlier in this same investigation) kept the
feature but added an opt-in fast path: when the requested correction gives
every subject the same scalar weight (`none`/`Z1theta`), `mean_i(w *
outer(stack_i, stack_i)) == w * (stacks.T @ stacks) / N` exactly (`sum_i
outer(x_i, x_i) == X.T @ X`), so the per-subject tensor never needed to be
materialized for those two values. `Z2theta`/`Z3theta` need genuinely
per-subject leverage weights and were left on the original
`jax.vmap(jnp.outer)` path.

In practice, nothing in this repo, its test suite, or its sibling
`adjusted-sandwich-user` runner repo ever passed anything other than
`"none"` for `small_sample_correction` -- a repo-wide grep confirmed
`Z1theta`/`Z2theta`/`Z3theta` appeared nowhere outside the enum and the
module implementing them. Given that, the feature was removed outright at
the user's explicit request rather than kept behind an opt-in flag:
`SmallSampleCorrections`, `small_sample_corrections.py`, and the
`small_sample_correction` parameter/CLI flag were deleted everywhere in this
repo (from `analyze_dataset`, `construct_classical_and_adjusted_sandwiches`,
the CLI, and every call/test site).
`get_avg_weighted_estimating_function_stacks_and_aux_values` now
unconditionally computes `stacks.T @ stacks` (and
`inference_component.T @ inference_component`) for the joint/classical meat
matrices -- the same identity as the fast path above, just with no
conditional and no other code path left that would ever need the
per-subject tensor. `construct_classical_and_adjusted_sandwiches` divides
those pre-summed matrices by `num_subjects` directly in place of the removed
`perform_desired_small_sample_correction` call; the two
`per_subject_*_corrections` outputs are kept as trivial `jnp.ones(num_subjects)`
placeholders purely so `analyze_dataset`'s `debug_pieces.pkl` output shape
(checked by `tests/utils.py`'s exact-key-list assertion) is undisturbed.

Verified numerically identical to pre-removal `"none"` behavior on both
`tests/benchmarks` golden fixtures (full suite re-run after the deletion,
same tolerances as before this step). At these fixtures' own scale
(`out_dim` 28 (small) / 56 (medium), `num_subjects` 20 / 100) the deleted
tensor was only ~0.06MB / ~1.2MB to begin with, so isolated-process peak RSS
for the whole `analyze_dataset` call is statistically unchanged
(`/usr/bin/time -l` "maximum resident set size": small ~702.6MB before this
step vs. ~704.2MB after; medium ~1163.1MB vs. ~1163.0MB) -- these fixtures
are far too small to exercise the cost this step removes, exactly as already
noted for Step 6 above. The `Peak RSS after compute_meat_matrices` log line
(same `resource.getrusage`-based instrumentation used throughout this file)
confirms the phase itself now adds only ~2.5MB on top of the preceding
`backward_vmap_chunks` phase at both scales -- consistent with never
constructing the per-subject tensor, even transiently. A real oralytics-scale
before/after comparison remains a deliberate follow-up (same watchdog
process as the original OOM repro), since this repo's own fixtures cannot
reach the ~1.96GB tensor size that motivated this step.

### Step 8: auto-enabled defaults for `combine_updates_into_one_vmap` and `jacobian_row_chunk_size` (done)

Steps 6's `combine_updates_into_one_vmap` and the chunked-backward
`jacobian_row_chunk_size` (both introduced above as explicit opt-ins) were
verified this same investigation to be *necessary* for the real
70-subject/31-update/`beta_dim=135` oralytics study (`out_dim` ~4185) to
complete on a 24GB machine -- which made "the user must know to pass both,
and pick a chunk size" too much to ask for the primary real use case. Both
now default to AUTO, with the resolution logged at INFO and explicit values
preserved as overrides in both directions:

- `combine_updates_into_one_vmap: bool | None = None` (was `bool = False`).
  `None` = auto: enabled exactly when eligible
  (`alg_update_func_args_mask_index >= 0` -- combining cannot be auto-enabled
  without the mask opt-in, since it needs a mask-aware `alg_update_func` --
  AND `alg_update_func_args_previous_betas_index < 0`), silently off
  otherwise; `True` keeps every loud ineligibility error verbatim; `False`
  forces the loop even when eligible. Resolution happens in
  `resolve_combine_updates_into_one_vmap`
  (`batched_weighted_estimating_function_stack.py`, pure, directly
  unit-tested) at the single choke point both real entry paths flow through
  (`get_avg_weighted_estimating_function_stacks_and_aux_values`, immediately
  before `build_update_layer_precompute`); the `precomputed_layers`
  diagnostic path bypasses it exactly as before. The two structural
  invariants only checkable *during* the combined-block precompute (one
  shape-bucket per update after global padding; identical non-ragged shapes
  across updates) fall back in auto mode -- `build_update_layer_precompute`
  grew a `combine_is_required: bool = True` argument, and when it is False
  (auto), the self-contained combined block's `ValueError` is caught, a
  WARNING naming the violated invariant is logged, and the precompute
  returns with `combined_*` left `None`, i.e. the shipped, golden-tested
  default loop. This is cheap by construction (the default per-update bucket
  structures are always fully built *before* the combined block starts --
  the block was deliberately written independent of the main loop's state)
  and never a correctness risk (the fallback IS the default path, and the
  two paths were already verified numerically identical at both fixture
  scales in Step 6). Erroring instead (option (b) considered) would have
  turned previously-working masked studies with unusual shape structure into
  hard failures purely because a speed heuristic got more ambitious.

- `jacobian_row_chunk_size: int | None = None`, with `None` now meaning
  auto rather than "never chunk": unchunked (the original single eager
  `jax.vmap(pullback)`) when `out_dim <= 512`, else
  `max(1, min(64, 65536 // out_dim))`; a NEW `0` sentinel forces the
  original unchunked path explicitly (0 was previously a `ValueError`); a
  positive int stays an explicit chunk size, unchanged, including the
  jitted-chunk mechanism. Resolution happens in
  `resolve_jacobian_row_chunk_size` (`post_deployment_analysis.py`, pure,
  directly unit-tested), called the moment `out_dim` is first known inside
  `construct_classical_and_adjusted_sandwiches`. The constants
  (`JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM = 512`,
  `JACOBIAN_AUTO_ROW_BUDGET = 65536`, `JACOBIAN_AUTO_MAX_CHUNK = 64`,
  `lifejacket/constants.py`) are grounded in this investigation's real
  oralytics measurements (all `beta_dim=135`, 24GB Mac, combined+masked):
  unchunked crashed at `out_dim` ~1500 while chunk 64 worked there (8.4s
  backward) and at ~2000 (21.8s); chunk 64 CRASHED at ~3100 while chunk 16
  worked (69-71s, peak RSS ~2.1GB); chunk 16 worked at ~4185 (174s, peak
  RSS ~3.1GB). Treating `chunk_size * out_dim` as the budget, the crash
  boundary sits in (128k, 198k]; 65,536 is roughly one-third of the crash
  point and at-or-below every verified-safe budget in the dangerous
  `out_dim >= 3100` regime, i.e. calibrated conservatively rather than to
  the edge (headroom over the last ~10% of speed). The 512 threshold keeps
  every problem at this repo's own fixture scales (`out_dim` 28/56, where
  the jitted chunk path measured ~2.4x SLOWER than eager -- see the
  chunking caveats above) on the untouched eager path; nothing was measured
  between `out_dim` 56 and ~1500, so the conservative end of that gap was
  chosen (worst case above the threshold: one ~1.4-1.8s compile, plus one
  for a remainder chunk, on problems whose backward already costs seconds;
  worst case below it would be an OOM'd machine). `out_dim` is honestly a
  single-variable PROXY (`U*beta_dim + theta_dim` correlates with, but does
  not equal, the true per-cotangent backward footprint, which also grows
  with `num_subjects x num_updates x history length`), calibrated on ONE
  study shape on ONE machine -- the explicit parameter is the documented
  escape hatch in both directions (smaller to use less memory, larger or 0
  to go faster when memory is plentiful).

Default behavior at this repo's own benchmark scales is unchanged by
construction (unmasked fixtures resolve combine to off; `out_dim` 28/56
resolve chunking to the untouched unchunked eager call), verified by
re-running the small and medium (`-m slow`) golden benchmark suites
unchanged after the change. New unit tests cover both resolvers' decision
matrices/boundaries, the auto-fallback-on-structural-violation path (both
invariants; explicit `True` still errors loudly), and numeric identity
between auto-resolved and explicitly-passed equivalents (combine: exact;
chunked-vs-unchunked backward: float32 noise only). CLI semantics follow the
existing conventions: `--combine_updates_into_one_vmap` keeps `type=bool`
with `default=None` (click's BOOL already parses True/False tokens, and the
`-1000` int-sentinel convention is reserved for argument-INDEX options, not
feature toggles); `--jacobian_row_chunk_size` keeps `type=int, default=None`
with `0` as the explicit "no chunking" value.

### Rejected alternatives

- **Migrate to float64 for speed.** This is CPU-only (no GPU/TPU config
  anywhere in the repo), and a direct microbenchmark showed eager-vs-jitted
  dispatch overhead differs by ~1000x for a representative nested
  `jax.jacrev` call -- two to three orders of magnitude more than any
  float32-vs-float64 compute difference on CPU. Not worth the ULP-level
  change in every existing regression fixture for no real speed benefit.
- **Move the post-hoc numpy linear algebra to `jax.numpy`**
  (`stabilize_joint_bread_if_necessary`, `form_adjusted_meat_adjustments_directly`)
  to be "more JAX-native." All of
  this runs *after* `jax.jacrev` has already returned concrete arrays, never
  inside a differentiated path, and operates on small one-off matrices
  outside any jit boundary -- numpy straight into LAPACK is the faster
  choice there, not a bug.

## Bugs found and fixed during this investigation

Three real correctness bugs surfaced while diagnosing the performance issue.
None is itself a performance problem, but all three are fixed as part of
this work since they were found here:

1. **`DeploymentConditioningMonitor`'s incremental Jacobian stitching was
   broken.** `construct_phi_dot_bar_so_far` used `jnp.block([a, zeros, b])`
   with a **flat** list, which concatenates along the last axis only
   (`hstack`), instead of a **nested** list, which is required to build a
   square 2D block matrix. Reproduced directly: after the first incremental
   update this silently produced a `(2, 8)` matrix instead of the required
   `(4, 4)`, corrupting `self.latest_phi_dot_bar` (and therefore every
   conditioning check downstream) from the second incremental call onward.
   This is why the file's two regression tests were
   `@pytest.mark.skip(reason="TODO fix if monitoring becomes more important")`.
   Fixed in `deployment_conditioning_monitor.py`; one of the two tests is now
   un-skipped, fixed (its fixture data had independent, pre-existing
   structural bugs -- wrong dict nesting, an off-by-one in which policy's
   update args were included, and a policy-numbering mismatch -- unrelated
   to the `jnp.block` bug), and extended to three successive updates so it
   exercises the case where the cached block is *larger* than the new one,
   not just equal to it. The other test remains skipped with an updated,
   accurate reason: it references an `analyze_dataset()` return key that no
   longer exists and passes `None` for required callables, independent of
   this fix, and needs its own rewrite.

   The fix was extracted into a shared, directly-tested helper,
   `append_new_block_row_to_block_lower_triangular_matrix`
   (`helper_functions.py`), and `get_datum_for_blowup_supervised_learning.py`'s
   `construct_premature_classical_and_adjusted_sandwiches` -- which already
   did this correctly with its own inline `jnp.block` call -- was switched to
   use the same helper. An adversarial code review of the initial patch
   correctly flagged that fixing the bug inline, without reusing or even
   referencing the already-correct implementation 400 lines away in another
   file, repeated exactly the kind of triplicated-logic pattern this ADR
   otherwise argues against.
2. **`analyze_dataset` swapped `raw_joint_bread_matrix` and
   `stabilized_joint_bread_matrix`** when unpacking
   `construct_classical_and_adjusted_sandwiches`'s return tuple. Didn't
   affect the returned `adjusted_sandwich_var_estimate` (that comes from a
   correctly-unpacked value), but it mislabeled both matrices in
   `debug_pieces.pkl` and fed the wrong one into the local-linearization
   diagnostic and the "Joint bread condition number" log line. Fixed by
   correcting the unpacking order; verified against the full unit suite, the
   closed-form regression tests, and all four integration tests.
3. **`stack_batched_arg_lists_into_tensors` (`vmap_helpers.py`) and its
   byte-identical duplicate `stack_batched_arg_lists_into_tensor`
   (`calculate_derivatives.py`) misclassify a 0-D (scalar-shaped) array.**
   Found while attempting Step 3 (below): the dispatch logic decides how to
   stack a batch of per-subject argument values by checking
   `isinstance(x, (jnp.ndarray, np.ndarray))`, then `x.ndim`. A plain Python
   `float`/`int` correctly takes the final "list of scalars" branch (it's
   never `isinstance(..., jnp.ndarray)`) -- but the *same value*, once it
   becomes a 0-D array (which is exactly what happens to a scalar argument
   crossing a `jax.jit` boundary, and can also happen with plain eager 0-D
   arrays passed directly), *is* `isinstance(..., jnp.ndarray)` yet has
   `ndim == 0`, not `1`, so it fell into the "vector (1D array)" branch and
   failed that branch's own `assert batched_arg_list[0].ndim == 1`. Fixed in
   both files by checking `ndim` explicitly at each branch (a 0-D array now
   correctly falls through to the scalar branch, same as a plain Python
   scalar) rather than assuming "isinstance array, not 2-D" implies 1-D.
   Regression-tested directly in the new `tests/unit_tests/test_vmap_helpers.py`
   (reverting the fix and re-running those tests reproduces the exact
   `AssertionError` this fix resolves). This bug was latent rather than
   already triggering some existing, shipped code path -- none of the
   *current* call sites of either function pass a 0-D array through it --
   but it will matter again for any future `jax.jit` attempt on code using
   this shared stacking helper (including a properly-`vmap`'d Step 3/4), so
   it's worth having fixed now rather than rediscovering later.

Two further findings were **not** fixed here, only flagged, since they're out
of scope for this ticket:

- The local-linearization diagnostic's `jnp.asarray(..., dtype=jnp.float64)`
  calls are silent no-ops (no `jax_enable_x64` is set anywhere in the repo),
  so the diagnostic's stated intent -- "ensure float64 for diagnostics even
  if upstream ran in float32" -- isn't actually happening. Both sides of its
  comparisons are float32 regardless, so this doesn't break the diagnostic's
  self-consistency, but the extra precision it's trying to buy for its
  condition-number thresholds (`1e12`, etc.) isn't real. Worth its own ticket
  if that precision turns out to matter.
- A **second**, unrelated bare `breakpoint()` landmine exists in
  `tests/simulators_and_runners/functions_to_pass_to_analysis/RL_least_squares_loss_regularized_previous_betas_as_args_hard_clipping.py`
  (separate from the one already known in
  `form_adjusted_meat_adjustments_directly.py:310`). It causes
  `test_RL_center_1_inf_center_1_steep_3_incremental_2_decs_btw_update_previous_betas_given`
  to fail with `BdbQuit` when run after other tests in the same pytest
  session (closed stdin), while passing cleanly in isolation -- an
  order-dependent flake, not something introduced by this work. Worth its
  own cleanup ticket.

## Consequences

- Every future step in this plan is gated by `tests/benchmarks` (numerical
  match at `rtol=1e-6`, plus a printed timing breakdown) in addition to the
  existing unit/integration suites -- run it before *and* after each change.
- There is no CI job that runs `pytest` at all today (`.github/workflows`
  only builds and checks the package). Until that changes, running the full
  suite (`unit`, `integration`, `benchmarks`) locally before merging any step
  of this plan is the only safety net.
- `tests/integration_tests` takes several minutes per test (confirmed: 162s
  for one `n=100, T=10` scenario) -- budget for this when validating later
  steps; it's slow because it's exercising exactly the code path this plan
  is about, not because of test-harness overhead.
- The full test suite, the correctness fixes, and the benchmark
  instrumentation were checked with an independent adversarial code review
  (four parallel finders: cleanup/simplification, a removed-behavior audit,
  direct re-verification of the rewritten monitor test, and a cross-file
  re-indentation trace) in addition to running the suites directly. All
  findings that indicated a real defect (a phase timer that logged a
  misleading duration for work that didn't run, a timer that silently summed
  two distinct operations, and the triplicated-block-logic issue noted above)
  were fixed; findings that were out of scope (the `run_local_synthetic.sh`
  timing-idiom inconsistency, per-parametrization fixture reloading in the
  benchmark test) were deliberately left as-is.
- **Measure the actual compile/run tradeoff before trusting a `jax.jit`
  design, even one that looks textbook-correct.** Step 3 followed the
  documented best practice (jit the outermost transform, close over static
  config, fix concretization breaks) and was still a severe regression,
  because compile cost for a fully-Python-unrolled graph isn't bounded the
  way eager dispatch cost is -- and it doesn't degrade gracefully; it got
  categorically worse (minutes, not a bounded multiple) at 5x the subject
  count. Do not extrapolate a small-scale JIT win to larger scale, or vice
  versa, without re-measuring at both.
- A second adversarial code review, run after Steps 1-3 (including the Step 3
  revert), found three more real issues, all fixed: (a)
  `require_action_probabilities_in_analysis_df_can_be_reconstructed`'s
  rewrite could crash with a bare `KeyError` on malformed input (non-blank
  args for an inactive row) instead of the clear, contract-referencing error
  the codebase's error-message conventions call for -- fixed by collecting
  such mismatches into an explicit `ValueError` listing the offending
  `(decision_time, subject_id)` pairs; (b) its actual-value lookup dict was
  built with `DataFrame.iterrows()`, a known-slow pattern inside a function
  whose whole point is speed -- replaced with a `zip`-over-`to_numpy()`-columns
  construction; (c) `stack_batched_arg_lists_into_tensor`
  (`calculate_derivatives.py`) was still a hand-duplicated copy of
  `stack_batched_arg_lists_into_tensors` (`vmap_helpers.py`) even after this
  same session fixed the identical 0-D-array bug in both copies separately --
  consolidated to one definition in `vmap_helpers.py`, matching the
  dedup pattern already used elsewhere in this work (bug 1's block-row helper).
  `tests/unit_tests/test_input_checks.py` and `test_vmap_helpers.py` are new;
  neither `input_checks.py` nor `vmap_helpers.py` had a dedicated test file
  before this session.
- A third adversarial code review, run after implementing Step 4, fanned out
  ten independent finder angles (line-by-line diff scan, removed-behavior
  audit, cross-file tracer, language-pitfall specialist, wrapper/proxy
  correctness, reuse, simplification, efficiency, altitude, and CLAUDE.md
  conventions) plus direct empirical re-verification (running the full test
  suite, an integration test, and hand-written JAX repros of the specific
  hazards claimed). The concrete correctness regressions it found are listed
  under "Step 4," above, and were fixed immediately; the lower-priority,
  deliberately-deferred gaps (contiguity-guard placement, the two
  independent weight-window implementations, the residual data-check cost,
  the repeated structural precompute) are also documented there rather than
  fixed, so the next person touching this code can tell "considered and
  deferred" from "not noticed."
