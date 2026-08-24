# 0001. Performance plan for the adjusted sandwich computation

- Status: Accepted (Steps 0-2 and three correctness bugs below are done; Step 3 was attempted and
  reverted -- see "Step 3 attempted and reverted" below; Steps 4-5 are proposed, not yet implemented)
- Date: 2026-08-24
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
| 3 | Wrap the `jax.jacrev` hot path in `jax.jit` | **Attempted and reverted** -- see below; net regression, not a win |
| 4 | Convert the ragged per-subject/per-update Python loops to padded + masked `jax.vmap`/`jax.lax.scan` | Proposed, high effort/risk -- now believed to be a **prerequisite for Step 3 to help at all**, not an independent follow-on |
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

### Rejected alternatives

- **Migrate to float64 for speed.** This is CPU-only (no GPU/TPU config
  anywhere in the repo), and a direct microbenchmark showed eager-vs-jitted
  dispatch overhead differs by ~1000x for a representative nested
  `jax.jacrev` call -- two to three orders of magnitude more than any
  float32-vs-float64 compute difference on CPU. Not worth the ULP-level
  change in every existing regression fixture for no real speed benefit.
- **Move the post-hoc numpy linear algebra to `jax.numpy`**
  (`stabilize_joint_bread_if_necessary`, `form_adjusted_meat_adjustments_directly`,
  the small-sample-correction inversion) to be "more JAX-native." All of
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
