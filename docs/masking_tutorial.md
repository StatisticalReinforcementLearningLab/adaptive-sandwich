# Masking tutorial: fast analysis of ragged (staggered-recruitment) studies

`analyze_dataset` (and the `lifejacket analyze` CLI) has an opt-in
padding/masking mechanism that can turn a slow or out-of-memory analysis of a
real staggered-recruitment study into one that completes in minutes. Opting in
requires a small, mechanical change to your algorithm-update (and/or
inference) function plus two extra arguments. This tutorial walks through the
whole thing using the two mask-aware functions that already ship with the
project.

## 1. When you need this (and when you don't)

Under incremental/staggered recruitment, each subject's per-decision-time
arrays (`state`, `action`, `rewards`, ...) grow with how long *that subject*
has been in the study, so at any given algorithm update different subjects
have different array lengths. `jax.vmap` requires identical shapes, so by
default lifejacket groups subjects into exact-shape "buckets" and dispatches
one vmap call per (update, bucket). With many distinct history lengths that
means many tiny dispatches — one real 70-subject study produced 146 buckets —
and both wall-clock time and memory suffer.

Check the log. Every run prints, unconditionally:

```
Algorithm shape-bucket fan-out: <U> update(s), <N> subject(s), <B> total bucket(s) (max <M> in a single update).
```

If the total bucket count is close to one per update, you probably don't need this —
that's already the ideal. That is typically the case when every subject is
recruited at once (uniform shapes), and small studies are usually fast
regardless. But if buckets grow toward `updates x subjects`, or your analysis
is slow or runs out of memory, read on.

## 2. The core idea: self-padding plus a mask

With masking on, lifejacket pads every subject's ragged arrays at each update
up to the longest length seen at that update, so every subject has the same
shape and the whole update runs as **one** vmap dispatch. Two details make
this correct:

- **Padding repeats the subject's own last real row** — never a fabricated
  zero. A zero can poison an otherwise-masked-away row through a non-linear op
  before the mask is applied (e.g. `1/act_prob` with `act_prob==0` gives
  `inf`, and `0 * inf` is `nan` in IEEE 754, not `0`). A repeated real row is
  always a valid domain point.
- **A validity mask is appended as a new last argument** to your function:
  a float array shaped `(num_decision_times,)`, `1.0` for a real row, `0.0`
  for a padded one.

The padded rows hold real-looking values on purpose. Correctness does not
come from the padded *values* being harmless — it comes from your function
zeroing their **contribution**: multiplying by the mask before every sum (or
mean, or matmul-that-sums) over the decision-time axis. Once every padded
row's contribution to every reduction is exactly zero, the padded call is
mathematically identical to the unpadded per-subject one, whatever the padded
rows contain.

## 3. Making your function mask-aware

Two steps, both mechanical:

1. Add `mask` as a new **last** parameter.
2. Multiply each per-decision-time quantity by `mask` (reshaped to broadcast,
   e.g. `mask.reshape(-1, 1)` against a 2D array) **before** any reduction
   over the decision-time axis.

Here is the shipped test example,
`tests/simulators_and_runners/functions_to_pass_to_analysis/RL_least_squares_loss_regularized.py`
(a least-squares loss whose only decision-time reduction is the sum of squared
residuals):

```python
def RL_least_squares_loss_regularized(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action1probtimes,
    action_centering,
    lambda_,
    n,
):
    ...
    return (
        jnp.einsum(
            "ij->",
            (
                rewards
                - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
                - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
            )
            ** 2,
        )
        + jnp.dot(beta_est, beta_est) * lambda_ / n
    )
```

and its mask-aware sibling `RL_least_squares_loss_regularized_masked.py`
(same directory). The entire diff is the new last parameter and the
mask-gated sum:

```python
def RL_least_squares_loss_regularized_masked(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action1probtimes,
    action_centering,
    lambda_,
    n,
    mask,          # NEW: appended last
):
    ...
    residual_sq = (
        rewards
        - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
        - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
    ) ** 2
    masked_sum = jnp.einsum("ij,i->", residual_sq, mask)   # was "ij->"
    return masked_sum + jnp.dot(beta_est, beta_est) * lambda_ / n
```

Note what did *not* change: the regularization term `jnp.dot(beta_est,
beta_est) * lambda_ / n` involves no per-decision-time rows, so it needs no
masking. Scalars and fixed-size arguments are never padded and never masked.

On real, unpadded data (mask all ones) the masked function is bit-for-bit
identical to the original.

### The danger case

The failure mode to fear is a reduction the mask doesn't gate: padded rows are
copies of a real row, so an unmasked sum silently adds real-looking terms and
**changes the answer without any error**. So check your work the same way the
shipped examples were checked: run the analysis twice on the same small (but
genuinely ragged) dataset — once with the original function and no mask
arguments, once with the masked function and the mask arguments — and compare
`theta_est`, `adjusted_sandwich_var_estimate`, and
`classical_sandwich_var_estimate`. They must agree to float32 noise. That is
exactly what
`tests/benchmarks/test_combine_updates_into_one_vmap_benchmark.py` does for
the pair above (`np.testing.assert_allclose` with `rtol=1e-5`).

## 4. A real study: the oralytics estimating function

The sibling `adjusted-sandwich-user` repo carries the real-study version:
`functions_to_pass_to_analysis/oralytics_RL_estimating_function.py`, a
Bayesian linear-regression update whose two decision-time reductions are both
matrix products over stacked features:

```python
def oralytics_RL_estimating_function(
    beta,            # 0
    n_users,         # 1
    state,           # 2   (num_decision_times, 5)
    action,          # 3   (num_decision_times, 1)
    act_prob,        # 4   (num_decision_times, 1)
    decision_times,  # 5   (num_decision_times,)
    rewards,         # 6   (num_decision_times, 1)
    prior_mu,        # 7
    prior_sigma_inv, # 8
    init_noise_var,  # 9
):
    ...
    stacked_phis = jnp.hstack([state, (act_prob * state), (action - act_prob) * state])
    stacked_phis_T = stacked_phis.T

    vector_1 = (
        stacked_phis_T @ (rewards.reshape(-1, 1) - stacked_phis @ mu)
    ).flatten() / init_noise_var
    ...
    matrix_3 = jnp.triu(stacked_phis_T @ stacked_phis) / init_noise_var + (...)
```

Its mask-aware sibling `oralytics_RL_estimating_function_masked.py` adds
`mask` as an 11th parameter (index 10) and zeroes each padded row of
`stacked_phis` once, before both reductions:

```python
    stacked_phis = jnp.hstack([state, (act_prob * state), (action - act_prob) * state])

    # Zero out every padded row BEFORE either decision-time sum below.
    masked_phis = stacked_phis * mask.reshape(-1, 1)
    masked_phis_T = masked_phis.T

    vector_1 = (
        masked_phis_T @ (rewards.reshape(-1, 1) - masked_phis @ mu)
    ).flatten() / init_noise_var
    ...
    matrix_3 = jnp.triu(masked_phis_T @ masked_phis) / init_noise_var + (...)
```

One multiplication suffices here even though `rewards` is never masked
directly: a zeroed row of `masked_phis` contributes a zero row/column to
`masked_phis_T @ (...)` and a zero outer product to
`masked_phis_T @ masked_phis`, so every term of both sums that touches a
padded row already contains a zero factor. You need the mask applied at least
once per row, per reduction — not once per array.

## 5. Wiring it up

Two pieces of information tell lifejacket how to pad and where the mask goes:

- **`mask_index`**: the index of the new mask argument. Because the mask is
  always appended as a new **last** argument, this must equal the *original*
  function's argument count (10 for both examples above — coincidentally, as
  each original function happens to take 10 arguments). Appending last means
  no other `*_index` option (`beta_index`, `previous_betas_index`,
  `action_prob_index`, ...) ever shifts.
- **`ragged_indices`**: every argument position shaped
  `(num_decision_times, ...)` — i.e. everything that grows with a subject's
  history. For the oralytics function that is `(2, 3, 4, 5, 6)` (`state`,
  `action`, `act_prob`, `decision_times`, `rewards`); for the RL test example
  it is `(1, 2, 3, 4, 5, 6)`.

Every ragged position must agree, per subject, on how many real rows that
subject has (they all represent the same "history so far"). If you list a
position that isn't actually ragged in that sense, padding fails immediately
with a `ValueError` naming the subject and the disagreeing lengths — a loud
error at padding time, never silent corruption. If you *miss* a genuinely
ragged position, that position keeps its per-subject lengths while its
neighbors get padded, and the mismatch surfaces as a loud shape error (extra
shape buckets, a combining-invariant warning, or a broadcasting error inside
your function) — again, not a silently wrong number.

**Python API** (`lifejacket.post_deployment_analysis.analyze_dataset`):

```python
analyze_dataset(
    ...,
    alg_update_func=load_function_from_same_named_file(
        "functions_to_pass_to_analysis/oralytics_RL_estimating_function_masked.py"
    ),
    alg_update_func_args_mask_index=10,
    alg_update_func_args_ragged_indices=(2, 3, 4, 5, 6),
    ...,
)
```

Both default to off (`alg_update_func_args_mask_index=-1`; any negative value
means "unused"). Note that `alg_update_func_args` themselves do **not**
change: you keep supplying each subject's real, unpadded argument tuples, and
lifejacket does the padding and appends the mask internally.

**CLI** (`lifejacket analyze`): same names as flags; `ragged_indices` is a
repeated option (once per position):

```bash
uv run lifejacket analyze \
  ... \
  --alg_update_func_filename=functions_to_pass_to_analysis/oralytics_RL_estimating_function_masked.py \
  --alg_update_func_args_mask_index=10 \
  --alg_update_func_args_ragged_indices=2 \
  --alg_update_func_args_ragged_indices=3 \
  --alg_update_func_args_ragged_indices=4 \
  --alg_update_func_args_ragged_indices=5 \
  --alg_update_func_args_ragged_indices=6
```

In the `adjusted-sandwich-user` repo, `run_local_oralytics.sh` already threads
these through (it also accepts the space-separated shorthand
`--alg_update_func_args_ragged_indices="2 3 4 5 6"`); this is the actual
working full-scale invocation:

```bash
bash run_local_oralytics.sh -n 70 -q 1 \
  -l functions_to_pass_to_analysis/oralytics_RL_estimating_function_masked.py \
  --alg_update_func_args_mask_index=10 \
  --alg_update_func_args_ragged_indices=2 \
  --alg_update_func_args_ragged_indices=3 \
  --alg_update_func_args_ragged_indices=4 \
  --alg_update_func_args_ragged_indices=5 \
  --alg_update_func_args_ragged_indices=6
```

(`-n 70` = subjects, `-q 1` = suppress all data checks, `-l` = the
algorithm-update function file.)

**Inference side.** If your `inference_func` also takes ragged
per-decision-time arrays, the same mechanism exists as
`inference_func_args_mask_index` / `inference_func_args_ragged_indices`
(identical contract; the mask is again appended last). The two sides are
independent opt-ins.

## 6. What you get for free once masking is on

- **One shape bucket per update.** The fan-out log line above drops to one
  bucket per update regardless of enrollment pattern — in the real oralytics
  study, from 146 buckets at 70 subjects to one per update.
- **`combine_updates_into_one_vmap` auto-enables.** With a mask-aware
  `alg_update_func` (and `alg_update_func_args_previous_betas_index` unused),
  the per-update Python loop collapses further into exactly one `jax.vmap`
  spanning every (subject, update) pair — numerically identical, measured
  ~61% faster with ~31% lower peak RSS at the medium benchmark scale. The one
  limitation: the combined path does not support
  `alg_update_func_args_previous_betas_index >= 0` (a per-update
  variable-length prefix of betas doesn't fit its fixed-shape tensor layout),
  so auto mode simply leaves combining off in that case — you still keep the
  bucket consolidation.
- **The jacobian backward pass auto-chunks.** For large problems
  (`out_dim > 512`) the `jax.jacrev` backward pass is automatically split
  into memory-bounded chunks; you do not need to set
  `--jacobian_row_chunk_size` (it remains an explicit escape hatch in both
  directions).

Net effect, measured on the real 70-subject/31-update oralytics study: a
configuration that OOM-crashed a 24GB machine at less than half scale now
completes end-to-end in about 5 minutes at ~4.6GB peak RSS, with no flags
beyond the mask wiring above.

## 7. Troubleshooting

All failure modes are loud. The ones you're most likely to meet:

- **Disagreeing lengths across ragged positions** — you listed a position
  that doesn't share the subject's per-decision-time row count (or your data
  really is inconsistent):

  > `Subject 3 has disagreeing row counts across ragged_indices=(2, 3, 4, 5, 6): {...} -- every ragged position must agree, per subject, on how many real rows that subject has.`

  Fix `ragged_indices` to name exactly the `(num_decision_times, ...)`-shaped
  positions. (A scalar/non-array position instead raises
  `... has a non-array/non-sequence value at one of ragged_indices=... -- cannot determine its row length to pad.`)

- **Wrong `mask_index`** — it must equal the original argument count:

  > `mask_index=9 must equal the argument count (10) -- the mask is always appended as a new last argument, never inserted in the middle.`

- **Forcing combining alongside `previous_betas`** — only if you explicitly
  pass `combine_updates_into_one_vmap=True` with
  `alg_update_func_args_previous_betas_index >= 0` (auto mode just stays
  off):

  > `NotImplementedError: combine_updates_into_one_vmap does not support alg_update_func_args_previous_betas_index >= 0: previous_betas is a per-update variable-length prefix of betas, ...`

- **A WARNING that combining fell back** — auto-enabled combining checks two
  structural invariants mid-precompute (one bucket per update after global
  padding; identical non-ragged shapes across updates). If your study
  violates one, you'll see
  `combine_updates_into_one_vmap was auto-enabled but this study's argument structure violates a combining invariant (...). Continuing on the default per-update/per-bucket loop instead -- results are unaffected, only the combined-vmap speedup is lost.`
  This is informational: you keep the bucket consolidation.

For the full design rationale (why self-padding rather than zeros, why
bucketing existed in the first place, the measured numbers), see
`docs/adr/0001-adaptive-sandwich-performance-plan.md` and the docstrings of
`self_pad_ragged_args_and_build_mask` and `build_update_layer_precompute` in
`lifejacket/batched_weighted_estimating_function_stack.py`.
