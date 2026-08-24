"""
Padded+masked, jax.vmap-batched replacement for the ragged per-subject/
per-update Python loops in post_deployment_analysis.py's
get_avg_weighted_estimating_function_stacks_and_aux_values. See
docs/adr/0001-adaptive-sandwich-performance-plan.md, Step 4, for the full
design rationale.

Subjects share a common global calendar-time axis and a common global set of
policy updates, but per (subject, decision-time) and per (subject,
policy-update) argument tuples are only sometimes present (an empty tuple ()
marks "not active/in-study then"). This module converts that ragged
representation into fixed-size, padded arrays plus validity masks so the
per-subject and per-update loops can be replaced with a small, fixed number
of jax.vmap calls instead of one Python-dispatched call per subject (or per
subject per update).

Two hazards drive most of the design here:

1. jax.vmap cannot skip invalid cells -- every cell must be called with SOME
   concrete, in-domain input. A fabricated dummy (e.g. zeros) at a
   function-internal singularity risks NaN surviving into a shared
   parameter's gradient via a masking multiplication (0 * NaN = NaN), even
   though the forward output is masked to a harmless value elsewhere. Every
   padded-cell input here is instead "self-padded": the same subject's own
   real data from their nearest active time, which is guaranteed to be a
   valid domain point. (build_threaded_action_prob_beta_tensor's
   jnp.where-based beta substitution has its own, separate select-based
   protection against this for the beta argument specifically -- see
   tests/unit_tests/test_batched_weighted_estimating_function_stack.py --
   but self-padding every argument position is kept as defense-in-depth
   for any position/pathway that doesn't go through a where-gated
   substitution, and to keep forward values at invalid cells meaningful
   rather than function-specific garbage.)

2. The Radon-Nikodym weight-window product must never be computed as a
   division of two prefix products (a legitimately-zero weight earlier in
   time -- e.g. a clipped action probability -- would produce a spurious
   0/0 for an unrelated window). Instead, invalid/out-of-window positions
   are reset to the multiplicative identity 1.0 before a single cumulative
   product runs.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .calculate_derivatives import group_user_args_by_shape
from .helper_functions import get_min_time_by_policy_num, get_radon_nikodym_weight
from .vmap_helpers import (
    build_batched_arg_lists_by_subject,
    stack_batched_arg_lists_into_tensors,
)


def get_global_time_axis(
    action_prob_func_args_by_subject_id_by_decision_time: dict[int, dict],
) -> np.ndarray:
    """
    Sorted array of every decision time in the study. NOTE: despite this
    dict's parameter name (matching post_deployment_analysis.py's own type
    hint), its OUTER key is decision_time, not subject_id -- confirmed by
    tracing into arg_threading_helpers.thread_action_prob_func_args, which
    receives this object unmodified and iterates over decision_time as the
    outer key. The outer keys span the full global calendar for every
    subject regardless of individual activity (every subject_id maps to
    real args or the () sentinel at every key).
    """
    return np.array(sorted(action_prob_func_args_by_subject_id_by_decision_time.keys()))


@dataclasses.dataclass(frozen=True)
class ActionProbLayerPrecompute:
    """
    Padded+masked replacement for the ragged action-prob-args /
    action-by-decision-time / policy-num-by-decision-time trio. Every array
    has leading axis N (subjects, in `subject_ids` order -- the same order
    analyze_dataset's `.unique()` construction produces, which
    per_subject_estimating_function_stacks and other positionally-keyed
    outputs already rely on) and a second axis T (the global calendar-time
    grid). Built ONCE per call in plain numpy -- never touches
    jax.grad/jax.vmap tracing, since it depends only on WHICH cells are
    active and WHICH policy/action applied, never on the differentiated
    betas/theta VALUES.
    """

    subject_ids: np.ndarray  # (N,)
    time_values: np.ndarray  # (T,)
    time_to_col: dict[int, int]
    active_mask: np.ndarray  # (N, T) bool
    fill_col_index: np.ndarray  # (N, T) int; nearest active column, per subject
    raw_arg_tensors: tuple[np.ndarray, ...]  # each (N, T, *shape_k), self-padded
    beta_row_index: np.ndarray  # (N, T) int; -1 = no substitution (initial policy / inactive)
    actions_grid: np.ndarray  # (N, T) int
    subject_start_idx: np.ndarray  # (N,)
    subject_end_idx: np.ndarray  # (N,)
    lo_idx: np.ndarray  # (N,); T sentinel == "first_time_after_first_update is None"
    min_time_by_policy_num: list[dict]  # per subject, passed through from get_min_time_by_policy_num
    subject_id_to_pos: dict[Any, int]
    T: int


def assert_no_intra_window_gaps(precompute: ActionProbLayerPrecompute) -> None:
    """
    Fails loudly if a subject was active, then inactive, then active again,
    strictly within their own [subject_start_time, subject_end_time]. A
    masked/padded design has no natural IndexError to fall back on (every
    cell is a valid array position by construction), so this exists purely
    to preserve "fail loudly on malformed input" rather than silently
    computing a wrong number.
    """
    N, _T = precompute.active_mask.shape
    violations = []
    for n in range(N):
        lo, hi = int(precompute.subject_start_idx[n]), int(precompute.subject_end_idx[n])
        window = precompute.active_mask[n, lo : hi + 1]
        if not window.all():
            gap_cols = np.nonzero(~window)[0] + lo
            violations.append(
                (
                    precompute.subject_ids[n].item(),
                    [precompute.time_values[c].item() for c in gap_cols],
                )
            )
    if violations:
        raise ValueError(
            "Found subject(s) with an intra-window gap (inactive at a decision "
            "time strictly between their own first and last active time), which "
            "violates the 'once active, stays active with no re-entry' invariant "
            "the Radon-Nikodym weight-window logic assumes: "
            + "; ".join(f"subject {sid} inactive at {times}" for sid, times in violations)
        )


def _stack_grid_to_tensor(value_grid: list[list[Any]], N: int, T: int) -> np.ndarray:
    flat = [value_grid[n][t] for n in range(N) for t in range(T)]
    (flat_tensor,), _ = stack_batched_arg_lists_into_tensors([flat])
    flat_tensor = np.asarray(flat_tensor)
    return flat_tensor.reshape((N, T) + flat_tensor.shape[1:])


def build_action_prob_layer_precompute(
    subject_ids: np.ndarray,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    action_by_decision_time_by_subject_id: dict[collections.abc.Hashable, dict[int, int]],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    beta_index_by_policy_num: dict[int | float, int],
    initial_policy_num: int | float,
    action_prob_func_args_beta_index: int,
) -> ActionProbLayerPrecompute:
    """
    One-time, plain-numpy precompute (structural/validity information only
    -- never the differentiated betas). Row n of every returned array
    corresponds to subject_ids[n], and this order is never re-sorted.
    """
    subject_ids = np.asarray(subject_ids)
    N = len(subject_ids)
    subject_id_to_pos = {sid: n for n, sid in enumerate(subject_ids.tolist())}
    time_values = get_global_time_axis(action_prob_func_args_by_subject_id_by_decision_time)
    T = len(time_values)
    time_to_col = {int(t): i for i, t in enumerate(time_values)}

    active_mask = np.zeros((N, T), dtype=bool)
    beta_row_index = np.full((N, T), -1, dtype=np.int64)
    actions_grid = np.zeros((N, T), dtype=np.int64)

    # Pass 1: active_mask / actions_grid / beta_row_index. O(N*T) plain-Python
    # work done ONCE -- never O(N*T) individually-dispatched JAX ops, which is
    # the actual current cost this design eliminates.
    for n, subject_id in enumerate(subject_ids.tolist()):
        subj_actions = action_by_decision_time_by_subject_id[subject_id]
        subj_policy_by_t = policy_num_by_decision_time_by_subject_id[subject_id]
        for decision_time, args_by_subject in action_prob_func_args_by_subject_id_by_decision_time.items():
            args = args_by_subject.get(subject_id, ())
            if not args:
                continue
            t_col = time_to_col[int(decision_time)]
            active_mask[n, t_col] = True
            actions_grid[n, t_col] = subj_actions[decision_time]
            policy_num = subj_policy_by_t[decision_time]
            beta_row_index[n, t_col] = (
                -1
                if policy_num == initial_policy_num
                # NOT .get(...): a truthy-args, non-initial policy is expected
                # to always be in beta_index_by_policy_num (fallback policies
                # have empty args). A KeyError here should surface loudly.
                else beta_index_by_policy_num[policy_num]
            )

    # fill_col_index: nearest active column per (subject, time) -- forward-fill
    # then backward-fill, never crossing subjects.
    fill_col_index = np.where(active_mask, np.tile(np.arange(T), (N, 1)), -1)
    for t in range(1, T):
        needs_fill = fill_col_index[:, t] == -1
        fill_col_index[needs_fill, t] = fill_col_index[needs_fill, t - 1]
    for t in range(T - 2, -1, -1):
        needs_fill = fill_col_index[:, t] == -1
        fill_col_index[needs_fill, t] = fill_col_index[needs_fill, t + 1]
    if (fill_col_index == -1).any():
        bad = subject_ids[np.any(fill_col_index == -1, axis=1)]
        raise ValueError(
            f"Subject(s) {bad.tolist()} have no active decision time anywhere; "
            "cannot self-pad their action-prob-args row."
        )

    # Self-pad actions_grid the same way as every other argument tensor below
    # (never a fabricated constant like 0 at invalid cells) -- its value at an
    # invalid cell is masked out downstream, but leaving it a bare zero-fill
    # would be a silent exception to this module's own padding invariant.
    actions_grid = np.take_along_axis(actions_grid, fill_col_index, axis=1)

    num_arg_positions = next(
        len(args)
        for args_by_subject in action_prob_func_args_by_subject_id_by_decision_time.values()
        for args in args_by_subject.values()
        if args
    )
    dummy_grids: list[list[list[Any]]] = [
        [[None] * T for _ in range(N)] for _ in range(num_arg_positions)
    ]
    for n, subject_id in enumerate(subject_ids.tolist()):
        for t_col in range(T):
            decision_time = int(time_values[t_col])
            args = action_prob_func_args_by_subject_id_by_decision_time[decision_time].get(
                subject_id, ()
            )
            if not args:
                # Self-pad directly here: fill_col_index[n, t_col] is this
                # subject's own nearest active column, so grabbing that
                # column's real args IS the self-padding (a separate
                # take_along_axis pass afterward would just recompute the
                # same values via the same index, since an active cell's own
                # fill_col_index entry always points at itself).
                fallback_col = int(fill_col_index[n, t_col])
                fallback_decision_time = int(time_values[fallback_col])
                args = action_prob_func_args_by_subject_id_by_decision_time[
                    fallback_decision_time
                ][subject_id]
            for k in range(num_arg_positions):
                dummy_grids[k][n][t_col] = args[k]
    raw_arg_tensors = tuple(
        _stack_grid_to_tensor(dummy_grids[k], N, T) for k in range(num_arg_positions)
    )

    subject_start_idx = np.zeros(N, dtype=np.int64)
    subject_end_idx = np.zeros(N, dtype=np.int64)
    lo_idx = np.full(N, T, dtype=np.int64)  # T == "no update ever applied" sentinel
    min_time_by_policy_num_list: list[dict] = []
    for n, subject_id in enumerate(subject_ids.tolist()):
        active_times = sorted(action_by_decision_time_by_subject_id[subject_id])
        if not active_times:
            raise ValueError(f"Subject {subject_id} has no active decision times.")
        subject_start_time, subject_end_time = active_times[0], active_times[-1]
        subject_start_idx[n] = time_to_col[subject_start_time]
        subject_end_idx[n] = time_to_col[subject_end_time]

        min_time_by_policy_num_n, first_time_after_first_update_n = get_min_time_by_policy_num(
            policy_num_by_decision_time_by_subject_id[subject_id],
            beta_index_by_policy_num,
        )
        min_time_by_policy_num_list.append(min_time_by_policy_num_n)
        if first_time_after_first_update_n is not None:
            lo_time_n = max(first_time_after_first_update_n, subject_start_time)
            lo_idx[n] = time_to_col[lo_time_n]

    precompute = ActionProbLayerPrecompute(
        subject_ids=subject_ids,
        time_values=time_values,
        time_to_col=time_to_col,
        active_mask=active_mask,
        fill_col_index=fill_col_index,
        raw_arg_tensors=raw_arg_tensors,
        beta_row_index=beta_row_index,
        actions_grid=actions_grid,
        subject_start_idx=subject_start_idx,
        subject_end_idx=subject_end_idx,
        lo_idx=lo_idx,
        min_time_by_policy_num=min_time_by_policy_num_list,
        subject_id_to_pos=subject_id_to_pos,
        T=T,
    )
    assert_no_intra_window_gaps(precompute)  # fails fast, before any expensive work below
    return precompute


@dataclasses.dataclass(frozen=True)
class UpdateArgBucket:
    """
    One shape-homogeneous group of subjects at a single algorithm update (or,
    reused as-is, at the single inference "update"). subject_positions
    indexes into 0..N-1 (global row order). No padding/filler is used within
    a bucket -- every subject here has real, identically-shaped args, so
    vmap'ing over the bucket reproduces the per-subject value exactly.
    """

    subject_positions: np.ndarray  # (bucket_size,)
    subject_ids_in_order: list[Any]
    raw_arg_lists: list[list]  # length num_args; each a bucket_size-long python list
    action_prob_times_by_subject: dict[Any, np.ndarray] | None


@dataclasses.dataclass(frozen=True)
class UpdateLayerPrecompute:
    policy_nums_by_update_index: list[int | float]  # length U
    buckets_by_update_index: list[list[UpdateArgBucket]]
    valid_update: np.ndarray  # (N, U) bool
    hi_idx: np.ndarray  # (N, U) int


def build_update_layer_precompute(
    subject_ids: np.ndarray,
    update_func_args_by_by_subject_id_by_policy_num: dict[int | float, dict[Any, tuple]],
    beta_index_by_policy_num: dict[int | float, int],
    alg_update_func_args_action_prob_times_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
) -> UpdateLayerPrecompute:
    """
    One-time, plain-numpy precompute. U = number of non-initial, non-fallback
    policies. policy_nums_by_update_index is built by sorting
    beta_index_by_policy_num by value (0..U-1) rather than trusting
    update_func_args_by_by_subject_id_by_policy_num's own dict order.

    Bucketing only groups the valid (non-()) subjects at each update by exact
    arg-tuple shape (reusing calculate_derivatives.group_user_args_by_shape --
    the same machinery input_checks.py's
    require_threaded_algorithm_estimating_function_args_equivalent already
    uses for this identical shape-heterogeneity problem).
    """
    N = len(subject_ids)
    subject_id_to_pos = action_prob_layer.subject_id_to_pos
    policy_nums_by_update_index = [
        policy_num
        for policy_num, _ in sorted(beta_index_by_policy_num.items(), key=lambda kv: kv[1])
    ]
    U = len(policy_nums_by_update_index)

    valid_update = np.zeros((N, U), dtype=bool)
    hi_idx = np.zeros((N, U), dtype=np.int64)
    buckets_by_update_index: list[list[UpdateArgBucket]] = []

    for u, policy_num in enumerate(policy_nums_by_update_index):
        args_by_subject_id = update_func_args_by_by_subject_id_by_policy_num[policy_num]

        for n, subject_id in enumerate(subject_ids.tolist()):
            args = args_by_subject_id.get(subject_id, ())
            valid_update[n, u] = bool(args)
            min_time = action_prob_layer.min_time_by_policy_num[n].get(policy_num, math.inf)
            subj_end_plus_1 = int(action_prob_layer.subject_end_idx[n]) + 1
            if math.isfinite(min_time):
                idx_candidate = action_prob_layer.time_to_col[int(min_time)]
            else:
                idx_candidate = subj_end_plus_1
            hi_idx[n, u] = min(idx_candidate, subj_end_plus_1)

        nontrivial = {sid: a for sid, a in args_by_subject_id.items() if a}
        buckets: list[UpdateArgBucket] = []
        for shape_group in group_user_args_by_shape(nontrivial):
            sorted_ids = sorted(shape_group.keys(), key=lambda sid: subject_id_to_pos[sid])
            raw_arg_lists = build_batched_arg_lists_by_subject(sorted_ids, shape_group)
            subject_positions = np.array(
                [subject_id_to_pos[sid] for sid in sorted_ids], dtype=np.int64
            )

            action_prob_times_by_subject = None
            if alg_update_func_args_action_prob_times_index >= 0:
                action_prob_times_by_subject = {
                    sid: np.asarray(shape_group[sid][alg_update_func_args_action_prob_times_index])
                    for sid in sorted_ids
                }

            buckets.append(
                UpdateArgBucket(
                    subject_positions=subject_positions,
                    subject_ids_in_order=sorted_ids,
                    raw_arg_lists=raw_arg_lists,
                    action_prob_times_by_subject=action_prob_times_by_subject,
                )
            )
        buckets_by_update_index.append(buckets)

    return UpdateLayerPrecompute(
        policy_nums_by_update_index=policy_nums_by_update_index,
        buckets_by_update_index=buckets_by_update_index,
        valid_update=valid_update,
        hi_idx=hi_idx,
    )


@dataclasses.dataclass(frozen=True)
class InferenceLayerPrecompute:
    buckets: list[UpdateArgBucket]  # reuses the same bucket shape


def build_inference_layer_precompute(
    inference_func_args_by_subject_id: dict[Any, tuple],
    inference_func_args_action_prob_index: int,
    inference_action_prob_decision_times_by_subject_id: dict[Any, Any],
    action_prob_layer: ActionProbLayerPrecompute,
) -> InferenceLayerPrecompute:
    """
    One-time, plain-numpy precompute. Every subject has a real (never-())
    inference-args tuple, so this layer needs shape-bucketing but no
    valid_* mask.
    """
    subject_id_to_pos = action_prob_layer.subject_id_to_pos
    buckets: list[UpdateArgBucket] = []
    for shape_group in group_user_args_by_shape(
        inference_func_args_by_subject_id, empty_allowed=False
    ):
        sorted_ids = sorted(shape_group.keys(), key=lambda sid: subject_id_to_pos[sid])
        raw_arg_lists = build_batched_arg_lists_by_subject(sorted_ids, shape_group)
        subject_positions = np.array(
            [subject_id_to_pos[sid] for sid in sorted_ids], dtype=np.int64
        )
        action_prob_times_by_subject = None
        if inference_func_args_action_prob_index >= 0:
            action_prob_times_by_subject = {
                sid: np.asarray(inference_action_prob_decision_times_by_subject_id[sid])
                for sid in sorted_ids
            }
        buckets.append(
            UpdateArgBucket(
                subject_positions=subject_positions,
                subject_ids_in_order=sorted_ids,
                raw_arg_lists=raw_arg_lists,
                action_prob_times_by_subject=action_prob_times_by_subject,
            )
        )
    return InferenceLayerPrecompute(buckets=buckets)


def build_threaded_action_prob_beta_tensor(
    precompute: ActionProbLayerPrecompute,
    betas: jnp.ndarray,  # (num_updates, beta_dim) -- the current, traced betas
    action_prob_func_args_beta_index: int,
) -> jnp.ndarray:
    """
    Per-call (depends on the current, traced betas). thread_action_prob_func_args
    only ever substitutes ONE position, so every other raw_arg_tensors[k] is
    reused unmodified every call, with no rebuild at all.
    """
    clamped = jnp.clip(precompute.beta_row_index, 0, max(betas.shape[0] - 1, 0))
    gathered = betas[clamped]  # (N, T, beta_dim), fancy-indexed
    needs_substitution = precompute.beta_row_index >= 0  # (N, T)
    raw_beta_tensor = precompute.raw_arg_tensors[action_prob_func_args_beta_index]  # self-padded, real
    return jnp.where(needs_substitution[..., None], gathered, raw_beta_tensor)


def compute_action_prob_layer_outputs(
    action_prob_func: collections.abc.Callable,
    action_prob_func_args_beta_index: int,
    precompute: ActionProbLayerPrecompute,
    betas: jnp.ndarray,
    need_pi_beta_grid: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """
    Per-call. Returns (raw_weight_grid, pi_beta_grid_or_None), each (N, T).
    raw_weight_grid[n, t] = get_radon_nikodym_weight at every cell, valid or
    not (every cell is guaranteed a real, self-padded domain point). This
    replaces O(N) individually-dispatched calls with exactly one jax.vmap
    call spanning every subject and every global time at once.
    """
    N, T = precompute.active_mask.shape
    NT = N * T

    threaded_beta_tensor = build_threaded_action_prob_beta_tensor(
        precompute, betas, action_prob_func_args_beta_index
    )
    threaded_arg_tensors = tuple(
        threaded_beta_tensor if k == action_prob_func_args_beta_index else precompute.raw_arg_tensors[k]
        for k in range(len(precompute.raw_arg_tensors))
    )
    threaded_args_flat = tuple(t.reshape((NT,) + t.shape[2:]) for t in threaded_arg_tensors)

    beta_target_tensor = precompute.raw_arg_tensors[action_prob_func_args_beta_index]
    beta_target_flat = beta_target_tensor.reshape((NT,) + beta_target_tensor.shape[2:])
    actions_flat = precompute.actions_grid.reshape(NT)

    raw_weights_flat = jax.vmap(
        fun=get_radon_nikodym_weight,
        in_axes=[0, None, None, 0] + [0] * len(threaded_args_flat),
        out_axes=0,
    )(
        beta_target_flat,
        action_prob_func,
        action_prob_func_args_beta_index,
        actions_flat,
        *threaded_args_flat,
    )
    raw_weight_grid = raw_weights_flat.reshape(N, T)

    pi_beta_grid = None
    if need_pi_beta_grid:
        pi_beta_flat = jax.vmap(fun=action_prob_func, in_axes=[0] * len(threaded_args_flat))(
            *threaded_args_flat
        )
        pi_beta_grid = jnp.reshape(pi_beta_flat, (N, T))

    return raw_weight_grid, pi_beta_grid


def compute_windowed_weight_products(
    raw_weights: jnp.ndarray,  # (N, T)
    active_mask: np.ndarray,  # (N, T) bool
    lo_idx: np.ndarray,  # (N,) int, in [0, T]; T == "no update yet" sentinel
    hi_idx: np.ndarray,  # (N, U) int, in [0, T]
    subject_end_idx: np.ndarray,  # (N,) int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Returns (rl_weight_products (N, U), inference_weight_products (N,)),
    reproducing the original per-subject
    `jnp.prod(all_weights[lo:hi]) if first_time_after_first_update is not
    None else 1` semantics exactly, via one cumulative product per subject
    plus O(1) gathers -- never a division of prefix products.

    Why not division: `full_cumprod[hi-1] / full_cumprod[lo-1]` is unsound
    whenever some genuinely-active weight strictly before lo is exactly 0
    (reachable here: action_prob_funcs in this repo are clip-based, i.e.
    built to be able to hit exact clip boundaries) -- every full_cumprod
    entry from that point on becomes 0, turning an unrelated window's
    extraction into a spurious 0/0 = NaN. The construction below resets
    everything strictly before lo to the multiplicative identity before any
    cumprod runs, so a zero before the window can never reach it.

    Key fact (re-derived from get_min_time_by_policy_num): lo is the same
    for every window a subject participates in (every RL update AND
    inference) -- it depends only on first_time_after_first_update and
    subject_start_time, never on policy_num -- so resetting once per subject
    at lo (not once per window) is sufficient; no (N, U, T) tensor is
    needed.
    """
    N, T = raw_weights.shape
    active_mask_j = jnp.asarray(active_mask)
    lo_idx_j = jnp.asarray(lo_idx)

    masked_weights = jnp.where(active_mask_j, raw_weights, 1.0)
    time_cols = jnp.arange(T)[None, :]
    is_on_or_after_lo = time_cols >= lo_idx_j[:, None]
    reset_weights = jnp.where(is_on_or_after_lo, masked_weights, 1.0)

    cum = jnp.cumprod(reset_weights, axis=1)
    cum_ext = jnp.concatenate([jnp.ones((N, 1), dtype=cum.dtype), cum], axis=1)  # (N, T+1)

    rl_weight_products = jnp.take_along_axis(cum_ext, jnp.asarray(hi_idx), axis=1)
    inference_hi_idx = (jnp.asarray(subject_end_idx) + 1)[:, None]
    inference_weight_products = jnp.take_along_axis(cum_ext, inference_hi_idx, axis=1)[:, 0]

    return rl_weight_products, inference_weight_products


def _gather_reconstructed_action_prob(
    pi_beta_grid: jnp.ndarray,
    time_to_col: dict[int, int],
    subject_positions: np.ndarray,
    times_by_subject: dict[Any, np.ndarray],
    subject_ids_in_order: list[Any],
    target_shape: tuple[int, ...],
) -> jnp.ndarray:
    """
    Single vectorized gather -- builds a (bucket_size, num_times) index
    matrix in plain numpy, then one jnp.take_along_axis pulls every
    subject's reconstructed action probabilities out of the
    already-computed (N, T) pi_beta_grid at once, instead of a second,
    redundant, per-subject-dispatched action_prob_func evaluation.
    """
    col_idx_matrix = np.stack(
        [
            np.array([time_to_col[int(t)] for t in times_by_subject[sid].flatten().tolist()])
            for sid in subject_ids_in_order
        ]
    )
    subject_rows = pi_beta_grid[subject_positions]  # (bucket_size, T)
    gathered = jnp.take_along_axis(subject_rows, jnp.asarray(col_idx_matrix), axis=1)
    return gathered.reshape((len(subject_ids_in_order),) + target_shape)


def _stackable_positions(raw_arg_lists: list[list], positions: list[int]) -> list[int]:
    """
    Filters out argument positions where every subject in the bucket holds a
    literal None -- a placeholder for an argument slot the estimating
    function itself never uses (e.g. an unused previous-betas/action-prob
    slot when the corresponding *_index is -1; several shipped fixtures and
    estimating functions rely on this). Stacking such a position would turn
    it into stack_batched_arg_lists_into_tensors' "list of scalars" branch,
    i.e. jnp.array([None, ...]) -- silently NaN today, and JAX has announced
    this becomes a hard error in a future release. Leaving it unstacked
    (its call_args/in_axes entries stay at their None default) passes the
    literal None straight through to jax.vmap, exactly matching what a
    plain per-subject call already does with it.
    """
    return [pos for pos in positions if not all(v is None for v in raw_arg_lists[pos])]


def compute_batched_algorithm_component(
    betas: jnp.ndarray,
    beta_dim: int,
    algorithm_estimating_func: collections.abc.Callable,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    update_layer: UpdateLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
    rl_weight_products: jnp.ndarray,  # (N, U)
) -> jnp.ndarray:
    """
    Per-call. Returns (N, U * beta_dim): row n is exactly
    concat([weight_u * algorithm_estimating_func(*update_args_u) for u in
    range(U)]), with the original "no args at this update -> zeros(beta_dim)"
    gate applied via valid_update. Replaces an O(N*U) Python-dispatched loop
    with O(U * shape_buckets_per_update) jax.vmap calls, each scattered back
    with a single .at[idx].set(...).
    """
    N = action_prob_layer.subject_ids.shape[0]
    U = len(update_layer.policy_nums_by_update_index)
    per_update_components = jnp.zeros((N, U, beta_dim), dtype=betas.dtype)

    for u, policy_num in enumerate(update_layer.policy_nums_by_update_index):
        beta_u = betas[u]
        for bucket in update_layer.buckets_by_update_index[u]:
            bucket_size = len(bucket.subject_ids_in_order)
            if bucket_size == 0:
                continue
            raw_arg_lists = list(bucket.raw_arg_lists)  # copy: about to override some positions

            override_position_values: dict[int, tuple[jnp.ndarray, int | None]] = {
                alg_update_func_args_beta_index: (beta_u, None)
            }

            if alg_update_func_args_previous_betas_index >= 0:
                prev_raw_list = raw_arg_lists[alg_update_func_args_previous_betas_index]
                # len(), not .shape[0]: the original thread_update_func_args used
                # len(...) on this argument (arg_threading_helpers.py), which
                # accepts a plain Python list/tuple as well as an ndarray --
                # calculate_derivatives.get_shape's own len()-fallback anticipates
                # exactly this for shape-bucketing, so this stays consistent.
                num_previous = len(prev_raw_list[0])
                if num_previous > betas.shape[0]:
                    raise ValueError(
                        f"A subject's previous_post_update_betas has length "
                        f"{num_previous} at policy_num={policy_num!r}, but only "
                        f"{betas.shape[0]} update(s) worth of betas are available -- "
                        "betas[:num_previous] would otherwise silently clamp to a "
                        "shorter-than-requested slice instead of raising."
                    )

                # The broadcast below is only correct if every subject in this
                # bucket was actually supplied the same previous-betas content
                # by the caller (true today for every alg_update_func and
                # data-collection function this repo ships, but not a
                # contract lifejacket enforces on
                # alg_update_func_args_previous_betas_index in general).
                # Check it explicitly, once per bucket, rather than assuming
                # it: this is O(bucket_size) numpy array comparisons,
                # negligible next to the vmap call itself, and turns a
                # possible silent wrong-number failure mode into a loud one.
                # np.allclose (not array_equal): only the CONTENT identity
                # across subjects matters here, and different floating-point
                # paths to the same logical value shouldn't spuriously trip
                # this check.
                if bucket_size > 1:
                    first_val = np.asarray(prev_raw_list[0])
                    for other in prev_raw_list[1:]:
                        if not np.allclose(np.asarray(other), first_val):
                            raise ValueError(
                                "alg_update_func_args_previous_betas_index does not have "
                                "identical raw values across every subject sharing this "
                                f"shape bucket at policy_num={policy_num!r}. The batched "
                                "implementation broadcasts ONE previous-betas value per "
                                "update across a whole bucket, which is only correct when "
                                "every subject at a given update was supplied the same "
                                "previous_post_update_betas content. If this fires, either "
                                "the calling code is supplying genuinely subject-specific "
                                "previous betas (fix the caller), or this bucket handling "
                                "needs a real per-subject gather instead of a broadcast."
                            )

                override_position_values[alg_update_func_args_previous_betas_index] = (
                    betas[:num_previous],
                    None,
                )

            if alg_update_func_args_action_prob_index >= 0:
                target_shape = raw_arg_lists[alg_update_func_args_action_prob_index][0].shape
                reconstructed = _gather_reconstructed_action_prob(
                    pi_beta_grid,
                    action_prob_layer.time_to_col,
                    bucket.subject_positions,
                    bucket.action_prob_times_by_subject,
                    bucket.subject_ids_in_order,
                    target_shape,
                )
                override_position_values[alg_update_func_args_action_prob_index] = (
                    reconstructed,
                    0,
                )

            num_args = len(raw_arg_lists)
            remaining_positions = [k for k in range(num_args) if k not in override_position_values]
            stack_positions = _stackable_positions(raw_arg_lists, remaining_positions)
            remaining_tensors, _ = stack_batched_arg_lists_into_tensors(
                [raw_arg_lists[k] for k in stack_positions]
            )
            call_args: list[Any] = [None] * num_args
            in_axes: list[Any] = [None] * num_args
            for pos, tensor in zip(stack_positions, remaining_tensors):
                call_args[pos] = tensor
                in_axes[pos] = 0
            for pos, (value, axis) in override_position_values.items():
                call_args[pos] = value
                in_axes[pos] = axis

            bucket_output = jax.vmap(algorithm_estimating_func, in_axes=in_axes)(*call_args)
            if bucket_output.shape != (bucket_size, beta_dim):
                # per_update_components is pre-allocated at (N, U, beta_dim), so
                # .at[].set() below would otherwise silently BROADCAST a
                # wrong-shaped-but-broadcastable bucket_output (e.g. (bucket_size, 1))
                # across every beta_dim slot instead of raising -- check explicitly,
                # matching the loud failure the original per-subject
                # `algorithm_component.size % beta_dim != 0` check gave for this
                # same class of bug (a malformed algorithm_estimating_func).
                raise ValueError(
                    f"algorithm_estimating_func returned shape {bucket_output.shape} "
                    f"for {bucket_size} subject(s) at policy_num={policy_num!r}; "
                    f"expected ({bucket_size}, {beta_dim})."
                )
            per_update_components = per_update_components.at[bucket.subject_positions, u].set(
                bucket_output
            )

    weighted = (
        per_update_components
        * update_layer.valid_update[:, :, None]
        * rl_weight_products[:, :, None]
    )
    return weighted.reshape(N, U * beta_dim)


def compute_batched_inference_outputs(
    theta: jnp.ndarray,
    theta_dim: int,
    inference_estimating_func: collections.abc.Callable,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    inference_layer: InferenceLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
    inference_weight_products: jnp.ndarray,  # (N,)
    need_hessians: bool,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    """
    Per-call. Returns (weighted_inference_component (N, theta_dim),
    inference_hessians (N, theta_dim, theta_dim) or None). inference_hessians
    is the unweighted per-subject Jacobian of inference_estimating_func wrt
    theta (classical bread contribution) -- matches the original
    jax.jacrev(inference_estimating_func, argnums=theta_index)(*threaded_args),
    with no Radon-Nikodym weight applied, exactly as before.
    """
    N = action_prob_layer.subject_ids.shape[0]
    component = jnp.zeros((N, theta_dim), dtype=theta.dtype)
    hessians = jnp.zeros((N, theta_dim, theta_dim), dtype=theta.dtype) if need_hessians else None

    for bucket in inference_layer.buckets:
        bucket_size = len(bucket.subject_ids_in_order)
        raw_arg_lists = list(bucket.raw_arg_lists)
        override_position_values: dict[int, tuple[jnp.ndarray, int | None]] = {
            inference_func_args_theta_index: (theta, None)
        }
        if inference_func_args_action_prob_index >= 0:
            target_shape = raw_arg_lists[inference_func_args_action_prob_index][0].shape
            reconstructed = _gather_reconstructed_action_prob(
                pi_beta_grid,
                action_prob_layer.time_to_col,
                bucket.subject_positions,
                bucket.action_prob_times_by_subject,
                bucket.subject_ids_in_order,
                target_shape,
            )
            override_position_values[inference_func_args_action_prob_index] = (reconstructed, 0)

        num_args = len(raw_arg_lists)
        remaining_positions = [k for k in range(num_args) if k not in override_position_values]
        stack_positions = _stackable_positions(raw_arg_lists, remaining_positions)
        remaining_tensors, _ = stack_batched_arg_lists_into_tensors(
            [raw_arg_lists[k] for k in stack_positions]
        )
        call_args: list[Any] = [None] * num_args
        in_axes: list[Any] = [None] * num_args
        for pos, tensor in zip(stack_positions, remaining_tensors):
            call_args[pos] = tensor
            in_axes[pos] = 0
        for pos, (value, axis) in override_position_values.items():
            call_args[pos] = value
            in_axes[pos] = axis

        bucket_component = jax.vmap(inference_estimating_func, in_axes=in_axes)(*call_args)
        if bucket_component.shape != (bucket_size, theta_dim):
            # See the analogous check in compute_batched_algorithm_component --
            # component is pre-allocated at (N, theta_dim), so .at[].set() would
            # otherwise silently broadcast a wrong-shaped output instead of
            # raising on a malformed inference_estimating_func.
            raise ValueError(
                f"inference_estimating_func returned shape {bucket_component.shape} "
                f"for {bucket_size} subject(s); expected ({bucket_size}, {theta_dim})."
            )
        component = component.at[bucket.subject_positions].set(bucket_component)

        if need_hessians:
            bucket_hessians = jax.vmap(
                jax.jacrev(inference_estimating_func, argnums=inference_func_args_theta_index),
                in_axes=in_axes,
            )(*call_args)
            if bucket_hessians.shape != (bucket_size, theta_dim, theta_dim):
                raise ValueError(
                    f"jax.jacrev(inference_estimating_func) returned shape "
                    f"{bucket_hessians.shape} for {bucket_size} subject(s); expected "
                    f"({bucket_size}, {theta_dim}, {theta_dim})."
                )
            hessians = hessians.at[bucket.subject_positions].set(bucket_hessians)

    weighted_component = component * inference_weight_products[:, None]
    return weighted_component, hessians
