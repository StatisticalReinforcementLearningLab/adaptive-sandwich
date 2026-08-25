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

Pipeline, in the order get_avg_weighted_estimating_function_stacks_and_aux_values
(post_deployment_analysis.py) calls it, once per differentiated (betas,
theta) value:

1. Structural precompute (plain numpy, rebuilt once per call, NEVER traced):
   build_action_prob_layer_precompute -> ActionProbLayerPrecompute,
   build_update_layer_precompute -> UpdateLayerPrecompute,
   build_inference_layer_precompute -> InferenceLayerPrecompute. These
   encode WHICH cells are active/valid and WHICH policy/action applied --
   never the differentiated values themselves -- so none of this needs to
   re-run inside jax.grad/jax.vmap tracing.
2. Batched forward pass (traced; depends on the current betas/theta):
   compute_action_prob_layer_outputs (ONE jax.vmap spanning every subject x
   every global decision time at once, for the Radon-Nikodym weights and,
   if needed, the reconstructed action probabilities) feeds
   compute_windowed_weight_products (one jnp.cumprod per subject, to slice
   out every update's/inference's weight-window product at once, without
   ever dividing prefix products), whose outputs feed
   compute_batched_algorithm_component / compute_batched_inference_outputs
   (one jax.vmap per (update, shape-bucket), or per inference shape-bucket).
3. Optional data checks (skipped when suppress_all_data_checks=True):
   check_batched_algorithm_estimating_function_args_equivalent /
   check_batched_inference_estimating_function_args_equivalent, which reuse
   the exact same bucket-override builders as step 2
   (_build_algorithm_bucket_overrides / _build_inference_bucket_overrides),
   so what gets checked is guaranteed identical to what the real computation
   used, not a separately-derived approximation of it.

Shapes recur across almost every function below and are defined once here
rather than re-derived in each docstring:
  N = number of subjects.
  T = number of distinct global decision times (every subject shares this
      axis, even though most are only active for a sub-range of it).
  U = number of non-initial, non-fallback policy updates.
  bucket_size = number of subjects sharing one exact argument shape at one
      update (or, for the inference layer, overall) -- see UpdateArgBucket.

Vs. the original per-subject implementation. This module has two spiritual
predecessors, both still present in the codebase (porting either is
deliberately out of scope here -- see the ADR's Step 4 "Scope" note):

  - post_deployment_analysis._reference_single_subject_weighted_estimating_function_stacker:
    a plain-Python, one-subject-at-a-time implementation, kept ONLY as a
    correctness oracle for tests -- no production code path calls it
    anymore. It loops in Python over threaded_update_func_args_by_policy_num,
    and for each update inline-computes one jnp.prod(all_weights[lo:hi])
    weight-window product (via helper_functions.compute_subject_radon_nikodym_weights'
    per-subject raw weights and lo/hi bookkeeping) before calling
    algorithm_estimating_func(*update_args) directly. This module replaces
    that per-subject, per-update Python double-loop with: the raw weights
    for EVERY subject via one compute_action_prob_layer_outputs jax.vmap
    call; EVERY subject's EVERY window product at once via
    compute_windowed_weight_products' cumulative-product-with-reset (instead
    of each caller re-deriving its own lo/hi and re-slicing); and the
    estimating-function evaluations themselves via
    compute_batched_algorithm_component's one jax.vmap per (update,
    shape-bucket), instead of one dispatch per (subject, update) pair.
  - helper_functions.compute_subject_radon_nikodym_weights: the shared,
    still-per-subject weight/bookkeeping helper that
    deployment_conditioning_monitor.py and
    get_datum_for_blowup_supervised_learning.py still call directly (a
    deliberate scope decision, not an oversight -- porting them is the
    natural next step if their performance ever becomes a concern).
    compute_action_prob_layer_outputs + compute_windowed_weight_products
    together are this module's batched, all-subjects-at-once counterpart,
    used only by post_deployment_analysis.py's hot path.

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
    beta_row_index: (
        np.ndarray
    )  # (N, T) int; -1 = no substitution (initial policy / inactive)
    actions_grid: np.ndarray  # (N, T) int
    subject_start_idx: np.ndarray  # (N,)
    subject_end_idx: np.ndarray  # (N,)
    lo_idx: np.ndarray  # (N,); T sentinel == "first_time_after_first_update is None"
    min_time_by_policy_num: list[
        dict
    ]  # per subject, passed through from get_min_time_by_policy_num
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
        lo, hi = (
            int(precompute.subject_start_idx[n]),
            int(precompute.subject_end_idx[n]),
        )
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
            + "; ".join(
                f"subject {sid} inactive at {times}" for sid, times in violations
            )
        )


def _stack_grid_to_tensor(value_grid: list[list[Any]], N: int, T: int) -> np.ndarray:
    """
    Pure-numpy stacking -- deliberately never jax.numpy. This always runs
    against static, never-differentiated data (the one-time structural
    precompute), which must stay concrete even when this whole module is
    called from inside a jax.jit trace (e.g. ADS-139 Step 3): under tracing,
    every jnp op inside the trace produces an abstract tracer regardless of
    whether its inputs are literal constants, which would make a later
    np.asarray(...) on the result raise TracerArrayConversionError -- as it
    did here when this used to call the jax.numpy-based
    stack_batched_arg_lists_into_tensors (vmap_helpers.py), which exists for
    the opposite case: building tensors that ARE meant to flow into a traced
    jax.vmap call.

    Unlike alg_update_func_args/inference_func_args (which may legitimately
    hold None at an unused, override-only argument position -- see
    _stackable_positions), every action_prob_func_args position here is
    REQUIRED: it always flows into a real call to action_prob_func, so None
    is never a valid value at this layer. Check for it explicitly (a clear
    ValueError naming the offending cell) rather than silently building an
    object-dtype array that would otherwise surface, confusingly, as a dtype
    error much later inside jax.vmap.
    """
    flat = []
    for n in range(N):
        for t in range(T):
            value = value_grid[n][t]
            if value is None:
                raise ValueError(
                    f"action_prob_func_args has a None value at subject index {n}, "
                    f"time index {t} -- every action_prob_func_args position must be a "
                    "real value, since it always flows into a call to action_prob_func "
                    "(unlike alg_update_func_args/inference_func_args, which may use "
                    "None for an unused, override-only argument position)."
                )
            flat.append(np.asarray(value))
    # Checked across every cell, not just flat[0]: recorded arg shapes could
    # in principle vary by decision time even for the same argument position.
    if any(v.ndim > 2 for v in flat):
        raise TypeError("Arrays with dimension greater than 2 are not supported.")
    stacked = np.stack(flat, axis=0)
    return stacked.reshape((N, T) + stacked.shape[1:])


def build_action_prob_layer_precompute(
    subject_ids: np.ndarray,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
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
    time_values = get_global_time_axis(
        action_prob_func_args_by_subject_id_by_decision_time
    )
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
        for (
            decision_time,
            args_by_subject,
        ) in action_prob_func_args_by_subject_id_by_decision_time.items():
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
            args = action_prob_func_args_by_subject_id_by_decision_time[
                decision_time
            ].get(subject_id, ())
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

        min_time_by_policy_num_n, first_time_after_first_update_n = (
            get_min_time_by_policy_num(
                policy_num_by_decision_time_by_subject_id[subject_id],
                beta_index_by_policy_num,
            )
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
    assert_no_intra_window_gaps(
        precompute
    )  # fails fast, before any expensive work below
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
    update_func_args_by_by_subject_id_by_policy_num: dict[
        int | float, dict[Any, tuple]
    ],
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
        for policy_num, _ in sorted(
            beta_index_by_policy_num.items(), key=lambda kv: kv[1]
        )
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
            min_time = action_prob_layer.min_time_by_policy_num[n].get(
                policy_num, math.inf
            )
            subj_end_plus_1 = int(action_prob_layer.subject_end_idx[n]) + 1
            if math.isfinite(min_time):
                idx_candidate = action_prob_layer.time_to_col[int(min_time)]
            else:
                idx_candidate = subj_end_plus_1
            hi_idx[n, u] = min(idx_candidate, subj_end_plus_1)

        nontrivial = {sid: a for sid, a in args_by_subject_id.items() if a}
        buckets: list[UpdateArgBucket] = []
        for shape_group in group_user_args_by_shape(nontrivial):
            sorted_ids = sorted(
                shape_group.keys(), key=lambda sid: subject_id_to_pos[sid]
            )
            raw_arg_lists = build_batched_arg_lists_by_subject(sorted_ids, shape_group)
            subject_positions = np.array(
                [subject_id_to_pos[sid] for sid in sorted_ids], dtype=np.int64
            )

            action_prob_times_by_subject = None
            if alg_update_func_args_action_prob_times_index >= 0:
                action_prob_times_by_subject = {
                    sid: np.asarray(
                        shape_group[sid][alg_update_func_args_action_prob_times_index]
                    )
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
    raw_beta_tensor = precompute.raw_arg_tensors[
        action_prob_func_args_beta_index
    ]  # self-padded, real
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
        threaded_beta_tensor
        if k == action_prob_func_args_beta_index
        else precompute.raw_arg_tensors[k]
        for k in range(len(precompute.raw_arg_tensors))
    )
    threaded_args_flat = tuple(
        t.reshape((NT,) + t.shape[2:]) for t in threaded_arg_tensors
    )

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
        pi_beta_flat = jax.vmap(
            fun=action_prob_func, in_axes=[0] * len(threaded_args_flat)
        )(*threaded_args_flat)
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
    cum_ext = jnp.concatenate(
        [jnp.ones((N, 1), dtype=cum.dtype), cum], axis=1
    )  # (N, T+1)

    rl_weight_products = jnp.take_along_axis(cum_ext, jnp.asarray(hi_idx), axis=1)
    inference_hi_idx = (jnp.asarray(subject_end_idx) + 1)[:, None]
    inference_weight_products = jnp.take_along_axis(cum_ext, inference_hi_idx, axis=1)[
        :, 0
    ]

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
            np.array(
                [time_to_col[int(t)] for t in times_by_subject[sid].flatten().tolist()]
            )
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


def _assemble_call_args_and_in_axes(
    raw_arg_lists: list[list],
    override_position_values: dict[int, tuple[Any, int | None]],
) -> tuple[list[Any], list[Any]]:
    """
    Builds the (call_args, in_axes) pair for one jax.vmap(estimating_func, ...)
    call over one shape bucket: every position not in
    override_position_values gets its raw per-subject values stacked into a
    batched tensor (in_axes=0, skipping all-None positions per
    _stackable_positions); every position in override_position_values uses
    the given (value, axis) directly (e.g. a shared beta with axis=None, or a
    per-subject reconstructed-action-prob tensor with axis=0).
    """
    num_args = len(raw_arg_lists)
    remaining_positions = [
        k for k in range(num_args) if k not in override_position_values
    ]
    stack_positions = _stackable_positions(raw_arg_lists, remaining_positions)
    remaining_tensors, _ = stack_batched_arg_lists_into_tensors(
        [raw_arg_lists[k] for k in stack_positions]
    )
    call_args: list[Any] = [None] * num_args
    in_axes: list[Any] = [None] * num_args
    for pos, tensor in zip(stack_positions, remaining_tensors, strict=False):
        call_args[pos] = tensor
        in_axes[pos] = 0
    for pos, (value, axis) in override_position_values.items():
        call_args[pos] = value
        in_axes[pos] = axis
    return call_args, in_axes


def _add_action_prob_override_if_used(
    override_position_values: dict[int, tuple[Any, int | None]],
    action_prob_index: int,
    bucket: UpdateArgBucket,
    action_prob_layer: ActionProbLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
) -> None:
    """
    Shared by _build_algorithm_bucket_overrides and
    _build_inference_bucket_overrides: if this estimating function takes a
    reconstructed-action-probability argument, gather it from the
    already-computed pi_beta_grid and add it to override_position_values
    in place. No-op if action_prob_index < 0.
    """
    if action_prob_index < 0:
        return
    target_shape = bucket.raw_arg_lists[action_prob_index][0].shape
    reconstructed = _gather_reconstructed_action_prob(
        pi_beta_grid,
        action_prob_layer.time_to_col,
        bucket.subject_positions,
        bucket.action_prob_times_by_subject,
        bucket.subject_ids_in_order,
        target_shape,
    )
    override_position_values[action_prob_index] = (reconstructed, 0)


def _build_algorithm_bucket_overrides(
    betas: jnp.ndarray,
    beta_u: jnp.ndarray,
    policy_num: int | float,
    bucket: UpdateArgBucket,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
) -> dict[int, tuple[Any, int | None]]:
    """
    The beta/previous-betas/action-prob override construction for one
    (update, shape-bucket) pair -- shared by compute_batched_algorithm_component
    (the main computation) and check_batched_algorithm_estimating_function_args_equivalent
    (the data-check), so the "threaded" arguments used by the check are
    guaranteed identical to what the main computation actually uses.
    """
    bucket_size = len(bucket.subject_ids_in_order)
    raw_arg_lists = bucket.raw_arg_lists
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

    _add_action_prob_override_if_used(
        override_position_values,
        alg_update_func_args_action_prob_index,
        bucket,
        action_prob_layer,
        pi_beta_grid,
    )
    return override_position_values


def _build_inference_bucket_overrides(
    theta: jnp.ndarray,
    bucket: UpdateArgBucket,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
) -> dict[int, tuple[Any, int | None]]:
    """
    The theta/action-prob override construction for one inference shape
    bucket -- shared by compute_batched_inference_outputs (the main
    computation) and check_batched_inference_estimating_function_args_equivalent
    (the data-check).
    """
    override_position_values: dict[int, tuple[jnp.ndarray, int | None]] = {
        inference_func_args_theta_index: (theta, None)
    }
    _add_action_prob_override_if_used(
        override_position_values,
        inference_func_args_action_prob_index,
        bucket,
        action_prob_layer,
        pi_beta_grid,
    )
    return override_position_values


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
) -> tuple[jnp.ndarray, list[jnp.ndarray]]:
    """
    Per-call. Returns (N, U * beta_dim): row n is exactly
    concat([weight_u * algorithm_estimating_func(*update_args_u) for u in
    range(U)]), with the original "no args at this update -> zeros(beta_dim)"
    gate applied via valid_update. Replaces an O(N*U) Python-dispatched loop
    with O(U * shape_buckets_per_update) jax.vmap calls, each scattered back
    with a single .at[idx].set(...).

    Also returns the raw, unweighted per-bucket outputs (one entry per
    non-empty (update, bucket) pair, in traversal order) -- this is exactly
    the "threaded" (reconstructed-action-prob) result
    check_batched_algorithm_estimating_function_args_equivalent would
    otherwise recompute from scratch via a second, identical
    _build_algorithm_bucket_overrides + jax.vmap call. Passing it back lets
    the data check reuse this call's work instead of duplicating it.
    """
    N = action_prob_layer.subject_ids.shape[0]
    U = len(update_layer.policy_nums_by_update_index)
    per_update_components = jnp.zeros((N, U, beta_dim), dtype=betas.dtype)
    bucket_outputs: list[jnp.ndarray] = []

    for u, policy_num in enumerate(update_layer.policy_nums_by_update_index):
        beta_u = betas[u]
        for bucket in update_layer.buckets_by_update_index[u]:
            bucket_size = len(bucket.subject_ids_in_order)
            if bucket_size == 0:
                continue

            override_position_values = _build_algorithm_bucket_overrides(
                betas,
                beta_u,
                policy_num,
                bucket,
                alg_update_func_args_beta_index,
                alg_update_func_args_previous_betas_index,
                alg_update_func_args_action_prob_index,
                action_prob_layer,
                pi_beta_grid,
            )
            call_args, in_axes = _assemble_call_args_and_in_axes(
                bucket.raw_arg_lists, override_position_values
            )

            bucket_output = jax.vmap(algorithm_estimating_func, in_axes=in_axes)(
                *call_args
            )
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
            bucket_outputs.append(bucket_output)
            per_update_components = per_update_components.at[
                bucket.subject_positions, u
            ].set(bucket_output)

    weighted = (
        per_update_components
        * update_layer.valid_update[:, :, None]
        * rl_weight_products[:, :, None]
    )
    return weighted.reshape(N, U * beta_dim), bucket_outputs


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
) -> tuple[jnp.ndarray, jnp.ndarray | None, list[jnp.ndarray]]:
    """
    Per-call. Returns (weighted_inference_component (N, theta_dim),
    inference_hessians (N, theta_dim, theta_dim) or None, bucket_outputs).
    inference_hessians is the unweighted per-subject Jacobian of
    inference_estimating_func wrt theta (classical bread contribution) --
    matches the original
    jax.jacrev(inference_estimating_func, argnums=theta_index)(*threaded_args),
    with no Radon-Nikodym weight applied, exactly as before.

    bucket_outputs is the raw, unweighted per-bucket output (one entry per
    bucket in inference_layer.buckets, in order) -- see
    compute_batched_algorithm_component's docstring for why this is returned
    (reused by check_batched_inference_estimating_function_args_equivalent
    instead of it recomputing the same "threaded" pass).
    """
    N = action_prob_layer.subject_ids.shape[0]
    component = jnp.zeros((N, theta_dim), dtype=theta.dtype)
    hessians = (
        jnp.zeros((N, theta_dim, theta_dim), dtype=theta.dtype)
        if need_hessians
        else None
    )
    bucket_outputs: list[jnp.ndarray] = []

    for bucket in inference_layer.buckets:
        bucket_size = len(bucket.subject_ids_in_order)

        override_position_values = _build_inference_bucket_overrides(
            theta,
            bucket,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            action_prob_layer,
            pi_beta_grid,
        )
        call_args, in_axes = _assemble_call_args_and_in_axes(
            bucket.raw_arg_lists, override_position_values
        )

        bucket_component = jax.vmap(inference_estimating_func, in_axes=in_axes)(
            *call_args
        )
        if bucket_component.shape != (bucket_size, theta_dim):
            # See the analogous check in compute_batched_algorithm_component --
            # component is pre-allocated at (N, theta_dim), so .at[].set() would
            # otherwise silently broadcast a wrong-shaped output instead of
            # raising on a malformed inference_estimating_func.
            raise ValueError(
                f"inference_estimating_func returned shape {bucket_component.shape} "
                f"for {bucket_size} subject(s); expected ({bucket_size}, {theta_dim})."
            )
        bucket_outputs.append(bucket_component)
        component = component.at[bucket.subject_positions].set(bucket_component)

        if need_hessians:
            bucket_hessians = jax.vmap(
                jax.jacrev(
                    inference_estimating_func, argnums=inference_func_args_theta_index
                ),
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
    return weighted_component, hessians, bucket_outputs


def _compute_threaded_bucket_result(
    estimating_func: collections.abc.Callable,
    bucket: UpdateArgBucket,
    threaded_overrides: dict[int, tuple[Any, int | None]],
) -> jnp.ndarray:
    """
    Builds the threaded (reconstructed-action-prob) call for one bucket and
    evaluates it -- exactly the same shape of call
    compute_batched_algorithm_component/compute_batched_inference_outputs
    already makes for this bucket. Only used as a fallback when the caller
    doesn't already have that result on hand (see
    check_batched_algorithm_estimating_function_args_equivalent's
    precomputed_threaded_results parameter).
    """
    threaded_call_args, threaded_in_axes = _assemble_call_args_and_in_axes(
        bucket.raw_arg_lists, threaded_overrides
    )
    return jax.vmap(estimating_func, in_axes=threaded_in_axes)(*threaded_call_args)


def _assert_original_and_threaded_bucket_results_agree(
    estimating_func: collections.abc.Callable,
    bucket: UpdateArgBucket,
    threaded_result: jnp.ndarray,
    atol: float,
    rtol: float,
) -> None:
    """
    Shared comparison step for both check_batched_*_equivalent functions:
    calls estimating_func over one bucket with the original, entirely
    un-substituted arguments, and asserts the result agrees with the given
    threaded_result within tolerance. Factored out specifically so the two
    callers' tolerances share one definition instead of two literal copies
    that can silently drift apart (as happened once already: the inference
    side had been copied from the algorithm side's atol=1e-7, rtol=1e-3
    instead of matching its own original,
    input_checks.require_threaded_inference_estimating_function_args_equivalent's
    looser rtol=1e-2, no-atol tolerance).
    """
    original_call_args, original_in_axes = _assemble_call_args_and_in_axes(
        bucket.raw_arg_lists, {}
    )
    original_result = jax.vmap(estimating_func, in_axes=original_in_axes)(
        *original_call_args
    )
    # Need to stop gradient here: threaded_result traces back to betas/theta,
    # which are being differentiated in the real jax.jacrev call this check
    # runs alongside, and np.asarray can't convert a traced value.
    np.testing.assert_allclose(
        np.asarray(original_result),
        np.asarray(jax.lax.stop_gradient(threaded_result)),
        atol=atol,
        rtol=rtol,
    )


def check_batched_algorithm_estimating_function_args_equivalent(
    algorithm_estimating_func: collections.abc.Callable,
    betas: jnp.ndarray,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    update_layer: UpdateLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
    precomputed_threaded_results: list[jnp.ndarray] | None = None,
) -> None:
    """
    Batched equivalent of
    input_checks.require_threaded_algorithm_estimating_function_args_equivalent:
    for every (update, shape-bucket), checks that substituting the shared
    betas and RECONSTRUCTED action probabilities (exactly what
    compute_batched_algorithm_component uses, via the same
    _build_algorithm_bucket_overrides helper) into algorithm_estimating_func
    produces the same result as the ORIGINAL, un-substituted arguments.

    Reuses the already-computed pi_beta_grid instead of re-deriving
    reconstructed action probabilities via a second, per-subject/per-update
    dispatched pass through arg_threading_helpers.thread_update_func_args --
    that old path was itself exactly the O(subjects) individually-dispatched
    pattern the rest of this module exists to eliminate, just relocated to
    feed this check instead of the main computation. No-op if action
    probabilities are not used in the algorithm estimating function.

    precomputed_threaded_results, if given, must be
    compute_batched_algorithm_component's own bucket_outputs return value
    (one entry per non-empty (update, bucket) pair, in the same traversal
    order used here) -- reusing it instead of recomputing the threaded call
    from scratch avoids doubling this check's cost on top of the main
    computation's identical work. If None, the threaded result is computed
    fresh here (used by callers that only have this check's own inputs, e.g.
    standalone tests).
    """
    if alg_update_func_args_action_prob_index < 0:
        return
    result_idx = 0
    for u, policy_num in enumerate(update_layer.policy_nums_by_update_index):
        beta_u = betas[u]
        for bucket in update_layer.buckets_by_update_index[u]:
            if len(bucket.subject_ids_in_order) == 0:
                continue

            if precomputed_threaded_results is not None:
                threaded_result = precomputed_threaded_results[result_idx]
            else:
                threaded_overrides = _build_algorithm_bucket_overrides(
                    betas,
                    beta_u,
                    policy_num,
                    bucket,
                    alg_update_func_args_beta_index,
                    alg_update_func_args_previous_betas_index,
                    alg_update_func_args_action_prob_index,
                    action_prob_layer,
                    pi_beta_grid,
                )
                threaded_result = _compute_threaded_bucket_result(
                    algorithm_estimating_func, bucket, threaded_overrides
                )
            result_idx += 1
            # Tolerance matches
            # input_checks.require_threaded_algorithm_estimating_function_args_equivalent
            # exactly -- see _assert_original_and_threaded_bucket_results_agree.
            _assert_original_and_threaded_bucket_results_agree(
                algorithm_estimating_func,
                bucket,
                threaded_result,
                atol=1e-7,
                rtol=1e-3,
            )


def check_batched_inference_estimating_function_args_equivalent(
    inference_estimating_func: collections.abc.Callable,
    theta: jnp.ndarray,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_layer: ActionProbLayerPrecompute,
    inference_layer: InferenceLayerPrecompute,
    pi_beta_grid: jnp.ndarray | None,
    precomputed_threaded_results: list[jnp.ndarray] | None = None,
) -> None:
    """
    Batched equivalent of
    input_checks.require_threaded_inference_estimating_function_args_equivalent
    -- see check_batched_algorithm_estimating_function_args_equivalent's
    docstring for the shared rationale, including precomputed_threaded_results
    (here, compute_batched_inference_outputs's own bucket_outputs). No-op if
    action probabilities are not used in the inference estimating function.
    """
    if inference_func_args_action_prob_index < 0:
        return
    result_idx = 0
    for bucket in inference_layer.buckets:
        if len(bucket.subject_ids_in_order) == 0:
            continue

        if precomputed_threaded_results is not None:
            threaded_result = precomputed_threaded_results[result_idx]
        else:
            threaded_overrides = _build_inference_bucket_overrides(
                theta,
                bucket,
                inference_func_args_theta_index,
                inference_func_args_action_prob_index,
                action_prob_layer,
                pi_beta_grid,
            )
            threaded_result = _compute_threaded_bucket_result(
                inference_estimating_func, bucket, threaded_overrides
            )
        result_idx += 1
        # Tolerance matches
        # input_checks.require_threaded_inference_estimating_function_args_equivalent
        # exactly (a looser rtol, no atol, than the algorithm-side check
        # above -- NOT the same value, and previously copy-pasted wrong; see
        # _assert_original_and_threaded_bucket_results_agree).
        _assert_original_and_threaded_bucket_results_agree(
            inference_estimating_func, bucket, threaded_result, atol=0.0, rtol=1e-2
        )
