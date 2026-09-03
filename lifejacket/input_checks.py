import collections
import inspect
import logging
import math
import numbers
import os
import pathlib
from typing import Any

import jax
import numpy as np
import pandas as pd
import plotext as plt
from jax import numpy as jnp

from .calculate_derivatives import (
    get_batched_arg_lists_and_involved_user_ids,
    group_user_args_by_shape,
)
from .constants import FunctionTypes
from .helper_functions import (
    confirm_input_check_result,
    construct_beta_index_by_policy_num_map,
)
from .vmap_helpers import batch_args_by_subject, stack_batched_arg_lists_into_tensors

# When we print out objects for debugging, show the whole thing.
np.set_printoptions(threshold=np.inf)

logger = logging.getLogger(__name__)


def perform_unconditional_zeroth_wave_input_checks(
    output_dir,
    percentile_bootstrap_draws,
    percentile_bootstrap_alpha,
    percentile_bootstrap_seed,
):
    """
    The checks that run FIRST and are never suppressed, not even by suppress_all_data_checks.

    Both earn their exemption by guarding something that would otherwise be discarded or
    silently wrong rather than merely unchecked:

    - The output directory: every result is written to output_dir only at the very end of the
      run and nothing creates it, so a bad path would otherwise throw away the entire analysis
      at the finish line. See require_output_dir_ready for what this does and does not
      protect against.
    - The percentile-bootstrap settings, for the sharper reason: percentile_bootstrap_alpha is
      used directly as a quantile level, so an out-of-range value does not fail -- it silently
      reports a meaningless interval as percentile_bootstrap_ci.

    Neither one looks at the data, which is why suppress_all_data_checks has no bearing on
    them.

    Args:
        output_dir (str | os.PathLike): The directory every result will be written to.
        percentile_bootstrap_draws (int | None): The number of multiplier-bootstrap draws, or
            None when the percentile bootstrap was not requested.
        percentile_bootstrap_alpha (float): The percentile-interval level.
        percentile_bootstrap_seed (int | None): The seed for the multiplicity draws.
    """
    require_output_dir_ready(output_dir)
    require_valid_percentile_bootstrap_settings(
        percentile_bootstrap_draws,
        percentile_bootstrap_alpha,
        percentile_bootstrap_seed,
    )


def perform_conditional_zeroth_wave_dataframe_checks(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
    reward_col_name,
):
    """
    The dataframe-only prerequisites, run before the caller's theta_calculation_func.

    The user-supplied theta_calculation_func consumes analysis_df BEFORE the first wave can run
    (the wave needs the theta it returns), so a malformed frame used to fail inside user code
    with whatever error that code happened to raise. These are exactly the checks that need
    nothing but the frame, and whose actionable messages a broken callback input would
    otherwise preempt -- so they run here, ahead of it, and NOT again in the first wave.

    Conditional: skipped, along with the first wave and the diagnostic suite, under
    suppress_all_data_checks. Everything downstream of them -- including
    require_all_named_columns_not_object_type_in_analysis_df and every check that hashes a
    subject id -- assumes they have already passed, so a caller invoking the first wave
    directly must call this first.

    Args:
        analysis_df (pd.DataFrame): The analysis DataFrame.
        active_col_name (str): The name of the active column.
        action_col_name (str): The name of the action column.
        policy_num_col_name (str): The name of the policy number column.
        calendar_t_col_name (str): The name of the calendar time column.
        subject_id_col_name (str): The name of the subject ID column.
        action_prob_col_name (str): The name of the action probability column.
        reward_col_name (str): The name of the reward column.
    """
    require_analysis_df_nonempty(analysis_df)
    # Before everything else that touches analysis_df, because almost every other check
    # indexes it by one of these names. Until 2026-09-02 the presence check ran in the first
    # wave's analysis_df section, AFTER verify_analysis_df_summary_satisfactory had already
    # read six of the columns to build its summary plots -- so a single typo'd column name
    # surfaced as a bare KeyError from inside a plotting routine instead of the purpose-built
    # message here.
    require_all_named_columns_present_in_analysis_df(
        analysis_df,
        active_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_col_name,
        reward_col_name,
    )
    # Immediately after the column checks, and BEFORE anything that hashes a subject id.
    # Previously this ran at the end of the first wave's analysis_df block, where it could
    # never fire: an unhashable id makes pandas raise a bare "TypeError: unhashable type:
    # 'list'" from the .unique()/groupby()/duplicated() calls in the checks that run before it
    # (verified). Its own docstring claims it is what stands between such an id and an
    # unhelpful failure, so it has to run here to deliver that.
    require_hashable_subject_ids(analysis_df, active_col_name, subject_id_col_name)


# TODO: any checks needed here about alg update function type?
def perform_first_wave_input_checks(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
    reward_col_name,
    action_prob_func,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    alg_update_func_args_previous_betas_index,
    theta_est,
    beta_dim,
    suppress_interactive_data_checks,
    *,
    alg_update_func,
    alg_update_func_type,
    inference_func,
    inference_func_type,
    inference_func_args_theta_index,
    alg_update_func_args_mask_index=-1,
    alg_update_func_args_ragged_indices=(),
):
    """
    Perform the first wave of input checks on the analysis DataFrame and algorithm update function arguments.
    These are generally to make sure that the inputs are wired correctly and to verify the
    options selected are correct.

    Assumes perform_conditional_zeroth_wave_dataframe_checks has already passed on this frame -- the
    zeroth wave runs before theta_calculation_func produces the theta_est this wave needs, so
    its checks are not repeated here. Call it first if you are invoking this wave directly.

    Args:
        analysis_df (pd.DataFrame): The analysis DataFrame.
        active_col_name (str): The name of the active column.
        action_col_name (str): The name of the action column.
        policy_num_col_name (str): The name of the policy number column.
        calendar_t_col_name (str): The name of the calendar time column.
        subject_id_col_name (str): The name of the subject ID column.
        action_prob_col_name (str): The name of the action probability column.
        reward_col_name (str): The name of the reward column.
        action_prob_func (callable): The action probability function.
        action_prob_func_args (list): The arguments for the action probability function.
        action_prob_func_args_beta_index (int): The index of the beta argument in the action probability function arguments.
        alg_update_func_args (list): The arguments for the algorithm update function.
        alg_update_func_args_beta_index (int): The index of the beta argument in the algorithm update function arguments.
        alg_update_func_args_action_prob_index (int | None): The index of the action probability argument in the algorithm update function arguments.
        alg_update_func_args_action_prob_times_index (int | None): The index of the action probability times argument in the algorithm update function arguments.
        alg_update_func_args_previous_betas_index (int): The index of the previous betas argument in the algorithm update function arguments.
        theta_est (jnp.ndarray): The estimated theta.
        beta_dim (int): The dimension of the beta parameter.
        suppress_interactive_data_checks (bool): Whether to suppress interactive data checks.
        alg_update_func (callable): The algorithm update loss/estimating function, needed to
            validate the supplied argument tuples against its declared arity.
        alg_update_func_type (str): The algorithm update function type, one of
            lifejacket.constants.FunctionTypes.
        inference_func (callable): The inference loss/estimating function, needed to validate
            inference_func_args_theta_index against its declared arity.
        inference_func_type (str): The inference function type, one of
            lifejacket.constants.FunctionTypes.
        inference_func_args_theta_index (int): The index of the theta argument in the inference
            function's arguments.
        alg_update_func_args_mask_index (int): The index at which a validity mask will be
            appended to the algorithm update function args, or -1 when mask padding is off.
        alg_update_func_args_ragged_indices (tuple[int, ...]): The algorithm update function arg
            positions to self-pad when mask padding is on.

    Returns:
        dict: measurements from the checks that produce one (currently the action-probability
        reconstruction's agreement), for the diagnostic summary to report.
    """
    ### Validate the analysis DataFrame's columns FIRST.
    #
    # The frame being non-empty, every named column being PRESENT, and the subject ids being
    # hashable are perform_conditional_zeroth_wave_dataframe_checks' job, not repeated here: that wave
    # runs ahead of the caller's theta_calculation_func, which this wave's theta_est comes
    # from, so by the time we are called they have already passed. The dtype check below does
    # index analysis_df by these names and so depends on the presence check having run.
    require_named_columns_distinct(
        {
            "active_col_name": active_col_name,
            "action_col_name": action_col_name,
            "policy_num_col_name": policy_num_col_name,
            "calendar_t_col_name": calendar_t_col_name,
            "subject_id_col_name": subject_id_col_name,
            "action_prob_col_name": action_prob_col_name,
            "reward_col_name": reward_col_name,
        }
    )
    require_all_named_columns_not_object_type_in_analysis_df(
        analysis_df,
        active_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_col_name,
        reward_col_name,
    )

    ### Validate the supplied functions and the argument indices that address their args.
    #
    # These run FIRST because they are the cheapest checks in the file and every failure they
    # name otherwise surfaces only from deep inside the derivative precompute -- as a bare
    # IndexError, as a generic "Unknown update function type.", or (for a theta index that
    # addresses no position at all) as no error whatsoever -- after all the setup work has
    # already been done and with nothing naming the parameter that was misconfigured.
    require_valid_function_types(alg_update_func_type, inference_func_type)

    action_prob_arg_tuple_length = require_arg_tuple_lengths_consistent_and_callable(
        action_prob_func,
        action_prob_func_args,
        func_description="action_prob_func",
        key_description="decision_time",
    )
    alg_update_arg_tuple_length = require_arg_tuple_lengths_consistent_and_callable(
        alg_update_func,
        alg_update_func_args,
        func_description="alg_update_func",
        key_description="policy_num",
        mask_index=alg_update_func_args_mask_index,
    )

    require_arg_indices_supplied(
        {"action_prob_func_args_beta_index": action_prob_func_args_beta_index},
        "action_prob_func",
    )
    # A None length means nothing non-blank was supplied, so there are no positions for the
    # indices to address; the emptiness itself is other checks' finding.
    if action_prob_arg_tuple_length is not None:
        require_arg_indices_in_range(
            {"action_prob_func_args_beta_index": action_prob_func_args_beta_index},
            action_prob_arg_tuple_length,
            "action_prob_func",
        )

    alg_update_indices_by_name = {
        "alg_update_func_args_beta_index": alg_update_func_args_beta_index,
        "alg_update_func_args_action_prob_index": alg_update_func_args_action_prob_index,
        "alg_update_func_args_action_prob_times_index": (
            alg_update_func_args_action_prob_times_index
        ),
        "alg_update_func_args_previous_betas_index": (
            alg_update_func_args_previous_betas_index
        ),
    }
    require_arg_indices_supplied(
        {"alg_update_func_args_beta_index": alg_update_func_args_beta_index},
        "alg_update_func",
    )
    require_arg_indices_distinct(alg_update_indices_by_name, "alg_update_func")
    if alg_update_arg_tuple_length is not None:
        require_arg_indices_in_range(
            alg_update_indices_by_name, alg_update_arg_tuple_length, "alg_update_func"
        )
        require_mask_index_appends_after_supplied_args(
            alg_update_func_args_mask_index,
            alg_update_arg_tuple_length,
            "alg_update_func",
        )
        require_ragged_indices_valid(
            alg_update_func_args_ragged_indices,
            alg_update_func_args_mask_index,
            alg_update_arg_tuple_length,
            {
                "alg_update_func_args_beta_index": alg_update_func_args_beta_index,
                "alg_update_func_args_previous_betas_index": (
                    alg_update_func_args_previous_betas_index
                ),
            },
            "alg_update_func",
        )

    # The inference args dictionary is built internally by
    # post_deployment_analysis.process_inference_func_args, one entry per DECLARED PARAMETER of
    # inference_func, so the theta index is validated against that parameter count. There is
    # deliberately no inference-side mask/ragged validation to do: those public options were
    # removed on 2026-09-02 (they could not work -- see build_inference_layer_precompute's
    # docstring), so masking is an algorithm-side feature only.
    require_arg_indices_supplied(
        {"inference_func_args_theta_index": inference_func_args_theta_index},
        "inference_func",
    )
    require_arg_indices_in_range(
        {"inference_func_args_theta_index": inference_func_args_theta_index},
        _get_declared_positional_parameter_count(inference_func, "inference_func"),
        "inference_func",
    )
    # Runs after the theta index has been validated, since it uses that index to decide which
    # parameter is exempt from having to name a column.
    require_inference_func_parameter_names_are_analysis_df_columns(
        inference_func, inference_func_args_theta_index, analysis_df
    )

    ### Validate algorithm loss/estimating function and args
    require_supplied_args_finite(
        action_prob_func_args, "action_prob_func", "decision_time"
    )
    require_supplied_args_finite(alg_update_func_args, "alg_update_func", "policy_num")
    require_alg_update_args_given_for_all_subjects_at_each_update(
        analysis_df, subject_id_col_name, alg_update_func_args
    )
    require_no_policy_numbers_present_in_alg_update_args_but_not_analysis_df(
        analysis_df, policy_num_col_name, alg_update_func_args
    )
    require_beta_is_1D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_beta_index
    )
    require_previous_betas_is_2D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_previous_betas_index
    )
    # Derived here rather than passed in so this function stays callable on its own; it is the
    # same mapping construct_beta_index_by_policy_num_map builds for the estimator.
    beta_index_by_policy_num, _ = construct_beta_index_by_policy_num_map(
        analysis_df, policy_num_col_name, active_col_name
    )
    require_beta_dimensions_consistent(
        action_prob_func_args,
        action_prob_func_args_beta_index,
        alg_update_func_args,
        alg_update_func_args_beta_index,
        alg_update_func_args_previous_betas_index,
        beta_index_by_policy_num,
        beta_dim,
    )
    require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
    )
    require_every_update_policy_has_at_least_one_nonblank_arg_tuple(
        analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
    )

    # Interactive check
    confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
        alg_update_func_args_action_prob_index,
        alg_update_func_args_previous_betas_index,
        suppress_interactive_data_checks,
    )
    require_action_prob_times_given_if_index_supplied(
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
    )
    require_action_prob_index_given_if_times_supplied(
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
    )
    require_betas_match_in_alg_update_args_each_update(
        alg_update_func_args, alg_update_func_args_beta_index
    )
    require_previous_betas_match_in_alg_update_args_each_update(
        alg_update_func_args, alg_update_func_args_previous_betas_index
    )
    require_action_prob_args_in_alg_update_func_correspond_to_analysis_df(
        analysis_df,
        action_prob_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
    )
    require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        calendar_t_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_times_index,
    )

    ### Validate action prob function and args
    require_action_prob_func_args_given_for_all_subjects_at_each_decision(
        analysis_df, subject_id_col_name, action_prob_func_args
    )
    require_action_prob_func_args_given_for_all_decision_times(
        analysis_df, calendar_t_col_name, action_prob_func_args
    )
    action_prob_reconstruction = (
        require_action_probabilities_in_analysis_df_can_be_reconstructed(
            analysis_df,
            action_prob_col_name,
            calendar_t_col_name,
            subject_id_col_name,
            active_col_name,
            action_prob_func_args,
            action_prob_func,
        )
    )

    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
        analysis_df,
        calendar_t_col_name,
        action_prob_func_args,
        active_col_name,
        subject_id_col_name,
    )
    require_beta_is_1D_array_in_action_prob_args(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    require_betas_match_in_action_prob_func_args_each_policy(
        analysis_df,
        active_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        policy_num_col_name,
        action_prob_func_args,
        action_prob_func_args_beta_index,
    )
    # Runs after the reconstruction check above, which is what rejects the duplicate active
    # (decision_time, subject_id) rows that would otherwise make "the policy in force at this
    # cell" ambiguous for this check.
    require_recorded_action_prob_betas_match_update_betas_for_their_policy(
        analysis_df,
        active_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        policy_num_col_name,
        action_prob_func_args,
        action_prob_func_args_beta_index,
        alg_update_func_args,
        alg_update_func_args_beta_index,
    )

    ### Validate analysis_df

    # Interactive check
    verify_analysis_df_summary_satisfactory(
        analysis_df,
        subject_id_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        active_col_name,
        action_prob_col_name,
        reward_col_name,
        beta_dim,
        len(theta_est),
        suppress_interactive_data_checks,
    )

    require_all_subjects_have_all_times_in_analysis_df(
        analysis_df, calendar_t_col_name, subject_id_col_name
    )
    # The column presence and dtype checks that used to sit here now run at the very top of
    # this function, ahead of everything that indexes analysis_df by a caller-supplied name.
    #
    # The two consecutive-integer checks run BEFORE the per-subject structural checks below,
    # because they describe the study's time and policy axes as a whole. A global calendar-time
    # gap (times 1, 2, 3, 5) makes EVERY subject's active times non-contiguous, so with
    # require_contiguous_participation first the user was told "1 subject(s) leave the study
    # and return" about what is really one missing decision time for everybody.
    require_consecutive_integer_calendar_times(analysis_df, calendar_t_col_name)
    require_consecutive_integer_policy_numbers(
        analysis_df, active_col_name, policy_num_col_name
    )
    require_no_duplicate_subject_time_rows(
        analysis_df, calendar_t_col_name, subject_id_col_name
    )
    require_contiguous_participation(
        analysis_df, active_col_name, calendar_t_col_name, subject_id_col_name
    )
    require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        active_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        policy_num_col_name,
    )
    require_analysis_df_values_finite(
        analysis_df,
        active_col_name,
        action_col_name,
        policy_num_col_name,
        action_prob_col_name,
        reward_col_name,
    )
    require_binary_actions(analysis_df, active_col_name, action_col_name)
    require_binary_active_indicators(analysis_df, active_col_name)
    # The two consecutive-integer checks moved above, ahead of the per-subject structural
    # checks; require_hashable_subject_ids moved to the top of this function, ahead of every
    # check that hashes a subject id.
    # Positivity/overlap (recorded probabilities strictly inside (0, 1)) is enforced by
    # lifejacket.diagnostics.check_exploration_and_weights instead -- see the removed
    # require_action_probabilities_in_range_0_to_1's history for why: some legitimate
    # near-deterministic policies legitimately produce recorded probabilities of exactly 0.0/1.0
    # after floating-point rounding, so a hard assertion here would have been a
    # backward-incompatible break to the always-on input-check path.

    ### Validate theta estimation
    require_theta_is_1D_array(theta_est)
    require_theta_estimate_is_finite_and_nonempty(theta_est)

    # The measurements worth carrying into the diagnostic summary. Everything else in the first
    # wave is a black-and-white wiring question with nothing to report beyond having run.
    return {"action_prob_reconstruction": action_prob_reconstruction}


# TODO: Give a hard-to-use option to loosen this check somehow
def require_action_probabilities_in_analysis_df_can_be_reconstructed(
    analysis_df,
    action_prob_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    active_col_name,
    action_prob_func_args,
    action_prob_func,
):
    """
    Check that the action probabilities in the analysis DataFrame can be reconstructed from the supplied
    action probability function and its arguments.

    NOTE THAT THIS IS A HARD FAILURE IF THE RECONSTRUCTION DOESN'T PASS.
    """
    logger.info("Reconstructing action probabilities from function and arguments.")

    active_df = analysis_df[analysis_df[active_col_name] == 1]

    # A dict keyed by (decision_time, subject_id) would silently keep only the
    # last row for any duplicate active (calendar_t, subject_id) pair, quietly
    # dropping the rest from this check's coverage instead of comparing every
    # row like the old per-row DataFrame.apply() implementation did -- fail
    # loudly instead so bad input data can't silently defang this check.
    dup_mask = active_df.duplicated(
        subset=[calendar_t_col_name, subject_id_col_name], keep=False
    )
    if dup_mask.any():
        examples = (
            active_df.loc[dup_mask, [calendar_t_col_name, subject_id_col_name]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "analysis_df contains duplicate active rows for the same "
            f"(decision_time, subject_id) key (e.g. {examples}). This makes "
            "action probability reconstruction ambiguous; please deduplicate "
            "or fix the input data."
        )

    # Keyed by (decision_time, subject_id) so the "actual" value gathered for
    # comparison always lines up with the exact same key used to build the
    # "reconstructed" one below -- never relying on two different iteration
    # orders (a pandas row order vs. a dict's) coincidentally matching, which
    # could otherwise let a subtle reordering bug silently defang this check
    # instead of correctly failing it. Built via zip over numpy columns
    # rather than DataFrame.iterrows(), which is much slower for a check
    # whose whole point is to replace a slow per-row loop.
    actual_action_prob_by_key = dict(
        zip(
            zip(
                active_df[calendar_t_col_name].to_numpy(),
                active_df[subject_id_col_name].to_numpy(),
                strict=False,
            ),
            active_df[action_prob_col_name].to_numpy(),
            strict=False,
        )
    )

    reconstructed_chunks = []
    actual_chunks = []
    visited_keys = []
    unexpected_keys = []
    for decision_time, args_by_subject_id in action_prob_func_args.items():
        nontrivial_args_by_subject_id = {
            subject_id: args for subject_id, args in args_by_subject_id.items() if args
        }
        if not nontrivial_args_by_subject_id:
            continue
        for shape_group in group_user_args_by_shape(nontrivial_args_by_subject_id):
            group_subject_ids = sorted(shape_group.keys())
            batched_arg_lists, _ = get_batched_arg_lists_and_involved_user_ids(
                action_prob_func, group_subject_ids, shape_group
            )
            batched_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensors(
                batched_arg_lists
            )
            reconstructed_chunks.append(
                jax.vmap(action_prob_func, in_axes=batch_axes)(*batched_arg_tensors)
            )
            # Use .get with a placeholder rather than indexing directly, so a
            # malformed input (non-blank args for a decision_time/subject_id
            # analysis_df doesn't mark active) collects into a clear error
            # below instead of crashing here with a bare KeyError -- which
            # would preempt require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times,
            # the check purpose-built to report exactly this mismatch clearly.
            for subject_id in group_subject_ids:
                key = (decision_time, subject_id)
                if key not in actual_action_prob_by_key:
                    unexpected_keys.append(key)
                visited_keys.append(key)
            actual_chunks.append(
                jnp.array(
                    [
                        actual_action_prob_by_key.get(
                            (decision_time, subject_id), jnp.nan
                        )
                        for subject_id in group_subject_ids
                    ]
                )
            )

    if unexpected_keys:
        raise ValueError(
            "require_action_probabilities_in_analysis_df_can_be_reconstructed found "
            f"non-blank action_prob_func_args for {len(unexpected_keys)} "
            "(decision_time, subject_id) pair(s) the analysis DataFrame does not mark "
            f"active, e.g. {sorted(unexpected_keys)[:5]}. This means action_prob_func_args "
            "has real (non-empty-tuple) entries for times/subjects the analysis DataFrame "
            "marks inactive. Please see the contract for details."
        )

    # Every active (decision_time, subject_id) pair must have been visited --
    # otherwise this check would silently validate a strict subset of
    # active_df's rows instead of failing loudly, the way the original
    # per-row implementation would (it would raise on any active row whose
    # args were an empty tuple, since calling action_prob_func(*()) fails).
    missing_keys = set(actual_action_prob_by_key.keys()) - set(visited_keys)
    if missing_keys:
        raise ValueError(
            "require_action_probabilities_in_analysis_df_can_be_reconstructed could not "
            f"reconstruct a prediction for {len(missing_keys)} active (decision_time, "
            f"subject_id) row(s), e.g. {sorted(missing_keys)[:5]}. This means "
            "action_prob_func_args has blank (empty-tuple) entries for times/subjects "
            "the analysis DataFrame marks as active. Please see the contract for details."
        )

    reconstructed_action_probs = (
        jnp.concatenate(reconstructed_chunks) if reconstructed_chunks else jnp.array([])
    )
    actual_action_probs = (
        jnp.concatenate(actual_chunks) if actual_chunks else jnp.array([])
    )

    # An ABSOLUTE tolerance is dimensionally correct here, unlike for estimating-function
    # values: probabilities are unitless and bounded in (0, 1), so there is no reward-scale
    # exposure, and 1e-6 leaves ~10x headroom over float32 evaluation noise at typical
    # probabilities. Do not "fix" this to a relative/scale-aware tolerance.
    actual_values = np.asarray(actual_action_probs, dtype="float64")
    reconstructed_values = np.asarray(reconstructed_action_probs, dtype="float64")
    np.testing.assert_allclose(actual_values, reconstructed_values, atol=1e-6)
    # Returned so the diagnostic summary can report the measured agreement rather than prose
    # boilerplate; callers that ignore the return value are unaffected. NaN, not 0.0, for an
    # empty study: "agreed to within 0" would claim a measurement where none was made.
    return {
        "max_abs_difference": (
            float(np.max(np.abs(actual_values - reconstructed_values)))
            if actual_values.size
            else math.nan
        ),
        "num_cells": int(actual_values.size),
        "atol": 1e-6,
    }


def require_all_subjects_have_all_times_in_analysis_df(
    analysis_df, calendar_t_col_name, subject_id_col_name
):
    logger.info(
        "Checking that all subjects have the same set of unique calendar times."
    )
    # Get the unique calendar times
    unique_calendar_times = set(analysis_df[calendar_t_col_name].unique())

    # Group by subject ID and aggregate the unique calendar times for each subject
    subject_calendar_times = analysis_df.groupby(subject_id_col_name)[
        calendar_t_col_name
    ].apply(set)

    # Check if all subjects have the same set of unique calendar times
    if not subject_calendar_times.apply(lambda x: x == unique_calendar_times).all():
        raise AssertionError(
            "Not all subjects have all calendar times in the analysis DataFrame. Please see the contract for details."
        )


def require_alg_update_args_given_for_all_subjects_at_each_update(
    analysis_df, subject_id_col_name, alg_update_func_args
):
    logger.info(
        "Checking that algorithm update function args are given for all subjects at each update."
    )
    all_subject_ids = set(analysis_df[subject_id_col_name].unique())
    for policy_num in alg_update_func_args:
        assert set(alg_update_func_args[policy_num].keys()) == all_subject_ids, (
            f"Not all subjects present in algorithm update function args for policy number {policy_num}. Please see the contract for details."
        )


def require_action_prob_args_in_alg_update_func_correspond_to_analysis_df(
    analysis_df,
    action_prob_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    alg_update_func_args,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
):
    logger.info(
        "Checking that the action probabilities supplied in the algorithm update function args, if"
        " any, correspond to those in the analysis DataFrame for the corresponding subjects and decision"
        " times."
    )
    if alg_update_func_args_action_prob_index < 0:
        return

    # Precompute a lookup dictionary for faster access
    analysis_df_lookup = analysis_df.set_index(
        [calendar_t_col_name, subject_id_col_name]
    )[action_prob_col_name].to_dict()

    for policy_num, subject_args in alg_update_func_args.items():
        for subject_id, args in subject_args.items():
            if not args:
                continue
            arg_action_probs = args[alg_update_func_args_action_prob_index]
            action_prob_times = args[
                alg_update_func_args_action_prob_times_index
            ].flatten()

            # Use the precomputed lookup dictionary
            analysis_df_action_probs = [
                analysis_df_lookup[(decision_time.item(), subject_id)]
                for decision_time in action_prob_times
            ]

            # Explicit tolerances (previously np.allclose's silent defaults). Probabilities
            # are unitless and bounded, so absolute+relative tolerances at these fixed values
            # are dimensionally sound -- see the reconstruction check's comment above.
            assert np.allclose(
                arg_action_probs.flatten(),
                analysis_df_action_probs,
                rtol=1e-5,
                atol=1e-8,
            ), (
                f"There is a mismatch for subject {subject_id} between the action probabilities supplied"
                f" in the args to the algorithm update function at policy {policy_num} and those in"
                " the analysis DataFrame for the supplied times. Please see the contract for details."
            )


def require_action_prob_func_args_given_for_all_subjects_at_each_decision(
    analysis_df,
    subject_id_col_name,
    action_prob_func_args,
):
    logger.info(
        "Checking that action prob function args are given for all subjects at each decision time."
    )
    all_subject_ids = set(analysis_df[subject_id_col_name].unique())
    for decision_time in action_prob_func_args:
        assert set(action_prob_func_args[decision_time].keys()) == all_subject_ids, (
            f"Not all subjects present in algorithm update function args for decision time {decision_time}. Please see the contract for details."
        )


def require_action_prob_func_args_given_for_all_decision_times(
    analysis_df, calendar_t_col_name, action_prob_func_args
):
    logger.info(
        "Checking that action prob function args are given for all decision times."
    )
    all_times = set(analysis_df[calendar_t_col_name].unique())

    assert set(action_prob_func_args.keys()) == all_times, (
        "Not all decision times present in action prob function args. Please see the contract for details."
    )


def require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
    analysis_df: pd.DataFrame,
    calendar_t_col_name: str,
    action_prob_func_args: dict[str, dict[str, tuple[Any, ...]]],
    active_col_name,
    subject_id_col_name,
):
    logger.info(
        "Checking that action probability function args are blank for exactly the times each subject"
        "is not in the study according to the analysis DataFrame."
    )
    inactive_df = analysis_df[analysis_df[active_col_name] == 0]
    inactive_times_by_subject_according_to_analysis_df = (
        inactive_df.groupby(subject_id_col_name)[calendar_t_col_name]
        .apply(set)
        .to_dict()
    )

    inactive_times_by_subject_according_to_action_prob_func_args = (
        collections.defaultdict(set)
    )
    for decision_time, action_prob_args_by_subject in action_prob_func_args.items():
        for subject_id, action_prob_args in action_prob_args_by_subject.items():
            if not action_prob_args:
                inactive_times_by_subject_according_to_action_prob_func_args[
                    subject_id
                ].add(decision_time)

    assert (
        inactive_times_by_subject_according_to_analysis_df
        == inactive_times_by_subject_according_to_action_prob_func_args
    ), (
        "Inactive decision times according to the analysis DataFrame do not match up with the"
        " times for which action probability arguments are blank for all subjects. Please see the"
        " contract for details."
    )


def require_analysis_df_nonempty(analysis_df):
    """
    The analysis DataFrame must have at least one row.

    An empty frame is not caught anywhere else and produces a bare, misdirecting exception from
    whichever downstream helper reaches it first: helper_functions.
    construct_beta_index_by_policy_num_map indexes element [0] of the sorted non-fallback active
    policy numbers, so an empty (or entirely fallback) study raises IndexError("list index out
    of range") from inside it, naming nothing about the input.
    """
    logger.info("Checking that the analysis DataFrame is non-empty.")
    assert len(analysis_df) > 0, (
        "The analysis DataFrame is empty; there is nothing to analyze."
    )


def require_named_columns_distinct(col_names_by_parameter):
    """
    No two of the caller's column-name parameters may name the SAME column.

    A collision passes both the presence and dtype checks -- the column exists and is
    well-typed -- and then quietly misreads the study: reward_col_name accidentally set to the
    action column's name (an easy copy-paste in a wide analysis_df) makes
    verify_analysis_df_summary_satisfactory chart actions as rewards, and every check that
    compares two of these columns compares one column against itself.
    """
    logger.info("Checking that the named columns are distinct.")
    parameters_by_col_name = collections.defaultdict(list)
    for parameter, col_name in col_names_by_parameter.items():
        parameters_by_col_name[col_name].append(parameter)
    collisions = {
        col_name: sorted(parameters)
        for col_name, parameters in parameters_by_col_name.items()
        if len(parameters) > 1
    }
    assert not collisions, (
        f"These column-name parameters refer to the same analysis DataFrame column -- column "
        f"-> the parameters that name it: {collisions}. Each must name a distinct column. "
        "Please see the contract for details."
    )


def require_all_named_columns_present_in_analysis_df(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
    reward_col_name,
):
    """
    Every column the caller named must actually be in the analysis DataFrame.

    reward_col_name was missing from this check (and from the object-dtype one below) until
    2026-09-02, even though it is a required argument of analyze_dataset and is read by
    verify_analysis_df_summary_satisfactory -- so a typo in it produced a bare KeyError from
    inside a plotting routine rather than the message here.
    """
    logger.info(
        "Checking that all named columns are present in the analysis DataFrame."
    )
    missing_col_names = [
        col_name
        for col_name in (
            active_col_name,
            action_col_name,
            policy_num_col_name,
            calendar_t_col_name,
            subject_id_col_name,
            action_prob_col_name,
            reward_col_name,
        )
        if col_name not in analysis_df.columns
    ]
    # All of them at once rather than one assert per column: a caller who mis-wired their
    # column names has usually mis-wired more than one, and reporting them together (with the
    # columns that ARE present) saves a round of guess-and-rerun.
    assert not missing_col_names, (
        f"These named columns are not in the analysis DataFrame: {missing_col_names}. "
        f"Columns present: {sorted(map(str, analysis_df.columns))}."
    )


def require_all_named_columns_not_object_type_in_analysis_df(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
    reward_col_name,
):
    logger.info("Checking that all named columns are not type object.")
    # subject_id_col_name is deliberately NOT in this tuple. Every other column here is
    # consumed numerically -- compared to 0/1, cast to int64, fed through jnp.array, run
    # through min/max/range -- so object dtype genuinely breaks them. Subject IDs are only ever
    # opaque keys: they index per-subject dictionaries and are zipped with results, never
    # arithmetic. Excluding this column is what allows STRING subject IDs, which pandas stores
    # as object dtype and which this check rejected outright until 2026-09-02. The subject-id
    # column's actual contract is enforced by require_hashable_subject_ids instead.
    #
    # The parameter is kept for call-site compatibility (perform_first_wave_input_checks passes
    # these positionally).
    object_dtype_col_names = [
        col_name
        for col_name in (
            active_col_name,
            action_col_name,
            policy_num_col_name,
            calendar_t_col_name,
            action_prob_col_name,
            reward_col_name,
        )
        if analysis_df[col_name].dtype == "object"
    ]
    # Every offender at once, with the remedy. The previous message named only the FIRST one
    # ("At least {colname} is of object type...") because the check asserted inside the loop,
    # so a caller with two object columns fixed one and re-ran to find the next.
    assert not object_dtype_col_names, (
        f"These analysis DataFrame columns are of object type, but are consumed numerically: "
        f'{object_dtype_col_names}. Convert them with .astype("float64"); object dtype usually '
        "means the column holds mixed types, strings, or None. Note this is not a mere "
        "formality -- such a column fails later anyway, in "
        "helper_functions.get_active_df_column, as a bare "
        '"Value ... with dtype object is not a valid JAX array type" that names no column. '
        "(The subject-id column is deliberately exempt: subject ids are opaque keys, never "
        "used numerically, and may legitimately be strings.)"
    )


def require_binary_actions(analysis_df, active_col_name, action_col_name):
    logger.info("Checking that actions are binary.")
    # Actions are only meaningful while a subject is active, so the active filter stays (the
    # column legitimately holds NaN for out-of-study rows). The .astype("int64") this check
    # used until 2026-09-02 did not: it TRUNCATED toward zero before testing membership, so a
    # fractional action of 0.5 became 0 and passed.
    actions = analysis_df.loc[analysis_df[active_col_name] == 1, action_col_name]
    assert actions.isin([0, 1]).all(), (
        f"Actions are not binary. {action_col_name} takes the value(s) "
        f"{sorted(set(actions[~actions.isin([0, 1])].unique().tolist()), key=str)[:5]} on "
        "active rows; only 0 and 1 are allowed."
    )


def require_binary_active_indicators(analysis_df, active_col_name):
    logger.info("Checking that active indicators are binary.")
    # The WHOLE column, and no .astype("int64") -- both are the fix for a check that was
    # vacuous until 2026-09-02. It filtered to active == 1 and then asserted those values were
    # in {0, 1}, which they are by construction, so it could never fail: an active column of
    # {1, 2} passed it, and the integer cast additionally truncated 0.5 to 0. This column
    # decides which rows every other check and the whole estimator look at, so it is validated
    # over every row, including the out-of-study ones.
    active_indicators = analysis_df[active_col_name]
    offending_values = sorted(
        set(active_indicators[~active_indicators.isin([0, 1])].unique().tolist()),
        key=str,
    )
    assert not offending_values, (
        f"In-study indicators are not binary. {active_col_name} takes the value(s) "
        f"{offending_values[:5]}; only 0 and 1 are allowed."
    )


def _unique_sorted_integer_values(values, value_description):
    """
    The unique values of a numeric column as a SORTED integer array.

    Sorted because pandas' Series.unique() returns values in order of first APPEARANCE, so
    comparing it against a range() -- as the two consecutive-integer checks below do -- silently
    made those checks depend on analysis_df's row order and spuriously fail on a frame that was
    not already ordered by the column being checked.

    Integrality is asserted rather than assumed. Floats are supported for legacy reasons, but
    casting a genuinely fractional value with .astype("int64") truncates it toward zero, which
    would let a policy number of 1.5 masquerade as the perfectly consecutive 1.
    """
    unique_values = values.unique()
    if isinstance(unique_values, pd.api.extensions.ExtensionArray):
        # pandas' nullable dtypes (Int64, Float64, ...) return an ExtensionArray here, which
        # np.asarray turns into an OBJECT array -- and np.isfinite below then fails with a bare
        # "ufunc 'isfinite' not supported for the input types" TypeError instead of this
        # function's own message. Such a column passes
        # require_all_named_columns_not_object_type_in_analysis_df (its dtype is "Int64", not
        # "object"), so this path is genuinely reachable. Routing through to_numpy also turns
        # pd.NA into a NaN that the non-finite assertion below can report properly.
        try:
            unique_values = unique_values.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError) as e:
            raise AssertionError(
                f"{value_description} has dtype {values.dtype}, whose values could not be "
                f"interpreted as numbers: {e}."
            ) from e
    unique_values = np.sort(np.asarray(unique_values))
    if unique_values.size == 0:
        return unique_values
    if not np.issubdtype(unique_values.dtype, np.integer):
        assert np.all(np.isfinite(unique_values)), (
            f"{value_description} contains non-finite values: "
            f"{unique_values[~np.isfinite(unique_values)][:5].tolist()}."
        )
        non_integral = unique_values[unique_values != np.rint(unique_values)]
        assert non_integral.size == 0, (
            f"{value_description} contains non-integer values: "
            f"{non_integral[:5].tolist()}."
        )
        unique_values = unique_values.astype("int64")
    return unique_values


def require_consecutive_integer_policy_numbers(
    analysis_df, active_col_name, policy_num_col_name
):
    # TODO: This is a somewhat rough check of this, could also check nondecreasing temporally

    logger.info(
        "Checking that in-study, non-fallback policy numbers are consecutive integers."
    )

    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonnegative_policy_nums = active_df.loc[
        active_df[policy_num_col_name] >= 0, policy_num_col_name
    ]
    if nonnegative_policy_nums.empty:
        # Every active row used a fallback policy. There is no consecutive-integer sequence to
        # check; whether that is a sane study is another check's question.
        return
    unique_sorted = _unique_sorted_integer_values(
        nonnegative_policy_nums, f"Non-fallback, in-study {policy_num_col_name}"
    )
    assert np.array_equal(
        unique_sorted, range(unique_sorted.min(), unique_sorted.max() + 1)
    ), (
        f"Policy numbers are not consecutive integers. Non-fallback, in-study "
        f"{policy_num_col_name} values present: {unique_sorted.tolist()}."
    )


def require_consecutive_integer_calendar_times(analysis_df, calendar_t_col_name):
    # This is a somewhat rough check of this, more like checking there are no
    # gaps in the integers covered.  But we have other checks that all subjects
    # have same times, etc.
    # Note these times should be well-formed even when the subject is not in the study.
    logger.info("Checking that calendar times are consecutive integers.")
    if analysis_df.empty:
        return
    unique_sorted = _unique_sorted_integer_values(
        analysis_df[calendar_t_col_name], calendar_t_col_name
    )
    assert np.array_equal(
        unique_sorted, range(unique_sorted.min(), unique_sorted.max() + 1)
    ), (
        f"Calendar times are not consecutive integers. {calendar_t_col_name} values "
        f"present: {unique_sorted.tolist()}."
    )


def require_hashable_subject_ids(analysis_df, active_col_name, subject_id_col_name):
    """
    Subject IDs must be hashable, and mutually comparable, since they key every per-subject
    argument dictionary in the package and several call sites sort them.

    This is now the subject-id column's ONLY structural guard, and it is load-bearing.
    require_all_named_columns_not_object_type_in_analysis_df used to reject that column for
    being object dtype, which incidentally blocked every unhashable value (pandas has no
    non-object dtype that can hold one) -- but it also blocked ordinary STRING subject IDs, so
    the column was exempted from that check on 2026-09-02 and this check took over.

    Mutual comparability is checked alongside hashability because relaxing the dtype gate makes
    a MIXED id column (say int and str together, which object dtype permits) reachable for the
    first time. Nothing would catch it: it is hashable, so it keys dictionaries fine, and it
    would instead surface much later as a bare "TypeError: '<' not supported between instances
    of 'str' and 'int'" from one of the sorted(...) calls over subject ids in
    calculate_derivatives and in this module.
    """
    logger.info("Checking that subject IDs are hashable.")
    # An actual assert, over every active subject id. Until 2026-09-02 this called isinstance()
    # and THREW THE RESULT AWAY, so it enforced nothing at all, and it reached its single
    # sampled value with label-based [0] -- a KeyError whenever the row labeled 0 was not
    # active. Subject ids key every per-subject argument dictionary in the package, so an
    # unhashable one cannot be looked up at all.
    #
    # Iterating the raw values rather than calling .unique()/.drop_duplicates() is deliberate:
    # those hash internally, so an unhashable id would surface as a bare TypeError from pandas
    # instead of the message below.
    # The WHOLE column, not just the active rows. Everything that hashes subject ids does so
    # frame-wide -- analysis_df[subject_id].unique(), groupby(subject_id), and
    # duplicated(subset=[subject_id, calendar_t]) -- so an unhashable or mixed-type id sitting
    # only on an out-of-study row still breaks them. active_col_name is retained in the
    # signature for call-site compatibility.
    active_subject_ids = analysis_df[subject_id_col_name].tolist()
    unhashable_subject_ids = [
        subject_id
        for subject_id in active_subject_ids
        if not isinstance(subject_id, collections.abc.Hashable)
    ]
    assert not unhashable_subject_ids, (
        f"Subject IDs must be hashable; found {len(unhashable_subject_ids)} unhashable "
        f"value(s), e.g. {unhashable_subject_ids[:5]}. Subject IDs key every per-subject "
        "argument dictionary. Please see the contract for details."
    )

    # Mutual comparability, in two layers. Several call sites sort subject ids --
    # sorted(user_ids) in calculate_derivatives and sorted(shape_group.keys()) here -- and a
    # failed sort raises a bare TypeError from deep inside the derivative precompute that
    # names neither the column nor the offending values. The type grouping below gives the
    # sharper message for the common mistake (a MIXED column); the trial sort after it is the
    # actual contract, catching same-type ids that simply do not order (complex numbers, a
    # hashable class without __lt__). bool/int are grouped because Python orders them
    # together, as are the numeric types.
    def _id_type_name(subject_id):
        if isinstance(subject_id, (bool, int, float, np.integer, np.floating)):
            return "number"
        if isinstance(subject_id, (str, np.str_)):
            return "string"
        return type(subject_id).__name__

    type_names = {_id_type_name(subject_id) for subject_id in active_subject_ids}
    assert len(type_names) <= 1, (
        f"Subject IDs must all be of one comparable type, but the active rows of "
        f"{subject_id_col_name} mix {sorted(type_names)}. Subject ids are sorted in several "
        "places, and sorting a mixed column raises an uninformative TypeError from inside the "
        "derivative precompute. Please see the contract for details."
    )

    # The trial sort, on the unique ids (hashability was established above, so dict.fromkeys
    # is safe and keeps this O(u log u) in the unique count rather than the row count).
    unique_subject_ids = list(dict.fromkeys(active_subject_ids))
    try:
        sorted(unique_subject_ids)
    except TypeError as exc:
        raise AssertionError(
            f"Subject IDs in {subject_id_col_name} are not sortable ({exc}). Subject ids are "
            "sorted in several places (per-subject argument batching, derivative "
            "precomputation), so ids of a type without an ordering -- complex numbers, or a "
            "custom class that defines __hash__ but not __lt__ -- cannot be used. Please see "
            "the contract for details."
        ) from exc


def require_no_policy_numbers_present_in_alg_update_args_but_not_analysis_df(
    analysis_df, policy_num_col_name, alg_update_func_args
):
    logger.info(
        "Checking that policy numbers in algorithm update function args are present in the analysis DataFrame."
    )
    alg_update_policy_nums = sorted(alg_update_func_args.keys())
    analysis_df_policy_nums = sorted(analysis_df[policy_num_col_name].unique())
    assert set(alg_update_policy_nums).issubset(set(analysis_df_policy_nums)), (
        f"There are policy numbers present in algorithm update function args but not in the analysis DataFrame. "
        f"\nalg_update_func_args policy numbers: {alg_update_policy_nums}"
        f"\nanalysis_df policy numbers: {analysis_df_policy_nums}.\nPlease see the contract for details."
    )


def require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
    analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
):
    logger.info(
        "Checking that all policy numbers in the analysis DataFrame are present in the algorithm update function args."
    )
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonnegative_policy_nums = active_df.loc[
        active_df[policy_num_col_name] >= 0, policy_num_col_name
    ]
    if nonnegative_policy_nums.empty:
        # Every active row used a fallback policy, so no update-produced policy is required.
        return
    # Get the number of the initial policy. 0 is recommended but not required.
    #
    # .min() -- NOT the bare Series. Until 2026-09-02 this was the Series itself, which made
    # the comparison below an index-ALIGNED elementwise comparison of the column against
    # itself. That silenced this check completely: with no fallback policies the two objects
    # were identically labeled, every comparison was value > itself, and the resulting empty
    # set trivially satisfied issubset; with fallback policies present the labels differed and
    # pandas raised "Can only compare identically-labeled Series objects" instead.
    initial_policy_num = nonnegative_policy_nums.min()
    # Computed once and reused in the message, which previously recomputed it with a hardcoded
    # > 0 -- wrong for any study whose initial policy number is not 0.
    missing_policy_nums = set(
        active_df.loc[
            active_df[policy_num_col_name] > initial_policy_num, policy_num_col_name
        ].unique()
    ) - set(alg_update_func_args.keys())
    assert not missing_policy_nums, (
        f"There are non-fallback, non-initial policy numbers in the analysis DataFrame that "
        f"are not in the update function args: {sorted(missing_policy_nums)} (the initial "
        f"policy number is {initial_policy_num}). Please see the contract for details."
    )


def confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
    alg_update_func_args_action_prob_index,
    alg_update_func_args_previous_betas_index,
    suppress_interactive_data_checks,
):
    logger.info(
        "Confirming that action probabilities are not in algorithm update function args IF their index is not specified"
    )
    if (
        alg_update_func_args_action_prob_index < 0
        and alg_update_func_args_previous_betas_index < 0
    ):
        confirm_input_check_result(
            "\nYou specified that the algorithm update function supplied does not have action probabilities or previous betas in its arguments. Please verify this is correct.\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
        )


def require_action_prob_times_given_if_index_supplied(
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob times are given if index is supplied.")
    if alg_update_func_args_action_prob_index >= 0:
        assert alg_update_func_args_action_prob_times_index >= 0 and (
            alg_update_func_args_action_prob_times_index
            != alg_update_func_args_action_prob_index
        )


def require_action_prob_index_given_if_times_supplied(
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob index is given if times are supplied.")
    if alg_update_func_args_action_prob_times_index >= 0:
        assert alg_update_func_args_action_prob_index >= 0 and (
            alg_update_func_args_action_prob_times_index
            != alg_update_func_args_action_prob_index
        )


def require_beta_is_1D_array_in_alg_update_args(
    alg_update_func_args, alg_update_func_args_beta_index
):
    for policy_num in alg_update_func_args:
        for subject_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][subject_id]:
                continue
            assert (
                alg_update_func_args[policy_num][subject_id][
                    alg_update_func_args_beta_index
                ].ndim
                == 1
            ), "Beta is not a 1D array in the algorithm update function args."


def require_previous_betas_is_2D_array_in_alg_update_args(
    alg_update_func_args, alg_update_func_args_previous_betas_index
):
    if alg_update_func_args_previous_betas_index < 0:
        return

    for policy_num in alg_update_func_args:
        for subject_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][subject_id]:
                continue
            assert (
                alg_update_func_args[policy_num][subject_id][
                    alg_update_func_args_previous_betas_index
                ].ndim
                == 2
            ), "Previous betas is not a 2D array in the algorithm update function args."


def require_beta_is_1D_array_in_action_prob_args(
    action_prob_func_args, action_prob_func_args_beta_index
):
    for decision_time in action_prob_func_args:
        for subject_id in action_prob_func_args[decision_time]:
            if not action_prob_func_args[decision_time][subject_id]:
                continue
            assert (
                action_prob_func_args[decision_time][subject_id][
                    action_prob_func_args_beta_index
                ].ndim
                == 1
            ), "Beta is not a 1D array in the action probability function args."


def require_theta_is_1D_array(theta_est):
    assert theta_est.ndim == 1, "Theta is not a 1D array."


def require_output_dir_ready(output_dir):
    """
    The output directory must exist and be writable BEFORE the analysis runs.

    Everything analyze_dataset produces is written at the very END -- analysis.pkl,
    debug_pieces.pkl, diagnostic_report.pkl -- and nothing anywhere creates the directory, so a
    missing or unwritable output_dir throws away the whole run at the finish line: seconds on a
    test fixture, but hours of compute on a real oralytics-scale study.

    Deliberately NOT gated behind suppress_all_data_checks, unlike everything else in this
    module. Suppressing the data checks is a statement about trusting the DATA; it is not a
    request to discard a completed analysis because of a typo in a path.

    HONEST LIMIT: this cannot protect against the directory disappearing DURING the run -- a
    concurrent process removing it, which tests/integration_tests' own fixture does to its
    shared output tree on every invocation. It catches the ordinary cases: a path that never
    existed, a file where a directory was expected, and a directory that cannot be written to.
    """
    logger.info("Checking that the output directory exists and is writable.")
    output_path = pathlib.Path(output_dir)
    assert output_path.exists(), (
        f"output_dir {str(output_path)!r} does not exist. Every result this analysis produces "
        "is written there at the end of the run, so it must exist before the run starts; "
        "create it first."
    )
    assert output_path.is_dir(), f"output_dir {str(output_path)!r} is not a directory."
    # An actual write rather than os.access, which consults the real uid and the permission
    # bits and so can disagree with reality on an ACL-governed or read-only-mounted filesystem.
    write_probe_path = output_path / f".lifejacket_write_probe_{os.getpid()}"
    try:
        with open(write_probe_path, "wb"):
            pass
    except OSError as e:
        raise AssertionError(
            f"output_dir {str(output_path)!r} exists but could not be written to: {e}. Every "
            "result this analysis produces is written there at the end of the run."
        ) from e
    finally:
        try:
            write_probe_path.unlink()
        except OSError:
            # Never let probe cleanup mask the analysis; a stray dotfile is harmless.
            pass


def require_inference_func_parameter_names_are_analysis_df_columns(
    inference_func, inference_func_args_theta_index, analysis_df
):
    """
    Every parameter of inference_func except the theta one must be the name of an analysis_df
    column, because that is literally how its arguments get built.

    post_deployment_analysis.process_inference_func_args constructs the inference argument
    tuples itself, one entry per declared parameter, filling each from the analysis_df column
    whose name EQUALS that parameter's name. This contract is not written down anywhere else,
    and violating it produces a bare KeyError from helper_functions.get_active_df_column with
    nothing to indicate that a parameter name was the problem.

    Mirrors that function's own introspection exactly -- __code__.co_argcount / co_varnames,
    not inspect.signature -- so this check cannot pass on a function that then fails there. The
    __code__ requirement is asserted rather than worked around for the same reason: a
    jax.jit-wrapped inference_func has no usable __code__, so process_inference_func_args
    cannot build its arguments at all, and saying so here beats an AttributeError later.
    """
    logger.info(
        "Checking that inference function parameter names are analysis DataFrame columns."
    )
    assert hasattr(inference_func, "__code__"), (
        f"inference_func ({inference_func!r}) has no __code__ attribute, so its argument names "
        "cannot be read. Its arguments are built by matching each parameter NAME to an "
        "analysis DataFrame column of the same name, which requires a plain Python function "
        "-- a jax.jit-wrapped one will not work here. Please see the contract for details."
    )
    parameter_names = inference_func.__code__.co_varnames[
        : inference_func.__code__.co_argcount
    ]
    missing_column_names = {
        position: name
        for position, name in enumerate(parameter_names)
        # The theta position is supplied from theta_est, not from a column, so its name is
        # unconstrained.
        if position != inference_func_args_theta_index
        and name not in analysis_df.columns
    }
    assert not missing_column_names, (
        f"These inference_func parameters are not analysis DataFrame columns (position -> "
        f"parameter name): {missing_column_names}. Every inference_func parameter except the "
        f"one at inference_func_args_theta_index={inference_func_args_theta_index} is filled "
        f"from the analysis DataFrame column of the same name, so each must name a real "
        f"column. Columns present: {sorted(map(str, analysis_df.columns))}. Please see the "
        "contract for details."
    )


def require_valid_percentile_bootstrap_settings(
    percentile_bootstrap_draws, percentile_bootstrap_alpha, percentile_bootstrap_seed
):
    """
    The refit percentile bootstrap's settings must be in range.

    alpha is the one that matters: it is used directly as a quantile level, so alpha=95 (from
    thinking in percent rather than proportions) does not fail -- it silently produces a
    meaningless interval that is then REPORTED as percentile_bootstrap_ci in analysis.pkl. A
    negative seed, by contrast, fails loudly, but only from np.random.default_rng at the very
    end of a completed analysis.

    Validated even when draws == 0 (the bootstrap off) so that a bad alpha is caught the first
    time it is passed rather than the first time it is used, and NOT gated behind
    suppress_all_data_checks: these are configuration values, not data, and suppressing the
    data checks is not a request to report a nonsense confidence interval.
    """
    logger.info("Checking that the percentile bootstrap settings are in range.")
    assert (
        isinstance(percentile_bootstrap_draws, (int, np.integer))
        and not isinstance(percentile_bootstrap_draws, bool)
        and percentile_bootstrap_draws >= 0
    ), (
        f"percentile_bootstrap_draws must be a non-negative integer, got "
        f"{percentile_bootstrap_draws!r} (0 disables the bootstrap)."
    )
    # 1-9 are rejected, not merely discouraged: refit_percentile_bootstrap refuses to compute
    # quantiles from fewer than max(10, half the requested draws) surviving draws, so any
    # accepted count in 1-9 was GUARANTEED to report an all-NaN interval even when every
    # refit converged -- a configuration error dressed up as a bootstrap result.
    assert percentile_bootstrap_draws == 0 or percentile_bootstrap_draws >= 10, (
        f"percentile_bootstrap_draws must be 0 (off) or at least 10, got "
        f"{percentile_bootstrap_draws!r}: the quantile step requires at least "
        f"max(10, half the requested draws) surviving draws, so 1-9 draws always produce an "
        f"all-NaN interval. In practice use hundreds (docs/adr/0003 used 300)."
    )
    assert 0.0 < percentile_bootstrap_alpha < 1.0, (
        f"percentile_bootstrap_alpha must be strictly between 0 and 1 -- it is a PROPORTION, "
        f"not a percentage, so pass 0.05 for a 95% interval -- got "
        f"{percentile_bootstrap_alpha!r}."
    )
    assert percentile_bootstrap_seed is None or (
        isinstance(percentile_bootstrap_seed, (int, np.integer))
        and not isinstance(percentile_bootstrap_seed, bool)
        and percentile_bootstrap_seed >= 0
    ), (
        f"percentile_bootstrap_seed must be None or a non-negative integer "
        f"(np.random.default_rng rejects negative seeds), got "
        f"{percentile_bootstrap_seed!r}."
    )


def require_no_duplicate_subject_time_rows(
    analysis_df, calendar_t_col_name, subject_id_col_name
):
    """
    (subject_id, calendar_t) must identify at most one row in the WHOLE analysis DataFrame.

    Every per-subject-per-time lookup in the package is a dictionary keyed this way, so a
    duplicate silently keeps whichever row happened to be built last and drops the other.
    require_all_subjects_have_all_times_in_analysis_df compares SETS of times, so duplicates
    sail through it, and the only existing duplicate detection lives inside
    require_action_probabilities_in_analysis_df_can_be_reconstructed and looks at ACTIVE rows
    only.
    """
    logger.info("Checking that (subject_id, calendar_t) identifies at most one row.")
    duplicate_mask = analysis_df.duplicated(
        subset=[subject_id_col_name, calendar_t_col_name], keep=False
    )
    if not duplicate_mask.any():
        return
    examples = (
        analysis_df.loc[duplicate_mask, [subject_id_col_name, calendar_t_col_name]]
        .drop_duplicates()
        .head(5)
        .to_dict("records")
    )
    raise AssertionError(
        f"The analysis DataFrame has {int(duplicate_mask.sum())} row(s) sharing a "
        f"(subject_id, calendar_t) key, e.g. {examples}. Every per-subject-per-time structure "
        "in the package is keyed this way, so duplicates silently drop data. Please "
        "deduplicate the input."
    )


def require_nondecreasing_policy_numbers_over_time(
    analysis_df,
    active_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    policy_num_col_name,
):
    """
    Each subject's NON-FALLBACK policy numbers must not decrease as calendar time advances.

    The estimator maps each decision time to the policy in force and from there to the beta
    that policy's update produced, so a subject acting under an EARLIER policy after a later
    one contradicts the temporal structure the whole adjustment is built on -- the recorded
    action probabilities and the betas being differentiated would come from different points in
    the algorithm's history. This is the check the TODO on
    require_consecutive_integer_policy_numbers has always asked for ("could also check
    nondecreasing temporally"); that check only verifies the SET of policy numbers has no gaps.

    Fallback (negative) policy numbers are excluded rather than ordered: they mark "the
    algorithm was bypassed here", carry no position in the update sequence, and are expected to
    interleave freely. The check is applied to the non-negative subsequence, so a fallback in
    the middle of a run neither breaks monotonicity nor hides a genuine regression around it.
    """
    logger.info(
        "Checking that non-fallback policy numbers do not decrease over time per subject."
    )
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    offending_subject_ids = []
    for subject_id, subject_df in active_df.groupby(subject_id_col_name, sort=False):
        ordered = subject_df.sort_values(calendar_t_col_name)[policy_num_col_name]
        nonnegative = ordered[ordered >= 0].to_numpy()
        if nonnegative.size > 1 and np.any(np.diff(nonnegative) < 0):
            offending_subject_ids.append(subject_id)
    assert not offending_subject_ids, (
        f"{len(offending_subject_ids)} subject(s) have non-fallback policy numbers that "
        f"DECREASE as calendar time advances, e.g. "
        f"{sorted(offending_subject_ids, key=str)[:5]}. A subject cannot act under an earlier "
        "policy after a later one; the estimator routes each decision time's beta by the "
        "policy in force. Please see the contract for details."
    )


def require_contiguous_participation(
    analysis_df, active_col_name, calendar_t_col_name, subject_id_col_name
):
    """
    Each subject must be active for one unbroken stretch of calendar time, and for at least one
    decision time.

    Staggered recruitment means subjects legitimately start late and leave early, but LEAVING
    AND RETURNING silently changes what "this subject's history so far" means: the per-subject
    ragged argument positions are built as one contiguous run of decision times, and the
    padding/masking path assumes a single active window per subject (see
    batched_weighted_estimating_function_stack.assert_no_intra_window_gaps, which enforces
    exactly this on the action-probability grid but only once the precompute is reached).

    A subject with NO active decision times is rejected in the same breath: it contributes an
    empty argument tuple to every layer, forms its own degenerate shape bucket, and adds a row
    of zeros to the per-subject stacks that then counts in the denominator of every average.
    """
    logger.info(
        "Checking that each subject is active for one unbroken stretch of decision times."
    )
    noncontiguous_subject_ids = []
    inactive_subject_ids = []
    for subject_id, subject_df in analysis_df.groupby(subject_id_col_name, sort=False):
        # Measured on the calendar-time VALUES, not on row positions within this subject's
        # rows. Position-based spans read a subject whose intermediate row is simply MISSING
        # (rather than present-and-inactive) as contiguous, which makes the check depend on
        # require_all_subjects_have_all_times_in_analysis_df having run first; comparing the
        # values keeps it self-sufficient for direct callers too.
        active_times = subject_df.loc[
            subject_df[active_col_name] == 1, calendar_t_col_name
        ].to_numpy()
        if active_times.size == 0:
            inactive_subject_ids.append(subject_id)
            continue
        distinct_times = np.unique(active_times)
        span = distinct_times[-1] - distinct_times[0] + 1
        if span != distinct_times.size:
            noncontiguous_subject_ids.append(subject_id)
    assert not inactive_subject_ids, (
        f"{len(inactive_subject_ids)} subject(s) are never active at any decision time, e.g. "
        f"{sorted(inactive_subject_ids, key=str)[:5]}. Such a subject contributes nothing but "
        "still counts in the denominator of every per-subject average; drop them from the "
        "analysis DataFrame instead. Please see the contract for details."
    )
    assert not noncontiguous_subject_ids, (
        f"{len(noncontiguous_subject_ids)} subject(s) leave the study and return, e.g. "
        f"{sorted(noncontiguous_subject_ids, key=str)[:5]}. Each subject must be active for "
        "one unbroken stretch of decision times; a gap changes what that subject's recorded "
        "history means. Please see the contract for details."
    )


def _numeric_values_or_none(column):
    """
    A float64 numpy view of a pandas column, or None when it is not numeric at all.

    Exists because Series.to_numpy() on a pandas NULLABLE dtype (Int64, Float64) returns an
    OBJECT array, so a plain np.issubdtype(..., np.number) test reads such a column as
    non-numeric and skips it. That was a silent hole: a nullable-dtype column also passes
    require_all_named_columns_not_object_type_in_analysis_df (its dtype is "Float64", not
    "object"), so a pd.NA reward on an active row was reported by NEITHER check -- precisely
    what the finiteness check exists to catch. Verified reachable before this fix.

    Object-dtype columns holding real numbers are converted too rather than skipped, for the
    same reason: they pass the object-dtype check for the exempt subject-id column's sake and
    would otherwise hide a NaN.
    """
    if isinstance(
        column.dtype, pd.api.types.pandas_dtype("Int64").__class__
    ) or isinstance(column.array, pd.api.extensions.ExtensionArray):
        try:
            return column.to_numpy(dtype="float64", na_value=np.nan)
        except (TypeError, ValueError):
            return None
    values = column.to_numpy()
    if np.issubdtype(values.dtype, np.number):
        return values.astype("float64")
    try:
        # An object array of genuine numbers converts cleanly; anything else raises and is
        # correctly treated as non-numeric.
        return values.astype("float64")
    except (TypeError, ValueError):
        return None


def require_analysis_df_values_finite(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    action_prob_col_name,
    reward_col_name,
):
    """
    The numeric analysis DataFrame columns must be finite ON ACTIVE ROWS.

    Active rows ONLY, and this is the whole scope decision: policy_num, action, action_prob and
    reward all legitimately hold NaN where a subject is out of study -- verified on this repo's
    own fixtures, where every one of those four columns is non-finite over the whole column and
    finite over the active rows. Checking them frame-wide would reject every real study.

    Not covered here because another check already covers it better: the active indicator
    (require_binary_active_indicators, which rejects NaN as non-binary over the whole column)
    and calendar_t (require_consecutive_integer_calendar_times, which asserts finiteness and
    integrality over the whole column, since times must be well-formed even out of study).

    Why it matters: a single NaN reward propagates silently through theta estimation into the
    bread, the meat and the reported variance, and currently surfaces only as a diagnostic
    finding (diagnostics' bread-stability check) after the entire computation has run.
    """
    logger.info("Checking that the numeric analysis DataFrame columns are finite.")
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonfinite_counts = {}
    for col_name in (
        action_col_name,
        policy_num_col_name,
        action_prob_col_name,
        reward_col_name,
    ):
        values = _numeric_values_or_none(active_df[col_name])
        if values is None:
            # A genuinely non-numeric column is
            # require_all_named_columns_not_object_type_in_analysis_df's finding.
            continue
        nonfinite = int(np.count_nonzero(~np.isfinite(values)))
        if nonfinite:
            nonfinite_counts[col_name] = nonfinite
    assert not nonfinite_counts, (
        f"These analysis DataFrame columns contain non-finite values (NaN or inf) on ACTIVE "
        f"rows -- column -> count: {nonfinite_counts}. Non-finite inputs propagate silently "
        "into the estimates and the reported variance. Please see the contract for details."
    )


def require_supplied_args_finite(
    args_by_subject_id_by_key, func_description, key_description
):
    """
    Every numeric value in the supplied argument tuples must be finite.

    One vectorized np.isfinite per supplied array. That is a full pass over the argument data,
    which on a large study is not free -- but it is a single pass over arrays already in memory,
    negligible beside the JAX precompute that follows, and a non-finite argument otherwise
    reaches the estimator and turns the entire stacked system into NaN.

    Non-numeric entries are skipped rather than rejected: argument tuples legitimately carry
    non-array scalars (the fixtures pass floats like lower_clip and integers like n), and
    anything that is not a number has no finiteness to test.
    """
    logger.info("Checking that supplied %s arguments are finite.", func_description)
    offenders = {}
    for key, args_by_subject_id in args_by_subject_id_by_key.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            for position, arg in enumerate(args):
                try:
                    values = np.asarray(arg)
                except ValueError:
                    # A ragged nested sequence cannot be arrayed; the arg-shape checks and the
                    # JAX precompute both speak to that, and it is not a finiteness question.
                    continue
                if not np.issubdtype(values.dtype, np.number):
                    # An OBJECT array of real numbers is converted rather than skipped, since
                    # skipping it would hide any NaN inside. Everything else -- None, strings,
                    # nested objects -- is genuinely not a number and has no finiteness to
                    # test; note astype("float64") would turn None into a NaN and report it as
                    # a non-finite VALUE, which is a misleading way to describe a None.
                    flat_values = values.ravel().tolist()
                    if not flat_values or not all(
                        isinstance(value, numbers.Number) for value in flat_values
                    ):
                        continue
                    values = values.astype("float64")
                if not np.all(np.isfinite(values)):
                    offenders[(key, subject_id, position)] = int(
                        np.count_nonzero(~np.isfinite(values))
                    )
    example_offenders = dict(
        sorted(offenders.items(), key=lambda item: str(item[0]))[:5]
    )
    assert not offenders, (
        f"{len(offenders)} supplied {func_description} argument position(s) contain non-finite "
        f"values (NaN or inf). Offending ({key_description}, subject_id, arg position) -> "
        f"count, up to 5 shown: {example_offenders}. Please see the contract for details."
    )


def require_theta_estimate_is_finite_and_nonempty(theta_est):
    """
    theta_calculation_func's output must be a non-empty, finite vector.

    theta_est is the inferential target: it seeds the joint estimating system, so a NaN or inf
    in it makes every downstream quantity NaN, and an EMPTY theta makes theta_dim zero, which
    turns the stacked-system arithmetic into a silently degenerate no-op rather than an error.
    require_theta_is_1D_array checks only the number of dimensions.
    """
    logger.info("Checking that the theta estimate is finite and non-empty.")
    theta = np.asarray(theta_est)
    assert theta.size > 0, (
        "theta_calculation_func returned an empty theta estimate; the inferential target must "
        "have at least one component."
    )
    assert np.issubdtype(theta.dtype, np.number), (
        f"theta_calculation_func returned a non-numeric theta estimate (dtype {theta.dtype})."
    )
    nonfinite_positions = np.flatnonzero(~np.isfinite(theta))
    assert nonfinite_positions.size == 0, (
        f"theta_calculation_func returned a theta estimate with non-finite values at "
        f"component(s) {nonfinite_positions[:5].tolist()}: {theta}. Every downstream quantity "
        "is computed from theta, so this would make the whole result NaN."
    )


def require_beta_dimensions_consistent(
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
    alg_update_func_args_previous_betas_index,
    beta_index_by_policy_num,
    beta_dim,
):
    """
    Every recorded beta must have length beta_dim, and every previous-betas block must be
    shaped (number of PRIOR updates, beta_dim).

    beta_dim is read off the FIRST non-blank action-prob argument tuple
    (helper_functions.calculate_beta_dim) and then used to slice and index the stacked system
    everywhere, so a beta of a different length elsewhere is not a local error -- it silently
    changes which components of the joint bread and meat matrices mean what.
    helper_functions.collect_all_post_update_betas eventually calls jnp.array over the
    per-policy betas, which fails on ragged input with a message that names no policy.

    The previous-betas shape is asserted exactly, not loosely:
    arg_threading_helpers.thread_update_func_args substitutes
    all_post_update_betas[: len(recorded_previous_betas)], i.e. it uses the RECORDED ROW COUNT
    as the slice length, so that row count must equal the number of updates preceding this
    policy -- which is precisely this policy's index in beta_index_by_policy_num. Verified
    against the repo's own previous-betas fixture, where all eight update policies satisfy
    shape == (beta_index, beta_dim) exactly. A wrong row count would quietly thread in the wrong
    slice of history.
    """
    logger.info("Checking that all recorded betas have consistent dimensions.")
    wrong_action_prob_beta_lengths = {}
    for decision_time, args_by_subject_id in action_prob_func_args.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            beta = np.asarray(args[action_prob_func_args_beta_index])
            if beta.shape != (beta_dim,):
                wrong_action_prob_beta_lengths[(decision_time, subject_id)] = beta.shape
    example_action_prob_beta_lengths = dict(
        sorted(wrong_action_prob_beta_lengths.items(), key=lambda item: str(item[0]))[
            :5
        ]
    )
    assert not wrong_action_prob_beta_lengths, (
        f"{len(wrong_action_prob_beta_lengths)} action_prob_func_args beta(s) are not shaped "
        f"({beta_dim},), the dimension taken from the first supplied beta. Offending "
        f"(decision_time, subject_id) -> shape, up to 5 shown: "
        f"{example_action_prob_beta_lengths}. Please see the contract for details."
    )

    wrong_update_beta_lengths = {}
    wrong_previous_betas_shapes = {}
    for policy_num, args_by_subject_id in alg_update_func_args.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            beta = np.asarray(args[alg_update_func_args_beta_index])
            if beta.shape != (beta_dim,):
                wrong_update_beta_lengths[(policy_num, subject_id)] = beta.shape
            if alg_update_func_args_previous_betas_index < 0:
                continue
            # A policy absent from beta_index_by_policy_num is the initial policy or a
            # fallback; neither has a position in the update sequence, so there is no expected
            # row count to check against. Skipped BEFORE indexing the tuple, so a short tuple
            # at such a policy cannot raise a bare IndexError from inside this check.
            if policy_num not in beta_index_by_policy_num:
                continue
            previous_betas = np.asarray(args[alg_update_func_args_previous_betas_index])
            expected_shape = (beta_index_by_policy_num[policy_num], beta_dim)
            if previous_betas.shape != expected_shape:
                wrong_previous_betas_shapes[(policy_num, subject_id)] = (
                    f"{previous_betas.shape} != expected {expected_shape}"
                )
    example_update_beta_lengths = dict(
        sorted(wrong_update_beta_lengths.items(), key=lambda item: str(item[0]))[:5]
    )
    assert not wrong_update_beta_lengths, (
        f"{len(wrong_update_beta_lengths)} alg_update_func_args beta(s) are not shaped "
        f"({beta_dim},). Offending (policy_num, subject_id) -> shape, up to 5 shown: "
        f"{example_update_beta_lengths}. Please see the contract for details."
    )
    example_previous_betas_shapes = dict(
        sorted(wrong_previous_betas_shapes.items(), key=lambda item: str(item[0]))[:5]
    )
    assert not wrong_previous_betas_shapes, (
        f"{len(wrong_previous_betas_shapes)} alg_update_func_args previous-betas block(s) have "
        f"the wrong shape; each must be (number of updates before that policy, {beta_dim}), "
        f"because the threading code uses the recorded row count as the slice length into the "
        f"shared beta history. Offending (policy_num, subject_id) -> found != expected, up to "
        f"5 shown: {example_previous_betas_shapes}. Please see the contract for details."
    )


def require_betas_match_in_action_prob_func_args_each_policy(
    analysis_df,
    active_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    policy_num_col_name,
    action_prob_func_args,
    action_prob_func_args_beta_index,
):
    """
    The beta recorded for the action probabilities must be a function of the POLICY in force,
    not of the decision time.

    This replaces require_betas_match_in_action_prob_func_args_each_decision, which required
    every subject at a given calendar time to share one beta. That was the wrong invariant:
    multiple policies at a single decision time is a supported configuration (fallback policies
    interleave with the current one, and verify_analysis_df_summary_satisfactory reports the
    count of such times as an ordinary statistic), and whenever two active subjects at one time
    were on different policies -- so legitimately had different betas -- the old check fired
    with a message about decision times that named nothing actionable.

    Keying on the policy number is both more permissive where it should be (different policies
    at one time are fine) and stricter where it matters (the SAME policy must present the same
    beta at every time it is in force, which is exactly what the threading code assumes when it
    substitutes one shared beta per policy).

    Fallback (negative) policy numbers are skipped, not grouped: a negative number marks "the
    algorithm was bypassed" rather than identifying a particular parameter vector, so two cells
    both labelled -1 need not share a beta, and the threading code never substitutes a beta for
    them.
    """
    logger.info(
        "Checking that betas match across cells sharing a policy number in the action prob args."
    )
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    policy_num_by_key = dict(
        zip(
            zip(
                active_df[calendar_t_col_name].to_numpy(),
                active_df[subject_id_col_name].to_numpy(),
                strict=False,
            ),
            active_df[policy_num_col_name].to_numpy(),
            strict=False,
        )
    )

    first_beta_by_policy_num = {}
    first_key_by_policy_num = {}
    mismatches = []
    for decision_time, args_by_subject_id in action_prob_func_args.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            policy_num = policy_num_by_key.get((decision_time, subject_id))
            if policy_num is None or policy_num < 0:
                continue
            beta = np.asarray(args[action_prob_func_args_beta_index])
            if policy_num not in first_beta_by_policy_num:
                first_beta_by_policy_num[policy_num] = beta
                first_key_by_policy_num[policy_num] = (decision_time, subject_id)
                continue
            reference = first_beta_by_policy_num[policy_num]
            if beta.shape != reference.shape or not np.array_equal(beta, reference):
                mismatches.append(
                    (
                        policy_num,
                        first_key_by_policy_num[policy_num],
                        (decision_time, subject_id),
                    )
                )

    assert not mismatches, (
        f"The action prob args record different betas for cells sharing a policy number, at "
        f"{len(mismatches)} cell(s). A policy is one parameter vector, and the estimator "
        f"substitutes a single shared beta per policy, so every cell in force under a given "
        f"policy must record the same beta. Up to 5 shown as (policy_num, first cell, "
        f"disagreeing cell): {sorted(mismatches, key=str)[:5]}. Please see the contract for "
        "details."
    )


def require_valid_function_types(alg_update_func_type, inference_func_type):
    """
    The supplied function types must be ones the derivative code recognizes.

    Without this, a typo ("Loss", "estimating_function") is not caught until
    calculate_derivatives raises a bare "Unknown update function type." from inside the
    derivative precompute -- after theta estimation, every input check and all of the
    data-structure preparation have already run.
    """
    logger.info("Checking that the supplied function types are recognized.")
    valid_function_types = (FunctionTypes.LOSS, FunctionTypes.ESTIMATING)
    for name, supplied_type in (
        ("alg_update_func_type", alg_update_func_type),
        ("inference_func_type", inference_func_type),
    ):
        assert supplied_type in valid_function_types, (
            f"{name}={supplied_type!r} is not a recognized function type; it must be one of "
            f"{list(valid_function_types)} (lifejacket.constants.FunctionTypes). Please see "
            "the contract for details."
        )


def _get_positional_parameter_bounds(func, func_description):
    """
    The (minimum, maximum) number of positional arguments func can accept, where maximum is
    math.inf when it declares *args.

    Resolved with inspect.signature, NOT func.__code__.co_argcount, so a jax.jit-wrapped
    function reports the signature of the function it wraps.

    Deliberately a RANGE rather than a single count. The production argument-batching code
    derives the argument count from the DATA, not from the signature -- see
    vmap_helpers.build_batched_arg_lists_by_subject, whose docstring says exactly that and
    explains why (function introspection is wrong for a wrapped function). So a thin wrapper
    analyzes perfectly well: both a forwarding shim declaring *args and a function carrying an
    extra defaulted hyperparameter were verified end to end on this repo's own fixture to
    return estimates BIT-IDENTICAL to the unwrapped function. Demanding that the supplied tuple
    length EQUAL the declared parameter count would reject both, and an always-on input check
    that rejects data the estimator handles correctly is worse than no check at all -- there is
    no opt-out short of suppress_all_data_checks.
    """
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError) as e:
        raise AssertionError(
            f"Could not inspect the signature of {func_description} ({func!r}): {e}. The "
            "package needs a function whose signature can be inspected. Please see the "
            "contract for details."
        ) from e

    minimum_positional = 0
    maximum_positional = 0
    for parameter in parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            maximum_positional += 1
            if parameter.default is inspect.Parameter.empty:
                minimum_positional += 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            maximum_positional = math.inf
    return minimum_positional, maximum_positional


def _describe_positional_bounds(minimum_positional, maximum_positional):
    if maximum_positional == math.inf:
        return f"at least {minimum_positional}"
    if minimum_positional == maximum_positional:
        return f"exactly {minimum_positional}"
    return f"{minimum_positional} to {maximum_positional}"


def _get_declared_positional_parameter_count(func, func_description):
    """
    The number of positional parameters func declares -- the count
    post_deployment_analysis.process_inference_func_args effectively uses (via
    __code__.co_argcount) when it builds one inference argument per declared parameter, filling
    each from the analysis_df column of the same name. *args is excluded, matching co_argcount.
    """
    _, maximum_positional = _get_positional_parameter_bounds(func, func_description)
    if maximum_positional == math.inf:
        parameters = inspect.signature(func).parameters
        return sum(
            1
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )
    return maximum_positional


def require_arg_tuple_lengths_consistent_and_callable(
    func,
    args_by_subject_id_by_key,
    *,
    func_description,
    key_description,
    mask_index=-1,
):
    """
    Every non-blank supplied argument tuple must have the SAME length, and the function must be
    able to accept that many positional arguments.

    The consistency half is the substantive one, and it is what catches the real wiring error:
    argument tuples of the wrong width for the function they are meant to feed. It is stated
    over the supplied DATA because that is what the production batching code actually reads
    (vmap_helpers.build_batched_arg_lists_by_subject takes its argument count from the first
    subject in each bucket and indexes every other subject's tuple by that range).

    The callability half is deliberately a RANGE check, not an equality one -- see
    _get_positional_parameter_bounds for the wrapper cases an equality check would wrongly
    reject.

    When mask_index >= 0 the function is called with one MORE argument than is supplied, since
    the padding path appends the validity mask as a new last argument
    (batched_weighted_estimating_function_stack.self_pad_ragged_args_and_build_mask).

    Returns the common supplied tuple length, or None when no non-blank tuple was supplied at
    all (a degenerate study other checks speak to), in which case there is nothing for the
    caller's argument indices to address.
    """
    logger.info(
        "Checking that supplied %s argument tuples are consistent and callable.",
        func_description,
    )
    supplied_lengths_by_key = {}
    for key, args_by_subject_id in args_by_subject_id_by_key.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            supplied_lengths_by_key[(key, subject_id)] = len(args)
    if not supplied_lengths_by_key:
        return None

    distinct_lengths = sorted(set(supplied_lengths_by_key.values()))
    if len(distinct_lengths) > 1:
        # One example key per length, chosen deterministically. Sorted by str() because the
        # keys mix types across studies (int or float policy numbers, arbitrary hashable
        # subject ids), which are not mutually orderable.
        example_by_length = {
            length: min(
                (
                    key
                    for key, value in supplied_lengths_by_key.items()
                    if value == length
                ),
                key=str,
            )
            for length in distinct_lengths
        }
        raise AssertionError(
            f"Supplied {func_description} argument tuples do not all have the same length; "
            f"lengths {distinct_lengths} are all present. One example "
            f"({key_description}, subject_id) per length: {example_by_length}. Every non-blank "
            "tuple must carry the same arguments in the same positions. Please see the contract "
            "for details."
        )

    (supplied_length,) = distinct_lengths
    call_length = supplied_length + 1 if mask_index >= 0 else supplied_length
    minimum_positional, maximum_positional = _get_positional_parameter_bounds(
        func, func_description
    )
    mask_note = (
        f" (the {supplied_length} supplied, plus the validity mask that will be appended at "
        f"mask index {mask_index})"
        if mask_index >= 0
        else ""
    )
    assert minimum_positional <= call_length <= maximum_positional, (
        f"{func_description} will be called with {call_length} positional argument(s)"
        f"{mask_note}, but its signature accepts "
        f"{_describe_positional_bounds(minimum_positional, maximum_positional)}. Please see "
        "the contract for details."
    )
    return supplied_length


def require_arg_indices_supplied(indices_by_name, func_description):
    """
    Indices for the positions the package ALWAYS threads a shared parameter into -- beta and
    theta -- must actually be supplied, i.e. be non-negative.

    A negative index is read as "this position is absent" by the threading code, which then
    never substitutes the shared parameter at all. For theta that failure is SILENT rather than
    loud: post_deployment_analysis.process_inference_func_args substitutes theta only at the
    position whose index matches, so an index matching no position leaves every parameter
    filled from an analysis_df column, and inference is differentiated with respect to a theta
    that was never inserted.
    """
    unsupplied_indices = {
        name: index for name, index in indices_by_name.items() if index < 0
    }
    assert not unsupplied_indices, (
        f"These {func_description} argument indices are required but were not supplied (a "
        f"negative index means absent): {unsupplied_indices}. Please see the contract for "
        "details."
    )


def require_arg_indices_in_range(indices_by_name, arg_tuple_length, func_description):
    """
    Every SUPPLIED (non-negative) index must address a real position in the argument tuples.

    Negative indices mark the genuinely optional positions -- action probabilities, their
    times, previous betas -- as absent, and are skipped. An out-of-range index otherwise
    surfaces as a bare IndexError from whichever check or precompute step reaches it first,
    naming neither the parameter nor the index that was wrong.
    """
    out_of_range_indices = {
        name: index
        for name, index in indices_by_name.items()
        if index >= arg_tuple_length
    }
    assert not out_of_range_indices, (
        f"These {func_description} argument indices do not address a position in argument "
        f"tuples of length {arg_tuple_length} (valid positions are 0 through "
        f"{arg_tuple_length - 1}): {out_of_range_indices}. Please see the contract for details."
    )


def require_arg_indices_distinct(indices_by_name, func_description):
    """
    No two SUPPLIED indices may name the same argument position.

    Only the action-prob/action-prob-times pair was checked for this before. A collision
    anywhere else means the threading code overwrites one shared parameter with another: with
    beta_index == action_prob_index, for instance, arg_threading_helpers.thread_update_func_args
    writes the reconstructed action probabilities over the beta position, so the beta the
    estimator is differentiating with respect to never reaches the update function at all.
    """
    names_by_index = collections.defaultdict(list)
    for name, index in indices_by_name.items():
        if index >= 0:
            names_by_index[index].append(name)
    colliding_names_by_index = {
        index: sorted(names)
        for index, names in names_by_index.items()
        if len(names) > 1
    }
    assert not colliding_names_by_index, (
        f"These {func_description} argument indices collide -- position -> the parameters that "
        f"claim it: {colliding_names_by_index}. Each supplied index must name a distinct "
        "position. Please see the contract for details."
    )


def require_mask_index_appends_after_supplied_args(
    mask_index, arg_tuple_length, func_description
):
    """
    The validity mask is always APPENDED as a new last argument, never inserted, so its index
    must be exactly the supplied tuple length.

    self_pad_ragged_args_and_build_mask enforces this too, but only once it reaches a non-blank
    shape group mid-precompute; up front it costs nothing.
    """
    if mask_index < 0:
        return
    assert mask_index == arg_tuple_length, (
        f"{func_description} mask index {mask_index} must equal the supplied argument tuple "
        f"length ({arg_tuple_length}): the validity mask is appended as a new last argument, "
        f"so it belongs at position {arg_tuple_length}. Please see the contract for details."
    )


def require_ragged_indices_valid(
    ragged_indices,
    mask_index,
    arg_tuple_length,
    shared_indices_by_name,
    func_description,
):
    """
    The self-padding positions must be non-empty when padding is requested, must address real
    supplied positions, and must not name a parameter that is SHARED across subjects.

    Self-padding repeats a position's last row up to the group's maximum length, so applying it
    to a shared parameter silently changes that parameter rather than merely adding padding
    rows: beta is a (beta_dim,) vector, so padding it appends copies of its last component and
    changes its dimension, and previous betas are shared across subjects the same way.
    self_pad_ragged_args_and_build_mask notices only in the case where the corrupted position
    happens to disagree in row count with the genuinely ragged ones.
    """
    if mask_index < 0:
        # Padding is off, and the stacking code ignores ragged_indices entirely in that case.
        return
    assert ragged_indices, (
        f"{func_description} requests mask padding (mask index {mask_index}) but supplied no "
        "ragged argument positions to pad. Please see the contract for details."
    )
    out_of_range_ragged_indices = sorted(
        index for index in ragged_indices if index < 0 or index >= arg_tuple_length
    )
    assert not out_of_range_ragged_indices, (
        f"These {func_description} ragged argument positions do not address a supplied "
        f"position in argument tuples of length {arg_tuple_length}: "
        f"{out_of_range_ragged_indices}. Please see the contract for details."
    )
    ragged_index_set = set(ragged_indices)
    shared_ragged_indices_by_name = {
        name: index
        for name, index in shared_indices_by_name.items()
        if index >= 0 and index in ragged_index_set
    }
    assert not shared_ragged_indices_by_name, (
        f"These {func_description} parameters are shared across subjects but were listed as "
        f"ragged (self-padding) positions: {shared_ragged_indices_by_name}. Padding repeats a "
        "position's last row, which would silently change these parameters instead of only "
        "adding padding rows. Please see the contract for details."
    )


def require_every_update_policy_has_at_least_one_nonblank_arg_tuple(
    analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
):
    """
    Every policy whose beta an algorithm update produced must have at least one subject with a
    non-blank argument tuple, because that tuple is the ONLY place the package reads that
    policy's beta from.

    helper_functions.collect_all_post_update_betas builds all_post_update_betas by walking the
    update policies in sorted order and appending the first non-blank subject's beta for each,
    while helper_functions.construct_beta_index_by_policy_num_map independently assigns each
    policy an index by its position in that same sorted order. A policy that is present as a
    key but blank for EVERY subject contributes no entry, so every later policy's beta silently
    shifts down one position. Verified on policies {2, 3, 4} with policy 2 blank: policy 2 then
    resolves to policy 3's beta, policy 3 to policy 4's, and policy 4 to policy 4's -- no
    error raised, and every policy wrong.

    require_alg_update_args_given_for_all_subjects_at_each_update compares only KEY sets, so a
    dict whose every value is an empty tuple satisfies it; and
    require_all_policy_numbers_..._present_in_alg_update_args only checks that the policy is a
    key. Neither catches this.
    """
    logger.info(
        "Checking that every update policy has at least one non-blank argument tuple."
    )
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonnegative_policy_nums = active_df.loc[
        active_df[policy_num_col_name] >= 0, policy_num_col_name
    ]
    if nonnegative_policy_nums.empty:
        return
    initial_policy_num = nonnegative_policy_nums.min()
    update_policy_nums = sorted(
        set(
            nonnegative_policy_nums[nonnegative_policy_nums > initial_policy_num]
            .unique()
            .tolist()
        )
    )

    all_blank_policy_nums = [
        policy_num
        for policy_num in update_policy_nums
        # A policy missing from alg_update_func_args entirely is
        # require_all_policy_numbers_..._present_in_alg_update_args' finding, not this one.
        if policy_num in alg_update_func_args
        and not any(alg_update_func_args[policy_num].values())
    ]
    assert not all_blank_policy_nums, (
        f"These update policies are present in alg_update_func_args but have a blank "
        f"(empty-tuple) argument tuple for every subject: {all_blank_policy_nums}. Each "
        "policy's beta is read from its first non-blank tuple, so a policy with none shifts "
        "every later policy's beta down one position instead of failing. Please see the "
        "contract for details."
    )


def require_recorded_action_prob_betas_match_update_betas_for_their_policy(
    analysis_df,
    active_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    policy_num_col_name,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
):
    """
    The beta recorded in action_prob_func_args at a decision time must be the beta recorded in
    alg_update_func_args by the update that produced the policy in force at that decision time.

    This is the action-probability function's missing original-vs-threaded equivalence check.
    arg_threading_helpers.thread_action_prob_func_args DISCARDS the beta recorded in
    action_prob_func_args and substitutes
    all_post_update_betas[beta_index_by_policy_num[policy_num]], which
    helper_functions.collect_all_post_update_betas harvests from alg_update_func_args. So when
    the two disagree, the estimator differentiates the action probabilities at betas the
    recorded probabilities did not come from, and nothing else notices:
    require_action_probabilities_in_analysis_df_can_be_reconstructed evaluates the ORIGINAL
    args, so it still passes. The algorithm and inference estimating functions each already
    have such a check (require_threaded_algorithm_estimating_function_args_equivalent and
    require_threaded_inference_estimating_function_args_equivalent); the action probability
    function, the one whose beta is actually replaced, had none.

    Only non-initial, non-fallback policies are compared, matching exactly what the threading
    code substitutes: the initial policy's beta was produced by no update and is passed through
    untouched, and fallback (negative) policies are expected to carry blank args. The initial
    policy number is derived the way helper_functions.construct_beta_index_by_policy_num_map
    derives it -- the minimum non-negative policy number among active rows, which is NOT
    necessarily 0.

    The representative update beta for a policy is the first non-blank subject's, mirroring
    collect_all_post_update_betas' own rule;
    require_betas_match_in_alg_update_args_each_update, which runs before this, is what makes
    that representative well-defined.
    """
    logger.info(
        "Checking that the beta recorded for each decision time's action probabilities matches "
        "the beta recorded by the update that produced that decision time's policy."
    )
    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonnegative_policy_nums = active_df.loc[
        active_df[policy_num_col_name] >= 0, policy_num_col_name
    ]
    if nonnegative_policy_nums.empty:
        return
    initial_policy_num = nonnegative_policy_nums.min()

    # Built from numpy columns rather than a per-row iteration, and keyed exactly the way
    # require_action_probabilities_in_analysis_df_can_be_reconstructed keys its own lookup.
    policy_num_by_key = dict(
        zip(
            zip(
                active_df[calendar_t_col_name].to_numpy(),
                active_df[subject_id_col_name].to_numpy(),
                strict=False,
            ),
            active_df[policy_num_col_name].to_numpy(),
            strict=False,
        )
    )

    update_beta_by_policy_num = {}
    for policy_num, args_by_subject_id in alg_update_func_args.items():
        for args in args_by_subject_id.values():
            if args:
                update_beta_by_policy_num[policy_num] = np.asarray(
                    args[alg_update_func_args_beta_index]
                )
                break

    mismatches = []
    for decision_time, args_by_subject_id in action_prob_func_args.items():
        for subject_id, args in args_by_subject_id.items():
            if not args:
                continue
            policy_num = policy_num_by_key.get((decision_time, subject_id))
            # Three cases are deliberately skipped rather than reported here, because each is
            # some other check's finding and each has a purpose-built message for it: a
            # non-blank tuple at a cell analysis_df does not mark active belongs to
            # require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times, and
            # a fallback or initial policy has no update-produced beta to compare against by
            # design.
            if policy_num is None or policy_num < 0 or policy_num == initial_policy_num:
                continue
            if policy_num not in update_beta_by_policy_num:
                # No update beta resolved for this policy: either it is absent from
                # alg_update_func_args (require_all_policy_numbers_..._present_in_alg_update_args)
                # or it is present but blank for every subject
                # (require_every_update_policy_has_at_least_one_nonblank_arg_tuple, which runs
                # before this and is what makes this skip safe rather than a silent hole).
                continue

            recorded_beta = np.asarray(args[action_prob_func_args_beta_index])
            update_beta = update_beta_by_policy_num[policy_num]
            # Shapes FIRST: numpy would broadcast e.g. a (1,) beta against a (4,) one and find
            # them equal, so a dimension mismatch has to be caught before any value comparison.
            if recorded_beta.shape != update_beta.shape:
                mismatches.append(
                    (
                        decision_time,
                        subject_id,
                        policy_num,
                        f"shape {recorded_beta.shape} vs {update_beta.shape}",
                    )
                )
                continue
            if np.array_equal(recorded_beta, update_beta):
                continue
            # The tolerance absorbs float32-vs-float64 STORAGE differences of what is meant to
            # be the same recorded value (jnp arrays are float32 here by default) and nothing
            # more. The wiring errors this check exists to catch -- an off-by-one in policy
            # numbering, or the pre-update beta recorded where the post-update one belongs --
            # differ by many orders of magnitude more than this.
            if np.allclose(recorded_beta, update_beta, rtol=1e-6, atol=1e-12):
                continue
            with np.errstate(invalid="ignore"):
                max_abs_difference = float(
                    np.max(
                        np.abs(
                            recorded_beta.astype(np.float64)
                            - update_beta.astype(np.float64)
                        )
                    )
                )
            mismatches.append(
                (
                    decision_time,
                    subject_id,
                    policy_num,
                    f"max |difference| {max_abs_difference:.8g}",
                )
            )

    assert not mismatches, (
        f"The beta recorded in action_prob_func_args disagrees with the beta recorded in "
        f"alg_update_func_args for the policy in force, at {len(mismatches)} "
        f"(decision_time, subject_id) cell(s). The computation THREADS IN the update's beta "
        f"and discards the recorded one, so the action probabilities would be differentiated "
        f"at betas the recorded probabilities did not come from. Up to 5 shown as "
        f"(decision_time, subject_id, policy_num, disagreement): "
        f"{sorted(mismatches, key=str)[:5]}. Please see the contract for details."
    )


def verify_analysis_df_summary_satisfactory(
    analysis_df,
    subject_id_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    active_col_name,
    action_prob_col_name,
    reward_col_name,
    beta_dim,
    theta_dim,
    suppress_interactive_data_checks,
):

    active_df = analysis_df[analysis_df[active_col_name] == 1]
    num_subjects = active_df[subject_id_col_name].nunique()
    num_non_initial_or_fallback_policies = active_df[
        active_df[policy_num_col_name] > 0
    ][policy_num_col_name].nunique()
    num_decision_times_with_fallback_policies = len(
        active_df[active_df[policy_num_col_name] < 0]
    )
    num_decision_times = active_df[calendar_t_col_name].nunique()
    avg_decisions_per_subject = len(active_df) / num_subjects
    num_decision_times_with_multiple_policies = (
        active_df[active_df[policy_num_col_name] >= 0]
        .groupby(calendar_t_col_name)[policy_num_col_name]
        .nunique()
        > 1
    ).sum()
    min_action_prob = active_df[action_prob_col_name].min()
    max_action_prob = active_df[action_prob_col_name].max()
    min_non_fallback_policy_num = active_df[active_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ].min()
    num_data_points_before_first_update = len(
        active_df[active_df[policy_num_col_name] == min_non_fallback_policy_num]
    )

    median_action_probabilities = (
        active_df.groupby(calendar_t_col_name)[action_prob_col_name].median().to_numpy()
    )
    quartiles = active_df.groupby(calendar_t_col_name)[action_prob_col_name].quantile(
        [0.25, 0.75]
    )
    q25_action_probabilities = quartiles.xs(0.25, level=1).to_numpy()
    q75_action_probabilities = quartiles.xs(0.75, level=1).to_numpy()

    avg_rewards = active_df.groupby(calendar_t_col_name)[reward_col_name].mean()

    # Plot action probability quartile trajectories
    plt.clear_figure()
    plt.title("Action 1 Probability 25/50/75 Quantile Trajectories")
    plt.xlabel("Decision Time")
    plt.ylabel("Action 1 Probability Quantiles")
    plt.error(
        median_action_probabilities,
        yerr=q75_action_probabilities - q25_action_probabilities,
        color="blue+",
    )
    plt.grid(True)
    plt.xticks(
        range(
            0,
            len(median_action_probabilities),
            max(1, len(median_action_probabilities) // 10),
        )
    )
    action_prob_trajectory_plot = plt.build()

    # Plot avg reward trajectory
    plt.clear_figure()
    plt.title("Avg Reward Trajectory")
    plt.xlabel("Decision Time")
    plt.ylabel("Avg Reward")
    plt.scatter(avg_rewards, color="blue+", marker="*")
    plt.grid(True)
    plt.xticks(
        range(
            0,
            len(avg_rewards),
            max(1, len(avg_rewards) // 10),
        )
    )
    avg_reward_trajectory_plot = plt.build()

    confirm_input_check_result(
        f"\nYou provided an analysis DataFrame reflecting a study with"
        f"\n* {num_subjects} subjects"
        f"\n* {num_non_initial_or_fallback_policies} policy updates"
        f"\n* {num_decision_times} decision times, for an average of {avg_decisions_per_subject}"
        f" decisions per subject"
        f"\n* RL parameters of dimension {beta_dim} per update"
        f"\n* Inferential target of dimension {theta_dim}"
        f"\n* {num_data_points_before_first_update} data points before the first update"
        f"\n* {num_decision_times_with_fallback_policies} decision times"
        f" ({num_decision_times_with_fallback_policies * 100 / num_decision_times}%) for which"
        f" fallback policies were used"
        f"\n* {num_decision_times_with_multiple_policies} decision times"
        f" ({num_decision_times_with_multiple_policies * 100 / num_decision_times}%)"
        f" for which multiple non-fallback policies were used"
        f"\n* Minimum action probability {min_action_prob}"
        f"\n* Maximum action probability {max_action_prob}"
        f"\n* The following trajectories of action probability quartiles over time:\n {action_prob_trajectory_plot}"
        f"\n* The following average reward trajectory over time:\n {avg_reward_trajectory_plot}"
        f" \n\nDoes this meet expectations? (y/n)\n",
        suppress_interactive_data_checks,
    )


def require_betas_match_in_alg_update_args_each_update(
    alg_update_func_args, alg_update_func_args_beta_index
):
    logger.info(
        "Checking that betas match across subjects for each update in the algorithm update function args."
    )
    for policy_num in alg_update_func_args:
        first_beta = None
        for subject_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][subject_id]:
                continue
            beta = alg_update_func_args[policy_num][subject_id][
                alg_update_func_args_beta_index
            ]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(beta, first_beta), (
                    f"Betas do not match across subjects in the algorithm update function args for policy number {policy_num}. Please see the contract for details."
                )


def require_previous_betas_match_in_alg_update_args_each_update(
    alg_update_func_args, alg_update_func_args_previous_betas_index
):
    logger.info(
        "Checking that previous betas match across subjects for each update in the algorithm update function args."
    )
    if alg_update_func_args_previous_betas_index < 0:
        return

    for policy_num in alg_update_func_args:
        first_previous_betas = None
        for subject_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][subject_id]:
                continue
            previous_betas = alg_update_func_args[policy_num][subject_id][
                alg_update_func_args_previous_betas_index
            ]
            if first_previous_betas is None:
                first_previous_betas = previous_betas
            else:
                assert np.array_equal(previous_betas, first_previous_betas), (
                    f"Previous betas do not match across subjects in the algorithm update function args for policy number {policy_num}. Please see the contract for details."
                )


def require_valid_action_prob_times_given_if_index_supplied(
    analysis_df,
    calendar_t_col_name,
    alg_update_func_args,
    alg_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob times are valid if index is supplied.")

    if alg_update_func_args_action_prob_times_index < 0:
        return

    min_time = analysis_df[calendar_t_col_name].min()
    max_time = analysis_df[calendar_t_col_name].max()
    for policy_idx, args_by_subject in alg_update_func_args.items():
        for subject_id, args in args_by_subject.items():
            if not args:
                continue
            times = args[alg_update_func_args_action_prob_times_index]
            assert all(times[i] > times[i - 1] for i in range(1, len(times))), (
                f"Non-strictly-increasing times were given for action probabilities in the algorithm update function args for subject {subject_id} and policy {policy_idx}. Please see the contract for details."
            )
            assert times[0] >= min_time and times[-1] <= max_time, (
                f"Times not present in the study were given for action probabilities in the algorithm update function args. The min and max times in the analysis DataFrame are {min_time} and {max_time}, while subject {subject_id} has times {times} supplied for policy {policy_idx}. Please see the contract for details."
            )


def require_estimating_functions_sum_to_zero_se_standardized(
    per_subject_estimating_function_stacks: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    suppress_interactive_data_checks: bool,
    *,
    soft_tolerance_se: float = 0.01,
    hard_tolerance_se: float = 0.1,
    relative_noise_floor: float = 100 * float(np.finfo(np.float32).eps),
):
    """
    SE-standardized replacement for require_estimating_functions_sum_to_zero. Same underlying
    question -- do the recorded parameters actually root the claimed update/inference equations
    on the realized data (a value-level test of the stacked model of the algorithm) -- but each
    component of the residual is measured against its own standard error, taken directly from
    the per-subject values that were averaged:

        r_j = mean_i psi_ij               (the residual: the quantity that must be ~0)
        s_j = sqrt(mean_i psi_ij^2)       (RMS size of the terms that had to cancel)
        a_j = |r_j| / (s_j / sqrt(n))     (the residual in units of its own SE)

    Why not the raw absolute tolerance: the raw residual carries the estimating equations' own
    (reward-scale) units, so any fixed atol is non-portable -- measured on real wave-2 ADS-142
    runs, the raw max residual grew ~2e-5 -> ~2e-4 -> ~5e-3 as reward noise variance went
    1 -> 10 -> 100, crossing the legacy 5e-4 gate on healthy runs, while s_j grows in exact
    proportion and keeps a_j flat. See docs/adr/0002, corrections 2026-08-31 and 2026-09-02.

    Why not the displacement form (a_j = |(B^-1 r)_j| / SE_j, this function's first
    incarnation): routing through the bread and sandwich made the check inherit their
    degeneracies -- masked entirely when a meat-driven sandwich blow-up inflated the SEs
    (the U5/B_influence regime), a silent pass when every SE collapsed to exactly zero, an
    astronomical false "does not sum to zero" on a denormal SE, and an abort on a singular
    bread. This form touches neither matrix, so none of those regimes exist for it, and it is
    self-limiting where the old one was unbounded: by Cauchy-Schwarz a_j <= sqrt(n).

    Components whose RMS sits at or below the stack's numerical noise floor
    (s_j <= relative_noise_floor * max_k s_k) are trivially rooted and reported as a_j = 0
    rather than judged: for such a component the statistic would compare float noise to float
    noise and read O(sqrt(n)) by construction, not by any failure. Observed concretely on the
    smoke fixture: an update-1 component whose only nonzero per-subject values were two
    identical 2-ulp float32 rounding residues (2.4e-7) read exactly sqrt(2) SE. The skip can
    never mask a real failure, because Cauchy-Schwarz also gives |r_j| <= s_j: anything skipped
    has a residual itself below the noise floor (s_j == 0 is the exact special case -- it
    forces r_j == 0). The floor's default is ~1.2e-5 RELATIVE to the stack's largest component
    (assumed to carry real data variation), so it is portable across reward scales; it only
    needs to dominate float32 evaluation noise, a hardware constant, not a data constant --
    under enable_x64 it is merely conservative.

    Known residual limitation: a single heavy-tailed subject dominating s_j dilutes a bug
    confined to the typical subjects. The displacement form had the identical weakness (the
    same subject dominates the meat), and that regime belongs to the diagnostic suite's
    influence and solver-health checks, so it is deliberately not patched here.

    Raises on a_max > hard_tolerance_se; interactively confirms on a_max > soft_tolerance_se.
    """
    logger.info(
        "Checking that estimating functions average to zero across subjects "
        "(SE-standardized residual form)"
    )
    stacks = np.asarray(per_subject_estimating_function_stacks, dtype=np.float64)
    assert stacks.ndim == 2
    num_subjects, dim = stacks.shape
    assert (dim - theta_dim) % beta_dim == 0
    num_updates = (dim - theta_dim) // beta_dim

    if not np.all(np.isfinite(stacks)):
        # Without this guard a nonfinite subject poisons r and s into nan, `s > 0` reads
        # False everywhere it matters, and the check would silently pass with a_max = 0.
        raise AssertionError(
            "Per-subject estimating function stacks contain nonfinite values, so the "
            "sum-to-zero question cannot be evaluated. This points at the estimating "
            "functions or their recorded arguments, not at this check."
        )

    r = stacks.mean(axis=0)
    s = np.sqrt(np.mean(stacks**2, axis=0))
    a = np.zeros(dim)
    # s_j / sqrt(n) is the SE of the mean estimating function itself (uncentered second
    # moment; centering is immaterial anywhere near a pass, where r ~ 0). Components at or
    # below the stack's noise floor resolve to a_j = 0 -- provably benign, since |r_j| <= s_j
    # puts their residual below the floor too (see the docstring).
    noise_floor = relative_noise_floor * np.max(s)
    judged = s > noise_floor
    np.divide(np.abs(r), s / np.sqrt(num_subjects), out=a, where=judged)
    if not np.all(judged):
        logger.info(
            "%d of %d stacked components sit at or below the stack's numerical noise floor "
            "(RMS <= %.3g) and are trivially rooted: their residuals are themselves bounded "
            "by the floor. Reported as a residual of 0 SE.",
            int((~judged).sum()),
            dim,
            noise_floor,
        )

    # The offending component is named in the failure text itself -- the per-block breakdown
    # below prints one line per block whatever fails, so it cannot attribute anything on its
    # own.
    worst_index = int(np.argmax(a))
    a_max = float(a[worst_index])
    worst_block = (
        num_updates
        if worst_index >= num_updates * beta_dim
        else worst_index // beta_dim
    )
    worst_label = (
        f"inference component {worst_index - num_updates * beta_dim}"
        if worst_block == num_updates
        else f"update {worst_block + 1} component {worst_index % beta_dim}"
    )

    def _block_breakdown() -> str:
        lines = []
        for i in range(num_updates):
            marker = "   <-- largest" if i == worst_block else ""
            lines.append(
                f"update {i + 1}: max residual "
                f"{np.max(a[i * beta_dim : (i + 1) * beta_dim]):.4g} SE{marker}"
            )
        marker = "   <-- largest" if worst_block == num_updates else ""
        lines.append(f"inference: max residual {np.max(a[-theta_dim:]):.4g} SE{marker}")
        return "\n".join(lines)

    if a_max > hard_tolerance_se:
        breakdown = _block_breakdown()
        logger.info(
            "Estimating-function residual exceeds the hard tolerance. "
            "Per-block breakdown:\n%s",
            breakdown,
        )
        raise AssertionError(
            f"Estimating functions do not sum to zero: the residual for {worst_label} "
            f"is {a_max:.4g} of its own standard error "
            f"(hard tolerance {hard_tolerance_se}). Per-block breakdown:\n{breakdown}"
        )
    if a_max > soft_tolerance_se:
        breakdown = _block_breakdown()
        logger.info(
            "Estimating-function residual exceeds the soft tolerance. "
            "Per-block breakdown:\n%s",
            breakdown,
        )
        # The breakdown is repeated in the prompt rather than pointed at: this package
        # configures no logging handler, so the logger.info above reaches the user only if they
        # set one up themselves.
        confirm_input_check_result(
            f"\nEstimating functions do not average to within tolerance of zero: the residual "
            f"for {worst_label} is {a_max:.4g} of its own standard error "
            f"(soft tolerance {soft_tolerance_se}). Please decide if this is a reasonable "
            f"result given the per-block breakdown:\n{breakdown}\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
        )
    logger.info(
        "Estimating functions sum to zero within tolerance (max SE-standardized "
        "residual %.4g).",
        a_max,
    )
    # Returned, not just logged, so analyze_dataset can record this outcome as a row in the
    # diagnostic summary. Returning None (as this did until 2026-09-02) left the check's result
    # visible only in a log line, which this package configures no handler for. Callers that
    # ignore the return value are unaffected.
    return {
        "max_residual_se": a_max,
        "worst_label": worst_label,
        "soft_tolerance_se": soft_tolerance_se,
        "hard_tolerance_se": hard_tolerance_se,
    }


def require_estimating_functions_sum_to_zero(
    mean_estimating_function_stack: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    suppress_interactive_data_checks: bool,
):
    """
    SUPERSEDED by require_estimating_functions_sum_to_zero_se_standardized (kept for backward
    compatibility): this version compares the raw residual against fixed absolute tolerances,
    which carry the estimating equations' own reward-scale units and are therefore non-portable
    across deployments -- empirically documented to false-alarm on healthy high-noise runs
    (docs/adr/0002).

    This is a test that the correct loss/estimating functions have
    been given for both the algorithm updates and inference. If that is true, then the
    loss/estimating functions when evaluated should sum to approximately zero across subjects.  These
    values have been stacked and averaged across subjects in mean_estimating_function_stack, which
    we simply compare to the zero vector.  We can isolate components for each update and inference
    by considering the dimensions of the beta vectors and the theta vector.

    Inputs:
    mean_estimating_function_stack:
        The mean of the estimating function stack (a component for each algorithm update and
        inference) across subjects. This should be a 1D array.
    beta_dim:
        The dimension of the beta vectors that parameterize the algorithm.
    theta_dim:
        The dimension of the theta vector that we estimate during after-study analysis.

    Returns:
    None
    """

    logger.info("Checking that estimating functions average to zero across subjects")

    # Have a looser hard failure cutoff before the typical interactive check
    try:
        np.testing.assert_allclose(
            mean_estimating_function_stack,
            jnp.zeros(mean_estimating_function_stack.size),
            atol=1e-2,
        )
    except AssertionError as e:
        logger.info(
            "Estimating function stacks do not average to within loose tolerance of zero across subjects.  Drilling in to specific updates and inference component."
        )
        # If this is not true there is an internal problem in the package.
        assert (mean_estimating_function_stack.size - theta_dim) % beta_dim == 0
        num_updates = (mean_estimating_function_stack.size - theta_dim) // beta_dim
        for i in range(num_updates):
            logger.info(
                "Mean estimating function contribution for update %s:\n%s",
                i + 1,
                mean_estimating_function_stack[i * beta_dim : (i + 1) * beta_dim],
            )
        logger.info(
            "Mean estimating function contribution for inference:\n%s",
            mean_estimating_function_stack[-theta_dim:],
        )

        raise e

    logger.info(
        "Estimating functions pass loose tolerance check, proceeding to tighter check."
    )
    try:
        np.testing.assert_allclose(
            mean_estimating_function_stack,
            jnp.zeros(mean_estimating_function_stack.size),
            atol=5e-4,
        )
    except AssertionError as e:
        logger.info(
            "Estimating function stacks do not average to within specified tolerance of zero across subjects.  Drilling in to specific updates and inference component."
        )
        # If this is not true there is an internal problem in the package.
        assert (mean_estimating_function_stack.size - theta_dim) % beta_dim == 0
        num_updates = (mean_estimating_function_stack.size - theta_dim) // beta_dim
        for i in range(num_updates):
            logger.info(
                "Mean estimating function contribution for update %s:\n%s",
                i + 1,
                mean_estimating_function_stack[i * beta_dim : (i + 1) * beta_dim],
            )
        logger.info(
            "Mean estimating function contribution for inference:\n%s",
            mean_estimating_function_stack[-theta_dim:],
        )
        confirm_input_check_result(
            f"\nEstimating functions do not average to within default tolerance of zero vector. Please decide if the following is a reasonable result, taking into account the above breakdown by update number and inference. If not, there are several possible reasons for failure mentioned in the contract. Results:\n{str(e)}\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
            e,
        )


# Shared default for the relative_floor parameters below: 100 ulps of the component's own scale
# in float32 (~1.19e-5), the precision every estimating-function evaluation in this package runs
# at. The previous 1e-6 left only ~8 float32 ulps of slack between two independently ordered
# evaluations of the same cancelling sum over O(100) decision times, and a healthy seed-0 local
# oralytics run tripped it (2026-09-02): one near-fully-cancelling component (update 31,
# component 134) differed by 1.7e-6 of its own scale (~14 ulps) while the other 8,774 compared
# values -- and the direct recorded-vs-reconstructed action probability check -- agreed to
# machine precision. 100 ulps gives ~7x headroom over that observed healthy noise while staying
# ~100x below the tightest rtol these comparisons use (1e-3) measured at each component's full
# scale.
ORIGINAL_VS_THREADED_RELATIVE_FLOOR = float(100 * np.finfo(np.float32).eps)


def componentwise_absolute_tolerance(
    reference: np.ndarray, relative_floor: float = ORIGINAL_VS_THREADED_RELATIVE_FLOOR
) -> np.ndarray:
    """
    Per-component absolute-tolerance floor for comparing two floating-point computations of the
    SAME batched quantity. Returns an array shaped to broadcast against `reference`, whose FIRST
    axis is the subject/batch axis (as produced by every jax.vmap call in this package).

    Replaces the scalar floor this package used until 2026-08-31 (relative_floor * max|reference|
    over the whole array; deleted with its call sites). The dynamic range that matters here is the
    one ACROSS components, which carry different units -- an intercept score, a covariate-weighted
    score and a reward-scaled score are simply not the same size -- whereas subjects share a
    component's units. Under a single global floor, one large component sets the absolute slack
    for every small one: at a realistic 1e4-vs-0.5 spread the large component hands the O(1) one
    an absolute slack 2e4x its own floor (atol ~0.12 at the default relative_floor), silently
    swallowing reconstruction inconsistencies of tens of percent there. Reducing over the subject
    axis only keeps the scale-awareness that fixes the reward-unit fixed-atol bug while confining
    each component's slack to its own scale.

    The default relative_floor is ORIGINAL_VS_THREADED_RELATIVE_FLOOR, 100 float32 ulps of the
    component's scale -- see its comment above for the calibration story (an earlier 1e-6
    hard-failed a healthy run on ~14 ulps of cancellation noise).

    Two degenerate tiers, in order:
      - a component that is exactly zero for every subject has no scale of its own, so it borrows
        the SMALLEST nonzero component scale in the array. Borrowing the array's overall scale
        instead would reintroduce the very cross-component leak this function exists to close: at
        a 1e4-vs-0.5 spread an all-zero component would inherit the large component's slack
        (atol ~0.12 at the default relative_floor), so a mismatch anywhere in the ~2e4-wide band
        between the two scales' floors would pass -- looser than the fixed 1e-7 this replaced.
        The smallest observed scale is the most conservative stand-in that is still nonzero;
      - an entirely zero reference has no observable scale at all, so the floor falls back to a
        unit scale, i.e. relative_floor itself. Returning 0.0 there (as the scalar helper does)
        would demand BIT-EXACT agreement between two float32 computations of a cancelling sum --
        precisely the zero-true-value fragility an atol floor exists to prevent. 0.0 is reserved
        for the empty reference, where there is nothing to compare in the first place.
    """
    magnitudes = np.abs(np.asarray(reference, dtype=np.float64))
    if magnitudes.size == 0:
        return np.zeros_like(magnitudes)
    if magnitudes.ndim == 0:
        magnitudes = magnitudes.reshape(1)
    # Nonfinite entries must not poison the scale into nan/inf and reject every component; the
    # comparison itself handles them (matching nans/infs agree, mismatched ones fail).
    magnitudes = np.where(np.isfinite(magnitudes), magnitudes, 0.0)
    component_scale = np.max(magnitudes, axis=0, keepdims=True)
    nonzero_scales = component_scale[component_scale > 0.0]
    fallback_scale = float(np.min(nonzero_scales)) if nonzero_scales.size else 1.0
    return relative_floor * np.where(
        component_scale > 0.0, component_scale, fallback_scale
    )


def require_original_and_threaded_results_agree(
    original,
    threaded,
    *,
    rtol: float,
    context: str,
    relative_floor: float = ORIGINAL_VS_THREADED_RELATIVE_FLOOR,
) -> None:
    """
    Assert that an estimating function's outputs on the ORIGINAL, recorded arguments and on the
    THREADED ones (the shared betas/theta and RECONSTRUCTED action probabilities the real
    computation substitutes in) agree component by component; a disagreement means the recorded
    action probabilities and the ones the supplied action-prob function reproduces are not the
    same data.

    One definition shared by every original-vs-threaded comparison in the package -- the two
    require_threaded_*_equivalent checks below and their batched counterparts in
    batched_weighted_estimating_function_stack -- so the tolerance rule cannot drift between them
    (it has drifted once already; see that module's
    _assert_original_and_threaded_bucket_results_agree). Only rtol legitimately differs per call
    site, so it is the one required tolerance argument.

    Replaces np.testing.assert_allclose because that accepts only a SCALAR atol, which cannot
    express componentwise_absolute_tolerance's per-component floor. The accept/reject rule is
    otherwise assert_allclose's own, |original - threaded| <= atol + rtol * |threaded|, and like
    assert_allclose it treats matching nans and matching signed infinities as agreement while
    REJECTING mismatched ones (+inf vs -inf, and inf vs any finite value). The tolerance branch is
    gated on both sides being finite for that reason: rtol * |threaded| is itself infinite at an
    infinite threaded component, so an ungated `difference <= allowed` would accept every possible
    original there -- silently blessing exactly the blow-up this check exists to catch.
    """
    original_np = np.atleast_1d(np.asarray(original, dtype=np.float64))
    threaded_np = np.atleast_1d(np.asarray(threaded, dtype=np.float64))
    if original_np.shape != threaded_np.shape:
        raise AssertionError(
            f"{context}\nOriginal and threaded estimating-function outputs have different "
            f"shapes: {original_np.shape} vs {threaded_np.shape}."
        )

    atol = componentwise_absolute_tolerance(original_np, relative_floor)
    allowed = atol + rtol * np.abs(threaded_np)
    # inf - inf is nan, which the agreement rule below handles explicitly; the numpy warning it
    # would otherwise raise here says nothing this check doesn't already report.
    with np.errstate(invalid="ignore"):
        difference = np.abs(original_np - threaded_np)
    both_finite = np.isfinite(original_np) & np.isfinite(threaded_np)
    agrees = (
        (both_finite & (difference <= allowed))
        | (original_np == threaded_np)
        | (np.isnan(original_np) & np.isnan(threaded_np))
    )
    if np.all(agrees):
        return

    # Rank the failures by how far past their OWN tolerance they are rather than by raw
    # difference: tolerances now vary by component, so the largest absolute gap is not
    # necessarily the most damning one.
    # inf/inf is nan here (a nonfinite mismatch, whose tolerance is itself infinite); it is mapped
    # to inf just below so those rank as the worst offenders rather than the mildest.
    with np.errstate(invalid="ignore"):
        excess = np.divide(
            difference,
            allowed,
            out=np.full(difference.shape, np.inf),
            where=allowed > 0.0,
        )
    excess = np.where(np.isnan(excess), np.inf, excess)
    disagree_count = int(np.count_nonzero(~agrees))
    ranked = np.argsort(np.where(agrees, -np.inf, excess), axis=None)[::-1]
    max_offenders_shown = 5
    atol_broadcast = np.broadcast_to(atol, difference.shape)
    offender_lines = []
    for flat_index in ranked[: min(max_offenders_shown, disagree_count)]:
        index = np.unravel_index(int(flat_index), difference.shape)
        offender_lines.append(
            f"  index {tuple(int(i) for i in index)}: original {original_np[index]:.8g}, "
            f"threaded {threaded_np[index]:.8g}, |difference| {difference[index]:.8g}, "
            f"allowed {allowed[index]:.8g} (atol {atol_broadcast[index]:.8g} + "
            f"rtol*|threaded| {rtol * abs(threaded_np[index]):.8g}); "
            f"{excess[index]:.3g}x allowed"
        )
    if disagree_count > max_offenders_shown:
        offender_lines.append(f"  ... and {disagree_count - max_offenders_shown} more.")
    # threshold/edgeitems are passed explicitly: the module-level
    # np.set_printoptions(threshold=np.inf) above would otherwise print every value of both
    # arrays -- thousands of terminal lines on a realistically sized bucket (observed: ~4,400
    # at 65 subjects x 135 components), burying the offender listing that actually localizes
    # the failure.
    raise AssertionError(
        f"{context}\n"
        f"Substituting the reconstructed action probabilities changed the estimating "
        f"function's output: {disagree_count} of {difference.size} values "
        f"disagree beyond tolerance (rtol={rtol} against |threaded|, plus a per-component "
        f"absolute floor of {relative_floor:.3g} x max|original| over the subject axis).\n"
        f"Worst offenders, ranked by multiple of their own tolerance (first index is the "
        f"subject's position in this group):\n" + "\n".join(offender_lines) + "\n"
        f"Max absolute difference: {np.max(difference):.8g}.\n"
        f"original (summarized):\n"
        f"{np.array2string(original_np, threshold=100, edgeitems=2)}\n"
        f"threaded (summarized):\n"
        f"{np.array2string(threaded_np, threshold=100, edgeitems=2)}"
    )


def require_threaded_algorithm_estimating_function_args_equivalent(
    algorithm_estimating_func,
    update_func_args_by_by_subject_id_by_policy_num,
    threaded_update_func_args_by_policy_num_by_subject_id,
    suppress_interactive_data_checks,
):
    """
    Check that the algorithm estimating function returns the same values
    when called with the original arguments and when called with the
    reconstructed action probabilities substituted in.
    """
    for (
        policy_num,
        update_func_args_by_subject_id,
    ) in update_func_args_by_by_subject_id_by_policy_num.items():
        nontrivial_args_by_subject_id = {
            subject_id: args
            for subject_id, args in update_func_args_by_subject_id.items()
            if args
        }
        if not nontrivial_args_by_subject_id:
            continue
        for shape_group in group_user_args_by_shape(nontrivial_args_by_subject_id):
            group_subject_ids = sorted(shape_group.keys())

            unthreaded_batched_arg_tensors, batch_axes = batch_args_by_subject(
                group_subject_ids, shape_group
            )

            threaded_shape_group = {
                subject_id: threaded_update_func_args_by_policy_num_by_subject_id[
                    subject_id
                ][policy_num]
                for subject_id in group_subject_ids
            }
            threaded_batched_arg_tensors, _ = batch_args_by_subject(
                group_subject_ids, threaded_shape_group
            )

            unthreaded_result = jax.vmap(algorithm_estimating_func, in_axes=batch_axes)(
                *unthreaded_batched_arg_tensors
            )
            # Need to stop gradient here because we can't convert a traced value to np array
            threaded_result = jax.lax.stop_gradient(
                jax.vmap(algorithm_estimating_func, in_axes=batch_axes)(
                    *threaded_batched_arg_tensors
                )
            )

            # The atol floor is scaled to the compared values, per component (estimating-function
            # outputs carry the reward's units, so a fixed absolute tolerance false-alarms on
            # healthy high-reward data) -- see componentwise_absolute_tolerance's docstring.
            require_original_and_threaded_results_agree(
                np.asarray(unthreaded_result),
                np.asarray(threaded_result),
                rtol=1e-3,
                context=(
                    "Algorithm estimating function args are not equivalent after threading "
                    f"for policy number {policy_num}, subjects {group_subject_ids}."
                ),
            )


def require_threaded_inference_estimating_function_args_equivalent(
    inference_estimating_func,
    inference_func_args_by_subject_id,
    threaded_inference_func_args_by_subject_id,
    suppress_interactive_data_checks,
):
    """
    Check that the inference estimating function returns the same values
    when called with the original arguments and when called with the
    reconstructed action probabilities substituted in.
    """
    nontrivial_args_by_subject_id = {
        subject_id: args
        for subject_id, args in inference_func_args_by_subject_id.items()
        if args
    }
    if not nontrivial_args_by_subject_id:
        return
    for shape_group in group_user_args_by_shape(nontrivial_args_by_subject_id):
        group_subject_ids = sorted(shape_group.keys())

        unthreaded_batched_arg_tensors, batch_axes = batch_args_by_subject(
            group_subject_ids, shape_group
        )

        threaded_shape_group = {
            subject_id: threaded_inference_func_args_by_subject_id[subject_id]
            for subject_id in group_subject_ids
        }
        threaded_batched_arg_tensors, _ = batch_args_by_subject(
            group_subject_ids, threaded_shape_group
        )

        unthreaded_result = jax.vmap(inference_estimating_func, in_axes=batch_axes)(
            *unthreaded_batched_arg_tensors
        )
        # Need to stop gradient here because we can't convert a traced value to np array
        threaded_result = jax.lax.stop_gradient(
            jax.vmap(inference_estimating_func, in_axes=batch_axes)(
                *threaded_batched_arg_tensors
            )
        )

        # The scale-aware atol floor also fixes this check's opposite latent fragility: with
        # no atol at all, a component whose true value is exactly zero fails any pure-rtol
        # comparison on any nonzero noise. See componentwise_absolute_tolerance's docstring.
        require_original_and_threaded_results_agree(
            np.asarray(unthreaded_result),
            np.asarray(threaded_result),
            rtol=1e-2,
            context=(
                "Inference estimating function args are not equivalent after threading for "
                f"subjects {group_subject_ids}."
            ),
        )
