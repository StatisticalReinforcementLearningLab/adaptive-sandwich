import collections
import logging
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
from .helper_functions import confirm_input_check_result
from .vmap_helpers import batch_args_by_subject, stack_batched_arg_lists_into_tensors

# When we print out objects for debugging, show the whole thing.
np.set_printoptions(threshold=np.inf)

logger = logging.getLogger(__name__)


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
):
    ### Validate algorithm loss/estimating function and args
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
    require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
    )
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
    require_action_probabilities_in_analysis_df_can_be_reconstructed(
        analysis_df,
        action_prob_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        active_col_name,
        action_prob_func_args,
        action_prob_func,
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
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )

    ### Validate analysis_df
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
    require_all_named_columns_present_in_analysis_df(
        analysis_df,
        active_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_col_name,
    )
    require_all_named_columns_not_object_type_in_analysis_df(
        analysis_df,
        active_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_col_name,
    )
    require_binary_actions(analysis_df, active_col_name, action_col_name)
    require_binary_active_indicators(analysis_df, active_col_name)
    require_consecutive_integer_policy_numbers(
        analysis_df, active_col_name, policy_num_col_name
    )
    require_consecutive_integer_calendar_times(analysis_df, calendar_t_col_name)
    require_hashable_subject_ids(analysis_df, active_col_name, subject_id_col_name)
    # Positivity/overlap (recorded probabilities strictly inside (0, 1)) is enforced by
    # lifejacket.diagnostics.check_exploration_and_weights instead -- see the removed
    # require_action_probabilities_in_range_0_to_1's history for why: some legitimate
    # near-deterministic policies legitimately produce recorded probabilities of exactly 0.0/1.0
    # after floating-point rounding, so a hard assertion here would have been a
    # backward-incompatible break to the always-on input-check path.

    ### Validate theta estimation
    require_theta_is_1D_array(theta_est)


def perform_alg_only_input_checks(
    analysis_df,
    active_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
    action_prob_func,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    alg_update_func_args_previous_betas_index,
    suppress_interactive_data_checks,
):
    ### Validate algorithm loss/estimating function and args
    require_alg_update_args_given_for_all_subjects_at_each_update(
        analysis_df, subject_id_col_name, alg_update_func_args
    )
    require_beta_is_1D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_beta_index
    )
    require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        analysis_df, active_col_name, policy_num_col_name, alg_update_func_args
    )
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
    require_action_probabilities_in_analysis_df_can_be_reconstructed(
        analysis_df,
        action_prob_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        active_col_name,
        action_prob_func_args,
        action_prob_func=action_prob_func,
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
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )


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
    np.testing.assert_allclose(
        np.asarray(actual_action_probs, dtype="float64"),
        np.asarray(reconstructed_action_probs, dtype="float64"),
        atol=1e-6,
    )


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


def require_all_named_columns_present_in_analysis_df(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
):
    logger.info(
        "Checking that all named columns are present in the analysis DataFrame."
    )
    assert active_col_name in analysis_df.columns, (
        f"{active_col_name} not in analysis DataFrame."
    )
    assert action_col_name in analysis_df.columns, (
        f"{action_col_name} not in analysis DataFrame."
    )
    assert policy_num_col_name in analysis_df.columns, (
        f"{policy_num_col_name} not in analysis DataFrame."
    )
    assert calendar_t_col_name in analysis_df.columns, (
        f"{calendar_t_col_name} not in analysis DataFrame."
    )
    assert subject_id_col_name in analysis_df.columns, (
        f"{subject_id_col_name} not in analysis DataFrame."
    )
    assert action_prob_col_name in analysis_df.columns, (
        f"{action_prob_col_name} not in analysis DataFrame."
    )


def require_all_named_columns_not_object_type_in_analysis_df(
    analysis_df,
    active_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    subject_id_col_name,
    action_prob_col_name,
):
    logger.info("Checking that all named columns are not type object.")
    for colname in (
        active_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_col_name,
    ):
        assert analysis_df[colname].dtype != "object", (
            f"At least {colname} is of object type in analysis DataFrame."
        )


def require_binary_actions(analysis_df, active_col_name, action_col_name):
    logger.info("Checking that actions are binary.")
    assert (
        analysis_df[analysis_df[active_col_name] == 1][action_col_name]
        .astype("int64")
        .isin([0, 1])
        .all()
    ), "Actions are not binary."


def require_binary_active_indicators(analysis_df, active_col_name):
    logger.info("Checking that active indicators are binary.")
    assert (
        analysis_df[analysis_df[active_col_name] == 1][active_col_name]
        .astype("int64")
        .isin([0, 1])
        .all()
    ), "In-study indicators are not binary."


def require_consecutive_integer_policy_numbers(
    analysis_df, active_col_name, policy_num_col_name
):
    # TODO: This is a somewhat rough check of this, could also check nondecreasing temporally

    logger.info(
        "Checking that in-study, non-fallback policy numbers are consecutive integers."
    )

    active_df = analysis_df[analysis_df[active_col_name] == 1]
    nonnegative_policy_df = active_df[active_df[policy_num_col_name] >= 0]
    # Ideally we actually have integers, but for legacy reasons we will support
    # floats as well.
    if nonnegative_policy_df[policy_num_col_name].dtype == "float64":
        nonnegative_policy_df[policy_num_col_name] = nonnegative_policy_df[
            policy_num_col_name
        ].astype("int64")
    assert np.array_equal(
        nonnegative_policy_df[policy_num_col_name].unique(),
        range(
            nonnegative_policy_df[policy_num_col_name].min(),
            nonnegative_policy_df[policy_num_col_name].max() + 1,
        ),
    ), "Policy numbers are not consecutive integers."


def require_consecutive_integer_calendar_times(analysis_df, calendar_t_col_name):
    # This is a somewhat rough check of this, more like checking there are no
    # gaps in the integers covered.  But we have other checks that all subjects
    # have same times, etc.
    # Note these times should be well-formed even when the subject is not in the study.
    logger.info("Checking that calendar times are consecutive integers.")
    assert np.array_equal(
        analysis_df[calendar_t_col_name].unique(),
        range(
            analysis_df[calendar_t_col_name].min(),
            analysis_df[calendar_t_col_name].max() + 1,
        ),
    ), "Calendar times are not consecutive integers."


def require_hashable_subject_ids(analysis_df, active_col_name, subject_id_col_name):
    logger.info("Checking that subject IDs are hashable.")
    isinstance(
        analysis_df[analysis_df[active_col_name] == 1][subject_id_col_name][0],
        collections.abc.Hashable,
    )


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
    # Get the number of the initial policy. 0 is recommended but not required.
    min_nonnegative_policy_number = active_df[active_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ]
    assert set(
        active_df[active_df[policy_num_col_name] > min_nonnegative_policy_number][
            policy_num_col_name
        ].unique()
    ).issubset(alg_update_func_args.keys()), (
        f"There are non-fallback, non-initial policy numbers in the analysis DataFrame that are not in the update function args: {set(active_df[active_df[policy_num_col_name] > 0][policy_num_col_name].unique()) - set(alg_update_func_args.keys())}. Please see the contract for details."
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


def require_betas_match_in_action_prob_func_args_each_decision(
    action_prob_func_args, action_prob_func_args_beta_index
):
    logger.info(
        "Checking that betas match across subjects for each decision time in the action prob args."
    )
    for decision_time in action_prob_func_args:
        first_beta = None
        for subject_id in action_prob_func_args[decision_time]:
            if not action_prob_func_args[decision_time][subject_id]:
                continue
            beta = action_prob_func_args[decision_time][subject_id][
                action_prob_func_args_beta_index
            ]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(beta, first_beta), (
                    f"Betas do not match across subjects in the action prob args for decision_time {decision_time}. Please see the contract for details."
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
    mean_estimating_function_stack: jnp.ndarray,
    joint_bread_matrix: np.ndarray,
    joint_sandwich_matrix: np.ndarray,
    beta_dim: int,
    theta_dim: int,
    suppress_interactive_data_checks: bool,
    *,
    soft_tolerance_se: float = 0.01,
    hard_tolerance_se: float = 0.1,
):
    """
    SE-standardized replacement for require_estimating_functions_sum_to_zero. Same underlying
    question -- do the recorded parameters actually root the claimed update/inference equations
    on the realized data (a value-level test of the stacked model of the algorithm) -- but the
    residual is measured in units that matter and that are portable across reward scales:

        displacement = B_hat^{-1} @ mean_stack        (how far the residual moves the estimates)
        a_j = |displacement_j| / SE(component_j)      (in units of each component's own SE)

    This is check_root_and_implementation's a_root construction extended from the theta targets
    to EVERY stacked component (each update's betas and inference's theta), so an update-block
    residual that barely moves theta still registers against that update's own beta SEs -- with
    per-update attribution preserved in the failure breakdown.

    Why not the raw absolute tolerance: the raw residual carries the estimating equations' own
    (reward-scale) units, so any fixed atol is non-portable. Measured on real wave-2 ADS-142
    runs, the raw max residual grew ~2e-5 -> ~2e-4 -> ~5e-3 as reward noise variance went
    1 -> 10 -> 100 (crossing the legacy 5e-4 gate and brushing its 1e-2 hard gate on healthy
    runs), while this statistic stayed flat at 4e-6..8e-5 across all three scales -- two-plus
    orders of magnitude below soft_tolerance_se. See docs/adr/0002, correction 2026-08-31.

    Components with (numerically) zero variance are excluded (a rank/identification problem for
    the diagnostic suite's bread-stability check to flag, not a sum-to-zero question), mirroring
    a_root's treatment.

    Inputs beyond the legacy check's: joint_bread_matrix (B_hat for the full stacked system) and
    joint_sandwich_matrix (B^-1 M B^-T / n, i.e. Cov(eta_hat) directly -- the same scaling
    convention as everywhere in lifejacket.diagnostics).

    Raises on a_max > hard_tolerance_se; interactively confirms on a_max > soft_tolerance_se.
    """
    logger.info(
        "Checking that estimating functions average to zero across subjects "
        "(SE-standardized displacement form)"
    )
    r = np.asarray(mean_estimating_function_stack, dtype=np.float64)
    B = np.asarray(joint_bread_matrix, dtype=np.float64)
    V = np.asarray(joint_sandwich_matrix, dtype=np.float64)
    assert (r.size - theta_dim) % beta_dim == 0
    num_updates = (r.size - theta_dim) // beta_dim

    singular_bread_message = (
        "Estimating-function residual displacement is unavailable -- the joint bread matrix "
        "is numerically singular or contains nonfinite values. See the diagnostic suite's "
        "root_and_implementation/bread_stability checks; the sum-to-zero question cannot "
        "be evaluated on this system."
    )
    try:
        displacement = np.linalg.solve(B, r)
    except np.linalg.LinAlgError as e:
        # An EXACTLY singular B raises here rather than returning inf/nan, so the nonfinite
        # guard below cannot be the only route to this message.
        raise AssertionError(singular_bread_message) from e
    if not np.all(np.isfinite(displacement)):
        raise AssertionError(singular_bread_message)
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    identified = se > 0
    a = np.full(r.size, np.nan)
    a[identified] = np.abs(displacement[identified]) / se[identified]
    if not np.all(identified):
        logger.info(
            "%d of %d stacked components have (numerically) zero variance and are excluded "
            "from the SE-standardized sum-to-zero check -- a rank/identification finding for "
            "the diagnostic suite's bread_stability check, not a sum-to-zero failure.",
            int((~identified).sum()),
            r.size,
        )

    finite_a = a[np.isfinite(a)]
    # argmax over a -inf-filled copy so excluded (nan) components can never win, and so the
    # offending component is named in the failure text itself -- the per-block breakdown below
    # prints one line per block whatever fails, so it cannot attribute anything on its own.
    worst_index = int(np.argmax(np.where(np.isfinite(a), a, -np.inf))) if a.size else 0
    a_max = float(a[worst_index]) if finite_a.size else 0.0
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
            block = a[i * beta_dim : (i + 1) * beta_dim]
            finite = block[np.isfinite(block)]
            marker = "   <-- largest" if i == worst_block else ""
            lines.append(
                f"update {i + 1}: max displacement {np.max(finite):.4g} SE{marker}"
                if finite.size
                else f"update {i + 1}: all excluded"
            )
        tail = a[-theta_dim:]
        finite = tail[np.isfinite(tail)]
        marker = "   <-- largest" if worst_block == num_updates else ""
        lines.append(
            f"inference: max displacement {np.max(finite):.4g} SE{marker}"
            if finite.size
            else "inference: all excluded"
        )
        return "\n".join(lines)

    if a_max > hard_tolerance_se:
        breakdown = _block_breakdown()
        logger.info(
            "Estimating-function residual displaces estimates beyond the hard tolerance. "
            "Per-block breakdown:\n%s",
            breakdown,
        )
        raise AssertionError(
            f"Estimating functions do not sum to zero: the residual displaces {worst_label} "
            f"by {a_max:.4g} of its own standard error "
            f"(hard tolerance {hard_tolerance_se}). Per-block breakdown:\n{breakdown}"
        )
    if a_max > soft_tolerance_se:
        breakdown = _block_breakdown()
        logger.info(
            "Estimating-function residual displacement exceeds the soft tolerance. "
            "Per-block breakdown:\n%s",
            breakdown,
        )
        # The breakdown is repeated in the prompt rather than pointed at: this package
        # configures no logging handler, so the logger.info above reaches the user only if they
        # set one up themselves.
        confirm_input_check_result(
            f"\nEstimating functions do not average to within tolerance of zero: the residual "
            f"displaces {worst_label} by {a_max:.4g} of its own standard error "
            f"(soft tolerance {soft_tolerance_se}). Please decide if this is a reasonable "
            f"result given the per-block breakdown:\n{breakdown}\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
        )
    logger.info(
        "Estimating functions sum to zero within tolerance (max SE-standardized "
        "displacement %.4g).",
        a_max,
    )


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


def componentwise_absolute_tolerance(
    reference: np.ndarray, relative_floor: float = 1e-6
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
    for every small one: at a realistic 1e4-vs-0.5 spread that is atol=1e-2, which silently
    swallows a 1% reconstruction inconsistency in every O(1) component. Reducing over the subject
    axis only keeps the scale-awareness that fixes the reward-unit fixed-atol bug while confining
    each component's slack to its own scale.

    Two degenerate tiers, in order:
      - a component that is exactly zero for every subject has no scale of its own, so it borrows
        the SMALLEST nonzero component scale in the array. Borrowing the array's overall scale
        instead would reintroduce the very cross-component leak this function exists to close: at
        a 1e4-vs-0.5 spread an all-zero component would inherit atol=1e-2, so a mismatch anywhere
        in 1e-7..1e-2 there would pass -- looser than the fixed 1e-7 this replaced. The smallest
        observed scale is the most conservative stand-in that is still nonzero;
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
    relative_floor: float = 1e-6,
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
    worst = np.unravel_index(
        int(np.argmax(np.where(agrees, -np.inf, excess))), difference.shape
    )
    raise AssertionError(
        f"{context}\n"
        f"Substituting the reconstructed action probabilities changed the estimating "
        f"function's output: {int(np.count_nonzero(~agrees))} of {difference.size} values "
        f"disagree beyond tolerance (rtol={rtol} against |threaded|, plus a per-component "
        f"absolute floor of {relative_floor} x max|original| over the subject axis).\n"
        f"Worst value at index {tuple(int(i) for i in worst)} (first axis is the subject's "
        f"position in this group): original {original_np[worst]:.8g}, threaded "
        f"{threaded_np[worst]:.8g}, |difference| {difference[worst]:.8g}, allowed "
        f"{allowed[worst]:.8g} (atol {np.broadcast_to(atol, difference.shape)[worst]:.8g} + "
        f"rtol*|threaded| {rtol * abs(threaded_np[worst]):.8g}).\n"
        f"Max absolute difference: {np.max(difference):.8g}.\n"
        f"original:\n{original_np}\nthreaded:\n{threaded_np}"
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
