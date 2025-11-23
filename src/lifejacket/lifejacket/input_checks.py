import collections
import logging
from typing import Any

import numpy as np
import jax
from jax import numpy as jnp
import pandas as pd
import plotext as plt

from .constants import InverseStabilizationMethods, SmallSampleCorrections
from .helper_functions import (
    confirm_input_check_result,
)

# When we print out objects for debugging, show the whole thing.
np.set_printoptions(threshold=np.inf)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


# TODO: any checks needed here about alg update function type?
def perform_first_wave_input_checks(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
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
    small_sample_correction,
):
    ### Validate algorithm loss/estimating function and args
    require_alg_update_args_given_for_all_users_at_each_update(
        study_df, user_id_col_name, alg_update_func_args
    )
    require_no_policy_numbers_present_in_alg_update_args_but_not_study_df(
        study_df, policy_num_col_name, alg_update_func_args
    )
    require_beta_is_1D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_beta_index
    )
    require_previous_betas_is_2D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_previous_betas_index
    )
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        study_df, in_study_col_name, policy_num_col_name, alg_update_func_args
    )
    confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
        alg_update_func_args_action_prob_index, suppress_interactive_data_checks
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
    require_action_prob_args_in_alg_update_func_correspond_to_study_df(
        study_df,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
    )
    require_valid_action_prob_times_given_if_index_supplied(
        study_df,
        calendar_t_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_times_index,
    )

    confirm_no_small_sample_correction_desired_if_not_requested(
        small_sample_correction, suppress_interactive_data_checks
    )

    ### Validate action prob function and args
    require_action_prob_func_args_given_for_all_users_at_each_decision(
        study_df, user_id_col_name, action_prob_func_args
    )
    require_action_prob_func_args_given_for_all_decision_times(
        study_df, calendar_t_col_name, action_prob_func_args
    )
    require_action_probabilities_in_study_df_can_be_reconstructed(
        study_df,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        in_study_col_name,
        action_prob_func_args,
        action_prob_func,
    )

    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
        study_df,
        calendar_t_col_name,
        action_prob_func_args,
        in_study_col_name,
        user_id_col_name,
    )
    require_beta_is_1D_array_in_action_prob_args(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )

    ### Validate study_df
    verify_study_df_summary_satisfactory(
        study_df,
        user_id_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        in_study_col_name,
        action_prob_col_name,
        reward_col_name,
        beta_dim,
        len(theta_est),
        suppress_interactive_data_checks,
    )

    require_all_users_have_all_times_in_study_df(
        study_df, calendar_t_col_name, user_id_col_name
    )
    require_all_named_columns_present_in_study_df(
        study_df,
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_col_name,
    )
    require_all_named_columns_not_object_type_in_study_df(
        study_df,
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_col_name,
    )
    require_binary_actions(study_df, in_study_col_name, action_col_name)
    require_binary_in_study_indicators(study_df, in_study_col_name)
    require_consecutive_integer_policy_numbers(
        study_df, in_study_col_name, policy_num_col_name
    )
    require_consecutive_integer_calendar_times(study_df, calendar_t_col_name)
    require_hashable_user_ids(study_df, in_study_col_name, user_id_col_name)
    require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name)

    ### Validate theta estimation
    require_theta_is_1D_array(theta_est)


def perform_alg_only_input_checks(
    study_df,
    in_study_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
    action_prob_func,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    suppress_interactive_data_checks,
):
    ### Validate algorithm loss/estimating function and args
    require_alg_update_args_given_for_all_users_at_each_update(
        study_df, user_id_col_name, alg_update_func_args
    )
    require_beta_is_1D_array_in_alg_update_args(
        alg_update_func_args, alg_update_func_args_beta_index
    )
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        study_df, in_study_col_name, policy_num_col_name, alg_update_func_args
    )
    confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
        alg_update_func_args_action_prob_index, suppress_interactive_data_checks
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
    require_action_prob_args_in_alg_update_func_correspond_to_study_df(
        study_df,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
    )
    require_valid_action_prob_times_given_if_index_supplied(
        study_df,
        calendar_t_col_name,
        alg_update_func_args,
        alg_update_func_args_action_prob_times_index,
    )

    ### Validate action prob function and args
    require_action_prob_func_args_given_for_all_users_at_each_decision(
        study_df, user_id_col_name, action_prob_func_args
    )
    require_action_prob_func_args_given_for_all_decision_times(
        study_df, calendar_t_col_name, action_prob_func_args
    )
    require_action_probabilities_in_study_df_can_be_reconstructed(
        study_df,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        in_study_col_name,
        action_prob_func_args,
        action_prob_func=action_prob_func,
    )

    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
        study_df,
        calendar_t_col_name,
        action_prob_func_args,
        in_study_col_name,
        user_id_col_name,
    )
    require_beta_is_1D_array_in_action_prob_args(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )


# TODO: Give a hard-to-use option to loosen this check somehow
def require_action_probabilities_in_study_df_can_be_reconstructed(
    study_df,
    action_prob_col_name,
    calendar_t_col_name,
    user_id_col_name,
    in_study_col_name,
    action_prob_func_args,
    action_prob_func,
):
    """
    Check that the action probabilities in the study dataframe can be reconstructed from the supplied
    action probability function and its arguments.

    NOTE THAT THIS IS A HARD FAILURE IF THE RECONSTRUCTION DOESN'T PASS.
    """
    logger.info("Reconstructing action probabilities from function and arguments.")

    in_study_df = study_df[study_df[in_study_col_name] == 1]
    reconstructed_action_probs = in_study_df.apply(
        lambda row: action_prob_func(
            *action_prob_func_args[row[calendar_t_col_name]][row[user_id_col_name]]
        ),
        axis=1,
    )

    np.testing.assert_allclose(
        in_study_df[action_prob_col_name].to_numpy(dtype="float64"),
        reconstructed_action_probs.to_numpy(dtype="float64"),
        atol=1e-6,
    )


def require_all_users_have_all_times_in_study_df(
    study_df, calendar_t_col_name, user_id_col_name
):
    logger.info("Checking that all users have the same set of unique calendar times.")
    # Get the unique calendar times
    unique_calendar_times = set(study_df[calendar_t_col_name].unique())

    # Group by user ID and aggregate the unique calendar times for each user
    user_calendar_times = study_df.groupby(user_id_col_name)[calendar_t_col_name].apply(
        set
    )

    # Check if all users have the same set of unique calendar times
    if not user_calendar_times.apply(lambda x: x == unique_calendar_times).all():
        raise AssertionError(
            "Not all users have all calendar times in the study dataframe. Please see the contract for details."
        )


def require_alg_update_args_given_for_all_users_at_each_update(
    study_df, user_id_col_name, alg_update_func_args
):
    logger.info(
        "Checking that algorithm update function args are given for all users at each update."
    )
    all_user_ids = set(study_df[user_id_col_name].unique())
    for policy_num in alg_update_func_args:
        assert (
            set(alg_update_func_args[policy_num].keys()) == all_user_ids
        ), f"Not all users present in algorithm update function args for policy number {policy_num}. Please see the contract for details."


def require_action_prob_args_in_alg_update_func_correspond_to_study_df(
    study_df,
    action_prob_col_name,
    calendar_t_col_name,
    user_id_col_name,
    alg_update_func_args,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
):
    logger.info(
        "Checking that the action probabilities supplied in the algorithm update function args, if"
        " any, correspond to those in the study dataframe for the corresponding users and decision"
        " times."
    )
    if alg_update_func_args_action_prob_index < 0:
        return

    # Precompute a lookup dictionary for faster access
    study_df_lookup = study_df.set_index([calendar_t_col_name, user_id_col_name])[
        action_prob_col_name
    ].to_dict()

    for policy_num, user_args in alg_update_func_args.items():
        for user_id, args in user_args.items():
            if not args:
                continue
            arg_action_probs = args[alg_update_func_args_action_prob_index]
            action_prob_times = args[
                alg_update_func_args_action_prob_times_index
            ].flatten()

            # Use the precomputed lookup dictionary
            study_df_action_probs = [
                study_df_lookup[(decision_time.item(), user_id)]
                for decision_time in action_prob_times
            ]

            assert np.allclose(
                arg_action_probs.flatten(),
                study_df_action_probs,
            ), (
                f"There is a mismatch for user {user_id} between the action probabilities supplied"
                f" in the args to the algorithm update function at policy {policy_num} and those in"
                " the study dataframe for the supplied times. Please see the contract for details."
            )


def require_action_prob_func_args_given_for_all_users_at_each_decision(
    study_df,
    user_id_col_name,
    action_prob_func_args,
):
    logger.info(
        "Checking that action prob function args are given for all users at each decision time."
    )
    all_user_ids = set(study_df[user_id_col_name].unique())
    for decision_time in action_prob_func_args:
        assert (
            set(action_prob_func_args[decision_time].keys()) == all_user_ids
        ), f"Not all users present in algorithm update function args for decision time {decision_time}. Please see the contract for details."


def require_action_prob_func_args_given_for_all_decision_times(
    study_df, calendar_t_col_name, action_prob_func_args
):
    logger.info(
        "Checking that action prob function args are given for all decision times."
    )
    all_times = set(study_df[calendar_t_col_name].unique())

    assert (
        set(action_prob_func_args.keys()) == all_times
    ), "Not all decision times present in action prob function args. Please see the contract for details."


def require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
    study_df: pd.DataFrame,
    calendar_t_col_name: str,
    action_prob_func_args: dict[str, dict[str, tuple[Any, ...]]],
    in_study_col_name,
    user_id_col_name,
):
    logger.info(
        "Checking that action probability function args are blank for exactly the times each user"
        " is not in the study according to the study dataframe."
    )
    out_of_study_df = study_df[study_df[in_study_col_name] == 0]
    out_of_study_times_by_user_according_to_study_df = (
        out_of_study_df.groupby(user_id_col_name)[calendar_t_col_name]
        .apply(set)
        .to_dict()
    )

    out_of_study_times_by_user_according_to_action_prob_func_args = (
        collections.defaultdict(set)
    )
    for decision_time, action_prob_args_by_user in action_prob_func_args.items():
        for user_id, action_prob_args in action_prob_args_by_user.items():
            if not action_prob_args:
                out_of_study_times_by_user_according_to_action_prob_func_args[
                    user_id
                ].add(decision_time)

    assert (
        out_of_study_times_by_user_according_to_study_df
        == out_of_study_times_by_user_according_to_action_prob_func_args
    ), (
        "Out-of-study decision times according to the study dataframe do not match up with the"
        " times for which action probability arguments are blank for all users. Please see the"
        " contract for details."
    )


def require_all_named_columns_present_in_study_df(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
):
    logger.info("Checking that all named columns are present in the study dataframe.")
    assert (
        in_study_col_name in study_df.columns
    ), f"{in_study_col_name} not in study df."
    assert action_col_name in study_df.columns, f"{action_col_name} not in study df."
    assert (
        policy_num_col_name in study_df.columns
    ), f"{policy_num_col_name} not in study df."
    assert (
        calendar_t_col_name in study_df.columns
    ), f"{calendar_t_col_name} not in study df."
    assert user_id_col_name in study_df.columns, f"{user_id_col_name} not in study df."
    assert (
        action_prob_col_name in study_df.columns
    ), f"{action_prob_col_name} not in study df."


def require_all_named_columns_not_object_type_in_study_df(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
):
    logger.info("Checking that all named columns are not type object.")
    for colname in (
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_col_name,
    ):
        assert (
            study_df[colname].dtype != "object"
        ), f"At least {colname} is of object type in study df."


def require_binary_actions(study_df, in_study_col_name, action_col_name):
    logger.info("Checking that actions are binary.")
    assert (
        study_df[study_df[in_study_col_name] == 1][action_col_name]
        .astype("int64")
        .isin([0, 1])
        .all()
    ), "Actions are not binary."


def require_binary_in_study_indicators(study_df, in_study_col_name):
    logger.info("Checking that in-study indicators are binary.")
    assert (
        study_df[study_df[in_study_col_name] == 1][in_study_col_name]
        .astype("int64")
        .isin([0, 1])
        .all()
    ), "In-study indicators are not binary."


def require_consecutive_integer_policy_numbers(
    study_df, in_study_col_name, policy_num_col_name
):
    # TODO: This is a somewhat rough check of this, could also check nondecreasing temporally

    logger.info(
        "Checking that in-study, non-fallback policy numbers are consecutive integers."
    )

    in_study_df = study_df[study_df[in_study_col_name] == 1]
    nonnegative_policy_df = in_study_df[in_study_df[policy_num_col_name] >= 0]
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


def require_consecutive_integer_calendar_times(study_df, calendar_t_col_name):
    # This is a somewhat rough check of this, more like checking there are no
    # gaps in the integers covered.  But we have other checks that all users
    # have same times, etc.
    # Note these times should be well-formed even when the user is not in the study.
    logger.info("Checking that calendar times are consecutive integers.")
    assert np.array_equal(
        study_df[calendar_t_col_name].unique(),
        range(
            study_df[calendar_t_col_name].min(), study_df[calendar_t_col_name].max() + 1
        ),
    ), "Calendar times are not consecutive integers."


def require_hashable_user_ids(study_df, in_study_col_name, user_id_col_name):
    logger.info("Checking that user IDs are hashable.")
    isinstance(
        study_df[study_df[in_study_col_name] == 1][user_id_col_name][0],
        collections.abc.Hashable,
    )


def require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name):
    logger.info("Checking that action probabilities are in the interval (0, 1).")
    study_df[action_prob_col_name].between(0, 1, inclusive="neither").all()


def require_no_policy_numbers_present_in_alg_update_args_but_not_study_df(
    study_df, policy_num_col_name, alg_update_func_args
):
    logger.info(
        "Checking that policy numbers in algorithm update function args are present in the study dataframe."
    )
    alg_update_policy_nums = sorted(alg_update_func_args.keys())
    study_df_policy_nums = sorted(study_df[policy_num_col_name].unique())
    assert set(alg_update_policy_nums).issubset(set(study_df_policy_nums)), (
        f"There are policy numbers present in algorithm update function args but not in the study dataframe. "
        f"\nalg_update_func_args policy numbers: {alg_update_policy_nums}"
        f"\nstudy_df policy numbers: {study_df_policy_nums}.\nPlease see the contract for details."
    )


def require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
    study_df, in_study_col_name, policy_num_col_name, alg_update_func_args
):
    logger.info(
        "Checking that all policy numbers in the study dataframe are present in the algorithm update function args."
    )
    in_study_df = study_df[study_df[in_study_col_name] == 1]
    # Get the number of the initial policy. 0 is recommended but not required.
    min_nonnegative_policy_number = in_study_df[in_study_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ]
    assert set(
        in_study_df[in_study_df[policy_num_col_name] > min_nonnegative_policy_number][
            policy_num_col_name
        ].unique()
    ).issubset(
        alg_update_func_args.keys()
    ), f"There are non-fallback, non-initial policy numbers in the study dataframe that are not in the update function args: {set(in_study_df[in_study_df[policy_num_col_name] > 0][policy_num_col_name].unique()) - set(alg_update_func_args.keys())}. Please see the contract for details."


def confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
    alg_update_func_args_action_prob_index,
    suppress_interactive_data_checks,
):
    logger.info(
        "Confirming that action probabilities are not in algorithm update function args IF their index is not specified"
    )
    if alg_update_func_args_action_prob_index < 0:
        confirm_input_check_result(
            "\nYou specified that the algorithm update function supplied does not have action probabilities as one of its arguments. Please verify this is correct.\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
        )


def confirm_no_small_sample_correction_desired_if_not_requested(
    small_sample_correction,
    suppress_interactive_data_checks,
):
    logger.info(
        "Confirming that no small sample correction is desired if it's not requested."
    )
    if small_sample_correction == SmallSampleCorrections.NONE:
        confirm_input_check_result(
            "\nYou specified that you would not like to perform any small-sample corrections. Please verify that this is correct.\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
        )


def confirm_no_adaptive_bread_inverse_stabilization_method_desired_if_not_requested(
    adaptive_bread_inverse_stabilization_method,
    suppress_interactive_data_checks,
):
    logger.info(
        "Confirming that no adaptive bread inverse stabilization method is desired if it's not requested."
    )
    if adaptive_bread_inverse_stabilization_method == InverseStabilizationMethods.NONE:
        confirm_input_check_result(
            "\nYou specified that you would not like to perform any inverse stabilization while forming the adaptive variance. This is not usually recommended. Please verify that it is correct or select one of the available options.\n\nContinue? (y/n)\n",
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
        for user_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][user_id]:
                continue
            assert (
                alg_update_func_args[policy_num][user_id][
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
        for user_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][user_id]:
                continue
            assert (
                alg_update_func_args[policy_num][user_id][
                    alg_update_func_args_previous_betas_index
                ].ndim
                == 2
            ), "Previous betas is not a 2D array in the algorithm update function args."


def require_beta_is_1D_array_in_action_prob_args(
    action_prob_func_args, action_prob_func_args_beta_index
):
    for decision_time in action_prob_func_args:
        for user_id in action_prob_func_args[decision_time]:
            if not action_prob_func_args[decision_time][user_id]:
                continue
            assert (
                action_prob_func_args[decision_time][user_id][
                    action_prob_func_args_beta_index
                ].ndim
                == 1
            ), "Beta is not a 1D array in the action probability function args."


def require_theta_is_1D_array(theta_est):
    assert theta_est.ndim == 1, "Theta is not a 1D array."


def verify_study_df_summary_satisfactory(
    study_df,
    user_id_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    in_study_col_name,
    action_prob_col_name,
    reward_col_name,
    beta_dim,
    theta_dim,
    suppress_interactive_data_checks,
):

    in_study_df = study_df[study_df[in_study_col_name] == 1]
    num_users = in_study_df[user_id_col_name].nunique()
    num_non_initial_or_fallback_policies = in_study_df[
        in_study_df[policy_num_col_name] > 0
    ][policy_num_col_name].nunique()
    num_decision_times_with_fallback_policies = len(
        in_study_df[in_study_df[policy_num_col_name] < 0]
    )
    num_decision_times = in_study_df[calendar_t_col_name].nunique()
    avg_decisions_per_user = len(in_study_df) / num_users
    num_decision_times_with_multiple_policies = (
        in_study_df[in_study_df[policy_num_col_name] >= 0]
        .groupby(calendar_t_col_name)[policy_num_col_name]
        .nunique()
        > 1
    ).sum()
    min_action_prob = in_study_df[action_prob_col_name].min()
    max_action_prob = in_study_df[action_prob_col_name].max()
    min_non_fallback_policy_num = in_study_df[in_study_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ].min()
    num_data_points_before_first_update = len(
        in_study_df[in_study_df[policy_num_col_name] == min_non_fallback_policy_num]
    )

    median_action_probabilities = (
        in_study_df.groupby(calendar_t_col_name)[action_prob_col_name]
        .median()
        .to_numpy()
    )
    quartiles = in_study_df.groupby(calendar_t_col_name)[action_prob_col_name].quantile(
        [0.25, 0.75]
    )
    q25_action_probabilities = quartiles.xs(0.25, level=1).to_numpy()
    q75_action_probabilities = quartiles.xs(0.75, level=1).to_numpy()

    avg_rewards = in_study_df.groupby(calendar_t_col_name)[reward_col_name].mean()

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
        f"\nYou provided a study dataframe reflecting a study with"
        f"\n* {num_users} users"
        f"\n* {num_non_initial_or_fallback_policies} policy updates"
        f"\n* {num_decision_times} decision times, for an average of {avg_decisions_per_user}"
        f" decisions per user"
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
        "Checking that betas match across users for each update in the algorithm update function args."
    )
    for policy_num in alg_update_func_args:
        first_beta = None
        for user_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][user_id]:
                continue
            beta = alg_update_func_args[policy_num][user_id][
                alg_update_func_args_beta_index
            ]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(
                    beta, first_beta
                ), f"Betas do not match across users in the algorithm update function args for policy number {policy_num}. Please see the contract for details."

def require_previous_betas_match_in_alg_update_args_each_update(
    alg_update_func_args, alg_update_func_args_previous_betas_index
):
    logger.info(
        "Checking that previous betas match across users for each update in the algorithm update function args."
    )
    if alg_update_func_args_previous_betas_index < 0:
        return 
    
    for policy_num in alg_update_func_args:
        first_previous_betas = None
        for user_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][user_id]:
                continue
            previous_betas = alg_update_func_args[policy_num][user_id][
                alg_update_func_args_previous_betas_index
            ]
            if first_previous_betas is None:
                first_previous_betas = previous_betas
            else:
                assert np.array_equal(
                    previous_betas, first_previous_betas
                ), f"Previous betas do not match across users in the algorithm update function args for policy number {policy_num}. Please see the contract for details."


def require_betas_match_in_action_prob_func_args_each_decision(
    action_prob_func_args, action_prob_func_args_beta_index
):
    logger.info(
        "Checking that betas match across users for each decision time in the action prob args."
    )
    for decision_time in action_prob_func_args:
        first_beta = None
        for user_id in action_prob_func_args[decision_time]:
            if not action_prob_func_args[decision_time][user_id]:
                continue
            beta = action_prob_func_args[decision_time][user_id][
                action_prob_func_args_beta_index
            ]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(
                    beta, first_beta
                ), f"Betas do not match across users in the action prob args for decision_time {decision_time}. Please see the contract for details."


def require_valid_action_prob_times_given_if_index_supplied(
    study_df,
    calendar_t_col_name,
    alg_update_func_args,
    alg_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob times are valid if index is supplied.")

    if alg_update_func_args_action_prob_times_index < 0:
        return

    min_time = study_df[calendar_t_col_name].min()
    max_time = study_df[calendar_t_col_name].max()
    for policy_idx, args_by_user in alg_update_func_args.items():
        for user_id, args in args_by_user.items():
            if not args:
                continue
            times = args[alg_update_func_args_action_prob_times_index]
            assert (
                times[i] > times[i - 1] for i in range(1, len(times))
            ), f"Non-strictly-increasing times were given for action probabilities in the algorithm update function args for user {user_id} and policy {policy_idx}. Please see the contract for details."
            assert (
                times[0] >= min_time and times[-1] <= max_time
            ), f"Times not present in the study were given for action probabilities in the algorithm update function args. The min and max times in the study dataframe are {min_time} and {max_time}, while user {user_id} has times {times} supplied for policy {policy_idx}. Please see the contract for details."


def require_estimating_functions_sum_to_zero(
    mean_estimating_function_stack: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    suppress_interactive_data_checks: bool,
):
    """
    This is a test that the correct loss/estimating functions have
    been given for both the algorithm updates and inference. If that is true, then the
    loss/estimating functions when evaluated should sum to approximately zero across users.  These
    values have been stacked and averaged across users in mean_estimating_function_stack, which
    we simply compare to the zero vector.  We can isolate components for each update and inference
    by considering the dimensions of the beta vectors and the theta vector.

    Inputs:
    mean_estimating_function_stack:
        The mean of the estimating function stack (a component for each algorithm update and
        inference) across users. This should be a 1D array.
    beta_dim:
        The dimension of the beta vectors that parameterize the algorithm.
    theta_dim:
        The dimension of the theta vector that we estimate during after-study analysis.

    Returns:
    None
    """

    logger.info("Checking that estimating functions average to zero across users")

    # Have a looser hard failure cutoff before the typical interactive check
    try:
        np.testing.assert_allclose(
            mean_estimating_function_stack,
            jnp.zeros(mean_estimating_function_stack.size),
            atol=1e-2,
        )
    except AssertionError as e:
        logger.info(
            "Estimating function stacks do not average to within loose tolerance of zero across users.  Drilling in to specific updates and inference component."
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
            atol=1e-5,
        )
    except AssertionError as e:
        logger.info(
            "Estimating function stacks do not average to within specified tolerance of zero across users.  Drilling in to specific updates and inference component."
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


def require_RL_estimating_functions_sum_to_zero(
    mean_estimating_function_stack: jnp.ndarray,
    beta_dim: int,
    suppress_interactive_data_checks: bool,
):
    """
    This is a test that the correct loss/estimating functions have
    been given for both the algorithm updates and inference. If that is true, then the
    loss/estimating functions when evaluated should sum to approximately zero across users.  These
    values have been stacked and averaged across users in mean_estimating_function_stack, which
    we simply compare to the zero vector.  We can isolate components for each update and inference
    by considering the dimensions of the beta vectors and the theta vector.

    Inputs:
    mean_estimating_function_stack:
        The mean of the estimating function stack (a component for each algorithm update and
        inference) across users. This should be a 1D array.
    beta_dim:
        The dimension of the beta vectors that parameterize the algorithm.
    theta_dim:
        The dimension of the theta vector that we estimate during after-study analysis.

    Returns:
    None
    """

    logger.info("Checking that RL estimating functions average to zero across users")

    # Have a looser hard failure cutoff before the typical interactive check
    try:
        np.testing.assert_allclose(
            mean_estimating_function_stack,
            jnp.zeros(mean_estimating_function_stack.size),
            atol=1e-2,
        )
    except AssertionError as e:
        logger.info(
            "RL estimating function stacks do not average to zero across users.  Drilling in to specific updates and inference component."
        )
        num_updates = (mean_estimating_function_stack.size) // beta_dim
        for i in range(num_updates):
            logger.info(
                "Mean estimating function contribution for update %s:\n%s",
                i + 1,
                mean_estimating_function_stack[i * beta_dim : (i + 1) * beta_dim],
            )
        # TODO: We may need to email instead of failing here for monitoring algorithm.
        raise e

    try:
        np.testing.assert_allclose(
            mean_estimating_function_stack,
            jnp.zeros(mean_estimating_function_stack.size),
            atol=1e-5,
        )
    except AssertionError as e:
        logger.info(
            "RL estimating function stacks do not average to zero across users.  Drilling in to specific updates and inference component."
        )
        num_updates = (mean_estimating_function_stack.size) // beta_dim
        for i in range(num_updates):
            logger.info(
                "Mean estimating function contribution for update %s:\n%s",
                i + 1,
                mean_estimating_function_stack[i * beta_dim : (i + 1) * beta_dim],
            )
        # TODO: Email instead of requiring user input for monitoring alg.
        confirm_input_check_result(
            f"\nEstimating functions do not average to within default tolerance of zero vector. Please decide if the following is a reasonable result, taking into account the above breakdown by update number and inference. If not, there are several possible reasons for failure mentioned in the contract. Results:\n{str(e)}\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
            e,
        )


def require_adaptive_bread_inverse_is_true_inverse(
    joint_adaptive_bread_matrix,
    joint_adaptive_bread_inverse_matrix,
    suppress_interactive_data_checks,
):
    """
    Check that the product of the joint adaptive bread matrix and its inverse is
    sufficiently close to the identity matrix.  This is a direct check that the
    joint_adaptive_bread_inverse_matrix we create is "well-conditioned".
    """
    should_be_identity = (
        joint_adaptive_bread_matrix @ joint_adaptive_bread_inverse_matrix
    )
    identity = np.eye(joint_adaptive_bread_matrix.shape[0])
    try:
        np.testing.assert_allclose(
            should_be_identity,
            identity,
            rtol=1e-5,
            atol=1e-5,
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"\nJoint adaptive bread is not exact inverse of the constructed matrix that was inverted to form it. This likely illustrates poor conditioning:\n{str(e)}\n\nContinue? (y/n)\n",
            suppress_interactive_data_checks,
            e,
        )

    # If we haven't already errored out, return some measures of how far off we are from identity
    diff = should_be_identity - identity
    logger.debug(
        "Difference between should-be-identity produced by multiplying joint adaptive bread inverse and its computed inverse and actual identity:\n%s",
        diff,
    )

    diff_abs_max = np.max(np.abs(diff))
    diff_frobenius_norm = np.linalg.norm(diff, "fro")

    logger.info("Maximum abs element of difference: %s", diff_abs_max)
    logger.info("Frobenius norm of difference: %s", diff_frobenius_norm)

    return diff_abs_max, diff_frobenius_norm


def require_threaded_algorithm_estimating_function_args_equivalent(
    algorithm_estimating_func,
    update_func_args_by_by_user_id_by_policy_num,
    threaded_update_func_args_by_policy_num_by_user_id,
    suppress_interactive_data_checks,
):
    """
    Check that the algorithm estimating function returns the same values
    when called with the original arguments and when called with the
    reconstructed action probabilities substituted in.
    """
    for (
        policy_num,
        update_func_args_by_user_id,
    ) in update_func_args_by_by_user_id_by_policy_num.items():
        for (
            user_id,
            unthreaded_args,
        ) in update_func_args_by_user_id.items():
            if not unthreaded_args:
                continue
            np.testing.assert_allclose(
                algorithm_estimating_func(*unthreaded_args),
                # Need to stop gradient here because we can't convert a traced value to np array
                jax.lax.stop_gradient(
                    algorithm_estimating_func(
                        *threaded_update_func_args_by_policy_num_by_user_id[user_id][
                            policy_num
                        ]
                    )
                ),
                atol=1e-7,
                rtol=1e-3,
            )


def require_threaded_inference_estimating_function_args_equivalent(
    inference_estimating_func,
    inference_func_args_by_user_id,
    threaded_inference_func_args_by_user_id,
    suppress_interactive_data_checks,
):
    """
    Check that the inference estimating function returns the same values
    when called with the original arguments and when called with the
    reconstructed action probabilities substituted in.
    """
    for user_id, unthreaded_args in inference_func_args_by_user_id.items():
        if not unthreaded_args:
            continue
        np.testing.assert_allclose(
            inference_estimating_func(*unthreaded_args),
            # Need to stop gradient here because we can't convert a traced value to np array
            jax.lax.stop_gradient(
                inference_estimating_func(
                    *threaded_inference_func_args_by_user_id[user_id]
                )
            ),
            rtol=1e-2,
        )
