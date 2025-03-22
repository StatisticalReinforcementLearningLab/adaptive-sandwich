import collections
import logging
from typing import Any

import numpy as np
from jax import numpy as jnp
import pandas as pd

from constants import SmallSampleCorrections
from helper_functions import (
    confirm_input_check_result,
    load_function_from_same_named_file,
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
    action_prob_func_filename,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    alg_update_func_args,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    theta_est,
    suppress_interactive_data_checks,
    small_sample_correction,
    meat_modifier_func_filename,
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
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        study_df, in_study_col_name, policy_num_col_name, alg_update_func_args
    )
    if not suppress_interactive_data_checks:
        confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
            alg_update_func_args_action_prob_index
        )
    require_action_prob_args_in_range_0_1_if_supplied(
        alg_update_func_args, alg_update_func_args_action_prob_index
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
    if not suppress_interactive_data_checks:
        require_action_probabilities_in_study_df_can_be_reconstructed(
            study_df,
            action_prob_col_name,
            calendar_t_col_name,
            user_id_col_name,
            in_study_col_name,
            action_prob_func_filename,
            action_prob_func_args,
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
    if not suppress_interactive_data_checks:
        verify_study_df_summary_satisfactory(
            study_df,
            user_id_col_name,
            policy_num_col_name,
            calendar_t_col_name,
            in_study_col_name,
            action_prob_col_name,
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

    ### Validate small sample correction
    require_custom_small_sample_correction_function_provided_if_selected(
        small_sample_correction, meat_modifier_func_filename
    )


def require_action_probabilities_in_study_df_can_be_reconstructed(
    study_df,
    action_prob_col_name,
    calendar_t_col_name,
    user_id_col_name,
    in_study_col_name,
    action_prob_func_filename,
    action_prob_func_args,
):
    logger.info("Reconstructing action probabilities from function and arguments.")
    action_prob_func = load_function_from_same_named_file(action_prob_func_filename)

    in_study_df = study_df[study_df[in_study_col_name] == 1]
    reconstructed_action_probs = in_study_df.apply(
        lambda row: action_prob_func(
            *action_prob_func_args[row[calendar_t_col_name]][row[user_id_col_name]]
        ),
        axis=1,
    )
    try:
        np.testing.assert_allclose(
            in_study_df[action_prob_col_name].to_numpy(dtype="float64"),
            reconstructed_action_probs.to_numpy(dtype="float64"),
            atol=1e-6,
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"\nThe action probabilities could not be exactly reconstructed by the function and arguments given. Please decide if the following result is acceptable. If not, see the contract for next steps:\n{str(e)}\n\nContinue? (y/n)\n",
            e,
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

    for policy_num in alg_update_func_args:
        for user_id in alg_update_func_args[policy_num]:
            if not alg_update_func_args[policy_num][user_id]:
                continue
            arg_action_probs = alg_update_func_args[policy_num][user_id][
                alg_update_func_args_action_prob_index
            ]
            action_prob_times = alg_update_func_args[policy_num][user_id][
                alg_update_func_args_action_prob_times_index
            ]
            study_df_action_probs = []
            for decision_time in action_prob_times:
                study_df_action_probs.append(
                    study_df[
                        (study_df[calendar_t_col_name] == decision_time)
                        & (study_df[user_id_col_name] == user_id)
                    ][action_prob_col_name].values[0]
                )

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
    logger.info("Checking that all named columns are present in the study dataframe.")
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

    # Maybe any negative number taken to be a fallback policy, everything else
    # consecutive integers. Consecutive might not be feasible tho given app
    # opening issue.

    # TODO: This probably isn't going to be a requirement when we move away from
    # update times... remove if so.
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
    assert set(alg_update_func_args.keys()).issubset(
        study_df[policy_num_col_name].unique()
    ), "There are policy numbers present in algorithm update function args but not in the study dataframe. Please see the contract for details."


def require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
    study_df, in_study_col_name, policy_num_col_name, alg_update_func_args
):
    logger.info(
        "Checking that all policy numbers in the study dataframe are present in the algorithm update function args."
    )
    in_study_df = study_df[study_df[in_study_col_name] == 1]
    min_positive_policy_num = in_study_df[in_study_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ].min()
    assert set(
        study_df[study_df[policy_num_col_name] > min_positive_policy_num][
            policy_num_col_name
        ].unique()
    ).issubset(
        alg_update_func_args.keys()
    ), "There are policy numbers present in algorithm update function args but not in the study dataframe. Please see the contract for details."


def confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
    alg_update_func_args_action_prob_index,
):
    logger.info(
        "Confirming that action probabilities are not in algorithm update function args IF their index is not specified"
    )
    if alg_update_func_args_action_prob_index < 0:
        confirm_input_check_result(
            "\nYou specified that the algorithm update function function supplied does not have action probabilities as one of its arguments. Please verify this is correct.\n\nContinue? (y/n)\n"
        )


def require_action_prob_args_in_range_0_1_if_supplied(
    alg_update_func_args, alg_update_func_args_action_prob_index
):
    # TODO: implement
    pass


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


# TODO: too basic?
def require_beta_is_1D_array_in_alg_update_args(
    alg_update_func_args, alg_update_func_args_beta_index
):
    pass


# TODO: too basic?
def require_beta_is_1D_array_in_action_prob_args(
    action_prob_func_args, action_prob_func_args_beta_index
):
    pass


# TODO: too basic?
def require_theta_is_1D_array(theta_est):
    pass


def verify_study_df_summary_satisfactory(
    study_df,
    user_id_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    in_study_col_name,
    action_prob_col_name,
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

    confirm_input_check_result(
        f"\nYou provided a study dataframe reflecting a study with"
        f"\n* {num_users} users"
        f"\n* {num_non_initial_or_fallback_policies} policy updates"
        f"\n* {num_decision_times} decision times, for an average of {avg_decisions_per_user}"
        f" decisions per user"
        f"\n* {num_decision_times_with_fallback_policies} decision times"
        f" ({num_decision_times_with_fallback_policies * 100 / num_decision_times}%) for which"
        f" fallback policies were used"
        f"\n* {num_decision_times_with_multiple_policies} decision times"
        f" ({num_decision_times_with_multiple_policies * 100 / num_decision_times}%)"
        f" for which multiple non-fallback policies were used"
        f"\n* Minimum action probability {min_action_prob}"
        f"\n* Maximum action probability {max_action_prob}"
        f" \n\nDoes this seem correct? (y/n)\n"
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
    mean_estimating_function_stack: jnp.ndarray, beta_dim: int, theta_dim: int
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

    logger.info("Checking that estimating functions sum to zero across users")
    try:
        np.testing.assert_allclose(
            jnp.zeros(mean_estimating_function_stack.size),
            mean_estimating_function_stack,
            rtol=1e-5,
            atol=1e-5,
        )
    except AssertionError as e:
        logger.info(
            "Estimating function stacks do not average to zero across users.  Drilling in to specific updates and inference component."
        )
        # If this is not true there is an interal problem in the package.
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
            e,
        )


# TODO: Either implement and remove NotImplementedError or remove the option.
def require_custom_small_sample_correction_function_provided_if_selected(
    small_sample_correction, meat_modifier_func_filename
):
    if small_sample_correction == SmallSampleCorrections.custom_meat_modifier:
        raise NotImplementedError(
            "Custom small sample correction function not yet implemented."
        )


def require_adaptive_bread_inverse_is_true_inverse(
    joint_adaptive_bread_matrix, joint_adaptive_bread_inverse_matrix
):
    """
    Check that the product of the joint adaptive bread matrix and its inverse is
    sufficiently close to the identity matrix.  This is a direct check that the
    joint_adaptive_bread_inverse_matrix we create is "well-conditioned".
    """
    try:
        np.testing.assert_allclose(
            joint_adaptive_bread_matrix @ joint_adaptive_bread_inverse_matrix,
            np.eye(joint_adaptive_bread_matrix.shape[0]),
            rtol=1e-5,
            atol=1e-5,
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"\nJoint adaptive bread is not exact inverse of the constructed matrix that was inverted to form it. This likely illustrates poor conditioning:\n{str(e)}\n\nContinue? (y/n)\n",
            e,
        )
