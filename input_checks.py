import collections
import logging

import numpy as np
from jax import numpy as jnp

from constants import SmallSampleCorrections
from helper_functions import (
    confirm_input_check_result,
    load_function_from_same_named_file,
)

# When we print out objects for debugging, show the whole thing.
np.set_printoptions(threshold=np.inf)

CONDITION_NUMBER_CUTOFF = 10**3

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


# TODO: any checks needed here about rl update function type?
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
    rl_update_func_args,
    rl_update_func_args_beta_index,
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
    theta_est,
    suppress_interactive_data_checks,
    small_sample_correction,
    meat_modifier_func_filename,
):
    # TODO: Also, maybe this wave shouldn't include loading functions
    # supplied--do action prob reconstruction, theta estimation, estimating function sum, etc. in a later wave.

    ### Validate RL loss/estimating function and args
    require_rl_update_args_given_for_all_users_at_each_update(
        study_df, user_id_col_name, rl_update_func_args
    )
    require_no_policy_numbers_present_in_rl_update_args_but_not_study_df(
        study_df, policy_num_col_name, rl_update_func_args
    )
    require_beta_is_1D_array_in_rl_update_args(
        rl_update_func_args, rl_update_func_args_beta_index
    )
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_update_args(
        study_df, in_study_col_name, policy_num_col_name, rl_update_func_args
    )
    if not suppress_interactive_data_checks:
        confirm_action_probabilities_not_in_rl_update_args_if_index_not_supplied(
            rl_update_func_args_action_prob_index
        )
    require_action_prob_args_in_range_0_1_if_supplied(
        rl_update_func_args, rl_update_func_args_action_prob_index
    )
    require_action_prob_times_given_if_index_supplied(
        rl_update_func_args_action_prob_index,
        rl_update_func_args_action_prob_times_index,
    )
    require_action_prob_index_given_if_times_supplied(
        rl_update_func_args_action_prob_index,
        rl_update_func_args_action_prob_times_index,
    )
    require_betas_match_in_rl_update_args_each_update(
        rl_update_func_args, rl_update_func_args_beta_index
    )
    require_valid_action_prob_times_given_if_index_supplied(
        study_df,
        calendar_t_col_name,
        rl_update_func_args,
        rl_update_func_args_action_prob_times_index,
    )

    ### Validate action prob function and args
    require_action_prob_func_args_given_for_all_users_at_each_decision(
        study_df, user_id_col_name, action_prob_func_args
    )
    require_action_prob_func_args_given_for_all_decision_times(
        study_df, calendar_t_col_name, action_prob_func_args
    )
    if not suppress_interactive_data_checks:
        require_action_probabilities_can_be_reconstructed(
            study_df,
            action_prob_col_name,
            calendar_t_col_name,
            user_id_col_name,
            in_study_col_name,
            action_prob_func_filename,
            action_prob_func_args,
        )
    require_beta_is_1D_array_in_action_prob_args(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )

    ### Validate study_df
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

    if not suppress_interactive_data_checks:
        verify_study_df_summary_satisfactory(study_df)

    ### Validate theta estimation
    require_theta_is_1D_array(theta_est)

    ### Validate small sample correction
    require_custom_small_sample_correction_function_provided_if_selected(
        small_sample_correction, meat_modifier_func_filename
    )


def require_action_probabilities_can_be_reconstructed(
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
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"The action probabilities could not be exactly reconstructed by the function and arguments given. Please decide if the following result is acceptable. If not, see the contract for next steps:\n{str(e)}\n\nContinue? (y/n)\n",
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
            "Not all users have the same calendar times in the study dataframe. Please see the contract for details."
        )


def require_rl_update_args_given_for_all_users_at_each_update(
    study_df, user_id_col_name, rl_update_func_args
):
    logger.info(
        "Checking that RL update function args are given for all users at each update."
    )
    all_user_ids = set(study_df[user_id_col_name].unique())
    for policy_num in rl_update_func_args:
        assert (
            set(rl_update_func_args[policy_num].keys()) == all_user_ids
        ), f"Not all users present in RL update function args for policy number {policy_num}. Please see the contract for details."


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
        ), f"Not all users present in RL update function args for decision time {decision_time}. Please see the contract for details."


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
        assert study_df[colname].dtype != "object"


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
    logger.info("Checking that action probabilities are in the range [0, 1].")
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    study_df[action_prob_col_name].between(0, 1).all()


def require_no_policy_numbers_present_in_rl_update_args_but_not_study_df(
    study_df, policy_num_col_name, rl_update_func_args
):
    logger.info(
        "Checking that policy numbers in RL update function args are present in the study dataframe."
    )
    assert set(rl_update_func_args.keys()).issubset(
        study_df[policy_num_col_name].unique()
    ), "There are policy numbers present in RL update function args but not in the study dataframe. Please see the contract for details."


def require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_update_args(
    study_df, in_study_col_name, policy_num_col_name, rl_update_func_args
):
    logger.info(
        "Checking that all policy numbers in the study dataframe are present in the RL update function args."
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
        rl_update_func_args.keys()
    ), "There are policy numbers present in RL update function args but not in the study dataframe. Please see the contract for details."


def confirm_action_probabilities_not_in_rl_update_args_if_index_not_supplied(
    rl_update_func_args_action_prob_index,
):
    logger.info(
        "Confirming that action probabilities are not in RL update function args IF their index is not specified"
    )
    if rl_update_func_args_action_prob_index < 0:
        confirm_input_check_result(
            "You specified that the RL update function function supplied does not have action probabilities as one of its arguments. Please verify this is correct.\n\nContinue? (y/n)\n"
        )


def require_action_prob_args_in_range_0_1_if_supplied(
    rl_update_func_args, rl_update_func_args_action_prob_index
):
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    pass


def require_action_prob_times_given_if_index_supplied(
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob times are given if index is supplied.")
    if rl_update_func_args_action_prob_index >= 0:
        assert rl_update_func_args_action_prob_times_index >= 0 and (
            rl_update_func_args_action_prob_times_index
            != rl_update_func_args_action_prob_index
        )


def require_action_prob_index_given_if_times_supplied(
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob index is given if times are supplied.")
    if rl_update_func_args_action_prob_times_index >= 0:
        assert rl_update_func_args_action_prob_index >= 0 and (
            rl_update_func_args_action_prob_times_index
            != rl_update_func_args_action_prob_index
        )


# TODO: too basic?
def require_beta_is_1D_array_in_rl_update_args(
    rl_update_func_args, rl_update_func_args_beta_index
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
):
    # TODO: Give a summary of the study dataframe and ask the user to verify that it
    # is satisfactory.  This should help avoid gross errors.
    pass


def require_betas_match_in_rl_update_args_each_update(
    rl_update_func_args, rl_update_func_args_beta_index
):
    logger.info(
        "Checking that betas match across users for each update in the RL update function args."
    )
    for policy_num in rl_update_func_args:
        first_beta = None
        for user_id in rl_update_func_args[policy_num]:
            if not rl_update_func_args[policy_num][user_id]:
                continue
            beta = rl_update_func_args[policy_num][user_id][
                rl_update_func_args_beta_index
            ]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(
                    beta, first_beta
                ), f"Betas do not match across users in the RL update function args for policy number {policy_num}. Please see the contract for details."


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
    rl_update_func_args,
    rl_update_func_args_action_prob_times_index,
):
    logger.info("Checking that action prob times are valid if index is supplied.")

    if rl_update_func_args_action_prob_times_index < 0:
        return

    min_time = study_df[calendar_t_col_name].min()
    max_time = study_df[calendar_t_col_name].max()
    for args_by_user in rl_update_func_args.values():
        for args in args_by_user.values():
            if not args:
                continue
            times = args[rl_update_func_args_action_prob_times_index]
            assert (
                times[i] > times[i - 1] for i in range(1, len(times))
            ), "Non-strictly-increasing times give for action proabilities in RL update function args. Please see the contract for details."
            assert (
                times[0] >= min_time and times[-1] <= max_time
            ), "Times not present in the study given for action proabilities in RL update function args. Please see the contract for details."


def require_theta_estimating_functions_sum_to_zero(
    inference_estimating_function_values, theta_dim
):
    # This is a test that the correct inference loss/estimating function has
    # been given, corresponding either to the theta estimation function provided
    # or the theta estimation procedure used to produce the theta estimate
    # provided.

    # If the theta estimate is directly provided, another possible failure mode
    # is that the study dataframe doesn't faithfully represent the data used
    # to produce that theta estimate.
    logger.info("Checking that theta estimating functions sum to zero across users")
    try:
        np.testing.assert_allclose(
            np.sum(inference_estimating_function_values, axis=0),
            jnp.zeros(theta_dim),
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"Theta estimating functions do not sum to within default tolerance of zero vector. Please decide if the following is a reasonable result. If not, there are several possible reasons for failure mentioned in the contract. Results:\n{str(e)}\n\nContinue? (y/n)\n",
            e,
        )


# TODO: Hotspot for replacing notion of update times
# TODO: Remove breakpoint eventually
def require_beta_estimating_functions_sum_to_zero(
    update_times, algorithm_statistics_by_calendar_t, beta_dim
):
    logger.info(
        "Checking that beta estimating functions sum to zero across users for each update"
    )
    # This is a test that the correct RL update function/estimating function has
    # been given, along with correct arguments for each update time.

    # If that is true, then the RL update function/estimating function should sum to zero
    # for each update time when the RESULTING beta estimate and the data used
    # to produce it are plugged in as the remaining args.

    # First we collect the specific times at which the beta estimating functions
    # don't sum to one. This is for easier debugging.
    failing_times = []
    all_update_sums = []
    for t in update_times:
        single_update_sum = sum(
            algorithm_statistics_by_calendar_t[t]["loss_gradients_by_user_id"].values(),
        )
        all_update_sums.append(single_update_sum)
        if not np.allclose(
            single_update_sum,
            jnp.zeros((beta_dim, 1)),
        ):
            failing_times.append(t)

    # Now, we actually run our assert on a concatenated sum across all update times,
    # because we are going to ask the user to verify that the max deviation is ok if
    # the sum across all update times is not within the default tolerance of zero.
    try:
        np.testing.assert_allclose(
            np.concatenate(all_update_sums),
            jnp.zeros(beta_dim * len(update_times)),
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"Beta estimating functions with args and provided beta estimate plugged in do not sum across users to within default tolerance of the zero vector for the updates first applying at times {failing_times}. Please decide if the maximum element-wise deviation from zero of the following concatenated vector sum across all update times is acceptably low. If not, there are several possible failure modes and next steps mentioned in the contract. Results:\n{str(e)}\n\nContinue? (y/n)\n",
            e,
        )


# TODO: Also have interactive check if condition number merely high?
# TODO: Hotspot for replacing notion of update times
def check_avg_hessian_condition_num_at_each_update(
    update_times,
    algorithm_statistics_by_calendar_t,
):
    logger.info(
        "Checking that average estimating function derivatives for RL are not singular at each update time"
    )
    poorly_conditioned_updates_dict = {}
    for t in update_times:
        condition_number = np.linalg.cond(
            algorithm_statistics_by_calendar_t[t]["avg_loss_hessian"]
        )
        logger.info("Condition number at update time %s: %s", t, condition_number)
        if condition_number > CONDITION_NUMBER_CUTOFF:
            poorly_conditioned_updates_dict[t] = condition_number

    if poorly_conditioned_updates_dict:
        confirm_input_check_result(
            f"Potentially poorly conditioned (possibly singular) average estimating function derivatives for RL at the following update times:\n{poorly_conditioned_updates_dict}\n\nPlease see the contract for details.\n\nContinue? (y/n)\n"
        )


def require_non_singular_avg_hessian_inference(
    inference_estimating_function_derivatives,
):
    logger.info(
        "Checking that average estimating function derivative for inference is not singular"
    )
    avg_inference_estimating_function_derivative = jnp.mean(
        inference_estimating_function_derivatives, axis=0
    )
    condition_number = np.linalg.cond(avg_inference_estimating_function_derivative)
    logger.info(
        "Condition number for average inference estimating function derivative: %s",
        condition_number,
    )

    if condition_number > CONDITION_NUMBER_CUTOFF:
        confirm_input_check_result(
            f"Potentially poorly conditioned (possibly singular) average estimating function derivative for inference. Condition number:\n\n{condition_number}.\n\nPlease see the contract for details.\n\nContinue? (y/n)\n"
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
    the identity matrix.  This is a direct check that the joint_adaptive_bread_inverse_matrix
    we create is "well-conditioned".
    """
    try:
        np.testing.assert_allclose(
            joint_adaptive_bread_matrix @ joint_adaptive_bread_inverse_matrix,
            np.eye(joint_adaptive_bread_matrix.shape[0]),
        )
    except AssertionError as e:
        confirm_input_check_result(
            f"Joint adaptive bread is not exact inverse of the constructed matrix that was inverted to form it. This likely illustrates poor conditioning:\n{str(e)}\n\nContinue? (y/n)\n",
            e,
        )
