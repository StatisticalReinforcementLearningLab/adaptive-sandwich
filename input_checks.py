import numpy as np
from helper_functions import load_function_from_same_named_file


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
    rl_loss_func_args,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    rl_loss_func_args_action_prob_times_index,
    theta_est,
):
    # TODO: Also, maybe this wave shouldn't include loading functions
    # supplied--do action prob reconstruction, theta estimation, estimating function sum, etc. in a later wave.

    ### Validate RL loss/estimating function and args
    require_rl_loss_args_given_for_all_users_at_each_update(
        study_df, user_id_col_name, rl_loss_func_args
    )
    require_no_policy_numbers_present_in_rl_loss_args_but_not_study_df(
        study_df, policy_num_col_name, rl_loss_func_args
    )
    require_beta_is_1D_array_in_rl_loss_args(
        rl_loss_func_args, rl_loss_func_args_beta_index
    )
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_loss_args(
        study_df, policy_num_col_name, rl_loss_func_args
    )
    confirm_action_probabilities_not_in_rl_loss_args_if_index_not_supplied(
        rl_loss_func_args_action_prob_index
    )
    require_action_prob_args_in_range_0_1_if_supplied(
        rl_loss_func_args, rl_loss_func_args_action_prob_index
    )
    require_action_prob_times_given_if_index_supplied(
        rl_loss_func_args,
        rl_loss_func_args_action_prob_index,
        rl_loss_func_args_action_prob_times_index,
    )
    require_betas_match_in_rl_loss_args_each_update(
        rl_loss_func_args, rl_loss_func_args_beta_index
    )
    require_valid_action_prob_times_given_if_index_supplied(
        rl_loss_func_args, rl_loss_func_args_action_prob_times_index
    )

    ### Validate action prob function and args
    require_action_probabilities_can_be_reconstructed(
        study_df,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        in_study_col_name,
        action_prob_func_filename,
        action_prob_func_args,
    )
    require_action_prob_func_args_given_for_all_users_at_each_decision(
        study_df, user_id_col_name, action_prob_func_args
    )
    require_action_prob_func_args_given_for_all_deicision_times(
        study_df, calendar_t_col_name, action_prob_func_args
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
    require_binary_actions(study_df, action_prob_col_name)
    require_binary_in_study_indicators(study_df, in_study_col_name)
    require_consecutive_integer_policy_numbers(study_df, policy_num_col_name)
    require_consecutive_integer_calendar_times(study_df, calendar_t_col_name)
    require_hashable_user_ids(study_df, user_id_col_name)
    require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name)
    verify_study_df_summary_satisfactory(study_df)

    ### Validate theta estimation
    require_theta_is_1D_array(theta_est)
    # Note that theta function can only take study df as an argument, but this
    # will have already failed by the time we get here when attempting to form
    # the theta estimate.


def require_action_probabilities_can_be_reconstructed(
    study_df,
    action_prob_col_name,
    calendar_t_col_name,
    user_id_col_name,
    in_study_col_name,
    action_prob_func_filename,
    action_prob_func_args,
):
    action_prob_func = load_function_from_same_named_file(action_prob_func_filename)

    in_study_df = study_df[study_df[in_study_col_name] == 1]
    reconstructed_action_probs = in_study_df.apply(
        lambda row: action_prob_func(
            *action_prob_func_args[row[calendar_t_col_name]][row[user_id_col_name]]
        ),
        axis=1,
    )
    try:
        assert np.testing.assert_allclose(
            in_study_df[action_prob_col_name].to_numpy(dtype="float64"),
            reconstructed_action_probs.to_numpy(dtype="float64"),
        )
    except AssertionError as e:
        # pylint: disable=bad-builtin
        answer = input(
            f"The action probabilities could not be exactly reconstructed by the function and arguments given:\n{str(e)}\n\nContinue? (y/n)\n"
        )
        # pylint: enable=bad-builtin
        if answer.lower() == "y":
            print("YES!!!")
        elif answer.lower() == "n":
            raise SystemExit from e
        else:
            print("Please enter 'y' or 'n'.")


def require_all_users_have_all_times_in_study_df(
    study_df, calendar_t_col_name, user_id_col_name
):
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


def require_rl_loss_args_given_for_all_users_at_each_update(
    study_df, user_id_col_name, rl_loss_func_args
):
    all_user_ids = set(study_df[user_id_col_name].unique())
    for policy_num in rl_loss_func_args:
        assert (
            set(rl_loss_func_args[policy_num].keys()) == all_user_ids
        ), f"Not all users present in RL loss args for olicy number {policy_num}. Please see the contract for details."


def require_action_prob_func_args_given_for_all_users_at_each_decision(
    study_df,
    user_id_col_name,
    action_prob_func_args,
):
    all_user_ids = set(study_df[user_id_col_name].unique())
    for decision_time in action_prob_func_args:
        assert (
            set(action_prob_func_args[decision_time].keys()) == all_user_ids
        ), f"Not all users present in RL loss args for olicy number {decision_time}. Please see the contract for details."


def require_action_prob_func_args_given_for_all_deicision_times(
    study_df, calendar_t_col_name, action_prob_func_args
):
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


def require_binary_actions(study_df, action_prob_col_name):
    pass


def require_binary_in_study_indicators(study_df, in_study_col_name):
    pass


def require_consecutive_integer_policy_numbers(study_df, policy_num_col_name):
    # Maybe any negative number taken to be a fallback policy, everything else
    # consecutive integers. Consecutive might not be feasible tho given app
    # opening issue.
    pass


def require_consecutive_integer_calendar_times(study_df, calendar_t_col_name):
    pass


def require_hashable_user_ids(study_df, user_id_col_name):
    pass


def require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name):
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    pass


def require_no_policy_numbers_present_in_rl_loss_args_but_not_study_df(
    study_df, policy_num_col_name, rl_loss_func_args
):
    assert set(rl_loss_func_args.keys()).issubset(
        study_df[policy_num_col_name].unique()
    ), "There are policy numbers present in RL loss args but not in the study dataframe. Please see the contract for details."


def require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_loss_args(
    study_df, policy_num_col_name, rl_loss_func_args
):
    min_positive_policy_num = study_df[study_df[policy_num_col_name] >= 0][
        policy_num_col_name
    ].min()
    assert set(
        study_df[study_df[policy_num_col_name] > min_positive_policy_num][
            policy_num_col_name
        ].unique()
    ).issubset(
        rl_loss_func_args.keys()
    ), "There are policy numbers present in RL loss args but not in the study dataframe. Please see the contract for details."


def confirm_action_probabilities_not_in_rl_loss_args_if_index_not_supplied(
    rl_loss_func_args_action_prob_index,
):
    pass


def require_action_prob_args_in_range_0_1_if_supplied(
    rl_loss_func_args, rl_loss_func_args_action_prob_index
):
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    pass


def require_action_prob_times_given_if_index_supplied(
    rl_loss_func_args,
    rl_loss_func_args_action_prob_index,
    rl_loss_func_args_action_prob_times_index,
):
    pass


def require_beta_is_1D_array_in_rl_loss_args(
    rl_loss_func_args, rl_loss_func_args_beta_index
):
    pass


def require_beta_is_1D_array_in_action_prob_args(
    action_prob_func_args, action_prob_func_args_beta_index
):
    pass


def require_theta_is_1D_array(theta_est):
    pass


def verify_study_df_summary_satisfactory(
    study_df,
):
    pass


def require_betas_match_in_rl_loss_args_each_update(
    rl_loss_func_args, rl_loss_func_args_beta_index
):
    for policy_num in rl_loss_func_args:
        first_beta = None
        for user_id in rl_loss_func_args[policy_num]:
            if not rl_loss_func_args[policy_num][user_id]:
                continue
            beta = rl_loss_func_args[policy_num][user_id][rl_loss_func_args_beta_index]
            if first_beta is None:
                first_beta = beta
            else:
                assert np.array_equal(
                    beta, first_beta
                ), f"Betas do not match across users in the RL loss args for policy number {policy_num}. Please see the contract for details."


def require_betas_match_in_action_prob_func_args_each_decision(
    action_prob_func_args, action_prob_func_args_beta_index
):
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
    rl_loss_func_args, rl_loss_func_args_action_prob_times_index
):
    # Must be strictly increasing. Possibly contiguous?
    pass
