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
    # TODO: Can group better, to make it easier to see what's required for each
    # input going forward.  Also, maybe this wave shouldn't include loading functions
    # supplied--do action prob reconstruction, theta estimation, estimating function sum, etc. in a later wave.
    require_action_probabilities_can_be_reconstructed(
        study_df, action_prob_col_name, action_prob_func_filename, action_prob_func_args
    )
    require_all_users_have_all_times_in_study_df(study_df, calendar_t_col_name)
    require_rl_loss_args_given_for_all_users_at_each_update(rl_loss_func_args)
    require_action_prob_func_args_given_for_all_users_at_each_decision(
        action_prob_func_args
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
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name)
    require_no_policy_numbers_present_in_rl_loss_args_but_not_study_df(
        study_df, rl_loss_func_args
    )

    # TODO: Possibly REQUIRE initial *isn't* in study df?
    require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_loss_args(
        study_df, rl_loss_func_args
    )

    confirm_action_probabilities_not_in_rl_loss_args_if_index_not_supplied(
        rl_loss_func_args_action_prob_index
    )
    # TODO: Can we even require not 0 or 1? Illustrates non-compliant RL algorithm
    require_action_prob_args_in_range_0_1_if_supplied(
        rl_loss_func_args, rl_loss_func_args_action_prob_index
    )
    require_action_prob_times_given_if_index_supplied(
        rl_loss_func_args,
        rl_loss_func_args_action_prob_index,
        rl_loss_func_args_action_prob_times_index,
    )
    require_beta_is_1D_array_in_rl_args(rl_loss_func_args, rl_loss_func_args_beta_index)
    require_beta_is_1D_array_in_action_prob_args(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    require_theta_is_1D_array(theta_est)

    verify_study_df_summary_satisfactory(study_df)

    require_betas_match_in_rl_loss_args_each_update(
        rl_loss_func_args, rl_loss_func_args_beta_index
    )
    require_betas_match_in_action_prob_func_args_each_decision(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    # Must be strictly increasing. Possibly contiguous?
    require_valid_action_prob_times_given_if_index_supplied(
        rl_loss_func_args, rl_loss_func_args_action_prob_times_index
    )


def require_action_probabilities_can_be_reconstructed(
    study_df, action_prob_col_name, action_prob_func_filename, action_prob_func_args
):
    pass


def require_all_users_have_all_times_in_study_df(study_df, calendar_t_col_name):
    pass


def require_rl_loss_args_given_for_all_users_at_each_update(rl_loss_func_args):
    pass


def require_action_prob_func_args_given_for_all_users_at_each_decision(
    action_prob_func_args,
):
    pass


def require_all_named_columns_present_in_study_df(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
):
    pass


def require_binary_actions(study_df, action_prob_col_name):
    pass


def require_binary_in_study_indicators(study_df, in_study_col_name):
    pass


def require_consecutive_integer_policy_numbers(study_df, policy_num_col_name):
    pass


def require_consecutive_integer_calendar_times(study_df, calendar_t_col_name):
    pass


def require_hashable_user_ids(study_df, user_id_col_name):
    pass


def require_action_probabilities_in_range_0_to_1(study_df, action_prob_col_name):
    pass


def require_no_policy_numbers_present_in_rl_loss_args_but_not_study_df(
    study_df, rl_loss_func_args
):
    pass


def require_all_policy_numbers_in_study_df_except_possibly_initial_and_fallback_present_in_rl_loss_args(
    study_df, rl_loss_func_args
):
    pass


def confirm_action_probabilities_not_in_rl_loss_args_if_index_not_supplied(
    rl_loss_func_args_action_prob_index,
):
    pass


def require_action_prob_args_in_range_0_1_if_supplied(
    rl_loss_func_args, rl_loss_func_args_action_prob_index
):
    pass


def require_action_prob_times_given_if_index_supplied(
    rl_loss_func_args,
    rl_loss_func_args_action_prob_index,
    rl_loss_func_args_action_prob_times_index,
):
    pass


def require_beta_is_1D_array_in_rl_args(
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
    pass


def require_betas_match_in_action_prob_func_args_each_decision(
    action_prob_func_args, action_prob_func_args_beta_index
):
    pass


def require_valid_action_prob_times_given_if_index_supplied(
    rl_loss_func_args, rl_loss_func_args_action_prob_times_index
):
    pass
