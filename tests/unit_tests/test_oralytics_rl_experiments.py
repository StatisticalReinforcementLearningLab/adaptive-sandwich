import rl_experiments


def test_create_base_data_df_1():
    template_users_list = ["user1", "user2", "user1"]
    per_user_weeks_in_study = 3
    users_per_recruitment = 1
    num_decision_times_per_user_per_day = 2
    weeks_between_recruitments = 1

    data_df = rl_experiments.create_base_data_df(
        template_users_list,
        per_user_weeks_in_study,
        users_per_recruitment,
        num_decision_times_per_user_per_day,
        weeks_between_recruitments,
    )

    # Check the shape of the DataFrame
    num_users = len(template_users_list)
    num_decision_times_per_user = (
        per_user_weeks_in_study * 7 * num_decision_times_per_user_per_day
    )

    last_entry_week = 2
    total_num_decision_times = (
        last_entry_week * 7 * num_decision_times_per_user_per_day
        + num_decision_times_per_user
    )

    # Should have all decision times for each user!
    assert data_df.shape[0] == total_num_decision_times * num_users

    # Check the columns of the DataFrame
    expected_columns = [
        "user_idx",
        "template_user_id",
        "user_entry_decision_t",
        "user_last_decision_t",
        "calendar_decision_t",
        "policy_idx",
        "action",
        "prob",
        "reward",
        "quality",
        "state.tod",
        "state.b.bar",
        "state.a.bar",
        "state.app.engage",
        "state.bias",
        "in_study",
    ]
    assert list(data_df.columns) == expected_columns

    # Check the values in the DataFrame
    assert data_df["user_idx"].nunique() == num_users
    assert data_df["user_entry_decision_t"].nunique() == last_entry_week + 1
    assert data_df["user_last_decision_t"].nunique() == last_entry_week + 1
    assert data_df["calendar_decision_t"].nunique() == total_num_decision_times

    # Check that the in_study column is correctly populated
    for user_idx in range(num_users):
        user_data = data_df[data_df["user_idx"] == user_idx]
        entry_time = user_data["user_entry_decision_t"].iloc[0]
        last_time = user_data["user_last_decision_t"].iloc[0]
        assert all(
            user_data.loc[
                (user_data["calendar_decision_t"] >= entry_time)
                & (user_data["calendar_decision_t"] <= last_time),
                "in_study",
            ]
            == 1
        )
        assert all(
            user_data.loc[
                (user_data["calendar_decision_t"] < entry_time)
                | (user_data["calendar_decision_t"] > last_time),
                "in_study",
            ]
            == 0
        )

    # Check that the FILL_IN_COLS are filled with NaN
    for col in [
        "policy_idx",
        "action",
        "prob",
        "reward",
        "quality",
        "state.tod",
        "state.b.bar",
        "state.a.bar",
        "state.app.engage",
        "state.bias",
    ]:
        assert data_df[col].isna().all()


def test_create_base_data_df_2():
    template_users_list = ["user1", "user2", "user1", "user3"]
    per_user_weeks_in_study = 10
    users_per_recruitment = 2
    num_decision_times_per_user_per_day = 2
    weeks_between_recruitments = 2

    data_df = rl_experiments.create_base_data_df(
        template_users_list,
        per_user_weeks_in_study,
        users_per_recruitment,
        num_decision_times_per_user_per_day,
        weeks_between_recruitments,
    )

    # Check the shape of the DataFrame
    num_users = len(template_users_list)
    num_decision_times_per_user = (
        per_user_weeks_in_study * 7 * num_decision_times_per_user_per_day
    )

    last_entry_week = 2
    total_num_decision_times = (
        last_entry_week * 7 * num_decision_times_per_user_per_day
        + num_decision_times_per_user
    )

    # Should have all decision times for each user!
    assert data_df.shape[0] == total_num_decision_times * num_users

    # Check the columns of the DataFrame
    expected_columns = [
        "user_idx",
        "template_user_id",
        "user_entry_decision_t",
        "user_last_decision_t",
        "calendar_decision_t",
        "policy_idx",
        "action",
        "prob",
        "reward",
        "quality",
        "state.tod",
        "state.b.bar",
        "state.a.bar",
        "state.app.engage",
        "state.bias",
        "in_study",
    ]
    assert list(data_df.columns) == expected_columns

    # Check the values in the DataFrame
    assert data_df["user_idx"].nunique() == num_users
    assert (
        data_df["user_entry_decision_t"].nunique()
        == last_entry_week / weeks_between_recruitments + 1
    )
    assert (
        data_df["user_last_decision_t"].nunique()
        == last_entry_week / weeks_between_recruitments + 1
    )
    assert data_df["calendar_decision_t"].nunique() == total_num_decision_times

    # Check that the in_study column is correctly populated
    for user_idx in range(num_users):
        user_data = data_df[data_df["user_idx"] == user_idx]
        entry_time = user_data["user_entry_decision_t"].iloc[0]
        last_time = user_data["user_last_decision_t"].iloc[0]
        assert all(
            user_data.loc[
                (user_data["calendar_decision_t"] >= entry_time)
                & (user_data["calendar_decision_t"] <= last_time),
                "in_study",
            ]
            == 1
        )
        assert all(
            user_data.loc[
                (user_data["calendar_decision_t"] < entry_time)
                | (user_data["calendar_decision_t"] > last_time),
                "in_study",
            ]
            == 0
        )

    # Check that the FILL_IN_COLS are filled with NaN
    for col in [
        "policy_idx",
        "action",
        "prob",
        "reward",
        "quality",
        "state.tod",
        "state.b.bar",
        "state.a.bar",
        "state.app.engage",
        "state.bias",
    ]:
        assert data_df[col].isna().all()
