from unittest import mock

import numpy as np
import pandas as pd

import rl_algorithm
import rl_experiments
import smoothing_function


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

    np.testing.assert_array_equal(
        data_df["user_idx"].to_numpy(),
        np.repeat([0, 1, 2], total_num_decision_times),
    )
    np.testing.assert_array_equal(
        data_df["user_entry_decision_t"].to_numpy(),
        np.repeat([0, 14, 28], total_num_decision_times),
    )
    np.testing.assert_array_equal(
        data_df["user_last_decision_t"].to_numpy(),
        np.repeat([41, 55, 69], total_num_decision_times),
    )

    np.testing.assert_array_equal(
        data_df["calendar_decision_t"].to_numpy(),
        np.tile(np.arange(total_num_decision_times), num_users),
    )

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
    np.testing.assert_array_equal(
        data_df["user_idx"].to_numpy(),
        np.repeat([0, 1, 2, 3], total_num_decision_times),
    )
    np.testing.assert_array_equal(
        data_df["user_entry_decision_t"].to_numpy(),
        np.repeat([0, 28], total_num_decision_times * 2),
    )
    np.testing.assert_array_equal(
        data_df["user_last_decision_t"].to_numpy(),
        np.repeat([139, 167], total_num_decision_times * 2),
    )

    np.testing.assert_array_equal(
        data_df["calendar_decision_t"].to_numpy(),
        np.tile(np.arange(total_num_decision_times), num_users),
    )

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


EXP_SETTINGS = {
    "sim_env_version": 3,
    "base_env_type": "NON_STAT",  # This indicates non-stationarity in the environment.
    "effect_size_scale": "None",  # Don't be alarmed, this is not a parameter for the V3 algorithm
    "delayed_effect_scale": "LOW_R",
    "alg_type": "BLR_AC_V3",
    "noise_var": "None",  # Don't be alarmed, this is not a paramter for the V3 algorithm
    "clipping_vals": [0.2, 0.8],
    "b_logistic": 0.515,
    "num_decision_times_between_updates": 14,
    "cluster_size": "full_pooling",
    "cost_params": [80, 40],
    "per_user_weeks_in_study": 10,
    "num_decision_times_per_day_per_user": 2,
    "weeks_between_recruitments": 2,
}


def test_run_incremental_recruitment_exp():

    L_min, L_max = EXP_SETTINGS["clipping_vals"]
    algorithm = rl_algorithm.BlrACV3(
        EXP_SETTINGS["cost_params"],
        EXP_SETTINGS["num_decision_times_between_updates"],
        smoothing_function.generalized_logistic_func_wrapper(
            L_min,
            L_max,
            EXP_SETTINGS["b_logistic"],
        ),
    )
    template_users_list = ["user1", "user2"] * 5
    users_per_recruitment = 2
    sim_env = mock.MagicMock()
    per_user_weeks_in_study = 4
    num_users_before_update = 6
    num_decision_times_per_user_per_day = 2
    weeks_between_recruitments = 2

    def generate_current_state_side_effect(
        user_idx, user_decision_time, weeks_in_study
    ):
        return np.array([0.0, -1.01117318, 0.0, 1.0, 0.0, 1.0, -0.97101449])

    def get_user_last_open_app_dt_side_effect(user_idx):
        return 0

    def generate_rewards_side_effect(user_idx, env_state, action):
        return 84

    sim_env.generate_current_state.side_effect = generate_current_state_side_effect
    sim_env.get_user_last_open_app_dt.side_effect = (
        get_user_last_open_app_dt_side_effect
    )
    sim_env.generate_rewards.side_effect = generate_rewards_side_effect

    data_df, update_df = rl_experiments.run_incremental_recruitment_exp(
        template_users_list,
        users_per_recruitment,
        algorithm,
        sim_env,
        per_user_weeks_in_study,
        num_users_before_update,
        num_decision_times_per_user_per_day,
        weeks_between_recruitments,
    )

    expected_update_df_rows = []
    expected_update_df_rows.append(
        [0]
        + np.concatenate([algorithm.PRIOR_MU, algorithm.PRIOR_SIGMA.flatten()]).tolist()
    )
    columns = (
        ["update_idx"]
        + [f"posterior_mu.{i}" for i in range(algorithm.feature_dim)]
        + [
            f"posterior_var.{i}.{j}"
            for i in range(algorithm.feature_dim)
            for j in range(algorithm.feature_dim)
        ]
    )
    expected_update_df = pd.DataFrame(expected_update_df_rows, columns=columns)

    pd.testing.assert_frame_equal(update_df, expected_update_df)
    pd.testing.assert_frame_equal(data_df, pd.DataFrame())
