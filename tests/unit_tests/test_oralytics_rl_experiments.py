from unittest import mock

import numpy as np
import pandas as pd

import rl_algorithm
import rl_experiments
import smoothing_function
from tests.unit_tests.test_utils import perform_bayesian_linear_regression


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


def test_run_incremental_recruitment_exp():
    """
    Test the function that contains essentially all important logic for oralytics
    simulator.

    Finding the right level of granularity for the test is a bit difficult, with
    so much going on.  I've chosen to largely assume the data_df is correct
    and check that the updates happened properly and analysis function arguments were collected
    correctly conditional on that.

    This enables a larger, more realistic test scenario and avoids having to mock out all the
    and state and reward generation functions.

    We may test the formation of data_df in small, separate test.
    """

    L_min, L_max = [0.2, 0.8]
    algorithm = rl_algorithm.BlrACV3(
        [80, 40],
        14,
        smoothing_function.generalized_logistic_func_wrapper(
            L_min,
            L_max,
            0.515,
        ),
    )
    template_users_list = ["user1", "user2"] * 5
    users_per_recruitment = 2
    sim_env = mock.MagicMock()
    per_user_weeks_in_study = 4
    num_users_before_update = 6
    num_decision_times_per_user_per_day = 2
    weeks_between_recruitments = 2

    # Week  0  1  2  3  4  5  6  7  8  9  10 11
    # Users 1  1  1  1  3  3  5  5  7  7  9  9
    #       2  2  2  2  4  4  6  6  8  8  10 10
    #             3  3  5  5  7  7  9  9
    #             4  4  6  6  8  8  10 10

    # First update should happen after week 4 and then weekly afterward,
    # for a total of 8 updates since we don't update after the last week.

    (
        data_df,
        update_df,
        study_df,
        alg_update_function_args,
        action_prob_function_args,
    ) = rl_experiments.run_incremental_recruitment_exp(
        template_users_list,
        users_per_recruitment,
        algorithm,
        sim_env,
        per_user_weeks_in_study,
        num_users_before_update,
        num_decision_times_per_user_per_day,
        weeks_between_recruitments,
    )

    expected_update_df_rows = [
        [0]
        + np.concatenate([algorithm.PRIOR_MU, algorithm.PRIOR_SIGMA.flatten()]).tolist()
    ]

    for policy_idx in range(1, 9):
        regression_features, rewards = generate_features_and_reward_for_update(
            data_df, policy_idx
        )
        posterior_mean, posterior_var = perform_bayesian_linear_regression(
            prior_mean=algorithm.PRIOR_MU,
            prior_variance=algorithm.PRIOR_SIGMA,
            features=regression_features,
            target=rewards,
            noise_variance=algorithm.SIGMA_N_2,
        )
        expected_update_df_rows.append(
            [policy_idx]
            + np.concatenate([posterior_mean, posterior_var.flatten()]).tolist()
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
    pd.testing.assert_frame_equal(study_df, pd.DataFrame())
    np.testing.assert_allclose(alg_update_function_args, {})
    np.testing.assert_allclose(action_prob_function_args, {})


def generate_features_and_reward_for_update(
    data_df: pd.DataFrame, policy_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates the features for an update based on the resulting policy index and the study data.

    Args:
        data_df (pd.DataFrame): The data_df for the whole study.
        policy_idx (int): The policy index resulting from the update under consideration.

    Returns:
        tuple[np.ndarray, np.ndarray]: The generated features and rewards going into this update.
    """

    feature_cols = [
        "state.tod",
        "state.b.bar",
        "state.a.bar",
        "state.app.engage",
        "state.bias",
    ]

    base_features = data_df[
        (data_df["policy_idx"] < policy_idx) & (data_df["in_study"] == 1)
    ][feature_cols].to_numpy()

    actions = (
        data_df[(data_df["policy_idx"] < policy_idx) & (data_df["in_study"] == 1)][
            "action"
        ]
        .to_numpy()
        .reshape(-1, 1)
    )

    action_probs = (
        data_df[(data_df["policy_idx"] < policy_idx) & (data_df["in_study"] == 1)][
            "prob"
        ]
        .to_numpy()
        .reshape(-1, 1)
    )

    rewards = (
        data_df[(data_df["policy_idx"] < policy_idx) & (data_df["in_study"] == 1)][
            "reward"
        ]
        .to_numpy()
        .reshape(-1, 1)
    )

    return (
        np.hstack(
            [
                base_features,
                action_probs * base_features,
                (actions - action_probs) * base_features,
            ]
        ),
        rewards,
    )
