import pandas as pd
import numpy as np

import after_study_analysis


def test_estimate_theta():
    pass


def test_form_meat_matrix():
    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
            "action1prob": [0.25, 0.5, 0.75, 0.1, 0.2, 0.3],
        }
    )

    theta_est = np.array([1.0, 2, 3, 4])
    state_feats = treat_feats = ["intercept", "past_reward"]
    update_times = [2, 3]
    beta_dim = 4
    algo_stats_dict = {
        2: {
            "pi_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
        },
        3: {
            "pi_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([2, 3, 4, 5], dtype="float32"),
                2: np.array([3, 4, 5, 6], dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)),
        },
    }

    # Final 4 entries computed as
    # -1.5 * np.array([1, 0, -.25, 0]) + 15 * np.array([1, 1, .5, .5]) - 2.5 * np.array([1, -1, .25, -.25])
    # with coefficients from running the following inside a breakpoint in get_loss:
    # ( rewards - jnp.matmul(base_states, theta_0) - jnp.matmul(actions * treat_states, theta_1) ) * -2
    user_1_meat_vector = np.array([1, 2, 3, 4, 2, 3, 4, 5, 11.0, 17.5, 7.25, 8.125])
    user_1_meat_contribution = np.outer(user_1_meat_vector, user_1_meat_vector)

    # Final 4 entries computed as
    # 16.599998 * np.array([1, 1, .9, .9]) + 17.2 * np.array([1, 1, .8, .8]) - 1.8000001 * np.array([1, 0, -.3, -0])
    # with coefficients from running the following inside a breakpoint in get_loss:
    # ( rewards - jnp.matmul(base_states, theta_0) - jnp.matmul(actions * treat_states, theta_1) ) * -2
    user_2_meat_vector = np.array(
        [2, 3, 4, 5, 3, 4, 5, 6, 31.9999979, 33.799998, 29.23999823, 28.6999982]
    )
    user_2_meat_contribution = np.outer(user_2_meat_vector, user_2_meat_vector)

    expected_meat_matrix = (user_1_meat_contribution + user_2_meat_contribution) / 2

    # Correct to 5 decimal places is perfectly sufficient
    np.testing.assert_allclose(
        after_study_analysis.form_meat_matrix(
            study_df,
            theta_est,
            state_feats,
            treat_feats,
            update_times,
            beta_dim,
            algo_stats_dict,
        ),
        expected_meat_matrix,
        rtol=1e-05,
    )


def form_bread_matrix():
    pass


def test_analyze_dataset():
    pass
