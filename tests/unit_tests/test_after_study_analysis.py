import pandas as pd
import numpy as np
import pytest

import after_study_analysis
import calculate_derivatives


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
            "in_study": [1, 1, 1, 1, 1, 1],
        }
    )

    theta_est = np.array([1.0, 2, 3, 4])
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

    # First 8 entries are just loss gradients taken right from the stats dict.
    # Final 4 entries computed as
    # -1.5 * np.array([1, 0, -.25, 0]) + 15 * np.array([1, 1, .5, .5]) - 2.5 * np.array([1, -1, .25, -.25])
    # with coefficients from running the following inside a breakpoint in get_loss:
    # ( rewards - jnp.matmul(base_states, theta_0) - jnp.matmul(actions * treat_states, theta_1) ) * -2
    user_1_meat_vector = np.array([1, 2, 3, 4, 2, 3, 4, 5, 11.0, 17.5, 7.25, 8.125])
    user_1_meat_contribution = np.outer(user_1_meat_vector, user_1_meat_vector)

    # First 8 entries are just loss gradients taken right from the stats dict.
    # Final 4 entries computed as
    # 16.599998 * np.array([1, 1, .9, .9]) + 17.2 * np.array([1, 1, .8, .8]) - 1.8000001 * np.array([1, 0, -.3, -0])
    # with coefficients from running the following inside a breakpoint in get_loss:
    # ( rewards - jnp.matmul(base_states, theta_0) - jnp.matmul(actions * treat_states, theta_1) ) * -2
    user_2_meat_vector = np.array(
        [2, 3, 4, 5, 3, 4, 5, 6, 31.9999979, 33.799998, 29.23999823, 28.6999982]
    )
    user_2_meat_contribution = np.outer(user_2_meat_vector, user_2_meat_vector)

    expected_meat_matrix = (user_1_meat_contribution + user_2_meat_contribution) / 2

    user_ids = [1, 2]
    loss_gradients, _, _ = calculate_derivatives.calculate_inference_loss_derivatives(
        study_df,
        theta_est,
        "functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py",
        0,
        user_ids,
        "user_id",
        "action1prob",
        "in_study",
        "calendar_t",
    )
    # Correct to 5 decimal places is perfectly sufficient
    np.testing.assert_allclose(
        after_study_analysis.form_joint_adaptive_meat_matrix(
            len(theta_est),
            update_times,
            beta_dim,
            algo_stats_dict,
            user_ids,
            loss_gradients,
        ),
        expected_meat_matrix,
        rtol=1e-05,
    )


def test_form_bread_inverse_matrix_1_decision_between_updates():
    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
            "action1prob": [0.25, 0.5, 0.75, 0.1, 0.2, 0.3],
            "in_study": [1, 1, 1, 1, 1, 1],
        }
    )

    theta_est = np.array([1.0, 2, 3, 4])
    update_times = [2, 3]
    beta_dim = 4
    algo_stats_dict = {
        2: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([2, 3, 4, 5], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            # Just make this zero because it should already be encoded
            # in provided upper left matrix: make sure we're not using this
            "avg_loss_hessian": np.zeros((4, 4)),
        },
        3: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 1, 1, 1], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([1, 1, 1, 1], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([2, 3, 4, 5], dtype="float32"),
                2: np.array([3, 4, 5, 6], dtype="float32"),
            },
            # Just make this zero because it should already be encoded
            # in provided upper left matrix: make sure we're not using this
            "avg_loss_hessian": np.zeros((4, 4)),
        },
    }

    # These were computed in the test for the meat matrix!
    user_1_psi = np.array([11.0, 17.5, 7.25, 8.125])
    user_2_psi = np.array([31.9999979, 33.799998, 29.23999823, 28.6999982])

    # psi times weight gradient at time 2 for each user,  then average
    user_1_block_1_non_cross_term_contribution = np.outer(
        user_1_psi, np.array([2, 3, 4, 5])
    )
    user_2_block_1_non_cross_term_contribution = np.outer(
        user_2_psi, np.array([0, 0, 0, 0])
    )
    block_1_non_cross_term = (
        user_1_block_1_non_cross_term_contribution
        + user_2_block_1_non_cross_term_contribution
    ) / 2

    theta_1 = np.array([3, 4])

    # TODO: Construct these more clearly with functions, the numbers in the
    # second part vs variables in the first part very misleading--actually
    # incorporates the long extra term in a very sneaky way
    user_1_state_2 = np.array([1, 1])
    user_1_block_1_cross_term_derivative_wrt_pi = -2 * np.dot(
        theta_1, user_1_state_2
    ) * np.array([1, 1, (1 - 0.5) * 1, ((1 - 0.5)) * 1]) + (
        -15 * np.array([0, 0, 1, 1])
    )
    user_1_block_1_cross_term_contribution = np.outer(
        user_1_block_1_cross_term_derivative_wrt_pi, np.array([1, 2, 3, 4])
    )

    block_1_cross_term = user_1_block_1_cross_term_contribution / 2

    block_1 = block_1_non_cross_term + block_1_cross_term

    user_1_block_2_non_cross_term_contribution = np.outer(
        user_1_psi, np.array([1, 2, 3, 4])
    )
    user_2_block_2_non_cross_term_contribution = np.outer(
        user_2_psi, np.array([1, 1, 1, 1])
    )

    block_2_non_cross_term = (
        user_1_block_2_non_cross_term_contribution
        + user_2_block_2_non_cross_term_contribution
    ) / 2

    theta_1 = np.array([3, 4])

    user_1_state_3 = np.array([1, -1])

    user_1_block_2_cross_term_contribution = -2 * np.dot(
        theta_1, user_1_state_3
    ) * np.outer(
        np.array([1, -1, (1 - 0.75) * 1, (1 - 0.75) * -1]), np.array([1, 1, 1, 1])
    ) + np.outer(
        2.5 * np.array([0, 0, 1, -1]),
        np.array([1, 1, 1, 1]),
    )

    user_2_state_3 = np.array([1, 0])
    user_2_block_2_cross_term_contribution = -2 * np.dot(
        theta_1, user_2_state_3
    ) * np.outer(
        np.array([1, 0, (0 - 0.3) * 1, (0 - 0.3) * 0]), np.array([2, 3, 4, 5])
    ) + np.outer(
        1.8000001 * np.array([0, 0, 1, 0]),
        np.array([2, 3, 4, 5]),
    )

    block_2_cross_term = (
        user_1_block_2_cross_term_contribution + user_2_block_2_cross_term_contribution
    ) / 2

    block_2 = block_2_non_cross_term + block_2_cross_term

    user_1_hessian = 2 * (
        np.block(
            [
                [
                    np.outer(np.array([1, 0]), np.array([1, 0])),
                    -0.25 * np.outer(np.array([1, 0]), np.array([1, 0])),
                ],
                [
                    -0.25 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    (-0.25) ** 2 * np.outer(np.array([1, 0]), np.array([1, 0])),
                ],
            ]
        )
        + np.block(
            [
                [
                    np.outer(np.array([1, 1]), np.array([1, 1])),
                    0.5 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
                [
                    0.5 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    (0.5) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
            ]
        )
        + np.block(
            [
                [
                    np.outer(np.array([1, -1]), np.array([1, -1])),
                    0.25 * np.outer(np.array([1, -1]), np.array([1, -1])),
                ],
                [
                    0.25 * np.outer(np.array([1, -1]), np.array([1, -1])),
                    (0.25) ** 2 * np.outer(np.array([1, -1]), np.array([1, -1])),
                ],
            ]
        )
    )

    user_2_hessian = 2 * (
        np.block(
            [
                [
                    np.outer(np.array([1, 1]), np.array([1, 1])),
                    0.9 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
                [
                    0.9 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    (0.9) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
            ]
        )
        + np.block(
            [
                [
                    np.outer(np.array([1, 1]), np.array([1, 1])),
                    0.8 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
                [
                    0.8 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    (0.8) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                ],
            ]
        )
        + np.block(
            [
                [
                    np.outer(np.array([1, 0]), np.array([1, 0])),
                    -0.3 * np.outer(np.array([1, 0]), np.array([1, 0])),
                ],
                [
                    -0.3 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    (-0.3) ** 2 * np.outer(np.array([1, 0]), np.array([1, 0])),
                ],
            ]
        )
    )
    theta_hessian = (user_1_hessian + user_2_hessian) / 2

    upper_left_bread_inverse = np.array(
        [
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [08.0, 11.0, 14.0, 17.0, 1.0, 1.0, 1.0, 1.0],
            [13.0, 18.0, 23.0, 28.0, 1.0, 1.0, 1.0, 1.0],
            [18.0, 25.0, 32.0, 39.0, 1.0, 1.0, 1.0, 1.0],
            [23.0, 32.0, 41.0, 50.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype="float32",
    )

    expected_bread_inverse = np.block(
        [
            [upper_left_bread_inverse, np.zeros((8, 4))],
            [block_1, block_2, theta_hessian],
        ]
    )

    user_ids = [1, 2]
    loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
        calculate_derivatives.calculate_inference_loss_derivatives(
            study_df,
            theta_est,
            "functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py",
            0,
            user_ids,
            "user_id",
            "action1prob",
            "in_study",
            "calendar_t",
        )
    )
    np.testing.assert_allclose(
        after_study_analysis.form_joint_adaptive_bread_inverse_matrix(
            upper_left_bread_inverse,
            study_df.calendar_t.max(),
            algo_stats_dict,
            update_times,
            beta_dim,
            len(theta_est),
            user_ids,
            loss_gradients,
            loss_hessians,
            loss_gradient_pi_derivatives,
        ),
        expected_bread_inverse,
        rtol=1e-05,
    )


# TODO: Modify to make one small part not just double the previous test,
# to fully exercise the piecing together of pi derivative logic
def test_form_bread_inverse_matrix_2_decisions_between_updates():
    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
            "calendar_t": [
                1,
                2,
                3,
                4,
                5,
                6,
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "action": [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            "reward": [1.0, 1, -1, -1, 0, 0, 1, 1, 0, 0, 1, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 0, 1, 1, -1, -1, 1, 1, 1, 1, 0, 0],
            "action1prob": [
                0.25,
                0.25,
                0.5,
                0.5,
                0.75,
                0.75,
                0.1,
                0.1,
                0.2,
                0.2,
                0.3,
                0.3,
            ],
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    theta_est = np.array([1.0, 2, 3, 4])
    update_times = [3, 5]
    beta_dim = 4
    algo_stats_dict = {
        3: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([2, 3, 4, 5], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
        },
        4: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([2, 3, 4, 5], dtype="float32"),
                2: np.array([0] * 4, dtype="float32"),
            },
        },
        5: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 1, 1, 1], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([1, 1, 1, 1], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * 1,
        },
        6: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 1, 1, 1], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([1, 1, 1, 1], dtype="float32"),
            },
        },
    }

    # These are two times what was computed in the test for the meat matrix,
    # since we just doubled up data in the study df for this test
    user_1_psi = 2 * np.array([11.0, 17.5, 7.25, 8.125])
    user_2_psi = 2 * np.array([31.9999979, 33.799998, 29.23999823, 28.6999982])

    # Again multiply by 2 to account for sum of two time steps set up to have
    # same data.
    # The psi is doubled above, meaning these are blown up by four
    # relative to the above test!
    user_1_block_1_non_cross_term_contribution = 2 * np.outer(
        user_1_psi, np.array([2, 3, 4, 5])
    )
    user_2_block_1_non_cross_term_contribution = 2 * np.outer(
        user_2_psi, np.array([0, 0, 0, 0])
    )
    block_1_non_cross_term = (
        user_1_block_1_non_cross_term_contribution
        + user_2_block_1_non_cross_term_contribution
    ) / 2

    theta_1 = np.array([3, 4])

    # Again multiply by 2 to account for sum of two time steps set up to have
    # same data.
    user_1_state_3_4 = np.array([1, 1])
    user_1_block_1_cross_term_derivative_wrt_pi = 2 * (
        -2
        * np.dot(theta_1, user_1_state_3_4)
        * np.array([1, 1, (1 - 0.5) * 1, ((1 - 0.5)) * 1])
        + (-15 * np.array([0, 0, 1, 1]))
    )
    user_1_block_1_cross_term_contribution = np.outer(
        user_1_block_1_cross_term_derivative_wrt_pi, np.array([1, 2, 3, 4])
    )

    block_1_cross_term = user_1_block_1_cross_term_contribution / 2

    block_1 = block_1_non_cross_term + block_1_cross_term

    user_1_block_2_non_cross_term_contribution = 2 * np.outer(
        user_1_psi, np.array([1, 2, 3, 4])
    )
    user_2_block_2_non_cross_term_contribution = 2 * np.outer(
        user_2_psi, np.array([1, 1, 1, 1])
    )

    block_2_non_cross_term = (
        user_1_block_2_non_cross_term_contribution
        + user_2_block_2_non_cross_term_contribution
    ) / 2

    theta_1 = np.array([3, 4])

    user_1_state_4_5 = np.array([1, -1])

    user_1_block_2_cross_term_contribution = 2 * (
        -2
        * np.dot(theta_1, user_1_state_4_5)
        * np.outer(
            np.array([1, -1, (1 - 0.75) * 1, (1 - 0.75) * -1]), np.array([1, 1, 1, 1])
        )
        + np.outer(
            2.5 * np.array([0, 0, 1, -1]),
            np.array([1, 1, 1, 1]),
        )
    )

    user_2_state_4_5 = np.array([1, 0])
    user_2_block_2_cross_term_contribution = 2 * (
        -2
        * np.dot(theta_1, user_2_state_4_5)
        * np.outer(
            np.array([1, 0, (0 - 0.3) * 1, (0 - 0.3) * 0]), np.array([2, 3, 4, 5])
        )
        + np.outer(
            1.8000001 * np.array([0, 0, 1, 0]),
            np.array([2, 3, 4, 5]),
        )
    )

    block_2_cross_term = (
        user_1_block_2_cross_term_contribution + user_2_block_2_cross_term_contribution
    ) / 2

    block_2 = block_2_non_cross_term + block_2_cross_term

    # 2 x the calculation for the previous test because of the way we doubled up
    # data in the study df
    user_1_hessian = (
        2
        * 2
        * (
            np.block(
                [
                    [
                        np.outer(np.array([1, 0]), np.array([1, 0])),
                        -0.25 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    ],
                    [
                        -0.25 * np.outer(np.array([1, 0]), np.array([1, 0])),
                        (-0.25) ** 2 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    ],
                ]
            )
            + np.block(
                [
                    [
                        np.outer(np.array([1, 1]), np.array([1, 1])),
                        0.5 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                    [
                        0.5 * np.outer(np.array([1, 1]), np.array([1, 1])),
                        (0.5) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                ]
            )
            + np.block(
                [
                    [
                        np.outer(np.array([1, -1]), np.array([1, -1])),
                        0.25 * np.outer(np.array([1, -1]), np.array([1, -1])),
                    ],
                    [
                        0.25 * np.outer(np.array([1, -1]), np.array([1, -1])),
                        (0.25) ** 2 * np.outer(np.array([1, -1]), np.array([1, -1])),
                    ],
                ]
            )
        )
    )

    # 2 x the calculation for the previous test because of the way we doubled up
    # data in the study df here
    user_2_hessian = (
        2
        * 2
        * (
            np.block(
                [
                    [
                        np.outer(np.array([1, 1]), np.array([1, 1])),
                        0.9 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                    [
                        0.9 * np.outer(np.array([1, 1]), np.array([1, 1])),
                        (0.9) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                ]
            )
            + np.block(
                [
                    [
                        np.outer(np.array([1, 1]), np.array([1, 1])),
                        0.8 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                    [
                        0.8 * np.outer(np.array([1, 1]), np.array([1, 1])),
                        (0.8) ** 2 * np.outer(np.array([1, 1]), np.array([1, 1])),
                    ],
                ]
            )
            + np.block(
                [
                    [
                        np.outer(np.array([1, 0]), np.array([1, 0])),
                        -0.3 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    ],
                    [
                        -0.3 * np.outer(np.array([1, 0]), np.array([1, 0])),
                        (-0.3) ** 2 * np.outer(np.array([1, 0]), np.array([1, 0])),
                    ],
                ]
            )
        )
    )
    theta_hessian = (user_1_hessian + user_2_hessian) / 2

    upper_left_bread_inverse = np.array(
        [
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            [08.0, 11.0, 14.0, 17.0, 1.0, 1.0, 1.0, 1.0],
            [13.0, 18.0, 23.0, 28.0, 1.0, 1.0, 1.0, 1.0],
            [18.0, 25.0, 32.0, 39.0, 1.0, 1.0, 1.0, 1.0],
            [23.0, 32.0, 41.0, 50.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype="float32",
    )

    expected_bread_inverse = np.block(
        [
            [upper_left_bread_inverse, np.zeros((8, 4))],
            [block_1, block_2, theta_hessian],
        ]
    )
    user_ids = [1, 2]
    loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
        calculate_derivatives.calculate_inference_loss_derivatives(
            study_df,
            theta_est,
            "functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py",
            0,
            user_ids,
            "user_id",
            "action1prob",
            "in_study",
            "calendar_t",
        )
    )
    np.testing.assert_allclose(
        after_study_analysis.form_joint_adaptive_bread_inverse_matrix(
            upper_left_bread_inverse,
            study_df.calendar_t.max(),
            algo_stats_dict,
            update_times,
            beta_dim,
            len(theta_est),
            user_ids,
            loss_gradients,
            loss_hessians,
            loss_gradient_pi_derivatives,
        ),
        expected_bread_inverse,
        rtol=1e-05,
    )


@pytest.mark.skip(reason="Nice to have")
def test_form_bread_inverse_matrix_incremental_recruitment():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_form_bread_inverse_matrix_no_action_centering():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_adaptive_and_classical_match_steepness_0():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_analyze_dataset():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_estimate_theta():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_form_classical_sandwich():
    raise NotImplementedError()


# Note that the testing of collect_derivatives is implicit here, as it's
# called along the way in several tests above.  This reflects the fact that
# the tests were originally constructed before it existed and it just reorganizes
# the same data.  That being said, a direct unit test would be nice to add.
# UPDATE: should add test for incremental recruitment case, unless incremental
# recruitment bread test is full end-to-end style like above


def test_calculate_upper_left_bread_inverse_update_every_decision_no_action_probs_in_loss():
    algorithm_statistics_by_calendar_t = {
        2: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.zeros((4, 1)),
                2: np.zeros((4, 1)),
            },
        },
        3: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
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
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.zeros((4, 2)),
                2: np.zeros((4, 2)),
            },
        },
    }
    upper_left_bread_inverse = after_study_analysis.calculate_upper_left_bread_inverse(
        pd.DataFrame({"user_id": [1, 2]}),
        "user_id",
        4,
        algorithm_statistics_by_calendar_t,
    )
    np.testing.assert_equal(
        upper_left_bread_inverse,
        # Note that this was constructed manually by inserting the correct
        # diagonal blocks and then averaging outer products of the
        # appropriate things in the above algorithm statistics dict
        np.array(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [04.0, 06.5, 09.0, 11.5, 1.0, 1.0, 1.0, 1.0],
                [05.5, 09.0, 12.5, 16.0, 1.0, 1.0, 1.0, 1.0],
                [07.0, 11.5, 16.0, 20.5, 1.0, 1.0, 1.0, 1.0],
                [08.5, 14.0, 19.5, 25.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype="float32",
        ),
    )


def test_calculate_upper_left_bread_inverse_update_every_decision_action_probs_in_loss():
    algorithm_statistics_by_calendar_t = {
        2: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([3, 4, 5, 6], dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.array(
                    [
                        [1],
                        [3],
                        [5],
                        [7],
                    ],
                    dtype="float32",
                ),
                2: np.array(
                    [
                        [1],
                        [4],
                        [6],
                        [8],
                    ],
                    dtype="float32",
                ),
            },
        },
        3: {
            "pi_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([3, 4, 5, 6], dtype="float32"),
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
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.array(
                    [
                        [1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8],
                    ],
                    dtype="float32",
                ),
                2: np.array(
                    [
                        [1, 1],
                        [3, 4],
                        [5, 6],
                        [7, 8],
                    ],
                    dtype="float32",
                ),
            },
        },
    }

    # The new contribution to the the bottom left relative to previoius test
    # due to loss_gradient_pi_derivatives_by_user_id being nonzero
    # is average of np.array([[2, 4, 6,  8],
    #                         [4, 8, 12, 16],
    #                         [6, 12, 18, 24],
    #                         [8, 16, 24, 32]])
    # and np.array([[3, 4, 5, 6],
    #               [12, 16, 20, 24],
    #               [18, 24, 30, 36],
    #               [24, 32, 40, 48]])
    # which is
    #     np.array([[ 2.5,  4. ,  5.5,  7. ],
    #               [ 8. , 12. , 16. , 20. ],
    #               [ 12. , 18. , 24. , 30. ],
    #               [ 16. , 24. , 32. , 40. ]])
    #

    upper_left_bread_inverse = after_study_analysis.calculate_upper_left_bread_inverse(
        pd.DataFrame({"user_id": [1, 2]}),
        "user_id",
        4,
        algorithm_statistics_by_calendar_t,
    )
    np.testing.assert_equal(
        upper_left_bread_inverse,
        # Note that this was constructed manually by inserting the correct
        # diagonal blocks and then averaging outer products of the
        # appropriate things in the above algorithm statistics dict
        np.array(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [06.5, 10.5, 14.5, 18.5, 1.0, 1.0, 1.0, 1.0],
                [13.5, 21.0, 28.5, 36.0, 1.0, 1.0, 1.0, 1.0],
                [19.0, 29.5, 40.0, 50.5, 1.0, 1.0, 1.0, 1.0],
                [24.5, 38.0, 51.5, 65.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype="float32",
        ),
    )


def test_calculate_upper_left_bread_inverse_2_decs_btwn_updates_no_action_probs_in_loss():
    algorithm_statistics_by_calendar_t = {
        3: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.zeros((4, 2)),
                2: np.zeros((4, 2)),
            },
        },
        4: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([3, 4, 5, 6], dtype="float32"),
                2: np.array([4, 5, 6, 7], dtype="float32"),
            },
        },
        5: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * 1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.zeros((4, 4)),
                2: np.zeros((4, 4)),
            },
        },
        6: {
            "pi_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
        },
    }
    upper_left_bread_inverse = after_study_analysis.calculate_upper_left_bread_inverse(
        pd.DataFrame({"user_id": [1, 2]}),
        "user_id",
        4,
        algorithm_statistics_by_calendar_t,
    )
    np.testing.assert_equal(
        upper_left_bread_inverse,
        # Note that this was constructed manually by inserting the correct
        # diagonal blocks and then summing weight gradients and averaging outer
        # products of appropriate things in the above algorithm statistics dict
        np.array(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [08.0, 11.0, 14.0, 17.0, 1.0, 1.0, 1.0, 1.0],
                [13.0, 18.0, 23.0, 28.0, 1.0, 1.0, 1.0, 1.0],
                [18.0, 25.0, 32.0, 39.0, 1.0, 1.0, 1.0, 1.0],
                [23.0, 32.0, 41.0, 50.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype="float32",
        ),
    )


def test_calculate_upper_left_bread_inverse_2_decs_btwn_updates_action_probs_in_loss():
    algorithm_statistics_by_calendar_t = {
        3: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * -1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.array(
                    [
                        [1, 2],
                        [3, 4],
                        [5, 6],
                        [7, 8],
                    ],
                    dtype="float32",
                ),
                2: np.array(
                    [
                        [1, 1],
                        [3, 4],
                        [5, 6],
                        [7, 8],
                    ],
                    dtype="float32",
                ),
            },
        },
        4: {
            "pi_gradients_by_user_id": {
                1: 2 * np.ones(4, dtype="float32"),
                2: np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([3, 4, 5, 6], dtype="float32"),
                2: np.array([4, 5, 6, 7], dtype="float32"),
            },
        },
        5: {
            "pi_gradients_by_user_id": {
                1: np.ones(4, dtype="float32"),
                2: 2 * np.ones(4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "loss_gradients_by_user_id": {
                1: np.array([1, 2, 3, 4], dtype="float32"),
                2: np.array([2, 3, 4, 5], dtype="float32"),
            },
            "avg_loss_hessian": np.ones((4, 4)) * 1,
            "loss_gradient_pi_derivatives_by_user_id": {
                1: np.array(
                    [
                        [1, 2, 1, 1],
                        [3, 4, 1, 2],
                        [5, 6, 1, 1],
                        [7, 8, 2, 2],
                    ],
                    dtype="float32",
                ),
                2: np.array(
                    [
                        [1, 1, 2, 1],
                        [3, 4, 3, 2],
                        [5, 6, 1, 1],
                        [7, 8, 1, 2],
                    ],
                    dtype="float32",
                ),
            },
        },
        6: {
            "pi_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
            "weight_gradients_by_user_id": {
                1: np.array([None] * 4, dtype="float32"),
                2: np.array([None] * 4, dtype="float32"),
            },
        },
    }

    # The new contribution to the the bottom left relative to previous test
    # due to loss_gradient_pi_derivatives_by_user_id being nonzero
    # is average of np.array([[3., 3., 3., 3.],
    #                         [5., 5., 5., 5.],
    #                         [3., 3., 3., 3.],
    #                         [6., 6., 6., 6.]], dtype=float32)
    # and np.array([[3., 3., 3., 3.],
    #               [5., 5., 5., 5.],
    #               [2., 2., 2., 2.],
    #               [3., 3., 3., 3.]], dtype=float32)
    # which is
    #     np.array([[ 3,  3. ,  3,  3. ],
    #               [ 5. , 5. , 5. , 5. ],
    #               [ 2.5 , 2.5 , 2.5 , 2.5 ],
    #               [ 4.5 , 4.5 , 4.5 , 4.5 ]])

    upper_left_bread_inverse = after_study_analysis.calculate_upper_left_bread_inverse(
        pd.DataFrame({"user_id": [1, 2]}),
        "user_id",
        4,
        algorithm_statistics_by_calendar_t,
    )
    np.testing.assert_equal(
        upper_left_bread_inverse,
        # Note that this was constructed manually by inserting the correct
        # diagonal blocks and then summing weight gradients and averaging outer
        # products of appropriate things in the above algorithm statistics dict
        np.array(
            [
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [11.0, 14.0, 17.0, 20.0, 1.0, 1.0, 1.0, 1.0],
                [18.0, 23.0, 28.0, 33.0, 1.0, 1.0, 1.0, 1.0],
                [20.5, 27.5, 34.5, 41.5, 1.0, 1.0, 1.0, 1.0],
                [27.5, 36.5, 45.5, 54.5, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype="float32",
        ),
    )


def test_invert_inverse_bread_matrix_2x2_block_diagonal():
    # Test case 1: Simple 2x2 block matrix
    inverse_bread = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]]
    )
    np.testing.assert_allclose(
        after_study_analysis.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_4x4_block_diagonal():
    inverse_bread = np.array([[4, 1, 0, 0], [1, 4, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [
            [0.26666667, -0.06666667, 0, 0],
            [-0.06666667, 0.26666667, 0, 0],
            [0, 0, 0.66666667, -0.33333333],
            [0, 0, -0.33333333, 0.66666667],
        ]
    )
    np.testing.assert_allclose(
        after_study_analysis.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_6x6_block_diagonal():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 0, 0, 3, 1],
            [0, 0, 0, 0, 1, 3],
        ]
    )
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [
            [0.26666667, -0.06666667, 0, 0, 0, 0],
            [-0.06666667, 0.26666667, 0, 0, 0, 0],
            [0, 0, 0.66666667, -0.33333333, 0, 0],
            [0, 0, -0.33333333, 0.66666667, 0, 0],
            [0, 0, 0, 0, 0.375, -0.125],
            [0, 0, 0, 0, -0.125, 0.375],
        ]
    )
    np.testing.assert_allclose(
        after_study_analysis.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_6x6_block_lower_triangular():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0],
            [7, 1, 2, 1, 0, 0],
            [1, 7, 1, 2, 0, 0],
            [5, 1, 6, 1, 3, 1],
            [1, 5, 1, 6, 1, 3],
        ]
    )
    beta_dim = 2
    theta_dim = 2

    expected_bread = np.linalg.inv(inverse_bread)

    np.testing.assert_allclose(
        after_study_analysis.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        atol=1e-12,
    )


def test_invert_inverse_bread_matrix_different_beta_theta_block_lower_triangular():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0, 0],
            [7, 1, 2, 1, 0, 0, 0],
            [1, 7, 1, 2, 0, 0, 0],
            [5, 1, 6, 1, 3, 1, 4],
            [1, 5, 1, 6, 1, 3, 4],
            [8, 7, 6, 5, 9, 3, 5],
        ]
    )
    beta_dim = 2
    theta_dim = 3

    expected_bread = np.linalg.inv(inverse_bread)

    np.testing.assert_allclose(
        after_study_analysis.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        atol=1e-12,
    )


@pytest.mark.skip(reason="Need to add")
def test_calculate_upper_left_bread_inverse_incremental_recruitment(self):
    raise NotImplementedError()


@pytest.mark.skip(reason="Need to add")
def test_calculate_upper_left_bread_agnostic_to_pi_times_that_start_before_first_update(
    self,
):
    # This was a bug.  We were automatically assuming the algo statistics dict
    # started at the first time after the first update, which used to be true
    # before we generalized things.  Now it typically starts at time 1, grabbing
    # pi and weight gradients from the beginning.  We don't want to assume
    # either way. Make sure both approaches work.
    raise NotImplementedError()


@pytest.mark.skip(reason="Need to add")
def test_form_bread_inverse_agnostic_to_pi_times_that_start_before_first_update(
    self,
):
    # As above for the RL side, this was a bug for the inference side.  We were
    # automatically assuming the algo statistics dict
    # started at the first time after the first update, which used to be true
    # before we generalized things.  Now it typically starts at time 1, grabbing
    # pi and weight gradients from the beginning.  We don't want to assume
    # either way. Make sure both approaches work.
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_calculate_upper_left_bread_inverse_three_updates(self):
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_custom_column_names_respected():
    raise NotImplementedError()


@pytest.mark.skip(reason="Nice to have")
def test_collect_algorithm_statistics():
    raise NotImplementedError()
