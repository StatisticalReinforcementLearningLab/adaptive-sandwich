from typing import Literal
from unittest.mock import patch, MagicMock

import jax
import pandas as pd
import numpy as np
import pytest
import jax.numpy as jnp

import after_study_analysis
import calculate_derivatives
from constants import FunctionTypes
from helper_functions import load_function_from_same_named_file, replace_tuple_index


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


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs():
    action_prob_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_action_1_prob_pure.py"
    action_prob_func_args_beta_index = 0
    alg_update_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_least_squares_loss_rl.py"
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    inference_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py"
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1, 4: 2, 5: 3}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {1: 0, 2: 1, 3: 0, 4: 1, 5: 0},
        2: {1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
    }
    policy_num_by_decision_time_by_user_id = {
        1: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        2: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                jnp.array([-(decision_time), 2.0, 3.0, 4.0], dtype="float32"),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # TODO: these don't build up over policy nums as they would in reality. OK?
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 2.0, 3.0, 4.0], dtype="float32"),
                jnp.array(
                    [
                        [user_id, policy_num],
                        [1.0, 1.0],
                        [1.0, -1.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [user_id, policy_num],
                        [1.0, 1.0],
                        [1.0, -1.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[0.0], [1.0], [1.0]], dtype="float32"),
                jnp.array([[1.0], [-1.0], [0.0]], dtype="float32"),
                jnp.array([[0.5], [0.6], [0.7]], dtype="float32"),
                None,
                0,
            )
            for user_id in (1, 2)
        }
        for policy_num in range(2, 6)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, 0, 1, 1, 1, 1, 0, 0],
            "reward": [1.0, -1, 0, 0, 2, 2, 1, 0, 1, 3],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 0, 0, 0, 2, 1, 0, 1],
            "in_study": [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            # These action probabilities are irrelevant for weights
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = after_study_analysis.process_inference_func_args(
        inference_func_filename,
        inference_func_args_theta_index,
        study_df,
        jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
        "action1prob",
        "calendar_t",
        "user_id",
        "in_study",
    )
    inference_action_prob_decision_times_by_user_id = {}

    return (
        action_prob_func_filename,
        action_prob_func_args_beta_index,
        alg_update_func_filename,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        inference_func_filename,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        beta_index_by_policy_num,
        initial_policy_num,
        action_by_decision_time_by_user_id,
        policy_num_by_decision_time_by_user_id,
        action_prob_func_args_by_user_id_by_decision_time,
        update_func_args_by_by_user_id_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
    )


def test_construct_single_user_weighted_estimating_function_stacker_simplest(
    setup_data_two_loss_functions_no_action_probs,
):
    """
    Test that the constructed function correctly computes a weighted estimating
    function stack for each of 2 users, at least in terms of the value. For now
    this test does not test that betas and thetas are threaded in correctly to
    enable differentiation.

    This test handles the simplest case: no incremental recruitment, no use of
    action probabilities in the loss/estimating functions for algorithm updates
    or inference, and only 1 decision time between updates.

    This test also has the realistic scenario where the betas in the action
    probability arguments (and the update function args) match the betas in
    all_post_update_betas, which prodeuces the scenario where the weights all
    end up being 1.  THIS MEANS WE DON'T MULTIPLY BY WEIGHTS IN THE EXPECTED
    VALUES!

    Another test below test breaks this setup, giving different betas
    in the action probability function args vs all_post_update_betas, which
    makes the weights not one, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the denominators of the weights.
    """
    (
        action_prob_func_filename,
        action_prob_func_args_beta_index,
        alg_update_func_filename,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        inference_func_filename,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        beta_index_by_policy_num,
        initial_policy_num,
        action_by_decision_time_by_user_id,
        policy_num_by_decision_time_by_user_id,
        action_prob_func_args_by_user_id_by_decision_time,
        update_func_args_by_by_user_id_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
    ) = setup_data_two_loss_functions_no_action_probs

    stacker = (
        after_study_analysis.construct_single_user_weighted_estimating_function_stacker(
            action_prob_func_filename,
            action_prob_func_args_beta_index,
            alg_update_func_filename,
            alg_update_func_type,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            inference_func_filename,
            inference_func_type,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            beta_index_by_policy_num,
            initial_policy_num,
            action_by_decision_time_by_user_id,
            policy_num_by_decision_time_by_user_id,
            action_prob_func_args_by_user_id_by_decision_time,
            update_func_args_by_by_user_id_by_policy_num,
            inference_func_args_by_user_id,
            inference_action_prob_decision_times_by_user_id,
        )
    )

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = [
        jnp.array([-2, 2, 3, 4], dtype="float32"),
        jnp.array([-3, 2, 3, 4], dtype="float32"),
        jnp.array([-4, 2, 3, 4], dtype="float32"),
        jnp.array([-5, 2, 3, 4], dtype="float32"),
    ]

    result_1 = stacker(theta, all_post_update_betas, 1)
    result_2 = stacker(theta, all_post_update_betas, 2)

    alg_loss_func = load_function_from_same_named_file(alg_update_func_filename)
    inference_loss_func = load_function_from_same_named_file(inference_func_filename)

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_loss_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_loss_func, allow_int=True)

    # Note that we don't multiply by the weights! Therefore we test that they
    # are all 1, as they should always be in practice, with the both beta going
    # into the numerator and denominator.
    expected_result_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[2][1][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[0],
                    *update_func_args_by_by_user_id_by_policy_num[2][1][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[3][1][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[1],
                    *update_func_args_by_by_user_id_by_policy_num[3][1][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[4][1][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[2],
                    *update_func_args_by_by_user_id_by_policy_num[4][1][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[5][1][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[3],
                    *update_func_args_by_by_user_id_by_policy_num[5][1][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *(
                    *inference_func_args_by_user_id[1][
                        :inference_func_args_theta_index
                    ],
                    theta,
                    *inference_func_args_by_user_id[1][
                        inference_func_args_theta_index + 1 :
                    ],
                )
            ),
        ]
    )
    expected_result_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[2][2][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[0],
                    *update_func_args_by_by_user_id_by_policy_num[2][2][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[3][2][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[1],
                    *update_func_args_by_by_user_id_by_policy_num[3][2][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[4][2][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[2],
                    *update_func_args_by_by_user_id_by_policy_num[4][2][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            alg_estimating_func(
                *(
                    *update_func_args_by_by_user_id_by_policy_num[5][2][
                        :alg_update_func_args_beta_index
                    ],
                    all_post_update_betas[3],
                    *update_func_args_by_by_user_id_by_policy_num[5][2][
                        alg_update_func_args_beta_index + 1 :
                    ],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *(
                    *inference_func_args_by_user_id[2][
                        :inference_func_args_theta_index
                    ],
                    theta,
                    *inference_func_args_by_user_id[2][
                        inference_func_args_theta_index + 1 :
                    ],
                )
            ),
        ]
    )
    np.testing.assert_allclose(result_1, expected_result_1)
    np.testing.assert_allclose(result_2, expected_result_2)


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs_different_betas():
    action_prob_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_action_1_prob_pure.py"
    action_prob_func_args_beta_index = 0
    alg_update_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_least_squares_loss_rl.py"
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    inference_func_filename = "/Users/nowellclosser/code/adaptive-sandwich/functions_to_pass_to_analysis/get_least_squares_loss_inference_no_action_centering.py"
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1, 4: 2, 5: 3}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {1: 0, 2: 1, 3: 0, 4: 1, 5: 0},
        2: {1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
    }
    policy_num_by_decision_time_by_user_id = {
        1: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        2: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                jnp.array([-(decision_time), 17.0, 18.0, 19.0], dtype="float32"),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # TODO: these don't build up over policy nums as they would in reality. OK?
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 17.0, 18.0, 19.0], dtype="float32"),
                jnp.array(
                    [
                        [user_id, policy_num],
                        [1.0, 1.0],
                        [1.0, -1.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [user_id, policy_num],
                        [1.0, 1.0],
                        [1.0, -1.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[0.0], [1.0], [1.0]], dtype="float32"),
                jnp.array([[1.0], [-1.0], [0.0]], dtype="float32"),
                jnp.array([[0.5], [0.6], [0.7]], dtype="float32"),
                None,
                0,
            )
            for user_id in (1, 2)
        }
        for policy_num in range(2, 6)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "reward": [1.0, -1, 0, 0, 2, 2, 1, 0, 1, 3],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 0, 0, 0, 2, 1, 0, 1],
            "in_study": [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            # These action probabilities are irrelevant for weights.
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = after_study_analysis.process_inference_func_args(
        inference_func_filename,
        inference_func_args_theta_index,
        study_df,
        jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
        "action1prob",
        "calendar_t",
        "user_id",
        "in_study",
    )
    inference_action_prob_decision_times_by_user_id = {}

    return (
        action_prob_func_filename,
        action_prob_func_args_beta_index,
        alg_update_func_filename,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        inference_func_filename,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        beta_index_by_policy_num,
        initial_policy_num,
        action_by_decision_time_by_user_id,
        policy_num_by_decision_time_by_user_id,
        action_prob_func_args_by_user_id_by_decision_time,
        update_func_args_by_by_user_id_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
    )


def test_construct_single_user_weighted_estimating_function_stacker_different_betas(
    setup_data_two_loss_functions_no_action_probs_different_betas,
):
    """
    Test that the constructed function correctly computes a weighted estimating
    function stack for each of 2 users, at least in terms of the value. For now
    this test does not test that betas and thetas are threaded in correctly to
    enable differentiation.

    This test handles the simplest case: no incremental recruitment, no use of
    action probabilities in the loss/estimating functions for algorithm updates
    or inference, and only 1 decision time between updates.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the denominators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the ones in all_post_update_betas are subbed in for use in the
    estimating function evaluations.
    """
    (
        action_prob_func_filename,
        action_prob_func_args_beta_index,
        alg_update_func_filename,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        inference_func_filename,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        beta_index_by_policy_num,
        initial_policy_num,
        action_by_decision_time_by_user_id,
        policy_num_by_decision_time_by_user_id,
        action_prob_func_args_by_user_id_by_decision_time,
        update_func_args_by_by_user_id_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
    ) = setup_data_two_loss_functions_no_action_probs_different_betas

    stacker = (
        after_study_analysis.construct_single_user_weighted_estimating_function_stacker(
            action_prob_func_filename,
            action_prob_func_args_beta_index,
            alg_update_func_filename,
            alg_update_func_type,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            inference_func_filename,
            inference_func_type,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            beta_index_by_policy_num,
            initial_policy_num,
            action_by_decision_time_by_user_id,
            policy_num_by_decision_time_by_user_id,
            action_prob_func_args_by_user_id_by_decision_time,
            update_func_args_by_by_user_id_by_policy_num,
            inference_func_args_by_user_id,
            inference_action_prob_decision_times_by_user_id,
        )
    )

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = [
        jnp.array([-2, 2, 3, 4], dtype="float32"),
        jnp.array([-3, 2, 3, 4], dtype="float32"),
        jnp.array([-4, 2, 3, 4], dtype="float32"),
        jnp.array([-5, 2, 3, 4], dtype="float32"),
    ]

    result_1 = stacker(theta, all_post_update_betas, 1)
    result_2 = stacker(theta, all_post_update_betas, 2)

    action_prob_func = load_function_from_same_named_file(action_prob_func_filename)
    alg_loss_func = load_function_from_same_named_file(alg_update_func_filename)
    inference_loss_func = load_function_from_same_named_file(inference_func_filename)

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_loss_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_loss_func, allow_int=True)

    expected_result_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[1],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )
    expected_result_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[2][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][2],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * after_study_analysis.get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                ),
            )
            * inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[2],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )

    np.testing.assert_allclose(result_1, expected_result_1)
    np.testing.assert_allclose(result_2, expected_result_2, rtol=1e-6)


def test_get_radon_nikodym_weight():
    def mock_action_prob_func(beta, arg1, arg2):
        return beta[0] + arg1 + arg2

    beta_target = jnp.array([0.5, 0.2, 0.3])
    action_prob_func_args_single_user = (jnp.array([0.1, 0.2, 0.3]), 0.01, 0.02)
    action_prob_func_args_beta_index = 0
    action = 1

    expected_numerator = mock_action_prob_func(*action_prob_func_args_single_user)
    expected_denominator_args = list(action_prob_func_args_single_user)
    expected_denominator_args[action_prob_func_args_beta_index] = beta_target
    expected_denominator = mock_action_prob_func(*expected_denominator_args)

    expected_result = expected_numerator / expected_denominator

    result = after_study_analysis.get_radon_nikodym_weight(
        beta_target,
        mock_action_prob_func,
        action_prob_func_args_beta_index,
        action,
        *action_prob_func_args_single_user,
    )

    np.testing.assert_allclose(result, expected_result)


def test_get_radon_nikodym_weight_action_0():
    def mock_action_prob_func(beta, arg1, arg2):
        return beta[0] + arg1 + arg2

    beta_target = jnp.array([0.5, 0.2, 0.3])
    action_prob_func_args_single_user = (jnp.array([0.1, 0.2, 0.3]), 0.01, 0.02)
    action_prob_func_args_beta_index = 0
    action = 0

    expected_numerator = mock_action_prob_func(*action_prob_func_args_single_user)
    expected_denominator_args = list(action_prob_func_args_single_user)
    expected_denominator_args[action_prob_func_args_beta_index] = beta_target
    expected_denominator = mock_action_prob_func(*expected_denominator_args)

    expected_result = (1 - expected_numerator) / (1 - expected_denominator)
    result = after_study_analysis.get_radon_nikodym_weight(
        beta_target,
        mock_action_prob_func,
        action_prob_func_args_beta_index,
        action,
        *action_prob_func_args_single_user,
    )

    np.testing.assert_allclose(result, expected_result)


def test_get_radon_nikodym_weight_same_beta():
    def mock_action_prob_func(beta, arg1, arg2):
        return beta[0] + arg1 + arg2

    beta_target = jnp.array([0.1, 0.2, 0.3])
    action_prob_func_args_single_user = (jnp.array([0.1, 0.2, 0.3]), 0.01, 0.02)
    action_prob_func_args_beta_index = 0
    action = 1

    expected_numerator = mock_action_prob_func(*action_prob_func_args_single_user)
    expected_denominator_args = list(action_prob_func_args_single_user)
    expected_denominator_args[action_prob_func_args_beta_index] = beta_target
    expected_denominator = mock_action_prob_func(*expected_denominator_args)

    expected_result = expected_numerator / expected_denominator

    result = after_study_analysis.get_radon_nikodym_weight(
        beta_target,
        mock_action_prob_func,
        action_prob_func_args_beta_index,
        action,
        *action_prob_func_args_single_user,
    )

    np.testing.assert_allclose(result, expected_result)
