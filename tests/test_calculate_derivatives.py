import pandas as pd
import numpy as np
import pytest
from jax import numpy as jnp

import calculate_derivatives
import functions_to_pass_to_analysis.get_action_1_prob_pure
import functions_to_pass_to_analysis.get_least_squares_loss_rl


def test_calculate_pi_and_weight_gradients_specific_t_positive_action_high_clip():
    """
    At time 3, User 1 takes a positive action, meaning positive gradient case, and User 2
    gets clipped at .9, meaning zero gradient.
    """

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
            "in_study": [1, 1, 1, 1, 1, 1],
            "action1prob": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3],
        }
    )
    np.testing.assert_equal(
        calculate_derivatives.calculate_pi_and_weight_gradients_specific_t(
            study_df,
            "in_study",
            "action",
            "calendar_t",
            "user_id",
            functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure,
            0,
            3,
            {
                1: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -1.0], dtype="float32"),
                ),
                2: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, 0.0], dtype="float32"),
                ),
            },
            [1, 2],
        ),
        (
            np.array(
                [
                    # derived by setting a breakpoint and calling
                    # self.get_action_prob_pure(curr_beta_est, self.args.lower_clip, self.args.upper_clip,
                    #                           self.get_user_states(current_data, 1)["treat_states" ][-1])
                    # for each user, then plugging into explicit formula. 0.26894143 for user 1, .9 for user 2
                    np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                    # Note that these are all zeros because this probability is clipped
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
            np.array(
                [
                    # derived using pi and pi gradients from above (two derivative cases depending
                    # on action are easy to calculate)
                    np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
        ),
    )


def test_calculate_pi_and_weight_gradients_specific_t_postive_action_nonzero_gradients():

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 2, 2],
            "calendar_t": [1, 2, 1, 2],
            "action": [1, 1, 1, 1],
            "reward": [-1, 0, -1, 0],
            "intercept": [1, 1, 1, 1],
            "past_reward": [1, -1, 1, -1],
            "in_study": [1, 1, 1, 1],
        }
    )
    np.testing.assert_equal(
        calculate_derivatives.calculate_pi_and_weight_gradients_specific_t(
            study_df,
            "in_study",
            "action",
            "calendar_t",
            "user_id",
            functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure,
            0,
            2,
            {
                1: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -1.0], dtype="float32"),
                ),
                2: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -1.0], dtype="float32"),
                ),
            },
            [1, 2],
        ),
        (
            np.array(
                [
                    np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                    np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                    np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                ]
            ),
        ),
    )


def test_calculate_pi_and_weight_gradients_specific_t_zero_action_low_clip():
    """
    User 1 takes no action, meaning negative gradient case, and User 2
    gets clipped at .1, meaning zero gradient.
    """

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 0, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, -10],
            "in_study": [1, 1, 1, 1, 1, 1],
        }
    )
    np.testing.assert_equal(
        calculate_derivatives.calculate_pi_and_weight_gradients_specific_t(
            study_df,
            "in_study",
            "action",
            "calendar_t",
            "user_id",
            functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure,
            0,
            3,
            {
                1: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -1.0], dtype="float32"),
                ),
                2: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -10.0], dtype="float32"),
                ),
            },
            [1, 2],
        ),
        (
            np.array(
                [
                    np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array([0, 0, -0.26894143, 0.26894143], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
        ),
    )


def test_calculate_pi_and_weight_gradients_specific_t_incremental_recruitment_1():
    study_df = pd.DataFrame(
        {
            "user_id": [
                1,
                2,
            ],
            "calendar_t": [
                1,
                1,
            ],
            "action": [
                1,
                None,
            ],
            "reward": [
                0,
                None,
            ],
            "intercept": [1, 1],
            "past_reward": [-1, None],
            "in_study": [1, 0],
        }
    )
    np.testing.assert_equal(
        calculate_derivatives.calculate_pi_and_weight_gradients_specific_t(
            study_df,
            "in_study",
            "action",
            "calendar_t",
            "user_id",
            functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure,
            0,
            1,
            {
                1: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, -1.0], dtype="float32"),
                ),
                2: (),
            },
            [1, 2],
        ),
        (
            # User 2 not in study
            np.array(
                [
                    np.array([0, 0, 0.19661194, -0.19661194], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array([0, 0, 0.73105858, -0.73105858], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
        ),
    )


def test_calculate_pi_and_weight_gradients_specific_t_incremental_recruitment_2():

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 1, 2, 3, 4],
            "action": [0, 1, 1, None, None, 1, 1, 0],
            "reward": [1.0, -1, 0, None, None, 1, 1, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, None, None, 0, 1, 0],
            "in_study": [1, 1, 1, 0, 0, 1, 1, 1],
        }
    )
    np.testing.assert_equal(
        calculate_derivatives.calculate_pi_and_weight_gradients_specific_t(
            study_df,
            "in_study",
            "action",
            "calendar_t",
            "user_id",
            functions_to_pass_to_analysis.get_action_1_prob_pure.get_action_1_prob_pure,
            0,
            4,
            {
                # Testing another falsey value vs. ()
                1: None,
                2: (
                    jnp.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([1, 0], dtype="float32"),
                ),
            },
            [1, 2],
        ),
        (
            np.array(
                [
                    np.array([0, 0, 0, 0], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array([0, 0, 0, 0], dtype="float32"),
                    np.array([0, 0, 0, 0], dtype="float32"),
                ]
            ),
        ),
    )


def test_calculate_loss_derivatives_no_action_centering():
    """
    Study df for reference
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
            "in_study": [1, 1, 1, 1, 1, 1],
            "action1prob": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3],
        }
    )

    Note that the pi derivatives are squeezed after this to get rid of a dimension
    """
    np.testing.assert_equal(
        calculate_derivatives.calculate_rl_loss_derivatives_specific_update(
            functions_to_pass_to_analysis.get_least_squares_loss_rl.get_least_squares_loss_rl,
            0,
            5,
            6,
            {
                1: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[0.0], [1.0], [1.0]], dtype="float32"),
                    jnp.array([[1.0], [-1.0], [0.0]], dtype="float32"),
                    jnp.array([[0.5], [0.6], [0.7]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    0,
                ),
                2: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[1.0], [1.0], [0.0]], dtype="float32"),
                    jnp.array([[1.0], [0.0], [1.0]], dtype="float32"),
                    jnp.array([[0.1], [0.2], [0.3]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    0,
                ),
            },
            [1, 2],
            4,
        ),
        (
            np.array(
                [
                    np.array([6, 26, 10, 26], dtype="float32"),
                    np.array([26, 30, 30, 30], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array(
                        [
                            [6, 0, 4, 0],
                            [0, 4, 0, 4],
                            [4, 0, 4, 0],
                            [0, 4, 0, 4],
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            [6, 4, 4, 4],
                            [4, 4, 4, 4],
                            [4, 4, 4, 4],
                            [4, 4, 4, 4],
                        ],
                        dtype="float32",
                    ),
                ]
            ),
            np.zeros((2, 4, 3, 1)),
        ),
    )


def test_calculate_loss_derivatives_no_action_probs_passed_to_function():
    """
    Just like previous test, but we pretend the loss function doesn't actually
    take action probabilities to get the same zero gradients. This is extra
    artificial because we are still passing in the arg that tells the times
    that action probs correspond to, but this doesn't affect the mechanics of
    the test. Just another unused argument.

    Note that the pi derivatives are squeezed after this to get rid of a dimension
    """
    np.testing.assert_equal(
        calculate_derivatives.calculate_rl_loss_derivatives_specific_update(
            functions_to_pass_to_analysis.get_least_squares_loss_rl.get_least_squares_loss_rl,
            0,
            -1,
            -1,
            {
                1: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[0.0], [1.0], [1.0]], dtype="float32"),
                    jnp.array([[1.0], [-1.0], [0.0]], dtype="float32"),
                    jnp.array([[0.5], [0.6], [0.7]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    0,
                ),
                2: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[1.0], [1.0], [0.0]], dtype="float32"),
                    jnp.array([[1.0], [0.0], [1.0]], dtype="float32"),
                    jnp.array([[0.1], [0.2], [0.3]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    0,
                ),
            },
            [1, 2],
            4,
        ),
        (
            np.array(
                [
                    np.array([6, 26, 10, 26], dtype="float32"),
                    np.array([26, 30, 30, 30], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array(
                        [
                            [6, 0, 4, 0],
                            [0, 4, 0, 4],
                            [4, 0, 4, 0],
                            [0, 4, 0, 4],
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            [6, 4, 4, 4],
                            [4, 4, 4, 4],
                            [4, 4, 4, 4],
                            [4, 4, 4, 4],
                        ],
                        dtype="float32",
                    ),
                ]
            ),
            np.zeros((2, 4, 3, 1)),
        ),
    )


def test_calculate_loss_derivatives_action_centering():
    """
    Study df for reference (look at first two times)
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2],
            "calendar_t": [1, 2, 3, 1, 2, 3],
            "action": [0, 1, 1, 1, 1, 0],
            "reward": [1.0, -1, 0, 1, 0, 1],
            "intercept": [1.0, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 1, 1, 0],
            "in_study": [1, 1, 1, 1, 1, 1],
            "action1prob": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3],
        }
    )

    Note that the pi derivatives are squeezed after this to get rid of a dimension
    """

    beta = np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32")

    user_1_centered_actions = np.array([0 - 0.5, 1 - 0.6, 1 - 0.7], dtype="float32")
    user_1_states = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]], dtype="float32")
    user_1_rewards = np.array([1.0, -1, 0], dtype="float32")
    user_1_loss_gradient = -2 * sum(
        (
            (
                user_1_rewards[i]
                - beta[:2] @ user_1_states[i]
                - beta[2:] @ (user_1_centered_actions[i] * user_1_states[i])
            )
            * np.concatenate(
                [
                    user_1_states[i],
                    user_1_centered_actions[i] * user_1_states[i],
                ]
            )
            for i in range(3)
        )
    )

    user_2_centered_actions = np.array([1 - 0.1, 1 - 0.2, 0 - 0.3], dtype="float32")
    user_2_states = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype="float32")
    user_2_rewards = np.array([1.0, 0, 1.0], dtype="float32")
    user_2_loss_gradient = -2 * sum(
        [
            (
                user_2_rewards[i]
                - beta[:2] @ user_2_states[i]
                - beta[2:] @ (user_2_centered_actions[i] * user_2_states[i])
            )
            * np.concatenate(
                [
                    user_2_states[i],
                    user_2_centered_actions[i] * user_2_states[i],
                ]
            )
            for i in range(3)
        ]
    )

    # There are small numerical differences between the above calculations
    # and the real results. Assert they are close here and then just use
    # the real results nested in the algorithm stats dict below
    # instead of ironing out floating point issues
    np.testing.assert_allclose(
        user_1_loss_gradient,
        np.array([-4.0000005, 16.199999, 5.3599997, 5.8199997], dtype="float32"),
        atol=1e-05,
    )
    np.testing.assert_allclose(
        user_2_loss_gradient,
        np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
        atol=1e-05,
    )

    # This is a little lazy but the loss gradients match up, which suggests action
    # centering is being incorporated correctly into the loss, and I
    # simply took the hessian and pi derivatives being computed and
    # use them as the expected values because the code is behaving correctly
    # in simulations and the addition of action centering doesn't add
    # further difficulties to the JAX gradient infrastructure.
    np.testing.assert_equal(
        calculate_derivatives.calculate_rl_loss_derivatives_specific_update(
            functions_to_pass_to_analysis.get_least_squares_loss_rl.get_least_squares_loss_rl,
            0,
            5,
            6,
            {
                1: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 0],
                            [1.0, 1.0],
                            [1.0, -1.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[0.0], [1.0], [1.0]], dtype="float32"),
                    jnp.array([[1.0], [-1.0], [0.0]], dtype="float32"),
                    jnp.array([[0.5], [0.6], [0.7]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    1,
                ),
                2: (
                    np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array(
                        [
                            [1.0, 1.0],
                            [1.0, 1.0],
                            [1.0, 0.0],
                        ],
                        dtype="float32",
                    ),
                    jnp.array([[1.0], [1.0], [0.0]], dtype="float32"),
                    jnp.array([[1.0], [0.0], [1.0]], dtype="float32"),
                    jnp.array([[0.1], [0.2], [0.3]], dtype="float32"),
                    jnp.array([[1], [2], [3]], dtype="int32"),
                    1,
                ),
            },
            [1, 2],
            4,
        ),
        (
            np.array(
                [
                    np.array(
                        [-4.0000005, 16.199999, 5.3599997, 5.8199997], dtype="float32"
                    ),
                    np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
                ]
            ),
            np.array(
                [
                    np.array(
                        [
                            [6.0, 0, 0.39999998, 0.19999993],
                            [0, 4.0, 0.19999993, 1.4],
                            [0.39999998, 0.19999993, 0.99999994, 0.13999996],
                            [0.19999993, 1.4, 0.13999996, 0.49999997],
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            [6.0, 4.0, 2.8000002, 3.4],
                            [4.0, 4.0, 3.4, 3.4],
                            [2.8000002, 3.4, 3.08, 2.8999999],
                            [3.4, 3.4, 2.8999999, 2.8999999],
                        ],
                        dtype="float32",
                    ),
                ]
            ),
            np.array(
                [
                    [
                        [[-6.0], [-14.0], [2.0]],
                        [[-0.0], [-14.0], [-2.0]],
                        [[10.0], [-15.199999], [7.2]],
                        [[-0.0], [-15.199999], [-7.2]],
                    ],
                    [
                        [[-14.0], [-14.0], [-6.0]],
                        [[-14.0], [-14.0], [-0.0]],
                        [[-25.2], [-24.4], [7.6000004]],
                        [[-25.199999], [-24.400002], [-0.0]],
                    ],
                ],
                dtype="float32",
            ),
        ),
    )


@pytest.mark.skip(reason="Nice to have")
def test_calculate_loss_derivatives_incremental_recruitment():
    raise NotImplementedError()


# Especially to exercise stacking function and get_batched_actions and dict
# formation
@pytest.mark.skip(reason="Nice to have")
def test_calculate_pi_and_weight_gradients_all_times():
    raise NotImplementedError()


# One for each of pi and loss?
@pytest.mark.skip(reason="Need to add")
def test_calculate_gradients_equal_with_and_without_zero_padding():
    raise NotImplementedError()


# This is tested indirectly through the after study analysis test
# for inference side
@pytest.mark.skip(reason="Nice to have")
def test_inference_loss_derivatives():
    raise NotImplementedError()


# Especially to exercise stacking function and get_first_applicable_time and
# squeeze of pi gradients and rest of dict formation
@pytest.mark.skip(reason="Nice to have")
def test_calculate_loss_derivatives_all_updates():
    raise NotImplementedError()


# Indirectly tested by testing its gradient is well-formed... but could add something direct.
@pytest.mark.skip(reason="Not sure if we actually need")
def test_get_radon_nikodym_weight():
    pass
