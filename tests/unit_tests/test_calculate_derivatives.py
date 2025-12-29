import pandas as pd
import numpy as np
from jax import numpy as jnp

from lifejacket import calculate_derivatives
import simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure
import simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_action_centering
import simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl


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
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure,
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
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure,
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
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure,
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


def test_calculate_pi_and_weight_gradients_specific_t_out_of_study_1():
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
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure,
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


def test_calculate_pi_and_weight_gradients_specific_t_out_of_study_2():

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
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure.synthetic_get_action_1_prob_pure,
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
            # User 1 not in study, User 2 clipped
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


def test_calculate_rl_update_derivatives_specific_update_no_action_centering():
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
        calculate_derivatives.calculate_rl_update_derivatives_specific_update(
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
            "loss",
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


def test_calculate_rl_update_derivatives_specific_update_no_action_probs_passed_to_function():
    """
    Just like previous test, but we pretend the loss function doesn't actually
    take action probabilities to get the same zero gradients. This is extra
    artificial because we are still passing in the arg that tells the times
    that action probs correspond to, but this doesn't affect the mechanics of
    the test. Just another unused argument.

    Note that the pi derivatives are squeezed after this to get rid of a dimension
    """
    np.testing.assert_equal(
        calculate_derivatives.calculate_rl_update_derivatives_specific_update(
            simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
            "loss",
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


def test_calculate_rl_update_derivatives_specific_update_action_centering():
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

    result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
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
    )
    np.testing.assert_allclose(
        result[0],
        np.array(
            [
                np.array(
                    [-4.0000005, 16.199999, 5.3599997, 5.8199997], dtype="float32"
                ),
                np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
            ]
        ),
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        result[1],
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
        rtol=1e-07,
    )
    np.testing.assert_allclose(
        result[2],
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
        rtol=1e-07,
    )


def test_calculate_rl_update_derivatives_specific_update_with_and_without_zero_padding():
    """
    Study df for reference
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, None, None, None, 1, 1, 0, None],
            "reward": [1.0, -1, 0, None, None, None, 1, 0, 1, None],
            "intercept": [1.0, 1, 1, None, None, None, 1, 1, 1, None],
            "past_reward": [0.0, 1, -1, None, None, None, 1, 1, 0, None],
            "in_study": [1, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            "action1prob": [0.5, 0.6, 0.7, None, None, None, 0.1, 0.2, 0.3, None],
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
    expected_result = (
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
                    [[-6.0], [-14.0], [2.0], [0], [0]],
                    [[-0.0], [-14.0], [-2.0], [0], [0]],
                    [[10.0], [-15.199999], [7.2], [0], [0]],
                    [[-0.0], [-15.199999], [-7.2], [0], [0]],
                ],
                [
                    [[0], [-14.0], [-14.0], [-6.0], [0]],
                    [[0], [-14.0], [-14.0], [-0.0], [0]],
                    [[0], [-25.2], [-24.4], [7.6000004], [0]],
                    [[0], [-25.199999], [-24.400002], [-0.0], [0]],
                ],
            ],
            dtype="float32",
        ),
    )

    # Pass the data in from the above dataframe by just passing in in-study data,
    # no padding, and get the resulting derivatives.
    non_zero_padded_result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
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
                jnp.array([[2], [3], [4]], dtype="int32"),
                1,
            ),
        },
        [1, 2],
        6,
    )
    np.testing.assert_allclose(
        non_zero_padded_result[0], expected_result[0], rtol=1e-07
    )
    np.testing.assert_allclose(
        non_zero_padded_result[1], expected_result[1], rtol=1e-07
    )
    np.testing.assert_allclose(
        non_zero_padded_result[2], expected_result[2], rtol=1e-07
    )

    assert len(non_zero_padded_result) == len(expected_result)

    # Pass the data in from the above dataframe by padding out of study values with zeros
    zero_padded_result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
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
                        [0, 0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [1.0, 0],
                        [1.0, 1.0],
                        [1.0, -1.0],
                        [0, 0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [1.0],
                        [-1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.5],
                        [0.6],
                        [0.7],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[1], [2], [3], [4], [5]], dtype="int32"),
                1,
            ),
            2: (
                np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                jnp.array(
                    [
                        [0, 0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0, 0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [0.0],
                        [1.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [0.1],
                        [0.2],
                        [0.3],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[1], [2], [3], [4], [5]], dtype="int32"),
                1,
            ),
        },
        [1, 2],
        6,
    )

    np.testing.assert_allclose(zero_padded_result[0], expected_result[0], rtol=1e-07)
    np.testing.assert_allclose(zero_padded_result[1], expected_result[1], rtol=1e-07)
    np.testing.assert_allclose(zero_padded_result[2], expected_result[2], rtol=1e-07)

    assert len(zero_padded_result) == len(expected_result)

    # NOTE: np isn't able to do the following comparison for some reason:
    # np.testing.assert_equal(non_zero_padded_result, zero_padded_result)


def test_calculate_rl_update_derivatives_specific_update_action_centering_incremental_recruitment_with_and_without_zero_padding_multiple_size_groups():
    """
    Study df for reference
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, 0, None, None, 1, 1, 0, None],
            "reward": [1.0, -1, 0, 0, None, None, 1, 0, 1, None],
            "intercept": [1.0, 1, 1, 0, None, None, 1, 1, 1, None],
            "past_reward": [0.0, 1, -1, 0, None, None, 1, 1, 0, None],
            "in_study": [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            "action1prob": [0.5, 0.6, 0.7, 0, None, None, 0.1, 0.2, 0.3, None],
        }
    )

    Note that the pi derivatives are squeezed after this to get rid of a dimension
    """

    beta = np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32")

    user_1_centered_actions = np.array(
        [0 - 0.5, 1 - 0.6, 1 - 0.7, 0 - 0], dtype="float32"
    )
    user_1_states = np.array(
        [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0, 0]], dtype="float32"
    )
    user_1_rewards = np.array([1.0, -1, 0, 0], dtype="float32")
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
            for i in range(4)
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
    expected_result = (
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
                    [[-6.0], [-14.0], [2.0], [0], [0]],
                    [[-0.0], [-14.0], [-2.0], [0], [0]],
                    [[10.0], [-15.199999], [7.2], [0], [0]],
                    [[-0.0], [-15.199999], [-7.2], [0], [0]],
                ],
                [
                    [[0], [-14.0], [-14.0], [-6.0], [0]],
                    [[0], [-14.0], [-14.0], [-0.0], [0]],
                    [[0], [-25.2], [-24.4], [7.6000004], [0]],
                    [[0], [-25.199999], [-24.400002], [-0.0], [0]],
                ],
            ],
            dtype="float32",
        ),
    )

    # Pass the data in from the above dataframe by just passing in in-study data,
    # no padding, and get the resulting derivatives.
    non_zero_padded_result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
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
                jnp.array([[2], [3], [4]], dtype="int32"),
                1,
            ),
        },
        [1, 2],
        6,
    )
    np.testing.assert_allclose(
        non_zero_padded_result[0], expected_result[0], rtol=1e-07
    )
    np.testing.assert_allclose(
        non_zero_padded_result[1], expected_result[1], rtol=1e-07
    )
    np.testing.assert_allclose(
        non_zero_padded_result[2], expected_result[2], rtol=1e-07
    )

    assert len(non_zero_padded_result) == len(expected_result)

    # Pass the data in from the above dataframe by padding out of study values with zeros
    zero_padded_result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
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
                        [0, 0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [1.0, 0],
                        [1.0, 1.0],
                        [1.0, -1.0],
                        [0, 0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [1.0],
                        [-1.0],
                        [0.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.5],
                        [0.6],
                        [0.7],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[1], [2], [3], [4], [5]], dtype="int32"),
                1,
            ),
            2: (
                np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32"),
                jnp.array(
                    [
                        [0, 0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0, 0],
                        [1.0, 1.0],
                        [1.0, 1.0],
                        [1.0, 0.0],
                        [0, 0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [1.0],
                        [0.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [1.0],
                        [0.0],
                        [1.0],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [0.0],
                        [0.1],
                        [0.2],
                        [0.3],
                        [0.0],
                    ],
                    dtype="float32",
                ),
                jnp.array([[1], [2], [3], [4], [5]], dtype="int32"),
                1,
            ),
        },
        [1, 2],
        6,
    )

    np.testing.assert_allclose(zero_padded_result[0], expected_result[0], rtol=1e-07)
    np.testing.assert_allclose(zero_padded_result[1], expected_result[1], rtol=1e-07)
    np.testing.assert_allclose(zero_padded_result[2], expected_result[2], rtol=1e-07)

    assert len(zero_padded_result) == len(expected_result)


def test_calculate_rl_update_derivatives_multiple_size_groups_real_bug_case():
    """
    The bug was occurring because instead of calling the padding function with
    the list of all user ids in the update, it was being called with just the
    list from the final size group (involved_user_ids from previous loop rather
    than built up all_involved_user_ids). Thus we were ending up with only
    one batch size worth of nonzero gradients after padding.

    So the main thing here is that we have forty nonzero loss gradients, not
    twenty.

    Real data taken from integration test 1, which is taken from a known-working
    branch.
    """

    result = calculate_derivatives.calculate_rl_update_derivatives_specific_update(
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl.synthetic_get_least_squares_loss_rl,
        "loss",
        0,
        5,
        6,
        {
            1: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.53085107], [1.0, 2.6111038]], dtype=np.float32),
                jnp.array([[1.0, -0.53085107], [1.0, 2.6111038]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[2.6111038], [1.4670815]], dtype=np.float32),
                jnp.array([[0.5], [0.1]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            2: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.30415758], [1.0, 0.6134206]], dtype=np.float32),
                jnp.array([[1.0, -0.30415758], [1.0, 0.6134206]], dtype=np.float32),
                jnp.array([[0.0], [0.0]], dtype=np.float32),
                jnp.array([[0.6134206], [-0.5266738]], dtype=np.float32),
                jnp.array([[0.5], [0.2937715]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            3: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.73139846], [1.0, -0.25626886]], dtype=np.float32),
                jnp.array([[1.0, -0.73139846], [1.0, -0.25626886]], dtype=np.float32),
                jnp.array([[0.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.25626886], [1.0295249]], dtype=np.float32),
                jnp.array([[0.5], [0.8897829]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            4: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.4485331], [1.0, -0.14482214]], dtype=np.float32),
                jnp.array([[1.0, -0.4485331], [1.0, -0.14482214]], dtype=np.float32),
                jnp.array([[1.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.14482214], [-0.30056202]], dtype=np.float32),
                jnp.array([[0.5], [0.8466402]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            5: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.01758752], [1.0, 1.0545937]], dtype=np.float32),
                jnp.array([[1.0, 0.01758752], [1.0, 1.0545937]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[1.0545937], [1.5804569]], dtype=np.float32),
                jnp.array([[0.5], [0.1]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            6: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.11458831], [1.0, 0.6334159]], dtype=np.float32),
                jnp.array([[1.0, -0.11458831], [1.0, 0.6334159]], dtype=np.float32),
                jnp.array([[1.0], [1.0]], dtype=np.float32),
                jnp.array([[0.6334159], [0.57918423]], dtype=np.float32),
                jnp.array([[0.5], [0.27982682]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            7: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.31894255], [1.0, 0.16913368]], dtype=np.float32),
                jnp.array([[1.0, -0.31894255], [1.0, 0.16913368]], dtype=np.float32),
                jnp.array([[1.0], [1.0]], dtype=np.float32),
                jnp.array([[0.16913368], [-0.9710794]], dtype=np.float32),
                jnp.array([[0.5], [0.6542769]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            8: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.86235404], [1.0, 0.40211704]], dtype=np.float32),
                jnp.array([[1.0, 0.86235404], [1.0, 0.40211704]], dtype=np.float32),
                jnp.array([[0.0], [1.0]], dtype=np.float32),
                jnp.array([[0.40211704], [-0.02794904]], dtype=np.float32),
                jnp.array([[0.5], [0.46093318]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            9: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.5527733], [1.0, 0.6505193]], dtype=np.float32),
                jnp.array([[1.0, 0.5527733], [1.0, 0.6505193]], dtype=np.float32),
                jnp.array([[0.0], [0.0]], dtype=np.float32),
                jnp.array([[0.6505193], [0.29432282]], dtype=np.float32),
                jnp.array([[0.5], [0.26822558]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            10: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.741625], [1.0, 0.8209995]], dtype=np.float32),
                jnp.array([[1.0, 0.741625], [1.0, 0.8209995]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[0.8209995], [0.9827795]], dtype=np.float32),
                jnp.array([[0.5], [0.17009057]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            11: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.6354397], [1.0, -1.1125957]], dtype=np.float32),
                jnp.array([[1.0, 0.6354397], [1.0, -1.1125957]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[-1.1125957], [-1.0791196]], dtype=np.float32),
                jnp.array([[0.5], [0.9]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            12: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.22552085], [1.0, 0.29941097]], dtype=np.float32),
                jnp.array([[1.0, -0.22552085], [1.0, 0.29941097]], dtype=np.float32),
                jnp.array([[0.0], [1.0]], dtype=np.float32),
                jnp.array([[0.29941097], [-0.25508976]], dtype=np.float32),
                jnp.array([[0.5], [0.5482603]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            13: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.656753], [1.0, 0.5994951]], dtype=np.float32),
                jnp.array([[1.0, 0.656753], [1.0, 0.5994951]], dtype=np.float32),
                jnp.array([[0.0], [0.0]], dtype=np.float32),
                jnp.array([[0.5994951], [0.18607414]], dtype=np.float32),
                jnp.array([[0.5], [0.303719]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            14: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.5498267], [1.0, -0.6448214]], dtype=np.float32),
                jnp.array([[1.0, -0.5498267], [1.0, -0.6448214]], dtype=np.float32),
                jnp.array([[0.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.6448214], [0.72349685]], dtype=np.float32),
                jnp.array([[0.5], [0.9]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            15: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 1.0223292], [1.0, -0.04148133]], dtype=np.float32),
                jnp.array([[1.0, 1.0223292], [1.0, -0.04148133]], dtype=np.float32),
                jnp.array([[0.0], [0.0]], dtype=np.float32),
                jnp.array([[-0.04148133], [-0.3352064]], dtype=np.float32),
                jnp.array([[0.5], [0.7951243]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            16: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.59958464], [1.0, -0.41987574]], dtype=np.float32),
                jnp.array([[1.0, 0.59958464], [1.0, -0.41987574]], dtype=np.float32),
                jnp.array([[1.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.41987574], [-0.9026403]], dtype=np.float32),
                jnp.array([[0.5], [0.9]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            17: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.10662533], [1.0, -0.48059478]], dtype=np.float32),
                jnp.array([[1.0, -0.10662533], [1.0, -0.48059478]], dtype=np.float32),
                jnp.array([[1.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.48059478], [-1.0119202]], dtype=np.float32),
                jnp.array([[0.5], [0.9]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            18: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.3879677], [1.0, 0.4745814]], dtype=np.float32),
                jnp.array([[1.0, 0.3879677], [1.0, 0.4745814]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[0.4745814], [-0.43253192]], dtype=np.float32),
                jnp.array([[0.5], [0.40042576]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            19: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.34608668], [1.0, 1.015439]], dtype=np.float32),
                jnp.array([[1.0, 0.34608668], [1.0, 1.015439]], dtype=np.float32),
                jnp.array([[1.0], [0.0]], dtype=np.float32),
                jnp.array([[1.015439], [-0.6503669]], dtype=np.float32),
                jnp.array([[0.5], [0.1]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            20: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.31254572], [1.0, -0.171983]], dtype=np.float32),
                jnp.array([[1.0, 0.31254572], [1.0, -0.171983]], dtype=np.float32),
                jnp.array([[0.0], [1.0]], dtype=np.float32),
                jnp.array([[-0.171983], [0.3554843]], dtype=np.float32),
                jnp.array([[0.5], [0.8582839]], dtype=np.float32),
                jnp.array([[1.0], [2.0]], dtype=np.float32),
                0,
            ),
            21: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.71152794]], dtype=np.float32),
                jnp.array([[1.0, 0.71152794]], dtype=np.float32),
                jnp.array([[0.0]], dtype=np.float32),
                jnp.array([[-1.4751961]], dtype=np.float32),
                jnp.array([[0.22940305]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            22: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.11297973]], dtype=np.float32),
                jnp.array([[1.0, -0.11297973]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-0.91365016]], dtype=np.float32),
                jnp.array([[0.83200526]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            23: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.7074563]], dtype=np.float32),
                jnp.array([[1.0, -0.7074563]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-0.42447996]], dtype=np.float32),
                jnp.array([[0.9]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            24: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.18363248]], dtype=np.float32),
                jnp.array([[1.0, -0.18363248]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[0.82809216]], dtype=np.float32),
                jnp.array([[0.86304724]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            25: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.26854706]], dtype=np.float32),
                jnp.array([[1.0, -0.26854706]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[1.4379805]], dtype=np.float32),
                jnp.array([[0.8938225]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            26: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 1.0880693]], dtype=np.float32),
                jnp.array([[1.0, 1.0880693]], dtype=np.float32),
                jnp.array([[0.0]], dtype=np.float32),
                jnp.array([[-1.4844414]], dtype=np.float32),
                jnp.array([[0.1]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            27: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.2346751]], dtype=np.float32),
                jnp.array([[1.0, 0.2346751]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[1.7322738]], dtype=np.float32),
                jnp.array([[0.60214114]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            28: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.4278203]], dtype=np.float32),
                jnp.array([[1.0, -0.4278203]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-0.2416297]], dtype=np.float32),
                jnp.array([[0.9]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            29: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.16230251]], dtype=np.float32),
                jnp.array([[1.0, -0.16230251]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-0.8412731]], dtype=np.float32),
                jnp.array([[0.854221]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            30: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.58896536]], dtype=np.float32),
                jnp.array([[1.0, 0.58896536]], dtype=np.float32),
                jnp.array([[0.0]], dtype=np.float32),
                jnp.array([[0.17993484]], dtype=np.float32),
                jnp.array([[0.31136543]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            31: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.42027605]], dtype=np.float32),
                jnp.array([[1.0, 0.42027605]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-1.7384543]], dtype=np.float32),
                jnp.array([[0.44558907]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            32: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.34141153]], dtype=np.float32),
                jnp.array([[1.0, -0.34141153]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[0.432817]], dtype=np.float32),
                jnp.array([[0.9]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            33: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.44227958]], dtype=np.float32),
                jnp.array([[1.0, 0.44227958]], dtype=np.float32),
                jnp.array([[0.0]], dtype=np.float32),
                jnp.array([[-0.08650005]], dtype=np.float32),
                jnp.array([[0.42713708]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            34: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.20521514]], dtype=np.float32),
                jnp.array([[1.0, 0.20521514]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[0.06941197]], dtype=np.float32),
                jnp.array([[0.62594366]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            35: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.10311087]], dtype=np.float32),
                jnp.array([[1.0, 0.10311087]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[1.7710041]], dtype=np.float32),
                jnp.array([[0.70329374]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            36: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.22611268]], dtype=np.float32),
                jnp.array([[1.0, 0.22611268]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-1.4893746]], dtype=np.float32),
                jnp.array([[0.60911477]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            37: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.29369444]], dtype=np.float32),
                jnp.array([[1.0, -0.29369444]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-0.61468214]], dtype=np.float32),
                jnp.array([[0.9]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            38: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.02824982]], dtype=np.float32),
                jnp.array([[1.0, -0.02824982]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[0.97335577]], dtype=np.float32),
                jnp.array([[0.7876763]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            39: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, 0.19699477]], dtype=np.float32),
                jnp.array([[1.0, 0.19699477]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[1.7229595]], dtype=np.float32),
                jnp.array([[0.6324834]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            40: (
                np.array(
                    [-0.19788389, 0.41594946, 0.32135132, -0.47109252], dtype=np.float32
                ),
                jnp.array([[1.0, -0.29904404]], dtype=np.float32),
                jnp.array([[1.0, -0.29904404]], dtype=np.float32),
                jnp.array([[1.0]], dtype=np.float32),
                jnp.array([[-1.929327]], dtype=np.float32),
                jnp.array([[0.9]], dtype=np.float32),
                jnp.array([[2.0]], dtype=np.float32),
                0,
            ),
            41: (),
            42: (),
            43: (),
            44: (),
            45: (),
            46: (),
            47: (),
            48: (),
            49: (),
            50: (),
            51: (),
            52: (),
            53: (),
            54: (),
            55: (),
            56: (),
            57: (),
            58: (),
            59: (),
            60: (),
            61: (),
            62: (),
            63: (),
            64: (),
            65: (),
            66: (),
            67: (),
            68: (),
            69: (),
            70: (),
            71: (),
            72: (),
            73: (),
            74: (),
            75: (),
            76: (),
            77: (),
            78: (),
            79: (),
            80: (),
            81: (),
            82: (),
            83: (),
            84: (),
            85: (),
            86: (),
            87: (),
            88: (),
            89: (),
            90: (),
            91: (),
            92: (),
            93: (),
            94: (),
            95: (),
            96: (),
            97: (),
            98: (),
            99: (),
            100: (),
        },
        list(range(1, 101)),
        3,
    )
    expected_result = (
        [
            jnp.array([-6.0744834, -0.4129725, -4.916727, 2.6100497], dtype="float32"),
            jnp.array([-0.70775354, 1.2868932, 0.0, 0.0], dtype="float32"),
            jnp.array(
                [-2.2755318, 0.81675947, -1.7838521, 0.45714575], dtype="float32"
            ),
            jnp.array(
                [1.4500768, -0.38799185, 1.4500768, -0.38799185], dtype="float32"
            ),
            jnp.array([-4.543558, -2.8584292, -1.864192, -0.03278651], dtype="float32"),
            jnp.array([-1.98855, -0.50614494, -1.98855, -0.50614494], dtype="float32"),
            jnp.array([2.114283, 0.38500565, 2.114283, 0.38500565], dtype="float32"),
            jnp.array(
                [-0.22412544, -0.31223986, 0.25848502, 0.10394123], dtype="float32"
            ),
            jnp.array([-1.680202, -0.97209644, 0.0, 0.0], dtype="float32"),
            jnp.array([-3.1551933, -2.4731874, -1.476855, -1.0952727], dtype="float32"),
            jnp.array([3.2389503, 0.59521943, 2.402046, 1.5263554], dtype="float32"),
            jnp.array(
                [-0.45810664, 0.48341236, 0.72409356, 0.21680155], dtype="float32"
            ),
            jnp.array([-1.3176026, -0.8499259, 0.0, 0.0], dtype="float32"),
            jnp.array([-0.692469, 0.48798165, -1.1289438, 0.72796714], dtype="float32"),
            jnp.array([0.77780616, 0.539714, 0.0, 0.0], dtype="float32"),
            jnp.array(
                [3.1190822, -0.26920593, 3.1190822, -0.26920593], dtype="float32"
            ),
            jnp.array([3.5436618, -1.2468662, 3.5436618, -1.2468662], dtype="float32"),
            jnp.array(
                [0.11908442, 0.12104378, -0.7450154, -0.2890419], dtype="float32"
            ),
            jnp.array(
                [-0.07240319, 1.1461139, -1.8221118, -0.6306086], dtype="float32"
            ),
            jnp.array(
                [-0.23686166, 0.14161731, -0.44506633, 0.07654385], dtype="float32"
            ),
            jnp.array([3.1465437, 2.2388537, 0.0, 0.0], dtype="float32"),
            jnp.array(
                [2.0866952, -0.23575427, 2.0866952, -0.23575427], dtype="float32"
            ),
            jnp.array(
                [1.1739174, -0.83049524, 1.1739174, -0.83049524], dtype="float32"
            ),
            jnp.array(
                [-1.3889973, 0.25506502, -1.3889973, 0.25506502], dtype="float32"
            ),
            jnp.array(
                [-2.5994093, 0.69806373, -2.5994093, 0.69806373], dtype="float32"
            ),
            jnp.array([3.4782786, 3.7846084, 0.0, 0.0], dtype="float32"),
            jnp.array([-3.243494, -0.7611673, -3.243494, -0.7611673], dtype="float32"),
            jnp.array(
                [0.7773769, -0.33257762, 0.7773769, -0.33257762], dtype="float32"
            ),
            jnp.array(
                [1.9473808, -0.31606477, 1.9473808, -0.31606477], dtype="float32"
            ),
            jnp.array([-0.2656778, -0.15647502, 0.0, 0.0], dtype="float32"),
            jnp.array([3.6774929, 1.5455621, 3.6774929, 1.5455621], dtype="float32"),
            jnp.array(
                [-0.5810461, 0.19837584, -0.5810461, 0.19837584], dtype="float32"
            ),
            jnp.array([0.14516422, 0.06420317, 0.0, 0.0], dtype="float32"),
            jnp.array(
                [0.08547851, 0.01754148, 0.08547851, 0.01754148], dtype="float32"
            ),
            jnp.array(
                [-3.3064451, -0.34093043, -3.3064451, -0.34093043], dtype="float32"
            ),
            jnp.array([3.200747, 0.7237295, 3.200747, 0.7237295], dtype="float32"),
            jnp.array(
                [1.5086896, -0.44309375, 1.5086896, -0.44309375], dtype="float32"
            ),
            jnp.array(
                [-1.6966612, 0.04793037, -1.6966612, 0.04793037], dtype="float32"
            ),
            jnp.array([-3.22071, -0.634463, -3.22071, -0.634463], dtype="float32"),
            jnp.array([4.1385694, -1.2376145, 4.1385694, -1.2376145], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
        ],
        [
            jnp.array(
                [
                    [4.0, 4.1605053, 2.0, -1.0617021],
                    [4.1605053, 14.199331, -1.0617021, 0.5636057],
                    [2.0, -1.0617021, 2.0, -1.0617021],
                    [-1.0617021, 0.5636057, -1.0617021, 0.5636057],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 0.61852604, 0.0, 0.0],
                    [0.61852604, 0.93759334, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -1.9753346, 2.0, -0.5125377],
                    [-1.9753346, 1.2012348, -0.5125377, 0.13134746],
                    [2.0, -0.5125377, 2.0, -0.5125377],
                    [-0.5125377, 0.13134746, -0.5125377, 0.13134746],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -1.1867105, 4.0, -1.1867105],
                    [-1.1867105, 0.44431075, -1.1867105, 0.44431075],
                    [4.0, -1.1867105, 4.0, -1.1867105],
                    [-1.1867105, 0.44431075, -1.1867105, 0.44431075],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0000000e00, 2.1443624e00, 2.0000000e00, 3.5175040e-02],
                    [2.1443624e00, 2.2249544e00, 3.5175040e-02, 6.1864173e-04],
                    [2.0000000e00, 3.5175040e-02, 2.0000000e00, 3.5175040e-02],
                    [3.5175040e-02, 6.1864173e-04, 3.5175040e-02, 6.1864173e-04],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 1.0376551, 4.0, 1.0376551],
                    [1.0376551, 0.8286923, 1.0376551, 0.8286923],
                    [4.0, 1.0376551, 4.0, 1.0376551],
                    [1.0376551, 0.8286923, 1.0376551, 0.8286923],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -0.29961774, 4.0, -0.29961774],
                    [-0.29961774, 0.2606611, -0.29961774, 0.2606611],
                    [4.0, -0.29961774, 4.0, -0.29961774],
                    [-0.29961774, 0.2606611, -0.29961774, 0.2606611],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 2.528942, 2.0, 0.8042341],
                    [2.528942, 1.8107052, 0.8042341, 0.32339624],
                    [2.0, 0.8042341, 2.0, 0.8042341],
                    [0.8042341, 0.32339624, 0.8042341, 0.32339624],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 2.4065852, 0.0, 0.0],
                    [2.4065852, 1.4574674, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 3.125249, 2.0, 1.48325],
                    [3.125249, 2.4480956, 1.48325, 1.1000153],
                    [2.0, 1.48325, 2.0, 1.48325],
                    [1.48325, 1.1000153, 1.48325, 1.1000153],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -0.95431197, 2.0, 1.2708794],
                    [-0.95431197, 3.2833054, 1.2708794, 0.80756724],
                    [2.0, 1.2708794, 2.0, 1.2708794],
                    [1.2708794, 0.80756724, 1.2708794, 0.80756724],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 0.14778024, 2.0, 0.59882194],
                    [0.14778024, 0.28101316, 0.59882194, 0.17929386],
                    [2.0, 0.59882194, 2.0, 0.59882194],
                    [0.59882194, 0.17929386, 0.59882194, 0.17929386],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 2.5124962, 0.0, 0.0],
                    [2.5124962, 1.5814378, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -2.389296, 2.0, -1.2896428],
                    [-2.389296, 1.4362081, -1.2896428, 0.8315893],
                    [2.0, -1.2896428, 2.0, -1.2896428],
                    [-1.2896428, 0.8315893, -1.2896428, 0.8315893],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 1.9616958, 0.0, 0.0],
                    [1.9616958, 2.0937555, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 0.3594178, 4.0, 0.3594178],
                    [0.3594178, 1.0715947, 0.3594178, 1.0715947],
                    [4.0, 0.3594178, 4.0, 0.3594178],
                    [0.3594178, 1.0715947, 0.3594178, 1.0715947],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, -1.1744403, 4.0, -1.1744403],
                    [-1.1744403, 0.48468062, -1.1744403, 0.48468062],
                    [4.0, -1.1744403, 4.0, -1.1744403],
                    [-1.1744403, 0.48468062, -1.1744403, 0.48468062],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 1.7250981, 2.0, 0.7759354],
                    [1.7250981, 0.75149286, 0.7759354, 0.30103788],
                    [2.0, 0.7759354, 2.0, 0.7759354],
                    [0.7759354, 0.30103788, 0.7759354, 0.30103788],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 2.7230515, 2.0, 0.69217336],
                    [2.7230515, 2.3017848, 0.69217336, 0.23955198],
                    [2.0, 0.69217336, 2.0, 0.69217336],
                    [0.69217336, 0.23955198, 0.69217336, 0.23955198],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [4.0, 0.28112543, 2.0, -0.343966],
                    [0.28112543, 0.25452596, -0.343966, 0.05915631],
                    [2.0, -0.343966, 2.0, -0.343966],
                    [-0.343966, 0.05915631, -0.343966, 0.05915631],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 1.4230559, 0.0, 0.0],
                    [1.4230559, 1.012544, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.22595946, 2.0, -0.22595946],
                    [-0.22595946, 0.02552884, -0.22595946, 0.02552884],
                    [2.0, -0.22595946, 2.0, -0.22595946],
                    [-0.22595946, 0.02552884, -0.22595946, 0.02552884],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -1.4149126, 2.0, -1.4149126],
                    [-1.4149126, 1.0009888, -1.4149126, 1.0009888],
                    [2.0, -1.4149126, 2.0, -1.4149126],
                    [-1.4149126, 1.0009888, -1.4149126, 1.0009888],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.36726496, 2.0, -0.36726496],
                    [-0.36726496, 0.06744178, -0.36726496, 0.06744178],
                    [2.0, -0.36726496, 2.0, -0.36726496],
                    [-0.36726496, 0.06744178, -0.36726496, 0.06744178],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.5370941, 2.0, -0.5370941],
                    [-0.5370941, 0.14423504, -0.5370941, 0.14423504],
                    [2.0, -0.5370941, 2.0, -0.5370941],
                    [-0.5370941, 0.14423504, -0.5370941, 0.14423504],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 2.1761386, 0.0, 0.0],
                    [2.1761386, 2.3677897, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.4693502, 2.0, 0.4693502],
                    [0.4693502, 0.1101448, 0.4693502, 0.1101448],
                    [2.0, 0.4693502, 2.0, 0.4693502],
                    [0.4693502, 0.1101448, 0.4693502, 0.1101448],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.8556406, 2.0, -0.8556406],
                    [-0.8556406, 0.3660604, -0.8556406, 0.3660604],
                    [2.0, -0.8556406, 2.0, -0.8556406],
                    [-0.8556406, 0.3660604, -0.8556406, 0.3660604],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.32460502, 2.0, -0.32460502],
                    [-0.32460502, 0.05268421, -0.32460502, 0.05268421],
                    [2.0, -0.32460502, 2.0, -0.32460502],
                    [-0.32460502, 0.05268421, -0.32460502, 0.05268421],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 1.1779307, 0.0, 0.0],
                    [1.1779307, 0.6937604, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.8405521, 2.0, 0.8405521],
                    [0.8405521, 0.3532639, 0.8405521, 0.3532639],
                    [2.0, 0.8405521, 2.0, 0.8405521],
                    [0.8405521, 0.3532639, 0.8405521, 0.3532639],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.68282306, 2.0, -0.68282306],
                    [-0.68282306, 0.23312366, -0.68282306, 0.23312366],
                    [2.0, -0.68282306, 2.0, -0.68282306],
                    [-0.68282306, 0.23312366, -0.68282306, 0.23312366],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.88455915, 0.0, 0.0],
                    [0.88455915, 0.39122245, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.41043028, 2.0, 0.41043028],
                    [0.41043028, 0.08422651, 0.41043028, 0.08422651],
                    [2.0, 0.41043028, 2.0, 0.41043028],
                    [0.41043028, 0.08422651, 0.41043028, 0.08422651],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.20622174, 2.0, 0.20622174],
                    [0.20622174, 0.0212637, 0.20622174, 0.0212637],
                    [2.0, 0.20622174, 2.0, 0.20622174],
                    [0.20622174, 0.0212637, 0.20622174, 0.0212637],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.45222536, 2.0, 0.45222536],
                    [0.45222536, 0.10225388, 0.45222536, 0.10225388],
                    [2.0, 0.45222536, 2.0, 0.45222536],
                    [0.45222536, 0.10225388, 0.45222536, 0.10225388],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.5873889, 2.0, -0.5873889],
                    [-0.5873889, 0.17251284, -0.5873889, 0.17251284],
                    [2.0, -0.5873889, 2.0, -0.5873889],
                    [-0.5873889, 0.17251284, -0.5873889, 0.17251284],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0000000e00, -5.6499641e-02, 2.0000000e00, -5.6499641e-02],
                    [-5.6499641e-02, 1.5961047e-03, -5.6499641e-02, 1.5961047e-03],
                    [2.0000000e00, -5.6499641e-02, 2.0000000e00, -5.6499641e-02],
                    [-5.6499641e-02, 1.5961047e-03, -5.6499641e-02, 1.5961047e-03],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, 0.39398953, 2.0, 0.39398953],
                    [0.39398953, 0.07761388, 0.39398953, 0.07761388],
                    [2.0, 0.39398953, 2.0, 0.39398953],
                    [0.39398953, 0.07761388, 0.39398953, 0.07761388],
                ],
                dtype="float32",
            ),
            jnp.array(
                [
                    [2.0, -0.5980881, 2.0, -0.5980881],
                    [-0.5980881, 0.17885467, -0.5980881, 0.17885467],
                    [2.0, -0.5980881, 2.0, -0.5980881],
                    [-0.5980881, 0.17885467, -0.5980881, 0.17885467],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype="float32",
            ),
        ],
        [
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                    [[-0.0], [-0.0]],
                ],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]], [[0.0], [-0.0]]]
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
            np.array(
                [[[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.0], [0.0]]],
                dtype="float32",
            ),
        ],
    )

    assert len(result) == len(expected_result)
    for idx, expected_element in enumerate(expected_result):
        np.testing.assert_allclose(result[idx], expected_element, atol=1e-07)


def test_calculate_inference_loss_derivatives_multiple_size_groups():
    """
    Note that the non-incremental case is tested via larger tests in
    test_after_study_analysis.py.

    """

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, 0, None, None, 1, 1, 0, None],
            "reward": [1.0, -1, 0, 0, None, None, 1, 0, 1, None],
            "intercept": [1.0, 1, 1, 0, None, None, 1, 1, 1, None],
            "past_reward": [0.0, 1, -1, 0, None, None, 1, 1, 0, None],
            "in_study": [1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
            "action1prob": [0.5, 0.6, 0.7, 0, None, None, 0.1, 0.2, 0.3, None],
        }
    )

    theta = np.array([-1.0, 2.0, 3.0, 4.0], dtype="float32")

    user_1_centered_actions = np.array(
        [0 - 0.5, 1 - 0.6, 1 - 0.7, 0 - 0], dtype="float32"
    )
    user_1_states = np.array(
        [[1.0, 0.0], [1.0, 1.0], [1.0, -1.0], [0, 0]], dtype="float32"
    )
    user_1_rewards = np.array([1.0, -1, 0, 0], dtype="float32")
    user_1_loss_gradient = -2 * sum(
        (
            (
                user_1_rewards[i]
                - theta[:2] @ user_1_states[i]
                - theta[2:] @ (user_1_centered_actions[i] * user_1_states[i])
            )
            * np.concatenate(
                [
                    user_1_states[i],
                    user_1_centered_actions[i] * user_1_states[i],
                ]
            )
            for i in range(4)
        )
    )

    user_2_centered_actions = np.array([1 - 0.1, 1 - 0.2, 0 - 0.3], dtype="float32")
    user_2_states = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]], dtype="float32")
    user_2_rewards = np.array([1.0, 0, 1.0], dtype="float32")
    user_2_loss_gradient = -2 * sum(
        [
            (
                user_2_rewards[i]
                - theta[:2] @ user_2_states[i]
                - theta[2:] @ (user_2_centered_actions[i] * user_2_states[i])
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
    # instead of ironing out floating point issues.
    np.testing.assert_allclose(
        user_1_loss_gradient,
        np.array([-4.0000005, 16.199999, 5.359999, 5.8199997], dtype="float32"),
        atol=1e-05,
    )
    np.testing.assert_allclose(
        user_2_loss_gradient,
        np.array([20.0, 25.8, 23.64, 21.9], dtype="float32"),
        atol=1e-05,
    )
    expected_result = (
        np.array(
            [
                np.array([-4.0000005, 16.199999, 5.359999, 5.8199997], dtype="float32"),
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
                    [[-6.0], [-14.0], [2.0], [0], [0]],
                    [[-0.0], [-14.0], [-2.0], [0], [0]],
                    [[10.0], [-15.199999], [7.2], [0], [0]],
                    [[-0.0], [-15.199999], [-7.2], [0], [0]],
                ],
                [
                    [[0], [-14.0], [-14.0], [-6.0], [0]],
                    [[0], [-14.0], [-14.0], [-0.0], [0]],
                    [[0], [-25.2], [-24.4], [7.6000004], [0]],
                    [[0], [-25.199999], [-24.400002], [-0.0], [0]],
                ],
            ],
            dtype="float32",
        ),
    )

    calculated_result = calculate_derivatives.calculate_inference_loss_derivatives(
        study_df,
        theta,
        simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_action_centering.synthetic_get_least_squares_loss_inference_action_centering,
        0,
        [1, 2],
        "user_id",
        "action1prob",
        "in_study",
        "calendar_t",
    )
    np.testing.assert_allclose(calculated_result[0], expected_result[0], rtol=1e-07)
    np.testing.assert_allclose(calculated_result[1], expected_result[1], rtol=1e-07)
    np.testing.assert_allclose(calculated_result[2], expected_result[2], rtol=1e-07)

    assert len(calculated_result) == len(expected_result)
