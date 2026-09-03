import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import post_deployment_analysis
from lifejacket.arg_threading_helpers import (
    replace_tuple_index,
    thread_action_prob_func_args,
    thread_inference_func_args,
    thread_update_func_args,
)
from lifejacket.constants import FunctionTypes
from lifejacket.helper_functions import get_radon_nikodym_weight
from simulators_and_runners.functions_to_pass_to_analysis.RL_least_squares_loss_regularized_previous_betas_as_args import (
    RL_least_squares_loss_regularized_previous_betas_as_args,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_generalized_logistic import (
    synthetic_get_action_1_prob_generalized_logistic,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_estimating_function_inference_no_action_centering import (
    synthetic_get_least_squares_estimating_function_inference_no_action_centering,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_estimating_function_rl import (
    synthetic_get_least_squares_estimating_function_rl,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_action_centering import (
    synthetic_get_least_squares_loss_inference_action_centering,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_inference_no_action_centering import (
    synthetic_get_least_squares_loss_inference_no_action_centering,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl import (
    synthetic_get_least_squares_loss_rl,
)

# TODO: Add checking of all aux values.


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs():

    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1

    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1

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
            # Note that only the last element of beta are used in the particular act
            # prob function used here. Important to have a different two components
            # for each decision time for robustness of the test. Further, originally
            # the different values here and in the threaded in betas actually resulted
            # in the same action probability by bad luck, but this is fixed now.
            user_id: (
                jnp.array([-(decision_time), 2.0, decision_time, 4.0], dtype="float32"),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These don't build up over time as they would in reality. This is fine.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 2.0, policy_num, 4.0], dtype="float32"),
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
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These action probabilities are irrelevant for weights
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_loss_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
    Test that the stacking function correctly computes a weighted estimating
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
    are appropriately subbed in to (only) the numerators of the weights.

    This test does check that given loss functions on both sides, they are
    differentiated properly, as I actually load the corresponding estimating
    functions directly below and work with them instead of differentiating the
    losses in the test to form expected values.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    # Note for this test these must match the betas in the action probability
    # function args, so that all weights end up being 1, as we do not multiply
    # by them below.
    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    alg_estimating_func = synthetic_get_least_squares_estimating_function_rl
    inference_estimating_func = (
        synthetic_get_least_squares_estimating_function_inference_no_action_centering
    )

    # Note that we don't multiply by the weights! Therefore we test that they
    # are all 1, as they should always be in practice, with the same beta going
    # into the numerator and denominator.
    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[1],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[2],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )
    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-6,
    )


@pytest.fixture
def setup_data_two_estimating_functions_no_action_probs():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.ESTIMATING
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    inference_func_type = FunctionTypes.ESTIMATING
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
            # Note that only the last element of beta are used in the particular act
            # prob function used here. Important to have a different two components
            # for each decision time for robustness of the test. Further, originally
            # the different values here and in the threaded in betas actually resulted
            # in the same action probability by bad luck, but this is fixed now.
            user_id: (
                jnp.array([-(decision_time), 2.0, decision_time, 4.0], dtype="float32"),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These don't build up over time as they would in reality. This is fine.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 2.0, policy_num, 4.0], dtype="float32"),
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
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These action probabilities are irrelevant for weights
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_estimating_function_inference_no_action_centering,
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
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_estimating_function_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_estimating_function_inference_no_action_centering,
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


def test_construct_single_user_weighted_estimating_function_stacker_estimating_functions_given(
    setup_data_two_estimating_functions_no_action_probs,
):
    """
    Just like the above test, but I give two estimating functions instead of
    loss functions to verify that use case works too (it should be simpler, no differentiation.)
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_estimating_functions_no_action_probs

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    # Note for this test these must match the betas in the action probability
    # function args, so that all weights end up being 1, as we do not multiply
    # by them below.
    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    alg_estimating_func = synthetic_get_least_squares_estimating_function_rl
    inference_estimating_func = (
        synthetic_get_least_squares_estimating_function_inference_no_action_centering
    )

    # Note that we don't multiply by the weights! Therefore we test that they
    # are all 1, as they should always be in practice, with the same beta going
    # into the numerator and denominator.
    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[1],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[3][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[5][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
            # Weighted theta estimating function value
            inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[2],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )
    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-6,
    )


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs_different_betas():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
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
                jnp.array(
                    [-(decision_time), 17.0, decision_time, 19.0], dtype="float32"
                ),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These don't build up over time as they would in reality. This is fine.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 17.0, policy_num, 19.0], dtype="float32"),
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
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These action probabilities are irrelevant for weights.
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_loss_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
    Test that the stacking function correctly computes a weighted estimating
    function stack for each of 2 users.

    This test handles the simplest case: no incremental recruitment, no use of
    action probabilities in the loss/estimating functions for algorithm updates
    or inference, and only 1 decision time between updates.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the numerators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the betas in all_post_update_betas are subbed in for use in the
    estimating function evaluations, because the estimating function values
    would be different were they not.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
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
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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

    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-6,
    )


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs_incremental_recruitment():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1, 4: 2, 5: 3}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {3: 0, 4: 1, 5: 0},
        2: {1: 1, 2: 0, 3: 1, 4: 0},
    }
    policy_num_by_decision_time_by_user_id = {
        1: {3: 3, 4: 4, 5: 5},
        2: {1: 1, 2: 2, 3: 3, 4: 4},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                (
                    jnp.array(
                        [-(decision_time), 17.0, decision_time, 19.0], dtype="float32"
                    ),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([user_id, -1.0], dtype="float32"),
                )
                # for inactive times, give empty tuple
                if (
                    (user_id == 1 and decision_time > 2)
                    or (user_id == 2 and decision_time < 5)
                )
                else ()
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These don't build up over time as they would in reality. This is fine.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                (
                    jnp.array([-policy_num, 17.0, policy_num, 19.0], dtype="float32"),
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
                if (
                    (
                        user_id == 1 and policy_num > 3
                    )  # policy 3 uses data from times 1 and 2, for which user 1 is inactive
                    or user_id == 2
                    # you never drop out of est eqns after entering study, so user 2 always has data
                )
                else ()
            )
            for user_id in (1, 2)
        }
        for policy_num in range(2, 6)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            "reward": [0, 0, 0, 0, 2.0, 2, 1, 0, 1, 0],
            "intercept": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            "past_reward": [0.0, 0, 0, 0, 0, 0, 2, 1, 0, 0],
            "in_study": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            # These action probabilities are irrelevant for weights.
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_loss_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_no_action_centering,
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


def test_construct_single_user_weighted_estimating_function_stacker_incremental_recruitment(
    setup_data_two_loss_functions_no_action_probs_incremental_recruitment,
):
    """
    Test that the stacking function correctly computes a weighted estimating
    function stack for each of 2 users.

    This test adds in incremental recruitment to the simplest case.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the numerators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the betas in all_post_update_betas are subbed in for use in the
    estimating function evaluations, because the estimating function values
    would be different were they not.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_no_action_probs_incremental_recruitment

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            # No contribution to the second policy (index 0 in post-update policy list)
            np.zeros(4),
            # No contribution to the third policy (index 1 in post-update policy list)
            np.zeros(4),
            # Finally, a contribution to the fourth policy (index 2 in post-update policy list),
            # where the first policy the user used was the third policy (index 1)
            get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[4][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
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
            get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
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
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            # No weight from time 5 because out of study
            * get_radon_nikodym_weight(
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
            * inference_estimating_func(
                *replace_tuple_index(
                    inference_func_args_by_user_id[2],
                    inference_func_args_theta_index,
                    theta,
                )
            ),
        ]
    )

    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-6,
    )


@pytest.fixture
def setup_data_two_loss_functions_no_action_probs_multiple_decisions_between_updates():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {1: 0, 2: 1, 3: 0, 4: 1, 5: 0},
        2: {1: 1, 2: 0, 3: 1, 4: 0, 5: 1},
    }
    # Note user 2 here: they are late to receive updates 2 and 3. This could arise
    # from something like the app-opening issue in Oralytics.  I am assuming
    # that they ARE contributing data as expected to both policies 2 and 3
    # despite not receiving update 2.
    policy_num_by_decision_time_by_user_id = {
        1: {1: 1, 2: 1, 3: 2, 4: 2, 5: 3},
        2: {1: 1, 2: 1, 3: 1, 4: 2, 5: 2},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                jnp.array(
                    [-(decision_time // 2), 17.0, decision_time // 2, 19.0],
                    dtype="float32",
                ),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These don't build up over time as they would in reality. This is fine.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 17.0, policy_num, 19.0], dtype="float32"),
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
        for policy_num in range(2, 4)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "reward": [1.0, -1, 0, 0, 2, 2, 1, 0, 1, 3],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 0, 0, 0, 2, 1, 0, 1],
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These action probabilities are irrelevant for weights.
            # Only the passed betas and the action probability func args matter.
            "action1prob": [0.5] * 10,
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_no_action_centering,
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
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_loss_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_no_action_centering,
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


def test_batched_and_reference_implementations_agree_per_subject_incremental_recruitment(
    setup_data_two_loss_functions_no_action_probs_incremental_recruitment,
):
    """
    Cross-checks the batched get_avg_weighted_estimating_function_stacks_and_aux_values
    (ADS-139 Step 4) against the retained, un-batched
    post_deployment_analysis._reference_single_subject_weighted_estimating_function_stacker
    (the pre-Step-4 per-subject implementation), row by row, on the same
    incremental-recruitment fixture used above. This is a strictly finer
    check than comparing only the averaged stack (test above) or per-subject
    outer products: a scatter/gather indexing bug in the batched path could
    still cancel out in an outer product but not in the raw per-subject
    stack compared here directly.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_no_action_probs_incremental_recruitment

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )
    user_ids = jnp.array([1, 2])
    beta_dim = all_post_update_betas.shape[1]

    batched_result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        beta_dim,
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )
    per_subject_stacks = batched_result[1][4]

    algorithm_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    (
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        action_prob_func_args_by_decision_time_by_subject_id,
    ) = thread_action_prob_func_args(
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        all_post_update_betas,
        beta_index_by_policy_num,
        action_prob_func_args_beta_index,
    )
    threaded_update_func_args_by_policy_num_by_subject_id = thread_update_func_args(
        update_func_args_by_by_user_id_by_policy_num,
        all_post_update_betas,
        beta_index_by_policy_num,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        action_prob_func,
    )
    threaded_inference_func_args_by_subject_id = thread_inference_func_args(
        inference_func_args_by_user_id,
        inference_func_args_theta_index,
        theta,
        inference_func_args_action_prob_index,
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        inference_action_prob_decision_times_by_user_id,
        action_prob_func,
    )

    for i, subject_id in enumerate(user_ids.tolist()):
        reference_stack = post_deployment_analysis._reference_single_subject_weighted_estimating_function_stacker(
            beta_dim,
            subject_id,
            action_prob_func,
            algorithm_estimating_func,
            inference_estimating_func,
            action_prob_func_args_beta_index,
            inference_func_args_theta_index,
            action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_update_func_args_by_policy_num_by_subject_id[subject_id],
            threaded_inference_func_args_by_subject_id[subject_id],
            policy_num_by_decision_time_by_user_id[subject_id],
            action_by_decision_time_by_user_id[subject_id],
            beta_index_by_policy_num,
            include_auxiliary_outputs=False,
        )
        np.testing.assert_allclose(
            per_subject_stacks[i],
            reference_stack,
            rtol=1e-6,
            err_msg=f"Batched and reference implementations disagree for subject {subject_id}.",
        )


def test_construct_single_user_weighted_estimating_function_stacker_multiple_decisions_between_updates(
    setup_data_two_loss_functions_no_action_probs_multiple_decisions_between_updates,
):
    """
    Test that the stacking function correctly computes a weighted estimating
    function stack for each of 2 users.

    This test adds the wrinkle of 2 decision times between updates to the
    simplest case above.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the numerators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the betas in all_post_update_betas are subbed in for use in the
    estimating function evaluations, because the estimating function values
    would be different were they not.

    User 2 is an interesting case here, late to receive updates. This situation
    arose in Oralytics due to the app-opening issue.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_no_action_probs_multiple_decisions_between_updates

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 3, 4], dtype="float32"),
            jnp.array([-3, 2, 0.3, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    update_func_args_by_by_user_id_by_policy_num[3][1],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
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
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    update_func_args_by_by_user_id_by_policy_num[2][2],
                    alg_update_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
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
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][2][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[2][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
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

    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-6,
    )


@pytest.fixture
def setup_data_two_loss_functions_use_action_probs_both_sides():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = 5
    alg_update_func_args_action_prob_times_index = 6
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1, 4: 2, 5: 3}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
        2: {1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
    }
    policy_num_by_decision_time_by_user_id = {
        1: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        2: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                jnp.array(
                    [-(decision_time), 17.0, decision_time, 19.0], dtype="float32"
                ),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These DO build up over time now.  The inference ones will be different since
    # these aren't derived from the study df, but that's okay.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 17.0, policy_num, 19.0], dtype="float32"),
                jnp.array(
                    [
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                    ][: policy_num - 1],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                    ][: policy_num - 1],
                    dtype="float32",
                ),
                jnp.array([0, 1, 1, 0, 1][: policy_num - 1], dtype="float32").reshape(
                    -1, 1
                ),
                (
                    jnp.array(
                        [[1], [-1], [0], [0], [2]][: policy_num - 1],
                        dtype="float32",
                    )
                    if user_id == 1
                    else jnp.array(
                        [[2], [1], [0], [1], [3]][: policy_num - 1],
                        dtype="float32",
                    )
                ),
                # Note that we DO NOT want these used in the estimating function
                # args. They should be replaced by probs computed from act prob
                # func args using the decision times arg.
                jnp.array(
                    (np.add([0.1, 0.2, 0.3, 0.4, 0.5], [0.1 * (user_id - 1)] * 5))[
                        : policy_num - 1
                    ]
                ).reshape(-1, 1),
                jnp.array([1, 2, 3, 4, 5][: policy_num - 1]),
                1,  # This turns action centering on
            )
            for user_id in (1, 2)
        }
        for policy_num in range(2, 6)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "reward": [1.0, -1, 0, 0, 2, 2, 1, 0, 1, 3],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 0, 0, 0, 2, 1, 0, 1],
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These action probabilities are irrelevant for weights, but this
            # test is checking that these are *not* the values used in the est
            # function args on either side, instead computing the probs from
            # action prob function arguments.
            "action1prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_action_centering,
        inference_func_args_theta_index,
        study_df,
        jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
        "action1prob",
        "calendar_t",
        "user_id",
        "in_study",
    )

    return (
        synthetic_get_action_1_prob_pure,
        action_prob_func_args_beta_index,
        synthetic_get_least_squares_loss_rl,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_action_centering,
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


def test_construct_single_user_weighted_estimating_function_stacker_use_action_probs_both_sides(
    setup_data_two_loss_functions_use_action_probs_both_sides,
):
    """
    Test that the stacking function correctly computes a weighted estimating
    function stack for each of 2 users.

    This test adds the wrinkle of action probabilities being used in both the
    algorithm and inference estimating functions.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the numerators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the betas in all_post_update_betas are subbed in for use in the
    estimating function evaluations, because the estimating function values
    would be different were they not.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_use_action_probs_both_sides

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    reconstructed_action_probs = {
        1: [
            action_prob_func(*action_prob_func_args_by_user_id_by_decision_time[1][1]),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
        ],
        2: [
            action_prob_func(*action_prob_func_args_by_user_id_by_decision_time[1][2]),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
        ],
    }

    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[2][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[0],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1][:1]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[3][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[1],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1][:2]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[4][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[2],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1][:3]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[5][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[3],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1][:4]).reshape(-1, 1),
                )
            ),
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                ),
            )
            * inference_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        inference_func_args_by_user_id[1],
                        inference_func_args_theta_index,
                        theta,
                    ),
                    inference_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1]).reshape(-1, 1),
                )
            ),
        ]
    )
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[2][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[0],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2][:1]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[3][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[1],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2][:2]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[4][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[2],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2][:3]).reshape(-1, 1),
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[5][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[3],
                    ),
                    alg_update_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2][:4]).reshape(-1, 1),
                )
            ),
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        inference_func_args_by_user_id[2],
                        inference_func_args_theta_index,
                        theta,
                    ),
                    inference_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2]).reshape(-1, 1),
                )
            ),
        ]
    )

    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-5,
    )


def test_batched_algorithm_data_check_detects_the_same_inconsistency_as_the_original(
    setup_data_two_loss_functions_use_action_probs_both_sides,
):
    """
    Fast, unit-level exercise of check_batched_algorithm_estimating_function_args_equivalent
    (the ADS-139 batched replacement for
    input_checks.require_threaded_algorithm_estimating_function_args_equivalent).
    Every other test in this file passes suppress_all_data_checks=True, so
    without a test like this, the new check function is only ever exercised
    by the slower tests/benchmarks and tests/integration_tests suites.

    Uses the same fixture as
    test_construct_single_user_weighted_estimating_function_stacker_use_action_probs_both_sides,
    which -- per its own docstring -- "intentionally breaks the assumption
    that the betas in the action probability function args match those in
    all_post_update_betas". That is exactly the inconsistency this data
    check exists to catch, so with suppress_all_data_checks=False this call
    is EXPECTED to raise. Confirmed directly (see the investigation in this
    change) that the original, un-batched
    input_checks.require_threaded_algorithm_estimating_function_args_equivalent
    raises on this exact fixture too -- this test proves the new batched
    check reproduces that same detection, not a new bug.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_use_action_probs_both_sides

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )
    user_ids = jnp.array([1, 2])

    common_args = (
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
    )

    # Sanity: checks-suppressed still computes fine (the inconsistency only
    # matters to the check itself, not to the main computation).
    post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        *common_args, True, True
    )

    with pytest.raises(AssertionError):
        post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
            *common_args, False, True
        )


@pytest.fixture
def setup_data_two_loss_functions_use_action_probs_from_betas_RL_action_probs_inference():
    action_prob_func_args_beta_index = 0
    alg_update_func_args_previous_betas_index = -1
    alg_update_func_type = FunctionTypes.LOSS
    alg_update_func_args_beta_index = 0
    alg_update_func_args_action_prob_index = -1
    alg_update_func_args_action_prob_times_index = -1
    alg_update_func_args_previous_betas_index = 7
    inference_func_type = FunctionTypes.LOSS
    inference_func_args_theta_index = 0
    beta_index_by_policy_num = {2: 0, 3: 1, 4: 2, 5: 3}
    initial_policy_num = 1
    action_by_decision_time_by_user_id = {
        1: {1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
        2: {1: 0, 2: 1, 3: 1, 4: 0, 5: 1},
    }
    policy_num_by_decision_time_by_user_id = {
        1: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        2: {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    }
    action_prob_func_args_by_user_id_by_decision_time = {
        decision_time: {
            user_id: (
                jnp.array(
                    [-(decision_time), 17.0, decision_time, 19.0], dtype="float32"
                ),
                0.1,
                1.0,
                0.9,
                jnp.array([user_id, -1.0], dtype="float32"),
            )
            for user_id in (1, 2)
        }
        for decision_time in range(1, 6)
    }
    # These DO build up over time now.  The inference ones will be different since
    # these aren't derived from the study df, but that's okay.
    update_func_args_by_by_user_id_by_policy_num = {
        policy_num: {
            user_id: (
                jnp.array([-policy_num, 17.0, policy_num, 19.0], dtype="float32"),
                jnp.array(
                    [
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                    ][: policy_num - 1],
                    dtype="float32",
                ),
                jnp.array(
                    [
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                        [user_id, 1.0],
                        [user_id, -1.0],
                    ][: policy_num - 1],
                    dtype="float32",
                ),
                jnp.array([0, 1, 1, 0, 1][: policy_num - 1], dtype="float32").reshape(
                    -1, 1
                ),
                (
                    jnp.array(
                        [[1], [-1], [0], [0], [2]][: policy_num - 1],
                        dtype="float32",
                    )
                    if user_id == 1
                    else jnp.array(
                        [[2], [1], [0], [1], [3]][: policy_num - 1],
                        dtype="float32",
                    )
                ),
                # "Original" action probs. These are not used.
                jnp.array(
                    (np.add([0.1, 0.2, 0.3, 0.4, 0.5], [0.1 * (user_id - 1)] * 5))[
                        : policy_num - 1
                    ]
                ).reshape(-1, 1),
                # First action prob. It is preupdate. Must align with actual action
                # prob function args for eventual estimating functions to evaluate to
                # same thing as previous test, which is desired.
                synthetic_get_action_1_prob_pure(
                    jnp.array([-1, 17.0, 1, 19.0], dtype="float32"),
                    0.1,
                    1.0,
                    0.9,
                    jnp.array([user_id, -1.0], dtype="float32"),
                ).reshape(-1, 1),  # pre update action 1 probs
                # NOTE: we want these to be replaced in the eventual estimating function
                # args, hence the crazy values to stick out if they are used.
                jnp.array(
                    [
                        [-2, 2, 2, 40000000],
                        [-3, 2, 3, 40000000],
                        [-4, 2, 4, 40000000],
                        [-5, 2, 0.5, 40000000],
                    ]
                )[: policy_num - 2],
                jnp.array(
                    [2, 3, 4, 5][: policy_num - 2]
                ),  # post update policy nums so far
                0.1,  # lower clip
                1.0,  # steepness
                0.9,  # upper clip
                1,  # This turns action centering on,
                0,  # zero lambda
                1,  # n, doesn't matter since lambda zero
            )
            for user_id in (1, 2)
        }
        for policy_num in range(2, 6)
    }

    study_df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "calendar_t": [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
            "action": [0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            "reward": [1.0, -1, 0, 0, 2, 2, 1, 0, 1, 3],
            "intercept": [1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "past_reward": [0.0, 1, -1, 0, 0, 0, 2, 1, 0, 1],
            "in_study": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            # These should be ignored here.  Any action probs needed are reconstructed
            # from betas.
            "action1prob": [0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = post_deployment_analysis.process_inference_func_args(
        synthetic_get_least_squares_loss_inference_action_centering,
        inference_func_args_theta_index,
        study_df,
        jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
        "action1prob",
        "calendar_t",
        "user_id",
        "in_study",
    )

    return (
        synthetic_get_action_1_prob_generalized_logistic,
        action_prob_func_args_beta_index,
        RL_least_squares_loss_regularized_previous_betas_as_args,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        synthetic_get_least_squares_loss_inference_action_centering,
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


def test_construct_single_user_weighted_estimating_function_stacker_use_action_probs_from_betas_RL_action_probs_inference(
    setup_data_two_loss_functions_use_action_probs_from_betas_RL_action_probs_inference,
):
    """
    Test that the stacking function correctly computes a weighted estimating
    function stack for each of 2 users.

    This test adds the wrinkle of action probabilities being used in both the
    algorithm and inference estimating functions.

    **This test intentionally breaks the assumption that the betas in the action
    probability function args match those in all_post_update_betas, which
    makes the weights not 1, allowing us to test that the *right* weights are
    multiplied by the right estimating functions and also that the shared betas
    are appropriately subbed in to (only) the numerators of the weights for
    differentiation.

    We also have different betas in the update args vs all_post_update_betas,
    testing that the betas in all_post_update_betas are subbed in for use in the
    estimating function evaluations, because the estimating function values
    would be different were they not.
    """
    (
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
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
    ) = setup_data_two_loss_functions_use_action_probs_from_betas_RL_action_probs_inference

    theta = jnp.array([1.0, 2.0, 3.0, 4.0], dtype="float32")

    all_post_update_betas = jnp.array(
        [
            jnp.array([-2, 2, 2, 4], dtype="float32"),
            jnp.array([-3, 2, 3, 4], dtype="float32"),
            jnp.array([-4, 2, 4, 4], dtype="float32"),
            jnp.array([-5, 2, 0.5, 4], dtype="float32"),
        ]
    )

    user_ids = jnp.array([1, 2])

    result = post_deployment_analysis.get_avg_weighted_estimating_function_stacks_and_aux_values(
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        user_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
        True,
        True,
    )

    # Quite odd that it complains about ints here and not in the real function... but alas.
    alg_estimating_func = jax.grad(alg_update_func, allow_int=True)
    inference_estimating_func = jax.grad(inference_func, allow_int=True)

    reconstructed_action_probs = {
        1: [
            action_prob_func(*action_prob_func_args_by_user_id_by_decision_time[1][1]),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
        ],
        2: [
            action_prob_func(*action_prob_func_args_by_user_id_by_decision_time[1][2]),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[2][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[0],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                )
            ),
            action_prob_func(
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][2],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                )
            ),
        ],
    }

    expected_weighted_stack_1 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[2][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[0],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:0],
                )
            ),
            get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[3][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[1],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:1],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[4][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[2],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:2],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[5][1],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[3],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:3],
                )
            ),
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[3][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][3],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[3][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[1],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[4][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][4],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[4][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[2],
                ),
            )
            * get_radon_nikodym_weight(
                action_prob_func_args_by_user_id_by_decision_time[5][1][
                    action_prob_func_args_beta_index
                ],
                action_prob_func,
                action_prob_func_args_beta_index,
                action_by_decision_time_by_user_id[1][5],
                *replace_tuple_index(
                    action_prob_func_args_by_user_id_by_decision_time[5][1],
                    action_prob_func_args_beta_index,
                    all_post_update_betas[3],
                ),
            )
            * inference_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        inference_func_args_by_user_id[1],
                        inference_func_args_theta_index,
                        theta,
                    ),
                    inference_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[1]).reshape(-1, 1),
                )
            ),
        ]
    )
    expected_weighted_stack_2 = jnp.concatenate(
        [
            # Weighted beta estimating function values
            alg_estimating_func(
                *replace_tuple_index(
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[2][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[0],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:0],
                )
            ),
            get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[3][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[1],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:1],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[4][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[2],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:2],
                )
            ),
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        update_func_args_by_by_user_id_by_policy_num[5][2],
                        alg_update_func_args_beta_index,
                        all_post_update_betas[3],
                    ),
                    alg_update_func_args_previous_betas_index,
                    all_post_update_betas[:3],
                )
            ),
            # Weighted theta estimating function value
            get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
            * get_radon_nikodym_weight(
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
                    replace_tuple_index(
                        inference_func_args_by_user_id[2],
                        inference_func_args_theta_index,
                        theta,
                    ),
                    inference_func_args_action_prob_index,
                    jnp.array(reconstructed_action_probs[2]).reshape(-1, 1),
                )
            ),
        ]
    )

    np.testing.assert_allclose(
        result[0],
        jnp.mean(
            jnp.array([expected_weighted_stack_1, expected_weighted_stack_2]), axis=0
        ),
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        result[1][0],
        result[0],
    )
    np.testing.assert_allclose(
        result[1][1],
        jnp.sum(
            jnp.array(
                [
                    jnp.outer(
                        expected_weighted_stack_1,
                        expected_weighted_stack_1,
                    ),
                    jnp.outer(
                        expected_weighted_stack_2,
                        expected_weighted_stack_2,
                    ),
                ]
            ),
            axis=0,
        ),
        rtol=1e-5,
    )


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

    result = get_radon_nikodym_weight(
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
    result = get_radon_nikodym_weight(
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

    expected_result = 1

    result = get_radon_nikodym_weight(
        beta_target,
        mock_action_prob_func,
        action_prob_func_args_beta_index,
        action,
        *action_prob_func_args_single_user,
    )

    np.testing.assert_allclose(result, expected_result)


def test_resolve_jacobian_row_chunk_size_auto_heuristic():
    """
    The None="auto" heuristic, pinned to the empirical calibration points in
    lifejacket/constants.py: unchunked (None) at and below the out_dim
    threshold (this repo's own benchmark fixtures, out_dim 28/56, land
    there, keeping the default path byte-identical); above it, chunk sizes
    at or below every budget verified safe at real oralytics scale.
    """
    resolve = post_deployment_analysis.resolve_jacobian_row_chunk_size

    # At/below the threshold: unchunked, exactly today's default path.
    assert resolve(None, 28) is None
    assert resolve(None, 56) is None
    assert resolve(None, 512) is None
    # Boundary: 513 is the first chunked out_dim, capped at 64 (the largest
    # chunk ever verified safe, at out_dim ~1500).
    assert resolve(None, 513) == 64
    assert resolve(None, 1024) == 64
    # Real-scale anchors (see constants.py): 43 at ~1500 and 32 at ~2000 are
    # both strictly smaller than the verified-safe 64 there; 21 at ~3100 and
    # 15 at ~4185 sit at/below the budgets verified safe at those scales.
    assert resolve(None, 1500) == 43
    assert resolve(None, 2000) == 32
    assert resolve(None, 3100) == 21
    assert resolve(None, 4185) == 15
    # Floor at 1 for absurdly large out_dim.
    assert resolve(None, 10**7) == 1


def test_resolve_jacobian_row_chunk_size_explicit_overrides():
    resolve = post_deployment_analysis.resolve_jacobian_row_chunk_size

    # 0 forces the unchunked single eager vmap even where auto would chunk.
    assert resolve(0, 4185) is None
    # A positive int is honored verbatim in both regimes.
    assert resolve(16, 4185) == 16
    assert resolve(16, 8) == 16
    # Negative is rejected.
    with pytest.raises(ValueError, match="jacobian_row_chunk_size"):
        resolve(-1, 100)


def test_pipeline_diagnostic_summary_rows_flag_extreme_condition_number():
    # Value from the real seed-0 local oralytics run that motivated the gate.
    rows, extreme_condition = (
        post_deployment_analysis.build_pipeline_diagnostic_summary_rows(1.473e13)
    )
    assert extreme_condition
    (condition_row,) = rows
    assert condition_row[0] == "joint_bread_condition_number"
    assert condition_row[1] == "failed"
    # The detail is a criteria list: a heading plus one CriterionResult whose own ok mirrors
    # the row status and whose value carries the measured condition number.
    heading, criterion = condition_row[2]
    assert heading == "criteria:"
    assert criterion.value == "1.473e+13"
    assert criterion.ok is False

    # A nonfinite condition number (a fully singular bread) is also extreme.
    _, extreme_condition = (
        post_deployment_analysis.build_pipeline_diagnostic_summary_rows(float("inf"))
    )
    assert extreme_condition


def test_pipeline_diagnostic_summary_rows_healthy_run_passes():
    rows, extreme_condition = (
        post_deployment_analysis.build_pipeline_diagnostic_summary_rows(1.0e5)
    )
    assert not extreme_condition
    (condition_row,) = rows
    assert condition_row[1] == "passed"
    _, criterion = condition_row[2]
    assert criterion.value == "1.000e+05"
    assert criterion.ok is True
