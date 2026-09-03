"""
Tests lifejacket.form_adjusted_meat_adjustments_directly, the diagnostic path
enabled by analyze_dataset's form_adjusted_meat_adjustments_explicitly=True.

Reuses the same small, fully-specified toy deployment (2 users, T=2 decision
points, one RL update, a sigmoid/Boltzmann policy, least squares on both the
RL and inference sides, no action-centering on either side) as
tests/unit_tests/test_sandwich_closed_form_notebook_example.py, since that
scenario is already independently validated there against a from-scratch
closed-form derivation -- but this file is about a different claim entirely
(the explicit-meat-adjustments equivalence below), not the notebook
regression check, so it lives on its own.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from lifejacket import post_deployment_analysis
from lifejacket.constants import FunctionTypes, SandwichFormationMethods
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl import (
    synthetic_get_least_squares_loss_rl,
)

STEEPNESS = 0.5
LOWER_CLIP = 0.1
UPPER_CLIP = 0.9

USER_IDS = np.array([1, 1, 2, 2])
CALENDAR_T = np.array([1, 2, 1, 2])
ACTIONS = np.array([0.0, 1.0, 1.0, 0.0])
REWARDS = np.array([-0.8565283183, -1.3169794352, -0.1061271784, -0.6967024415])


@pytest.fixture
def beta_and_theta_estimates():
    """
    Beta and theta estimates for the toy deployment, via plain OLS: beta_hat
    from the single user pair's t=1 data (the only data that fed the one RL
    update), and theta_hat from the full 4-row history.
    """
    design = np.column_stack([np.ones_like(ACTIONS), ACTIONS])
    t1_mask = CALENDAR_T == 1

    beta_hat = (
        LinearRegression(fit_intercept=False)
        .fit(design[t1_mask], REWARDS[t1_mask])
        .coef_
    )
    theta_hat = LinearRegression(fit_intercept=False).fit(design, REWARDS).coef_
    return beta_hat, theta_hat


def _run_package_pipeline_with_meat_adjustments(beta_hat, theta_hat):
    """
    Runs the real lifejacket pipeline on the toy deployment with
    form_adjusted_meat_adjustments_explicitly=True, returning the pieces
    needed to independently reconstruct the theta-only adjusted sandwich from
    the explicit per-subject meat adjustments (eq. 5.3 of
    https://arxiv.org/pdf/2202.07098) rather than from the joint bread/meat
    directly -- see test_adjusted_sandwich_matches_explicit_meat_adjustments
    below.

    form_adjusted_meat_adjustments_directly (called internally when
    form_adjusted_meat_adjustments_explicitly=True) ends by design in a bare
    breakpoint() -- see its own docstring -- which would hang a test, so the
    caller must monkeypatch it to a no-op before calling this.
    """
    beta_hat_jnp = jnp.array(beta_hat, dtype=jnp.float32)
    theta_hat_jnp = jnp.array(theta_hat, dtype=jnp.float32)
    all_post_update_betas = jnp.stack([beta_hat_jnp])

    initial_policy_num = 1
    updated_policy_num = 2
    beta_index_by_policy_num = {updated_policy_num: 0}

    def intercept_action_args(beta_for_denominator):
        return (
            beta_for_denominator,
            LOWER_CLIP,
            STEEPNESS,
            UPPER_CLIP,
            jnp.array([1.0]),
        )

    # At t=1 no update has happened yet, so the policy is the fixed, non-adaptive
    # initial one (encoded here via beta=0, which yields action probability 0.5 with
    # synthetic_get_action_1_prob_pure); at t=2 the policy is governed by beta_hat,
    # so the raw ("denominator") beta placed here must match beta_hat exactly for
    # the Radon-Nikodym weight to evaluate to 1 when differentiating at beta_hat.
    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {
            1: intercept_action_args(jnp.zeros(2, dtype=jnp.float32)),
            2: intercept_action_args(jnp.zeros(2, dtype=jnp.float32)),
        },
        2: {
            1: intercept_action_args(beta_hat_jnp),
            2: intercept_action_args(beta_hat_jnp),
        },
    }
    policy_num_by_decision_time_by_subject_id = {
        1: {1: initial_policy_num, 2: updated_policy_num},
        2: {1: initial_policy_num, 2: updated_policy_num},
    }
    action_by_decision_time_by_subject_id = {
        1: {1: 0, 2: 1},
        2: {1: 1, 2: 0},
    }

    def rl_update_args(user_id):
        user_mask = USER_IDS == user_id
        t1_index = np.nonzero(user_mask & (CALENDAR_T == 1))[0][0]
        return (
            jnp.zeros(2, dtype=jnp.float32),
            jnp.array([[1.0]], dtype=jnp.float32),
            jnp.array([[1.0]], dtype=jnp.float32),
            jnp.array([[ACTIONS[t1_index]]], dtype=jnp.float32),
            jnp.array([[REWARDS[t1_index]]], dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            jnp.zeros((1, 1), dtype=jnp.float32),
            False,
        )

    update_func_args_by_by_subject_id_by_policy_num = {
        updated_policy_num: {
            1: rl_update_args(1),
            2: rl_update_args(2),
        }
    }

    def inference_args(user_id):
        user_mask = USER_IDS == user_id
        return (
            jnp.zeros(2, dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.array(ACTIONS[user_mask], dtype=jnp.float32).reshape(-1, 1),
            jnp.array(REWARDS[user_mask], dtype=jnp.float32).reshape(-1, 1),
            jnp.zeros((2, 1), dtype=jnp.float32),
            jnp.zeros((2, 1), dtype=jnp.float32),
            False,
        )

    inference_func_args_by_subject_id = {
        1: inference_args(1),
        2: inference_args(2),
    }

    analysis_df = pd.DataFrame(
        {
            "user_id": USER_IDS,
            "calendar_t": CALENDAR_T,
            "action": ACTIONS,
            "in_study": np.ones_like(ACTIONS),
            # These columns match synthetic_get_least_squares_loss_rl's own
            # parameter names exactly (rather than the generic
            # action_col_name/etc. above): calculate_inference_loss_derivatives
            # (called by form_adjusted_meat_adjustments_directly, unlike the
            # main pipeline) reconstructs each subject's inference-function
            # args by looking up analysis_df columns named after inference_func's
            # own parameters -- see get_study_df_column. Values mirror
            # inference_args(user_id) above: intercept-only states (=1),
            # this subject's own actions/rewards, and no action-centering.
            "base_states": np.ones_like(ACTIONS),
            "treat_states": np.ones_like(ACTIONS),
            "actions": ACTIONS,
            "rewards": REWARDS,
            "action1probs": np.zeros_like(ACTIONS),
            "action1probtimes": np.zeros_like(ACTIONS),
            "action_centering": np.zeros_like(ACTIONS),
        }
    )

    (
        _joint_bread_matrix,
        _joint_adjusted_meat_matrix,
        joint_sandwich_matrix,
        classical_bread_matrix,
        _classical_meat_matrix,
        _classical_sandwich,
        _per_subject_estimating_function_stacks,
        per_subject_adjusted_meat_contributions,
    ) = post_deployment_analysis.construct_classical_and_adjusted_sandwiches(
        theta_hat_jnp,
        all_post_update_betas,
        jnp.array([1, 2]),
        synthetic_get_action_1_prob_pure,
        0,  # action_prob_func_args_beta_index
        synthetic_get_least_squares_loss_rl,
        FunctionTypes.LOSS,
        0,  # alg_update_func_args_beta_index
        -1,  # alg_update_func_args_action_prob_index (unused: no action-centering)
        -1,  # alg_update_func_args_action_prob_times_index (unused)
        -1,  # alg_update_func_args_previous_betas_index (only one update)
        synthetic_get_least_squares_loss_rl,
        FunctionTypes.LOSS,
        0,  # inference_func_args_theta_index
        -1,  # inference_func_args_action_prob_index (unused: no action-centering)
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_subject_id,
        {},  # inference_action_prob_decision_times_by_subject_id (unused)
        update_func_args_by_by_subject_id_by_policy_num,
        action_by_decision_time_by_subject_id,
        True,  # suppress_all_data_checks
        True,  # suppress_interactive_data_checks
        True,  # form_adjusted_meat_adjustments_explicitly
        analysis_df,
        "in_study",  # active_col_name
        "action",  # action_col_name
        "calendar_t",  # calendar_t_col_name
        "user_id",  # subject_id_col_name
        # Same dict as action_prob_func_args_by_subject_id_by_decision_time above:
        # calculate_pi_and_weight_gradients (called by
        # form_adjusted_meat_adjustments_directly) expects exactly that
        # {calendar_t: {subject_id: args}} shape.
        action_prob_func_args_by_subject_id_by_decision_time,
        None,  # action_prob_col_name (unused: no action-centering)
    )

    theta_only_adjusted_sandwich = joint_sandwich_matrix[-2:, -2:]
    return (
        theta_only_adjusted_sandwich,
        classical_bread_matrix,
        per_subject_adjusted_meat_contributions,
    )


def test_adjusted_sandwich_matches_explicit_meat_adjustments(
    beta_and_theta_estimates, monkeypatch
):
    """
    The theta-only adjusted sandwich, formed the normal way (the theta-theta
    block of the full joint bread^-1 @ meat @ bread^-T sandwich), must equal
    the same sandwich formed via the *explicit* per-subject meat-adjustment
    construction (form_adjusted_meat_adjustments_directly,
    form_adjusted_meat_adjustments_explicitly=True): classical_bread^-1 @
    adjusted_meat @ classical_bread^-T, where adjusted_meat is the average
    outer product of each subject's inference estimating function plus an
    explicit correction term. This equivalence is eq. 5.3 of
    https://arxiv.org/pdf/2202.07098 -- two different-looking formulas for the
    same theta-only adjusted covariance -- and is exactly the cross-check
    construct_classical_and_adjusted_sandwiches already computes internally
    when form_adjusted_meat_adjustments_explicitly=True, but only soft-warns
    on (see its own comments); this test promotes that into a real assertion.

    form_adjusted_meat_adjustments_directly ends by design in a bare
    breakpoint() (see its own docstring) -- monkeypatched to a no-op here
    purely so the test can run to completion. The whole point of that
    mechanism is interactive-only inspection of intermediate variables; a test
    couldn't and shouldn't assert on it.
    """
    monkeypatch.setattr(
        "lifejacket.form_adjusted_meat_adjustments_directly.breakpoint",
        lambda: None,
        raising=False,  # breakpoint() is a builtin, not already a module attribute
    )

    (
        theta_only_adjusted_sandwich,
        classical_bread_matrix,
        per_subject_adjusted_meat_contributions,
    ) = _run_package_pipeline_with_meat_adjustments(*beta_and_theta_estimates)

    theta_only_adjusted_meat_from_adjustments = jnp.mean(
        per_subject_adjusted_meat_contributions, axis=0
    )
    theta_only_adjusted_sandwich_from_adjustments = (
        post_deployment_analysis.form_sandwich_from_bread_and_meat(
            classical_bread_matrix,
            theta_only_adjusted_meat_from_adjustments,
            2,  # num_subjects
            method=SandwichFormationMethods.BREAD_T_QR,
        )
    )

    np.testing.assert_allclose(
        theta_only_adjusted_sandwich_from_adjustments,
        theta_only_adjusted_sandwich,
        rtol=1e-4,
        atol=1e-6,
    )
