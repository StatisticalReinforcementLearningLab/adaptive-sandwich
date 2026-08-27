"""
Regression test extracted from the "Nowell_Kelly_Small_Example_Calculations" notebook.

That notebook was written to independently verify the adaptive sandwich variance
computation for a small, fully-specified deployment (2 users, T=2 decision points,
one RL update, a Boltzmann/sigmoid policy with steepness > 0, least squares on both
the RL and inference sides, and no action-centering on either side) after observing
that the average adaptive sandwich variance was not sufficiently close to the
empirical variance in larger simulations with these settings.

Two independent checks are made against the same real lifejacket pipeline, run on
this toy deployment:

1. Rather than typing in the notebook's hand-derived intermediate numbers, the
   first test recomputes the bread, meat, and sandwich from first principles with
   plain numpy/sklearn (closed-form OLS and the sigmoid-policy derivative
   formulas) and compares that independent computation against what
   lifejacket.post_deployment_analysis actually produces.
2. The second test does type in literal numbers: the notebook's own recorded
   (pdb-captured) values from an actual historical run, used as an independent
   check that doesn't rely on this file's closed-form reimplementation of the
   math (test 1) being correct.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import expit
from sklearn.linear_model import LinearRegression

from lifejacket import post_deployment_analysis
from lifejacket.constants import FunctionTypes
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)
from simulators_and_runners.functions_to_pass_to_analysis.synthetic_get_least_squares_loss_rl import (
    synthetic_get_least_squares_loss_rl,
)

# Toy deployment settings from the notebook's "Particular Setting" section: T=2
# decision points, n=2 users, one RL update (after t=1), a sigmoid/Boltzmann policy
# using only the intercept as state (so beta and theta are each 2-dimensional: one
# non-action coefficient, one action coefficient), and no action-centering on either
# the RL or the inference side. The notebook's args namespace lists steepness=2.5,
# but the actual calculations (and the action probability of 0.5927150249 at t=2)
# only reproduce with steepness=0.5, so that is the value used here.
STEEPNESS = 0.5
LOWER_CLIP = 0.1
UPPER_CLIP = 0.9

# The single, seeded local simulation the notebook worked from (env=1713299542,
# alg=1713304542).
USER_IDS = np.array([1, 1, 2, 2])
CALENDAR_T = np.array([1, 2, 1, 2])
ACTIONS = np.array([0.0, 1.0, 1.0, 0.0])
REWARDS = np.array([-0.8565283183, -1.3169794352, -0.1061271784, -0.6967024415])

# Frozen values from the notebook's "Existing Code" section: an actual, independent
# run of this package on the same toy deployment, captured via pdb at the time the
# notebook was written. These are copied verbatim (not derived from the formulas
# below), so agreement with them is a check against the notebook's real historical
# artifact, not just against this file's own closed-form reimplementation of the
# math. The near-zero entries in the meat matrix's beta-beta and beta-theta blocks
# are floating point noise (the beta-side model is exactly identified with 2 users
# and 2 parameters, so its estimating function is analytically zero at beta_hat);
# their exact magnitude is not meaningful and is allowed a generous atol below.
NOTEBOOK_BREAD = np.array(
    [
        [2.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.34266952, 4.0, 2.0],
        [0.0, 0.30271304, 2.0, 2.0],
    ]
)
NOTEBOOK_MEAT = np.array(
    [
        [7.1054274e-15, 7.1054274e-15, -8.1698779e-08, -7.2172412e-08],
        [7.1054274e-15, 7.1054274e-15, -8.1698779e-08, -7.2172412e-08],
        [-8.1698779e-08, -8.1698779e-08, 1.8787583, 1.6596885],
        [-7.2172412e-08, -7.2172412e-08, 1.6596885, 1.4661629],
    ]
)
NOTEBOOK_THETA_SANDWICH = np.array(
    [
        [0.00319304, 0.02099766],
        [0.02099766, 0.13808201],
    ]
)


@pytest.fixture
def beta_and_theta_estimates():
    """
    Beta and theta estimates for the toy deployment, via plain OLS: beta_hat from
    the single user pair's t=1 data (the only data that fed the one RL update), and
    theta_hat from the full 4-row history. This matches the notebook's own
    "Beta Estimate"/"Theta Estimate" sections (the notebook additionally cross-
    checked these against sklearn, which is what's used directly here).
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


def _independent_bread_meat_sandwich(beta_hat, theta_hat):
    """
    Independently derives the joint bread, joint meat, and theta-only adjusted
    sandwich for the toy deployment above, using only plain numpy.

    This mirrors, but does not call, the actual estimating/loss functions handed
    to the package in _run_package_pipeline below -- it is meant to be a
    from-first-principles check, not a restatement of the code under test. The
    formulas follow the notebook's "Conceptual Setting" section: least squares loss
    on both sides gives an estimating function -2*sum((r - theta.x)*x), its Hessian
    is 2*sum(x x^T), and the joint bread's lower-left block is the average, over
    users, of the outer product of each user's inference estimating function value
    with the gradient (wrt beta) of the Radon-Nikodym weight applied to their
    post-update data.
    """
    num_subjects = 2
    t1_mask = CALENDAR_T == 1
    design = np.column_stack([np.ones_like(ACTIONS), ACTIONS])

    # Action-1 probability governing the (only) post-update decision time, t=2.
    # State is the intercept only (=1), so the action coefficient beta_hat[1] is
    # the only piece of beta that matters here.
    pi_t2 = np.clip(expit(STEEPNESS * beta_hat[1]), LOWER_CLIP, UPPER_CLIP)

    # Upper-left block: average (over both users, who both fed the single update)
    # Hessian of the RL least-squares loss wrt beta.
    design_beta = design[t1_mask]
    UL = 2 * design_beta.T @ design_beta / num_subjects

    # Lower-right block: average (over both users) Hessian of the inference
    # least-squares loss wrt theta.
    BR = 2 * design.T @ design / num_subjects

    est_beta_by_user = []
    psi_by_user = []
    weight_grad_by_user = []
    for user_id in (1, 2):
        user_mask = USER_IDS == user_id
        x_user = design[user_mask]
        r_user = REWARDS[user_mask]
        a_user_t2 = ACTIONS[user_mask][1]  # this user's action at t=2

        # RL-side estimating function, evaluated at beta_hat, using only this
        # user's t=1 data point (the only data that fed the single update).
        x_beta = x_user[0]
        est_beta_by_user.append(-2 * (r_user[0] - x_beta @ beta_hat) * x_beta)

        # Inference-side estimating function, evaluated at theta_hat, summed over
        # this user's full history (t=1 and t=2).
        residual = r_user - x_user @ theta_hat
        psi_by_user.append(-2 * (residual @ x_user))

        # Gradient (wrt beta) of the Radon-Nikodym weight applied to this user's
        # t=2 data. Only the action-coefficient slot of beta enters the sigmoid
        # (state is a constant intercept), so the non-action slot's gradient is 0.
        dpi_dbeta1 = STEEPNESS * pi_t2 * (1 - pi_t2)
        sign = 1.0 if a_user_t2 == 1 else -1.0
        q = pi_t2 if a_user_t2 == 1 else (1 - pi_t2)
        weight_grad_by_user.append(np.array([0.0, sign * dpi_dbeta1 / q]))

    # Lower-left block: average, over users, of the outer product of each user's
    # inference estimating function with their Radon-Nikodym weight gradient.
    BL = (
        sum(
            np.outer(psi, wg)
            for psi, wg in zip(psi_by_user, weight_grad_by_user, strict=False)
        )
        / num_subjects
    )

    bread = np.block([[UL, np.zeros((2, 2))], [BL, BR]])

    stacked_by_user = [
        np.concatenate([est_beta, psi])
        for est_beta, psi in zip(est_beta_by_user, psi_by_user, strict=False)
    ]
    meat = sum(np.outer(s, s) for s in stacked_by_user) / num_subjects

    bread_inv = np.linalg.inv(bread)
    joint_sandwich = bread_inv @ meat @ bread_inv.T / num_subjects
    theta_only_sandwich = joint_sandwich[-2:, -2:]

    return bread, meat, theta_only_sandwich


def _run_package_pipeline(beta_hat, theta_hat, jacobian_row_chunk_size=None):
    """
    Runs the actual, real lifejacket pipeline on the toy deployment: builds the
    minimal per-subject argument dictionaries construct_classical_and_adjusted_
    sandwiches needs, reusing the package's own real contract functions
    (synthetic_get_action_1_prob_pure, synthetic_get_least_squares_loss_rl -- the
    latter is generic enough to serve as both the RL loss and the inference loss
    here, since both sides use only the intercept as state), and returns the raw
    joint bread, joint meat, and theta-only adjusted sandwich it computes.

    jacobian_row_chunk_size is passed straight through to
    construct_classical_and_adjusted_sandwiches, so the chunked-backward
    equivalence test below can exercise every representation of the parameter
    against the identical toy deployment.
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

    (
        raw_joint_bread_matrix,
        joint_adjusted_meat_matrix,
        joint_sandwich_matrix,
        _classical_bread_matrix,
        _classical_meat_matrix,
        _classical_sandwich,
        _avg_estimating_function_stack,
        _per_subject_estimating_function_stacks,
        _per_subject_adjusted_corrections,
        _per_subject_classical_corrections,
        _per_subject_adjusted_meat_adjustments,
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
        False,  # form_adjusted_meat_adjustments_explicitly
        None,  # analysis_df (only needed if forming meat adjustments explicitly)
        None,  # active_col_name
        None,  # action_col_name
        None,  # calendar_t_col_name
        None,  # subject_id_col_name
        None,  # action_prob_func_args
        None,  # action_prob_col_name
        jacobian_row_chunk_size=jacobian_row_chunk_size,
    )

    theta_only_adjusted_sandwich = joint_sandwich_matrix[-2:, -2:]
    return (
        raw_joint_bread_matrix,
        joint_adjusted_meat_matrix,
        theta_only_adjusted_sandwich,
    )


@pytest.fixture
def package_pipeline_outputs(beta_and_theta_estimates):
    return _run_package_pipeline(*beta_and_theta_estimates)


def test_adaptive_sandwich_matches_independent_closed_form_calculation(
    package_pipeline_outputs, beta_and_theta_estimates
):
    """
    The package's bread/meat/theta-sandwich for the toy deployment should match a
    from-scratch, plain-numpy closed-form derivation of the same quantities.
    """
    bread, meat, theta_sandwich = package_pipeline_outputs
    expected_bread, expected_meat, expected_theta_sandwich = (
        _independent_bread_meat_sandwich(*beta_and_theta_estimates)
    )

    np.testing.assert_allclose(bread, expected_bread, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(meat, expected_meat, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(
        theta_sandwich, expected_theta_sandwich, rtol=1e-3, atol=1e-6
    )


def test_adaptive_sandwich_matches_notebook_recorded_values(package_pipeline_outputs):
    """
    The package's bread/meat/theta-sandwich for the toy deployment should also
    match the notebook's own recorded, pdb-captured numbers from an actual
    historical run -- an independent check that doesn't depend on this file's
    closed-form reimplementation of the math (see the test above) being correct.
    """
    bread, meat, theta_sandwich = package_pipeline_outputs

    np.testing.assert_allclose(bread, NOTEBOOK_BREAD, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(meat, NOTEBOOK_MEAT, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        theta_sandwich, NOTEBOOK_THETA_SANDWICH, rtol=1e-4, atol=1e-6
    )


def test_jacobian_row_chunk_size_representations_numerically_identical(
    package_pipeline_outputs, beta_and_theta_estimates
):
    """
    Every representation of jacobian_row_chunk_size must produce the same
    numbers on the identical toy deployment:

    - None (the default) = auto, which at this tiny out_dim (=4, far below
      the unchunked threshold) resolves to the single unchunked eager
      jax.vmap(pullback) -- the pre-auto default path.
    - 0 = force that same unchunked path explicitly, so it must be
      bitwise-identical to the auto default here.
    - 3 = an explicit chunk size (with a shorter remainder chunk of 1),
      exercising the jitted chunked backward path -- mathematically
      identical, allowed float32 noise only.
    """
    forced_unchunked = _run_package_pipeline(
        *beta_and_theta_estimates, jacobian_row_chunk_size=0
    )
    chunked = _run_package_pipeline(
        *beta_and_theta_estimates, jacobian_row_chunk_size=3
    )

    for auto_value, forced_value in zip(
        package_pipeline_outputs, forced_unchunked, strict=True
    ):
        np.testing.assert_array_equal(np.asarray(auto_value), np.asarray(forced_value))
    for auto_value, chunked_value in zip(
        package_pipeline_outputs, chunked, strict=True
    ):
        np.testing.assert_allclose(
            np.asarray(auto_value),
            np.asarray(chunked_value),
            rtol=1e-5,
            atol=1e-6,
        )
