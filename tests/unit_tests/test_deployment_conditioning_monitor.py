import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket.deployment_conditioning_monitor import DeploymentConditioningMonitor
from lifejacket.post_deployment_analysis import analyze_dataset


# Mock functions and arguments
def action_prob_func(beta, features):
    return jax.nn.sigmoid(beta @ features)


def alg_update_func(beta, states, actions, rewards):
    return jnp.sum((rewards - states @ beta) ** 2)


def inference_loss_func(theta, reward, states):
    return (reward - states @ theta) ** 2


@pytest.mark.skip(
    reason=(
        "Stale test: references analyze_dataset()['adjusted_bread_RL_block'], a key "
        "that does not exist in its current return dict (only theta_est, "
        "adjusted_sandwich_var_estimate, classical_sandwich_var_estimate), and passes "
        "inference_func=None / theta_calculation_func=None, which analyze_dataset calls "
        "unconditionally. Needs a rewrite against the current API, independent of the "
        "jnp.block incremental-stitching fix (see the sibling test in this file, which "
        "is unskipped and passing)."
    )
)
def test_incremental_phi_dot_bar_consistency_with_after_study_analysis():
    """
    Test that phi dot bars from incremental assess_update calls match
    successive upper left prefixes of the final RL bread.
    """
    # Mock study data with multiple updates
    n_users = 6
    n_decision_times = 9
    beta_dim = 3
    n_policies = 3

    # Create deterministic synthetic study dataframe
    study_data = []
    for user_id in range(n_users):
        for t in range(n_decision_times):
            policy_num = t // 3  # Policy changes every 3 time steps
            study_data.append(
                {
                    "user_id": user_id,
                    "calendar_t": t,
                    "in_study": 1,
                    "action": (user_id + t) % 2,  # Deterministic binary actions
                    "policy_num": policy_num,
                    "action_prob": 0.6,
                    "reward": 1.0 + 0.1 * user_id + 0.05 * t,  # Deterministic rewards
                }
            )

    study_df = pd.DataFrame(study_data)

    # Deterministic function arguments
    action_prob_func_args = {
        t: {
            0: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
            1: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
            2: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
            3: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
            4: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
            5: (jnp.array([0.1, 0.2, 0.3]), jnp.array([1.0, 1.5, 2.0])),
        }
        for t in range(n_decision_times)
    }

    # Create deterministic update function arguments
    alg_update_func_args = {}
    for policy_num in range(1, n_policies):
        alg_update_func_args[policy_num] = {}
        for user_id in range(n_users):
            alg_update_func_args[policy_num][user_id] = (
                jnp.array([0.1, 0.2, 0.3]),  # beta
                jnp.array([1.0 + user_id, 2.0 + user_id, 3.0 + user_id]),  # states
                jnp.array([0, 1, (user_id % 2)]),  # actions
                jnp.array([1.0 + user_id, 1.1 + user_id, 1.2 + user_id]),  # rewards
            )

    # Initialize monitor
    monitor = DeploymentConditioningMonitor()

    # Collect phi_dot_bars from incremental assess_update calls
    incremental_phi_dot_bars = []

    for policy_num in range(1, n_policies + 1):
        # Filter study_df to only include data up to previous policy
        filtered_df = study_df[study_df["policy_num"] < policy_num].copy()

        # Truncate args to only include policies up to policy_num
        truncated_action_prob_func_args = {
            k: v for k, v in action_prob_func_args.items() if k <= policy_num
        }

        truncated_alg_update_func_args = {
            k: v for k, v in alg_update_func_args.items() if k <= policy_num
        }

        monitor.assess_update(
            proposed_policy_num=policy_num,
            analysis_df=filtered_df,
            action_prob_func=action_prob_func,
            action_prob_func_args=truncated_action_prob_func_args,
            action_prob_func_args_beta_index=0,
            alg_update_func=alg_update_func,
            alg_update_func_type="loss",
            alg_update_func_args=truncated_alg_update_func_args,
            alg_update_func_args_beta_index=0,
            alg_update_func_args_action_prob_index=-1,
            alg_update_func_args_action_prob_times_index=-1,
            alg_update_func_args_previous_betas_index=-1,
            active_col_name="in_study",
            action_col_name="action",
            policy_num_col_name="policy_num",
            calendar_t_col_name="calendar_t",
            subject_id_col_name="user_id",
            action_prob_col_name="action_prob",
            suppress_interactive_data_checks=True,
            suppress_all_data_checks=True,
            incremental=True,
        )

        # Store the phi_dot_bar from this update
        incremental_phi_dot_bars.append(monitor.latest_phi_dot_bar)

    # Now run the full after_study_analysis to get the final RL bread
    final_analysis_results = analyze_dataset(
        output_dir=pathlib.Path("."),
        analysis_df=study_df,
        action_prob_func=action_prob_func,
        action_prob_func_args=action_prob_func_args,
        action_prob_func_args_beta_index=0,
        alg_update_func=alg_update_func,
        alg_update_func_type="loss",
        alg_update_func_args=alg_update_func_args,
        alg_update_func_args_beta_index=0,
        alg_update_func_args_action_prob_index=-1,
        alg_update_func_args_action_prob_times_index=-1,
        alg_update_func_args_previous_betas_index=-1,
        inference_func=None,
        inference_func_type="none",
        inference_func_args_theta_index=0,
        theta_calculation_func=None,
        active_col_name="in_study",
        action_col_name="action",
        policy_num_col_name="policy_num",
        calendar_t_col_name="calendar_t",
        subject_id_col_name="user_id",
        action_prob_col_name="action_prob",
        reward_col_name="reward",
        suppress_interactive_data_checks=True,
        suppress_all_data_checks=True,
        small_sample_correction="none",
        collect_data_for_blowup_supervised_learning=False,
        form_adjusted_meat_adjustments_explicitly=False,
        stabilize_joint_bread=False,
    )

    # Extract the RL portion of the final bread
    final_rl_bread_inv = final_analysis_results["adjusted_bread_RL_block"]

    # Verify that each incremental phi_dot_bar matches the corresponding upper-left prefix
    for i, phi_dot_bar in enumerate(incremental_phi_dot_bars):
        expected_size = (i + 1) * beta_dim

        # Extract the upper-left prefix of the final RL bread
        expected_prefix = final_rl_bread_inv[:expected_size, :expected_size]

        # Compare with the incremental phi_dot_bar
        assert phi_dot_bar.shape == expected_prefix.shape, (
            f"Shape mismatch at update {i + 1}: got {phi_dot_bar.shape}, "
            f"expected {expected_prefix.shape}"
        )

        np.testing.assert_allclose(
            phi_dot_bar,
            expected_prefix,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Phi dot bar mismatch at update {i + 1}",
        )


def test_incremental_phi_dot_bar_consistency_with_non_incremental():
    """
    Test that non-incremental and incremental assess_update calls produce consistent results.
    """
    # Mock study data spanning three non-initial policy updates (policy 0 is the
    # initial policy; policies 1, 2, 3 are each produced by an update). Three
    # updates specifically so the third incremental call must stitch a cached
    # (2*beta_dim, 2*beta_dim) block onto a new (beta_dim, 3*beta_dim) block row
    # -- a non-square-by-construction case, unlike a single update where the
    # cached and new blocks happen to have matching dimensions and a shape bug
    # in the stitching could go unnoticed.
    n_users = 4
    n_decision_times = 12
    beta_dim = 2
    n_policies = 4

    # Create deterministic synthetic study dataframe
    study_data = []
    for user_id in range(n_users):
        for t in range(n_decision_times):
            policy_num = t // 3  # Policy changes every 3 time steps
            study_data.append(
                {
                    "user_id": user_id,
                    "calendar_t": t,
                    "in_study": 1,
                    "action": (user_id * 2 + t) % 2,  # Deterministic binary actions
                    "policy_num": policy_num,
                    "action_prob": 0.7,
                    "reward": 2.0 + 0.2 * user_id - 0.1 * t,  # Deterministic rewards
                }
            )

    study_df = pd.DataFrame(study_data)

    # action_prob_func_args must be nested as {decision_time: {subject_id:
    # args_tuple}} to match calculate_beta_dim and the rest of the pipeline's
    # expected structure, and must have an entry for every decision time that
    # will appear in a truncated analysis_df, since thread_action_prob_func_args
    # looks up policy_num_by_decision_time_by_subject_id for every decision time
    # it's given.
    action_prob_func_args = {
        t: {
            user_id: (jnp.array([0.5, 0.8]), jnp.array([1.2, 1.8]))
            for user_id in range(n_users)
        }
        for t in range(n_decision_times)
    }

    # alg_update_func_args is keyed by the (non-initial) policy number an update
    # produces -- policy 0 is initial and has no entry, matching
    # construct_beta_index_by_policy_num_map's convention.
    alg_update_func_args = {
        policy_num: {
            user_id: (
                jnp.array([0.2 * (user_id + 1), 0.4 * (user_id + 1)]),  # beta
                jnp.array([2.0 + user_id, 3.0 + user_id]),  # states
                jnp.array([user_id % 2, (user_id + 1) % 2]),  # actions
                jnp.array([2.0 + user_id, 2.2 + user_id]),  # rewards
            )
            for user_id in range(n_users)
        }
        for policy_num in range(1, n_policies)
    }

    monitor_incremental = DeploymentConditioningMonitor()
    monitor_non_incremental = DeploymentConditioningMonitor()

    # Run both incremental and non-incremental versions
    for proposed_policy_num in range(1, n_policies):
        # Filter study_df to only include data available before this policy
        # took effect.
        filtered_df = study_df[study_df["policy_num"] < proposed_policy_num].copy()
        available_times = set(filtered_df["calendar_t"].unique())

        truncated_action_prob_func_args = {
            t: args for t, args in action_prob_func_args.items() if t in available_times
        }
        # Includes the args for the update that PRODUCES proposed_policy_num
        # itself, not just prior updates -- alg_update_func_args[k] holds the
        # data used to fit policy k's beta, which construct_phi_dot_bar_so_far
        # needs to compute that beta's Jacobian block row.
        truncated_alg_update_func_args = {
            k: v for k, v in alg_update_func_args.items() if k <= proposed_policy_num
        }

        common_kwargs = {
            "proposed_policy_num": proposed_policy_num,
            "analysis_df": filtered_df,
            "action_prob_func": action_prob_func,
            "action_prob_func_args": truncated_action_prob_func_args,
            "action_prob_func_args_beta_index": 0,
            "alg_update_func": alg_update_func,
            "alg_update_func_type": "loss",
            "alg_update_func_args": truncated_alg_update_func_args,
            "alg_update_func_args_beta_index": 0,
            "alg_update_func_args_action_prob_index": -1,
            "alg_update_func_args_action_prob_times_index": -1,
            "alg_update_func_args_previous_betas_index": -1,
            "active_col_name": "in_study",
            "action_col_name": "action",
            "policy_num_col_name": "policy_num",
            "calendar_t_col_name": "calendar_t",
            "subject_id_col_name": "user_id",
            "action_prob_col_name": "action_prob",
            "suppress_interactive_data_checks": True,
            "suppress_all_data_checks": True,
        }

        monitor_incremental.assess_update(**common_kwargs, incremental=True)
        monitor_non_incremental.assess_update(**common_kwargs, incremental=False)

        expected_shape = (
            proposed_policy_num * beta_dim,
            proposed_policy_num * beta_dim,
        )
        assert monitor_incremental.latest_phi_dot_bar.shape == expected_shape, (
            f"Incremental phi_dot_bar has shape "
            f"{monitor_incremental.latest_phi_dot_bar.shape} at policy "
            f"{proposed_policy_num}, expected square {expected_shape} -- a "
            "non-square result here is exactly the jnp.block flat-list bug "
            "this test guards against."
        )

        # They should produce identical phi_dot_bars
        np.testing.assert_allclose(
            monitor_incremental.latest_phi_dot_bar,
            monitor_non_incremental.latest_phi_dot_bar,
            rtol=1e-6,
            atol=1e-6,
            err_msg=f"Incremental vs non-incremental mismatch at policy {proposed_policy_num}",
        )
