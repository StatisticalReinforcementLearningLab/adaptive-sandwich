from __future__ import annotations

import logging
import math
from typing import Any
import collections

import numpy as np
from scipy.special import logit
import plotext as plt
import jax
from jax import numpy as jnp
import pandas as pd

from . import post_deployment_analysis
from .constants import FunctionTypes
from .vmap_helpers import stack_batched_arg_lists_into_tensors

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def get_datum_for_blowup_supervised_learning(
    joint_adjusted_bread_matrix,
    joint_adjusted_bread_cond,
    avg_estimating_function_stack,
    per_subject_estimating_function_stacks,
    all_post_update_betas,
    analysis_df,
    active_col_name,
    calendar_t_col_name,
    action_prob_col_name,
    subject_id_col_name,
    reward_col_name,
    theta_est,
    adjusted_sandwich_var_estimate,
    subject_ids,
    beta_dim,
    theta_dim,
    initial_policy_num,
    beta_index_by_policy_num,
    policy_num_by_decision_time_by_subject_id,
    theta_calculation_func,
    action_prob_func,
    action_prob_func_args_beta_index,
    inference_func,
    inference_func_type,
    inference_func_args_theta_index,
    inference_func_args_action_prob_index,
    inference_action_prob_decision_times_by_subject_id,
    action_prob_func_args,
    action_by_decision_time_by_subject_id,
) -> dict[str, Any]:
    """
    Collects a datum for supervised learning about adjusted sandwich blowup.

    The datum consists of features and the raw adjusted sandwich variance estimate as a label.

    A few plots are produced along the way to help visualize the data.

    Args:
        joint_adjusted_bread_matrix (jnp.ndarray):
            The joint adjusted bread matrix.
        joint_adjusted_bread_cond (float):
            The condition number of the joint adjusted bread matrix.
        avg_estimating_function_stack (jnp.ndarray):
            The average estimating function stack across subjects.
        per_subject_estimating_function_stacks (jnp.ndarray):
            The estimating function stacks for each subject.
        all_post_update_betas (jnp.ndarray):
            All post-update beta parameters.
        analysis_df (pd.DataFrame):
            The study DataFrame.
        active_col_name (str):
            Column name indicating if a subject is in the study in the study dataframe.
        calendar_t_col_name (str):
            Column name for calendar time in the study dataframe.
        action_prob_col_name (str):
            Column name for action probabilities in the study dataframe.
        subject_id_col_name (str):
            Column name for subject IDs in the study dataframe
        reward_col_name (str):
            Column name for rewards in the study dataframe.
        theta_est (jnp.ndarray):
            The estimate of the parameter vector theta.
        adjusted_sandwich_var_estimate (jnp.ndarray):
            The adjusted sandwich variance estimate for theta.
        subject_ids (jnp.ndarray):
            Array of unique subject IDs.
        beta_dim (int):
            Dimension of the beta parameter vector.
        theta_dim (int):
            Dimension of the theta parameter vector.
        initial_policy_num (int | float):
            The initial policy number used in the study.
        beta_index_by_policy_num (dict[int | float, int]):
            Mapping from policy numbers to indices in all_post_update_betas.
        policy_num_by_decision_time_by_subject_id (dict):
            Mapping from subject IDs to their policy numbers by decision time.
        theta_calculation_func (callable):
            The theta calculation function.
        action_prob_func (callable):
            The action probability function.
        action_prob_func_args_beta_index (int):
            Index for beta in action probability function arguments.
        inference_func (callable):
            The inference function.
        inference_func_type (str):
            Type of the inference function.
        inference_func_args_theta_index (int):
            Index for theta in inference function arguments.
        inference_func_args_action_prob_index (int):
            Index for action probability in inference function arguments.
        inference_action_prob_decision_times_by_subject_id (dict):
            Mapping from subject IDs to decision times for action probabilities used in inference.
        action_prob_func_args (dict):
            Arguments for the action probability function.
        action_by_decision_time_by_subject_id (dict):
            Mapping from subject IDs to their actions by decision time.
    Returns:
        dict[str, Any]: A dictionary containing features and the label for supervised learning.
    """
    num_diagonal_blocks = (
        (joint_adjusted_bread_matrix.shape[0] - theta_dim) // beta_dim
    ) + 1
    diagonal_block_sizes = ([beta_dim] * (num_diagonal_blocks - 1)) + [theta_dim]

    block_bounds = np.cumsum([0] + list(diagonal_block_sizes))
    num_block_rows_cols = len(diagonal_block_sizes)

    # collect diagonal and sub-diagonal block norms and diagonal condition numbers
    off_diag_block_norms = {}
    diag_norms = []
    diag_conds = []
    off_diag_row_norms = np.zeros(num_block_rows_cols)
    off_diag_col_norms = np.zeros(num_block_rows_cols)
    for i in range(num_block_rows_cols):
        for j in range(num_block_rows_cols):
            if i > j:  # below-diagonal blocks
                row_slice = slice(block_bounds[i], block_bounds[i + 1])
                col_slice = slice(block_bounds[j], block_bounds[j + 1])
                block_norm = np.linalg.norm(
                    joint_adjusted_bread_matrix[row_slice, col_slice],
                    ord="fro",
                )
                # We will sum here and take the square root later
                off_diag_row_norms[i] += block_norm
                off_diag_col_norms[j] += block_norm
                off_diag_block_norms[(i, j)] = block_norm

        # handle diagonal blocks
        sl = slice(block_bounds[i], block_bounds[i + 1])
        diag_norms.append(
            np.linalg.norm(joint_adjusted_bread_matrix[sl, sl], ord="fro")
        )
        diag_conds.append(np.linalg.cond(joint_adjusted_bread_matrix[sl, sl]))

    # Sqrt each row/col sum to truly get row/column norms.
    # Perhaps not necessary for learning, but more natural
    off_diag_row_norms = np.sqrt(off_diag_row_norms)
    off_diag_col_norms = np.sqrt(off_diag_col_norms)

    # Get the per-person estimating function stack norms
    estimating_function_stack_norms = np.linalg.norm(
        per_subject_estimating_function_stacks, axis=1
    )

    # Get the average estimating function stack norms by update/inference
    # Use the bounds variable from above to split the estimating function stacks
    # into blocks corresponding to the updates and inference.
    avg_estimating_function_stack_norms_per_segment = [
        np.mean(
            np.linalg.norm(
                per_subject_estimating_function_stacks[
                    :, block_bounds[i] : block_bounds[i + 1]
                ],
                axis=1,
            )
        )
        for i in range(len(block_bounds) - 1)
    ]

    # Compute the norms of each successive difference in all_post_update_betas.
    successive_beta_diffs = np.diff(np.array(all_post_update_betas), axis=0)
    successive_beta_diff_norms = np.linalg.norm(successive_beta_diffs, axis=1)
    max_successive_beta_diff_norm = np.max(successive_beta_diff_norms)
    std_successive_beta_diff_norm = np.std(successive_beta_diff_norms)

    # Add a column with logits of the action probabilities
    # Compute the average and standard deviation of the logits of the action probabilities at each decision time using analysis_df
    # action_prob_logit_means and action_prob_logit_stds are numpy arrays of mean and stddev at each decision time
    # Only compute logits for rows where subject is in the study; set others to NaN
    in_study_mask = analysis_df[active_col_name] == 1
    analysis_df["action_prob_logit"] = np.where(
        in_study_mask,
        logit(analysis_df[action_prob_col_name]),
        np.nan,
    )
    grouped_action_prob_logit = analysis_df.loc[in_study_mask].groupby(
        calendar_t_col_name
    )["action_prob_logit"]
    action_prob_logit_means_by_t = grouped_action_prob_logit.mean().values
    action_prob_logit_stds_by_t = grouped_action_prob_logit.std().values

    # Compute the average and standard deviation of the rewards at each decision time using analysis_df
    # reward_means and reward_stds are numpy arrays of mean and stddev at each decision time
    grouped_reward = analysis_df.loc[in_study_mask].groupby(calendar_t_col_name)[
        reward_col_name
    ]
    reward_means_by_t = grouped_reward.mean().values
    reward_stds_by_t = grouped_reward.std().values

    joint_bread_min_singular_value = np.linalg.svd(
        joint_adjusted_bread_matrix, compute_uv=False
    )[-1]

    max_reward = analysis_df.loc[in_study_mask][reward_col_name].max()

    norm_avg_estimating_function_stack = np.linalg.norm(avg_estimating_function_stack)
    max_estimating_function_stack_norm = np.max(estimating_function_stack_norms)

    (
        premature_thetas,
        premature_adjusted_sandwiches,
        premature_classical_sandwiches,
        premature_joint_adjusted_bread_condition_numbers,
        premature_avg_inference_estimating_functions,
    ) = calculate_sequence_of_premature_adjusted_estimates(
        analysis_df,
        initial_policy_num,
        beta_index_by_policy_num,
        policy_num_by_decision_time_by_subject_id,
        theta_calculation_func,
        calendar_t_col_name,
        action_prob_col_name,
        subject_id_col_name,
        active_col_name,
        all_post_update_betas,
        subject_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_subject_id,
        action_prob_func_args,
        action_by_decision_time_by_subject_id,
        joint_adjusted_bread_matrix,
        per_subject_estimating_function_stacks,
        beta_dim,
    )

    np.testing.assert_allclose(
        np.zeros_like(premature_avg_inference_estimating_functions),
        premature_avg_inference_estimating_functions,
        atol=1e-3,
    )

    # Plot premature joint adjusted bread log condition numbers
    plt.clear_figure()
    plt.title("Premature Joint Adjusted Bread Inverse Log Condition Numbers")
    plt.xlabel("Premature Update Index")
    plt.ylabel("Log Condition Number")
    plt.scatter(
        np.log(premature_joint_adjusted_bread_condition_numbers),
        color="blue+",
    )
    plt.grid(True)
    plt.xticks(
        range(
            0,
            len(premature_joint_adjusted_bread_condition_numbers),
            max(
                1,
                len(premature_joint_adjusted_bread_condition_numbers) // 10,
            ),
        )
    )
    plt.show()

    # Plot each diagonal element of premature adjusted sandwiches
    num_diag = premature_adjusted_sandwiches.shape[-1]
    for i in range(num_diag):
        plt.clear_figure()
        plt.title(f"Premature Adjusted Sandwich Diagonal Element {i}")
        plt.xlabel("Premature Update Index")
        plt.ylabel(f"Variance (Diagonal {i})")
        plt.scatter(np.array(premature_adjusted_sandwiches[:, i, i]), color="blue+")
        plt.grid(True)
        plt.xticks(
            range(
                0,
                int(premature_adjusted_sandwiches.shape[0]),
                max(1, int(premature_adjusted_sandwiches.shape[0]) // 10),
            )
        )
        plt.show()

        plt.clear_figure()
        plt.title(
            f"Premature Adjusted Sandwich Diagonal Element {i} Ratio to Classical"
        )
        plt.xlabel("Premature Update Index")
        plt.ylabel(f"Variance (Diagonal {i})")
        plt.scatter(
            np.array(premature_adjusted_sandwiches[:, i, i])
            / np.array(premature_classical_sandwiches[:, i, i]),
            color="red+",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                int(premature_adjusted_sandwiches.shape[0]),
                max(1, int(premature_adjusted_sandwiches.shape[0]) // 10),
            )
        )
        plt.show()

        plt.clear_figure()
        plt.title(f"Premature Theta Estimates At Index {i}")
        plt.xlabel("Premature Update Index")
        plt.ylabel(f"Theta element {i}")
        plt.scatter(np.array(premature_thetas[:, i]), color="green+")
        plt.grid(True)
        plt.xticks(
            range(
                0,
                int(premature_adjusted_sandwiches.shape[0]),
                max(1, int(premature_adjusted_sandwiches.shape[0]) // 10),
            )
        )
        plt.show()

        # Grab predictors related to premature Phi-dot-bars
        RL_stack_beta_derivatives_block = joint_adjusted_bread_matrix[
            :-theta_dim, :-theta_dim
        ]
        num_updates = RL_stack_beta_derivatives_block.shape[0] // beta_dim
        premature_RL_block_condition_numbers = []
        premature_RL_block_inverse_norms = []
        diagonal_RL_block_condition_numbers = []
        off_diagonal_RL_scaled_block_norm_sums = []
        for i in range(1, num_updates + 1):
            whole_block_size = i * beta_dim
            whole_block = RL_stack_beta_derivatives_block[
                :whole_block_size, :whole_block_size
            ]
            whole_RL_block_cond_number = np.linalg.cond(whole_block)
            premature_RL_block_condition_numbers.append(whole_RL_block_cond_number)
            logger.info(
                "Condition number of whole RL_stack_beta_derivatives_block (after update %s): %s",
                i,
                whole_RL_block_cond_number,
            )
            diagonal_block = RL_stack_beta_derivatives_block[
                (i - 1) * beta_dim : i * beta_dim, (i - 1) * beta_dim : i * beta_dim
            ]
            diagonal_RL_block_cond_number = np.linalg.cond(diagonal_block)
            diagonal_RL_block_condition_numbers.append(diagonal_RL_block_cond_number)
            logger.info(
                "Condition number of just RL_stack_beta_derivatives_block *diagonal block* for update %s: %s",
                i,
                diagonal_RL_block_cond_number,
            )

            premature_RL_block_inverse_norms.append(
                np.linalg.norm(np.linalg.inv(whole_block))
            )
            logger.info(
                "Norm of inverse of whole RL_stack_beta_derivatives_block (after update %s): %s",
                i,
                premature_RL_block_inverse_norms[-1],
            )

            off_diagonal_RL_scaled_block_norm_sum = 0
            for j in range(1, i):
                off_diagonal_block = RL_stack_beta_derivatives_block[
                    (i - 1) * beta_dim : i * beta_dim, (j - 1) * beta_dim : j * beta_dim
                ]
                off_diagonal_scaled_block_norm = np.linalg.norm(
                    np.linalg.solve(diagonal_block, off_diagonal_block)
                )
                off_diagonal_RL_scaled_block_norm_sum += off_diagonal_scaled_block_norm
            off_diagonal_RL_scaled_block_norm_sums.append(
                off_diagonal_RL_scaled_block_norm_sum
            )
            logger.info(
                "Sum of norms of off-diagonal blocks in row %s scaled by inverse of diagonal block: %s",
                i,
                off_diagonal_RL_scaled_block_norm_sum,
            )
    return {
        **{
            "joint_bread_condition_number": joint_adjusted_bread_cond,
            "joint_bread_min_singular_value": joint_bread_min_singular_value,
            "max_reward": max_reward,
            "norm_avg_estimating_function_stack": norm_avg_estimating_function_stack,
            "max_estimating_function_stack_norm": max_estimating_function_stack_norm,
            "max_successive_beta_diff_norm": max_successive_beta_diff_norm,
            "std_successive_beta_diff_norm": std_successive_beta_diff_norm,
            "label": adjusted_sandwich_var_estimate,
        },
        **{
            f"off_diag_block_{i}_{j}_norm": off_diag_block_norms[(i, j)]
            for i in range(num_block_rows_cols)
            for j in range(i)
        },
        **{f"diag_block_{i}_norm": diag_norms[i] for i in range(num_block_rows_cols)},
        **{f"diag_block_{i}_cond": diag_conds[i] for i in range(num_block_rows_cols)},
        **{
            f"off_diag_row_{i}_norm": off_diag_row_norms[i]
            for i in range(num_block_rows_cols)
        },
        **{
            f"off_diag_col_{i}_norm": off_diag_col_norms[i]
            for i in range(num_block_rows_cols)
        },
        **{
            f"estimating_function_stack_norm_subject_{subject_id}": estimating_function_stack_norms[
                i
            ]
            for i, subject_id in enumerate(subject_ids)
        },
        **{
            f"avg_estimating_function_stack_norm_segment_{i}": avg_estimating_function_stack_norms_per_segment[
                i
            ]
            for i in range(len(avg_estimating_function_stack_norms_per_segment))
        },
        **{
            f"successive_beta_diff_norm_{i}": successive_beta_diff_norms[i]
            for i in range(len(successive_beta_diff_norms))
        },
        **{
            f"action_prob_logit_mean_t_{t}": action_prob_logit_means_by_t[t]
            for t in range(len(action_prob_logit_means_by_t))
        },
        **{
            f"action_prob_logit_std_t_{t}": action_prob_logit_stds_by_t[t]
            for t in range(len(action_prob_logit_stds_by_t))
        },
        **{
            f"reward_mean_t_{t}": reward_means_by_t[t]
            for t in range(len(reward_means_by_t))
        },
        **{
            f"reward_std_t_{t}": reward_stds_by_t[t]
            for t in range(len(reward_stds_by_t))
        },
        **{f"theta_est_{i}": theta_est[i].item() for i in range(len(theta_est))},
        **{
            f"premature_joint_adjusted_bread_condition_number_{i}": premature_joint_adjusted_bread_condition_numbers[
                i
            ]
            for i in range(len(premature_joint_adjusted_bread_condition_numbers))
        },
        **{
            f"premature_adjusted_sandwich_update_{i}_diag_position_{j}": premature_adjusted_sandwich[
                j, j
            ]
            for premature_adjusted_sandwich in premature_adjusted_sandwiches
            for j in range(theta_dim)
        },
        **{
            f"premature_classical_sandwich_update_{i}_diag_position_{j}": premature_classical_sandwich[
                j, j
            ]
            for premature_classical_sandwich in premature_classical_sandwiches
            for j in range(theta_dim)
        },
        **{
            f"off_diagonal_RL_scaled_block_norm_sum_for_update_{i}": off_diagonal_RL_scaled_block_norm_sums[
                i
            ]
            for i in range(len(off_diagonal_RL_scaled_block_norm_sums))
        },
        **{
            f"premature_RL_block_condition_number_after_update_{i}": premature_RL_block_condition_numbers[
                i
            ]
            for i in range(len(premature_RL_block_condition_numbers))
        },
        **{
            f"premature_RL_block_inverse_norm_after_update_{i}": premature_RL_block_inverse_norms[
                i
            ]
            for i in range(len(premature_RL_block_inverse_norms))
        },
    }


def calculate_sequence_of_premature_adjusted_estimates(
    analysis_df: pd.DataFrame,
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    theta_calculation_func: str,
    calendar_t_col_name: str,
    action_prob_col_name: str,
    subject_id_col_name: str,
    active_col_name: str,
    all_post_update_betas: jnp.ndarray,
    subject_ids: jnp.ndarray,
    action_prob_func: str,
    action_prob_func_args_beta_index: int,
    inference_func: str,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    inference_action_prob_decision_times_by_subject_id: dict[
        collections.abc.Hashable, list[int]
    ],
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
    full_joint_adjusted_bread_matrix: jnp.ndarray,
    per_subject_estimating_function_stacks: jnp.ndarray,
    beta_dim: int,
) -> jnp.ndarray:
    """
    Calculates a sequence of premature adjusted estimates for the given study DataFrame, where we
    pretend the study ended after each update in sequence. The behavior of this sequence may provide
    insight into the stability of the final adjusted estimate.

    Args:
        analysis_df (pandas.DataFrame):
            The DataFrame containing the study data.
            initial_policy_num (int | float): The policy number of the initial policy before any updates.
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        policy_num_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of subject ids to dictionaries mapping decision times to the policy number in use.
            Only applies to in-study decision times!
        theta_calculation_func (callable):
            The filename for the theta calculation function.
        calendar_t_col_name (str):
            The name of the column in analysis_df representing calendar time.
        action_prob_col_name (str):
            The name of the column in analysis_df representing action probabilities.
        subject_id_col_name (str):
            The name of the column in analysis_df representing subject IDs.
        active_col_name (str):
            The name of the column in analysis_df indicating whether the subject is in the study at that time.
        all_post_update_betas (jnp.ndarray):
            A NumPy array containing all post-update beta values.
        subject_ids (jnp.ndarray):
            A NumPy array containing all subject IDs in the study.
        action_prob_func (callable):
            The action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        inference_func (callable):
            The inference function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        inference_action_prob_decision_times_by_subject_id (dict[collections.abc.Hashable, list[int]]):
            For each subject, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.
        action_prob_func_args_by_subject_id_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of subject ids to the function arguments
            required to compute action probabilities for this subject.
        action_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping subject IDs to their respective actions taken at each decision time.
            Only applies to in-study decision times!
        full_joint_adjusted_bread_matrix (jnp.ndarray):
            The full joint adjusted bread matrix as a NumPy array.
        per_subject_estimating_function_stacks (jnp.ndarray):
            A NumPy array containing all per-subject (weighted) estimating function stacks.
        beta_dim (int):
            The dimension of the beta parameters.
    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: A NumPy array containing the sequence of premature adjusted estimates.
    """

    # Loop through the non-initial (ie not before an update has occurred), non-final policy numbers in sorted order, forming adjusted and classical
    # variance estimates pretending that each was the final policy.
    premature_adjusted_sandwiches = []
    premature_thetas = []
    premature_joint_adjusted_bread_condition_numbers = []
    premature_avg_inference_estimating_functions = []
    premature_classical_sandwiches = []
    logger.info(
        "Calculating sequence of premature adjusted estimates by pretending the study ended after each update in sequence."
    )
    for policy_num in sorted(beta_index_by_policy_num):
        logger.info(
            "Calculating premature adjusted estimate assuming policy %s is the final one.",
            policy_num,
        )
        pretend_max_policy = policy_num

        truncated_joint_adjusted_bread_matrix = full_joint_adjusted_bread_matrix[
            : (beta_index_by_policy_num[pretend_max_policy] + 1) * beta_dim,
            : (beta_index_by_policy_num[pretend_max_policy] + 1) * beta_dim,
        ]

        max_decision_time = analysis_df[
            analysis_df["policy_num"] == pretend_max_policy
        ][calendar_t_col_name].max()

        truncated_analysis_df = analysis_df[
            analysis_df[calendar_t_col_name] <= max_decision_time
        ].copy()

        truncated_beta_index_by_policy_num = {
            k: v for k, v in beta_index_by_policy_num.items() if k <= pretend_max_policy
        }

        max_beta_index = max(truncated_beta_index_by_policy_num.values())

        truncated_all_post_update_betas = all_post_update_betas[: max_beta_index + 1, :]

        premature_theta = jnp.array(theta_calculation_func(truncated_analysis_df))

        truncated_action_prob_func_args_by_subject_id_by_decision_time = {
            decision_time: args_by_subject_id
            for decision_time, args_by_subject_id in action_prob_func_args_by_subject_id_by_decision_time.items()
            if decision_time <= max_decision_time
        }

        truncated_inference_func_args_by_subject_id, _, _ = (
            post_deployment_analysis.process_inference_func_args(
                inference_func,
                inference_func_args_theta_index,
                truncated_analysis_df,
                premature_theta,
                action_prob_col_name,
                calendar_t_col_name,
                subject_id_col_name,
                active_col_name,
            )
        )

        truncated_inference_action_prob_decision_times_by_subject_id = {
            subject_id: [
                decision_time
                for decision_time in inference_action_prob_decision_times_by_subject_id[
                    subject_id
                ]
                if decision_time <= max_decision_time
            ]
            # writing this way is important, handles empty dicts correctly
            for subject_id in inference_action_prob_decision_times_by_subject_id
        }

        truncated_action_by_decision_time_by_subject_id = {
            subject_id: {
                decision_time: action
                for decision_time, action in action_by_decision_time_by_subject_id[
                    subject_id
                ].items()
                if decision_time <= max_decision_time
            }
            for subject_id in action_by_decision_time_by_subject_id
        }

        truncated_per_subject_estimating_function_stacks = (
            per_subject_estimating_function_stacks[
                :,
                : (beta_index_by_policy_num[pretend_max_policy] + 1) * beta_dim,
            ]
        )

        (
            premature_adjusted_sandwich,
            premature_classical_sandwich,
            premature_avg_inference_estimating_function,
        ) = construct_premature_classical_and_adjusted_sandwiches(
            truncated_joint_adjusted_bread_matrix,
            truncated_per_subject_estimating_function_stacks,
            premature_theta,
            truncated_all_post_update_betas,
            subject_ids,
            action_prob_func,
            action_prob_func_args_beta_index,
            inference_func,
            inference_func_type,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            truncated_action_prob_func_args_by_subject_id_by_decision_time,
            policy_num_by_decision_time_by_subject_id,
            initial_policy_num,
            truncated_beta_index_by_policy_num,
            truncated_inference_func_args_by_subject_id,
            truncated_inference_action_prob_decision_times_by_subject_id,
            truncated_action_by_decision_time_by_subject_id,
        )

        premature_adjusted_sandwiches.append(premature_adjusted_sandwich)
        premature_classical_sandwiches.append(premature_classical_sandwich)
        premature_thetas.append(premature_theta)
        premature_avg_inference_estimating_functions.append(
            premature_avg_inference_estimating_function
        )
    return (
        jnp.array(premature_thetas),
        jnp.array(premature_adjusted_sandwiches),
        jnp.array(premature_classical_sandwiches),
        jnp.array(premature_joint_adjusted_bread_condition_numbers),
        jnp.array(premature_avg_inference_estimating_functions),
    )


def construct_premature_classical_and_adjusted_sandwiches(
    truncated_joint_adjusted_bread_matrix: jnp.ndarray,
    per_subject_truncated_estimating_function_stacks: jnp.ndarray,
    theta: jnp.ndarray,
    all_post_update_betas: jnp.ndarray,
    subject_ids: jnp.ndarray,
    action_prob_func: str,
    action_prob_func_args_beta_index: int,
    inference_func: str,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_subject_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_subject_id: dict[
        collections.abc.Hashable, list[int]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
) -> tuple[
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
]:
    """
    Constructs the classical bread and meat matrices, as well as the adjusted bread matrix
    and the average weighted inference estimating function for the premature variance estimation
    procedure.

    This is done by computing and differentiating the new average inference estimating function
    with respect to the betas and theta, and stitching this together with the existing
    adjusted bread matrix portion (corresponding to the updates still under consideration)
    to form the new premature joint adjusted bread matrix.

    Args:
        truncated_joint_adjusted_bread_matrix (jnp.ndarray):
            A 2-D JAX NumPy array holding the existing joint adjusted bread but
            with rows corresponding to updates not under consideration and inference dropped.
            We will stitch this together with the newly computed inference portion to form
            our "premature" joint adjusted bread matrix.
        per_subject_truncated_estimating_function_stacks (jnp.ndarray):
            A 2-D JAX NumPy array holding the existing per-subject weighted estimating function
            stacks but with rows corresponding to updates not under consideration dropped.
            We will stitch this together with the newly computed inference estimating functions
            to form our "premature" joint adjusted estimating function stacks from which the new
            adjusted meat matrix can be computed.
        theta (jnp.ndarray):
            A 1-D JAX NumPy array representing the parameter estimate for inference.
        all_post_update_betas (jnp.ndarray):
            A 2-D JAX NumPy array representing all parameter estimates for the algorithm updates.
        subject_ids (jnp.ndarray):
            A 1-D JAX NumPy array holding all subject IDs in the study.
        action_prob_func (callable):
            The action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        inference_func (callable):
            The inference loss or estimating function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_subject_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of subject ids to the function arguments
            required to compute action probabilities for this subject.
        policy_num_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of subject ids to dictionaries mapping decision times to the policy number in use.
            Only applies to in-study decision times!
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_subject_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping subject IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_subject_id (dict[collections.abc.Hashable, list[int]]):
            For each subject, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.
        action_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping subject IDs to their respective actions taken at each decision time.
            Only applies to in-study decision times!
    Returns:
        tuple[jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32],
              jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32],
              jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32]]:
            A tuple containing:
            - The joint adjusted bread matrix.
            - The joint adjusted bread matrix.
            - The joint adjusted meat matrix.
            - The classical bread matrix.
            - The classical bread matrix.
            - The classical meat matrix.
            - The average (weighted) inference estimating function.
            - The joint adjusted bread matrix condition number.
    """
    logger.info(
        "Differentiating average weighted inference estimating function stack and collecting auxiliary values."
    )
    # jax.jacobian may perform worse here--seemed to hang indefinitely while jacrev is merely very
    # slow.
    # Note that these "contributions" are per-subject Jacobians of the weighted estimating function stack.
    new_inference_block_row, (
        per_subject_inference_estimating_functions,
        avg_inference_estimating_function,
        per_subject_classical_meat_contributions,
        per_subject_classical_bread_contributions,
    ) = jax.jacrev(get_weighted_inference_estimating_functions_only, has_aux=True)(
        # While JAX can technically differentiate with respect to a list of JAX arrays,
        # it is more efficient to flatten them into a single array. This is done
        # here to improve performance. We can simply unflatten them inside the function.
        post_deployment_analysis.flatten_params(all_post_update_betas, theta),
        all_post_update_betas.shape[1],
        theta.shape[0],
        subject_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_subject_id,
        inference_action_prob_decision_times_by_subject_id,
        action_by_decision_time_by_subject_id,
    )

    joint_adjusted_bread_matrix = jnp.block(
        [
            [
                truncated_joint_adjusted_bread_matrix,
                np.zeros(
                    (
                        truncated_joint_adjusted_bread_matrix.shape[0],
                        new_inference_block_row.shape[0],
                    )
                ),
            ],
            [new_inference_block_row],
        ]
    )
    per_subject_estimating_function_stacks = jnp.concatenate(
        [
            per_subject_truncated_estimating_function_stacks,
            per_subject_inference_estimating_functions,
        ],
        axis=1,
    )
    per_subject_adjusted_meat_contributions = jnp.einsum(
        "ni,nj->nij",
        per_subject_estimating_function_stacks,
        per_subject_estimating_function_stacks,
    )

    joint_adjusted_meat_matrix = jnp.mean(
        per_subject_adjusted_meat_contributions, axis=0
    )

    classical_bread_matrix = jnp.mean(per_subject_classical_bread_contributions, axis=0)
    classical_meat_matrix = jnp.mean(per_subject_classical_meat_contributions, axis=0)

    num_subjects = subject_ids.shape[0]
    joint_adjusted_sandwich = (
        post_deployment_analysis.form_sandwich_from_bread_and_meat(
            joint_adjusted_bread_matrix,
            joint_adjusted_meat_matrix,
            num_subjects,
            method="bread_T_qr",
        )
    )
    adjusted_sandwich = joint_adjusted_sandwich[-theta.shape[0] :, -theta.shape[0] :]

    classical_bread_matrix = jnp.mean(per_subject_classical_bread_contributions, axis=0)
    classical_sandwich = post_deployment_analysis.form_sandwich_from_bread_and_meat(
        classical_bread_matrix,
        classical_meat_matrix,
        num_subjects,
        method="bread_T_qr",
    )

    # Stack the joint adjusted bread pieces together horizontally and return the auxiliary
    # values too. The joint adjusted bread should always be block lower triangular.
    return (
        adjusted_sandwich,
        classical_sandwich,
        avg_inference_estimating_function,
    )


def get_weighted_inference_estimating_functions_only(
    flattened_betas_and_theta: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    subject_ids: jnp.ndarray,
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    inference_func: callable,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_subject_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_subject_id: dict[
        collections.abc.Hashable, list[int]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
) -> tuple[
    jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    """
    Computes the average weighted inference estimating function across subjects, along with
    auxiliary values used to construct the adjusted and classical sandwich variances.

    Note that input data should have been adjusted to only correspond to updates/decision times
    that are being considered for the current "premature" variance estimation procedure.

    Args:
        flattened_betas_and_theta (jnp.ndarray):
            A list of JAX NumPy arrays representing the betas produced by all updates and the
            theta value, in that order. Important that this is a 1D array for efficiency reasons.
            We simply extract the betas and theta from this array below.
        beta_dim (int):
            The dimension of each of the beta parameters.
        theta_dim (int):
            The dimension of the theta parameter.
        subject_ids (jnp.ndarray):
            A 1D JAX NumPy array of subject IDs.
        action_prob_func (str):
            The action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        inference_func (str):
            The inference loss or estimating function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_subject_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of subject ids to the function arguments
            required to compute action probabilities for this subject.
        policy_num_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of subject ids to dictionaries mapping decision times to the policy number in use.
            Only applies to in-study decision times!
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_subject_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping subject IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_subject_id (dict[collections.abc.Hashable, list[int]]):
            For each subject, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.
        action_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping subject IDs to their respective actions taken at each decision time.
            Only applies to in-study decision times!

    Returns:
        jnp.ndarray:
            A 2D JAX NumPy array holding the average weighted inference estimating function.
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            A tuple containing
            1. the per-subject weighted inference estimating function stacks
            2. the average weighted inference estimating function
            3. the subject-level classical meat matrix contributions
            4. the subject-level inverse classical bread matrix contributions
            stacks.
    """

    inference_estimating_func = (
        jax.grad(inference_func, argnums=inference_func_args_theta_index)
        if (inference_func_type == FunctionTypes.LOSS)
        else inference_func
    )

    betas, theta = post_deployment_analysis.unflatten_params(
        flattened_betas_and_theta,
        beta_dim,
        theta_dim,
    )

    # 2. Thread in the betas and theta in all_post_update_betas_and_theta into the arguments
    # supplied for the above functions, so that differentiation works correctly.  The existing
    # values should be the same, but not connected to the parameter we are differentiating
    # with respect to. Note we will also find it useful below to have the action probability args
    # nested dict structure flipped to be subject_id -> decision_time -> args, so we do that here too.

    logger.info("Threading in betas to action probability arguments for all subjects.")
    (
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        action_prob_func_args_by_decision_time_by_subject_id,
    ) = post_deployment_analysis.thread_action_prob_func_args(
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id,
        initial_policy_num,
        betas,
        beta_index_by_policy_num,
        action_prob_func_args_beta_index,
    )

    # 4. Thread the central theta into the inference function arguments
    # and replace any action probabilities with reconstructed ones from the above
    # arguments with the central betas introduced.
    logger.info(
        "Threading in theta and beta-dependent action probabilities to inference update "
        "function args for all subjects"
    )
    threaded_inference_func_args_by_subject_id = (
        post_deployment_analysis.thread_inference_func_args(
            inference_func_args_by_subject_id,
            inference_func_args_theta_index,
            theta,
            inference_func_args_action_prob_index,
            threaded_action_prob_func_args_by_decision_time_by_subject_id,
            inference_action_prob_decision_times_by_subject_id,
            action_prob_func,
        )
    )

    # 5. Now we can compute the the weighted inference estimating functions for all subjects
    # as well as collect related values used to construct the adjusted and classical
    # sandwich variances.
    results = [
        single_subject_weighted_inference_estimating_function(
            subject_id,
            action_prob_func,
            inference_estimating_func,
            action_prob_func_args_beta_index,
            inference_func_args_theta_index,
            action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_inference_func_args_by_subject_id[subject_id],
            policy_num_by_decision_time_by_subject_id[subject_id],
            action_by_decision_time_by_subject_id[subject_id],
            beta_index_by_policy_num,
        )
        for subject_id in subject_ids.tolist()
    ]

    weighted_inference_estimating_functions = jnp.array(
        [result[0] for result in results]
    )
    inference_only_outer_products = jnp.array([result[1] for result in results])
    inference_hessians = jnp.array([result[2] for result in results])

    # 6. Note this strange return structure! We will differentiate the first output,
    # but the second tuple will be passed along without modification via has_aux=True and then used
    # for the adjusted meat matrix, estimating functions sum check, and classical meat and inverse
    # bread matrices. The raw per-subject estimating functions are also returned again for debugging
    # purposes.
    return jnp.mean(weighted_inference_estimating_functions, axis=0), (
        weighted_inference_estimating_functions,
        jnp.mean(weighted_inference_estimating_functions, axis=0),
        inference_only_outer_products,
        inference_hessians,
    )


def single_subject_weighted_inference_estimating_function(
    subject_id: collections.abc.Hashable,
    action_prob_func: callable,
    inference_estimating_func: callable,
    action_prob_func_args_beta_index: int,
    inference_func_args_theta_index: int,
    action_prob_func_args_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    threaded_action_prob_func_args_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    threaded_inference_func_args: dict[collections.abc.Hashable, tuple[Any, ...]],
    policy_num_by_decision_time: dict[collections.abc.Hashable, dict[int, int | float]],
    action_by_decision_time: dict[collections.abc.Hashable, dict[int, int]],
    beta_index_by_policy_num: dict[int | float, int],
) -> tuple[
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
]:
    """
    Computes a weighted inference estimating function for a given inference estimating function and arguments
    and action probability function and arguments if applicable.

    Args:
        subject_id (collections.abc.Hashable):
            The subject ID for which to compute the weighted estimating function stack.

        action_prob_func (callable):
            The function used to compute the probability of action 1 at a given decision time for
            a particular subject given their state and the algorithm parameters.

        inference_estimating_func (callable):
            The estimating function that corresponds to inference.

        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.

        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference loss or estimating function arguments.

        action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for this subject for the action
            probability function. This is for all decision times (args are an empty
            tuple if they are not in the study). Should be sorted by decision time. NOTE THAT THESE
            ARGS DO NOT CONTAIN THE SHARED BETAS, making them impervious to the differentiation that
            will occur.

        threaded_action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for the action
            probability function, with the shared betas threaded in for differentation. Decision
            times should be sorted.

        threaded_inference_func_args (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A tuple containing the arguments for the inference
            estimating function for this subject, with the shared betas threaded in for differentiation.

        policy_num_by_decision_time (dict[collections.abc.Hashable, dict[int, int | float]]):
            A dictionary mapping decision times to the policy number in use. This may be
            subject-specific. Should be sorted by decision time. Only applies to in-study decision
            times!

        action_by_decision_time (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to actions taken. Only applies to in-study decision
            times!

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

    Returns:
        jnp.ndarray: A 1-D JAX NumPy array representing the subject's weighted inference estimating function.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical bread contribution.
    """

    logger.info(
        "Computing only weighted inference estimating function stack for subject %s.",
        subject_id,
    )

    # First, reformat the supplied data into more convenient structures.

    # 1. Get the first time after the first update for convenience.
    # This is used to form the Radon-Nikodym weights for the right times.
    _, first_time_after_first_update = (
        post_deployment_analysis.get_min_time_by_policy_num(
            policy_num_by_decision_time,
            beta_index_by_policy_num,
        )
    )

    # 2. Get the start and end times for this subject.
    subject_start_time = math.inf
    subject_end_time = -math.inf
    for decision_time in action_by_decision_time:
        subject_start_time = min(subject_start_time, decision_time)
        subject_end_time = max(subject_end_time, decision_time)

    # 3. Calculate the Radon-Nikodym weights for the inference estimating function.
    in_study_action_prob_func_args = [
        args for args in action_prob_func_args_by_decision_time.values() if args
    ]
    in_study_betas_list_by_decision_time_index = jnp.array(
        [
            action_prob_func_args[action_prob_func_args_beta_index]
            for action_prob_func_args in in_study_action_prob_func_args
        ]
    )
    in_study_actions_list_by_decision_time_index = jnp.array(
        list(action_by_decision_time.values())
    )

    # Sort the threaded args by decision time to be cautious. We check if the
    # subject id is present in the subject args dict because we may call this on a
    # subset of the subject arg dict when we are batching arguments by shape
    sorted_threaded_action_prob_args_by_decision_time = {
        decision_time: threaded_action_prob_func_args_by_decision_time[decision_time]
        for decision_time in range(subject_start_time, subject_end_time + 1)
        if decision_time in threaded_action_prob_func_args_by_decision_time
    }

    num_args = None
    for args in sorted_threaded_action_prob_args_by_decision_time.values():
        if args:
            num_args = len(args)
            break

    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_threaded_arg_lists = [[] for _ in range(num_args)]
    for (
        decision_time,
        args,
    ) in sorted_threaded_action_prob_args_by_decision_time.items():
        if not args:
            continue
        for idx, arg in enumerate(args):
            batched_threaded_arg_lists[idx].append(arg)

    batched_threaded_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensors(
        batched_threaded_arg_lists
    )

    # Note that we do NOT use the shared betas in the first arg to the weight function,
    # since we don't want differentiation to happen with respect to them.
    # Just grab the original beta from the update function arguments. This is the same
    # value, but impervious to differentiation with respect to all_post_update_betas. The
    # args, on the other hand, are a function of all_post_update_betas.
    in_study_weights = jax.vmap(
        fun=post_deployment_analysis.get_radon_nikodym_weight,
        in_axes=[0, None, None, 0] + batch_axes,
        out_axes=0,
    )(
        in_study_betas_list_by_decision_time_index,
        action_prob_func,
        action_prob_func_args_beta_index,
        in_study_actions_list_by_decision_time_index,
        *batched_threaded_arg_tensors,
    )

    in_study_index = 0
    decision_time_to_all_weights_index_offset = min(
        sorted_threaded_action_prob_args_by_decision_time
    )
    all_weights_raw = []
    for (
        decision_time,
        args,
    ) in sorted_threaded_action_prob_args_by_decision_time.items():
        all_weights_raw.append(in_study_weights[in_study_index] if args else 1.0)
        in_study_index += 1
    all_weights = jnp.array(all_weights_raw)

    # 4. Form the weighted inference estimating equation.
    weighted_inference_estimating_function = jnp.prod(
        all_weights[
            max(first_time_after_first_update, subject_start_time)
            - decision_time_to_all_weights_index_offset : subject_end_time
            + 1
            - decision_time_to_all_weights_index_offset,
        ]
        # If the subject exited the study before there were any updates,
        # this variable will be None and the above code to grab a weight would
        # throw an error. Just use 1 to include the unweighted estimating function
        # if they have data to contribute here (pretty sure everyone should?)
        if first_time_after_first_update is not None
        else 1
    ) * inference_estimating_func(*threaded_inference_func_args)

    return (
        weighted_inference_estimating_function,
        jnp.outer(
            weighted_inference_estimating_function,
            weighted_inference_estimating_function,
        ),
        jax.jacrev(inference_estimating_func, argnums=inference_func_args_theta_index)(
            *threaded_inference_func_args
        ),
    )
