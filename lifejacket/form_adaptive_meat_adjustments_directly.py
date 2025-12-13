import collections
import logging

import pandas as pd
import jax.numpy as jnp
import numpy as np

from .calculate_derivatives import (
    calculate_inference_loss_derivatives,
    calculate_pi_and_weight_gradients,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def form_adaptive_meat_adjustments_directly(
    theta_dim: int,
    beta_dim: int,
    joint_adaptive_bread_inverse_matrix: jnp.ndarray,
    per_user_estimating_function_stacks: jnp.ndarray,
    study_df: pd.DataFrame,
    in_study_col_name: str,
    action_col_name: str,
    calendar_t_col_name: str,
    user_id_col_name: str,
    action_prob_func: callable,
    action_prob_func_args: dict,
    action_prob_func_args_beta_index: int,
    theta_est: jnp.ndarray,
    inference_func: callable,
    inference_func_args_theta_index: int,
    user_ids: list[collections.abc.Hashable],
    action_prob_col_name: str,
) -> jnp.ndarray:
    logger.info(
        "Explicitly forming the per-user meat adjustments that differentiate the adaptive sandwich from the classical sandwich."
    )

    # 1. Form the M-matrices, which are shared across users.
    # This is not quite the paper definition of the M-matrices, which
    # includes multiplication by the classical bread.  We don't care about
    # that here, since in forming the adjustments there is a multiplication
    # by the classical bread inverse that cancels it out.
    V_blocks_together = joint_adaptive_bread_inverse_matrix[-theta_dim:, :-theta_dim]
    RL_stack_beta_derivatives_block = joint_adaptive_bread_inverse_matrix[
        :-theta_dim, :-theta_dim
    ]
    effective_M_blocks_together = np.linalg.solve(
        RL_stack_beta_derivatives_block.T, V_blocks_together.T
    ).T

    # 2. Extract the RL-only parts of the per-user estimating function stacks
    per_user_RL_only_est_fn_stacks_together = per_user_estimating_function_stacks[
        :, :-theta_dim
    ]

    # 3. Split the effective M blocks into (theta_dim, beta_dim) blocks and the
    # estimating function stacks into (num_updates, beta_dim) stacks.

    # effective_M_blocks_together is shape (theta_dim, num_updates * beta_dim)
    # We want to split it into a list of (theta_dim, beta_dim) arrays
    M_blocks = np.split(
        effective_M_blocks_together,
        effective_M_blocks_together.shape[1] // beta_dim,
        axis=1,
    )
    # Now stack into a 3D array of shape (num_updates, theta_dim, beta_dim)
    M_blocks_stacked = np.stack(M_blocks, axis=0)

    # per_user_RL_only_est_fn_stacks is shape (num_users, num_updates * beta_dim)
    # We want to split it into a list of (num_updates, beta_dim) arrays per user
    per_user_RL_only_est_fns = np.split(
        per_user_RL_only_est_fn_stacks_together,
        per_user_RL_only_est_fn_stacks_together.shape[1] // beta_dim,
        axis=1,
    )
    # Stack into a 3D array of shape (num_users, num_updates, beta_dim)
    # Note the difference between this and the original format of these estimating functions,
    # which was not broken down by update
    per_user_RL_only_est_fns_stacked = np.stack(per_user_RL_only_est_fns, axis=1)

    # Now multiply the M matrices and the per-user estimating functions
    # and sum over the updates to get the per-user meat adjustments (to be more precise, what would
    # be added to each users inference estimating function before an outer product is taken with
    # itself to get each users's contributioan theta-only meat matrix).
    # Result is shape (num_users, theta_dim).
    # Form the per-user adaptive meat adjustments explicitly for diagnostic purposes.
    per_user_meat_adjustments_stacked = np.einsum(
        "utb,nub->nt", M_blocks_stacked, per_user_RL_only_est_fns_stacked
    )

    # Log some diagnostics about the pieces going into the adaptive meat adjustments
    # and the adjustments themselves.
    V_blocks = np.split(
        V_blocks_together, V_blocks_together.shape[1] // beta_dim, axis=1
    )
    logger.info("Examining adaptive meat adjustments.")
    # No scientific notation
    np.set_printoptions(suppress=True)

    per_user_inference_estimating_functions_stacked = (
        per_user_estimating_function_stacks[:, -theta_dim:]
    )
    # This actually logs way too much, so making these all debug level to not exhaust VScode
    # terminal buffer
    logger.debug(
        "Per-user inference estimating functions. Without adjustment, the average of the outer products of these is the classical meat: %s",
        per_user_inference_estimating_functions_stacked,
    )
    logger.debug(
        "Norms of per-user inference estimating functions: %s",
        np.linalg.norm(per_user_inference_estimating_functions_stacked, axis=1),
    )

    logger.debug(
        "Per-user adaptive meat adjustments, to be added to inference estimating functions before forming the meat. Formed from the sum of the products of the M-blocks and the corresponding RL update estimating functions for each user: %s",
        per_user_meat_adjustments_stacked,
    )
    logger.debug(
        "Norms of per-user adaptive meat adjustments: %s",
        np.linalg.norm(per_user_meat_adjustments_stacked, axis=1),
    )

    per_user_fractional_adjustments = (
        per_user_meat_adjustments_stacked
        / per_user_inference_estimating_functions_stacked
    )
    logger.debug(
        "Per-user fractional adjustments (elementwise ratio of adjustment to original inference estimating function): %s",
        per_user_fractional_adjustments,
    )
    logger.debug(
        "Norms of per-user fractional adjustments: %s",
        np.linalg.norm(per_user_fractional_adjustments, axis=1),
    )

    V_blocks_stacked = np.stack(V_blocks, axis=0)
    logger.debug(
        "V_blocks, one per update, each shape theta_dim x beta_dim. These measure the sensitivity of the estimating function for theta to the limiting policy parameters per update: %s",
        V_blocks_stacked,
    )
    logger.debug("Norms of V-blocks: %s", np.linalg.norm(V_blocks_stacked, axis=(1, 2)))

    logger.debug(
        "M_blocks, one per update, each shape theta_dim x beta_dim. The sum of the products "
        "of each of these times a user's corresponding RL estimating function forms their adaptive "
        "adjustment. The M's are the blocks of the the product of the V's concatened and the inverse of "
        "the RL-only upper-left block of the joint adaptive bread inverse. In other words, the lower "
        "left block of the joint adaptive bread. Also note that the inference estimating function "
        "derivative inverse is omitted here despite the definition of the M's in the paper, because "
        "that factor simply cancels later: %s",
        M_blocks_stacked,
    )
    logger.debug("Norms of M-blocks: %s", np.linalg.norm(M_blocks_stacked, axis=(1, 2)))

    logger.debug(
        "RL block of joint adaptive bread inverse. The *inverse* of this goes into the M's: %s",
        RL_stack_beta_derivatives_block,
    )
    logger.debug(
        "Norm of RL block of joint adaptive bread inverse: %s",
        np.linalg.norm(RL_stack_beta_derivatives_block),
    )

    inverse_RL_stack_beta_derivatives_block = np.linalg.inv(
        RL_stack_beta_derivatives_block
    )
    logger.debug(
        "Inverse of RL block of joint adaptive bread inverse. This goes into the M's: %s",
        inverse_RL_stack_beta_derivatives_block,
    )
    logger.debug(
        "Norm of Inverse of RL block of joint adaptive bread inverse: %s",
        np.linalg.norm(inverse_RL_stack_beta_derivatives_block),
    )

    logger.debug(
        "Per-update RL-only estimating function elementwise maxes across users: %s",
        np.max(per_user_RL_only_est_fns_stacked, axis=0),
    )
    logger.debug(
        "Per-update RL-only estimating function elementwise mins across users: %s",
        np.min(per_user_RL_only_est_fns_stacked, axis=0),
    )
    logger.debug(
        "Per-user average RL-only estimating functions across updates: %s",
        np.mean(per_user_RL_only_est_fns_stacked, axis=1),
    )
    logger.debug(
        "Per-update std of RL-only estimating functions across users: %s",
        np.std(per_user_RL_only_est_fns_stacked, axis=0),
    )
    logger.debug(
        "Norms of per-user RL-only estimating functions (num users x num updates): %s",
        np.linalg.norm(per_user_RL_only_est_fns_stacked, axis=2),
    )

    # Now dig even deeper to get weight derivatives and inference estimating function mixed
    # derivatives that go into the V's

    pi_and_weight_gradients_by_calendar_t = calculate_pi_and_weight_gradients(
        study_df,
        in_study_col_name,
        action_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_func,
        action_prob_func_args,
        action_prob_func_args_beta_index,
    )

    _, _, loss_gradient_pi_derivatives = calculate_inference_loss_derivatives(
        study_df,
        theta_est,
        inference_func,
        inference_func_args_theta_index,
        user_ids,
        user_id_col_name,
        action_prob_col_name,
        in_study_col_name,
        calendar_t_col_name,
    )
    # Take the outer product of each row of (per_user_meat_adjustments_stacked + per_user_inference_estimating_functions_stacked)
    per_user_adjusted_inference_estimating_functions_stacked = (
        per_user_meat_adjustments_stacked
        + per_user_inference_estimating_functions_stacked
    )
    per_user_theta_only_adaptive_meat_contributions = jnp.einsum(
        "ni,nj->nij",
        per_user_adjusted_inference_estimating_functions_stacked,
        per_user_adjusted_inference_estimating_functions_stacked,
    )
    adaptive_theta_only_meat_matrix = jnp.mean(
        per_user_theta_only_adaptive_meat_contributions, axis=0
    )
    logger.info(
        "Theta-only adaptive meat matrix (no small sample corrections): %s",
        adaptive_theta_only_meat_matrix,
    )
    classical_theta_only_meat_matrix = jnp.mean(
        jnp.einsum(
            "ni,nj->nij",
            per_user_inference_estimating_functions_stacked,
            per_user_inference_estimating_functions_stacked,
        ),
        axis=0,
    )
    logger.info(
        "Classical meat matrix (no small sample corrections): %s",
        classical_theta_only_meat_matrix,
    )

    # np.linalg.cond(RL_stack_beta_derivatives_block)

    # Print the condition number of each upper left block of RL_stack_beta_derivatives_block
    # as if we stopped after first update, then second update, etc, up to full beta_dim * num_updates
    num_updates = RL_stack_beta_derivatives_block.shape[0] // beta_dim
    whole_block_condition_numbers = []
    diagonal_block_condition_numbers = []
    for i in range(1, num_updates + 1):
        whole_block_size = i * beta_dim
        whole_block = RL_stack_beta_derivatives_block[
            :whole_block_size, :whole_block_size
        ]
        whole_block_cond_number = np.linalg.cond(whole_block)
        whole_block_condition_numbers.append(whole_block_cond_number)
        logger.info(
            "Condition number of whole RL_stack_beta_derivatives_block (after update %s): %s",
            i,
            whole_block_cond_number,
        )
        diagonal_block = RL_stack_beta_derivatives_block[
            (i - 1) * beta_dim : i * beta_dim, (i - 1) * beta_dim : i * beta_dim
        ]
        diagonal_block_cond_number = np.linalg.cond(diagonal_block)
        diagonal_block_condition_numbers.append(diagonal_block_cond_number)
        logger.info(
            "Condition number of just RL_stack_beta_derivatives_block *diagonal block* for update %s: %s",
            i,
            diagonal_block_cond_number,
        )

        off_diagonal_scaled_block_norm_sum = 0
        for j in range(1, i):
            off_diagonal_block = RL_stack_beta_derivatives_block[
                (i - 1) * beta_dim : i * beta_dim, (j - 1) * beta_dim : j * beta_dim
            ]
            off_diagonal_scaled_block_norm = np.linalg.norm(
                np.linalg.solve(diagonal_block, off_diagonal_block)
            )
            off_diagonal_scaled_block_norm_sum += off_diagonal_scaled_block_norm
            logger.debug(
                "Norm of off-diagonal block (%s, %s) scaled by inverse of diagonal block: %s",
                i,
                j,
                off_diagonal_scaled_block_norm,
            )

        logger.info(
            "Sum of norms of off-diagonal blocks in row %s scaled by inverse of diagonal block: %s",
            i,
            off_diagonal_scaled_block_norm_sum,
        )

    # Keeping a breakpoint here is the best way to dig in without logging too
    # much or being too opinionated about what to log.
    breakpoint()

    # # Visualize the inverse RL block of joint adaptive bread inverse using seaborn heatmap
    # pyplt.figure(figsize=(8, 6))
    # sns.heatmap(inverse_RL_stack_beta_derivatives_block, annot=False, cmap="viridis")
    # pyplt.title("Inverse RL Block of Joint Adaptive Bread Inverse")
    # pyplt.xlabel("Beta Index")
    # pyplt.ylabel("Beta Index")
    # pyplt.tight_layout()
    # pyplt.show()

    # # # Visualize the RL block of joint adaptive bread inverse using seaborn heatmap

    # pyplt.figure(figsize=(8, 6))
    # sns.heatmap(RL_stack_beta_derivatives_block, annot=False, cmap="viridis")
    # pyplt.title("RL Block of Joint Adaptive Bread Inverse")
    # pyplt.xlabel("Beta Index")
    # pyplt.ylabel("Beta Index")
    # pyplt.tight_layout()
    # pyplt.show()

    return per_user_theta_only_adaptive_meat_contributions
