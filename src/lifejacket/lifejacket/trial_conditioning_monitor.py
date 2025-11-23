from __future__ import annotations

import collections.abc
import logging
import math
from typing import Any

from jax import numpy as jnp
import jax
import numpy as np
import pandas as pd

from .arg_threading_helpers import thread_action_prob_func_args, thread_update_func_args
from .constants import FunctionTypes
from .helper_functions import (
    calculate_beta_dim,
    collect_all_post_update_betas,
    construct_beta_index_by_policy_num_map,
    extract_action_and_policy_by_decision_time_by_user_id,
    flatten_params,
    get_min_time_by_policy_num,
    get_radon_nikodym_weight,
    unflatten_params,
)
from . import input_checks
from .vmap_helpers import stack_batched_arg_lists_into_tensors


logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


class TrialConditioningMonitor:
    whole_RL_block_conditioning_threshold = None
    diagonal_RL_block_conditioning_threshold = None

    def __init__(
        self,
        whole_RL_block_conditioning_threshold: int = 1000,
        diagonal_RL_block_conditioning_threshold: int = 100,
    ):
        self.whole_RL_block_conditioning_threshold = (
            whole_RL_block_conditioning_threshold
        )
        self.diagonal_RL_block_conditioning_threshold = (
            diagonal_RL_block_conditioning_threshold
        )
        self.latest_phi_dot_bar = None

    def assess_update(
        self,
        proposed_policy_num: int | float,
        study_df: pd.DataFrame,
        action_prob_func: callable,
        action_prob_func_args: dict,
        action_prob_func_args_beta_index: int,
        alg_update_func: callable,
        alg_update_func_type: str,
        alg_update_func_args: dict,
        alg_update_func_args_beta_index: int,
        alg_update_func_args_action_prob_index: int,
        alg_update_func_args_action_prob_times_index: int,
        in_study_col_name: str,
        action_col_name: str,
        policy_num_col_name: str,
        calendar_t_col_name: str,
        user_id_col_name: str,
        action_prob_col_name: str,
        suppress_interactive_data_checks: bool,
        suppress_all_data_checks: bool,
        incremental: bool = True,
    ) -> None:
        """
        Analyzes a dataset to estimate parameters and variance using adaptive and classical sandwich estimators.

        Parameters:
        proposed_policy_num (int | float):
            The policy number of the proposed update.
        study_df (pd.DataFrame):
            DataFrame containing the study data.
        action_prob_func (str):
            Action probability function.
        action_prob_func_args (dict):
            Arguments for the action probability function.
        action_prob_func_args_beta_index (int):
            Index for beta in action probability function arguments.
        alg_update_func (str):
            Algorithm update function.
        alg_update_func_type (str):
            Type of the algorithm update function.
        alg_update_func_args (dict):
            Arguments for the algorithm update function.
        alg_update_func_args_beta_index (int):
            Index for beta in algorithm update function arguments.
        alg_update_func_args_action_prob_index (int):
            Index for action probability in algorithm update function arguments.
        alg_update_func_args_action_prob_times_index (int):
            Index for action probability times in algorithm update function arguments.
        in_study_col_name (str):
            Column name indicating if a user is in the study in the study dataframe.
        action_col_name (str):
            Column name for actions in the study dataframe.
        policy_num_col_name (str):
            Column name for policy numbers in the study dataframe.
        calendar_t_col_name (str):
            Column name for calendar time in the study dataframe.
        user_id_col_name (str):
            Column name for user IDs in the study dataframe.
        action_prob_col_name (str):
            Column name for action probabilities in the study dataframe.
        reward_col_name (str):
            Column name for rewards in the study dataframe.
        suppress_interactive_data_checks (bool):
            Whether to suppress interactive data checks. This should be used in simulations, for example.
        suppress_all_data_checks (bool):
            Whether to suppress all data checks. Not recommended.
        small_sample_correction (str):
            Type of small sample correction to apply.
        collect_data_for_blowup_supervised_learning (bool):
            Whether to collect data for doing supervised learning about adaptive sandwich blowup.
        form_adaptive_meat_adjustments_explicitly (bool):
            If True, explicitly forms the per-user meat adjustments that differentiate the adaptive
            sandwich from the classical sandwich. This is for diagnostic purposes, as the
            adaptive sandwich is formed without doing this.
        stabilize_joint_adaptive_bread_inverse (bool):
            If True, stabilizes the joint adaptive bread inverse matrix if it does not meet conditioning
            thresholds.

        Returns:
        None: The function writes analysis results and debug pieces to files in the same directory as
        the input files.
        """

        beta_dim = calculate_beta_dim(
            action_prob_func_args, action_prob_func_args_beta_index
        )

        if not suppress_all_data_checks:
            input_checks.perform_alg_only_input_checks(
                study_df,
                in_study_col_name,
                policy_num_col_name,
                calendar_t_col_name,
                user_id_col_name,
                action_prob_col_name,
                action_prob_func,
                action_prob_func_args,
                action_prob_func_args_beta_index,
                alg_update_func_args,
                alg_update_func_args_beta_index,
                alg_update_func_args_action_prob_index,
                alg_update_func_args_action_prob_times_index,
                suppress_interactive_data_checks,
            )

        beta_index_by_policy_num, initial_policy_num = (
            construct_beta_index_by_policy_num_map(
                study_df, policy_num_col_name, in_study_col_name
            )
        )
        # We augment the produced map to include the proposed policy num.
        # This is necessary because the logic above assumes all policies are present in the
        # study df, whereas for us we are only passing the data *used* for the current update,
        # i.e. up to the previous policy.
        beta_index_by_policy_num[proposed_policy_num] = len(beta_index_by_policy_num)

        all_post_update_betas = collect_all_post_update_betas(
            beta_index_by_policy_num,
            alg_update_func_args,
            alg_update_func_args_beta_index,
        )

        action_by_decision_time_by_user_id, policy_num_by_decision_time_by_user_id = (
            extract_action_and_policy_by_decision_time_by_user_id(
                study_df,
                user_id_col_name,
                in_study_col_name,
                calendar_t_col_name,
                action_col_name,
                policy_num_col_name,
            )
        )

        user_ids = jnp.array(study_df[user_id_col_name].unique())

        phi_dot_bar, avg_estimating_function_stack = self.construct_phi_dot_bar_so_far(
            all_post_update_betas,
            user_ids,
            action_prob_func,
            action_prob_func_args_beta_index,
            alg_update_func,
            alg_update_func_type,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            action_prob_func_args,
            policy_num_by_decision_time_by_user_id,
            initial_policy_num,
            beta_index_by_policy_num,
            alg_update_func_args,
            action_by_decision_time_by_user_id,
            suppress_all_data_checks,
            suppress_interactive_data_checks,
            incremental=incremental,
        )

        if not suppress_all_data_checks:
            input_checks.require_RL_estimating_functions_sum_to_zero(
                avg_estimating_function_stack,
                beta_dim,
                suppress_interactive_data_checks,
            )

        # Decide whether to accept or reject the update based on conditioning
        update_rejected = False
        rejection_reason = ""
        whole_RL_block_condition_number = np.linalg.cond(phi_dot_bar)
        new_diagonal_RL_block_condition_number = np.linalg.cond(
            phi_dot_bar[-beta_dim:, -beta_dim:]
        )

        if whole_RL_block_condition_number > self.whole_RL_block_conditioning_threshold:
            logger.warning(
                "The RL portion of the bread inverse up to this point exceeds the threshold set (condition number: %s, threshold: %s). Consider an alternative update strategy which produces less dependence on previous RL parameters (via the data they produced) and/or improves the conditioning of each update itself.  Regularization may help with both of these.",
                whole_RL_block_condition_number,
                self.whole_RL_block_conditioning_threshold,
            )
            update_rejected = True
            rejection_reason = "whole_block_poor_conditioning"
        elif (
            new_diagonal_RL_block_condition_number
            > self.diagonal_RL_block_conditioning_threshold
        ):
            logger.warning(
                "The diagonal RL block of the bread inverse up to this point exceeds the threshold set (condition number: %s, threshold: %s). This may illustrate a fundamental problem with the conditioning of the RL update procedure.",
                new_diagonal_RL_block_condition_number,
                self.diagonal_RL_block_conditioning_threshold,
            )
            update_rejected = True
            rejection_reason = "diagonal_block_poor_conditioning"

        # TODO: Regression -> prediction over going over threshold? Take in estimated num updates if so.

        ans = {
            "update_rejected": update_rejected,
            "rejection_reason": rejection_reason,
            "whole_RL_block_condition_number": whole_RL_block_condition_number,
            "whole_RL_block_conditioning_threshold": self.whole_RL_block_conditioning_threshold,
            "new_diagonal_RL_block_condition_number": new_diagonal_RL_block_condition_number,
            "diagonal_RL_block_conditioning_threshold": self.diagonal_RL_block_conditioning_threshold,
        }
        logger.info("Update assessment results: %s", ans)
        return ans

    def construct_phi_dot_bar_so_far(
        self,
        all_post_update_betas: jnp.ndarray,
        user_ids: jnp.ndarray,
        action_prob_func: callable,
        action_prob_func_args_beta_index: int,
        alg_update_func: callable,
        alg_update_func_type: str,
        alg_update_func_args_beta_index: int,
        alg_update_func_args_action_prob_index: int,
        alg_update_func_args_action_prob_times_index: int,
        action_prob_func_args_by_user_id_by_decision_time: dict[
            collections.abc.Hashable, dict[int, tuple[Any, ...]]
        ],
        policy_num_by_decision_time_by_user_id: dict[
            collections.abc.Hashable, dict[int, int | float]
        ],
        initial_policy_num: int | float,
        beta_index_by_policy_num: dict[int | float, int],
        update_func_args_by_by_user_id_by_policy_num: dict[
            collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
        ],
        action_by_decision_time_by_user_id: dict[
            collections.abc.Hashable, dict[int, int]
        ],
        suppress_all_data_checks: bool,
        suppress_interactive_data_checks: bool,
        incremental: bool,
    ) -> tuple[
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
    ]:
        """
        Constructs the classical and adaptive inverse bread and meat matrices, as well as the average
        estimating function stack and some other intermediate pieces.

        This is done by computing and differentiating the average weighted estimating function stack
        with respect to the betas and theta, using the resulting Jacobian to compute the inverse bread
        and meat matrices, and then stably computing sandwiches.

        Args:
            all_post_update_betas (jnp.ndarray):
                A 2-D JAX NumPy array representing all parameter estimates for the algorithm updates.
            user_ids (jnp.ndarray):
                A 1-D JAX NumPy array holding all user IDs in the study.
            action_prob_func (callable):
                The action probability function.
            action_prob_func_args_beta_index (int):
                The index of beta in the action probability function arguments tuples.
            alg_update_func (callable):
                The algorithm update function.
            alg_update_func_type (str):
                The type of the algorithm update function (loss or estimating).
            alg_update_func_args_beta_index (int):
                The index of beta in the update function arguments tuples.
            alg_update_func_args_action_prob_index (int):
                The index  of action probabilities in the update function arguments tuple, if
                applicable. -1 otherwise.
            alg_update_func_args_action_prob_times_index (int):
                The index in the update function arguments tuple where an array of times for which the
                given action probabilities apply is provided, if applicable. -1 otherwise.
            action_prob_func_args_by_user_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
                A dictionary mapping decision times to maps of user ids to the function arguments
                required to compute action probabilities for this user.
            policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
                A map of user ids to dictionaries mapping decision times to the policy number in use.
                Only applies to in-study decision times!
            initial_policy_num (int | float):
                The policy number of the initial policy before any updates.
            beta_index_by_policy_num (dict[int | float, int]):
                A dictionary mapping policy numbers to the index of the corresponding beta in
                all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
            update_func_args_by_by_user_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
                A dictionary where keys are policy numbers and values are dictionaries mapping user IDs
                to their respective update function arguments.
            action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
                A dictionary mapping user IDs to their respective actions taken at each decision time.
                Only applies to in-study decision times!
            suppress_all_data_checks (bool):
                If True, suppresses carrying out any data checks at all.
            suppress_interactive_data_checks (bool):
                If True, suppresses interactive data checks that would otherwise be performed to ensure
                the correctness of the threaded arguments. The checks are still performed, but
                any interactive prompts are suppressed.
            incremental (bool): Whether to form the whole phi-dot-bar so far, or just form the latest
                block row and add to the cached previous phi-dot-bar.
        """
        logger.info(
            "Differentiating average weighted estimating function stack and collecting auxiliary values."
        )
        beta_dim = all_post_update_betas.shape[1]

        if incremental and self.latest_phi_dot_bar is not None:
            # We only need to compute the latest block row of the Jacobian.
            (
                phi_dot_bar_latest_block,
                avg_RL_estimating_function_stack,
            ) = jax.jacrev(
                self.get_avg_weighted_RL_estimating_function_stacks_and_aux_values,
                has_aux=True,
            )(
                # While JAX can technically differentiate with respect to a list of JAX arrays,
                # it is apparently more efficient to flatten them into a single array. This is done
                # here to improve performance. We can simply unflatten them inside the function.
                flatten_params(all_post_update_betas, jnp.array([])),
                beta_dim,
                user_ids,
                action_prob_func,
                action_prob_func_args_beta_index,
                alg_update_func,
                alg_update_func_type,
                alg_update_func_args_beta_index,
                alg_update_func_args_action_prob_index,
                alg_update_func_args_action_prob_times_index,
                action_prob_func_args_by_user_id_by_decision_time,
                policy_num_by_decision_time_by_user_id,
                initial_policy_num,
                beta_index_by_policy_num,
                update_func_args_by_by_user_id_by_policy_num,
                action_by_decision_time_by_user_id,
                suppress_all_data_checks,
                suppress_interactive_data_checks,
                only_latest_block=True,
            )

            # Now we can just augment the cached previous phi-dot-bar with zeros
            # and the latest block row.
            phi_dot_bar = jnp.block(
                [
                    self.latest_phi_dot_bar,
                    jnp.zeros((beta_dim, beta_dim)),
                    phi_dot_bar_latest_block[-beta_dim:, :],
                ]
            )
        else:

            (
                phi_dot_bar,
                avg_RL_estimating_function_stack,
            ) = jax.jacrev(
                self.get_avg_weighted_RL_estimating_function_stacks_and_aux_values,
                has_aux=True,
            )(
                # While JAX can technically differentiate with respect to a list of JAX arrays,
                # it is apparently more efficient to flatten them into a single array. This is done
                # here to improve performance. We can simply unflatten them inside the function.
                flatten_params(all_post_update_betas, jnp.array([])),
                beta_dim,
                user_ids,
                action_prob_func,
                action_prob_func_args_beta_index,
                alg_update_func,
                alg_update_func_type,
                alg_update_func_args_beta_index,
                alg_update_func_args_action_prob_index,
                alg_update_func_args_action_prob_times_index,
                action_prob_func_args_by_user_id_by_decision_time,
                policy_num_by_decision_time_by_user_id,
                initial_policy_num,
                beta_index_by_policy_num,
                update_func_args_by_by_user_id_by_policy_num,
                action_by_decision_time_by_user_id,
                suppress_all_data_checks,
                suppress_interactive_data_checks,
            )

        self.latest_phi_dot_bar = phi_dot_bar
        return phi_dot_bar, avg_RL_estimating_function_stack

    def get_avg_weighted_RL_estimating_function_stacks_and_aux_values(
        self,
        flattened_betas_and_theta: jnp.ndarray,
        beta_dim: int,
        user_ids: jnp.ndarray,
        action_prob_func: callable,
        action_prob_func_args_beta_index: int,
        alg_update_func: callable,
        alg_update_func_type: str,
        alg_update_func_args_beta_index: int,
        alg_update_func_args_action_prob_index: int,
        alg_update_func_args_action_prob_times_index: int,
        action_prob_func_args_by_user_id_by_decision_time: dict[
            collections.abc.Hashable, dict[int, tuple[Any, ...]]
        ],
        policy_num_by_decision_time_by_user_id: dict[
            collections.abc.Hashable, dict[int, int | float]
        ],
        initial_policy_num: int | float,
        beta_index_by_policy_num: dict[int | float, int],
        update_func_args_by_by_user_id_by_policy_num: dict[
            collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
        ],
        action_by_decision_time_by_user_id: dict[
            collections.abc.Hashable, dict[int, int]
        ],
        suppress_all_data_checks: bool,
        suppress_interactive_data_checks: bool,
        only_latest_block: bool = False,
    ) -> tuple[
        jnp.ndarray,
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        """
        Computes the average weighted estimating function stack across all users, along with
        auxiliary values used to construct the adaptive and classical sandwich variances.

        If only_latest_block is True, only uses data from the most recent update.

        Args:
            flattened_betas_and_theta (jnp.ndarray):
                A list of JAX NumPy arrays representing the betas produced by all updates and the
                theta value, in that order. Important that this is a 1D array for efficiency reasons.
                We simply extract the betas and theta from this array below.
            beta_dim (int):
                The dimension of each of the beta parameters.
            user_ids (jnp.ndarray):
                A 1D JAX NumPy array of user IDs.
            action_prob_func (callable):
                The action probability function.
            action_prob_func_args_beta_index (int):
                The index of beta in the action probability function arguments tuples.
            alg_update_func (callable):
                The algorithm update function.
            alg_update_func_type (str):
                The type of the algorithm update function (loss or estimating).
            alg_update_func_args_beta_index (int):
                The index of beta in the update function arguments tuples.
            alg_update_func_args_action_prob_index (int):
                The index  of action probabilities in the update function arguments tuple, if
                applicable. -1 otherwise.
            alg_update_func_args_action_prob_times_index (int):
                The index in the update function arguments tuple where an array of times for which the
                given action probabilities apply is provided, if applicable. -1 otherwise.
            action_prob_func_args_by_user_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
                A dictionary mapping decision times to maps of user ids to the function arguments
                required to compute action probabilities for this user.
            policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
                A map of user ids to dictionaries mapping decision times to the policy number in use.
                Only applies to in-study decision times!
            initial_policy_num (int | float):
                The policy number of the initial policy before any updates.
            beta_index_by_policy_num (dict[int | float, int]):
                A dictionary mapping policy numbers to the index of the corresponding beta in
                all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
            update_func_args_by_by_user_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
                A dictionary where keys are policy numbers and values are dictionaries mapping user IDs
                to their respective update function arguments.
            action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
                A dictionary mapping user IDs to their respective actions taken at each decision time.
                Only applies to in-study decision times!
            suppress_all_data_checks (bool):
                If True, suppresses carrying out any data checks at all.
            suppress_interactive_data_checks (bool):
                If True, suppresses interactive data checks that would otherwise be performed to ensure
                the correctness of the threaded arguments. The checks are still performed, but
                any interactive prompts are suppressed.
            only_latest_block (bool):
                If True, only uses data from the most recent update.

        Returns:
            jnp.ndarray:
                A 2D JAX NumPy array holding the average weighted estimating function stack.
            jnp.ndarray:
                The same, again. We will differentiate the first output.
        """

        algorithm_estimating_func = (
            jax.grad(alg_update_func, argnums=alg_update_func_args_beta_index)
            if (alg_update_func_type == FunctionTypes.LOSS)
            else alg_update_func
        )

        betas, _ = unflatten_params(
            flattened_betas_and_theta,
            beta_dim,
            0,
        )
        # 1. If only_latest_block is True, we need to filter all the arguments to only
        # include those relevant to the latest update. We still need action probabilities
        # from the beginning for the weights, but the update function args can be trimmed
        # to the max policy so that the loop single_user_weighted_RL_estimating_function_stacker
        # is only over one policy.
        if only_latest_block:
            logger.info(
                "Filtering algorithm update function arguments to only include those relevant to the latest update."
            )
            max_policy_num = max(beta_index_by_policy_num)
            update_func_args_by_by_user_id_by_policy_num = {
                max_policy_num: update_func_args_by_by_user_id_by_policy_num[
                    max_policy_num
                ]
            }

        # 2. Thread in the betas and theta in all_post_update_betas_and_theta into the arguments
        # supplied for the above functions, so that differentiation works correctly.  The existing
        # values should be the same, but not connected to the parameter we are differentiating
        # with respect to. Note we will also find it useful below to have the action probability args
        # nested dict structure flipped to be user_id -> decision_time -> args, so we do that here too.

        logger.info("Threading in betas to action probability arguments for all users.")
        (
            threaded_action_prob_func_args_by_decision_time_by_user_id,
            action_prob_func_args_by_decision_time_by_user_id,
        ) = thread_action_prob_func_args(
            action_prob_func_args_by_user_id_by_decision_time,
            policy_num_by_decision_time_by_user_id,
            initial_policy_num,
            betas,
            beta_index_by_policy_num,
            action_prob_func_args_beta_index,
        )

        # 3. Thread the central betas into the algorithm update function arguments
        # and replace any action probabilities with reconstructed ones from the above
        # arguments with the central betas introduced.
        logger.info(
            "Threading in betas and beta-dependent action probabilities to algorithm update "
            "function args for all users"
        )
        threaded_update_func_args_by_policy_num_by_user_id = thread_update_func_args(
            update_func_args_by_by_user_id_by_policy_num,
            betas,
            beta_index_by_policy_num,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            threaded_action_prob_func_args_by_decision_time_by_user_id,
            action_prob_func,
        )

        # If action probabilities are used in the algorithm estimating function, make
        # sure that substituting in the reconstructed action probabilities is approximately
        # equivalent to using the original action probabilities.
        if not suppress_all_data_checks and alg_update_func_args_action_prob_index >= 0:
            input_checks.require_threaded_algorithm_estimating_function_args_equivalent(
                algorithm_estimating_func,
                update_func_args_by_by_user_id_by_policy_num,
                threaded_update_func_args_by_policy_num_by_user_id,
                suppress_interactive_data_checks,
            )

        # 5. Now we can compute the weighted estimating function stacks for all users
        # as well as collect related values used to construct the adaptive and classical
        # sandwich variances.
        RL_stacks = jnp.array(
            [
                self.single_user_weighted_RL_estimating_function_stacker(
                    beta_dim,
                    user_id,
                    action_prob_func,
                    algorithm_estimating_func,
                    action_prob_func_args_beta_index,
                    action_prob_func_args_by_decision_time_by_user_id[user_id],
                    threaded_action_prob_func_args_by_decision_time_by_user_id[user_id],
                    threaded_update_func_args_by_policy_num_by_user_id[user_id],
                    policy_num_by_decision_time_by_user_id[user_id],
                    action_by_decision_time_by_user_id[user_id],
                    beta_index_by_policy_num,
                )
                for user_id in user_ids.tolist()
            ]
        )

        # 6. We will differentiate the first output, while the second will be used
        # for an estimating function sum check.
        mean_stack_across_users = jnp.mean(RL_stacks, axis=0)
        return mean_stack_across_users, mean_stack_across_users

    def single_user_weighted_RL_estimating_function_stacker(
        self,
        beta_dim: int,
        user_id: collections.abc.Hashable,
        action_prob_func: callable,
        algorithm_estimating_func: callable,
        action_prob_func_args_beta_index: int,
        action_prob_func_args_by_decision_time: dict[
            int, dict[collections.abc.Hashable, tuple[Any, ...]]
        ],
        threaded_action_prob_func_args_by_decision_time: dict[
            collections.abc.Hashable, dict[int, tuple[Any, ...]]
        ],
        threaded_update_func_args_by_policy_num: dict[
            collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
        ],
        policy_num_by_decision_time: dict[
            collections.abc.Hashable, dict[int, int | float]
        ],
        action_by_decision_time: dict[collections.abc.Hashable, dict[int, int]],
        beta_index_by_policy_num: dict[int | float, int],
    ) -> tuple[
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
    ]:
        """
        Computes a weighted estimating function stack for a given algorithm estimating function
        and arguments, inference estimating functio and arguments, and action probability function and
        arguments.

        Args:
            beta_dim (list[jnp.ndarray]):
                A list of 1D JAX NumPy arrays corresponding to the betas produced by all updates.

            user_id (collections.abc.Hashable):
                The user ID for which to compute the weighted estimating function stack.

            action_prob_func (callable):
                The function used to compute the probability of action 1 at a given decision time for
                a particular user given their state and the algorithm parameters.

            algorithm_estimating_func (callable):
                The estimating function that corresponds to algorithm updates.

            action_prob_func_args_beta_index (int):
                The index of the beta argument in the action probability function's arguments.

            action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
                A map from decision times to tuples of arguments for this user for the action
                probability function. This is for all decision times (args are an empty
                tuple if they are not in the study). Should be sorted by decision time. NOTE THAT THESE
                ARGS DO NOT CONTAIN THE SHARED BETAS, making them impervious to the differentiation that
                will occur.

            threaded_action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
                A map from decision times to tuples of arguments for the action
                probability function, with the shared betas threaded in for differentation. Decision
                times should be sorted.

            threaded_update_func_args_by_policy_num (dict[int | float, dict[collections.abc.Hashable, tuple[Any, ...]]]):
                A map from policy numbers to tuples containing the arguments for
                the corresponding estimating functions for this user, with the shared betas threaded in
                for differentiation.  This is for all non-initial, non-fallback policies. Policy numbers
                should be sorted.

            policy_num_by_decision_time (dict[collections.abc.Hashable, dict[int, int | float]]):
                A dictionary mapping decision times to the policy number in use. This may be
                user-specific. Should be sorted by decision time. Only applies to in-study decision
                times!

            action_by_decision_time (dict[collections.abc.Hashable, dict[int, int]]):
                A dictionary mapping decision times to actions taken. Only applies to in-study decision
                times!

            beta_index_by_policy_num (dict[int | float, int]):
                A dictionary mapping policy numbers to the index of the corresponding beta in
                all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

        Returns:
            jnp.ndarray: A 1-D JAX NumPy array representing the RL portion of the user's weighted
            estimating function stack.
        """

        logger.info(
            "Computing weighted estimating function stack for user %s.", user_id
        )

        # First, reformat the supplied data into more convenient structures.

        # 1. Form a dictionary mapping policy numbers to the first time they were
        # applicable (for this user). Note that this includes ALL policies, initial
        # fallbacks included.
        # Collect the first time after the first update separately for convenience.
        # These are both used to form the Radon-Nikodym weights for the right times.
        min_time_by_policy_num, first_time_after_first_update = (
            get_min_time_by_policy_num(
                policy_num_by_decision_time,
                beta_index_by_policy_num,
            )
        )

        # 2. Get the start and end times for this user.
        user_start_time = math.inf
        user_end_time = -math.inf
        for decision_time in action_by_decision_time:
            user_start_time = min(user_start_time, decision_time)
            user_end_time = max(user_end_time, decision_time)

        # 3. Form a stack of weighted estimating equations, one for each update of the algorithm.
        logger.info(
            "Computing the algorithm component of the weighted estimating function stack for user %s.",
            user_id,
        )

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
        # user id is present in the user args dict because we may call this on a
        # subset of the user arg dict when we are batching arguments by shape
        sorted_threaded_action_prob_args_by_decision_time = {
            decision_time: threaded_action_prob_func_args_by_decision_time[
                decision_time
            ]
            for decision_time in range(user_start_time, user_end_time + 1)
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
            fun=get_radon_nikodym_weight,
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

        algorithm_component = jnp.concatenate(
            [
                # Here we compute a product of Radon-Nikodym weights
                # for all decision times after the first update and before the update
                # update under consideration took effect, for which the user was in the study.
                (
                    jnp.prod(
                        all_weights[
                            # The earliest time after the first update where the user was in
                            # the study
                            max(
                                first_time_after_first_update,
                                user_start_time,
                            )
                            - decision_time_to_all_weights_index_offset :
                            # One more than the latest time the user was in the study before the time
                            # the update under consideration first applied. Note the + 1 because range
                            # does not include the right endpoint.
                            min(
                                min_time_by_policy_num.get(policy_num, math.inf),
                                user_end_time + 1,
                            )
                            - decision_time_to_all_weights_index_offset,
                        ]
                        # If the user exited the study before there were any updates,
                        # this variable will be None and the above code to grab a weight would
                        # throw an error. Just use 1 to include the unweighted estimating function
                        # if they have data to contribute to the update.
                        if first_time_after_first_update is not None
                        else 1
                    )  # Now use the above to weight the alg estimating function for this update
                    * algorithm_estimating_func(*update_args)
                    # If there are no arguments for the update function, the user is not yet in the
                    # study, so we just add a zero vector contribution to the sum across users.
                    # Note that after they exit, they still contribute all their data to later
                    # updates.
                    if update_args
                    else jnp.zeros(beta_dim)
                )
                # vmapping over this would be tricky due to different shapes across updates
                for policy_num, update_args in threaded_update_func_args_by_policy_num.items()
            ]
        )

        if algorithm_component.size % beta_dim != 0:
            raise ValueError(
                "The algorithm component of the weighted estimating function stack does not have a "
                "size that is a multiple of the beta dimension. This likely means that the "
                "algorithm estimating function is not returning a vector of the correct size."
            )

        return algorithm_component
