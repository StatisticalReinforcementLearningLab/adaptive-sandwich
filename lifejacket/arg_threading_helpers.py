from __future__ import annotations

from typing import Any
import collections
import logging

import jax
import jax.numpy as jnp

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def replace_tuple_index(tupl, index, value):
    return tupl[:index] + (value,) + tupl[index + 1 :]


def thread_action_prob_func_args(
    action_prob_func_args_by_user_id_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    all_post_update_betas: jnp.ndarray,
    beta_index_by_policy_num: dict[int | float, int],
    action_prob_func_args_beta_index: int,
) -> tuple[
    dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]],
    dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]],
]:
    """
    Threads the shared betas into the action probability function arguments for each user and
    decision time to enable correct differentiation.

    Args:
        action_prob_func_args_by_user_id_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to maps of user ids to tuples of arguments for action
            probability function. This is for all decision times for all users (args are an empty
            tuple if they are not in the study). Should be sorted by decision time.

        policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A dictionary mapping decision times to the policy number in use. This may be user-specific.
            Should be sorted by decision time.

        initial_policy_num (int | float): The policy number of the initial policy before any
            updates.

        all_post_update_betas (jnp.ndarray):
            A 2D array of beta values to be introduced into arguments to
            facilitate differentiation.  They will be the same value as what they replace, but this
            introduces direct dependence on the parameter we will differentiate with respect to.

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to their respective
            beta indices in all_post_update_betas.

        action_prob_func_args_beta_index (int):
            The index in the action probability function arguments tuple
            where the beta value should be inserted.

    Returns:
        dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]:
            A map from user ids to maps of decision times to action probability function
            arguments tuples with the shared betas threaded in. Note the key order switch.
    """
    threaded_action_prob_func_args_by_decision_time_by_user_id = (
        collections.defaultdict(dict)
    )
    action_prob_func_args_by_decision_time_by_user_id = collections.defaultdict(dict)
    for (
        decision_time,
        action_prob_func_args_by_user_id,
    ) in action_prob_func_args_by_user_id_by_decision_time.items():
        for user_id, args in action_prob_func_args_by_user_id.items():
            # Always add a contribution to the reversed key order dictionary.
            action_prob_func_args_by_decision_time_by_user_id[user_id][
                decision_time
            ] = args

            # Now proceed with the threading, if necessary.
            if not args:
                threaded_action_prob_func_args_by_decision_time_by_user_id[user_id][
                    decision_time
                ] = ()
                continue

            policy_num = policy_num_by_decision_time_by_user_id[user_id][decision_time]

            # The expectation is that fallback policies have empty args, and the only other
            # policy not represented in beta_index_by_policy_num is the initial policy.
            if policy_num == initial_policy_num:
                threaded_action_prob_func_args_by_decision_time_by_user_id[user_id][
                    decision_time
                ] = action_prob_func_args_by_user_id[user_id]
                continue

            beta_to_introduce = all_post_update_betas[
                beta_index_by_policy_num[policy_num]
            ]
            threaded_action_prob_func_args_by_decision_time_by_user_id[user_id][
                decision_time
            ] = replace_tuple_index(
                action_prob_func_args_by_user_id[user_id],
                action_prob_func_args_beta_index,
                beta_to_introduce,
            )

    return (
        threaded_action_prob_func_args_by_decision_time_by_user_id,
        action_prob_func_args_by_decision_time_by_user_id,
    )


def thread_update_func_args(
    update_func_args_by_by_user_id_by_policy_num: dict[
        int | float, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    all_post_update_betas: jnp.ndarray,
    beta_index_by_policy_num: dict[int | float, int],
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    alg_update_func_args_previous_betas_index: int,
    threaded_action_prob_func_args_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    action_prob_func: callable,
) -> dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]:
    """
    Threads the shared betas into the algorithm update function arguments for each user and
    policy update to enable correct differentiation.  This is done by replacing the betas in the
    update function arguments with the shared betas, and if necessary replacing action probabilities
    with reconstructed action probabilities computed using the shared betas.

    Args:
        update_func_args_by_by_user_id_by_policy_num (dict[int | float, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A dictionary where keys are policy
            numbers and values are dictionaries mapping user IDs to their respective update function
            arguments.

        all_post_update_betas (jnp.ndarray):
            A 2D array of beta values to be introduced into arguments to
            facilitate differentiation.  They will be the same value as what they replace, but this
            introduces direct dependence on the parameter we will differentiate with respect to.

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to their respective
            beta indices in all_post_update_betas.

        alg_update_func_args_beta_index (int):
            The index in the update function arguments tuple
            where the beta value should be inserted.

        alg_update_func_args_action_prob_index (int):
            The index in the update function arguments
            tuple where new beta-threaded action probabilities should be inserted, if applicable.
            -1 otherwise.

        alg_update_func_args_action_prob_times_index (int):
            If action probabilities are supplied
            to the update function, this is the index in the arguments where an array of times for
            which the given action probabilities apply is provided.

        alg_update_func_args_previous_betas_index (int):
            The index in the update function with previous beta parameters

        threaded_action_prob_func_args_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to the function arguments required to compute action
            probabilities for this user, and with the shared betas thread in.

        action_prob_func (callable):
            A function that computes an action 1 probability given the appropriate arguments.

    Returns:
        dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]:
            A map from user ids to maps of policy numbers to update function
            arguments tuples for the specified user with the shared betas threaded in. Note the key
            order switch relative to the supplied args!
    """
    threaded_update_func_args_by_policy_num_by_user_id = collections.defaultdict(dict)
    for (
        policy_num,
        update_func_args_by_user_id,
    ) in update_func_args_by_by_user_id_by_policy_num.items():
        for user_id, args in update_func_args_by_user_id.items():
            if not args:
                threaded_update_func_args_by_policy_num_by_user_id[user_id][
                    policy_num
                ] = ()
                continue

            logger.debug(
                "Threading in shared betas to update function arguments for user %s and policy number %s.",
                user_id,
                policy_num,
            )

            beta_to_introduce = all_post_update_betas[
                beta_index_by_policy_num[policy_num]
            ]
            threaded_update_func_args_by_policy_num_by_user_id[user_id][policy_num] = (
                replace_tuple_index(
                    update_func_args_by_user_id[user_id],
                    alg_update_func_args_beta_index,
                    beta_to_introduce,
                )
            )
            if alg_update_func_args_previous_betas_index >= 0:
                previous_betas_to_introduce = all_post_update_betas[
                    : len(
                        update_func_args_by_user_id[user_id][
                            alg_update_func_args_previous_betas_index
                        ]
                    )
                ]
                if previous_betas_to_introduce.size > 0:
                    threaded_update_func_args_by_policy_num_by_user_id[user_id][
                        policy_num
                    ] = replace_tuple_index(
                        threaded_update_func_args_by_policy_num_by_user_id[user_id][
                            policy_num
                        ],
                        alg_update_func_args_previous_betas_index,
                        previous_betas_to_introduce,
                    )

            if alg_update_func_args_action_prob_index >= 0:
                logger.debug(
                    "Action probabilities are used in the algorithm update function. Reconstructing them using the shared betas."
                )
                action_prob_times = update_func_args_by_user_id[user_id][
                    alg_update_func_args_action_prob_times_index
                ]
                # Vectorized computation of action_probs_to_introduce using jax.vmap
                flattened_times = action_prob_times.flatten()
                args_list = [
                    threaded_action_prob_func_args_by_decision_time_by_user_id[user_id][
                        int(t)
                    ]
                    for t in flattened_times.tolist()
                ]
                if len(args_list) == 0:
                    action_probs_to_introduce = jnp.array([]).reshape(
                        update_func_args_by_user_id[user_id][
                            alg_update_func_args_action_prob_index
                        ].shape
                    )
                else:
                    batched_args = list(zip(*args_list))
                    # Ensure each argument is at least 2D for batching, to avoid shape issues with scalars
                    batched_tensors = []
                    for arg_group in batched_args:
                        arr = jnp.array(arg_group)
                        if arr.ndim == 1:
                            arr = arr[:, None]
                        batched_tensors.append(arr)
                    vmapped_func = jax.vmap(
                        action_prob_func, in_axes=tuple(0 for _ in batched_tensors)
                    )
                    action_probs_to_introduce = vmapped_func(*batched_tensors).reshape(
                        update_func_args_by_user_id[user_id][
                            alg_update_func_args_action_prob_index
                        ].shape
                    )
                threaded_update_func_args_by_policy_num_by_user_id[user_id][
                    policy_num
                ] = replace_tuple_index(
                    threaded_update_func_args_by_policy_num_by_user_id[user_id][
                        policy_num
                    ],
                    alg_update_func_args_action_prob_index,
                    action_probs_to_introduce,
                )
    return threaded_update_func_args_by_policy_num_by_user_id


def thread_inference_func_args(
    inference_func_args_by_user_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_func_args_theta_index: int,
    theta: jnp.ndarray,
    inference_func_args_action_prob_index: int,
    threaded_action_prob_func_args_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    inference_action_prob_decision_times_by_user_id: dict[
        collections.abc.Hashable, list[int]
    ],
    action_prob_func: callable,
) -> dict[collections.abc.Hashable, tuple[Any, ...]]:
    """
    Threads the shared theta into the inference function arguments for each user to enable correct
    differentiation.  This is done by replacing the theta in the inference function arguments with
    theta. If applicable, action probabilities are also replaced with reconstructed action
    probabilities computed using the shared betas.

    Args:
        inference_func_args_by_user_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping user IDs to their respective inference function arguments.

        inference_func_args_theta_index (int):
            The index in the inference function arguments tuple
            where the theta value should be inserted.

        theta (jnp.ndarray):
            The theta value to be threaded into the inference function arguments.

        inference_func_args_action_prob_index (int):
            The index in the inference function arguments
            tuple where new beta-threaded action probabilities should be inserted, if applicable.
            -1 otherwise.

        threaded_action_prob_func_args_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to the function arguments required to compute action
            probabilities for this user, and with the shared betas thread in.

        inference_action_prob_decision_times_by_user_id (dict[collections.abc.Hashable, list[int]]):
            For each user, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.

        action_prob_func (callable):
            A function that computes an action 1 probability given the appropriate arguments.
    Returns:
        dict[collections.abc.Hashable, tuple[Any, ...]]:
            A map from user ids to tuples of inference function arguments with the shared theta
            threaded in.
    """

    threaded_inference_func_args_by_user_id = {}
    for user_id, args in inference_func_args_by_user_id.items():
        threaded_inference_func_args_by_user_id[user_id] = replace_tuple_index(
            args,
            inference_func_args_theta_index,
            theta,
        )

        if inference_func_args_action_prob_index >= 0:
            # Use a vmap-like pattern to compute action probabilities in batch.
            action_prob_times_flattened = (
                inference_action_prob_decision_times_by_user_id[user_id].flatten()
            )
            args_list = [
                threaded_action_prob_func_args_by_decision_time_by_user_id[user_id][
                    int(t)
                ]
                for t in action_prob_times_flattened.tolist()
            ]
            if len(args_list) == 0:
                action_probs_to_introduce = jnp.array([]).reshape(
                    args[inference_func_args_action_prob_index].shape
                )
            else:
                batched_args = list(zip(*args_list))
                batched_tensors = []
                for arg_group in batched_args:
                    arr = jnp.array(arg_group)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    batched_tensors.append(arr)
                vmapped_func = jax.vmap(
                    action_prob_func, in_axes=tuple(0 for _ in batched_tensors)
                )
                action_probs_to_introduce = vmapped_func(*batched_tensors).reshape(
                    args[inference_func_args_action_prob_index].shape
                )
            threaded_inference_func_args_by_user_id[user_id] = replace_tuple_index(
                threaded_inference_func_args_by_user_id[user_id],
                inference_func_args_action_prob_index,
                action_probs_to_introduce,
            )
    return threaded_inference_func_args_by_user_id
