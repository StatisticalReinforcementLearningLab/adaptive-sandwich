import logging
import os

import jax
from jax import numpy as jnp
import numpy as np

from helper_functions import (
    conditional_x_or_one_minus_x,
    load_module_from_source_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def calculate_pi_and_weight_gradients(
    study_df,
    in_study_col_name,
    action_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_func_filename,
    action_prob_func_args,
    action_prob_func_args_beta_index,
):
    """
    For all users, compute the gradient with respect to beta of both the pi function
    (which takes an action and state and gives the probability of selecting the action)
    and the Radon-Nikodym weight (derived from pi functions as described in the paper)
    for the current time, evaluated at the most recent beta.
    """

    logger.info("Calculating pi and weight gradients with respect to beta.")

    # Retrieve the action prob function from file
    action_probs_module = load_module_from_source_file(
        "action_probs", action_prob_func_filename
    )
    # NOTE the assumption that the function and file have the same name
    action_prob_func_name = os.path.basename(action_prob_func_filename)
    try:
        action_prob_func = getattr(action_probs_module, action_prob_func_name)
    except AttributeError as e:
        raise ValueError(
            "Unable to import action probability function.  Please verify the file has the same name as the function of interest."
        ) from e

    # TODO: Fallback for users missing or require complete?
    pi_and_weight_gradients_by_calendar_t = {}
    for calendar_t, user_args_dict in action_prob_func_args.items():

        # Sort users to be cautious
        # TODO: Can do sorting of user list and arg length determination just once
        user_ids = list(user_args_dict.keys())
        sorted_user_ids = sorted(user_ids)
        sorted_user_args_dict = {
            user_id: user_args_dict[user_id] for user_id in sorted_user_ids
        }

        num_args = len(user_args_dict[user_ids[0]])
        batched_arg_lists = [[]] * num_args
        for user_args in sorted_user_args_dict.values():
            for idx, arg in user_args:
                batched_arg_lists[idx].append(arg)

        # Note this stacking works with incremental recruitment only because we
        # fill in states for out-of-study times such that all users have the
        # same state matrix size
        # TODO: Are there cases where an arg can't be tensorized into a numpy array
        # because of type?
        logger.info("Reforming batched data lists into tensors.")
        batched_arg_tensors = [
            jnp.vstack(batched_arg_list) for batched_arg_list in batched_arg_lists
        ]

        logger.info("Forming pi gradients with respect to beta.")
        # Note that we care about the probability of action 1 specifically,
        # not the taken action.
        pi_gradients = get_pi_gradients_batched(
            action_prob_func, action_prob_func_args_beta_index, *batched_arg_tensors
        )

        # TODO: betas should be verified to be the same across users now or earlier
        logger.info("Forming weight gradients with respect to beta.")
        batched_actions_tensor = collect_batched_actions(
            study_df,
            calendar_t,
            sorted_user_ids,
            in_study_col_name,
            action_col_name,
            calendar_t_col_name,
            user_id_col_name,
        )
        weight_gradients = get_weight_gradients_batched(
            action_prob_func,
            action_prob_func_args_beta_index,
            batched_actions_tensor,
            *batched_arg_tensors,
        )

        logger.info("Collecting pi gradients into algorithm stats dictionary.")
        pi_and_weight_gradients_by_calendar_t.setdefault(calendar_t, {})[
            "pi_gradients_by_user_id"
        ] = {user_id: pi_gradients[i] for i, user_id in enumerate(user_ids)}

        logger.info("Collecting weight gradients into algorithm stats dictionary.")
        pi_and_weight_gradients_by_calendar_t.setdefault(calendar_t, {})[
            "weight_gradients_by_user_id"
        ] = {user_id: weight_gradients[i] for i, user_id in enumerate(user_ids)}
    return pi_and_weight_gradients_by_calendar_t


def collect_batched_actions(
    study_df,
    calendar_t,
    sorted_user_ids,
    in_study_col_name,
    action_col_name,
    calendar_t_col_name,
    user_id_col_name,
):
    fallback_action = 0

    batched_actions_list = []
    for user_id in sorted_user_ids:
        filtered_user_data = study_df.loc[
            study_df[user_id_col_name] == user_id,
            study_df[calendar_t_col_name] == calendar_t,
        ]
        in_study = filtered_user_data[in_study_col_name].values[0]
        action = (
            filtered_user_data[action_col_name].values[0]
            if in_study
            else fallback_action
        )
        batched_actions_list.append(action)

    return jnp.dstack(batched_actions_list)


def get_radon_nikodym_weight(
    beta_target,
    action_prob_func,
    action_prob_func_args_beta_index,
    action,
    *action_prob_func_args_single_user,
):

    beta_target_action_prob_func_args_single_user = [*action_prob_func_args_single_user]
    beta_target_action_prob_func_args_single_user[action_prob_func_args_beta_index] = (
        beta_target
    )

    # TODO: The [()] is probably not generally necessary. specify format
    # of function clearly to make it either way
    pi_beta = action_prob_func(*action_prob_func_args_single_user)[()]
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_user)[
        ()
    ]
    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


# TODO: Docstring
def get_pi_gradients_batched(
    action_prob_func, action_prob_func_args_beta_index, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.grad(action_prob_func, action_prob_func_args_beta_index),
        in_axes=[0] * len(batched_arg_tensors),
        out_axes=0,
    )(*batched_arg_tensors)


# TODO: Docstring
def get_weight_gradients_batched(
    beta_target,
    action_prob_func,
    action_prob_func_args_beta_index,
    batched_actions_tensor,
    *batched_arg_tensors,
):
    # NOTE the (4 + index) is due to the fact that we have four fixed args in
    # the above dynamic definition of the weight function before passing in the
    # action prob args
    return jax.vmap(
        fun=jax.grad(get_radon_nikodym_weight, 4 + action_prob_func_args_beta_index),
        in_axes=[None, None, None, 0] + [0] * len(batched_arg_tensors),
        out_axes=0,
    )(
        beta_target,
        action_prob_func,
        action_prob_func_args_beta_index,
        batched_actions_tensor,
        *batched_arg_tensors,
    )


# TODO: Docstring
# TODO: JIT whole function? or just gradient and hessian batch functions
def calculate_loss_derivatives(
    study_df,
    RL_loss_func_filename,
    RL_loss_func_args,
    RL_loss_func_args_beta_index,
    RL_loss_func_args_action_prob_index,
    policy_num_col_name,
    calendar_t_col_name,
):
    logger.info("Calculating loss gradients and hessians with respect to beta.")

    # Because we perform algorithm updates at the *end* of a timestep, the
    # first timestep they apply to is one more than the time of the update.
    # Hence we add 1 here for notational consistency with the paper.

    # TODO: Note that we don't need the loss gradient for the first update
    # time... include anyway?

    # TODO: Do this pandas filtering only once. Also consider doing it in numpy?
    # Or just do everything in matrix form somehow?
    # https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array

    # Retrieve the RL function from file
    RL_loss_module = load_module_from_source_file("RL_loss", RL_loss_func_filename)
    # NOTE the assumption that the function and file have the same name
    RL_loss_func_name = os.path.basename(RL_loss_func_filename)
    try:
        RL_loss_func = getattr(RL_loss_module, RL_loss_func_name)
    except AttributeError as e:
        raise ValueError(
            "Unable to import RL loss function.  Please verify the file has the same name as the function of interest."
        ) from e

    # TODO: Fallback for users missing or require complete?
    RL_update_derivatives_by_calendar_t = {}

    for policy_num, user_args_dict in RL_loss_func_args.items():

        # Sort users to be cautious
        # TODO: Can do sorting of user list and arg length determination just once
        user_ids = list(user_args_dict.keys())
        sorted_user_ids = sorted(user_ids)
        sorted_user_args_dict = {
            user_id: user_args_dict[user_id] for user_id in sorted_user_ids
        }

        num_args = len(user_args_dict[user_ids[0]])
        batched_arg_lists = [[]] * num_args
        for user_args in sorted_user_args_dict.values():
            for idx, arg in user_args:
                batched_arg_lists[idx].append(arg)

        # Note this stacking works with incremental recruitment only because we
        # fill in states for out-of-study times such that all users have the
        # same state matrix size
        # TODO: Are there cases where an arg can't be tensorized into a numpy array
        # because of type?
        logger.info("Reforming batched data lists into tensors.")
        batched_arg_tensors = [
            jnp.dstack(batched_arg_list) for batched_arg_list in batched_arg_lists
        ]

        logger.info("Forming loss gradients with respect to beta.")
        loss_gradients = get_RL_loss_gradients_batched(
            RL_loss_func, RL_loss_func_args_beta_index, *batched_arg_tensors
        )
        logger.info("Forming loss hessians with respect to beta")
        loss_hessians = get_RL_loss_hessians_batched(
            RL_loss_func, RL_loss_func_args_beta_index, *batched_arg_tensors
        )
        logger.info(
            "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time"
        )
        loss_gradient_pi_derivatives = get_RL_loss_gradient_derivatives_wrt_pi_batched(
            RL_loss_func,
            RL_loss_func_args_beta_index,
            RL_loss_func_args_action_prob_index,
            *batched_arg_tensors,
        )

        # We store these loss gradients by the first time the resulting parameters
        # apply to, so determine this time.
        first_applicable_time = get_first_applicable_time(
            study_df, policy_num, policy_num_col_name, calendar_t_col_name
        )
        RL_update_derivatives_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {user_id: loss_gradients[i] for i, user_id in enumerate(user_ids)}

        RL_update_derivatives_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = np.mean(loss_hessians, axis=0)

        RL_update_derivatives_by_calendar_t[first_applicable_time][
            "loss_gradient_pi_derivatives_by_user_id"
        ] = {
            user_id: loss_gradient_pi_derivatives[i].squeeze()
            for i, user_id in enumerate(user_ids)
        }
    return RL_update_derivatives_by_calendar_t


# TODO: note the 2s for in axes vs 0s for action prob func args. It seems right
# roughly, but think about generality. It occurs because of dstack here vs vstack
# there
def get_RL_loss_gradients_batched(
    RL_loss_func, RL_loss_func_args_beta_index, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.grad(RL_loss_func, RL_loss_func_args_beta_index),
        in_axes=[2] * len(batched_arg_tensors),
        out_axes=0,
    )(*batched_arg_tensors)


def get_RL_loss_hessians_batched(
    RL_loss_func, RL_loss_func_args_beta_index, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.hessian(RL_loss_func, RL_loss_func_args_beta_index),
        in_axes=[2] * len(batched_arg_tensors),
        out_axes=0,
    )(*batched_arg_tensors)


def get_RL_loss_gradient_derivatives_wrt_pi_batched(
    RL_loss_func,
    RL_loss_func_args_beta_index,
    RL_loss_func_args_action_prob_index,
    *batched_arg_tensors,
):
    return jax.vmap(
        fun=jax.jacrev(
            jax.grad(RL_loss_func, RL_loss_func_args_beta_index),
            RL_loss_func_args_action_prob_index,
        ),
        in_axes=[2] * len(batched_arg_tensors),
        out_axes=0,
    )(*batched_arg_tensors)


#  TODO: Is there a better way to calculate this? This seems like it should
# be reliable, not messing up when a policy was actually available. If study
# df says policy was used, that should be correct.
def get_first_applicable_time(
    study_df, policy_num, policy_num_col_name, calendar_t_col_name
):
    return study_df[
        study_df[policy_num_col_name] == policy_num, calendar_t_col_name
    ].min()
