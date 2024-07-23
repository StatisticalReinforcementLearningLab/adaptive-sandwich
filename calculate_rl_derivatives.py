import logging
import os
import collections

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


# Note this stacking works with incremental recruitment only because we
# fill in states for out-of-study times such that all users have the
# same state matrix size
# TODO: Are there cases where an arg can't be tensorized into a numpy array
# because of type?
# TODO: I am assuming args can be placed in numpy array here
# The casework here is annoying, but to play nicely with vmap we need
# to stack arrays and non-arrays differently. We must be able to iterate
# along the first dimension to get the values for different users in the batch.
# If we vstack a 1d array of scalars we get a 2d array--list of 1-element lists--so
# iterating along first dimension gives 1-element arrays now. We do not
# want that.
# TODO: Try except and nice error message
def stack_batched_arg_list_into_tensor_and_get_batch_axes(batched_arg_lists):

    batched_arg_tensors = []

    # This ends up being all zeros because of the way we are (now) doing the
    # stacking, but better to not assume that externally and send out what
    # we've done with this list.
    batch_axes = []

    for batched_arg_list in batched_arg_lists:
        if (
            isinstance(
                batched_arg_list[0],
                (jnp.ndarray, np.ndarray),
            )
            and batched_arg_list[0].ndim == 2
        ):
            # We have a matrix (2D array) arg
            batched_arg_tensors.append(jnp.stack(batched_arg_list, 0))
            batch_axes.append(0)
        elif isinstance(
            batched_arg_list[0],
            (collections.abc.Sequence, jnp.ndarray, np.ndarray),
        ) and not isinstance(batched_arg_list[0], str):
            # We have a vector (1D array) arg
            batched_arg_tensors.append(jnp.vstack(batched_arg_list))
            batch_axes.append(0)
        else:
            # Otherwise we should have a list of scalars. Just turn into a
            # jnp array.
            batched_arg_tensors.append(jnp.array(batched_arg_list))
            batch_axes.append(0)

    return batched_arg_tensors, batch_axes


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
    for all update times.
    """

    logger.info("Calculating pi and weight gradients with respect to beta.")

    # Retrieve the action prob function from file
    action_probs_module = load_module_from_source_file(
        "action_probs", action_prob_func_filename
    )
    # NOTE the assumption that the function and file have the same name
    action_prob_func_name = os.path.basename(action_prob_func_filename).split(".")[0]
    try:
        action_prob_func = getattr(action_probs_module, action_prob_func_name)
    except AttributeError as e:
        raise ValueError(
            "Unable to import action probability function.  Please verify the file has the same name as the function of interest."
        ) from e
    # TODO: Fallback for users missing or require complete?
    pi_and_weight_gradients_by_calendar_t = {}
    user_ids = list(next(iter(action_prob_func_args.values())).keys())
    sorted_user_ids = sorted(user_ids)
    for calendar_t, user_args_dict in action_prob_func_args.items():

        pi_gradients, weight_gradients = calculate_pi_and_weight_gradients_specific_t(
            study_df,
            in_study_col_name,
            action_col_name,
            calendar_t_col_name,
            user_id_col_name,
            action_prob_func,
            action_prob_func_args_beta_index,
            calendar_t,
            user_args_dict,
            sorted_user_ids,
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


def calculate_pi_and_weight_gradients_specific_t(
    study_df,
    in_study_col_name,
    action_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_func,
    action_prob_func_args_beta_index,
    calendar_t,
    user_args_dict,
    sorted_user_ids,
):
    # Sort users to be cautious
    sorted_user_args_dict = {
        user_id: user_args_dict[user_id] for user_id in sorted_user_ids
    }

    num_args = len(sorted_user_args_dict[sorted_user_ids[0]])
    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]
    for user_args in sorted_user_args_dict.values():
        for idx, arg in enumerate(user_args):
            batched_arg_lists[idx].append(arg)

    logger.info("Reforming batched data lists into tensors.")
    batched_arg_tensors, batch_axes = (
        stack_batched_arg_list_into_tensor_and_get_batch_axes(batched_arg_lists)
    )

    logger.info("Forming pi gradients with respect to beta.")
    # Note that we care about the probability of action 1 specifically,
    # not the taken action.
    pi_gradients = get_pi_gradients_batched(
        action_prob_func,
        action_prob_func_args_beta_index,
        batch_axes,
        *batched_arg_tensors,
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
    # Note the first argument here: we extract the betas to pass in
    # again as the "target" denominator betas, whereas we differentiate with
    # respect to the betas in the numerator. Also note that these betas are
    # redundant across users: it's just the same thing repeated num users
    # times.
    weight_gradients = get_weight_gradients_batched(
        batched_arg_tensors[action_prob_func_args_beta_index],
        action_prob_func,
        action_prob_func_args_beta_index,
        batched_actions_tensor,
        batch_axes,
        *batched_arg_tensors,
    )

    return pi_gradients, weight_gradients


# TODO: is it ok to get the action from the study df? No issues with actions taken
# but not known about?
# TODO: This fallback action business is a bit weird and probably should not
# happen.  Probably just need to communicate they must set an appropriate fallback
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
            (study_df[user_id_col_name] == user_id)
            & (study_df[calendar_t_col_name] == calendar_t)
        ]
        in_study = filtered_user_data[in_study_col_name].values[0]
        action = (
            filtered_user_data[action_col_name].values[0]
            if in_study
            else fallback_action
        )
        batched_actions_list.append(action)

    return jnp.array(batched_actions_list)


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
    action_prob_func, action_prob_func_args_beta_index, batch_axes, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.grad(action_prob_func, action_prob_func_args_beta_index),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


# TODO: Docstring
def get_weight_gradients_batched(
    batched_beta_target_tensor,
    action_prob_func,
    action_prob_func_args_beta_index,
    batched_actions_tensor,
    batch_axes,
    *batched_arg_tensors,
):
    # NOTE the (4 + index) is due to the fact that we have four fixed args in
    # the above definition of the weight function before passing in the action
    # prob args
    return jax.vmap(
        fun=jax.grad(get_radon_nikodym_weight, 4 + action_prob_func_args_beta_index),
        in_axes=[0, None, None, 0] + batch_axes,
        out_axes=0,
    )(
        batched_beta_target_tensor,
        action_prob_func,
        action_prob_func_args_beta_index,
        batched_actions_tensor,
        *batched_arg_tensors,
    )


# TODO: Docstring
# TODO: JIT whole function? or just gradient and hessian batch functions
def calculate_rl_loss_derivatives(
    study_df,
    rl_loss_func_filename,
    rl_loss_func_args,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    policy_num_col_name,
    calendar_t_col_name,
):
    logger.info("Calculating loss gradients and hessians with respect to beta.")

    # Because we perform algorithm updates at the *end* of a timestep, the
    # first timestep they apply to is one more than the time of the update.

    # TODO: Note that we don't need the loss gradient for the first update
    # time... include anyway?

    # Retrieve the RL function from file
    rl_loss_module = load_module_from_source_file("rl_loss", rl_loss_func_filename)
    # NOTE the assumption that the function and file have the same name
    rl_loss_func_name = os.path.basename(rl_loss_func_filename).split(".")[0]
    try:
        rl_loss_func = getattr(rl_loss_module, rl_loss_func_name)
    except AttributeError as e:
        raise ValueError(
            "Unable to import RL loss function.  Please verify the file has the same name as the function of interest."
        ) from e

    # TODO: Fallback for users missing or require complete?
    rl_update_derivatives_by_calendar_t = {}
    user_ids = list(next(iter(rl_loss_func_args.values())).keys())
    sorted_user_ids = sorted(user_ids)
    for policy_num, user_args_dict in rl_loss_func_args.items():
        loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
            calculate_rl_loss_derivatives_specific_update(
                rl_loss_func,
                rl_loss_func_args_beta_index,
                rl_loss_func_args_action_prob_index,
                user_args_dict,
                sorted_user_ids,
            )
        )

        # We store these loss gradients by the first time the resulting parameters
        # apply to, so determine this time.
        # Because we perform algorithm updates at the *end* of a timestep, the
        # first timestep they apply to is one more than the time at which the
        # update data is gathered.
        first_applicable_time = get_first_applicable_time(
            study_df, policy_num, policy_num_col_name, calendar_t_col_name
        )
        rl_update_derivatives_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {user_id: loss_gradients[i] for i, user_id in enumerate(user_ids)}

        rl_update_derivatives_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = np.mean(loss_hessians, axis=0)

        rl_update_derivatives_by_calendar_t[first_applicable_time][
            "loss_gradient_pi_derivatives_by_user_id"
        ] = {
            # NOTE the squeeze here... it is very important.Without it we have
            # a 4D shape (x,y,z,1) array of gradients, and the use of these
            # probabilities assumes (x,y,z).  The squeezing should arguably
            # happen above, but the vmap call spits out a 4D array so in that
            # sense that's the most natural return value.
            user_id: loss_gradient_pi_derivatives[i].squeeze()
            for i, user_id in enumerate(user_ids)
        }
    return rl_update_derivatives_by_calendar_t


def calculate_rl_loss_derivatives_specific_update(
    rl_loss_func,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    user_args_dict,
    sorted_user_ids,
):
    # Sort users to be cautious
    sorted_user_args_dict = {
        user_id: user_args_dict[user_id] for user_id in sorted_user_ids
    }

    num_args = len(sorted_user_args_dict[sorted_user_ids[0]])
    batched_arg_lists = [[] for _ in range(num_args)]
    for user_args in sorted_user_args_dict.values():
        for idx, arg in enumerate(user_args):
            batched_arg_lists[idx].append(arg)

    # Note this stacking works with incremental recruitment only because we
    # fill in states for out-of-study times such that all users have the
    # same state matrix size
    # TODO: Articulate requirement that each arg can be tensorized into a numpy array
    # because of type
    logger.info("Reforming batched data lists into tensors.")
    batched_arg_tensors, batch_axes = (
        stack_batched_arg_list_into_tensor_and_get_batch_axes(batched_arg_lists)
    )

    logger.info("Forming loss gradients with respect to beta.")
    loss_gradients = get_rl_loss_gradients_batched(
        rl_loss_func, rl_loss_func_args_beta_index, batch_axes, *batched_arg_tensors
    )
    logger.info("Forming loss hessians with respect to beta")
    loss_hessians = get_rl_loss_hessians_batched(
        rl_loss_func, rl_loss_func_args_beta_index, batch_axes, *batched_arg_tensors
    )
    logger.info(
        "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time"
    )
    loss_gradient_pi_derivatives = get_rl_loss_gradient_derivatives_wrt_pi_batched(
        rl_loss_func,
        rl_loss_func_args_beta_index,
        rl_loss_func_args_action_prob_index,
        batch_axes,
        *batched_arg_tensors,
    )

    return loss_gradients, loss_hessians, loss_gradient_pi_derivatives


def get_rl_loss_gradients_batched(
    rl_loss_func, rl_loss_func_args_beta_index, batch_axes, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.grad(rl_loss_func, rl_loss_func_args_beta_index),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


def get_rl_loss_hessians_batched(
    rl_loss_func, rl_loss_func_args_beta_index, batch_axes, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.hessian(rl_loss_func, rl_loss_func_args_beta_index),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


def get_rl_loss_gradient_derivatives_wrt_pi_batched(
    rl_loss_func,
    rl_loss_func_args_beta_index,
    rl_loss_func_args_action_prob_index,
    batch_axes,
    *batched_arg_tensors,
):
    return jax.vmap(
        fun=jax.jacrev(
            jax.grad(rl_loss_func, rl_loss_func_args_beta_index),
            rl_loss_func_args_action_prob_index,
        ),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


#  TODO: Is there a better way to calculate this? This seems like it should
# be reliable, not messing up when a policy was actually available. If study
# df says policy was used, that should be correct.
def get_first_applicable_time(
    study_df, policy_num, policy_num_col_name, calendar_t_col_name
):
    return study_df[study_df[policy_num_col_name] == policy_num][
        calendar_t_col_name
    ].min()
