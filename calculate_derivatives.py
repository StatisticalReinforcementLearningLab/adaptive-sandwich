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

# TODO: Consolidate function loading logic


# TODO: Check for exactly the required types earlier
# TODO: Try except and nice error message
# TODO: This is complicated enough to deserve its own unit tests
# TODO: Technically dont need to pad betas but I think handling that is too messy
# TODO: This shape based approach doesn't actually work, because you can't vmap over different shapes duh
# Maybe some kind of grouping by shape? That should actually work. There should be a common trimming
# needed per active recruitment group barring special cases.
def stack_batched_arg_lists_into_tensor(batched_arg_lists):
    """
    Stack a simple Python list of function arguments (across all users for a specific arg position)
    into an array that can be supplied to vmap as batch arguments. vmap requires all elements of
    such a batched array to be the same shape, as do the stacking functions we use here.  Thus
    we take the original argument lists and pad each to constant size within itself, recording the
    original sizes via an array of zeros so they can be unpadded later (we can trim only based the
    size of arguments inside of vmap (see JAX sharp bits)). We also supply the axes one must
    iterate over to get each users's args in a batch.
    """

    padded_batched_arg_tensors = []

    # This ends up being all zeros because of the way we are (now) doing the
    # stacking, but better to not assume that externally and send out what
    # we've done with this list.
    batch_axes = []

    batched_zeros_like_arrays = []
    for batched_arg_list in batched_arg_lists:
        if (
            isinstance(
                batched_arg_list[0],
                (jnp.ndarray, np.ndarray),
            )
            and batched_arg_list[0].ndim > 2
        ):
            raise TypeError("Arrays with dimension greater that 2 are not supported.")
        if (
            isinstance(
                batched_arg_list[0],
                (jnp.ndarray, np.ndarray),
            )
            and batched_arg_list[0].ndim == 2
        ):
            ########## We have a matrix (2D array) arg

            padded_batched_arg_list, batched_zeros_like_list = (
                pad_batched_arg_list_to_max_on_each_dimension(batched_arg_list)
            )

            padded_batched_arg_tensors.append(jnp.stack(padded_batched_arg_list, 0))
            batched_zeros_like_arrays.append(jnp.vstack(batched_zeros_like_list))
            batch_axes.append(0)
        elif isinstance(
            batched_arg_list[0],
            (collections.abc.Sequence, jnp.ndarray, np.ndarray),
        ) and not isinstance(batched_arg_list[0], str):
            ########## We have a vector (1D array) arg
            if not isinstance(batched_arg_list[0], (jnp.ndarray, np.ndarray)):
                try:
                    batched_arg_list = [jnp.ndarray(x) for x in batched_arg_list]
                except Exception as e:
                    raise TypeError(
                        "Argument of sequence type that cannot be cast to JAX numpy array"
                    ) from e
            assert batched_arg_list[0].ndim == 1

            padded_batched_arg_list, batched_zeros_like_list = (
                pad_batched_arg_list_to_max_on_each_dimension(batched_arg_list)
            )

            padded_batched_arg_tensors.append(jnp.vstack(padded_batched_arg_list))
            batched_zeros_like_arrays.append(jnp.vstack(batched_zeros_like_list))
            batch_axes.append(0)
        else:
            ########## Otherwise we should have a list of scalars.
            # Just turn into a jnp array.
            padded_batched_arg_tensors.append(jnp.array(batched_arg_list))
            batched_zeros_like_arrays.append(
                # We can't put None here, because vmap doesn't accept an
                # batch array of NoneType.  We also can't communicate that
                # we have non-trimmable values here by the *values* in our input
                # array. So just pass a batched 5D array that could not have arisen
                # above due to shape constraints, and communicate not to trim
                # by only the SIZE of this array.
                np.zeros((len(batched_arg_list), 1, 1, 1, 1, 1))
            )
            batch_axes.append(0)

    return padded_batched_arg_tensors, batch_axes, batched_zeros_like_arrays


# TODO: Could consider padding with an actual value from the array, but zero
# should be pretty safe.
def pad_batched_arg_list_to_max_on_each_dimension(batched_arg_list):
    assert isinstance(batched_arg_list[0], (np.ndarray, jnp.ndarray))

    # Find the maximum size along each axis
    num_dims = batched_arg_list[0].ndim
    max_size_per_dim = [0] * num_dims
    for arg in batched_arg_list:
        for dim in range(num_dims):
            if arg.shape[dim] > max_size_per_dim[dim]:
                max_size_per_dim[dim] = arg.shape[dim]

    # Pad each array with zeros to those max sizes, noop if not needed, and
    # record original sizes for later trimming
    batched_zeros_like_list = []
    padded_batched_arg_list = []
    for arg in batched_arg_list:
        batched_zeros_like_list.append(jnp.zeros_like(arg))
        padded_batched_arg_list.append(
            jnp.pad(
                arg,
                pad_width=[
                    (0, max_size_per_dim[dim] - arg.shape[dim])
                    for dim in range(num_dims)
                ],
            )
        )

    return padded_batched_arg_list, batched_zeros_like_list


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

    pi_and_weight_gradients_by_calendar_t = {}

    # This is a reliable way to get all user ids since we require all user ids
    # at all decision times
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

    num_args = action_prob_func.__code__.co_argcount
    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]
    in_study_user_ids = set()
    for user_id, user_args in sorted_user_args_dict.items():
        if not user_args:
            continue
        in_study_user_ids.add(user_id)
        for idx, arg in enumerate(user_args):
            batched_arg_lists[idx].append(arg)

    logger.info("Reforming batched data lists into tensors.")
    padded_batched_arg_tensors, batch_axes, batched_zeros_like_arrays = (
        stack_batched_arg_lists_into_tensor(batched_arg_lists)
    )

    logger.info("Forming pi gradients with respect to beta.")
    # Note that we care about the probability of action 1 specifically,
    # not the taken action.
    in_study_pi_gradients = get_pi_gradients_batched(
        action_prob_func,
        action_prob_func_args_beta_index,
        batch_axes,
        padded_batched_arg_tensors,
        batched_zeros_like_arrays,
    )
    pi_gradients = pad_in_study_derivatives_with_zeros(
        in_study_pi_gradients, sorted_user_ids, in_study_user_ids
    )

    # TODO: betas should be verified to be the same across users now or earlier
    logger.info("Forming weight gradients with respect to beta.")
    in_study_batched_actions_tensor = collect_batched_in_study_actions(
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
    in_study_weight_gradients = get_weight_gradients_batched(
        padded_batched_arg_tensors[action_prob_func_args_beta_index],
        action_prob_func,
        action_prob_func_args_beta_index,
        in_study_batched_actions_tensor,
        batch_axes,
        padded_batched_arg_tensors,
        batched_zeros_like_arrays,
    )
    weight_gradients = pad_in_study_derivatives_with_zeros(
        in_study_weight_gradients, sorted_user_ids, in_study_user_ids
    )

    return pi_gradients, weight_gradients


# TODO: is it ok to get the action from the study df? No issues with actions taken
# but not known about?
# TODO: Test this at least with an incremental recruitment collect pi gradients
# case, possibly directly.
def collect_batched_in_study_actions(
    study_df,
    calendar_t,
    sorted_user_ids,
    in_study_col_name,
    action_col_name,
    calendar_t_col_name,
    user_id_col_name,
):

    # TODO: This for loop can be removed, just grabbing the actions col after
    # filtering and sorting, and converting to jnp array.  It's just an artifact
    # from when the loop used to be more complicated.
    batched_actions_list = []
    for user_id in sorted_user_ids:
        filtered_user_data = study_df.loc[
            (study_df[user_id_col_name] == user_id)
            & (study_df[calendar_t_col_name] == calendar_t)
            & (study_df[in_study_col_name] == 1)
        ]
        if not filtered_user_data.empty:
            batched_actions_list.append(filtered_user_data[action_col_name].values[0])

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

    # TODO: How could it be that [()] after each of the prob func calls doesn't
    # make any difference here. ?? Understand. I expect these to be scalars...

    pi_beta = action_prob_func(*action_prob_func_args_single_user)
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_user)
    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


def arg_unpadding_wrapper(func, num_initial_no_pad_args, *args):
    """
    This is a little tricky but *args should be of size 2n + k, where the first
    k elements are not to be unpadded and the next n elements
    are potentially padded args. The next n are in the shapes of these args
    pre-padding.
    """
    half_num_pad_args = (len(args) - num_initial_no_pad_args) // 2
    unpadded_args = []
    for i, arg in enumerate(
        args[num_initial_no_pad_args : num_initial_no_pad_args + half_num_pad_args]
    ):
        trim_shape = args[num_initial_no_pad_args + half_num_pad_args + i].shape
        if len(trim_shape) != 5:
            slice_tuple = tuple(slice(size) for size in trim_shape)
            unpadded_args.append(arg[slice_tuple])
        else:
            unpadded_args.append(arg)
    return func(*args[:num_initial_no_pad_args], *unpadded_args)


# TODO: Docstring
def get_pi_gradients_batched(
    action_prob_func,
    action_prob_func_args_beta_index,
    batch_axes,
    padded_batched_arg_tensors,
    batched_zeros_like_arrays,
):
    # NOTE the (2 + index) is due to the fact that we have 2 fixed args in the
    # unpadding function
    return jax.vmap(
        fun=jax.grad(arg_unpadding_wrapper, 2 + action_prob_func_args_beta_index),
        in_axes=[None, None] + batch_axes * 2,
        out_axes=0,
    )(action_prob_func, 0, *padded_batched_arg_tensors, *batched_zeros_like_arrays)


# TODO: Docstring
def get_weight_gradients_batched(
    batched_beta_target_tensor,
    action_prob_func,
    action_prob_func_args_beta_index,
    batched_actions_tensor,
    batch_axes,
    padded_batched_arg_tensors,
    batched_zeros_like_arrays,
):
    # NOTE the (2 + 4 + index) is due to the fact that we have four fixed args in
    # the above definition of the weight function before passing in the action
    # prob args and 2 fixed args in the unpadding function
    return jax.vmap(
        fun=jax.grad(arg_unpadding_wrapper, 2 + 4 + action_prob_func_args_beta_index),
        in_axes=[None, None, 0, None, None, 0] + batch_axes * 2,
        out_axes=0,
    )(
        get_radon_nikodym_weight,
        4,
        batched_beta_target_tensor,
        action_prob_func,
        action_prob_func_args_beta_index,
        batched_actions_tensor,
        *padded_batched_arg_tensors,
        *batched_zeros_like_arrays,
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
        # We store these loss gradients by the first time the resulting parameters
        # apply to, so determine this time.
        # Because we perform algorithm updates at the *end* of a timestep, the
        # first timestep they apply to is one more than the time at which the
        # update data is gathered.
        first_applicable_time = get_first_applicable_time(
            study_df, policy_num, policy_num_col_name, calendar_t_col_name
        )
        loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
            calculate_rl_loss_derivatives_specific_update(
                rl_loss_func,
                rl_loss_func_args_beta_index,
                rl_loss_func_args_action_prob_index,
                user_args_dict,
                sorted_user_ids,
                first_applicable_time,
            )
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
            # NOTE the squeeze here... it is very important. Without it we have
            # a shape (x,y,z,1) array of gradients, and the use of these
            # probabilities assumes (x,y,z).  The squeezing should arguably
            # happen above, but the vmap call spits out a 4D array so in that
            # sense that's the most natural return value.
            # TODO: This probably has to do with the dimension of the action
            # probabilities... we may need to specify that they are scalars in the
            # loss function args, rather than 1-element vectors. Or one will
            # have to say so.  Test both of these cases.  Can probably check
            # dimensions and squeeze if necessary.
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
    first_applicable_time,
):
    # Sort users to be cautious
    sorted_user_args_dict = {
        user_id: user_args_dict[user_id] for user_id in sorted_user_ids
    }

    num_args = rl_loss_func.__code__.co_argcount
    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]
    in_study_user_ids = set()
    for user_id, user_args in sorted_user_args_dict.items():
        if not user_args:
            continue
        in_study_user_ids.add(user_id)
        for idx, arg in enumerate(user_args):
            batched_arg_lists[idx].append(arg)

    # Note this stacking works with incremental recruitment only because we
    # fill in states for out-of-study times such that all users have the
    # same state matrix size
    # TODO: Articulate requirement that each arg can be tensorized into a numpy array
    # because of type
    logger.info("Reforming batched data lists into tensors.")
    padded_batched_arg_tensors, batch_axes, batched_trim_shape_arrays = (
        stack_batched_arg_lists_into_tensor(batched_arg_lists)
    )

    logger.info("Forming loss gradients with respect to beta.")
    in_study_loss_gradients = get_loss_gradients_batched(
        rl_loss_func,
        rl_loss_func_args_beta_index,
        batch_axes,
        *padded_batched_arg_tensors,
    )
    loss_gradients = pad_in_study_derivatives_with_zeros(
        in_study_loss_gradients, sorted_user_ids, in_study_user_ids
    )

    logger.info("Forming loss hessians with respect to beta")
    in_study_loss_hessians = get_loss_hessians_batched(
        rl_loss_func,
        rl_loss_func_args_beta_index,
        batch_axes,
        *padded_batched_arg_tensors,
    )
    loss_hessians = pad_in_study_derivatives_with_zeros(
        in_study_loss_hessians, sorted_user_ids, in_study_user_ids
    )
    logger.info(
        "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time"
    )
    # If there is NOT an action probability argument in the loss, we need to
    # simply return zero gradients of the correct shape.
    if rl_loss_func_args_action_prob_index < 0:
        num_users = len(sorted_user_ids)
        beta_dim = batched_arg_lists[rl_loss_func_args_beta_index][0].size
        timesteps_included = first_applicable_time - 1

        loss_gradient_pi_derivatives = np.zeros(
            (num_users, beta_dim, timesteps_included, 1)
        )
    # Otherwise, actually differentiate with respect to action probabilities.
    else:
        in_study_loss_gradient_pi_derivatives = (
            get_loss_gradient_derivatives_wrt_pi_batched(
                rl_loss_func,
                rl_loss_func_args_beta_index,
                rl_loss_func_args_action_prob_index,
                batch_axes,
                *padded_batched_arg_tensors,
            )
        )
        loss_gradient_pi_derivatives = pad_in_study_derivatives_with_zeros(
            in_study_loss_gradient_pi_derivatives,
            sorted_user_ids,
            in_study_user_ids,
        )

    return loss_gradients, loss_hessians, loss_gradient_pi_derivatives


def pad_in_study_derivatives_with_zeros(
    in_study_derivatives, sorted_user_ids, in_study_user_ids
):
    """
    This fills in zero gradients for users not currently in the study given the
    derivatives computed for those in it.
    """
    all_derivatives = []
    in_study_next_idx = 0
    for user_id in sorted_user_ids:
        if user_id in in_study_user_ids:
            all_derivatives.append(in_study_derivatives[in_study_next_idx])
            in_study_next_idx += 1
        else:
            all_derivatives.append(np.zeros_like(in_study_derivatives[0]))

    return all_derivatives


def get_loss_gradients_batched(
    loss_func, loss_func_args_beta_index, batch_axes, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.grad(loss_func, loss_func_args_beta_index),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


def get_loss_hessians_batched(
    loss_func, loss_func_args_beta_index, batch_axes, *batched_arg_tensors
):
    return jax.vmap(
        fun=jax.hessian(loss_func, loss_func_args_beta_index),
        in_axes=batch_axes,
        out_axes=0,
    )(*batched_arg_tensors)


def get_loss_gradient_derivatives_wrt_pi_batched(
    loss_func,
    loss_func_args_beta_index,
    loss_func_args_action_prob_index,
    batch_axes,
    *batched_arg_tensors,
):
    return jax.vmap(
        fun=jax.jacrev(
            jax.grad(loss_func, loss_func_args_beta_index),
            loss_func_args_action_prob_index,
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


def calculate_inference_loss_derivatives(
    study_df,
    theta_est,
    inference_loss_func_filename,
    inference_loss_func_args_theta_index,
    user_ids,
    user_id_col_name,
    action_prob_col_name,
    in_study_col_name,
    calendar_t_col_name,
):
    # Retrieve the inference loss function from file
    inference_loss_module = load_module_from_source_file(
        "inference_loss", inference_loss_func_filename
    )
    # NOTE the assumption that the function and file have the same name
    inference_loss_func_name = os.path.basename(inference_loss_func_filename).split(
        "."
    )[0]
    try:
        inference_loss_func = getattr(inference_loss_module, inference_loss_func_name)
    except AttributeError as e:
        raise ValueError(
            "Unable to import RL loss function.  Please verify the file has the same name as the function of interest."
        ) from e

    num_args = inference_loss_func.__code__.co_argcount
    inference_loss_func_arg_names = inference_loss_func.__code__.co_varnames[:num_args]
    num_args = len(inference_loss_func_arg_names)
    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]

    for user_id in user_ids:
        filtered_user_data = study_df.loc[study_df[user_id_col_name] == user_id]
        for idx, col_name in enumerate(inference_loss_func_arg_names):
            if idx == inference_loss_func_args_theta_index:
                batched_arg_lists[idx].append(theta_est)
            else:
                batched_arg_lists[idx].append(
                    get_study_df_column(filtered_user_data, col_name, in_study_col_name)
                )

    # Note this stacking works with incremental recruitment only because we
    # fill in states for out-of-study times such that all users have the
    # same state matrix size
    logger.info("Reforming batched data lists into tensors.")
    batched_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensor(
        batched_arg_lists
    )

    logger.info("Forming loss gradients with respect to beta.")
    loss_gradients = get_loss_gradients_batched(
        inference_loss_func,
        inference_loss_func_args_theta_index,
        batch_axes,
        *batched_arg_tensors,
    )

    logger.info("Forming loss hessians with respect to beta")
    loss_hessians = get_loss_hessians_batched(
        inference_loss_func,
        inference_loss_func_args_theta_index,
        batch_axes,
        *batched_arg_tensors,
    )
    logger.info(
        "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time"
    )
    # If there is NOT an action probability argument in the loss, we need to
    # simply return zero gradients of the correct shape.
    if action_prob_col_name in inference_loss_func_arg_names:
        inference_loss_func_args_action_prob_index = (
            inference_loss_func_arg_names.index(action_prob_col_name)
        )
        loss_gradient_pi_derivatives = get_loss_gradient_derivatives_wrt_pi_batched(
            inference_loss_func,
            inference_loss_func_args_theta_index,
            inference_loss_func_args_action_prob_index,
            batch_axes,
            *batched_arg_tensors,
        )
    # Otherwise, actually differentiate with respect to action probabilities.
    else:
        num_users = len(user_ids)
        theta_dim = theta_est.size
        timesteps_included = study_df[calendar_t_col_name].nunique()

        loss_gradient_pi_derivatives = np.zeros(
            (num_users, theta_dim, timesteps_included, 1)
        )

    return loss_gradients, loss_hessians, loss_gradient_pi_derivatives


def get_study_df_column(study_df, col_name, in_study_col_name):
    # TODO: This assumes we can simply take 0 for out-of-study values. This is not
    # appropriate in general, and needs to be fixed.
    # See https://nowell-closser.atlassian.net/browse/ADS-28
    study_df.loc[study_df[in_study_col_name] == 0, col_name] = 0
    return jnp.array(study_df[col_name].to_numpy().reshape(-1, 1))
