import collections
import logging

import jax
from jax import numpy as jnp
import numpy as np

from .constants import FunctionTypes
from .helper_functions import (
    conditional_x_or_one_minus_x,
    load_function_from_same_named_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# TODO: Consolidate function loading logic


def get_batched_arg_lists_and_involved_user_ids(func, sorted_user_ids, args_by_user_id):
    """
    Collect a dict of arg tuples by user id into a list of lists, each containing
    all the args at a particular index across users. We make sure the list is
    ordered according to sorted_user_ids.
    """
    # Sort users to be cautious. We check if the user id is present in the user args dict
    # because we may call this on a subset of the user arg dict when we are batching
    # arguments by shape
    sorted_args_by_user_id = {
        user_id: args_by_user_id[user_id]
        for user_id in sorted_user_ids
        if user_id in args_by_user_id
    }

    # Just a quick way to get the arg count instead of iterating through args
    # for the first Truthy tuple
    # TODO: If there are arguments with defaults and not supplied, this will break.
    # Should probably in fact iterate through to first Truthy tuple.
    num_args = func.__code__.co_argcount

    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]
    involved_user_ids = set()
    for user_id, user_args in sorted_args_by_user_id.items():
        if not user_args:
            continue
        involved_user_ids.add(user_id)
        for idx, arg in enumerate(user_args):
            batched_arg_lists[idx].append(arg)

    return batched_arg_lists, involved_user_ids


def get_shape(obj):
    if hasattr(obj, "shape"):
        return obj.shape
    if isinstance(obj, str):
        return None
    try:
        return len(obj)
    except Exception:
        return None


def group_user_args_by_shape(user_arg_dict, empty_allowed=True):
    user_arg_dicts_by_shape = collections.defaultdict(dict)
    for user_id, args in user_arg_dict.items():
        if not args:
            if not empty_allowed:
                raise ValueError("There shouldn't be a user with no data at this time")
            continue
        shape_id = tuple(get_shape(arg) for arg in args)
        user_arg_dicts_by_shape[shape_id][user_id] = args
    return user_arg_dicts_by_shape.values()


# TODO: Check for exactly the required types earlier
# TODO: Try except and nice error message
# TODO: This is complicated enough to deserve its own unit tests
def stack_batched_arg_lists_into_tensor(batched_arg_lists):
    """
    Stack a simple Python list of lists of function arguments (across all users for a specific arg position)
    into a list of jnp arrays that can be supplied to vmap as batch arguments. vmap requires all elements of
    such a batched array to be the same shape, as do the stacking functions we use here.  Thus we require
    this be called on batches of users with the same data shape. We also supply the axes one must
    iterate over to get each users's args in a batch.
    """

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

            batched_arg_tensors.append(jnp.stack(batched_arg_list, 0))
            batch_axes.append(0)
        elif isinstance(
            batched_arg_list[0],
            (collections.abc.Sequence, jnp.ndarray, np.ndarray),
        ) and not isinstance(batched_arg_list[0], str):
            ########## We have a vector (1D array) arg
            if not isinstance(batched_arg_list[0], (jnp.ndarray, np.ndarray)):
                try:
                    batched_arg_list = [jnp.array(x) for x in batched_arg_list]
                except Exception as e:
                    raise TypeError(
                        "Argument of sequence type that cannot be cast to JAX numpy array"
                    ) from e
            assert batched_arg_list[0].ndim == 1

            batched_arg_tensors.append(jnp.vstack(batched_arg_list))
            batch_axes.append(0)
        else:
            ########## Otherwise we should have a list of scalars.
            # Just turn into a jnp array.
            batched_arg_tensors.append(jnp.array(batched_arg_list))
            batch_axes.append(0)

    return (
        batched_arg_tensors,
        batch_axes,
    )


# TODO: Add clarity on why we need gradients at all times.
def pad_loss_gradient_pi_derivative_outside_supplied_action_probabilites(
    loss_gradient_pi_derivative,
    action_prob_times,
    first_time_after,
):
    """
    This fills in zero gradients for the times for which action probabilites
    were not supplied, for users currently in the study.  Compare to the below
    padding which is about filling in full sets of zero gradients for all users
    not currently in the study. This is about filling in zero gradients for
    times 1,2,3,4,9,10,11,12 if action probabilities are given for times 5,6,7,
    8.
    """
    zero_gradient = np.zeros((loss_gradient_pi_derivative.shape[0], 1, 1))
    gradients_to_stack = []
    next_column_index_to_grab = 0
    for t in range(1, first_time_after):
        if t in action_prob_times:
            gradients_to_stack.append(
                np.expand_dims(
                    loss_gradient_pi_derivative[:, next_column_index_to_grab, :], 2
                )
            )
            next_column_index_to_grab += 1
        else:
            gradients_to_stack.append(zero_gradient)

    return np.hstack(gradients_to_stack)


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


def calculate_pi_and_weight_gradients(
    study_df,
    in_study_col_name,
    action_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_func,
    action_prob_func_args,
    action_prob_func_args_beta_index,
):
    """
    For all decision times, for all users, compute the gradient with respect to
    beta of both the pi function (which takes a state and gives the probability
    of selecting action 1) and the Radon-Nikodym weight (derived from pi
    functions as described in the paper)
    """

    logger.debug("Calculating pi and weight gradients with respect to beta.")

    pi_and_weight_gradients_by_calendar_t = {}

    # This is a reliable way to get all user ids since we require all user ids
    # at all decision times
    user_ids = list(next(iter(action_prob_func_args.values())).keys())
    sorted_user_ids = sorted(user_ids)

    for calendar_t, args_by_user_id in action_prob_func_args.items():

        pi_gradients, weight_gradients = calculate_pi_and_weight_gradients_specific_t(
            study_df,
            in_study_col_name,
            action_col_name,
            calendar_t_col_name,
            user_id_col_name,
            action_prob_func,
            action_prob_func_args_beta_index,
            calendar_t,
            args_by_user_id,
            sorted_user_ids,
        )

        logger.debug("Collecting pi gradients into algorithm stats dictionary.")
        pi_and_weight_gradients_by_calendar_t.setdefault(calendar_t, {})[
            "pi_gradients_by_user_id"
        ] = {user_id: pi_gradients[i] for i, user_id in enumerate(sorted_user_ids)}

        logger.debug("Collecting weight gradients into algorithm stats dictionary.")
        pi_and_weight_gradients_by_calendar_t.setdefault(calendar_t, {})[
            "weight_gradients_by_user_id"
        ] = {user_id: weight_gradients[i] for i, user_id in enumerate(sorted_user_ids)}

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
    args_by_user_id,
    sorted_user_ids,
):
    logger.debug(
        "Calculating pi and weight gradients for decision time %d.",
        calendar_t,
    )
    # Get a list of subdicts of the user args dict, with each united by having
    # the same shapes across all arguments.  We will then vmap the gradients needed
    # for each subdict separately, and later combine the results.  In the worst
    # case we may have a batch per user, if, say, everyone starts on a different
    # date, and this will be slow. If this is problematic we can pad the data
    # with some values that don't affect computations, producing one batch here.
    # This also supports very large simulations by making things fast as long
    # as there is 1 or a small number of shape batches.
    nontrivial_user_args_grouped_by_shape = group_user_args_by_shape(args_by_user_id)
    logger.debug(
        "Found %d set(s) of users with different arg shapes.",
        len(nontrivial_user_args_grouped_by_shape),
    )

    # Loop over each set of user args and vmap to get their pi and weight gradients
    in_study_pi_gradients_by_user_id = {}
    in_study_weight_gradients_by_user_id = {}
    all_involved_user_ids = set()
    for args_by_user_id_subset in nontrivial_user_args_grouped_by_shape:
        # Now that we are grouping by arg shape and excluding the out of study
        # group, all the users should be involved in the study in this loop,
        # but... just keep this logic that works for heterogeneous-shaped
        # batches too.
        batched_arg_lists, involved_user_ids = (
            get_batched_arg_lists_and_involved_user_ids(
                action_prob_func, sorted_user_ids, args_by_user_id_subset
            )
        )
        all_involved_user_ids |= involved_user_ids

        if not batched_arg_lists[0]:
            continue

        logger.debug("Reforming batched data lists into tensors.")
        batched_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensor(
            batched_arg_lists
        )

        logger.debug("Forming pi gradients with respect to beta.")
        # Note that we care about the probability of action 1 specifically,
        # not the taken action.
        in_study_pi_gradients_subset = get_pi_gradients_batched(
            action_prob_func,
            action_prob_func_args_beta_index,
            batch_axes,
            batched_arg_tensors,
        )

        # TODO: betas should be verified to be the same across users now or earlier
        logger.debug("Forming weight gradients with respect to beta.")
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
        in_study_weight_gradients_subset = get_weight_gradients_batched(
            batched_arg_tensors[action_prob_func_args_beta_index],
            action_prob_func,
            action_prob_func_args_beta_index,
            in_study_batched_actions_tensor,
            batch_axes,
            batched_arg_tensors,
        )

        # Collect the gradients for the in-study users in this group into the
        # overall dict.
        in_batch_index = 0
        for user_id in sorted_user_ids:
            if user_id not in involved_user_ids:
                continue
            in_study_pi_gradients_by_user_id[user_id] = in_study_pi_gradients_subset[
                in_batch_index
            ]
            in_study_weight_gradients_by_user_id[user_id] = (
                in_study_weight_gradients_subset[in_batch_index]
            )
            in_batch_index += 1

    in_study_pi_gradients = [
        in_study_pi_gradients_by_user_id[user_id]
        for user_id in sorted_user_ids
        if user_id in all_involved_user_ids
    ]
    in_study_weight_gradients = [
        in_study_weight_gradients_by_user_id[user_id]
        for user_id in sorted_user_ids
        if user_id in all_involved_user_ids
    ]
    # TODO: These padding methods assume someone was in the study at this time.
    pi_gradients = pad_in_study_derivatives_with_zeros(
        in_study_pi_gradients, sorted_user_ids, all_involved_user_ids
    )
    weight_gradients = pad_in_study_derivatives_with_zeros(
        in_study_weight_gradients, sorted_user_ids, all_involved_user_ids
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


# TODO: Docstring
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

    pi_beta = action_prob_func(*action_prob_func_args_single_user)
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_user)
    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


# TODO: Docstring
def get_pi_gradients_batched(
    action_prob_func,
    action_prob_func_args_beta_index,
    batch_axes,
    batched_arg_tensors,
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
    batched_arg_tensors,
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
# TODO: This is a hotspot for moving away from update times
def calculate_rl_update_derivatives(
    study_df,
    rl_update_func_filename,
    rl_update_func_args,
    rl_update_func_type,
    rl_update_func_args_beta_index,
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
    policy_num_col_name,
    calendar_t_col_name,
):
    logger.debug(
        "Calculating RL loss gradients and hessians with respect to beta and mixed beta/action probability derivatives for each user at all update times."
    )
    rl_update_func = load_function_from_same_named_file(rl_update_func_filename)

    rl_update_derivatives_by_calendar_t = {}
    user_ids = list(next(iter(rl_update_func_args.values())).keys())
    sorted_user_ids = sorted(user_ids)
    for policy_num, args_by_user_id in rl_update_func_args.items():
        # We store these loss gradients by the first time the resulting parameters
        # apply to, so determine this time.
        # Because we perform algorithm updates at the *end* of a timestep, the
        # first timestep they apply to is one more than the time at which the
        # update data is gathered.
        first_applicable_time = get_first_applicable_time(
            study_df, policy_num, policy_num_col_name, calendar_t_col_name
        )
        loss_gradients, loss_hessians, loss_gradient_pi_derivatives = (
            calculate_rl_update_derivatives_specific_update(
                rl_update_func,
                rl_update_func_type,
                rl_update_func_args_beta_index,
                rl_update_func_args_action_prob_index,
                rl_update_func_args_action_prob_times_index,
                args_by_user_id,
                sorted_user_ids,
                first_applicable_time,
            )
        )
        rl_update_derivatives_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {user_id: loss_gradients[i] for i, user_id in enumerate(sorted_user_ids)}
        rl_update_derivatives_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = np.mean(loss_hessians, axis=0)

        rl_update_derivatives_by_calendar_t[first_applicable_time][
            "loss_gradient_pi_derivatives_by_user_id"
        ] = {
            # NOTE the [..., 0] here... it is very important. Without it we have
            # a shape (x,y,z,1) array of gradients, and the use of these
            # probabilities assumes (x,y,z).  This should arguably
            # happen above, but the vmap call spits out a 4D array so in that
            # sense that's the most natural return value. Note that we don't
            # simply squeeze because that would remove the beta dimension
            # if it were one.
            # TODO: This probably has to do with the dimension of the action
            # probabilities... we may need to specify that they are scalars in the
            # loss function args, rather than 1-element vectors. Or one will
            # have to say so.  Test both of these cases.  Can probably check
            # dimensions and squeeze if necessary.
            user_id: loss_gradient_pi_derivatives[i][..., 0]
            for i, user_id in enumerate(sorted_user_ids)
        }
    return rl_update_derivatives_by_calendar_t


def calculate_rl_update_derivatives_specific_update(
    rl_update_func,
    rl_update_func_type,
    rl_update_func_args_beta_index,
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
    args_by_user_id,
    sorted_user_ids,
    first_applicable_time,
):
    logger.debug(
        "Calculating RL update derivatives for the update that first applies at time %d.",
        first_applicable_time,
    )
    # Get a list of subdicts of the user args dict, with each united by having
    # the same shapes across all arguments.  We will then vmap the gradients needed
    # for each subdict separately, and later combine the results.  In the worst
    # case we may have a batch per user, if, say, everyone starts on a different
    # date, and this will be slow. If this is problematic we can pad the data
    # with some values that don't affect computations, producing one batch here.
    # This also supports very large simulations by making things fast as long
    # as there is 1 or a small number of shape batches.
    # NOTE: Susan and Kelly think we might actually have uniqueish shapes pretty often
    nontrivial_user_args_grouped_by_shape = group_user_args_by_shape(args_by_user_id)
    logger.debug(
        "Found %d set(s) of users with different arg shapes.",
        len(nontrivial_user_args_grouped_by_shape),
    )

    # Loop over each set of user args and vmap to get their pi and weight gradients
    in_study_loss_gradients_by_user_id = {}
    in_study_loss_hessians_by_user_id = {}
    in_study_loss_gradient_pi_derivatives_by_user_id = {}
    all_involved_user_ids = set()
    for args_by_user_id_subset in nontrivial_user_args_grouped_by_shape:
        # Pivot the loss args for the involved users into a list of lists, each
        # representing all the args at a particular index across users. Note
        # that users not in the study at this time are filtered out by this
        # function when it checks for truthiness of the supplied args.
        batched_arg_lists, involved_user_ids = (
            get_batched_arg_lists_and_involved_user_ids(
                rl_update_func, sorted_user_ids, args_by_user_id_subset
            )
        )
        all_involved_user_ids |= involved_user_ids

        if not batched_arg_lists[0]:
            continue

        logger.debug("Reforming batched data lists into tensors.")
        # Now just transform the previous list of lists into a jnp array for each
        # index (a tensor for each argument).  This is for passing to vmap.
        batched_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensor(
            batched_arg_lists
        )

        logger.debug("Forming loss gradients with respect to beta.")
        in_study_loss_gradients_subset = get_loss_gradients_batched(
            rl_update_func,
            rl_update_func_type,
            rl_update_func_args_beta_index,
            batch_axes,
            *batched_arg_tensors,
        )

        logger.debug("Forming loss hessians with respect to beta.")
        in_study_loss_hessians_subset = get_loss_hessians_batched(
            rl_update_func,
            rl_update_func_type,
            rl_update_func_args_beta_index,
            batch_axes,
            *batched_arg_tensors,
        )
        logger.debug(
            "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time."
        )
        if rl_update_func_args_action_prob_index >= 0:
            in_study_loss_gradient_pi_derivatives_subset = (
                get_loss_gradient_derivatives_wrt_pi_batched(
                    rl_update_func,
                    rl_update_func_type,
                    rl_update_func_args_beta_index,
                    rl_update_func_args_action_prob_index,
                    batch_axes,
                    *batched_arg_tensors,
                )
            )
        # Collect the gradients for the in-study users in this group into the
        # overall dict.
        in_batch_index = 0
        for user_id in sorted_user_ids:
            if user_id not in involved_user_ids:
                continue
            in_study_loss_gradients_by_user_id[user_id] = (
                in_study_loss_gradients_subset[in_batch_index]
            )
            in_study_loss_hessians_by_user_id[user_id] = in_study_loss_hessians_subset[
                in_batch_index
            ]
            if rl_update_func_args_action_prob_index >= 0:
                in_study_loss_gradient_pi_derivatives_by_user_id[user_id] = (
                    pad_loss_gradient_pi_derivative_outside_supplied_action_probabilites(
                        in_study_loss_gradient_pi_derivatives_subset[in_batch_index],
                        args_by_user_id[user_id][
                            rl_update_func_args_action_prob_times_index
                        ],
                        first_applicable_time,
                    )
                )
            in_batch_index += 1
    in_study_loss_gradients = [
        in_study_loss_gradients_by_user_id[user_id]
        for user_id in sorted_user_ids
        if user_id in all_involved_user_ids
    ]
    in_study_loss_hessians = [
        in_study_loss_hessians_by_user_id[user_id]
        for user_id in sorted_user_ids
        if user_id in all_involved_user_ids
    ]
    if rl_update_func_args_action_prob_index >= 0:
        in_study_loss_gradient_pi_derivatives = [
            in_study_loss_gradient_pi_derivatives_by_user_id[user_id]
            for user_id in sorted_user_ids
            if user_id in all_involved_user_ids
        ]
    # TODO: These padding methods assume *someone* had study data at this time.
    loss_gradients = pad_in_study_derivatives_with_zeros(
        in_study_loss_gradients, sorted_user_ids, all_involved_user_ids
    )
    loss_hessians = pad_in_study_derivatives_with_zeros(
        in_study_loss_hessians, sorted_user_ids, all_involved_user_ids
    )

    # If there is an action probability argument in the RL update function, we need to
    # pad the derivatives calculated already with zeros for those users not currently
    # in the study. Otherwise simply return all zero gradients of the correct shape.
    if rl_update_func_args_action_prob_index >= 0:
        loss_gradient_pi_derivatives = pad_in_study_derivatives_with_zeros(
            in_study_loss_gradient_pi_derivatives,
            sorted_user_ids,
            all_involved_user_ids,
        )
    else:
        num_users = len(sorted_user_ids)
        beta_dim = batched_arg_lists[rl_update_func_args_beta_index][0].size
        timesteps_included = first_applicable_time - 1

        loss_gradient_pi_derivatives = np.zeros(
            (num_users, beta_dim, timesteps_included, 1)
        )

    return loss_gradients, loss_hessians, loss_gradient_pi_derivatives


def get_loss_gradients_batched(
    update_func,
    update_func_type,
    update_func_args_beta_index,
    batch_axes,
    *batched_arg_tensors,
):
    if update_func_type == FunctionTypes.LOSS:
        return jax.vmap(
            fun=jax.grad(update_func, update_func_args_beta_index),
            in_axes=batch_axes,
            out_axes=0,
        )(*batched_arg_tensors)
    if update_func_type == FunctionTypes.ESTIMATING:
        return jax.vmap(
            fun=update_func,
            in_axes=batch_axes,
            out_axes=0,
        )(*batched_arg_tensors)
    raise ValueError("Unknown update function type.")


def get_loss_hessians_batched(
    update_func,
    update_func_type,
    update_func_args_beta_index,
    batch_axes,
    *batched_arg_tensors,
):
    if update_func_type == FunctionTypes.LOSS:
        return jax.vmap(
            fun=jax.hessian(update_func, update_func_args_beta_index),
            in_axes=batch_axes,
            out_axes=0,
        )(*batched_arg_tensors)
    if update_func_type == FunctionTypes.ESTIMATING:
        return jax.vmap(
            fun=jax.jacrev(update_func, update_func_args_beta_index),
            in_axes=batch_axes,
            out_axes=0,
        )(*batched_arg_tensors)
    raise ValueError("Unknown update function type.")


def get_loss_gradient_derivatives_wrt_pi_batched(
    update_func,
    update_func_type,
    update_func_args_beta_index,
    update_func_args_action_prob_index,
    batch_axes,
    *batched_arg_tensors,
):
    if update_func_type == FunctionTypes.LOSS:
        return jax.jit(  # pylint: disable=not-callable
            jax.vmap(
                fun=jax.jacrev(
                    jax.grad(update_func, update_func_args_beta_index),
                    update_func_args_action_prob_index,
                ),
                in_axes=batch_axes,
                out_axes=0,
            )
        )(*batched_arg_tensors)
    if update_func_type == FunctionTypes.ESTIMATING:
        return jax.jit(  # pylint: disable=not-callable
            jax.vmap(
                fun=jax.jacrev(
                    update_func,
                    update_func_args_action_prob_index,
                ),
                in_axes=batch_axes,
                out_axes=0,
            )
        )(*batched_arg_tensors)
    raise ValueError("Unknown update function type.")


# TODO: Is there a better way to calculate this? This seems like it should
# be reliable, not messing up when a policy was actually available. If study
# df says policy was used, that should be correct.  May not play nicely with
# pure exploration phase though.
def get_first_applicable_time(
    study_df, policy_num, policy_num_col_name, calendar_t_col_name
):
    return study_df[study_df[policy_num_col_name] == policy_num][
        calendar_t_col_name
    ].min()


def calculate_inference_loss_derivatives(
    study_df,
    theta_est,
    inference_func,
    inference_func_args_theta_index,
    user_ids,
    user_id_col_name,
    action_prob_col_name,
    in_study_col_name,
    calendar_t_col_name,
    inference_func_type=FunctionTypes.LOSS,
):
    logger.debug("Calculating inference loss derivatives.")

    # Convert to list if needed (from jnp array, etc)
    try:
        user_ids = user_ids.tolist()
    except Exception:
        pass

    num_args = inference_func.__code__.co_argcount
    inference_func_arg_names = inference_func.__code__.co_varnames[:num_args]
    # NOTE: Cannot do [[]] * num_args here! Then all lists point
    # same object...
    batched_arg_lists = [[] for _ in range(num_args)]

    # We begin by constructing a dict of loss function arg tuples of the type we get from file
    # for the RL data; because we have to group user args by shape anyway, we
    # might as well collect them in this format and then use previous machinery
    # to process them. There are a few extra loops but more shared code this way.
    args_by_user_id = {}
    using_action_probs = action_prob_col_name in inference_func_arg_names
    if using_action_probs:
        inference_func_args_action_prob_index = inference_func_arg_names.index(
            action_prob_col_name
        )
        action_prob_decision_times_by_user_id = {}
        max_calendar_time = study_df[calendar_t_col_name].max()
    for user_id in user_ids:
        user_args_list = []
        filtered_user_data = study_df.loc[study_df[user_id_col_name] == user_id]
        for idx, col_name in enumerate(inference_func_arg_names):
            if idx == inference_func_args_theta_index:
                user_args_list.append(theta_est)
            else:
                user_args_list.append(
                    get_study_df_column(filtered_user_data, col_name, in_study_col_name)
                )
        args_by_user_id[user_id] = tuple(user_args_list)
        if using_action_probs:
            action_prob_decision_times_by_user_id[user_id] = get_study_df_column(
                filtered_user_data, calendar_t_col_name, in_study_col_name
            )

    # Get a list of subdicts of the user args dict, with each united by having
    # the same shapes across all arguments.  We will then vmap the gradients needed
    # for each subdict separately, and later combine the results.  In the worst
    # case we may have a batch per user, if, say, everyone starts on a different
    # date, and this will be slow. If this is problematic we can pad the data
    # with some values that don't affect computations, producing one batch here.
    # This also supports very large simulations by making things fast as long
    # as there is 1 or a small number of shape batches.
    # NOTE: As opposed to the RL updates, we should expect a small number of
    # batches here. It is only users having different numbers of decision times
    # that contributes additional batches.
    nontrivial_user_args_grouped_by_shape = group_user_args_by_shape(
        args_by_user_id, empty_allowed=False
    )
    logger.debug(
        "Found %d set(s) of users with different arg shapes.",
        len(nontrivial_user_args_grouped_by_shape),
    )

    loss_gradients_by_user_id = {}
    loss_hessians_by_user_id = {}
    loss_gradient_pi_derivatives_by_user_id = {}
    all_involved_user_ids = set()
    sorted_user_ids = sorted(user_ids)
    for args_by_user_id_subset in nontrivial_user_args_grouped_by_shape:
        batched_arg_lists, involved_user_ids = (
            get_batched_arg_lists_and_involved_user_ids(
                inference_func, sorted_user_ids, args_by_user_id_subset
            )
        )
        all_involved_user_ids |= involved_user_ids

        logger.debug("Reforming batched data lists into tensors.")
        batched_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensor(
            batched_arg_lists
        )

        logger.debug("Forming loss gradients with respect to theta.")
        loss_gradients_subset = get_loss_gradients_batched(
            inference_func,
            inference_func_type,
            inference_func_args_theta_index,
            batch_axes,
            *batched_arg_tensors,
        )

        logger.debug("Forming loss hessians with respect to theta.")
        loss_hessians_subset = get_loss_hessians_batched(
            inference_func,
            inference_func_type,
            inference_func_args_theta_index,
            batch_axes,
            *batched_arg_tensors,
        )
        logger.debug(
            "Forming derivatives of loss with respect to theta and then the action probabilities vector at each time."
        )
        # If there is an action probability argument in the loss,
        # actually differentiate with respect to action probabilities
        if using_action_probs:
            loss_gradient_pi_derivatives_subset = (
                get_loss_gradient_derivatives_wrt_pi_batched(
                    inference_func,
                    inference_func_type,
                    inference_func_args_theta_index,
                    inference_func_args_action_prob_index,
                    batch_axes,
                    *batched_arg_tensors,
                )
            )
        # Collect the gradients for the in-study users in this group into the
        # overall dict.
        in_batch_index = 0
        for user_id in sorted_user_ids:
            if user_id not in involved_user_ids:
                continue
            loss_gradients_by_user_id[user_id] = loss_gradients_subset[in_batch_index]
            loss_hessians_by_user_id[user_id] = loss_hessians_subset[in_batch_index]
            if using_action_probs:
                loss_gradient_pi_derivatives_by_user_id[user_id] = (
                    pad_loss_gradient_pi_derivative_outside_supplied_action_probabilites(
                        loss_gradient_pi_derivatives_subset[in_batch_index],
                        action_prob_decision_times_by_user_id[user_id],
                        max_calendar_time + 1,
                    )
                )
            in_batch_index += 1
    loss_gradients = np.array(
        [
            loss_gradients_by_user_id[user_id]
            for user_id in sorted_user_ids
            if user_id in all_involved_user_ids
        ]
    )
    loss_hessians = np.array(
        [
            loss_hessians_by_user_id[user_id]
            for user_id in sorted_user_ids
            if user_id in all_involved_user_ids
        ]
    )
    # If using action probs, collect the mixed theta pi derivatives computed
    # so far.
    if using_action_probs:
        loss_gradient_pi_derivatives = np.array(
            [
                loss_gradient_pi_derivatives_by_user_id[user_id]
                for user_id in sorted_user_ids
                if user_id in all_involved_user_ids
            ]
        )
    # Otherwise, we need to simply return zero gradients of the correct shape.
    else:
        num_users = len(user_ids)
        theta_dim = theta_est.size
        timesteps_included = study_df[calendar_t_col_name].nunique()

        loss_gradient_pi_derivatives = np.zeros(
            (num_users, theta_dim, timesteps_included, 1)
        )

    return loss_gradients, loss_hessians, loss_gradient_pi_derivatives


def get_study_df_column(study_df, col_name, in_study_col_name):
    return jnp.array(
        study_df.loc[study_df[in_study_col_name] == 1, col_name]
        .to_numpy()
        .reshape(-1, 1)
    )
