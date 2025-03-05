import collections
import pickle
import os
import logging
import pathlib
import glob
import math
from typing import Any

import click
import jax
import numpy as np
from jax import numpy as jnp
import scipy
import pandas

import calculate_derivatives
from constants import FunctionTypes, SmallSampleCorrections
import input_checks


from helper_functions import (
    conditional_x_or_one_minus_x,
    get_in_study_df_column,
    invert_matrix_and_check_conditioning,
    load_function_from_same_named_file,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# TODO: Break this file up


@click.group()
def cli():
    pass


# TODO: Add option to give per-user loss OR estimating function. Just loss now
# TODO: take in theta instead of forming it, and use to check estimating function.
#       Yet we still want to support large simulation case where we DO calculate theta.
#       I think you have to pass in theta-forming function OR theta itself.
# TODO: Take in requirements files for action prob and loss and take derivatives
# in corresponding sandbox. For now we just assume the dependencies in this package
# suffice.
# TODO: Handle raw timestamps instead of calendar time index? For now I'm requiring it.
# More generally, handle decision times being different across different users? Would like
# to consolidate.
# TODO: Check all help strings for accuracy.
# TODO: Don't use theta and beta jargon?? Need a legend if I do.
# TODO: Make run scripts that hardcode to action centering or not on both RL and inference sides
# TODO: Need to support pure exploration phase with more flags than just in study. Maybe in study, receiving updates
# TODO: Deal with NA, -1, etc policy numbers
@cli.command()
@click.option(
    "--study_df_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme)",
    required=True,
)
@click.option(
    "--action_prob_func_filename",
    type=click.Path(exists=True),
    help="File that contains the action probability function and relevant imports.  The filename without its extension will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--action_prob_func_args_pickle",
    type=click.File("rb"),
    help="Pickled dictionary that contains the action probability function arguments for all decision times for all users",
    required=True,
)
@click.option(
    "--action_prob_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the algorithm parameter vector beta in the tuple of action probability func args",
)
@click.option(
    "--alg_update_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user update function used to determine the algorithm parameters at each update and relevant imports.  The filename without its extension will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--alg_update_func_type",
    type=click.Choice([FunctionTypes.LOSS, FunctionTypes.ESTIMATING]),
    help="Type of function used to summarize the algorithm updates.  If loss, an update should correspond to choosing parameters to minimize it.  If estimating, an update should correspond to setting the function equal to zero and solving for the parameters.",
    required=True,
)
@click.option(
    "--alg_update_func_args_pickle",
    type=click.File("rb"),
    help="Pickled dictionary that contains the algorithm update function arguments for all update times for all users",
    required=True,
)
@click.option(
    "--alg_update_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the algorithm parameter vector beta in the tuple of algorithm update func args",
)
@click.option(
    "--alg_update_func_args_action_prob_index",
    type=int,
    default=-1000,
    help="Index of the action probability in the tuple of algorithm update func args, if applicable",
)
@click.option(
    "--alg_update_func_args_action_prob_times_index",
    type=int,
    default=-1000,
    help="Index of the argument holding the decision times the action probabilities correspond to in the tuple of algorithm update func args, if applicable",
)
@click.option(
    "--inference_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user loss/estimating function used to determine the inference estimate and relevant imports.  The filename without its extension will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--inference_func_type",
    type=click.Choice([FunctionTypes.LOSS, FunctionTypes.ESTIMATING]),
    help="Type of function used to summarize inference.  If loss, inference should correspond to choosing parameters to minimize it.  If estimating, inference should correspond to setting the function equal to zero and solving for the parameters.",
    required=True,
)
@click.option(
    "--inference_func_args_theta_index",
    type=int,
    required=True,
    help="Index of the algorithm parameter vector beta in the tuple of inference loss/estimating func args",
)
@click.option(
    "--theta_calculation_func_filename",
    type=click.Path(exists=True),
    help="Path to file that allows one to actually calculate a theta estimate given the study dataframe only. One must supply either this or a precomputed theta estimate. The filename without its extension will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--in_study_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates whether a user is in the study",
)
@click.option(
    "--action_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates which action was taken",
)
@click.option(
    "--policy_num_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates the policy number in use",
)
@click.option(
    "--calendar_t_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates calendar time (shared integer index across users).",
)
@click.option(
    "--user_id_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates user id",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that gives action probabilities",
)
@click.option(
    "--suppress_interactive_data_checks",
    type=bool,
    default=False,
    help="Flag to suppress any data checks that require user input. This is suitable for tests.",
)
@click.option(
    "--suppress_all_data_checks",
    type=bool,
    default=False,
    help="Flag to suppress all data checks. This is suitable for large simulations.",
)
@click.option(
    "--small_sample_correction",
    type=click.Choice(
        [
            SmallSampleCorrections.none,
            SmallSampleCorrections.HC1,
            SmallSampleCorrections.custom_meat_modifier,
        ]
    ),
    default=SmallSampleCorrections.HC1,
    help="Type of small sample correction to apply to the variance estimate",
)
@click.option(
    "--meat_modifier_func_filename",
    type=click.Path(exists=True),
    help="File that contains the meat matrix modifier function and relevant imports.  The filename without its extension will be assumed to match the function name.",
)
def analyze_dataset(
    study_df_pickle,
    action_prob_func_filename,
    action_prob_func_args_pickle,
    action_prob_func_args_beta_index,
    alg_update_func_filename,
    alg_update_func_type,
    alg_update_func_args_pickle,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    inference_func_filename,
    inference_func_type,
    inference_func_args_theta_index,
    theta_calculation_func_filename,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_col_name,
    suppress_interactive_data_checks,
    suppress_all_data_checks,
    small_sample_correction,
    meat_modifier_func_filename,
):
    """
    Make sure in study is never on for more than one stretch EDIT: unclear if
    this will remain an invariant as we deal with more complicated data missingness

    I think I'm agnostic to indexing of calendar times but should check because
    otherwise need to add a check here to verify required format.

    Currently assuming function args can be placed in a numpy array. Must be scalar, 1d or 2d array.
    Higher dimensional objects not supported.  Not entirely sure what kind of "scalars" apply.

    Beta must be vector (not matrix)

    Codify assumptions that make get_first_applicable_time work.  The main
    thing is an assumption that users don't get different policies at the same
    time.  EDIT: Well... users can have different policies at the same time. So
    we can't codify this and have to rewrite that function.

    Codify assumptions used for collect_batched_in_study_actions

    Make the user give the min and max probabilities, and I'll enforce it

    I assume someone is in the study at each decision time. Check for this or
    see if it shouldn't always be true. EDIT: Is this true that I assume this?

    I also assume someone has some data to contribute at each update time. Check
    for this or see if shouldn't always be true. EDIT: Is it true that I assume this?
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    study_df = pickle.load(study_df_pickle)
    # TODO: Should I sort? Check how slow it is, for one.
    # study_df = pickle.load(study_df_pickle).sort_values(
    #     by=[user_id_col_name, calendar_t_col_name]
    # )
    action_prob_func_args = pickle.load(action_prob_func_args_pickle)
    alg_update_func_args = pickle.load(alg_update_func_args_pickle)

    theta_est = estimate_theta(study_df, theta_calculation_func_filename)

    # This does the first round of input validation, before computing any
    # gradients
    if not suppress_all_data_checks:
        input_checks.perform_first_wave_input_checks(
            study_df,
            in_study_col_name,
            action_col_name,
            policy_num_col_name,
            calendar_t_col_name,
            user_id_col_name,
            action_prob_col_name,
            action_prob_func_filename,
            action_prob_func_args,
            action_prob_func_args_beta_index,
            alg_update_func_args,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            theta_est,
            suppress_interactive_data_checks,
            small_sample_correction,
            meat_modifier_func_filename,
        )

    # TODO: Perhaps add a check that the supplied action probabilities in args
    # can also be reconstructed from the action probability function.

    policy_num_by_decision_time_by_user = (
        construct_policy_num_by_decision_time_by_user_map(
            study_df, user_id_col_name, calendar_t_col_name, policy_num_col_name
        )
    )
    beta_index_by_policy_num = construct_beta_index_by_policy_num_map(
        policy_num_by_decision_time_by_user,
        policy_num_col_name,
    )

    all_betas = collect_all_betas(
        beta_index_by_policy_num, alg_update_func_args, alg_update_func_args_beta_index
    )

    action_by_decision_time_by_user_id = {}
    policy_num_by_decision_time_by_user_id = {}
    for user_id, user_df in study_df.groupby(user_id_col_name):
        in_study_df = user_df[user_df[in_study_col_name] == 1]
        action_by_decision_time_by_user_id[user_id] = dict(
            zip(in_study_df[calendar_t_col_name], in_study_df[action_col_name])
        )
        policy_num_by_decision_time_by_user_id[user_id] = dict(
            zip(in_study_df[calendar_t_col_name], in_study_df[policy_num_col_name])
        )

    user_ids = study_df[user_id_col_name].unique()

    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = process_inference_func_args(
        inference_func_filename,
        study_df,
        user_ids,
        action_prob_col_name,
        calendar_t_col_name,
        user_id_col_name,
        in_study_col_name,
    )

    single_user_weighted_estimating_function_stacker = (
        construct_single_user_weighted_estimating_function_stacker(
            action_prob_func_filename,
            action_prob_func_args_beta_index,
            alg_update_func_filename,
            alg_update_func_type,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            inference_func_filename,
            inference_func_type,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            beta_index_by_policy_num,
            action_by_decision_time_by_user_id,
            policy_num_by_decision_time_by_user_id,
            action_prob_func_args,
            alg_update_func_args,
            inference_func_args_by_user_id,
            inference_action_prob_decision_times_by_user_id,
        )
    )

    weighted_estimating_function_stacks = [
        single_user_weighted_estimating_function_stacker(
            theta_est,
            all_betas,
            user_id,
        )
        for user_id in user_ids
    ]

    # roadmap: vmap the derivatives of the above vectors over users (if I can, shapes may differ...) and then average
    inverse_bread = construct_inverse_bread(
        single_user_weighted_estimating_function_stacker,
        action_prob_func_args,
        action_prob_func_args_beta_index,
        alg_update_func_args,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        study_df,
        in_study_col_name,
        action_col_name,
        policy_num_col_name,
        calendar_t_col_name,
        user_id_col_name,
        action_prob_col_name,
        theta_est,
    )

    # TODO: reinsert estimating function sum check.  not sure about hessian
    # condition number checks

    # Write analysis results to same directory that input files are in
    folder_path = pathlib.Path(study_df_pickle.name).parent.resolve()
    with open(f"{folder_path}/analysis.pkl", "wb") as f:
        pickle.dump(
            {
                "theta_est": theta_est,
                "adaptive_sandwich_var_estimate": adaptive_sandwich_var_estimate,
                "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
            },
            f,
        )

    with open(f"{folder_path}/debug_pieces.pkl", "wb") as f:
        pickle.dump(
            {
                "theta_est": theta_est,
                "adaptive_sandwich_var_estimate": adaptive_sandwich_var_estimate,
                "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
                "joint_bread_inverse_matrix": joint_adaptive_bread_inverse_matrix,
                "joint_meat_matrix": joint_meat_matrix,
                "inference_loss_gradients": inference_loss_gradients,
                "inference_loss_hessians": inference_loss_hessians,
                "inference_loss_gradient_pi_derivatives": inference_loss_gradient_pi_derivatives,
                "algorithm_statistics_by_calendar_t": algorithm_statistics_by_calendar_t,
                "upper_left_bread_inverse": upper_left_bread_inverse,
            },
            f,
        )

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdaptive sandwich variance estimate:\n {adaptive_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )


def construct_beta_index_by_policy_num_map(study_df, policy_num_col_name):
    """
    Constructs a mapping from non-initial, non-fallback policy numbers to the index of the
    corresponding beta in all_betas.

    This is useful because differentiating the stacked estimating functions with respect to all the
    betas is simplest if they are passed in a single list. This auxiliary data then allows us to
    route the right beta to the right policy number at each time.

    If we really keep the enforcement of consecutive policy numbers, we don't actually need all
    this logic and can just pass around the initial policy number, but I'd like to have this
    handle the increasing (non-fallback) case even though upstream we currently do require no
    gaps.
    """

    unique_non_fallback_policy_nums = (
        study_df[study_df[policy_num_col_name] >= 0, policy_num_col_name]
        .unique()
        .tolist()
        .sort()
    )
    return {
        policy_num: i for i, policy_num in enumerate(unique_non_fallback_policy_nums)
    }


def construct_policy_num_by_decision_time_by_user_map(
    study_df: pandas.DataFrame,
    user_id_col_name: str,
    calendar_t_col_name: str,
    policy_num_col_name: str,
) -> dict[collections.abc.Hashable, dict[int, int]]:
    """
    Constructs a mapping from decision times to policy numbers for each user.
    """
    policy_num_by_decision_time_by_user = {}
    for user_id, user_df in study_df.groupby(user_id_col_name):
        policy_num_by_decision_time_by_user[user_id] = dict(
            zip(user_df[calendar_t_col_name], user_df[policy_num_col_name])
        )
    return policy_num_by_decision_time_by_user


def collect_all_betas(
    beta_index_by_policy_num, alg_update_func_args, alg_update_func_args_beta_index
):
    """
    Collects all betas produced by the algorithm updates in an ordered list.

    This data structure is chosen because it makes for the most convenient
    differentiation of the stacked estimating functions with respect to all the
    betas. Otherwise a dictionary keyed on policy number would be more natural.
    """
    all_betas = []
    for policy_num in sorted(beta_index_by_policy_num.keys()):
        for user_id in alg_update_func_args[policy_num]:
            if alg_update_func_args[policy_num][user_id]:
                all_betas.append(
                    alg_update_func_args[policy_num][user_id][
                        alg_update_func_args_beta_index
                    ]
                )
                break
    return jnp.array(all_betas)


def process_inference_func_args(
    inference_func_filename,
    study_df,
    user_ids,
    action_prob_col_name,
    calendar_t_col_name,
    user_id_col_name,
    in_study_col_name,
) -> tuple[dict[collections.abc.Hashable, tuple[Any, ...]], int]:
    """
    Collects the inference function arguments for each user.

    Note that theta and action probabilities, if present, will be replaced later
    so that the function can be differentiated with respect to shared versions
    of them.
    """

    inference_func = load_function_from_same_named_file(inference_func_filename)
    num_args = inference_func.__code__.co_argcount
    inference_func_arg_names = inference_func.__code__.co_varnames[:num_args]
    inference_func_args_by_user_id = {}

    inference_func_args_action_prob_index = -1
    inference_action_prob_decision_times_by_user_id = {}

    using_action_probs = action_prob_col_name in inference_func_arg_names
    if using_action_probs:
        inference_func_args_action_prob_index = inference_func_arg_names.index(
            action_prob_col_name
        )
    for user_id in user_ids:
        user_args_list = []
        filtered_user_data = study_df.loc[study_df[user_id_col_name] == user_id]
        for col_name in enumerate(inference_func_arg_names):
            user_args_list.append(
                get_in_study_df_column(filtered_user_data, col_name, in_study_col_name)
            )
        inference_func_args_by_user_id[user_id] = tuple(user_args_list)
        if using_action_probs:
            inference_action_prob_decision_times_by_user_id[user_id] = (
                get_in_study_df_column(
                    filtered_user_data, calendar_t_col_name, in_study_col_name
                )
            )

    return (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    )


# TODO: Docstring
def get_radon_nikodym_weight(
    beta_target: jnp.ndarray,
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    action: int,
    *action_prob_func_args_single_user: tuple[Any, ...],
):
    """
    Computes a ratio of action probabilities where in the denominator beta_target is substituted into the rest of the
    action probability function arguments in place of whatever is given, and in the numerator the
    original value is used.  Even though in practice we call this in such a way that the beta value
    is the same in numerator and denominator, it is important to define the function this way so
    that differentiation, which is with respect to the numerator beta, is done correctly.

    """

    # numerator
    pi_beta = action_prob_func(*action_prob_func_args_single_user)

    # denominator
    beta_target_action_prob_func_args_single_user = [*action_prob_func_args_single_user]
    beta_target_action_prob_func_args_single_user[action_prob_func_args_beta_index] = (
        beta_target
    )
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_user)

    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


def construct_single_user_weighted_estimating_function_stacker(
    action_prob_func_filename: str,
    action_prob_func_args_beta_index: int,
    alg_update_func_filename: str,
    alg_update_func_type: str,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    inference_func_filename: str,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    beta_index_by_policy_num: dict[int, int],
    action_by_decision_time_by_user_id: dict[collections.abc.Hashable, dict[int, int]],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
    action_prob_func_args_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    update_func_args_by_policy_num_by_user_id: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    inference_func_args_by_user_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_user_id: dict[
        collections.abc.Hashable, list[int]
    ],
) -> callable:
    """
    Returns a function that computes a weighted estimating function stack for a single user. This
    includes a vertical stack of weighted estimating functions for each update and then one at the
    bottom for inference.

    Arguments that are not user-specific are provided at this level, whereas user-specific args are
    provided when the returned function is called.

    Args:
        action_prob_func_filename (str):
            The filename of the action probability function to be loaded.

        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.

        alg_update_func_filename (str):
            The filename of the algorithm update function to be loaded.

        alg_update_func_type (str):
            The type of the algorithm update function. FunctionTypes.LOSS or
            FunctionTypes.ESTIMATING.

        alg_update_func_args_beta_index (int):
            The index of the beta argument in the algorithm update function's arguments.

        alg_update_func_args_action_prob_index (int):
            The index of the action probabilities argument in the algorithm update function's
            arguments.

        alg_update_func_args_action_prob_times_index (int):
            The index of the argument holding the decision times the action probabilities correspond
            to in the algorithm update function's arguments.

        inference_func_filename (str):
            The filename of the inference loss function to be loaded.

        inference_func_type (str):
            The type of the inference function. FunctionTypes.LOSS or FunctionTypes.ESTIMATING.

        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference loss function arguments.

        inference_func_args_action_prob_index (int):
            The index of the action probabilities argument in the inference loss function's
            arguments.

        beta_index_by_policy_num (dict[int, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_betas.

        action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to actions taken.

        policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to the policy number in use.
            This may be user-specific. Should be sorted by decision time.

        action_prob_func_args_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for action probability function
            EXCEPT beta. Should be sorted by decision time.

        update_func_args_by_policy_num_by_user_id (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A map from policy numbers to tuples containing the arguments for the algorithm
            loss or estimating functions EXCEPT beta when producing this policy.
            Should be sorted by policy number.

        inference_func_args_by_user_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A tuple containing the arguments for the loss function EXCEPT beta for this user.

        inference_action_prob_decision_times_by_user_id (dict[collections.abc.Hashable, list[int]]):
            For each user, a list of decision times for which action probabilities are needed.
            Essentially just in-study times.
    """
    action_prob_func = load_function_from_same_named_file(action_prob_func_filename)
    alg_update_func = load_function_from_same_named_file(alg_update_func_filename)
    inference_func = load_function_from_same_named_file(inference_func_filename)

    algorithm_estimating_func = (
        jax.grad(alg_update_func, argnums=alg_update_func_args_beta_index)
        if (alg_update_func_type == FunctionTypes.LOSS)
        else alg_update_func
    )

    inference_estimating_func = jax.grad(
        jax.grad(inference_func, argnums=inference_func_args_theta_index)
        if (inference_func_type == FunctionTypes.LOSS)
        else inference_func
    )

    # TODO: Decide whether to deal with or mask multiple active policy numbers a
    # at the same time (e.g. app opening issue.) Have I already dealt with it?
    # TODO: Break into smaller functions.
    def single_user_weighted_algorithm_estimating_function_stacker(
        theta: jnp.ndarray,
        all_betas: jnp.ndarray,
        user_id: collections.abc.Hashable,
    ) -> jnp.ndarray:
        """
        Computes a weighted estimating function stack for a given set of algorithm update function arguments,
        and action probability function arguments.

        Args:
            theta (jnp.ndarray):
                The estimate of the parameter vector.

            all_betas (jnp.ndarray):
                A 2D JAX NumPy array containing the betas produced by all updates.

            user_id (collections.abc.Hashable):
                The user ID for which to compute the weighted estimating function stack.

        Returns:
            jnp.ndarray: A JAX NumPy array representing the weighted estimating function stack.
        """

        # First, reformat the supplied data into more convienent structures.

        # 1. Form a dictionary mapping policy numbers to the first time they were
        # applicable (for this user). Collect the first time after the first update
        # separately for convenience.  These are both used to form the Radon-Nikodym
        # weights for the right times.
        min_time_by_policy_num = {}
        first_time_after_first_update = None
        for decision_time, policy_num in policy_num_by_decision_time_by_user_id[
            user_id
        ].items():
            if policy_num not in min_time_by_policy_num:
                min_time_by_policy_num[policy_num] = decision_time

            if (
                policy_num in beta_index_by_policy_num
                and first_time_after_first_update is None
            ):
                first_time_after_first_update = decision_time

        # 2. Thread the central betas into the action probability arguments
        # for this particular user. This enables differentiation of the Radon-Nikodym
        # weights and action probabilities with respect to these shared betas.
        threaded_single_user_action_prob_func_args_by_decision_time = {}
        for (
            decision_time,
            action_prob_action_prob_args_by_user_id,
        ) in action_prob_func_args_by_decision_time_by_user_id.items():
            if not action_prob_action_prob_args_by_user_id[user_id]:
                threaded_single_user_action_prob_func_args_by_decision_time[
                    decision_time
                ] = ()
                continue

            beta_to_introduce = all_betas[
                beta_index_by_policy_num[
                    policy_num_by_decision_time_by_user_id[user_id][decision_time]
                ]
            ]
            threaded_single_user_action_prob_func_args_by_decision_time[
                decision_time
            ] = (
                action_prob_action_prob_args_by_user_id[user_id][
                    :action_prob_func_args_beta_index
                ]
                + (beta_to_introduce,)
                + action_prob_action_prob_args_by_user_id[user_id][
                    action_prob_func_args_beta_index + 1 :
                ]
            )

        # 3. Thread the central betas into the algorithm update function arguments
        # and replace any action probabilities with reconstructed ones from the above
        # arguments with the central betas introduced.
        threaded_single_user_update_func_args_by_policy_num = {}
        for (
            policy_num,
            update_func_args_by_user_id,
        ) in update_func_args_by_policy_num_by_user_id.items():
            if not update_func_args_by_user_id[user_id]:
                threaded_single_user_update_func_args_by_policy_num[policy_num] = ()
                continue

            beta_to_introduce = all_betas[beta_index_by_policy_num[policy_num]]
            threaded_single_user_update_func_args_by_policy_num[policy_num] = (
                update_func_args_by_user_id[:alg_update_func_args_beta_index]
                + (beta_to_introduce,)
                + update_func_args_by_user_id[alg_update_func_args_beta_index + 1 :]
            )

            if alg_update_func_args_action_prob_index >= 0:
                action_prob_times = update_func_args_by_user_id[user_id][
                    alg_update_func_args_action_prob_times_index
                ]
                action_probs_to_introduce = jnp.array(
                    [
                        action_prob_func(
                            *threaded_single_user_action_prob_func_args_by_decision_time[
                                t
                            ]
                        )
                        for t in action_prob_times
                    ]
                ).reshape(
                    update_func_args_by_user_id[user_id][
                        alg_update_func_args_action_prob_index
                    ].shape
                )
                threaded_single_user_update_func_args_by_policy_num[policy_num] = (
                    threaded_single_user_update_func_args_by_policy_num[policy_num][
                        :alg_update_func_args_action_prob_index
                    ]
                    + (action_probs_to_introduce,)
                    + threaded_single_user_update_func_args_by_policy_num[policy_num][
                        alg_update_func_args_action_prob_index + 1 :
                    ]
                )

        # 3. Thread the central betas into the inference function arguments
        # and replace any action probabilities with reconstructed ones from the above
        # arguments with the central betas introduced.
        single_user_inference_func_args = inference_func_args_by_user_id[user_id]

        threaded_single_user_inference_func_args = (
            single_user_inference_func_args[:inference_func_args_theta_index]
            + (theta,)
            + single_user_inference_func_args[inference_func_args_theta_index + 1 :]
        )

        if inference_func_args_action_prob_index >= 0:
            action_probs_to_introduce = jnp.array(
                [
                    action_prob_func(
                        *threaded_single_user_action_prob_func_args_by_decision_time[t]
                    )
                    for t in inference_action_prob_decision_times_by_user_id[user_id]
                ]
            ).reshape(
                single_user_inference_func_args[
                    inference_func_args_action_prob_index
                ].shape
            )
            threaded_single_user_inference_func_args = (
                threaded_single_user_inference_func_args[
                    :inference_func_args_action_prob_index
                ]
                + (action_probs_to_introduce,)
                + threaded_single_user_inference_func_args[
                    inference_func_args_action_prob_index + 1 :
                ]
            )

        # Actually do a loop since we want both the min and max
        # TODO: perhaps move away from dictionary structure if slow
        # when batched over users.
        # TODO: Make sure action_by_decision_time has correct structure, only
        # in-study decision times present
        user_start_time = math.inf
        user_end_time = -math.inf
        for decision_time in action_by_decision_time_by_user_id[user_id]:
            user_start_time = min(user_start_time, decision_time)
            user_end_time = max(user_end_time, decision_time)

        # TODO: Handle case where action probabilities are (optionally) used in
        # estimating function
        # TODO: Make sure shapes are appropriate... code uses a 1d vector, but math is column vector
        # and they behave a little differently. It might be nice to define as a function of betas as
        # column vectors so that differentation is as plug-and-play as possible, but the first step
        # is to stack all the vectors together.
        return jnp.concatenate(
            # Algorithm stack
            [
                # Here we compute a product of Radon-Nikodym weights
                # for all decision times after the first update and before the update
                # update under consideration took effect, for which the user was in the study.
                (
                    jnp.prod(
                        [
                            # Note that we do NOT use the shared betas in the first arg, for which
                            # we don't wan't differentiation to happen with respect to. Just grab
                            # the original beta from the update function arguments. This is the same
                            # value, but impervious to differentiation with respect tot all_betas.
                            # The args, on the other hand, are a function of all_betas.
                            get_radon_nikodym_weight(
                                action_prob_func_args_by_decision_time_by_user_id[t][
                                    action_prob_func_args_beta_index
                                ],
                                action_prob_func,
                                action_prob_func_args_beta_index,
                                action_by_decision_time_by_user_id[user_id][t],
                                *threaded_single_user_action_prob_func_args_by_decision_time[
                                    t
                                ],
                            )
                            for t in range(
                                # The earliest time after the first update where the user was in the study
                                max(
                                    first_time_after_first_update,
                                    user_start_time,
                                ),
                                # The latest time the user was in the study before the time the update
                                # under consideration first applied. Note the + 1 because range
                                # does not include the right endpoint.
                                min(
                                    min_time_by_policy_num[policy_num],
                                    user_end_time + 1,
                                ),
                            )
                        ]
                    )  # Now use the above to weight the algo estimating function for this update
                    * algorithm_estimating_func(*update_args)
                    # If there are no arguments for the update function, the user is not yet in the
                    # study, so we just add a zero vector contribution to the sum across users.
                    # Note that after they exit, they still contribute all their data to later
                    # updates.
                    if update_args
                    else jnp.zeros(len(all_betas[0]))
                )
                for policy_num, update_args in threaded_single_user_update_func_args_by_policy_num.items()
            ]
            # Inference contribution
            # Here we compute a product of Radon-Nikodym weights
            # for all decision times after the first update for which the user was in the study
            + jnp.prod(
                [
                    # Note: as a above, the first arg is the original beta, not the shared one.
                    get_radon_nikodym_weight(
                        action_prob_func_args_by_decision_time_by_user_id[t][
                            action_prob_func_args_beta_index
                        ],
                        action_prob_func,
                        action_prob_func_args_beta_index,
                        action_by_decision_time_by_user_id[user_id][t],
                        *threaded_single_user_action_prob_func_args_by_decision_time[t],
                    )
                    # Go from the first time for the user that is after the first
                    # update, to their last active time
                    for t in range(
                        max(first_time_after_first_update, user_start_time),
                        user_end_time + 1,
                    )
                ]
            )
            * inference_estimating_func(*threaded_single_user_inference_func_args)
        )

    return single_user_weighted_algorithm_estimating_function_stacker


def calculate_beta_dim(alg_update_func_args, alg_update_func_args_beta_index):
    for user_args_dict in alg_update_func_args.values():
        for args in user_args_dict.values():
            if args:
                return args[alg_update_func_args_beta_index].size


# TODO: Docstring
def estimate_theta(study_df, theta_calculation_func_filename):
    logger.info("Forming theta estimate.")
    theta_calculation_func = load_function_from_same_named_file(
        theta_calculation_func_filename
    )

    return theta_calculation_func(study_df)


# TODO: docstring
# TODO: One of the hotspots for update time logic to be removed
def calculate_algorithm_statistics(
    study_df,
    in_study_col_name,
    action_col_name,
    policy_num_col_name,
    calendar_t_col_name,
    user_id_col_name,
    action_prob_func_filename,
    action_prob_func_args,
    action_prob_func_args_beta_index,
    rl_update_func_filename,
    rl_update_func_type,
    rl_update_func_args,
    rl_update_func_args_beta_index,
    rl_update_func_args_action_prob_index,
    rl_update_func_args_action_prob_times_index,
):
    pi_and_weight_gradients_by_calendar_t = (
        calculate_derivatives.calculate_pi_and_weight_gradients(
            study_df,
            in_study_col_name,
            action_col_name,
            calendar_t_col_name,
            user_id_col_name,
            action_prob_func_filename,
            action_prob_func_args,
            action_prob_func_args_beta_index,
        )
    )
    rl_update_derivatives_by_calendar_t = (
        calculate_derivatives.calculate_rl_update_derivatives(
            study_df,
            rl_update_func_filename,
            rl_update_func_args,
            rl_update_func_type,
            rl_update_func_args_beta_index,
            rl_update_func_args_action_prob_index,
            rl_update_func_args_action_prob_times_index,
            policy_num_col_name,
            calendar_t_col_name,
        )
    )

    merged_dict = {}
    for t, t_dict in pi_and_weight_gradients_by_calendar_t.items():
        merged_dict[t] = {
            **t_dict,
            **rl_update_derivatives_by_calendar_t.get(t, {}),
        }

    return merged_dict


# TODO: docstring
# TODO: One of the hotspots for update time logic to be removed
def calculate_upper_left_bread_inverse(
    study_df, user_id_col_name, beta_dim, algorithm_statistics_by_calendar_t
):

    # List of times that were the first applicable time for some update
    # TODO: sort to not rely on insertion order?
    # TODO: use policy_num in df? alg statistics potentially ok too though.
    next_times_after_update = [
        t
        for t, value in algorithm_statistics_by_calendar_t.items()
        if "loss_gradients_by_user_id" in value
    ]

    # Form the dimensions for our bread matrix portion (pre-inverting)
    num_updates = len(next_times_after_update)
    overall_dim = beta_dim * num_updates
    output_matrix = jnp.zeros((overall_dim, overall_dim))

    user_ids = study_df[user_id_col_name].unique()
    num_users = len(user_ids)

    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user. The one complication is that we add some
    # padding of zeros for decision times before the first update to make
    # indexing simpler below.
    # NOTE there was a bug here that ASSUMED the padding needed to happen,
    # in particular that the algo statistics started at
    # next_times_after_update[0].  This is not necessarily true, and is now
    # dictated by the args passed to us.  Because I want to allow the user to
    # pass pi args for all decision times (in fact this should be the default),
    # I instead will make this the time I deal with that. I will just zero out
    # any pi gradients until after the first update.  Note that isn't necessary;
    # we could do nothing, because this is just about getting the right values
    # at the right index.  But then we are assuming that we have pi gradients
    # from the beginning.  Instead just take this heavy-handed approach and
    # ensure we have the shape we want whether data starts immediately after or
    # sometime before the first update.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_dict["pi_gradients_by_user_id"][user_id]
                    for t, t_dict in algorithm_statistics_by_calendar_t.items()
                    if t >= next_times_after_update[0]
                ]
            ),
            pad_width=((next_times_after_update[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # This loop iterates over all times that were the first applicable time
    # for a non-initial policy. Take care to note that update_idx starts at 0.
    # Think of each iteration of this loop as creating a (block) row of the matrix
    for update_idx, next_t_after_update in enumerate(next_times_after_update):
        logger.info(
            "Processing the update that first applied at time %s.", next_t_after_update
        )
        t_stats_dict = algorithm_statistics_by_calendar_t[next_t_after_update]

        # This loop creates the non-diagonal terms for the current update
        # Think of each iteration of this loop as creating one term in the current (block) row
        logger.info("Creating the non-diagonal terms for the current update.")
        for i in range(update_idx):
            lower_t = next_times_after_update[i]
            upper_t = next_times_after_update[i + 1]
            running_entry_holder = jnp.zeros((beta_dim, beta_dim))

            # This loop calculates the per-user quantities that will be
            # averaged for the final matrix entries
            for user_id, loss_gradient in t_stats_dict[
                "loss_gradients_by_user_id"
            ].items():
                weight_gradient_sum = jnp.zeros(beta_dim)

                # This loop iterates over decision times in slices
                # according to what was used for each update to collect the
                # right weight gradients
                for t in range(
                    lower_t,
                    upper_t,
                ):
                    weight_gradient_sum += algorithm_statistics_by_calendar_t[t][
                        "weight_gradients_by_user_id"
                    ][user_id]

                running_entry_holder += jnp.outer(
                    loss_gradient,
                    weight_gradient_sum,
                )

                # TODO: Detailed comment explaining this logic and the data
                # orientation that makes it work.  Also note the assumption
                # that the estimating function is additive across times
                # so that matrix multiplication is the right operation. Also
                # place this comment on the after study analysis logic or
                # link to the same explanation in both places.
                # Maybe link to a document with a picture...
                # TODO: This assumes indexing starts at 1
                mixed_beta_loss_derivative = jnp.matmul(
                    t_stats_dict["loss_gradient_pi_derivatives_by_user_id"][user_id][
                        :,
                        lower_t - 1 : upper_t - 1,
                    ],
                    pi_derivatives_by_user_id[user_id][
                        lower_t - 1 : upper_t - 1,
                        :,
                    ],
                )
                running_entry_holder += mixed_beta_loss_derivative
            # TODO: Use jnp.block instead of indexing
            output_matrix = output_matrix.at[
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                i * beta_dim : (i + 1) * beta_dim,
            ].set(running_entry_holder / num_users)

        # Add the diagonal hessian entry (which is already an average)
        # TODO: Use jnp.block instead of indexing
        output_matrix = output_matrix.at[
            (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
            (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
        ].set(t_stats_dict["avg_loss_hessian"])

    return output_matrix


# TODO: Docstring
# TODO: One of the hotspots for update time logic to be removed
def compute_variance_estimates(
    study_df,
    beta_dim,
    theta_est,
    algorithm_statistics_by_calendar_t,
    update_times,
    upper_left_bread_inverse,
    inference_loss_func_filename,
    inference_loss_func_args_theta_index,
    in_study_col_name,
    user_id_col_name,
    action_prob_col_name,
    calendar_t_col_name,
    suppress_interactive_data_checks,
    suppress_all_data_checks,
    small_sample_correction,
    meat_modifier_func_filename,
):
    # Collect list of user ids to guarantee we have a shared, fixed order
    # to iterate through in a variety of places.
    user_ids = study_df[user_id_col_name].unique()

    theta_dim = len(theta_est)

    logger.info("Forming adaptive sandwich variance estimator.")

    logger.info("Calculating all inference-side derivatives needed with JAX.")
    (
        inference_loss_gradients,
        inference_loss_hessians,
        inference_loss_gradient_pi_derivatives,
    ) = calculate_derivatives.calculate_inference_loss_derivatives(
        study_df,
        theta_est,
        inference_loss_func_filename,
        inference_loss_func_args_theta_index,
        user_ids,
        user_id_col_name,
        action_prob_col_name,
        in_study_col_name,
        calendar_t_col_name,
    )

    if not suppress_interactive_data_checks and not suppress_all_data_checks:
        input_checks.require_theta_estimating_functions_sum_to_zero(
            inference_loss_gradients, theta_dim
        )
        input_checks.require_non_singular_avg_hessian_inference(
            inference_loss_hessians,
        )

    logger.info("Forming adaptive joint meat.")
    joint_adaptive_meat_matrix = form_joint_adaptive_meat_matrix(
        theta_dim,
        update_times,
        beta_dim,
        algorithm_statistics_by_calendar_t,
        user_ids,
        inference_loss_gradients,
    )
    logger.debug("Adaptive joint meat:")
    logger.debug(joint_adaptive_meat_matrix)

    logger.info("Forming adaptive joint bread inverse and inverting.")
    max_t = study_df[calendar_t_col_name].max()
    joint_adaptive_bread_inverse_matrix = form_joint_adaptive_bread_inverse_matrix(
        upper_left_bread_inverse,
        max_t,
        algorithm_statistics_by_calendar_t,
        update_times,
        beta_dim,
        theta_dim,
        user_ids,
        inference_loss_gradients,
        inference_loss_hessians,
        inference_loss_gradient_pi_derivatives,
    )
    logger.debug("Adaptive joint bread inverse:")
    logger.debug(joint_adaptive_bread_inverse_matrix)

    # TODO: decide whether to in fact scrap the structure-based inversion
    # joint_adaptive_bread_matrix = invert_inverse_bread_matrix(
    #     joint_adaptive_bread_inverse_matrix, beta_dim, theta_dim
    # )

    joint_adaptive_bread_matrix = np.linalg.inv(joint_adaptive_bread_inverse_matrix)

    if not suppress_interactive_data_checks and not suppress_all_data_checks:
        input_checks.require_adaptive_bread_inverse_is_true_inverse(
            joint_adaptive_bread_matrix, joint_adaptive_bread_inverse_matrix
        )

    logger.info("Combining sandwich ingredients.")
    # Note the normalization here: underlying the calculations we have asymptotic normality
    # at rate sqrt(n), so in finite samples we approximate the observed variance of theta_hat
    # by dividing the variance of that limiting normal by a factor of n.
    joint_adaptive_variance = (
        joint_adaptive_bread_matrix
        @ joint_adaptive_meat_matrix
        @ joint_adaptive_bread_matrix.T
    ) / len(user_ids)
    logger.info("Finished forming adaptive sandwich variance estimator.")

    # This bottom right corner of the joint variance matrix is the portion
    # corresponding to just theta.  This distinguishes this matrix from the
    # *joint* adaptive variance matrix above, which covers both beta and theta.
    adaptive_sandwich_var = joint_adaptive_variance[
        -len(theta_est) :, -len(theta_est) :
    ]

    # We will also calculate the classical sandwich variance estimator for comparison.
    # But we take it piece by piece and also extract the *inverse* of the classical bread
    # because it is useful for extracting the non-joint adaptive meat.  This is
    # needed in case we are adjusting this meat matrix with a small sample correction.
    classical_bread, classical_meat, _ = get_classical_sandwich_var_pieces(
        theta_dim,
        inference_loss_gradients,
        inference_loss_hessians,
    )

    # Apply small sample correction if requested, and form the final variance estimates
    adaptive_sandwich_var, classical_sandwich_var = apply_small_sample_correction(
        adaptive_sandwich_var,
        classical_bread,
        classical_meat,
        len(user_ids),
        theta_dim,
        small_sample_correction,
    )

    return (
        adaptive_sandwich_var,
        classical_sandwich_var,
        # The following are returned for debugging purposes
        joint_adaptive_bread_inverse_matrix,
        joint_adaptive_meat_matrix,
        inference_loss_gradients,
        inference_loss_hessians,
        inference_loss_gradient_pi_derivatives,
    )


def invert_inverse_bread_matrix(inverse_bread, beta_dim, theta_dim):
    """
    Invert the inverse bread matrix to get the bread matrix.  This is a special
    function in order to take advantage of the block lower triangular structure.

    The procedure is as follows:
    1. Initialize the inverse matrix B = A^{-1} as a block lower triangular matrix
       with the same block structure as A.

    2. Compute the diagonal blocks B_{ii}:
       For each diagonal block A_{ii}, calculate:
           B_{ii} = A_{ii}^{-1}

    3. Compute the off-diagonal blocks B_{ij} for i > j:
       For each off-diagonal block B_{ij} (where i > j), compute:
           B_{ij} = -A_{ii}^{-1} * sum(A_{ik} * B_{kj} for k in range(j, i))
    """
    blocks = []
    num_beta_block_rows = (inverse_bread.shape[0] - theta_dim) // beta_dim

    # Create upper rows of block of bread (just the beta portion)
    for i in range(0, num_beta_block_rows):
        beta_block_row = []
        beta_diag_inverse = invert_matrix_and_check_conditioning(
            inverse_bread[
                beta_dim * i : beta_dim * (i + 1),
                beta_dim * i : beta_dim * (i + 1),
            ],
            try_tikhonov_if_poorly_conditioned=True,
        )
        for j in range(0, num_beta_block_rows):
            if i > j:
                beta_block_row.append(
                    -beta_diag_inverse
                    @ sum(
                        inverse_bread[
                            beta_dim * i : beta_dim * (i + 1),
                            beta_dim * k : beta_dim * (k + 1),
                        ]
                        @ blocks[k][j]
                        for k in range(j, i)
                    )
                )
            elif i == j:
                beta_block_row.append(beta_diag_inverse)
            else:
                beta_block_row.append(np.zeros((beta_dim, beta_dim)).astype(np.float32))

        # Extra beta * theta zero block. This is the last block of the row.
        # Any other zeros in the row have already been handled above.
        beta_block_row.append(np.zeros((beta_dim, theta_dim)))

        blocks.append(beta_block_row)

    # Create the bottom block row of bread (the theta portion)
    theta_block_row = []
    theta_diag_inverse = invert_matrix_and_check_conditioning(
        inverse_bread[
            -theta_dim:,
            -theta_dim:,
        ]
    )
    for k in range(0, num_beta_block_rows):
        theta_block_row.append(
            -theta_diag_inverse
            @ sum(
                inverse_bread[
                    -theta_dim:,
                    beta_dim * h : beta_dim * (h + 1),
                ]
                @ blocks[h][k]
                for h in range(k, num_beta_block_rows)
            )
        )

    theta_block_row.append(theta_diag_inverse)
    blocks.append(theta_block_row)

    return np.block(blocks)


# TODO: doc string
# TODO: This is a hotspot for update time logic to be removed
def form_joint_adaptive_meat_matrix(
    theta_dim,
    update_times,
    beta_dim,
    algo_stats_dict,
    user_ids,
    inference_loss_gradients,
):
    num_rows_and_cols = beta_dim * len(update_times) + theta_dim
    running_meat_matrix = jnp.zeros((num_rows_and_cols, num_rows_and_cols)).astype(
        jnp.float32
    )

    for i, user_id in enumerate(user_ids):
        user_meat_vector = jnp.concatenate(
            # beta estimating functions
            [
                algo_stats_dict[t]["loss_gradients_by_user_id"][user_id]
                for t in update_times
            ]
            # theta estimating function
            + [inference_loss_gradients[i]],
        ).reshape(-1, 1)
        running_meat_matrix += jnp.outer(user_meat_vector, user_meat_vector)

    return running_meat_matrix / len(user_ids)


# TODO: doc string
# TODO: This is a hotspot for update time logic to be removed
def form_joint_adaptive_bread_inverse_matrix(
    upper_left_bread_inverse,
    max_t,
    algo_stats_dict,
    update_times,
    beta_dim,
    theta_dim,
    user_ids,
    inference_loss_gradients,
    inference_loss_hessians,
    inference_loss_gradient_derivatives_wrt_pi,
):
    existing_rows = upper_left_bread_inverse.shape[0]

    # This is useful for sweeping through the decision times between updates
    # but critically also those after the final update
    update_times_and_upper_limit = (
        update_times if update_times[-1] == max_t + 1 else update_times + [max_t + 1]
    )

    # Begin by creating a few convenience data structures for the mixed theta/beta derivatives
    # that are most easily created for many decision times at once, whereas the following loop is
    # over update times.  We will pull appropriate quantities from here during iterations of the
    # loop.

    # This computes derivatives of the theta estimating function wrt the action probabilities
    # vector, which importantly has an element for *every* decision time.  We will later do the
    # work to multiply these by derivatives of pi with respect to beta, thus getting the quantities
    # we really want via the chain rule, and also summing terms that correspond to the *same* betas
    # behind the scenes.
    # NOTE THAT COLUMN INDEX i CORRESPONDS TO DECISION TIME i+1!
    # TODO: This [..., 0] might be nice to do earlier. Note there is also a corresponding RL-side
    # [..., 0] on the pi derivatives, but it happens closer to
    # the gradient computation.  On the other hand, it happens in a layer of the RL logic
    # that doesn't exist on the inference side (the layer that takes in the results from each
    # update).  As on the RL side, we [..., 0] instead of squeezing to not collapse the
    # parameter dimension if it's 1D.
    mixed_theta_pi_loss_derivatives_by_user_id = {
        user_id: inference_loss_gradient_derivatives_wrt_pi[i][..., 0]
        for i, user_id in enumerate(user_ids)
    }

    # This simply collects the pi derivatives with respect to betas for all
    # decision times for each user, reorganizing existing data from the RL side.
    # The one complication is that we add some padding of zeros for decision
    # times before the first update to be in correspondence with the above data
    # structure.
    # See the analogous comment in the construction of the RL portion of the
    # matrix for more details on why we limit to t >= update_times[0]. In short
    # we do not need the values before that, but want to allow them to be given.
    # Whether they are given or not, we choose to put zeros in their place.
    # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
    pi_derivatives_by_user_id = {
        user_id: jnp.pad(
            jnp.array(
                [
                    t_stats_dict["pi_gradients_by_user_id"][user_id]
                    for t, t_stats_dict in algo_stats_dict.items()
                    if t >= update_times[0]
                ]
            ),
            pad_width=((update_times[0] - 1, 0), (0, 0)),
        )
        for user_id in user_ids
    }

    # Think of each iteration of this loop as creating one off-diagonal term in
    # the final (block) row
    bottom_left_row_blocks = []
    for i in range(len(update_times)):
        lower_t = update_times_and_upper_limit[i]
        upper_t = update_times_and_upper_limit[i + 1]
        running_entry_holder = jnp.zeros((theta_dim, beta_dim))

        # This loop calculates the per-user quantities that will be
        # averaged for the final matrix entries
        for j, user_id in enumerate(user_ids):
            # 1. We first form the outer product of the estimating equation for theta
            # and the sum of the weight gradients with respect to beta for the
            # corresponding decision times

            theta_loss_gradient = inference_loss_gradients[j]

            weight_gradient_sum = jnp.zeros(beta_dim)

            # This loop iterates over decision times in slices between updates
            # to collect the right weight gradients
            # Note these may look more sparse than expected due to clipping, which
            # produces zero gradients when limits are hit.
            for t in range(lower_t, upper_t):
                weight_gradient_sum += algo_stats_dict[t][
                    "weight_gradients_by_user_id"
                ][user_id]
            running_entry_holder += jnp.outer(theta_loss_gradient, weight_gradient_sum)

            # 2. We now calculate mixed derivatives of the loss wrt theta and then beta. This piece
            # is a bit intricate; we only have the theta loss function in terms of the pis,
            # and the *values* of the pi derivatives wrt to betas available here, since the actual
            # pi functions are the domain of the RL side. The loss function also gets an action
            # probability for all decision times, not knowing which correspond to which
            # betas behind the scenes, so our tasks are to
            # 1. multiply these theta derivatives wrt pi for each relevant decision
            #    time by the corresponding pi derivative wrt beta
            # 2. sum together the products from the previous step that actually
            #    correspond to the same betas
            # The loop we are currently in is doing this for just the bucket of decision
            # times currently under consideration.

            # Multiply just the appropriate segments of the precomputed
            # mixed theta pi loss derivative matrix for the given user and
            # the precollected pi beta derivative matrix for the user. These
            # segments are simply those that correspond to all the decision times
            # in the current slice between updates under consideration.
            # NOTE THAT OUR HELPER DATA STRUCTURES ARE 0-INDEXED, SO WE SUBTRACT
            # 1 FROM OUR TIME BOUNDS.
            # NOTE we could also do something like a join on policy number,
            # then multiply and sum in groups--may be simpler to think about
            # than dealing with spans of update times
            mixed_theta_beta_loss_derivative = jnp.matmul(
                mixed_theta_pi_loss_derivatives_by_user_id[user_id][
                    :,
                    lower_t - 1 : upper_t - 1,
                ],
                pi_derivatives_by_user_id[user_id][
                    lower_t - 1 : upper_t - 1,
                    :,
                ],
            )

            running_entry_holder += mixed_theta_beta_loss_derivative

        bottom_left_row_blocks.append(running_entry_holder / len(user_ids))
    bottom_right_hessian = jnp.mean(inference_loss_hessians, axis=0)
    return jnp.block(
        [
            [
                upper_left_bread_inverse,
                jnp.zeros((existing_rows, theta_dim)),
            ],
            [
                jnp.block(bottom_left_row_blocks),
                bottom_right_hessian,
            ],
        ]
    )


# TODO: Needs tests
# TODO: Complete docstring
def get_classical_sandwich_var_pieces(theta_dim, loss_gradients, loss_hessians):
    """
    Forms standard sandwich variance estimator for inference (thetahat)

    Input:

    Output:
    - Sandwich variance estimator matrix (size dim_theta by dim_theta)
    """

    logger.info("Forming classical sandwich variance estimator.")
    num_users = len(loss_gradients)

    logger.info("Forming classical meat.")
    running_meat_matrix = np.zeros((theta_dim, theta_dim)).astype(jnp.float32)
    for loss_gradient in loss_gradients:
        user_meat_vector = loss_gradient.reshape(-1, 1)
        running_meat_matrix += np.outer(user_meat_vector, user_meat_vector)

    meat = running_meat_matrix / num_users

    logger.info("Forming classical bread inverse.")
    normalized_hessian = np.mean(loss_hessians, axis=0)

    # degrees of freedom adjustment
    # TODO: Reinstate? Provide reference? Mentioned in sandwich package
    # This is HC1 correction
    # Should we use something other than theta_dim for d?
    # meat = meat * (num_users - 1) / (num_users - theta_dim)

    logger.info("Inverting classical bread and combining ingredients.")
    inv_hessian = invert_matrix_and_check_conditioning(normalized_hessian)

    logger.info("Finished forming classical sandwich variance estimator.")

    # We return the bread and the meat, and also the inverse of the bread
    # because it may be needed for a small-sample corrections and we want to
    # avoid another inverse.
    return inv_hessian, meat, normalized_hessian


@cli.command()
@click.option(
    "--input_glob",
    help="A glob that captures all of the analyses to be collected.  Leaf folders will be searched for analyses",
    required=True,
)
@click.option(
    "--index_to_check_ci_coverage",
    type=int,
    help="The index of the parameter to check coverage for.  If not provided, coverage will not be checked.",
)
def collect_existing_analyses(input_glob, index_to_check_ci_coverage):

    raw_theta_estimates = []
    raw_adaptive_sandwich_var_estimates = []
    raw_classical_sandwich_var_estimates = []
    filenames = glob.glob(input_glob)

    logger.info("Found %d files under the glob %s", len(filenames), input_glob)
    if len(filenames) == 0:
        raise RuntimeError("Aborting because no files found. Please check path.")

    for i, filename in enumerate(filenames):
        if i and len(filenames) >= 10 and i % (len(filenames) // 10) == 0:
            logger.info("A(nother) tenth of files processed.")
        if not os.stat(filename).st_size:
            raise RuntimeError(
                "Empty analysis pickle.  This means there were probably timeouts or other failures during simulations."
            )
        with open(filename, "rb") as f:
            analysis_dict = pickle.load(f)
            (
                theta_est,
                adaptive_sandwich_var,
                classical_sandwich_var,
            ) = (
                analysis_dict["theta_est"],
                analysis_dict["adaptive_sandwich_var_estimate"],
                analysis_dict["classical_sandwich_var_estimate"],
            )
            raw_theta_estimates.append(theta_est)
            raw_adaptive_sandwich_var_estimates.append(adaptive_sandwich_var)
            raw_classical_sandwich_var_estimates.append(classical_sandwich_var)

    theta_estimates = np.array(raw_theta_estimates)
    adaptive_sandwich_var_estimates = np.array(raw_adaptive_sandwich_var_estimates)
    classical_sandwich_var_estimates = np.array(raw_classical_sandwich_var_estimates)

    theta_estimate = np.mean(theta_estimates, axis=0)
    empirical_var_normalized = empirical_var_normalized = np.atleast_2d(
        np.cov(theta_estimates.T, ddof=0)
    )
    mean_adaptive_sandwich_var_estimate = np.mean(
        adaptive_sandwich_var_estimates, axis=0
    )
    mean_classical_sandwich_var_estimate = np.mean(
        classical_sandwich_var_estimates, axis=0
    )

    # Calculate standard error (or corresponding variance) of variance estimate for each
    # component of theta.  This is done by finding an unbiased estimator of the standard
    # formula for the standard error of a variance from iid observations.
    # Population standard error formula: https://en.wikipedia.org/wiki/Variance
    # Unbiased estimator: https://stats.stackexchange.com/questions/307537/unbiased-estimator-of-the-variance-of-the-sample-variance
    theta_component_variance_std_errors = []
    for i in range(len(theta_estimate)):
        component_estimates = [estimate[i] for estimate in theta_estimates]
        second_central_moment = scipy.stats.moment(component_estimates, moment=4)
        fourth_central_moment = scipy.stats.moment(component_estimates, moment=4)
        n = len(theta_estimates)
        theta_component_variance_std_errors.append(
            np.sqrt(
                n
                * (
                    ((n) ** 2 - 3) * (second_central_moment) ** 2
                    + ((n - 1) ** 2) * fourth_central_moment
                )
                / ((n - 3) * (n - 2) * ((n - 1) ** 2))
            )
        )

    approximate_standard_errors = np.empty_like(empirical_var_normalized)
    for i, j in np.ndindex(approximate_standard_errors.shape):
        approximate_standard_errors[i, j] = max(
            theta_component_variance_std_errors[i],
            theta_component_variance_std_errors[j],
        )

    print(f"\nParameter estimate:\n{theta_estimate}")
    print(f"\nEmpirical variance:\n{empirical_var_normalized}")
    print(
        f"\nEmpirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):\n{approximate_standard_errors}"
    )
    print(
        f"\nAdaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nClassical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}\n",
    )
    print(
        f"\nAdaptive sandwich variance estimate std errors from empirical:\n{(mean_adaptive_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nClassical sandwich variance estimate std errors from empirical:\n{(mean_classical_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}\n",
    )

    if theta_estimates[0].size == 1:
        index_to_check_ci_coverage = 0
    if index_to_check_ci_coverage is not None:
        adaptive_cover_count = 0
        classical_cover_count = 0
        scalar_mean_theta = theta_estimate[index_to_check_ci_coverage]
        for single_theta_est in theta_estimates:
            scalar_single_theta = single_theta_est[index_to_check_ci_coverage]
            adaptive_se = math.sqrt(
                mean_adaptive_sandwich_var_estimate[index_to_check_ci_coverage][
                    index_to_check_ci_coverage
                ]
            )
            classical_se = math.sqrt(
                mean_classical_sandwich_var_estimate[index_to_check_ci_coverage][
                    index_to_check_ci_coverage
                ]
            )
            if (
                scalar_mean_theta - 1.96 * adaptive_se
                <= scalar_single_theta
                <= scalar_mean_theta + 1.96 * adaptive_se
            ):
                adaptive_cover_count += 1
            if (
                scalar_mean_theta - 1.96 * classical_se
                <= scalar_single_theta
                <= scalar_mean_theta + 1.96 * classical_se
            ):
                classical_cover_count += 1

        print(
            f"\nAdaptive sandwich 95% CI coverage:\n{adaptive_cover_count / len(theta_estimates)}",
        )
        print(
            f"\nClassical sandwich 95% CI coverage:\n{classical_cover_count / len(theta_estimates)}",
        )


def apply_small_sample_correction(
    adaptive_sandwich_var,
    classical_bread,
    classical_meat,
    num_users,
    theta_dim,
    small_sample_correction,
):
    classical_sandwich_var = (
        classical_bread @ classical_meat @ classical_bread.T
    ) / num_users
    if small_sample_correction == SmallSampleCorrections.none:
        return adaptive_sandwich_var, classical_sandwich_var
    if small_sample_correction == SmallSampleCorrections.HC1:
        correction = num_users / (num_users - theta_dim)
        return adaptive_sandwich_var * correction, classical_sandwich_var * correction
    if small_sample_correction == SmallSampleCorrections.custom_meat_modifier:
        # TODO: If we go this route, we need a function for making the custom H matrices to add into
        # the meat between X and the residual per person.  This is not very hard for the classical case,
        # though the structure of the function will probably be pretty particular. It would perhaps
        # be most natural to have a weighted linear regression mode for inference where the features
        # and weights are specified, and then we could automatically form the H_ii matrices ala Mancl
        # and DeRouen.  But the larger question is how to incorporate the H's into the adaptive version
        # of the meat.  Does it just affect the classical portion or also the adjustment? And regardless,
        # we'd have to directly construct the adaptive meat in a way that we currently are not to inject
        # the H's. Also, the time for this correction to happen would not be here. We would have to reach
        # into the initial meat construction logic in each case.
        raise NotImplementedError("Custom meat modifier not yet implemented.")
    raise ValueError("Invalid small sample correction type.")


if __name__ == "__main__":
    cli()
