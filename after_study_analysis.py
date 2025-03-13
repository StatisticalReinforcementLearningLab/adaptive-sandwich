from __future__ import annotations

import collections
import pathlib
import pickle
import os
import logging
import glob
import math
from typing import Any

import click
import jax
import numpy as np
from jax import numpy as jnp
import scipy
import pandas

from constants import FunctionTypes, SmallSampleCorrections
import input_checks


from helper_functions import (
    conditional_x_or_one_minus_x,
    get_in_study_df_column,
    invert_matrix_and_check_conditioning,
    load_function_from_same_named_file,
    replace_tuple_index,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

# TODO: Break this file up?


@click.group()
def cli():
    pass


# TODO: Take in requirements files for action prob and loss and take derivatives
# in corresponding sandbox. For now we just assume the dependencies in this package
# suffice.
# TODO: Handle raw timestamps instead of calendar time index? For now I'm requiring it.
# More generally, handle decision times being different across different users? Would like
# to consolidate.
# TODO: Check all help strings for accuracy.
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

    theta_est = jnp.array(estimate_theta(study_df, theta_calculation_func_filename))

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

    beta_index_by_policy_num, initial_policy_num = (
        construct_beta_index_by_policy_num_map(
            study_df, policy_num_col_name, in_study_col_name
        )
    )

    all_post_update_betas = collect_all_post_update_betas(
        beta_index_by_policy_num, alg_update_func_args, alg_update_func_args_beta_index
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

    (
        inference_func_args_by_user_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_user_id,
    ) = process_inference_func_args(
        inference_func_filename,
        inference_func_args_theta_index,
        study_df,
        theta_est,
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
            initial_policy_num,
            action_by_decision_time_by_user_id,
            policy_num_by_decision_time_by_user_id,
            action_prob_func_args,
            alg_update_func_args,
            inference_func_args_by_user_id,
            inference_action_prob_decision_times_by_user_id,
        )
    )

    logger.info("Constructing joint adaptive bread inverse matrix.")
    user_ids = jnp.array(study_df[user_id_col_name].unique())
    # roadmap: vmap the derivatives of the above vectors over users (if I can, shapes may differ...) and then average
    (
        joint_adaptive_bread_inverse_matrix,
        joint_adaptive_meat_matrix,
        classical_bread_inverse_matrix,
        classical_meat_matrix,
        avg_estimating_function_stack,
    ) = construct_classical_and_adaptive_inverse_bread_and_meat_and_avg_estimating_function_stack(
        single_user_weighted_estimating_function_stacker,
        theta_est,
        all_post_update_betas,
        user_ids,
    )

    if not suppress_interactive_data_checks and not suppress_all_data_checks:
        input_checks.require_estimating_functions_sum_to_zero(
            avg_estimating_function_stack
        )

    logger.info("Forming classical sandwich variance estimate...")
    classical_bread_matrix = invert_matrix_and_check_conditioning(
        classical_bread_inverse_matrix
    )
    classical_sandwich_var_estimate = (
        classical_bread_matrix @ classical_meat_matrix @ classical_bread_matrix.T
    ) / len(user_ids)

    # TODO: decide whether to in fact scrap the structure-based inversion
    logger.info("Inverting joint bread inverse matrix...")
    joint_adaptive_bread_matrix = invert_matrix_and_check_conditioning(
        joint_adaptive_bread_inverse_matrix, try_tikhonov_if_poorly_conditioned=True
    )

    if not suppress_interactive_data_checks and not suppress_all_data_checks:
        input_checks.require_adaptive_bread_inverse_is_true_inverse(
            joint_adaptive_bread_matrix, joint_adaptive_bread_inverse_matrix
        )

    logger.info("Forming joint adaptive sandwich variance estimate...")
    joint_adaptive_var_estimate = (
        joint_adaptive_bread_matrix
        @ joint_adaptive_meat_matrix
        @ joint_adaptive_bread_matrix.T
    ) / len(user_ids)

    # This bottom right corner of the joint variance matrix is the portion
    # corresponding to just theta.  This distinguishes this matrix from the
    # *joint* adaptive variance matrix above, which covers both beta and theta.
    adaptive_sandwich_var_estimate = joint_adaptive_var_estimate[
        -len(theta_est) :, -len(theta_est) :
    ]

    logger.info("Writing results to file...")
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
                "joint_meat_matrix": joint_adaptive_meat_matrix,
            },
            f,
        )

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdaptive sandwich variance estimate:\n {adaptive_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )


def construct_beta_index_by_policy_num_map(
    study_df, policy_num_col_name, in_study_col_name
):
    """
    Constructs a mapping from non-initial, non-fallback policy numbers to the index of the
    corresponding beta in all_post_update_betas.

    This is useful because differentiating the stacked estimating functions with respect to all the
    betas is simplest if they are passed in a single list. This auxiliary data then allows us to
    route the right beta to the right policy number at each time.

    If we really keep the enforcement of consecutive policy numbers, we don't actually need all
    this logic and can just pass around the initial policy number, but I'd like to have this
    handle the merely increasing (non-fallback) case even though upstream we currently do require no
    gaps.
    """

    unique_sorted_non_fallback_policy_nums = sorted(
        study_df[
            (study_df[policy_num_col_name] >= 0) & (study_df[in_study_col_name] == 1)
        ][policy_num_col_name]
        .unique()
        .tolist()
    )
    # This assumes only the first policy is an initial policy not produced by an update.
    # Hence the [1:] slice.
    return {
        policy_num: i
        for i, policy_num in enumerate(unique_sorted_non_fallback_policy_nums[1:])
    }, unique_sorted_non_fallback_policy_nums[0]


def collect_all_post_update_betas(
    beta_index_by_policy_num, alg_update_func_args, alg_update_func_args_beta_index
):
    """
    Collects all betas produced by the algorithm updates in an ordered list.

    This data structure is chosen because it makes for the most convenient
    differentiation of the stacked estimating functions with respect to all the
    betas. Otherwise a dictionary keyed on policy number would be more natural.
    """
    all_post_update_betas = []
    for policy_num in sorted(beta_index_by_policy_num.keys()):
        for user_id in alg_update_func_args[policy_num]:
            if alg_update_func_args[policy_num][user_id]:
                all_post_update_betas.append(
                    alg_update_func_args[policy_num][user_id][
                        alg_update_func_args_beta_index
                    ]
                )
                break
    return jnp.array(all_post_update_betas)


def extract_action_and_policy_by_decision_time_by_user_id(
    study_df,
    user_id_col_name,
    in_study_col_name,
    calendar_t_col_name,
    action_col_name,
    policy_num_col_name,
):
    action_by_decision_time_by_user_id = {}
    policy_num_by_decision_time_by_user_id = {}
    for user_id, user_df in study_df.groupby(user_id_col_name):
        in_study_user_df = user_df[user_df[in_study_col_name] == 1]
        action_by_decision_time_by_user_id[user_id] = dict(
            zip(
                in_study_user_df[calendar_t_col_name], in_study_user_df[action_col_name]
            )
        )
        policy_num_by_decision_time_by_user_id[user_id] = dict(
            zip(
                in_study_user_df[calendar_t_col_name],
                in_study_user_df[policy_num_col_name],
            )
        )
    return action_by_decision_time_by_user_id, policy_num_by_decision_time_by_user_id


def process_inference_func_args(
    inference_func_filename,
    inference_func_args_theta_index,
    study_df,
    theta_est,
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

    # Convert to list from jnp array so extraction is simplest
    for user_id in study_df[user_id_col_name].unique():
        user_args_list = []
        filtered_user_data = study_df.loc[study_df[user_id_col_name] == user_id]
        for idx, col_name in enumerate(inference_func_arg_names):
            if idx == inference_func_args_theta_index:
                user_args_list.append(theta_est)
                continue
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
    Computes a ratio of action probabilities where in the denominator beta_target is substituted
    into the rest of the action probability function arguments in place of whatever is given, and
    in the numerator the original value is used.  Even though in practice we call this in such a way
    that the beta value is the same in numerator and denominator, it is important to define the
    function this way so that differentiation, which is with respect to the numerator beta, is done
    correctly.
    """

    # numerator
    pi_beta = action_prob_func(*action_prob_func_args_single_user)

    # denominator, where we thread in beta_target so that differentiation with respect to the
    # original beta in the arguments leaves this alone.
    beta_target_action_prob_func_args_single_user = [*action_prob_func_args_single_user]
    beta_target_action_prob_func_args_single_user[action_prob_func_args_beta_index] = (
        beta_target
    )
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_user)

    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


def get_min_time_by_policy_num(
    single_user_policy_num_by_decision_time, beta_index_by_policy_num
):
    """
    Returns a dictionary mapping each policy number to the first time it was applicable,
    and the first time after the first update.
    """
    min_time_by_policy_num = {}
    first_time_after_first_update = None
    for decision_time, policy_num in single_user_policy_num_by_decision_time.items():
        if policy_num not in min_time_by_policy_num:
            min_time_by_policy_num[policy_num] = decision_time

        # Grab the first time where a non-initial, non-fallback policy is used.
        # Assumes single_user_policy_num_by_decision_time is sorted.
        if (
            policy_num in beta_index_by_policy_num
            and first_time_after_first_update is None
        ):
            first_time_after_first_update = decision_time

    return min_time_by_policy_num, first_time_after_first_update


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
    beta_index_by_policy_num: dict[int | float, int],
    initial_policy_num: int | float,
    action_by_decision_time_by_user_id: dict[collections.abc.Hashable, dict[int, int]],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    action_prob_func_args_by_user_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    update_func_args_by_by_user_id_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
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
            The filename of the inference loss or estimating function to be loaded.

        inference_func_type (str):
            The type of the inference function. FunctionTypes.LOSS or FunctionTypes.ESTIMATING.

        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference loss or estimating function arguments.
            This parameter must be present, so this should always be nonnegative.

        inference_func_args_action_prob_index (int):
            The index of the action probabilities argument in the inference loss or estimating
            function's arguments. -1 if not present.

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

        initial_policy_num (int | float): The policy number of the initial policy before any
            updates.

        action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to actions taken.

        policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A dictionary mapping decision times to the policy number in use.
            This may be user-specific. Should be sorted by decision time.

        action_prob_func_args_by_user_id_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for action probability function.
            This is for all decision times for all users (ars are an empty tuple if they are not in
            the study). Should be sorted by decision time.

        update_func_args_by_by_user_id_by_policy_num (dict[int | float, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from policy numbers to tuples containing the arguments for the algorithm
            loss or estimating functions when producing this policy.  This is for all non-initial,
            non-fallback policies. Should be sorted by policy number.

        inference_func_args_by_user_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A tuple containing the arguments for the inference loss or estimating function for this
            user.

        inference_action_prob_decision_times_by_user_id (dict[collections.abc.Hashable, list[int]]):
            For each user, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.

    Returns:
        callable: A function that computes a weighted estimating function stack for a single user
            (and some auxiliary values described in its docstring).
    """
    action_prob_func = load_function_from_same_named_file(action_prob_func_filename)
    alg_update_func = load_function_from_same_named_file(alg_update_func_filename)
    inference_func = load_function_from_same_named_file(inference_func_filename)

    algorithm_estimating_func = (
        jax.grad(alg_update_func, argnums=alg_update_func_args_beta_index)
        if (alg_update_func_type == FunctionTypes.LOSS)
        else alg_update_func
    )

    inference_estimating_func = (
        jax.grad(inference_func, argnums=inference_func_args_theta_index)
        if (inference_func_type == FunctionTypes.LOSS)
        else inference_func
    )

    # TODO: Break into smaller functions.
    def single_user_weighted_algorithm_estimating_function_stacker(
        theta: jnp.ndarray,
        all_post_update_betas: list[jnp.ndarray],
        user_id: collections.abc.Hashable,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Computes a weighted estimating function stack for a given set of algorithm update function
        arguments, and action probability function arguments.

        Args:
            theta (jnp.ndarray):
                The estimate of the parameter vector.

            all_post_update_betas (list[jnp.ndarray]):
                A list of 1D JAX NumPy arrays corresponding to the betas produced by all updates.

            user_id (collections.abc.Hashable):
                The user ID for which to compute the weighted estimating function stack.

        Returns:
            jnp.ndarray: A JAX NumPy array representing the weighted estimating function stack.
            jnp.ndarray: A JAX NumPy matrix representing the users' adaptive meat contribution.
            jnp.ndarray: A JAX NumPy matrix representing the users's classical meat contribution.
            jnp.ndarray: A JAX NumPy matrix representing the user's classical bread contribution.
        """

        logger.info(
            "Computing weighted estimating function stack for user %s.", user_id
        )

        # First, reformat the supplied data into more convienent structures.

        # 1. Form a dictionary mapping policy numbers to the first time they were
        # applicable (for this user). Note that this includes ALL policies, initial
        # fallbacks included.
        # Collect the first time after the first update separately for convenience.
        # These are both used to form the Radon-Nikodym weights for the right times.
        min_time_by_policy_num, first_time_after_first_update = (
            get_min_time_by_policy_num(
                policy_num_by_decision_time_by_user_id[user_id],
                beta_index_by_policy_num,
            )
        )

        # TODO: Can threading be done for all users at once somehow? Seems like yes, outside
        # of average over users.  Think about this while setting up vmap.

        # 2. Thread the central betas into the action probability arguments
        # for this particular user. This enables differentiation of the Radon-Nikodym
        # weights and action probabilities with respect to these shared betas.
        logger.info(
            "Threading in betas to action probability arguments for user %s.", user_id
        )
        threaded_single_user_action_prob_func_args_by_decision_time = (
            thread_action_prob_func_args(
                action_prob_func_args_by_user_id_by_decision_time,
                user_id,
                policy_num_by_decision_time_by_user_id,
                initial_policy_num,
                all_post_update_betas,
                beta_index_by_policy_num,
                action_prob_func_args_beta_index,
            )
        )

        # 3. Thread the central betas into the algorithm update function arguments
        # and replace any action probabilities with reconstructed ones from the above
        # arguments with the central betas introduced.
        logger.info(
            "Threading in betas and beta-dependent action probabilities to algorithm update "
            "function args for user %s.",
            user_id,
        )
        threaded_single_user_update_func_args_by_policy_num = thread_update_func_args(
            update_func_args_by_by_user_id_by_policy_num,
            user_id,
            all_post_update_betas,
            beta_index_by_policy_num,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            threaded_single_user_action_prob_func_args_by_decision_time,
            action_prob_func,
        )

        # 4. Thread the central betas into the inference function arguments
        # and replace any action probabilities with reconstructed ones from the above
        # arguments with the central betas introduced.
        logger.info(
            "Threading in theta and beta-dependent action probabilities to inference update "
            "function args for user %s.",
            user_id,
        )
        threaded_single_user_inference_func_args = thread_inference_func_args(
            user_id,
            inference_func_args_by_user_id,
            inference_func_args_theta_index,
            theta,
            inference_func_args_action_prob_index,
            threaded_single_user_action_prob_func_args_by_decision_time,
            inference_action_prob_decision_times_by_user_id,
            action_prob_func,
        )

        # 5. Get the start and end times for this user.
        logger.info("Calculating start and end times for user %s.", user_id)
        user_start_time = math.inf
        user_end_time = -math.inf
        for decision_time in action_by_decision_time_by_user_id[user_id]:
            user_start_time = min(user_start_time, decision_time)
            user_end_time = max(user_end_time, decision_time)

        # 6. Form a stack of weighted estimating equations, one for each update of the algorithm.
        logger.info(
            "Computing the algorithm component of the weighted estimating function stack for user %s.",
            user_id,
        )
        algorithm_component = jnp.concatenate(
            [
                # Here we compute a product of Radon-Nikodym weights
                # for all decision times after the first update and before the update
                # update under consideration took effect, for which the user was in the study.
                (
                    jnp.prod(
                        jnp.array(
                            [
                                # Note that we do NOT use the shared betas in the first arg, for
                                # which we don't want differentiation to happen with respect to.
                                # Just grab the original beta from the update function arguments.
                                # This is the same value, but impervious to differentiation with
                                # respect to all_post_update_betas. The args, on the other hand,
                                # are a function of all_post_update_betas.
                                get_radon_nikodym_weight(
                                    action_prob_func_args_by_user_id_by_decision_time[
                                        t
                                    ][user_id][action_prob_func_args_beta_index],
                                    action_prob_func,
                                    action_prob_func_args_beta_index,
                                    action_by_decision_time_by_user_id[user_id][t],
                                    *threaded_single_user_action_prob_func_args_by_decision_time[
                                        t
                                    ],
                                )
                                for t in range(
                                    # The earliest time after the first update where the user was in
                                    #  the study
                                    max(
                                        first_time_after_first_update,
                                        user_start_time,
                                    ),
                                    # The latest time the user was in the study before the time the
                                    # update under consideration first applied. Note the + 1 because
                                    # range does not include the right endpoint.
                                    # TODO: Is there any reason for the policy to not be in min time
                                    # by policy num? I can't think of one currently.
                                    min(
                                        min_time_by_policy_num.get(
                                            policy_num, math.inf
                                        ),
                                        user_end_time + 1,
                                    ),
                                )
                            ]
                        )
                    )  # Now use the above to weight the alg estimating function for this update
                    * algorithm_estimating_func(*update_args)
                    # If there are no arguments for the update function, the user is not yet in the
                    # study, so we just add a zero vector contribution to the sum across users.
                    # Note that after they exit, they still contribute all their data to later
                    # updates.
                    if update_args
                    else jnp.zeros(len(all_post_update_betas[0]))
                )
                for policy_num, update_args in threaded_single_user_update_func_args_by_policy_num.items()
            ]
        )
        # 7. Form the weighted inference estimating equation.
        logger.info(
            "Computing the inference component of the weighted estimating function stack for user %s.",
            user_id,
        )
        inference_component = jnp.prod(
            jnp.array(
                [
                    # Note: as above, the first arg is the original beta, not the shared one.
                    get_radon_nikodym_weight(
                        action_prob_func_args_by_user_id_by_decision_time[t][user_id][
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
        ) * inference_estimating_func(*threaded_single_user_inference_func_args)

        # 8. Concatenate the two components to form the weighted estimating function stack for this
        # user.
        weighted_stack = jnp.concatenate([algorithm_component, inference_component])
        # 9. Return the stack and auxiliary outputs described below.
        # Note the 4 outputs:
        #
        # 1. The first is simply the weighted estimating function stack for this user. The average
        # of these is what we differentiate with respect to theta to form the inverse adaptive bread
        # matrix, and we also compare that average to zero to check the estimating functions'
        # fidelity.
        # 2. The average outer product of these per-user stacks is the meat matrix, hence the second
        # output.
        # 3. The third output is averaged across users to obtain the classical meat matrix.
        # 4. The fourth output is averaged across users to obtatin the inverse classical bread
        # matrix.
        return (
            weighted_stack,
            jnp.outer(weighted_stack, weighted_stack),
            jnp.outer(inference_component, inference_component),
            jax.jacrev(
                inference_estimating_func, argnums=inference_func_args_theta_index
            )(*threaded_single_user_inference_func_args),
        )

    return single_user_weighted_algorithm_estimating_function_stacker


def thread_action_prob_func_args(
    action_prob_func_args_by_user_id_by_decision_time,
    user_id,
    policy_num_by_decision_time_by_user_id,
    initial_policy_num,
    all_post_update_betas,
    beta_index_by_policy_num,
    action_prob_func_args_beta_index,
):
    threaded_single_user_action_prob_func_args_by_decision_time = {}
    for (
        decision_time,
        action_prob_func_args_by_user_id,
    ) in action_prob_func_args_by_user_id_by_decision_time.items():
        if not action_prob_func_args_by_user_id[user_id]:
            threaded_single_user_action_prob_func_args_by_decision_time[
                decision_time
            ] = ()
            continue

        policy_num = policy_num_by_decision_time_by_user_id[user_id][decision_time]

        # The expectation is that fallback policies are empty, and the only other
        # policy not represented in beta_index_by_policy_num is the initial policy.
        # Specifically check for this to be a little more robust than simply checking
        # for the policy number in the dictionary.
        if policy_num == initial_policy_num:
            threaded_single_user_action_prob_func_args_by_decision_time[
                decision_time
            ] = action_prob_func_args_by_user_id[user_id]
            continue

        beta_to_introduce = all_post_update_betas[beta_index_by_policy_num[policy_num]]
        threaded_single_user_action_prob_func_args_by_decision_time[decision_time] = (
            replace_tuple_index(
                action_prob_func_args_by_user_id[user_id],
                action_prob_func_args_beta_index,
                beta_to_introduce,
            )
        )

    return threaded_single_user_action_prob_func_args_by_decision_time


def thread_update_func_args(
    update_func_args_by_by_user_id_by_policy_num,
    user_id,
    all_post_update_betas,
    beta_index_by_policy_num,
    alg_update_func_args_beta_index,
    alg_update_func_args_action_prob_index,
    alg_update_func_args_action_prob_times_index,
    threaded_single_user_action_prob_func_args_by_decision_time,
    action_prob_func,
):
    threaded_single_user_update_func_args_by_policy_num = {}
    for (
        policy_num,
        update_func_args_by_user_id,
    ) in update_func_args_by_by_user_id_by_policy_num.items():
        if not update_func_args_by_user_id[user_id]:
            threaded_single_user_update_func_args_by_policy_num[policy_num] = ()
            continue

        beta_to_introduce = all_post_update_betas[beta_index_by_policy_num[policy_num]]
        threaded_single_user_update_func_args_by_policy_num[policy_num] = (
            replace_tuple_index(
                update_func_args_by_user_id[user_id],
                alg_update_func_args_beta_index,
                beta_to_introduce,
            )
        )

        if alg_update_func_args_action_prob_index >= 0:
            action_prob_times = update_func_args_by_user_id[user_id][
                alg_update_func_args_action_prob_times_index
            ]
            action_probs_to_introduce = jnp.array(
                [
                    action_prob_func(
                        *threaded_single_user_action_prob_func_args_by_decision_time[t]
                    )
                    for t in action_prob_times.flatten().tolist()
                ]
            ).reshape(
                update_func_args_by_user_id[user_id][
                    alg_update_func_args_action_prob_index
                ].shape
            )
            threaded_single_user_update_func_args_by_policy_num[policy_num] = (
                replace_tuple_index(
                    threaded_single_user_update_func_args_by_policy_num[policy_num],
                    alg_update_func_args_action_prob_index,
                    action_probs_to_introduce,
                )
            )
    return threaded_single_user_update_func_args_by_policy_num


def thread_inference_func_args(
    user_id,
    inference_func_args_by_user_id,
    inference_func_args_theta_index,
    theta,
    inference_func_args_action_prob_index,
    threaded_single_user_action_prob_func_args_by_decision_time,
    inference_action_prob_decision_times_by_user_id,
    action_prob_func,
):
    single_user_inference_func_args = inference_func_args_by_user_id[user_id]

    threaded_single_user_inference_func_args = replace_tuple_index(
        single_user_inference_func_args,
        inference_func_args_theta_index,
        theta,
    )

    if inference_func_args_action_prob_index >= 0:
        action_probs_to_introduce = jnp.array(
            [
                action_prob_func(
                    *threaded_single_user_action_prob_func_args_by_decision_time[t]
                )
                for t in inference_action_prob_decision_times_by_user_id[user_id]
                .flatten()
                .tolist()
            ]
        ).reshape(
            single_user_inference_func_args[inference_func_args_action_prob_index].shape
        )
        threaded_single_user_inference_func_args = replace_tuple_index(
            threaded_single_user_inference_func_args,
            inference_func_args_action_prob_index,
            action_probs_to_introduce,
        )
    return threaded_single_user_inference_func_args


# TODO: vmap
def get_avg_weighted_estimating_function_stack_and_aux_values(
    all_post_update_betas_and_theta: list[jnp.ndarray],
    single_user_weighted_estimating_function_stacker: callable,
    user_ids: jnp.ndarray,
):
    results = [
        single_user_weighted_estimating_function_stacker(
            all_post_update_betas_and_theta[-1],
            all_post_update_betas_and_theta[:-1],
            user_id,
        )
        for user_id in user_ids.tolist()
    ]

    stacks = jnp.array([result[0] for result in results])
    outer_products = jnp.array([result[1] for result in results])
    inference_only_outer_products = jnp.array([result[2] for result in results])
    inference_hessians = jnp.array([result[3] for result in results])

    # Note this strange return structure! We will differentiate with respect to the first output,
    # but the second output will be passed along without modification via has_aux=True and then used
    # for the adaptive meat matrix, estimating functions sum check, and classical meat and inverse
    # bread matrices.
    return jnp.mean(stacks, axis=0), (
        jnp.mean(stacks, axis=0),
        jnp.mean(outer_products, axis=0),
        jnp.mean(inference_only_outer_products, axis=0),
        jnp.mean(inference_hessians, axis=0),
    )


def construct_classical_and_adaptive_inverse_bread_and_meat_and_avg_estimating_function_stack(
    single_user_weighted_estimating_function_stacker: callable,
    theta: jnp.ndarray,
    all_post_update_betas: jnp.ndarray,
    user_ids: jnp.ndarray,
):
    logger.info("Differentiating avg weighted estimating function stack.")
    # Interestingly, jax.jacobian does not seem to work here... just hangs in
    # the oralytics case, while it works fine in the simpler synthetic case.
    joint_adaptive_bread_inverse_pieces, (
        avg_estimating_function_stack,
        joint_adaptive_meat,
        classical_meat,
        classical_bread_inverse,
    ) = jax.jacrev(
        get_avg_weighted_estimating_function_stack_and_aux_values, has_aux=True
    )(
        # Note how this is a list of jnp arrays; it cannot be a jnp array itself
        # because theta and the betas need not be the same size.  But JAX can still
        # differentiate with respect to the all betas and thetas at once if they
        # are collected like so.
        list(all_post_update_betas) + [theta],
        single_user_weighted_estimating_function_stacker,
        user_ids,
    )

    # Stack the joint adaptive inverse bread pieces together horizontally and return the auxiliary values.
    # The bread will always be block lower triangular.  If this is not the case there
    # is an error (but it is almost certainly the package's fault, not the user's,
    # so no live check for this.)
    logger.info("Stacking bread pieces horizontally into full matrix.")
    return (
        jnp.hstack(joint_adaptive_bread_inverse_pieces),
        joint_adaptive_meat,
        classical_bread_inverse,
        classical_meat,
        avg_estimating_function_stack,
    )


# TODO: Docstring
def estimate_theta(study_df, theta_calculation_func_filename):
    logger.info("Forming theta estimate.")
    theta_calculation_func = load_function_from_same_named_file(
        theta_calculation_func_filename
    )

    return theta_calculation_func(study_df)


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

    print(f"\n(Mean) parameter estimate:\n{theta_estimate}")
    print(f"\nEmpirical variance:\n{empirical_var_normalized}")
    print(
        f"\nEmpirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):\n{approximate_standard_errors}"
    )
    print(
        f"\n(Mean) adaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\n(Mean) classical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}\n",
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


if __name__ == "__main__":
    cli()
