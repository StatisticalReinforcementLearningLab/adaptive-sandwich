from __future__ import annotations

import collections
import pathlib
import pickle
import os
import logging
import glob
import math
import sys
from typing import Any

import click
import jax
import numpy as np
from jax import numpy as jnp
import scipy
import pandas
import plotext as plt

from constants import FunctionTypes, SmallSampleCorrections
import input_checks
from helper_functions import (
    conditional_x_or_one_minus_x,
    get_action_prob_variance,
    get_in_study_df_column,
    invert_matrix_and_check_conditioning,
    load_function_from_same_named_file,
    replace_tuple_index,
    get_action_1_fraction,
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


# TODO: Check all help strings for accuracy.
# TODO: Deal with NA, -1, etc policy numbers
# TODO: Make sure in study is never on for more than one stretch EDIT: unclear if
# this will remain an invariant as we deal with more complicated data missingness
# TODO: I think I'm agnostic to indexing of calendar times but should check because
# otherwise need to add a check here to verify required format.
# TODO: Currently assuming function args can be placed in a numpy array. Must be scalar, 1d or 2d array.
# Higher dimensional objects not supported.  Not entirely sure what kind of "scalars" apply.
# TODO: Make the user give the min and max probabilities, and I'll enforce it?
@cli.command()
@click.option(
    "--study_df_pickle",
    type=click.File("rb"),
    help="Pickled pandas dataframe in correct format (see contract/readme).",
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
    help="Pickled dictionary that contains the action probability function arguments for all decision times for all users.",
    required=True,
)
@click.option(
    "--action_prob_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the algorithm parameter vector beta in the tuple of action probability func args.",
)
@click.option(
    "--alg_update_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-user update function used to determine the algorithm parameters at each update and relevant imports. May be a loss or estimating function, specified in a separate argument.  The filename without its extension will be assumed to match the function name.",
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
    help="Pickled dictionary that contains the algorithm update function arguments for all update times for all users.",
    required=True,
)
@click.option(
    "--alg_update_func_args_beta_index",
    type=int,
    required=True,
    help="Index of the algorithm parameter vector beta in the tuple of algorithm update func args.",
)
@click.option(
    "--alg_update_func_args_action_prob_index",
    type=int,
    default=-1000,
    help="Index of the action probability in the tuple of algorithm update func args, if applicable.",
)
@click.option(
    "--alg_update_func_args_action_prob_times_index",
    type=int,
    default=-1000,
    help="Index of the argument holding the decision times the action probabilities correspond to in the tuple of algorithm update func args, if applicable.",
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
    help="Index of the algorithm parameter vector beta in the tuple of inference loss/estimating func args.",
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
    help="Name of the binary column in the study dataframe that indicates whether a user is in the study.",
)
@click.option(
    "--action_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the study dataframe that indicates which action was taken.",
)
@click.option(
    "--policy_num_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that indicates the policy number in use.",
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
    help="Name of the column in the study dataframe that indicates user id.",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    required=True,
    help="Name of the column in the study dataframe that gives action one probabilities.",
)
@click.option(
    "--suppress_interactive_data_checks",
    type=bool,
    default=False,
    help="Flag to suppress any data checks that require user input. This is suitable for tests and large simulations",
)
@click.option(
    "--suppress_all_data_checks",
    type=bool,
    default=False,
    help="Flag to suppress all data checks. Not usually recommended, as suppressing only interactive checks suffices to keep tests/simulations running and is safer.",
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
def analyze_dataset(
    study_df_pickle: click.File,
    action_prob_func_filename: str,
    action_prob_func_args_pickle: click.File,
    action_prob_func_args_beta_index: int,
    alg_update_func_filename: str,
    alg_update_func_type: str,
    alg_update_func_args_pickle: click.File,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    inference_func_filename: str,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    theta_calculation_func_filename: str,
    in_study_col_name: str,
    action_col_name: str,
    policy_num_col_name: str,
    calendar_t_col_name: str,
    user_id_col_name: str,
    action_prob_col_name: str,
    suppress_interactive_data_checks: bool,
    suppress_all_data_checks: bool,
    small_sample_correction: str,
) -> None:
    """
    Analyzes a dataset to estimate parameters and variance using adaptive and classical sandwich estimators.

    Parameters:
    study_df_pickle (click.File):
        Pickle file containing the study DataFrame.
    action_prob_func_filename (str):
        Filename of the action probability function.
    action_prob_func_args_pickle (click.File):
        Pickle file containing arguments for the action probability function.
    action_prob_func_args_beta_index (int):
        Index for beta in action probability function arguments.
    alg_update_func_filename (str):
        Filename of the algorithm update function.
    alg_update_func_type (str):
        Type of the algorithm update function.
    alg_update_func_args_pickle (click.File):
        Pickle file containing arguments for the algorithm update function.
    alg_update_func_args_beta_index (int):
        Index for beta in algorithm update function arguments.
    alg_update_func_args_action_prob_index (int):
        Index for action probability in algorithm update function arguments.
    alg_update_func_args_action_prob_times_index (int):
        Index for action probability times in algorithm update function arguments.
    inference_func_filename (str):
        Filename of the inference function.
    inference_func_type (str):
        Type of the inference function.
    inference_func_args_theta_index (int):
        Index for theta in inference function arguments.
    theta_calculation_func_filename (str):
        Filename of the theta calculation function.
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
    suppress_interactive_data_checks (bool):
        Whether to suppress interactive data checks. This should be used in simulations, for example.
    suppress_all_data_checks (bool):
        Whether to suppress all data checks. Not recommended.
    small_sample_correction (str): Type of small sample correction to apply.

    Returns:
    None: The function writes analysis results and debug pieces to files in the same directory as the input files.
    """

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    study_df = pickle.load(study_df_pickle)
    action_prob_func_args = pickle.load(action_prob_func_args_pickle)
    alg_update_func_args = pickle.load(alg_update_func_args_pickle)

    theta_est = jnp.array(estimate_theta(study_df, theta_calculation_func_filename))

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
        )

    ### Begin collecting data structures that will be used to compute the joint bread matrix.

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

    # Use a per-user weighted estimating function stacking functino to derive classical and joint
    # adaptive meat and inverse bread matrices.  This is facilitated because the *value* of the
    # weighted and unweighted stacks are the same, as the weights evaluate to 1 pre-differentiation.
    logger.info(
        "Constructing joint adaptive bread inverse matrix, joint adaptive meat matrix, the classical analogs, and the avg estimating function stack across users."
    )
    user_ids = jnp.array(study_df[user_id_col_name].unique())
    # TODO: roadmap: vmap the derivatives of the above vectors over users (if I can, shapes may differ...) and then average
    (
        joint_adaptive_bread_inverse_matrix,
        joint_adaptive_meat_matrix,
        classical_bread_inverse_matrix,
        classical_meat_matrix,
        avg_estimating_function_stack,
        all_per_user_estimating_function_stacks,
    ) = construct_classical_and_adaptive_inverse_bread_and_meat_and_avg_estimating_function_stack(
        theta_est,
        all_post_update_betas,
        user_ids,
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
        action_prob_func_args,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        alg_update_func_args,
        action_by_decision_time_by_user_id,
    )

    beta_dim = len(all_post_update_betas[0])
    theta_dim = len(theta_est)
    if not suppress_all_data_checks:
        input_checks.require_estimating_functions_sum_to_zero(
            avg_estimating_function_stack,
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks,
        )

    logger.info("Forming classical sandwich variance estimate...")
    classical_bread_matrix = invert_matrix_and_check_conditioning(
        classical_bread_inverse_matrix
    )[0]
    classical_sandwich_var_estimate = (
        classical_bread_matrix @ classical_meat_matrix @ classical_bread_matrix.T
    ) / len(user_ids)

    # TODO: Small sample correction?

    # TODO: decide whether to in fact scrap the structure-based inversion
    # TODO: Could inspect condition number of each of the diagonal matrices
    logger.info("Inverting joint bread inverse matrix...")
    joint_adaptive_bread_matrix, joint_adaptive_bread_inverse_cond = (
        invert_matrix_and_check_conditioning(joint_adaptive_bread_inverse_matrix)
    )

    if not suppress_all_data_checks:
        input_checks.require_adaptive_bread_inverse_is_true_inverse(
            joint_adaptive_bread_matrix,
            joint_adaptive_bread_inverse_matrix,
            suppress_interactive_data_checks,
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
        -theta_dim:, -theta_dim:
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
                "joint_bread_matrix": joint_adaptive_bread_matrix,
                "joint_meat_matrix": joint_adaptive_meat_matrix,
                "classical_bread_inverse_matrix": classical_bread_inverse_matrix,
                "classical_bread_matrix": classical_bread_matrix,
                "classical_meat_matrix": classical_meat_matrix,
                "all_estimating_function_stacks": all_per_user_estimating_function_stacks,
                "joint_bread_inverse_condition_number": joint_adaptive_bread_inverse_cond,
            },
            f,
        )

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdaptive sandwich variance estimate:\n {adaptive_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )


def construct_beta_index_by_policy_num_map(
    study_df: pandas.DataFrame, policy_num_col_name: str, in_study_col_name: str
) -> tuple[dict[int | float, int], int | float]:
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
    inference_func_filename: str,
    inference_func_args_theta_index: int,
    study_df: pandas.DataFrame,
    theta_est: jnp.ndarray,
    action_prob_col_name: str,
    calendar_t_col_name: str,
    user_id_col_name: str,
    in_study_col_name: str,
) -> tuple[dict[collections.abc.Hashable, tuple[Any, ...]], int]:
    """
    Collects the inference function arguments for each user.

    Note that theta and action probabilities, if present, will be replaced later
    so that the function can be differentiated with respect to shared versions
    of them.

    Args:
        inference_func_filename (str):
            The filename of the inference function to be loaded.
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function's arguments.
        study_df (pandas.DataFrame):
            The study DataFrame.
        theta_est (jnp.ndarray):
            The estimate of the parameter vector.
        action_prob_col_name (str):
            The name of the column in the study DataFrame that gives action probabilities.
        calendar_t_col_name (str):
            The name of the column in the study DataFrame that indicates calendar time.
        user_id_col_name (str):
            The name of the column in the study DataFrame that indicates user ID.
        in_study_col_name (str):
            The name of the binary column in the study DataFrame that indicates whether a user is in the study.
    Returns:
        tuple[dict[collections.abc.Hashable, tuple[Any, ...]], int, dict[collections.abc.Hashable, jnp.ndarray[int]]]:
            A tuple containing
                - the inference function arguments dictionary for each user
                - the index of the action probabilities argument
                - a dictionary mapping user IDs to the decision times to which action probabilities correspond
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


def get_radon_nikodym_weight(
    beta_target: jnp.ndarray[jnp.float32],
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    action: int,
    *action_prob_func_args_single_user: tuple[Any, ...],
):
    """
    Computes a ratio of action probabilities under two sets of algorithm parameters:
    in the denominator, beta_target is substituted in with the the rest of the supplied action
    probability function arguments, and in the numerator the original value is used.  The action
    actually taken at the relevant decision time is also supplied, which is used to determine
    whether to use action 1 probabilities or action 0 probabilities in the ratio.

    Even though in practice we call this in such a way that the beta value is the same in numerator
    and denominator, it is important to define the function this way so that differentiation, which
    is with respect to the numerator beta, is done correctly.

    Args:
        beta_target (jnp.ndarray[jnp.float32]):
            The beta value to use in the denominator. NOT involved in differentation!
        action_prob_func (callable):
            The function used to compute the probability of action 1 at a given decision time for
            a particular user given their state and the algorithm parameters.
        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.
        action (int):
            The actual taken action at the relevant decision time.
        *action_prob_func_args_single_user (tuple[Any, ...]):
            The arguments to the action probability function for the relevant user at this time.

    Returns:
        jnp.float32: The Radon-Nikodym weight.

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


def single_user_weighted_algorithm_estimating_function_stacker(
    beta_dim: int,
    user_id: collections.abc.Hashable,
    action_prob_func: callable,
    algorithm_estimating_func: callable,
    inference_estimating_func: callable,
    action_prob_func_args_beta_index: int,
    inference_func_args_theta_index: int,
    action_prob_func_args_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    threaded_action_prob_func_args_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    threaded_update_func_args_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
    ],
    threaded_inference_func_args: dict[collections.abc.Hashable, tuple[Any, ...]],
    policy_num_by_decision_time: dict[collections.abc.Hashable, dict[int, int | float]],
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

        inference_estimating_func (callable):
            The estimating function that corresponds to inference.

        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.

        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference loss or estimating function arguments.

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

        threaded_inference_func_args (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A tuple containing the arguments for the inference
            estimating function for this user, with the shared betas threaded in for differentiation.

        policy_num_by_decision_time (dict[collections.abc.Hashable, dict[int, int | float]]):
            A dictionary mapping decision times to the policy number in use. This may be user-specific.
            Should be sorted by decision time.

        action_by_decision_time (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to actions taken.

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

    Returns:
        jnp.ndarray: A 1-D JAX NumPy array representing the user's weighted estimating function
            stack.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the user's adaptive meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the user's classical meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the user's classical bread contribution.
    """

    logger.info("Computing weighted estimating function stack for user %s.", user_id)

    # First, reformat the supplied data into more convienent structures.

    # 1. Form a dictionary mapping policy numbers to the first time they were
    # applicable (for this user). Note that this includes ALL policies, initial
    # fallbacks included.
    # Collect the first time after the first update separately for convenience.
    # These are both used to form the Radon-Nikodym weights for the right times.
    min_time_by_policy_num, first_time_after_first_update = get_min_time_by_policy_num(
        policy_num_by_decision_time,
        beta_index_by_policy_num,
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
    # TODO: This loop could be vmapped to be much faster.
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
                                action_prob_func_args_by_decision_time[t][
                                    action_prob_func_args_beta_index
                                ],
                                action_prob_func,
                                action_prob_func_args_beta_index,
                                action_by_decision_time[t],
                                *threaded_action_prob_func_args_by_decision_time[t],
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
                                    min_time_by_policy_num.get(policy_num, math.inf),
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
                else jnp.zeros(beta_dim)
            )
            for policy_num, update_args in threaded_update_func_args_by_policy_num.items()
        ]
    )
    # 4. Form the weighted inference estimating equation.
    logger.info(
        "Computing the inference component of the weighted estimating function stack for user %s.",
        user_id,
    )
    inference_component = jnp.prod(
        jnp.array(
            [
                # Note: as above, the first arg is the original beta, not the shared one.
                get_radon_nikodym_weight(
                    action_prob_func_args_by_decision_time[t][
                        action_prob_func_args_beta_index
                    ],
                    action_prob_func,
                    action_prob_func_args_beta_index,
                    action_by_decision_time[t],
                    *threaded_action_prob_func_args_by_decision_time[t],
                )
                # Go from the first time for the user that is after the first
                # update to their last active time
                for t in range(
                    max(first_time_after_first_update, user_start_time),
                    user_end_time + 1,
                )
            ]
        )
    ) * inference_estimating_func(*threaded_inference_func_args)

    # 5. Concatenate the two components to form the weighted estimating function stack for this
    # user.
    weighted_stack = jnp.concatenate([algorithm_component, inference_component])

    # 6. Return the following outputs:
    # a. The first is simply the weighted estimating function stack for this user. The average
    # of these is what we differentiate with respect to theta to form the inverse adaptive joint
    # bread matrix, and we also compare that average to zero to check the estimating functions'
    # fidelity.
    # b. The average outer product of these per-user stacks across users is the adaptive joint meat
    # matrix, hence the second output.
    # c. The third output is averaged across users to obtain the classical meat matrix.
    # d. The fourth output is averaged across users to obtatin the inverse classical bread
    # matrix.
    return (
        weighted_stack,
        jnp.outer(weighted_stack, weighted_stack),
        jnp.outer(inference_component, inference_component),
        jax.jacrev(inference_estimating_func, argnums=inference_func_args_theta_index)(
            *threaded_inference_func_args
        ),
    )


def thread_action_prob_func_args(
    action_prob_func_args_by_user_id_by_decision_time: dict[
        int, dict[collections.abc.Hashable, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    all_post_update_betas: list[jnp.ndarray],
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

        all_post_update_betas (list[jnp.ndarray]):
            A list of beta values to be introduced into arguments to
            facilitate differentiation.  They will be the same value as what they replace, but this
            introduces direct dependence on the parameter we will differentiate with respect to.

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

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
    all_post_update_betas: list[jnp.ndarray],
    beta_index_by_policy_num: dict[int | float, int],
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
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

        all_post_update_betas (list[jnp.ndarray]):
            A list of beta values to be introduced into arguments to
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

            if alg_update_func_args_action_prob_index >= 0:
                action_prob_times = update_func_args_by_user_id[user_id][
                    alg_update_func_args_action_prob_times_index
                ]
                action_probs_to_introduce = jnp.array(
                    [
                        action_prob_func(
                            *threaded_action_prob_func_args_by_decision_time_by_user_id[
                                user_id
                            ][t]
                        )
                        for t in action_prob_times.flatten().tolist()
                    ]
                ).reshape(
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
            action_probs_to_introduce = jnp.array(
                [
                    action_prob_func(
                        *threaded_action_prob_func_args_by_decision_time_by_user_id[
                            user_id
                        ][t]
                    )
                    for t in inference_action_prob_decision_times_by_user_id[user_id]
                    .flatten()
                    .tolist()
                ]
            ).reshape(args[inference_func_args_action_prob_index].shape)
            threaded_inference_func_args_by_user_id[user_id] = replace_tuple_index(
                threaded_inference_func_args_by_user_id[user_id],
                inference_func_args_action_prob_index,
                action_probs_to_introduce,
            )
    return threaded_inference_func_args_by_user_id


# TODO: vmap
def get_avg_weighted_estimating_function_stack_and_aux_values(
    all_post_update_betas_and_theta: list[jnp.ndarray],
    user_ids: jnp.ndarray,
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
    action_prob_func_args_by_user_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_user_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_user_id: dict[
        collections.abc.Hashable, list[int]
    ],
    update_func_args_by_by_user_id_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
    ],
    action_by_decision_time_by_user_id: dict[collections.abc.Hashable, dict[int, int]],
) -> tuple[
    jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    """
    Computes the average of the weighted estimating function stacks for all users, along with
    auxiliary values used to construct the adaptive and classical sandwich variances.

    Args:
        all_post_update_betas_and_theta (list[jnp.ndarray]):
            A list of JAX NumPy arrays representing the betas produced by all updates and the
            theta value, in that order.
        user_ids (jnp.ndarray):
            A 1D JAX NumPy array of user IDs.
        action_prob_func_filename (str):
            The name of the file containing the action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        alg_update_func_filename (str):
            The name of the file containing the algorithm update function.
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
        inference_func_filename (str):
            The name of the file containing the inference function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of theta in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_user_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of user ids to the function arguments
            required to compute action probabilities for this user.
        policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of user ids to dictionaries mapping decision times to the policy number in use.
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_user_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping user IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_user_id (dict[collections.abc.Hashable, list[int]]):
            For each user, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.
        update_func_args_by_by_user_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
            A dictionary where keys are policy numbers and values are dictionaries mapping user IDs
            to their respective update function arguments.
        action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping user IDs to their respective actions taken at each decision time.

    Returns:
        jnp.ndarray:
            A 1D JAX NumPy array representing the average weighted estimating function stack.
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            A tuple containing the average weighted estimating function stack, the adaptive meat
            matrix, the classical meat matrix, the inverse classical bread matrix, and the raw
            per-user weighted estimating function stacks.
    """

    # 1. Collect the necessary function objects
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
        all_post_update_betas_and_theta[:-1],
        beta_index_by_policy_num,
        action_prob_func_args_beta_index,
    )

    # 3. Thread the central betas into the algorithm update function arguments
    # and replace any action probabilities with reconstructed ones from the above
    # arguments with the central betas introduced.
    logger.info(
        "Threading in betas and beta-dependent action probabilities to algorithm update "
        "function args for all users."
    )
    threaded_update_func_args_by_policy_num_by_user_id = thread_update_func_args(
        update_func_args_by_by_user_id_by_policy_num,
        all_post_update_betas_and_theta[:-1],
        beta_index_by_policy_num,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        threaded_action_prob_func_args_by_decision_time_by_user_id,
        action_prob_func,
    )

    # 4. Thread the central theta into the inference function arguments
    # and replace any action probabilities with reconstructed ones from the above
    # arguments with the central betas introduced.
    logger.info(
        "Threading in theta and beta-dependent action probabilities to inference update "
        "function args for all users"
    )
    threaded_inference_func_args_by_user_id = thread_inference_func_args(
        inference_func_args_by_user_id,
        inference_func_args_theta_index,
        all_post_update_betas_and_theta[-1],
        inference_func_args_action_prob_index,
        threaded_action_prob_func_args_by_decision_time_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        action_prob_func,
    )

    # 5. Now we can compute the average of the weighted estimating function stacks for all users
    # as well as collect related values used to construct the adaptive and classical
    # sandwich variances.
    results = [
        single_user_weighted_algorithm_estimating_function_stacker(
            len(all_post_update_betas_and_theta[0]),
            user_id,
            action_prob_func,
            algorithm_estimating_func,
            inference_estimating_func,
            action_prob_func_args_beta_index,
            inference_func_args_theta_index,
            action_prob_func_args_by_decision_time_by_user_id[user_id],
            threaded_action_prob_func_args_by_decision_time_by_user_id[user_id],
            threaded_update_func_args_by_policy_num_by_user_id[user_id],
            threaded_inference_func_args_by_user_id[user_id],
            policy_num_by_decision_time_by_user_id[user_id],
            action_by_decision_time_by_user_id[user_id],
            beta_index_by_policy_num,
        )
        for user_id in user_ids.tolist()
    ]

    stacks = jnp.array([result[0] for result in results])
    outer_products = jnp.array([result[1] for result in results])
    inference_only_outer_products = jnp.array([result[2] for result in results])
    inference_hessians = jnp.array([result[3] for result in results])

    # 6. Note this strange return structure! We will differentiate the first output,
    # but the second output will be passed along without modification via has_aux=True and then used
    # for the adaptive meat matrix, estimating functions sum check, and classical meat and inverse
    # bread matrices. The raw per-user stacks are also returned for debugging purposes.
    return jnp.mean(stacks, axis=0), (
        jnp.mean(stacks, axis=0),
        jnp.mean(outer_products, axis=0),
        jnp.mean(inference_only_outer_products, axis=0),
        jnp.mean(inference_hessians, axis=0),
        stacks,
    )


def construct_classical_and_adaptive_inverse_bread_and_meat_and_avg_estimating_function_stack(
    theta: jnp.ndarray,
    all_post_update_betas: jnp.ndarray,
    user_ids: jnp.ndarray,
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
    action_prob_func_args_by_user_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_user_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_user_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_user_id: dict[
        collections.abc.Hashable, list[int]
    ],
    update_func_args_by_by_user_id_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
    ],
    action_by_decision_time_by_user_id: dict[collections.abc.Hashable, dict[int, int]],
) -> tuple[
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
]:
    """
    Constructs the classical and adaptive inverse bread and meat matrices, as well as the average
    estimating function stack.
    This is done by computing and differentiating the average weighted estimating function stack
    with respect to the betas and theta, and then using the resulting Jacobian to compute the inverse bread and meat
    matrices.

    Args:
        theta (jnp.ndarray):
            A 1-D JAX NumPy array representing the parameter estimate for inference.
        all_post_update_betas (jnp.ndarray):
            A 2-D JAX NumPy array representing all parameter estimates for the algorithm updates.
        user_ids (jnp.ndarray):
            A 1-D JAX NumPy array holding all user IDs in the study.
        action_prob_func_filename (str):
            The name of the file containing the action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        alg_update_func_filename (str):
            The name of the file containing the algorithm update function.
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
        inference_func_filename (str):
            The name of the file containing the inference function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of theta in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_user_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of user ids to the function arguments
            required to compute action probabilities for this user.
        policy_num_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of user ids to dictionaries mapping decision times to the policy number in use.
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_user_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping user IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_user_id (dict[collections.abc.Hashable, list[int]]):
            For each user, a list of decision times to which action probabilities correspond if
            provided. Typically just in-study times if action probabilites are used in the inference
            loss or estimating function.
        update_func_args_by_by_user_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
            A dictionary where keys are policy numbers and values are dictionaries mapping user IDs
            to their respective update function arguments.
        action_by_decision_time_by_user_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping user IDs to their respective actions taken at each decision time.
    Returns:
        tuple[jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32]]:
            A tuple containing:
            - The joint adaptive inverse bread matrix.
            - The joint adaptive meat matrix.
            - The classical inverse bread matrix.
            - The classical meat matrix.
            - The average weighted estimating function stack.
            - All per-user weighted estimating function stacks.
    """
    logger.info(
        "Differentiating average weighted estimating function stack and collecting auxiliary values."
    )
    # jax.jacobian may perform worse here--seemed to hang indefinitely while jacrev is merely very
    # slow.
    joint_adaptive_bread_inverse_pieces, (
        avg_estimating_function_stack,
        joint_adaptive_meat,
        classical_meat,
        classical_bread_inverse,
        all_per_user_estimating_function_stacks,
    ) = jax.jacrev(
        get_avg_weighted_estimating_function_stack_and_aux_values, has_aux=True
    )(
        # Note how this is a list of jnp arrays; it cannot easily be a jnp array itself
        # because theta and the betas need not be the same size.  But JAX can still
        # differentiate with respect to all betas and thetas at once if they
        # are collected like so.
        list(all_post_update_betas) + [theta],
        user_ids,
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
        action_prob_func_args_by_user_id_by_decision_time,
        policy_num_by_decision_time_by_user_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_user_id,
        inference_action_prob_decision_times_by_user_id,
        update_func_args_by_by_user_id_by_policy_num,
        action_by_decision_time_by_user_id,
    )

    # Stack the joint adaptive inverse bread pieces together horizontally and return the auxiliary
    # values too. The joint adaptive bread inverse should always be block lower triangular.
    return (
        jnp.hstack(joint_adaptive_bread_inverse_pieces),
        joint_adaptive_meat,
        classical_bread_inverse,
        classical_meat,
        avg_estimating_function_stack,
        all_per_user_estimating_function_stacks,
    )


def estimate_theta(
    study_df: pandas.DataFrame, theta_calculation_func_filename: str
) -> jnp.ndarray[jnp.float32]:
    """
    Estimates theta using the provided study DataFrame and the specified theta calculation
    function.
    Args:
        study_df (pandas.DataFrame): The DataFrame containing the study data.
        theta_calculation_func_filename (str): The filename of the theta calculation function.
    Returns:
        jnp.ndarray[jnp.float32]: The estimated theta (1-D).
    """

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
@click.option("--num_users", type=int, required=True)
@click.option(
    "--index_to_check_ci_coverage",
    type=int,
    help="The index of the parameter to check confidence interval coverage for across runs.  If not provided, coverage will not be checked.",
)
@click.option(
    "--in_study_col_name",
    type=str,
    help="Name of the binary column in the study dataframe that indicates whether a user is in the study.",
)
@click.option(
    "--action_col_name",
    type=str,
    help="Name of the column in the study dataframe that indicates the action taken by the user.",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    help="Name of the column in the study dataframe that indicates the probability of taking action 1.",
)
def collect_existing_analyses(
    input_glob: str,
    num_users: int,
    index_to_check_ci_coverage: int,
    in_study_col_name: str,
    action_col_name: str,
    action_prob_col_name: str,
) -> None:
    """
    Collects existing analyses from the specified input glob and computes the mean parameter estimate,
    empirical variance, and adaptive/classical sandwich variance estimates.
    Optionally checks confidence interval coverage for a specified parameter index.

    Args:
        input_glob (str): The glob pattern to search for analysis files.
        num_users (int): The number of users in the study.
        index_to_check_ci_coverage (int, optional): The index of the parameter to check confidence
            interval coverage for. If not provided, coverage will not be checked.
        in_study_col_name (str, optional): The name of the column indicating whether a user is in
            the study.
        action_col_name (str, optional): The name of the column indicating the action taken by the
            user.
        action_prob_col_name (str, optional): The name of the column indicating the probability of
            taking action 1.
    """

    raw_theta_estimates = []
    raw_adaptive_sandwich_var_estimates = []
    raw_classical_sandwich_var_estimates = []
    all_debug_pieces = []
    study_dfs = []
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
        with open(filename.replace("analysis.pkl", "debug_pieces.pkl"), "rb") as f:
            all_debug_pieces.append(pickle.load(f))
        with open(filename.replace("analysis.pkl", "study_df.pkl"), "rb") as f:
            study_dfs.append(pandas.read_pickle(f))

    theta_estimates = np.array(raw_theta_estimates)
    adaptive_sandwich_var_estimates = np.array(raw_adaptive_sandwich_var_estimates)
    classical_sandwich_var_estimates = np.array(raw_classical_sandwich_var_estimates)

    mean_theta_estimate = np.mean(theta_estimates, axis=0)
    empirical_var_normalized = np.atleast_2d(np.cov(theta_estimates.T, ddof=1))

    mean_adaptive_sandwich_var_estimate = np.mean(
        adaptive_sandwich_var_estimates, axis=0
    )
    median_adaptive_sandwich_var_estimate = np.median(
        adaptive_sandwich_var_estimates, axis=0
    )
    adaptive_sandwich_var_estimate_std_deviations = np.sqrt(
        np.var(adaptive_sandwich_var_estimates, axis=0, ddof=1)
    )
    adaptive_sandwich_var_estimate_mins = np.min(
        adaptive_sandwich_var_estimates, axis=0
    )
    adaptive_sandwich_var_estimate_maxes = np.max(
        adaptive_sandwich_var_estimates, axis=0
    )

    mean_classical_sandwich_var_estimate = np.mean(
        classical_sandwich_var_estimates, axis=0
    )
    median_classical_sandwich_var_estimate = np.median(
        classical_sandwich_var_estimates, axis=0
    )
    classical_sandwich_var_estimate_std_deviations = np.sqrt(
        np.var(classical_sandwich_var_estimates, axis=0, ddof=1)
    )
    classical_sandwich_var_estimate_mins = np.min(
        classical_sandwich_var_estimates, axis=0
    )
    classical_sandwich_var_estimate_maxes = np.max(
        classical_sandwich_var_estimates, axis=0
    )

    # Calculate standard error (or corresponding variance) of variance estimate for each
    # component of theta.  This is done by finding an unbiased estimator of the standard
    # formula for the standard error of a variance from iid observations.
    # Population standard error formula: https://en.wikipedia.org/wiki/Variance
    # Unbiased estimator: https://stats.stackexchange.com/questions/307537/unbiased-estimator-of-the-variance-of-the-sample-variance
    theta_component_variance_std_errors = []
    for i in range(len(mean_theta_estimate)):
        component_estimates = [estimate[i] for estimate in theta_estimates]
        second_central_moment = scipy.stats.moment(component_estimates, moment=4)
        fourth_central_moment = scipy.stats.moment(component_estimates, moment=4)
        N = len(theta_estimates)
        theta_component_variance_std_errors.append(
            np.sqrt(
                N
                * (
                    ((N) ** 2 - 3) * (second_central_moment) ** 2
                    + ((N - 1) ** 2) * fourth_central_moment
                )
                / ((N - 3) * (N - 2) * ((N - 1) ** 2))
            )
        )

    approximate_standard_errors = np.empty_like(empirical_var_normalized)
    for i, j in np.ndindex(approximate_standard_errors.shape):
        approximate_standard_errors[i, j] = max(
            theta_component_variance_std_errors[i],
            theta_component_variance_std_errors[j],
        )

    print(f"\nMean parameter estimate:\n{mean_theta_estimate}")
    print(f"\nEmpirical variance of parameter estimates:\n{empirical_var_normalized}")
    print(
        f"\nEmpirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):\n{approximate_standard_errors}"
    )
    print(
        f"\nMean adaptive sandwich variance estimate:\n{mean_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nMean classical sandwich variance estimate:\n{mean_classical_sandwich_var_estimate}",
    )
    print(
        f"\nMedian adaptive sandwich variance estimate:\n{median_adaptive_sandwich_var_estimate}",
    )
    print(
        f"\nMedian classical sandwich variance estimate:\n{median_classical_sandwich_var_estimate}",
    )
    print(
        f"\nAdaptive sandwich variance estimate std errors from empirical:\n{(mean_adaptive_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nClassical sandwich variance estimate std errors from empirical:\n{(mean_classical_sandwich_var_estimate - empirical_var_normalized) / approximate_standard_errors}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise standard deviations:\n{adaptive_sandwich_var_estimate_std_deviations}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise standard deviations:\n{classical_sandwich_var_estimate_std_deviations}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise mins:\n{adaptive_sandwich_var_estimate_mins}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise mins:\n{classical_sandwich_var_estimate_mins}",
    )
    print(
        f"\nAdaptive sandwich variance estimate elementwise maxes:\n{adaptive_sandwich_var_estimate_maxes}",
    )
    print(
        f"\nClassical sandwich variance estimate elementwise maxes:\n{classical_sandwich_var_estimate_maxes}\n",
    )

    if theta_estimates[0].size == 1:
        index_to_check_ci_coverage = 0
    if index_to_check_ci_coverage is not None:
        # We take this to be the "true" value
        scalar_mean_theta = mean_theta_estimate[index_to_check_ci_coverage]
        diffs = np.abs(
            theta_estimates[:, index_to_check_ci_coverage] - scalar_mean_theta
        )

        adaptive_standard_errors = np.sqrt(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        classical_standard_errors = np.sqrt(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )
        NOMINAL_COVERAGE = 0.95
        UPPER_PERCENTILE = 1 - (1 - NOMINAL_COVERAGE) / 2

        adaptive_z_covers = (
            diffs < scipy.stats.norm.ppf(UPPER_PERCENTILE) * adaptive_standard_errors
        )
        classical_z_covers = (
            diffs < scipy.stats.norm.ppf(UPPER_PERCENTILE) * classical_standard_errors
        )

        adaptive_t_covers = (
            diffs
            < scipy.stats.t.ppf(UPPER_PERCENTILE, num_users - 1)
            * adaptive_standard_errors
        )
        classical_t_covers = (
            diffs
            < scipy.stats.t.ppf(UPPER_PERCENTILE, num_users - 1)
            * classical_standard_errors
        )

        print(
            f"\nAdaptive sandwich {NOMINAL_COVERAGE * 100}% standard normal CI coverage:\n{np.mean(adaptive_z_covers)}\n",
        )
        print(
            f"\nClassical sandwich {NOMINAL_COVERAGE * 100}% standard normal CI coverage:\n{np.mean(classical_z_covers)}\n",
        )
        print(
            f"\nAdaptive sandwich {NOMINAL_COVERAGE * 100}% t({num_users - 1}) CI coverage:\n{np.mean(adaptive_t_covers)}\n",
        )
        print(
            f"\nClassical sandwich {NOMINAL_COVERAGE * 100}% t({num_users - 1}) CI coverage:\n{np.mean(classical_t_covers)}\n",
        )

        print("\nNow examining stability.\n")

        condition_numbers = None
        if "joint_adaptive_bread_inverse_condition_number" in all_debug_pieces[0]:
            condition_numbers = [
                debug_pieces["joint_adaptive_bread_inverse_condition_number"]
                for debug_pieces in all_debug_pieces
            ]
        if "joint_adaptive_bread_inverse" in all_debug_pieces[0]:
            condition_numbers = [
                np.linalg.cond(debug_pieces["joint_adaptive_bread_inverse"])
                for debug_pieces in all_debug_pieces
            ]

        action_1_fractions = [
            get_action_1_fraction(study_df, in_study_col_name, action_col_name)
            for study_df in study_dfs
        ]

        action_prob_variances = [
            get_action_prob_variance(study_df, in_study_col_name, action_prob_col_name)
            for study_df in study_dfs
        ]

        # Make sure previous output is flushed and not cleared
        sys.stdout.flush()
        plt.clear_terminal(False)

        # Plot the theta estimates to see variation
        plt.clear_figure()
        plt.title(f"Index {index_to_check_ci_coverage} of Theta Estimates")
        plt.xlabel("Simulation Index")
        plt.ylabel("Theta Estimate")
        plt.scatter(
            theta_estimates[:, index_to_check_ci_coverage],
            color="blue",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(theta_estimates[:, index_to_check_ci_coverage]),
                max(1, len(theta_estimates[:, index_to_check_ci_coverage]) // 10),
            )
        )
        plt.show()

        # Plot the adaptive sandwich variance estimates to look for blowup
        plt.clear_figure()
        plt.title(f"Index {index_to_check_ci_coverage} of Adaptive Variance Estimates")
        plt.xlabel("Simulation Index")
        plt.ylabel("Adaptive Variance Estimate")
        plt.scatter(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="green",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    adaptive_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        adaptive_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.show()

        # Plot the classical sandwich variance estimates to look for blowup
        plt.clear_figure()
        plt.title(f"Index {index_to_check_ci_coverage} of Classical Variance Estimates")
        plt.xlabel("Simulation Index")
        plt.ylabel("Classical Variance Estimate")
        plt.scatter(
            classical_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="green",
        )
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(
                    classical_sandwich_var_estimates[
                        :, index_to_check_ci_coverage, index_to_check_ci_coverage
                    ]
                ),
                max(
                    1,
                    len(
                        classical_sandwich_var_estimates[
                            :, index_to_check_ci_coverage, index_to_check_ci_coverage
                        ]
                    )
                    // 10,
                ),
            )
        )
        plt.show()

        # Plot the classical sandwich variance estimates for the top 5% experiments ranked by adaptive variance estimate size
        num_top_experiments = max(1, len(adaptive_sandwich_var_estimates) * 5 // 100)
        top_indices = np.argsort(
            adaptive_sandwich_var_estimates[
                :, index_to_check_ci_coverage, index_to_check_ci_coverage
            ]
        )[-num_top_experiments:]

        top_classical_var_estimates = classical_sandwich_var_estimates[
            top_indices, index_to_check_ci_coverage, index_to_check_ci_coverage
        ]

        plt.clear_figure()
        plt.title(
            f"Classical Estimates vs. Median for Highest {num_top_experiments} *Adaptive* Estimates at Index {index_to_check_ci_coverage}"
        )
        plt.xlabel("Experiment Rank (by Adaptive Variance)")
        plt.ylabel("Classical Variance Estimate")
        plt.scatter(
            top_classical_var_estimates,
            color="orange",
        )
        plt.horizontal_line(
            median_classical_sandwich_var_estimate[
                index_to_check_ci_coverage, index_to_check_ci_coverage
            ],
            color="blue",
        )
        plt.xticks(range(1, num_top_experiments + 1, max(1, num_top_experiments // 10)))
        plt.show()

        if condition_numbers:
            # Plot all condition numbers
            plt.clear_figure()
            plt.title("Condition Numbers for All Simulations")
            plt.xlabel("Simulation Index")
            plt.ylabel("Condition Number")
            plt.scatter(condition_numbers, color="purple")
            plt.grid(True)
            plt.xticks(
                range(
                    0,
                    len(condition_numbers),
                    max(1, len(condition_numbers) // 10),
                )
            )
            plt.show()

            # Plot condition numbers for the top 5% of adaptive variance estimates
            top_condition_numbers = [condition_numbers[i] for i in top_indices]
            plt.clear_figure()
            plt.title(
                f"Condition Numbers for Top {num_top_experiments} Adaptive Variance Estimates"
            )
            plt.xlabel("Experiment Rank (by Adaptive Variance)")
            plt.ylabel("Condition Number")
            plt.scatter(top_condition_numbers, color="purple")
            plt.xticks(
                range(1, num_top_experiments + 1, max(1, num_top_experiments // 10))
            )
            plt.grid(True)
            plt.show()

        # Plot all action_1_fractions
        plt.clear_figure()
        plt.title("Action 1 Fractions for All Simulations")
        plt.xlabel("Simulation Index")
        plt.ylabel("Action 1 Fraction")
        plt.scatter(action_1_fractions, color="red")
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(action_1_fractions),
                max(1, len(action_1_fractions) // 10),
            )
        )
        plt.show()

        # Plot action_1_fractions for the top 5% of adaptive variance estimates
        top_action_1_fractions = [action_1_fractions[i] for i in top_indices]
        plt.clear_figure()
        plt.title(
            f"Action 1 Fractions for Top {num_top_experiments} Adaptive Variance Estimates"
        )
        plt.xlabel("Experiment Rank (by Adaptive Variance)")
        plt.ylabel("Action 1 Fraction")
        plt.scatter(top_action_1_fractions, color="red")
        plt.xticks(range(1, num_top_experiments + 1, max(1, num_top_experiments // 10)))
        plt.grid(True)
        plt.show()

        # Plot all action probability variances
        plt.clear_figure()
        plt.title("Action Probability Variances for All Simulations")
        plt.xlabel("Simulation Index")
        plt.ylabel("Action Probability Variance")
        plt.scatter(action_prob_variances, color="blue")
        plt.grid(True)
        plt.xticks(
            range(
                0,
                len(action_prob_variances),
                max(1, len(action_prob_variances) // 10),
            )
        )
        plt.show()

        # Plot action probability variances for the top 5% of adaptive variance estimates
        top_action_prob_variances = [action_prob_variances[i] for i in top_indices]
        plt.clear_figure()
        plt.title(
            f"Action Probability Variances for Top {num_top_experiments} Adaptive Variance Estimates"
        )
        plt.xlabel("Experiment Rank (by Adaptive Variance)")
        plt.ylabel("Action Probability Variance")
        plt.scatter(top_action_prob_variances, color="blue")
        plt.xticks(range(1, num_top_experiments + 1, max(1, num_top_experiments // 10)))
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    cli()
