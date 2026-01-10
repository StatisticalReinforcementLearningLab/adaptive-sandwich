from __future__ import annotations

import collections
import pathlib
import pickle
import logging
import math
from typing import Any, Callable

import click
import jax
import numpy as np
from jax import numpy as jnp
import scipy
import pandas as pd

from .arg_threading_helpers import (
    thread_action_prob_func_args,
    thread_inference_func_args,
    thread_update_func_args,
)
from .constants import (
    FunctionTypes,
    SandwichFormationMethods,
    SmallSampleCorrections,
)
from .form_adjusted_meat_adjustments_directly import (
    form_adjusted_meat_adjustments_directly,
)
from . import input_checks
from . import get_datum_for_blowup_supervised_learning
from .small_sample_corrections import perform_desired_small_sample_correction
from .vmap_helpers import stack_batched_arg_lists_into_tensors


from .helper_functions import (
    calculate_beta_dim,
    collect_all_post_update_betas,
    construct_beta_index_by_policy_num_map,
    extract_action_and_policy_by_decision_time_by_subject_id,
    flatten_params,
    get_active_df_column,
    get_min_time_by_policy_num,
    get_radon_nikodym_weight,
    load_function_from_same_named_file,
    unflatten_params,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


@click.group()
def cli():
    pass


# TODO: Check all help strings for accuracy.
# TODO: Deal with NA, -1, etc policy numbers
# TODO: Make sure in deployment is never on for more than one stretch EDIT: unclear if
# this will remain an invariant as we deal with more complicated data missingness
# TODO: I think I'm agnostic to indexing of calendar times but should check because
# otherwise need to add a check here to verify required format.
# TODO: Currently assuming function args can be placed in a numpy array. Must be scalar, 1d or 2d array.
# Higher dimensional objects not supported.  Not entirely sure what kind of "scalars" apply.
@cli.command(name="analyze")
@click.option(
    "--analysis_df_pickle",
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
    help="Pickled dictionary that contains the action probability function arguments for all decision times for all subjects.",
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
    help="File that contains the per-subject update function used to determine the algorithm parameters at each update and relevant imports. May be a loss or estimating function, specified in a separate argument.  The filename without its extension will be assumed to match the function name.",
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
    help="Pickled dictionary that contains the algorithm update function arguments for all update times for all subjects.",
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
    "--alg_update_func_args_previous_betas_index",
    type=int,
    default=-1000,
    help="Index of the previous betas array in the tuple of algorithm update func args, if applicable. Note that these are only post-update betas. Sometimes a beta_0 may be defined pre-update; this should not be in here.",
)
@click.option(
    "--inference_func_filename",
    type=click.Path(exists=True),
    help="File that contains the per-subject loss/estimating function used to determine the inference estimate and relevant imports.  The filename without its extension will be assumed to match the function name.",
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
    help="Path to file that allows one to actually calculate a theta estimate given the analysis dataframe only. One must supply either this or a precomputed theta estimate. The filename without its extension will be assumed to match the function name.",
    required=True,
)
@click.option(
    "--active_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the analysis dataframe that indicates whether a subject is in the deployment.",
)
@click.option(
    "--action_col_name",
    type=str,
    required=True,
    help="Name of the binary column in the analysis dataframe that indicates which action was taken.",
)
@click.option(
    "--policy_num_col_name",
    type=str,
    required=True,
    help="Name of the column in the analysis dataframe that indicates the policy number in use.",
)
@click.option(
    "--calendar_t_col_name",
    type=str,
    required=True,
    help="Name of the column in the analysis dataframe that indicates calendar time (shared integer index across subjects).",
)
@click.option(
    "--subject_id_col_name",
    type=str,
    required=True,
    help="Name of the column in the analysis dataframe that indicates subject id.",
)
@click.option(
    "--action_prob_col_name",
    type=str,
    required=True,
    help="Name of the column in the analysis dataframe that gives action one probabilities.",
)
@click.option(
    "--reward_col_name",
    type=str,
    required=True,
    help="Name of the column in the analysis dataframe that gives rewards.",
)
@click.option(
    "--suppress_interactive_data_checks",
    type=bool,
    default=False,
    help="Flag to suppress any data checks that require subject input. This is suitable for tests and large simulations",
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
            SmallSampleCorrections.NONE,
            SmallSampleCorrections.Z1theta,
            SmallSampleCorrections.Z2theta,
            SmallSampleCorrections.Z3theta,
        ]
    ),
    default=SmallSampleCorrections.NONE,
    help="Type of small sample correction to apply to the variance estimate",
)
@click.option(
    "--collect_data_for_blowup_supervised_learning",
    type=bool,
    default=False,
    help="Flag to collect data for supervised learning blowup detection. This will write a single datum and label to a file in the same directory as the input files.",
)
@click.option(
    "--form_adjusted_meat_adjustments_explicitly",
    type=bool,
    default=False,
    help="If True, explicitly forms the per-subject meat adjustments that differentiate the adjusted sandwich from the classical sandwich. This is for diagnostic purposes, as the adjusted sandwich is formed without doing this.",
)
@click.option(
    "--stabilize_joint_bread",
    type=bool,
    default=True,
    help="If True, stabilizes the joint bread matrix if it does not meet conditioning thresholds.",
)
def analyze_dataset_wrapper(**kwargs):
    """
    This function is a wrapper around analyze_dataset to facilitate command line use.

    From the command line, we will take pickles and filenames for Python objects.
    We unpickle/load files here for passing to the implementation function, which
    may also be called in its own right with in-memory objects.

    See analyze_dataset for the underlying details.

    Returns: None
    """

    # Pass along the folder the analysis dataframe is in as the output folder.
    # Do it now because we will be removing the analysis dataframe pickle from kwargs.
    kwargs["output_dir"] = pathlib.Path(
        kwargs["analysis_df_pickle"].name
    ).parent.resolve()

    # Unpickle pickles and replace those args in kwargs
    kwargs["analysis_df"] = pickle.load(kwargs["analysis_df_pickle"])
    kwargs["action_prob_func_args"] = pickle.load(
        kwargs["action_prob_func_args_pickle"]
    )
    kwargs["alg_update_func_args"] = pickle.load(kwargs["alg_update_func_args_pickle"])

    kwargs.pop("analysis_df_pickle")
    kwargs.pop("action_prob_func_args_pickle")
    kwargs.pop("alg_update_func_args_pickle")

    # Load functions from filenames and replace those args in kwargs
    kwargs["action_prob_func"] = load_function_from_same_named_file(
        kwargs["action_prob_func_filename"]
    )
    kwargs["alg_update_func"] = load_function_from_same_named_file(
        kwargs["alg_update_func_filename"]
    )
    kwargs["inference_func"] = load_function_from_same_named_file(
        kwargs["inference_func_filename"]
    )
    kwargs["theta_calculation_func"] = load_function_from_same_named_file(
        kwargs["theta_calculation_func_filename"]
    )

    kwargs.pop("action_prob_func_filename")
    kwargs.pop("alg_update_func_filename")
    kwargs.pop("inference_func_filename")
    kwargs.pop("theta_calculation_func_filename")

    analyze_dataset(**kwargs)


def analyze_dataset(
    output_dir: pathlib.Path | str,
    analysis_df: pd.DataFrame,
    action_prob_func: Callable,
    action_prob_func_args: dict[int, Any],
    action_prob_func_args_beta_index: int,
    alg_update_func: Callable,
    alg_update_func_type: str,
    alg_update_func_args: dict[int, Any],
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    alg_update_func_args_previous_betas_index: int,
    inference_func: Callable,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    theta_calculation_func: Callable[[pd.DataFrame], jnp.ndarray],
    active_col_name: str,
    action_col_name: str,
    policy_num_col_name: str,
    calendar_t_col_name: str,
    subject_id_col_name: str,
    action_prob_col_name: str,
    reward_col_name: str,
    suppress_interactive_data_checks: bool,
    suppress_all_data_checks: bool,
    small_sample_correction: str,
    collect_data_for_blowup_supervised_learning: bool,
    form_adjusted_meat_adjustments_explicitly: bool,
    stabilize_joint_bread: bool,
) -> None:
    """
    Analyzes a dataset to provide a parameter estimate and an estimate of its variance using  and classical sandwich estimators.

    There are two modes of use for this function.

    First, it may be called indirectly from the command line by passing through
    analyze_dataset_wrapper.

    Second, it may be called directly from Python code with in-memory objects.

    Parameters:
    output_dir (pathlib.Path | str):
        Directory in which to save output files.
    analysis_df (pd.DataFrame):
        DataFrame containing the deployment data.
    action_prob_func (callable):
        Action probability function.
    action_prob_func_args (dict[int, Any]):
        Arguments for the action probability function.
    action_prob_func_args_beta_index (int):
        Index for beta in action probability function arguments.
    alg_update_func (callable):
        Algorithm update function.
    alg_update_func_type (str):
        Type of the algorithm update function.
    alg_update_func_args (dict[int, Any]):
        Arguments for the algorithm update function.
    alg_update_func_args_beta_index (int):
        Index for beta in algorithm update function arguments.
    alg_update_func_args_action_prob_index (int):
        Index for action probability in algorithm update function arguments.
    alg_update_func_args_action_prob_times_index (int):
        Index for action probability times in algorithm update function arguments.
    inference_func (callable):
        Inference loss or estimating function.
    inference_func_type (str):
        Type of the inference function.
    inference_func_args_theta_index (int):
        Index for theta in inference function arguments.
    theta_calculation_func (callable):
        Function to estimate theta from the analysis dataframe.
    active_col_name (str):
        Column name indicating if a subject is active in the analysis dataframe.
    action_col_name (str):
        Column name for actions in the analysis dataframe.
    policy_num_col_name (str):
        Column name for policy numbers in the analysis dataframe.
    calendar_t_col_name (str):
        Column name for calendar time in the analysis dataframe.
    subject_id_col_name (str):
        Column name for subject IDs in the analysis dataframe.
    action_prob_col_name (str):
        Column name for action probabilities in the analysis dataframe.
    reward_col_name (str):
        Column name for rewards in the analysis dataframe.
    suppress_interactive_data_checks (bool):
        Whether to suppress interactive data checks. This should be used in simulations, for example.
    suppress_all_data_checks (bool):
        Whether to suppress all data checks. Not recommended.
    small_sample_correction (str):
        Type of small sample correction to apply.
    collect_data_for_blowup_supervised_learning (bool):
        Whether to collect data for doing supervised learning about adjusted sandwich blowup.
    form_adjusted_meat_adjustments_explicitly (bool):
        If True, explicitly forms the per-subject meat adjustments that differentiate the
        sandwich from the classical sandwich. This is for diagnostic purposes, as the
        adjusted sandwich is formed without doing this.
    stabilize_joint_bread (bool):
        If True, stabilizes the joint bread matrix if it does not meet conditioning
        thresholds.

    Returns:
    dict: A dictionary containing the theta estimate, adjusted sandwich variance estimate, and
    classical sandwich variance estimate.
    """

    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )

    theta_est = jnp.array(theta_calculation_func(analysis_df))

    beta_dim = calculate_beta_dim(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    if not suppress_all_data_checks:
        input_checks.perform_first_wave_input_checks(
            analysis_df,
            active_col_name,
            action_col_name,
            policy_num_col_name,
            calendar_t_col_name,
            subject_id_col_name,
            action_prob_col_name,
            reward_col_name,
            action_prob_func,
            action_prob_func_args,
            action_prob_func_args_beta_index,
            alg_update_func_args,
            alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index,
            alg_update_func_args_action_prob_times_index,
            alg_update_func_args_previous_betas_index,
            theta_est,
            beta_dim,
            suppress_interactive_data_checks,
            small_sample_correction,
        )

    ### Begin collecting data structures that will be used to compute the joint bread matrix.
    beta_index_by_policy_num, initial_policy_num = (
        construct_beta_index_by_policy_num_map(
            analysis_df, policy_num_col_name, active_col_name
        )
    )

    all_post_update_betas = collect_all_post_update_betas(
        beta_index_by_policy_num, alg_update_func_args, alg_update_func_args_beta_index
    )

    action_by_decision_time_by_subject_id, policy_num_by_decision_time_by_subject_id = (
        extract_action_and_policy_by_decision_time_by_subject_id(
            analysis_df,
            subject_id_col_name,
            active_col_name,
            calendar_t_col_name,
            action_col_name,
            policy_num_col_name,
        )
    )

    (
        inference_func_args_by_subject_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_subject_id,
    ) = process_inference_func_args(
        inference_func,
        inference_func_args_theta_index,
        analysis_df,
        theta_est,
        action_prob_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        active_col_name,
    )

    # Use a per-subject weighted estimating function stacking function to derive classical and joint
    # meat and bread matrices.  This is facilitated because the *value* of the
    # weighted and unweighted stacks are the same, as the weights evaluate to 1 pre-differentiation.
    logger.info(
        "Constructing joint bread matrix, joint meat matrix, the classical analogs, and the avg estimating function stack across subjects."
    )

    subject_ids = jnp.array(analysis_df[subject_id_col_name].unique())
    (
        stabilized_joint_adjusted_bread_matrix,
        raw_joint_adjusted_bread_matrix,
        joint_adjusted_meat_matrix,
        joint_adjusted_sandwich_matrix,
        classical_bread_matrix,
        classical_meat_matrix,
        classical_sandwich_var_estimate,
        avg_estimating_function_stack,
        per_subject_estimating_function_stacks,
        per_subject_adjusted_corrections,
        per_subject_classical_corrections,
        per_subject_adjusted_meat_adjustments,
    ) = construct_classical_and_adjusted_sandwiches(
        theta_est,
        all_post_update_betas,
        subject_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args,
        policy_num_by_decision_time_by_subject_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_subject_id,
        inference_action_prob_decision_times_by_subject_id,
        alg_update_func_args,
        action_by_decision_time_by_subject_id,
        suppress_all_data_checks,
        suppress_interactive_data_checks,
        small_sample_correction,
        form_adjusted_meat_adjustments_explicitly,
        stabilize_joint_bread,
        analysis_df,
        active_col_name,
        action_col_name,
        calendar_t_col_name,
        subject_id_col_name,
        action_prob_func_args,
        action_prob_col_name,
    )

    theta_dim = len(theta_est)
    if not suppress_all_data_checks:
        input_checks.require_estimating_functions_sum_to_zero(
            avg_estimating_function_stack,
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks,
        )

    # This bottom right corner of the joint (betas and theta) variance matrix is the portion
    # corresponding to just theta.
    adjusted_sandwich_var_estimate = joint_adjusted_sandwich_matrix[
        -theta_dim:, -theta_dim:
    ]

    # Check for negative diagonal elements and set them to zero if found
    adjusted_diagonal = np.diag(adjusted_sandwich_var_estimate)
    if np.any(adjusted_diagonal < 0):
        logger.warning(
            "Found negative diagonal elements in adjusted sandwich variance estimate. Setting them to zero."
        )
        np.fill_diagonal(
            adjusted_sandwich_var_estimate, np.maximum(adjusted_diagonal, 0)
        )

    logger.info("Writing results to file...")
    output_folder_abs_path = pathlib.Path(output_dir).resolve()

    analysis_dict = {
        "theta_est": theta_est,
        "adjusted_sandwich_var_estimate": adjusted_sandwich_var_estimate,
        "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
    }
    with open(output_folder_abs_path / "analysis.pkl", "wb") as f:
        pickle.dump(
            analysis_dict,
            f,
        )

    joint_adjusted_bread_cond = jnp.linalg.cond(raw_joint_adjusted_bread_matrix)
    logger.info(
        "Joint adjusted bread condition number: %f",
        joint_adjusted_bread_cond,
    )

    # calculate the max eigenvalue of the joint adjusted sandwich
    max_eigenvalue = scipy.linalg.eigvalsh(joint_adjusted_sandwich_matrix).max()
    logger.info(
        "Max eigenvalue of joint adjusted sandwich matrix: %f",
        max_eigenvalue,
    )

    debug_pieces_dict = {
        "theta_est": theta_est,
        "adjusted_sandwich_var_estimate": adjusted_sandwich_var_estimate,
        "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
        "raw_joint_bread_matrix": raw_joint_adjusted_bread_matrix,
        "stabilized_joint_bread_matrix": stabilized_joint_adjusted_bread_matrix,
        "joint_meat_matrix": joint_adjusted_meat_matrix,
        "classical_bread_matrix": classical_bread_matrix,
        "classical_meat_matrix": classical_meat_matrix,
        "all_estimating_function_stacks": per_subject_estimating_function_stacks,
        "joint_bread_condition_number": joint_adjusted_bread_cond,
        "max_eigenvalue_joint_adjusted_sandwich": max_eigenvalue,
        "all_post_update_betas": all_post_update_betas,
        "per_subject_adjusted_corrections": per_subject_adjusted_corrections,
        "per_subject_classical_corrections": per_subject_classical_corrections,
        "per_subject_adjusted_meat_adjustments": per_subject_adjusted_meat_adjustments,
    }
    with open(output_folder_abs_path / "debug_pieces.pkl", "wb") as f:
        pickle.dump(
            debug_pieces_dict,
            f,
        )

    if collect_data_for_blowup_supervised_learning:
        datum_and_label_dict = get_datum_for_blowup_supervised_learning.get_datum_for_blowup_supervised_learning(
            raw_joint_adjusted_bread_matrix,
            joint_adjusted_bread_cond,
            avg_estimating_function_stack,
            per_subject_estimating_function_stacks,
            all_post_update_betas,
            analysis_df,
            active_col_name,
            calendar_t_col_name,
            action_prob_col_name,
            subject_id_col_name,
            reward_col_name,
            theta_est,
            adjusted_sandwich_var_estimate,
            subject_ids,
            beta_dim,
            theta_dim,
            initial_policy_num,
            beta_index_by_policy_num,
            policy_num_by_decision_time_by_subject_id,
            theta_calculation_func,
            action_prob_func,
            action_prob_func_args_beta_index,
            inference_func,
            inference_func_type,
            inference_func_args_theta_index,
            inference_func_args_action_prob_index,
            inference_action_prob_decision_times_by_subject_id,
            action_prob_func_args,
            action_by_decision_time_by_subject_id,
        )

        with open(output_folder_abs_path / "supervised_learning_datum.pkl", "wb") as f:
            pickle.dump(datum_and_label_dict, f)

    print(f"\nParameter estimate:\n {theta_est}")
    print(f"\nAdjusted sandwich variance estimate:\n {adjusted_sandwich_var_estimate}")
    print(
        f"\nClassical sandwich variance estimate:\n {classical_sandwich_var_estimate}\n"
    )

    return analysis_dict


def process_inference_func_args(
    inference_func: callable,
    inference_func_args_theta_index: int,
    analysis_df: pd.DataFrame,
    theta_est: jnp.ndarray,
    action_prob_col_name: str,
    calendar_t_col_name: str,
    subject_id_col_name: str,
    active_col_name: str,
) -> tuple[dict[collections.abc.Hashable, tuple[Any, ...]], int]:
    """
    Collects the inference function arguments for each subject from the analysis DataFrame.

    Note that theta and action probabilities, if present, will be replaced later
    so that the function can be differentiated with respect to shared versions
    of them.

    Args:
        inference_func (callable):
            The inference function to be used.
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function's arguments.
        analysis_df (pandas.DataFrame):
            The analysis DataFrame.
        theta_est (jnp.ndarray):
            The estimate of the parameter vector.
        action_prob_col_name (str):
            The name of the column in the analysis DataFrame that gives action probabilities.
        calendar_t_col_name (str):
            The name of the column in the analysis DataFrame that indicates calendar time.
        subject_id_col_name (str):
            The name of the column in the analysis DataFrame that indicates subject ID.
        active_col_name (str):
            The name of the binary column in the analysis DataFrame that indicates whether a subject is in the deployment.
    Returns:
        tuple[dict[collections.abc.Hashable, tuple[Any, ...]], int, dict[collections.abc.Hashable, jnp.ndarray[int]]]:
            A tuple containing
                - the inference function arguments dictionary for each subject
                - the index of the action probabilities argument
                - a dictionary mapping subject IDs to the decision times to which action probabilities correspond
    """

    num_args = inference_func.__code__.co_argcount
    inference_func_arg_names = inference_func.__code__.co_varnames[:num_args]
    inference_func_args_by_subject_id = {}

    inference_func_args_action_prob_index = -1
    inference_action_prob_decision_times_by_subject_id = {}

    using_action_probs = action_prob_col_name in inference_func_arg_names
    if using_action_probs:
        inference_func_args_action_prob_index = inference_func_arg_names.index(
            action_prob_col_name
        )

    for subject_id in analysis_df[subject_id_col_name].unique():
        subject_args_list = []
        filtered_subject_data = analysis_df.loc[
            analysis_df[subject_id_col_name] == subject_id
        ]
        for idx, col_name in enumerate(inference_func_arg_names):
            if idx == inference_func_args_theta_index:
                subject_args_list.append(theta_est)
                continue
            subject_args_list.append(
                get_active_df_column(filtered_subject_data, col_name, active_col_name)
            )
        inference_func_args_by_subject_id[subject_id] = tuple(subject_args_list)
        if using_action_probs:
            inference_action_prob_decision_times_by_subject_id[subject_id] = (
                get_active_df_column(
                    filtered_subject_data, calendar_t_col_name, active_col_name
                )
            )

    return (
        inference_func_args_by_subject_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_subject_id,
    )


def single_subject_weighted_estimating_function_stacker(
    beta_dim: int,
    subject_id: collections.abc.Hashable,
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

        subject_id (collections.abc.Hashable):
            The subject ID for which to compute the weighted estimating function stack.

        action_prob_func (callable):
            The function used to compute the probability of action 1 at a given decision time for
            a particular subject given their state and the algorithm parameters.

        algorithm_estimating_func (callable):
            The estimating function that corresponds to algorithm updates.

        inference_estimating_func (callable):
            The estimating function that corresponds to inference.

        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.

        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference loss or estimating function arguments.

        action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for this subject for the action
            probability function. This is for all decision times (args are an empty
            tuple if they are not in the deployment). Should be sorted by decision time. NOTE THAT THESE
            ARGS DO NOT CONTAIN THE SHARED BETAS, making them impervious to the differentiation that
            will occur.

        threaded_action_prob_func_args_by_decision_time (dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from decision times to tuples of arguments for the action
            probability function, with the shared betas threaded in for differentation. Decision
            times should be sorted.

        threaded_update_func_args_by_policy_num (dict[int | float, dict[collections.abc.Hashable, tuple[Any, ...]]]):
            A map from policy numbers to tuples containing the arguments for
            the corresponding estimating functions for this subject, with the shared betas threaded in
            for differentiation.  This is for all non-initial, non-fallback policies. Policy numbers
            should be sorted.

        threaded_inference_func_args (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A tuple containing the arguments for the inference
            estimating function for this subject, with the shared betas threaded in for differentiation.

        policy_num_by_decision_time (dict[collections.abc.Hashable, dict[int, int | float]]):
            A dictionary mapping decision times to the policy number in use. This may be
            subject-specific. Should be sorted by decision time. Only applies to active decision
            times!

        action_by_decision_time (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping decision times to actions taken. Only applies to active decision
            times!

        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.

    Returns:
        jnp.ndarray: A 1-D JAX NumPy array representing the subject's weighted estimating function
            stack.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's adjusted meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical bread contribution.
    """

    logger.info(
        "Computing weighted estimating function stack for subject %s.", subject_id
    )

    # First, reformat the supplied data into more convenient structures.

    # 1. Form a dictionary mapping policy numbers to the first time they were
    # applicable (for this subject). Note that this includes ALL policies, initial
    # fallbacks included.
    # Collect the first time after the first update separately for convenience.
    # These are both used to form the Radon-Nikodym weights for the right times.
    min_time_by_policy_num, first_time_after_first_update = get_min_time_by_policy_num(
        policy_num_by_decision_time,
        beta_index_by_policy_num,
    )

    # 2. Get the start and end times for this subject.
    subject_start_time = math.inf
    subject_end_time = -math.inf
    for decision_time in action_by_decision_time:
        subject_start_time = min(subject_start_time, decision_time)
        subject_end_time = max(subject_end_time, decision_time)

    # 3. Form a stack of weighted estimating equations, one for each update of the algorithm.
    logger.info(
        "Computing the algorithm component of the weighted estimating function stack for subject %s.",
        subject_id,
    )

    active_action_prob_func_args = [
        args for args in action_prob_func_args_by_decision_time.values() if args
    ]
    active_betas_list_by_decision_time_index = jnp.array(
        [
            action_prob_func_args[action_prob_func_args_beta_index]
            for action_prob_func_args in active_action_prob_func_args
        ]
    )
    active_actions_list_by_decision_time_index = jnp.array(
        list(action_by_decision_time.values())
    )

    # Sort the threaded args by decision time to be cautious. We check if the
    # subject id is present in the subject args dict because we may call this on a
    # subset of the subject arg dict when we are batching arguments by shape
    sorted_threaded_action_prob_args_by_decision_time = {
        decision_time: threaded_action_prob_func_args_by_decision_time[decision_time]
        for decision_time in range(subject_start_time, subject_end_time + 1)
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
    active_weights = jax.vmap(
        fun=get_radon_nikodym_weight,
        in_axes=[0, None, None, 0] + batch_axes,
        out_axes=0,
    )(
        active_betas_list_by_decision_time_index,
        action_prob_func,
        action_prob_func_args_beta_index,
        active_actions_list_by_decision_time_index,
        *batched_threaded_arg_tensors,
    )

    active_index = 0
    decision_time_to_all_weights_index_offset = min(
        sorted_threaded_action_prob_args_by_decision_time
    )
    all_weights_raw = []
    for (
        decision_time,
        args,
    ) in sorted_threaded_action_prob_args_by_decision_time.items():
        all_weights_raw.append(active_weights[active_index] if args else 1.0)
        active_index += 1
    all_weights = jnp.array(all_weights_raw)

    algorithm_component = jnp.concatenate(
        [
            # Here we compute a product of Radon-Nikodym weights
            # for all decision times after the first update and before the update
            # update under consideration took effect, for which the subject was in the deployment.
            (
                jnp.prod(
                    all_weights[
                        # The earliest time after the first update where the subject was in
                        # the deployment
                        max(
                            first_time_after_first_update,
                            subject_start_time,
                        )
                        - decision_time_to_all_weights_index_offset :
                        # One more than the latest time the subject was in the deployment before the time
                        # the update under consideration first applied. Note the + 1 because range
                        # does not include the right endpoint.
                        min(
                            min_time_by_policy_num.get(policy_num, math.inf),
                            subject_end_time + 1,
                        )
                        - decision_time_to_all_weights_index_offset,
                    ]
                    # If the subject exited the deployment before there were any updates,
                    # this variable will be None and the above code to grab a weight would
                    # throw an error. Just use 1 to include the unweighted estimating function
                    # if they have data to contribute to the update.
                    if first_time_after_first_update is not None
                    else 1
                )  # Now use the above to weight the alg estimating function for this update
                * algorithm_estimating_func(*update_args)
                # If there are no arguments for the update function, the subject is not yet in the
                # deployment, so we just add a zero vector contribution to the sum across subjects.
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
    # 4. Form the weighted inference estimating equation.
    logger.info(
        "Computing the inference component of the weighted estimating function stack for subject %s.",
        subject_id,
    )
    inference_component = jnp.prod(
        all_weights[
            max(first_time_after_first_update, subject_start_time)
            - decision_time_to_all_weights_index_offset : subject_end_time
            + 1
            - decision_time_to_all_weights_index_offset,
        ]
        # If the subject exited the deployment before there were any updates,
        # this variable will be None and the above code to grab a weight would
        # throw an error. Just use 1 to include the unweighted estimating function
        # if they have data to contribute here (pretty sure everyone should?)
        if first_time_after_first_update is not None
        else 1
    ) * inference_estimating_func(*threaded_inference_func_args)

    # 5. Concatenate the two components to form the weighted estimating function stack for this
    # subject.
    weighted_stack = jnp.concatenate([algorithm_component, inference_component])

    # 6. Return the following outputs:
    # a. The first is simply the weighted estimating function stack for this subject. The average
    # of these is what we differentiate with respect to theta to form the joint
    # bread matrix, and we also compare that average to zero to check the estimating functions'
    # fidelity.
    # b. The average outer product of these per-subject stacks across subjects is the adjusted joint meat
    # matrix, hence the second output.
    # c. The third output is averaged across subjects to obtain the classical meat matrix.
    # d. The fourth output is averaged across subjects to obtain the inverse classical bread
    # matrix.
    return (
        weighted_stack,
        jnp.outer(weighted_stack, weighted_stack),
        jnp.outer(inference_component, inference_component),
        jax.jacrev(inference_estimating_func, argnums=inference_func_args_theta_index)(
            *threaded_inference_func_args
        ),
    )


def get_avg_weighted_estimating_function_stacks_and_aux_values(
    flattened_betas_and_theta: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    subject_ids: jnp.ndarray,
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    alg_update_func: callable,
    alg_update_func_type: str,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    alg_update_func_args_previous_betas_index: int,
    inference_func: callable,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_subject_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_subject_id: dict[
        collections.abc.Hashable, list[int]
    ],
    update_func_args_by_by_subject_id_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
    suppress_all_data_checks: bool,
    suppress_interactive_data_checks: bool,
) -> tuple[
    jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
]:
    """
    Computes the average weighted estimating function stack across all subjects, along with
    auxiliary values used to construct the adjusted and classical sandwich variances.

    Args:
        flattened_betas_and_theta (jnp.ndarray):
            A list of JAX NumPy arrays representing the betas produced by all updates and the
            theta value, in that order. Important that this is a 1D array for efficiency reasons.
            We simply extract the betas and theta from this array below.
        beta_dim (int):
            The dimension of each of the beta parameters.
        theta_dim (int):
            The dimension of the theta parameter.
        subject_ids (jnp.ndarray):
            A 1D JAX NumPy array of subject IDs.
        action_prob_func (callable):
            The action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        alg_update_func (callable):
            The algorithm update estimating or loss function.
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
        alg_update_func_args_previous_betas_index (int):
            The index in the update function arguments tuple where previous betas are provided.
        inference_func (callable):
            The inference loss or estimating function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_subject_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of subject ids to the function arguments
            required to compute action probabilities for this subject.
        policy_num_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of subject ids to dictionaries mapping decision times to the policy number in use.
            Only applies to active decision times!
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_subject_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping subject IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_subject_id (dict[collections.abc.Hashable, list[int]]):
            For each subject, a list of decision times to which action probabilities correspond if
            provided. Typically just active times if action probabilites are used in the inference
            loss or estimating function.
        update_func_args_by_by_subject_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
            A dictionary where keys are policy numbers and values are dictionaries mapping subject IDs
            to their respective update function arguments.
        action_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping subject IDs to their respective actions taken at each decision time.
            Only applies to active decision times!
        suppress_all_data_checks (bool):
            If True, suppresses carrying out any data checks at all.
        suppress_interactive_data_checks (bool):
            If True, suppresses interactive data checks that would otherwise be performed to ensure
            the correctness of the threaded arguments. The checks are still performed, but
            any interactive prompts are suppressed.

    Returns:
        jnp.ndarray:
            A 2D JAX NumPy array holding the average weighted estimating function stack.
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            A tuple containing
            1. the average weighted estimating function stack
            2. the subject-level adjusted meat matrix contributions
            3. the subject-level classical meat matrix contributions
            4. the subject-level inverse classical bread matrix contributions
            5. raw per-subject weighted estimating function
            stacks.
    """

    # 1. Collect estimating functions by differentiating the loss functions if needed.
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

    betas, theta = unflatten_params(
        flattened_betas_and_theta,
        beta_dim,
        theta_dim,
    )

    # 2. Thread in the betas and theta in all_post_update_betas_and_theta into the arguments
    # supplied for the above functions, so that differentiation works correctly.  The existing
    # values should be the same, but not connected to the parameter we are differentiating
    # with respect to. Note we will also find it useful below to have the action probability args
    # nested dict structure flipped to be subject_id -> decision_time -> args, so we do that here too.

    logger.info("Threading in betas to action probability arguments for all subjects.")
    (
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        action_prob_func_args_by_decision_time_by_subject_id,
    ) = thread_action_prob_func_args(
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id,
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
        "function args for all subjects"
    )
    threaded_update_func_args_by_policy_num_by_subject_id = thread_update_func_args(
        update_func_args_by_by_subject_id_by_policy_num,
        betas,
        beta_index_by_policy_num,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        action_prob_func,
    )

    # If action probabilites are used in the algorithm estimating function, make
    # sure that substituting in the reconstructed action probabilities is approximately
    # equivalent to using the original action probabilities.
    if not suppress_all_data_checks and alg_update_func_args_action_prob_index >= 0:
        input_checks.require_threaded_algorithm_estimating_function_args_equivalent(
            algorithm_estimating_func,
            update_func_args_by_by_subject_id_by_policy_num,
            threaded_update_func_args_by_policy_num_by_subject_id,
            suppress_interactive_data_checks,
        )

    # 4. Thread the central theta into the inference function arguments
    # and replace any action probabilities with reconstructed ones from the above
    # arguments with the central betas introduced.
    logger.info(
        "Threading in theta and beta-dependent action probabilities to inference update "
        "function args for all subjects"
    )
    threaded_inference_func_args_by_subject_id = thread_inference_func_args(
        inference_func_args_by_subject_id,
        inference_func_args_theta_index,
        theta,
        inference_func_args_action_prob_index,
        threaded_action_prob_func_args_by_decision_time_by_subject_id,
        inference_action_prob_decision_times_by_subject_id,
        action_prob_func,
    )

    # If action probabilites are used in the inference estimating function, make
    # sure that substituting in the reconstructed action probabilities is approximately
    # equivalent to using the original action probabilities.
    if not suppress_all_data_checks and inference_func_args_action_prob_index >= 0:
        input_checks.require_threaded_inference_estimating_function_args_equivalent(
            inference_estimating_func,
            inference_func_args_by_subject_id,
            threaded_inference_func_args_by_subject_id,
            suppress_interactive_data_checks,
        )

    # 5. Now we can compute the weighted estimating function stacks for all subjects
    # as well as collect related values used to construct the adjusted and classical
    # sandwich variances.
    results = [
        single_subject_weighted_estimating_function_stacker(
            beta_dim,
            subject_id,
            action_prob_func,
            algorithm_estimating_func,
            inference_estimating_func,
            action_prob_func_args_beta_index,
            inference_func_args_theta_index,
            action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_action_prob_func_args_by_decision_time_by_subject_id[subject_id],
            threaded_update_func_args_by_policy_num_by_subject_id[subject_id],
            threaded_inference_func_args_by_subject_id[subject_id],
            policy_num_by_decision_time_by_subject_id[subject_id],
            action_by_decision_time_by_subject_id[subject_id],
            beta_index_by_policy_num,
        )
        for subject_id in subject_ids.tolist()
    ]

    stacks = jnp.array([result[0] for result in results])
    outer_products = jnp.array([result[1] for result in results])
    inference_only_outer_products = jnp.array([result[2] for result in results])
    inference_hessians = jnp.array([result[3] for result in results])

    # 6. Note this strange return structure! We will differentiate the first output,
    # but the second tuple will be passed along without modification via has_aux=True and then used
    # for the estimating functions sum check, per_subject_classical_bread_contributions, and
    # classical meat and inverse read matrices. The raw per-subject stacks are also returned for
    # debugging purposes.

    # Note that returning the raw stacks here as the first argument is potentially
    # memory-intensive when combined with differentiation. Keep this in mind if the per-subject bread
    # inverse contributions are needed for something like CR2/CR3 small-sample corrections.
    return jnp.mean(stacks, axis=0), (
        jnp.mean(stacks, axis=0),
        outer_products,
        inference_only_outer_products,
        inference_hessians,
        stacks,
    )


def construct_classical_and_adjusted_sandwiches(
    theta_est: jnp.ndarray,
    all_post_update_betas: jnp.ndarray,
    subject_ids: jnp.ndarray,
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    alg_update_func: callable,
    alg_update_func_type: str,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    alg_update_func_args_previous_betas_index: int,
    inference_func: callable,
    inference_func_type: str,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
    action_prob_func_args_by_subject_id_by_decision_time: dict[
        collections.abc.Hashable, dict[int, tuple[Any, ...]]
    ],
    policy_num_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int | float]
    ],
    initial_policy_num: int | float,
    beta_index_by_policy_num: dict[int | float, int],
    inference_func_args_by_subject_id: dict[collections.abc.Hashable, tuple[Any, ...]],
    inference_action_prob_decision_times_by_subject_id: dict[
        collections.abc.Hashable, list[int]
    ],
    update_func_args_by_by_subject_id_by_policy_num: dict[
        collections.abc.Hashable, dict[int | float, tuple[Any, ...]]
    ],
    action_by_decision_time_by_subject_id: dict[
        collections.abc.Hashable, dict[int, int]
    ],
    suppress_all_data_checks: bool,
    suppress_interactive_data_checks: bool,
    small_sample_correction: str,
    form_adjusted_meat_adjustments_explicitly: bool,
    stabilize_joint_bread: bool,
    analysis_df: pd.DataFrame | None,
    active_col_name: str | None,
    action_col_name: str | None,
    calendar_t_col_name: str | None,
    subject_id_col_name: str | None,
    action_prob_func_args: tuple | None,
    action_prob_col_name: str | None,
) -> tuple[
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
    jnp.ndarray[jnp.float32],
]:
    """
    Constructs the classical and adjusted sandwich matrices, as well as various
    intermediate pieces in their consruction.

    This is done by computing and differentiating the average weighted estimating function stack
    with respect to the betas and theta, using the resulting Jacobian to compute the bread
    and meat matrices, and then stably computing sandwiches.

    Args:
        theta_est (jnp.ndarray):
            A 1-D JAX NumPy array representing the parameter estimate for inference.
        all_post_update_betas (jnp.ndarray):
            A 2-D JAX NumPy array representing all parameter estimates for the algorithm updates.
        subject_ids (jnp.ndarray):
            A 1-D JAX NumPy array holding all subject IDs in the deployment.
        action_prob_func (callable):
            The action probability function.
        action_prob_func_args_beta_index (int):
            The index of beta in the action probability function arguments tuples.
        alg_update_func (callable):
            The algorithm update loss/estimating function.
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
        alg_update_func_args_previous_betas_index (int):
            The index in the update function arguments tuple where the previous betas are provided.
        inference_func (callable):
            The inference loss or estimating function.
        inference_func_type (str):
            The type of the inference function (loss or estimating).
        inference_func_args_theta_index (int):
            The index of the theta parameter in the inference function arguments tuples.
        inference_func_args_action_prob_index (int):
            The index of action probabilities in the inference function arguments tuple, if
            applicable. -1 otherwise.
        action_prob_func_args_by_subject_id_by_decision_time (dict[collections.abc.Hashable, dict[int, tuple[Any, ...]]]):
            A dictionary mapping decision times to maps of subject ids to the function arguments
            required to compute action probabilities for this subject.
        policy_num_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int | float]]):
            A map of subject ids to dictionaries mapping decision times to the policy number in use.
            Only applies to active decision times!
        initial_policy_num (int | float):
            The policy number of the initial policy before any updates.
        beta_index_by_policy_num (dict[int | float, int]):
            A dictionary mapping policy numbers to the index of the corresponding beta in
            all_post_update_betas. Note that this is only for non-initial, non-fallback policies.
        inference_func_args_by_subject_id (dict[collections.abc.Hashable, tuple[Any, ...]]):
            A dictionary mapping subject IDs to their respective inference function arguments.
        inference_action_prob_decision_times_by_subject_id (dict[collections.abc.Hashable, list[int]]):
            For each subject, a list of decision times to which action probabilities correspond if
            provided. Typically just active times if action probabilites are used in the inference
            loss or estimating function.
        update_func_args_by_by_subject_id_by_policy_num (dict[collections.abc.Hashable, dict[int | float, tuple[Any, ...]]]):
            A dictionary where keys are policy numbers and values are dictionaries mapping subject IDs
            to their respective update function arguments.
        action_by_decision_time_by_subject_id (dict[collections.abc.Hashable, dict[int, int]]):
            A dictionary mapping subject IDs to their respective actions taken at each decision time.
            Only applies to active decision times!
        suppress_all_data_checks (bool):
            If True, suppresses carrying out any data checks at all.
        suppress_interactive_data_checks (bool):
            If True, suppresses interactive data checks that would otherwise be performed to ensure
            the correctness of the threaded arguments. The checks are still performed, but
            any interactive prompts are suppressed.
        small_sample_correction (str):
            The type of small sample correction to apply. See SmallSampleCorrections class for
            options.
        form_adjusted_meat_adjustments_explicitly (bool):
            If True, explicitly forms the per-subject meat adjustments that differentiate the adjusted
            sandwich from the classical sandwich. This is for diagnostic purposes, as the
            adjusted sandwich is formed without doing this.
        stabilize_joint_bread (bool):
            If True, will apply various techniques to stabilize the joint bread if necessary.
        analysis_df (pd.DataFrame):
            The full analysis dataframe, needed if forming the adjusted meat adjustments explicitly.
        active_col_name (str):
            The name of the column in analysis_df indicating whether a subject is active at a given decision time.
        action_col_name (str):
            The name of the column in analysis_df indicating the action taken at a given decision time.
        calendar_t_col_name (str):
            The name of the column in analysis_df indicating the calendar time of a given decision time.
        subject_id_col_name (str):
            The name of the column in analysis_df indicating the subject ID.
        action_prob_func_args (tuple):
            The arguments to be passed to the action probability function, needed if forming the
            adjusted meat adjustments explicitly.
        action_prob_col_name (str):
            The name of the column in analysis_df indicating the action probability of the action taken,
            needed if forming the adjusted meat adjustments explicitly.
    Returns:
        tuple[jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32]]:
            A tuple containing:
            - The raw joint bread matrix.
            - The (possibly) stabilized joint bread matrix.
            - The joint meat matrix.
            - The joint sandwich matrix.
            - The classical bread matrix.
            - The classical meat matrix.
            - The classical sandwich matrix.
            - The average weighted estimating function stack.
            - All per-subject weighted estimating function stacks.
            - The per-subject adjusted meat small-sample corrections.
            - The per-subject classical meat small-sample corrections.
            - The per-subject adjusted meat adjustments, if form_adjusted_meat_adjustments_explicitly
              is True, otherwise an array of NaNs.
    """
    logger.info(
        "Differentiating average weighted estimating function stack and collecting auxiliary values."
    )
    theta_dim = theta_est.shape[0]
    beta_dim = all_post_update_betas.shape[1]
    # Note that these "contributions" are per-subject Jacobians of the weighted estimating function stack.
    raw_joint_adjusted_bread_matrix, (
        avg_estimating_function_stack,
        per_subject_joint_adjusted_meat_contributions,
        per_subject_classical_meat_contributions,
        per_subject_classical_bread_contributions,
        per_subject_estimating_function_stacks,
    ) = jax.jacrev(
        get_avg_weighted_estimating_function_stacks_and_aux_values, has_aux=True
    )(
        # While JAX can technically differentiate with respect to a list of JAX arrays,
        # it is apparently more efficient to flatten them into a single array. This is done
        # here to improve performance. We can simply unflatten them inside the function.
        flatten_params(all_post_update_betas, theta_est),
        beta_dim,
        theta_dim,
        subject_ids,
        action_prob_func,
        action_prob_func_args_beta_index,
        alg_update_func,
        alg_update_func_type,
        alg_update_func_args_beta_index,
        alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index,
        inference_func,
        inference_func_type,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id,
        initial_policy_num,
        beta_index_by_policy_num,
        inference_func_args_by_subject_id,
        inference_action_prob_decision_times_by_subject_id,
        update_func_args_by_by_subject_id_by_policy_num,
        action_by_decision_time_by_subject_id,
        suppress_all_data_checks,
        suppress_interactive_data_checks,
    )

    num_subjects = len(subject_ids)

    (
        joint_adjusted_meat_matrix,
        classical_meat_matrix,
        per_subject_adjusted_corrections,
        per_subject_classical_corrections,
    ) = perform_desired_small_sample_correction(
        small_sample_correction,
        per_subject_joint_adjusted_meat_contributions,
        per_subject_classical_meat_contributions,
        per_subject_classical_bread_contributions,
        num_subjects,
        theta_dim,
    )

    # Increase diagonal block dominance possibly improve conditioning of diagonal
    # blocks as necessary, to ensure mathematical stability of joint bread
    stabilized_joint_adjusted_bread_matrix = (
        (
            stabilize_joint_bread_if_necessary(
                raw_joint_adjusted_bread_matrix,
                beta_dim,
                theta_dim,
            )
        )
        if stabilize_joint_bread
        else raw_joint_adjusted_bread_matrix
    )

    # Now stably (no explicit inversion) form our sandwiches.
    joint_adjusted_sandwich = form_sandwich_from_bread_and_meat(
        stabilized_joint_adjusted_bread_matrix,
        joint_adjusted_meat_matrix,
        num_subjects,
        method=SandwichFormationMethods.BREAD_T_QR,
    )
    classical_bread_matrix = jnp.mean(per_subject_classical_bread_contributions, axis=0)
    classical_sandwich = form_sandwich_from_bread_and_meat(
        classical_bread_matrix,
        classical_meat_matrix,
        num_subjects,
        method=SandwichFormationMethods.BREAD_T_QR,
    )

    per_subject_adjusted_meat_adjustments = jnp.full(
        (len(subject_ids), theta_dim, theta_dim), jnp.nan
    )
    if form_adjusted_meat_adjustments_explicitly:
        per_subject_adjusted_classical_meat_contributions = (
            form_adjusted_meat_adjustments_directly(
                theta_dim,
                all_post_update_betas.shape[1],
                stabilized_joint_adjusted_bread_matrix,
                per_subject_estimating_function_stacks,
                analysis_df,
                active_col_name,
                action_col_name,
                calendar_t_col_name,
                subject_id_col_name,
                action_prob_func,
                action_prob_func_args,
                action_prob_func_args_beta_index,
                theta_est,
                inference_func,
                inference_func_args_theta_index,
                subject_ids,
                action_prob_col_name,
            )
        )
        # Validate that the adjusted meat adjustments we just formed are accurate by constructing
        # the theta-only adjusted sandwich from them and checking that it matches the standard result
        # we get by taking a subset of the joint sandwich.
        # First just apply any small-sample correction for parity.
        (
            _,
            theta_only_adjusted_meat_matrix_v2,
            _,
            _,
        ) = perform_desired_small_sample_correction(
            small_sample_correction,
            per_subject_joint_adjusted_meat_contributions,
            per_subject_adjusted_classical_meat_contributions,
            per_subject_classical_bread_contributions,
            num_subjects,
            theta_dim,
        )
        theta_only_adjusted_sandwich_from_adjustments = (
            form_sandwich_from_bread_and_meat(
                classical_bread_matrix,
                theta_only_adjusted_meat_matrix_v2,
                num_subjects,
                method=SandwichFormationMethods.BREAD_T_QR,
            )
        )
        theta_only_adjusted_sandwich = joint_adjusted_sandwich[-theta_dim:, -theta_dim:]

        if not np.allclose(
            theta_only_adjusted_sandwich,
            theta_only_adjusted_sandwich_from_adjustments,
            rtol=3e-2,
        ):
            logger.warning(
                "There may be a bug in the explicit meat adjustment calculation (this doesn't affect the actual calculation, just diagnostics). We've calculated the theta-only adjusted sandwich two different ways and they do not match sufficiently."
            )

    # Stack the joint bread pieces together horizontally and return the auxiliary
    # values too. The joint bread should always be block lower triangular.
    return (
        raw_joint_adjusted_bread_matrix,
        stabilized_joint_adjusted_bread_matrix,
        joint_adjusted_meat_matrix,
        joint_adjusted_sandwich,
        classical_bread_matrix,
        classical_meat_matrix,
        classical_sandwich,
        avg_estimating_function_stack,
        per_subject_estimating_function_stacks,
        per_subject_adjusted_corrections,
        per_subject_classical_corrections,
        per_subject_adjusted_meat_adjustments,
    )


# TODO: I think there should be interaction to confirm stabilization.  It is
# important for the subject to know if this is happening. Even if enabled, it is important
# that the subject know it actually kicks in.
def stabilize_joint_bread_if_necessary(
    joint_adjusted_bread_matrix: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
) -> jnp.ndarray:
    """
    Stabilizes the joint bread matrix if necessary by increasing diagonal block
    dominance and/or adding a small ridge penalty to the diagonal blocks.

    Args:
        joint_adjusted_bread_matrix (jnp.ndarray):
            A 2-D JAX NumPy array representing the joint bread matrix.
        beta_dim (int):
            The dimension of each beta parameter.
        theta_dim (int):
            The dimension of the theta parameter.
    Returns:
        jnp.ndarray:
            A 2-D NumPy array representing the stabilized joint bread matrix.
    """

    # TODO: come up with more sophisticated settings here. These are maybe a little loose,
    # but I especially want to avoid adding ridge penalties if possible.
    # Would be interested in dividing each by 10, though.

    # Set thresholds to guide stabilization.
    diagonal_block_cond_threshold = 2e2
    whole_RL_block_cond_threshold = 1e4

    # Grab just the RL block and convert numpy array for easier manipulation.
    RL_stack_beta_derivatives_block = np.array(
        joint_adjusted_bread_matrix[:-theta_dim, :-theta_dim]
    )
    num_updates = RL_stack_beta_derivatives_block.shape[0] // beta_dim
    for i in range(1, num_updates + 1):

        # Add ridge penalty to diagonal block to control its condition number if necessary.
        # Define the slice for the current diagonal block
        diagonal_block_slice = slice((i - 1) * beta_dim, i * beta_dim)
        diagonal_block = RL_stack_beta_derivatives_block[
            diagonal_block_slice, diagonal_block_slice
        ]
        diagonal_block_cond_number = np.linalg.cond(diagonal_block)
        svs = np.linalg.svd(diagonal_block, compute_uv=False)
        max_sv = svs[0]
        min_sv = svs[-1]

        ridge_penalty = max(
            0,
            (max_sv - diagonal_block_cond_threshold * min_sv)
            / (diagonal_block_cond_threshold + 1),
        )

        if ridge_penalty:
            new_block = diagonal_block + ridge_penalty * np.eye(beta_dim)
            new_diagonal_block_cond_number = np.linalg.cond(new_block)
            RL_stack_beta_derivatives_block[
                diagonal_block_slice, diagonal_block_slice
            ] = diagonal_block + ridge_penalty * np.eye(beta_dim)
            # TODO: Require subject input here in interactive settings?
            logger.info(
                "Added ridge penalty of %s to diagonal block for update %s to improve conditioning from %s to %s",
                ridge_penalty,
                i,
                diagonal_block_cond_number,
                new_diagonal_block_cond_number,
            )

        # Damp off-diagonal blocks to improve conditioning of whole RL block if necessary.
        off_diagonal_block_row_slices = (
            slice((i - 1) * beta_dim, i * beta_dim),
            slice((i - 1) * beta_dim),
        )
        whole_block_cur_update_size = i * beta_dim
        initial_whole_block_cond_number = None
        incremental_damping_factor = 0.9
        max_iterations = 50
        damping_applied = 1

        for _ in range(max_iterations):
            whole_block_cur_update = RL_stack_beta_derivatives_block[
                :whole_block_cur_update_size, :whole_block_cur_update_size
            ]
            whole_block_cur_update_cond_number = np.linalg.cond(whole_block_cur_update)
            if initial_whole_block_cond_number is None:
                initial_whole_block_cond_number = whole_block_cur_update_cond_number

            if whole_block_cur_update_cond_number <= whole_RL_block_cond_threshold:
                break

            damping_applied *= incremental_damping_factor
            RL_stack_beta_derivatives_block[
                off_diagonal_block_row_slices
            ] *= incremental_damping_factor
        else:
            damping_applied = 0
            RL_stack_beta_derivatives_block[off_diagonal_block_row_slices] *= 0

            # TODO: Maybe in this case, roll back through previous rows and damp off diagonals
            # instead of adding ridge?  Feels a little safer because if we zeroed everything
            # off-diagonal and didnt touch diagonal, we'd get classical.
            if whole_block_cur_update_cond_number > whole_RL_block_cond_threshold:
                logger.warning(
                    "Off-diagonal blocks were zeroed for update %s, but conditioning is still poor: %s > %s. Adding extra ridge penalty to entire RL block so far.",
                    i,
                    whole_block_cur_update_cond_number,
                    whole_RL_block_cond_threshold,
                )

            svs = np.linalg.svd(whole_block_cur_update, compute_uv=False)
            max_sv = svs[0]
            min_sv = svs[-1]

            ridge_penalty = max(
                0,
                (max_sv - whole_RL_block_cond_threshold * min_sv)
                / (whole_RL_block_cond_threshold + 1),
            )

            # TODO: This is highly questionable, potentially modifying the matrix very significantly.
            new_block = whole_block_cur_update + ridge_penalty * np.eye(
                whole_block_cur_update_size
            )
            new_whole_block_cond_number = np.linalg.cond(new_block)
            RL_stack_beta_derivatives_block[
                :whole_block_cur_update_size, :whole_block_cur_update_size
            ] += ridge_penalty * np.eye(whole_block_cur_update_size)
            logger.info(
                "Added ridge penalty of %s to entire RL block up to update %s to improve conditioning from %s to %s",
                ridge_penalty,
                i,
                whole_block_cur_update_cond_number,
                new_whole_block_cond_number,
            )

            # Add ridge penalty to off-diagonal blocks if necessary.

        if damping_applied < 1:
            logger.info(
                "Applied damping factor of %s to off-diagonal blocks for update %s to improve conditioning of whole RL block up to that update from %s to %s",
                damping_applied,
                i,
                initial_whole_block_cond_number,
                whole_block_cur_update_cond_number,
            )

    return np.block(
        [
            [
                RL_stack_beta_derivatives_block,
                joint_adjusted_bread_matrix[:-theta_dim, -theta_dim:],
            ],
            [
                joint_adjusted_bread_matrix[-theta_dim:, :-theta_dim],
                joint_adjusted_bread_matrix[-theta_dim:, -theta_dim:],
            ],
        ]
    )


def form_sandwich_from_bread_and_meat(
    bread: jnp.ndarray,
    meat: jnp.ndarray,
    num_subjects: int,
    method: str = SandwichFormationMethods.BREAD_T_QR,
) -> jnp.ndarray:
    """
    Forms a sandwich variance matrix from the provided bread and meat matrices.

    Attempts to do so STABLY without ever forming the bread inverse matrix itself
    (except with naive option).

    Args:
        bread (jnp.ndarray):
            A 2-D JAX NumPy array representing the bread matrix.
        meat (jnp.ndarray):
            A 2-D JAX NumPy array representing the meat matrix.
        num_subjects (int):
            The number of subjects in the deployment, used to scale the sandwich appropriately.
        method (str):
            The method to use for forming the sandwich.

            SandwichFormationMethods.BREAD_T_QR uses the QR decomposition of the transpose
            of the bread matrix.

            SandwichFormationMethods.MEAT_SVD_SOLVE uses a decomposition of the meat matrix.

            SandwichFormationMethods.NAIVE simply inverts the bread and forms the sandwich.


    Returns:
        jnp.ndarray:
            A 2-D JAX NumPy array representing the sandwich variance matrix.
    """

    if method == SandwichFormationMethods.BREAD_T_QR:
        # QR of B^T → Q orthogonal, R upper triangular; L = R^T lower triangular
        Q, R = np.linalg.qr(bread.T, mode="reduced")
        L = R.T

        new_meat = scipy.linalg.solve_triangular(
            L, scipy.linalg.solve_triangular(L, meat.T, lower=True).T, lower=True
        )

        return Q @ new_meat @ Q.T / num_subjects
    elif method == SandwichFormationMethods.MEAT_SVD_SOLVE:
        # Factor the meat via SVD without any symmetrization or truncation.
        # For general (possibly slightly nonsymmetric) M, SVD gives M = U @ diag(s) @ Vh.
        # We construct two square-root factors C_left = U * sqrt(s) and C_right = V * sqrt(s)
        # so that M = C_left @ C_right.T exactly, then solve once per factor.
        U, s, Vh = scipy.linalg.svd(meat, full_matrices=False)
        C_left = U * np.sqrt(s)
        C_right = Vh.T * np.sqrt(s)

        # Solve B W_left = C_left and B W_right = C_right (no explicit inverses).
        W_left = scipy.linalg.solve(bread, C_left)
        W_right = scipy.linalg.solve(bread, C_right)

        # Return the exact sandwich: V = (B^{-1} C_left) (B^{-1} C_right)^T / num_subjects
        return W_left @ W_right.T / num_subjects

    elif method == SandwichFormationMethods.NAIVE:
        # Simply invert the bread and form the sandwich directly.
        # This is NOT numerically stable and is only included for comparison purposes.
        bread_inverse = np.linalg.inv(bread)
        return bread_inverse @ meat @ bread_inverse.T / num_subjects

    else:
        raise ValueError(
            f"Unknown sandwich method: {method}. Please use 'bread_t_qr' or 'meat_decomposition_solve'."
        )


if __name__ == "__main__":
    cli()
