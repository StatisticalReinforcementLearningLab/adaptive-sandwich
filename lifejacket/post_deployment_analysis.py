from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pathlib
import pickle
import resource
import sys
import time
import typing
from collections.abc import Callable
from typing import Any

import click
import jax
import numpy as np
import pandas as pd
import scipy
from jax import numpy as jnp

from . import diagnostics, input_checks
from .batched_weighted_estimating_function_stack import (
    ActionProbLayerPrecompute,
    InferenceLayerPrecompute,
    UpdateArgBucket,
    UpdateLayerPrecompute,
    _stackable_positions,
    build_action_prob_layer_precompute,
    build_inference_layer_precompute,
    build_update_layer_precompute,
    check_batched_algorithm_estimating_function_args_equivalent,
    check_batched_inference_estimating_function_args_equivalent,
    compute_action_prob_layer_outputs,
    compute_batched_algorithm_component,
    compute_batched_inference_outputs,
    compute_windowed_weight_products,
    resolve_combine_updates_into_one_vmap,
)
from .constants import (
    JACOBIAN_AUTO_MAX_CHUNK,
    JACOBIAN_AUTO_ROW_BUDGET,
    JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM,
    FunctionTypes,
    SandwichFormationMethods,
)
from .form_adjusted_meat_adjustments_directly import (
    form_adjusted_meat_adjustments_directly,
)
from .helper_functions import (
    calculate_beta_dim,
    collect_all_post_update_betas,
    compute_subject_radon_nikodym_weights,
    construct_beta_index_by_policy_num_map,
    extract_action_and_policy_by_decision_time_by_subject_id,
    flatten_params,
    get_active_df_column,
    load_function_from_same_named_file,
    log_phase_duration,
    unflatten_params,
)
from .vmap_helpers import stack_batched_arg_lists_into_tensors

logger = logging.getLogger(__name__)


def _peak_rss_mb() -> float:
    """
    This process's peak resident set size ("high-water mark"), in MB.

    ru_maxrss's units are platform-dependent -- bytes on macOS/BSD, kibibytes
    on Linux (see each platform's getrusage(2)) -- so the divisor must be
    chosen per-platform rather than assuming one or the other; a fixed
    bytes-assuming divisor under-reports by ~1024x on the Linux clusters real
    analyses run on.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024 if sys.platform == "darwin" else 1024)


@click.group()
def cli():
    """
    lifejacket command-line interface.

    Logging is configured here, at the CLI entry point, rather than at import
    time in library modules -- see
    https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library.
    Code that imports lifejacket directly (rather than through this CLI) gets
    silent library loggers by default, as recommended there.
    """
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )


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
    "--alg_update_func_args_mask_index",
    type=int,
    default=-1000,
    help="Opt-in (default: unused): index of a new, LAST argument alg_update_func accepts a per-subject validity mask at (1.0=real row, 0.0=padded row), if applicable. When given (>= 0), every shape-bucket at each update is consolidated into one by self-padding every alg_update_func_args_ragged_indices position instead of grouping subjects by exact arg-tuple shape -- fixes real-world incremental-recruitment studies producing far more shape-buckets than subject count alone would suggest. Only usable with an alg_update_func written to accept and correctly multiply by this mask.",
)
@click.option(
    "--alg_update_func_args_ragged_indices",
    type=int,
    multiple=True,
    help="Which alg_update_func_args positions to self-pad when alg_update_func_args_mask_index is given (repeat this option once per position) -- must be non-empty in that case. Ignored otherwise.",
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
    "--inference_func_args_mask_index",
    type=int,
    default=-1000,
    help="Same as alg_update_func_args_mask_index, for inference_func.",
)
@click.option(
    "--inference_func_args_ragged_indices",
    type=int,
    multiple=True,
    help="Same as alg_update_func_args_ragged_indices, for inference_func.",
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
    "--form_adjusted_meat_adjustments_explicitly",
    type=bool,
    default=False,
    help="If True, explicitly forms the per-subject meat adjustments that differentiate the adjusted sandwich from the classical sandwich. This is for diagnostic purposes, as the adjusted sandwich is formed without doing this. WARNING: this ends by dropping into an interactive debugger (breakpoint()) to allow inspecting intermediate variables -- only use this in an interactive session; it will hang or fail in CI/batch/SLURM or any other non-interactive context.",
)
@click.option(
    "--jacobian_row_chunk_size",
    type=int,
    default=None,
    help="Unset (default) = AUTO: the jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values) backward pass runs as a single unchunked jax.vmap over the full output basis for small problems (out_dim <= 512), and is split into memory-bounded chunks of at most max(1, min(64, 65536 // out_dim)) output-basis rows for large ones -- numerically identical either way; the heuristic is calibrated on one real oralytics-scale study on one 24GB machine, with out_dim as a proxy for the true memory footprint (see resolve_jacobian_row_chunk_size's docstring). 0 = force the single unchunked vmap (the fastest option when it fits in memory). A positive int = explicit chunk size, honored verbatim: pass a smaller value (e.g. 8 or 4) to use less memory on a smaller machine or bigger study (or after a crash under auto), a larger one to go faster when memory is known-plentiful.",
)
@click.option(
    "--combine_updates_into_one_vmap",
    type=bool,
    default=None,
    help="Unset (default) = AUTO: replaces compute_batched_algorithm_component's per-update, per-shape-bucket jax.vmap loop with exactly one jax.vmap call spanning every (subject, update) pair at once (fewer, bigger dispatches; numerically identical) whenever eligible -- alg_update_func_args_mask_index >= 0 (a mask-aware alg_update_func is a prerequisite, so this cannot be auto-enabled without that opt-in) and alg_update_func_args_previous_betas_index < 0 -- and silently stays on the original loop otherwise; if a combining invariant only checkable mid-precompute is violated, auto mode falls back to the original loop with a WARNING log. True = force on, raising loudly when ineligible rather than silently mis-batching. False = force off even when eligible. Not applied to the separate local-linearization diagnostic, which always uses the original loop.",
)
@click.option(
    "--run_diagnostics",
    type=bool,
    default=True,
    help="If True (the default), runs the extended diagnostic suite (see lifejacket.diagnostics) "
    "after computing the adjusted sandwich and writes diagnostic_report.pkl. Does not affect the "
    "adjusted sandwich computation itself. Defaults to DiagnosticConfig()'s own cheap-checks-only "
    "settings (root/implementation, local nonlinearity, bread stability, influence concentration, "
    "exploration/weights) -- the expensive exact-nonlinear-perturbation/Jacobian-drift checks stay "
    "opt-in (see --diagnostic_config_pickle) even when this is True.",
)
@click.option(
    "--diagnostic_config_pickle",
    type=click.File("rb"),
    default=None,
    help="Optional pickled lifejacket.diagnostics.DiagnosticConfig controlling the diagnostic "
    "suite. Only used when --run_diagnostics is True; DiagnosticConfig() defaults are used "
    "otherwise.",
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

    diagnostic_config_pickle = kwargs.pop("diagnostic_config_pickle", None)
    kwargs["diagnostic_config"] = (
        pickle.load(diagnostic_config_pickle)
        if diagnostic_config_pickle is not None
        else None
    )

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
    form_adjusted_meat_adjustments_explicitly: bool,
    alg_update_func_args_mask_index: int = -1,
    alg_update_func_args_ragged_indices: tuple[int, ...] = (),
    inference_func_args_mask_index: int = -1,
    inference_func_args_ragged_indices: tuple[int, ...] = (),
    jacobian_row_chunk_size: int | None = None,
    combine_updates_into_one_vmap: bool | None = None,
    run_diagnostics: bool = True,
    diagnostic_config: diagnostics.DiagnosticConfig | None = None,
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
    form_adjusted_meat_adjustments_explicitly (bool):
        If True, explicitly forms the per-subject meat adjustments that differentiate the
        sandwich from the classical sandwich. This is for diagnostic purposes, as the
        adjusted sandwich is formed without doing this. WARNING: this ends by dropping
        into an interactive debugger (breakpoint()) -- only enable it in an interactive
        session; see form_adjusted_meat_adjustments_directly's own docstring.
    alg_update_func_args_mask_index (int):
        Opt-in (default -1 = off, zero behavior change): if >= 0, consolidates every
        shape-bucket at each algorithm update into one by self-padding every
        alg_update_func_args_ragged_indices position and appending a validity mask
        (1.0 real / 0.0 padded) as a new last argument to alg_update_func, instead of
        grouping subjects by exact arg-tuple shape. Fixes real-world
        incremental-recruitment studies producing far more shape-buckets than
        subject count alone would suggest (e.g. 146 buckets at just 70 subjects in
        one observed case). Only usable with an alg_update_func written to accept
        and correctly multiply by the appended mask before any row-wise sum -- see
        batched_weighted_estimating_function_stack.self_pad_ragged_args_and_build_mask's
        own docstring for the full padding/masking contract.
    alg_update_func_args_ragged_indices (tuple[int, ...]):
        Which alg_update_func_args positions to self-pad when
        alg_update_func_args_mask_index >= 0 -- e.g. every position shaped
        (num_decision_times_so_far, ...) that varies per subject under staggered
        recruitment. Must be non-empty in that case; ignored otherwise.
    inference_func_args_mask_index (int):
        Same as alg_update_func_args_mask_index, for inference_func.
    inference_func_args_ragged_indices (tuple[int, ...]):
        Same as alg_update_func_args_ragged_indices, for inference_func.
    jacobian_row_chunk_size (int | None):
        Passed straight through to construct_classical_and_adjusted_sandwiches's
        jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values) call.
        None (the default) = AUTO: a single unchunked backward vmap for small
        problems (out_dim <= 512), a conservative heuristic chunk size for
        large ones -- calibrated on one real oralytics-scale study on one
        24GB machine, with out_dim as a proxy for the true memory footprint
        (see resolve_jacobian_row_chunk_size's docstring for the details and
        caveats). 0 = force the single unchunked vmap (the pre-auto default;
        fastest when it fits in memory). A positive int = explicit chunk
        size, honored verbatim (pass a smaller one to use less memory, a
        larger one to go faster when memory is known-plentiful).
    combine_updates_into_one_vmap (bool | None):
        Passed straight through to construct_classical_and_adjusted_sandwiches
        (and, from there, to get_avg_weighted_estimating_function_stacks_and_aux_values,
        which resolves it); see those docstrings for the full semantics. None
        (the default) = AUTO: the per-update loop collapse is enabled
        whenever eligible (alg_update_func_args_mask_index >= 0 and
        alg_update_func_args_previous_betas_index < 0 -- it cannot be
        auto-enabled otherwise, since it needs a mask-aware alg_update_func),
        silently left off otherwise, and falls back to the default loop with
        a WARNING log if a structural invariant only checkable mid-precompute
        is violated -- results are numerically identical either way. True =
        force on, with loud errors when ineligible. False = force off. NOT
        passed to the separate local-linearization diagnostic below
        (compute_local_linearization_error_ratio), which continues to use the
        original per-update/per-bucket loop regardless.
    run_diagnostics (bool):
        If True (the default), runs the extended diagnostic suite (lifejacket.diagnostics.
        run_diagnostic_suite) after the adjusted sandwich has been computed and writes its
        DiagnosticReport to diagnostic_report.pkl in output_dir. Does not otherwise affect the
        adjusted sandwich computation. With no diagnostic_config override, this runs only
        DiagnosticConfig()'s cheap checks (root/implementation, local nonlinearity, bread
        stability, influence concentration, exploration/weights) -- the expensive exact-
        nonlinear-perturbation/Jacobian-drift checks (compute_exact_nonlinear_roots) stay
        opt-in regardless of this flag.
    diagnostic_config (lifejacket.diagnostics.DiagnosticConfig | None):
        Configuration for the diagnostic suite, used only when run_diagnostics is True. Defaults
        to DiagnosticConfig() when not supplied.

    Returns:
    dict: A dictionary containing the theta estimate, adjusted sandwich variance estimate, and
    classical sandwich variance estimate.
    """

    with log_phase_duration("theta_calculation_func"):
        theta_est = jnp.array(theta_calculation_func(analysis_df))

    beta_dim = calculate_beta_dim(
        action_prob_func_args, action_prob_func_args_beta_index
    )
    if not suppress_all_data_checks:
        with log_phase_duration("input_checks"):
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
            )

    ### Begin collecting data structures that will be used to compute the joint bread matrix.
    with log_phase_duration(
        "data_structure_prep.construct_beta_index_by_policy_num_map"
    ):
        beta_index_by_policy_num, initial_policy_num = (
            construct_beta_index_by_policy_num_map(
                analysis_df, policy_num_col_name, active_col_name
            )
        )

    with log_phase_duration("data_structure_prep.collect_all_post_update_betas"):
        all_post_update_betas = collect_all_post_update_betas(
            beta_index_by_policy_num,
            alg_update_func_args,
            alg_update_func_args_beta_index,
        )

    with log_phase_duration(
        "data_structure_prep.extract_action_and_policy_by_decision_time_by_subject_id"
    ):
        (
            action_by_decision_time_by_subject_id,
            policy_num_by_decision_time_by_subject_id,
        ) = extract_action_and_policy_by_decision_time_by_subject_id(
            analysis_df,
            subject_id_col_name,
            active_col_name,
            calendar_t_col_name,
            action_col_name,
            policy_num_col_name,
        )

    with log_phase_duration("data_structure_prep.process_inference_func_args"):
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
    with log_phase_duration("construct_classical_and_adjusted_sandwiches"):
        (
            raw_joint_bread_matrix,
            joint_adjusted_meat_matrix,
            joint_sandwich_matrix,
            classical_bread_matrix,
            classical_meat_matrix,
            classical_sandwich_var_estimate,
            avg_estimating_function_stack,
            per_subject_estimating_function_stacks,
            per_subject_adjusted_corrections,
            per_subject_classical_corrections,
            per_subject_adjusted_meat_contributions,
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
            form_adjusted_meat_adjustments_explicitly,
            analysis_df,
            active_col_name,
            action_col_name,
            calendar_t_col_name,
            subject_id_col_name,
            action_prob_func_args,
            action_prob_col_name,
            alg_update_func_args_mask_index=alg_update_func_args_mask_index,
            alg_update_func_args_ragged_indices=alg_update_func_args_ragged_indices,
            inference_func_args_mask_index=inference_func_args_mask_index,
            inference_func_args_ragged_indices=inference_func_args_ragged_indices,
            jacobian_row_chunk_size=jacobian_row_chunk_size,
            combine_updates_into_one_vmap=combine_updates_into_one_vmap,
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
    adjusted_sandwich_var_estimate = joint_sandwich_matrix[-theta_dim:, -theta_dim:]

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

    with log_phase_duration("eigenvalue_and_condition_diagnostics"):
        (
            joint_bread_cond,
            max_eig_joint,
            eigvals_joint_sandwich,
            max_to_median_ratio_joint_sandwich,
            max_eig_theta,
            eigvals_theta_only_adjusted_sandwich,
            max_to_median_ratio_theta_only_adjusted_sandwich,
        ) = compute_eigenvalue_and_condition_diagnostics(
            raw_joint_bread_matrix,
            joint_sandwich_matrix,
            adjusted_sandwich_var_estimate,
        )

    # Local linearization validity diagnostic (single-run) -- see
    # compute_local_linearization_error_ratio's docstring for the full explanation.
    with log_phase_duration("local_linearization_diagnostic"):
        try:
            local_error_ratio_median, local_error_ratio_p90, local_error_ratio_max = (
                compute_local_linearization_error_ratio(
                    raw_joint_bread_matrix,
                    avg_estimating_function_stack,
                    per_subject_estimating_function_stacks,
                    all_post_update_betas,
                    theta_est,
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
                    action_prob_func_args,
                    policy_num_by_decision_time_by_subject_id,
                    initial_policy_num,
                    beta_index_by_policy_num,
                    inference_func_args_by_subject_id,
                    inference_action_prob_decision_times_by_subject_id,
                    alg_update_func_args,
                    action_by_decision_time_by_subject_id,
                    alg_update_func_args_mask_index=alg_update_func_args_mask_index,
                    alg_update_func_args_ragged_indices=alg_update_func_args_ragged_indices,
                    inference_func_args_mask_index=inference_func_args_mask_index,
                    inference_func_args_ragged_indices=inference_func_args_ragged_indices,
                )
            )
        except Exception as e:
            # This diagnostic is best-effort; failure should not break analysis.
            logger.warning(
                "Failed to compute local linearization error ratio diagnostic: %s",
                str(e),
            )
            local_error_ratio_median = math.nan
            local_error_ratio_p90 = math.nan
            local_error_ratio_max = math.nan

    debug_pieces_dict = {
        "theta_est": theta_est,
        "adjusted_sandwich_var_estimate": adjusted_sandwich_var_estimate,
        "classical_sandwich_var_estimate": classical_sandwich_var_estimate,
        "raw_joint_bread_matrix": raw_joint_bread_matrix,
        "joint_meat_matrix": joint_adjusted_meat_matrix,
        "classical_bread_matrix": classical_bread_matrix,
        "classical_meat_matrix": classical_meat_matrix,
        "all_estimating_function_stacks": per_subject_estimating_function_stacks,
        "joint_bread_condition_number": joint_bread_cond,
        "max_eigenvalue_joint_sandwich": max_eig_joint,
        "all_eigenvalues_joint_sandwich": eigvals_joint_sandwich,
        "max_to_median_ratio_joint_sandwich": max_to_median_ratio_joint_sandwich,
        "max_eigenvalue_theta_only_adjusted_sandwich": max_eig_theta,
        "all_eigenvalues_theta_only_adjusted_sandwich": eigvals_theta_only_adjusted_sandwich,
        "max_to_median_ratio_theta_only_adjusted_sandwich": max_to_median_ratio_theta_only_adjusted_sandwich,
        "local_linearization_error_ratio_median": local_error_ratio_median,
        "local_linearization_error_ratio_p90": local_error_ratio_p90,
        "local_linearization_error_ratio_max": local_error_ratio_max,
        "all_post_update_betas": all_post_update_betas,
        "per_subject_adjusted_corrections": per_subject_adjusted_corrections,
        "per_subject_classical_corrections": per_subject_classical_corrections,
        "per_subject_adjusted_meat_adjustments": per_subject_adjusted_meat_contributions,
    }
    with open(output_folder_abs_path / "debug_pieces.pkl", "wb") as f:
        pickle.dump(
            debug_pieces_dict,
            f,
        )

    if run_diagnostics:
        logger.info("Running extended diagnostic suite (lifejacket.diagnostics).")

        def _diagnostics_g_tilde(flattened_betas_and_theta: jnp.ndarray) -> jnp.ndarray:
            # The diagnostic suite calls this dozens of times (once per sampled
            # perturbation direction/radius across several checks), and
            # get_avg_weighted_estimating_function_stacks_and_aux_values
            # unconditionally logs shape-bucket-fan-out/phase-duration INFO
            # lines meant to describe the one real analysis call above --
            # without this, those lines flood the log with identical repeats.
            # Raised/restored around just this call so the real analysis run's
            # own logging above, and the diagnostic summary/warnings below,
            # are unaffected.
            root_logger = logging.getLogger()
            previous_level = root_logger.level
            root_logger.setLevel(logging.WARNING)
            try:
                return jnp.asarray(
                    get_avg_weighted_estimating_function_stacks_and_aux_values(
                        flattened_betas_and_theta,
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
                        action_prob_func_args,
                        policy_num_by_decision_time_by_subject_id,
                        initial_policy_num,
                        beta_index_by_policy_num,
                        inference_func_args_by_subject_id,
                        inference_action_prob_decision_times_by_subject_id,
                        alg_update_func_args,
                        action_by_decision_time_by_subject_id,
                        True,  # suppress_all_data_checks
                        True,  # suppress_interactive_data_checks
                        False,  # include_auxiliary_outputs
                    )
                )
            finally:
                root_logger.setLevel(previous_level)

        try:
            diagnostic_report = diagnostics.run_diagnostic_suite(
                _diagnostics_g_tilde,
                flatten_params(all_post_update_betas, theta_est),
                raw_joint_bread_matrix,
                joint_adjusted_meat_matrix,
                joint_sandwich_matrix,
                per_subject_estimating_function_stacks,
                beta_dim,
                theta_dim,
                len(subject_ids),
                diagnostic_config or diagnostics.DiagnosticConfig(),
                # Only the reconstruction check is re-run here: it has no numeric equivalent
                # inside the diagnostic suite. require_estimating_functions_sum_to_zero (still run
                # unconditionally above, interactively, as part of the main pipeline) is
                # deliberately NOT also wired in here anymore -- check_root_and_implementation's
                # own a_root_max already answers the same underlying question ("was the equation
                # actually solved?"), and does so in SE-standardized, portable units rather than
                # the legacy check's arbitrary absolute tolerances (the same non-portability
                # problem discussed for r_j/q_j) -- so re-running it here was redundant, not an
                # independent signal.
                legacy_check_callables=[
                    (
                        "action_probabilities_reconstructed",
                        lambda: (
                            input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
                                analysis_df,
                                action_prob_col_name,
                                calendar_t_col_name,
                                subject_id_col_name,
                                active_col_name,
                                action_prob_func_args,
                                action_prob_func,
                            )
                        ),
                    ),
                ],
                analysis_df=analysis_df,
                active_col_name=active_col_name,
                calendar_t_col_name=calendar_t_col_name,
                action_prob_col_name=action_prob_col_name,
                action_prob_func=action_prob_func,
                action_prob_func_args=action_prob_func_args,
                action_prob_func_args_beta_index=action_prob_func_args_beta_index,
                action_by_decision_time_by_subject_id=action_by_decision_time_by_subject_id,
                policy_num_by_decision_time_by_subject_id=policy_num_by_decision_time_by_subject_id,
                initial_policy_num=initial_policy_num,
                beta_index_by_policy_num=beta_index_by_policy_num,
                subject_ids=subject_ids,
            )
            logger.info(
                "Diagnostic suite classification: %s", diagnostic_report.classification
            )
            for warning_message in diagnostic_report.warnings:
                logger.warning(warning_message)
            with open(output_folder_abs_path / "diagnostic_report.pkl", "wb") as f:
                pickle.dump(diagnostic_report, f)
        except Exception as e:  # noqa: BLE001
            # As with the local linearization diagnostic above, the diagnostic suite is
            # best-effort and failure to run it should not break the underlying analysis.
            logger.warning("Failed to run the extended diagnostic suite: %s", str(e))

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


# Kept only as a slow, obviously-correct oracle for cross-checking the
# batched implementation in tests (see
# test_batched_and_reference_implementations_agree_per_subject_incremental_recruitment
# in tests/unit_tests/test_post_deployment_analysis.py) -- unused by
# get_avg_weighted_estimating_function_stacks_and_aux_values or any other
# production code path since ADS-139 Step 4. See
# docs/adr/0001-adaptive-sandwich-performance-plan.md.
def _reference_single_subject_weighted_estimating_function_stacker(
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
    include_auxiliary_outputs: bool = True,
) -> (
    tuple[
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
        jnp.ndarray[jnp.float32],
    ]
    | jnp.ndarray[jnp.float32]
):
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

        include_auxiliary_outputs (bool):
            If True, returns the adjusted meat, classical meat, and classical bread contributions in
            a second returned tuple. If False, only returns the weighted estimating function stack.

    Returns:
        jnp.ndarray: A 1-D JAX NumPy array representing the subject's weighted estimating function
            stack.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's adjusted meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical meat contribution.
        jnp.ndarray: A 2-D JAX NumPy matrix representing the subject's classical bread contribution.

        or

        jnp.ndarray: A 1-D JAX NumPy array representing the subject's weighted estimating function
            stack.

        depending on the value of include_auxiliary_outputs.
    """

    logger.info(
        "Computing weighted estimating function stack for subject %s.", subject_id
    )

    # First, reformat the supplied data into more convenient structures.

    logger.info(
        "Computing the algorithm component of the weighted estimating function stack for subject %s.",
        subject_id,
    )

    (
        all_weights,
        decision_time_to_all_weights_index_offset,
        first_time_after_first_update,
        min_time_by_policy_num,
        subject_start_time,
        subject_end_time,
    ) = compute_subject_radon_nikodym_weights(
        action_prob_func,
        action_prob_func_args_beta_index,
        action_prob_func_args_by_decision_time,
        threaded_action_prob_func_args_by_decision_time,
        policy_num_by_decision_time,
        action_by_decision_time,
        beta_index_by_policy_num,
    )

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
                        - decision_time_to_all_weights_index_offset
                        # One more than the latest time the subject was in the deployment before the time
                        # the update under consideration first applied. Note the + 1 because range
                        # does not include the right endpoint.
                         : min(
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
    if include_auxiliary_outputs:
        return (
            weighted_stack,
            jnp.outer(weighted_stack, weighted_stack),
            jnp.outer(inference_component, inference_component),
            jax.jacrev(
                inference_estimating_func, argnums=inference_func_args_theta_index
            )(*threaded_inference_func_args),
        )

    else:
        return weighted_stack


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
    include_auxiliary_outputs: bool = True,
    precomputed_layers: tuple[
        ActionProbLayerPrecompute, UpdateLayerPrecompute, InferenceLayerPrecompute
    ]
    | None = None,
    alg_update_func_args_mask_index: int = -1,
    alg_update_func_args_ragged_indices: tuple[int, ...] = (),
    inference_func_args_mask_index: int = -1,
    inference_func_args_ragged_indices: tuple[int, ...] = (),
    combine_updates_into_one_vmap: bool | None = None,
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
        include_auxiliary_outputs (bool):
            If True, returns the adjusted meat, classical meat, and classical bread contributions in addition to the average weighted estimating function stack.
            If False, returns only the average weighted estimating function stack.
        precomputed_layers (tuple[ActionProbLayerPrecompute, UpdateLayerPrecompute, InferenceLayerPrecompute] | None):
            If given, used directly in place of building
            action_prob_layer/update_layer/inference_layer from the raw
            action_prob_func_args_by_subject_id_by_decision_time/
            update_func_args_by_by_subject_id_by_policy_num/
            inference_func_args_by_subject_id arguments (the three
            build_*_precompute calls are skipped entirely). This exists so a
            caller that has already built these once (e.g. the
            local-linearization diagnostic in
            compute_local_linearization_error_ratio, which calls this
            function 15 times with identical structural data and only the
            differentiated betas/theta values changing across calls) can
            reuse them, and -- when the caller is a jax.jit trace -- pass a
            reconstruction of them built from genuine traced arguments
            instead of Python-closed-over concrete arrays, which is what
            actually keeps large (N, T, ...)-shaped precompute data from
            being embedded as XLA literal constants in the compiled
            program. If None (the default), the three objects are built
            fresh from the raw arguments as before -- this is what the one
            real, non-jitted jax.jacrev call site in
            construct_classical_and_adjusted_sandwiches still does. Ignored
            (along with the four *_mask_index/*_ragged_indices arguments
            below) when precomputed_layers is given -- the caller already
            decided how those layers were built.
        alg_update_func_args_mask_index (int):
            Opt-in (default -1 = off, zero behavior change): if >= 0,
            consolidates every shape-bucket at each update into one by
            self-padding every alg_update_func_args_ragged_indices position
            and appending a validity mask as a new last argument to
            alg_update_func, instead of grouping subjects by exact arg-tuple
            shape. Fixes real-world incremental-recruitment studies producing
            far more shape-buckets than any subject-count-only benchmark
            exercises. Only usable with an alg_update_func written to accept
            and correctly multiply by the appended mask -- see
            batched_weighted_estimating_function_stack.self_pad_ragged_args_and_build_mask's
            own docstring.
        alg_update_func_args_ragged_indices (tuple[int, ...]):
            Which alg_update_func_args positions to self-pad when
            alg_update_func_args_mask_index >= 0 -- must be non-empty in that
            case. Ignored otherwise.
        inference_func_args_mask_index (int):
            Same as alg_update_func_args_mask_index, for inference_func.
        inference_func_args_ragged_indices (tuple[int, ...]):
            Same as alg_update_func_args_ragged_indices, for inference_func.
        combine_updates_into_one_vmap (bool | None):
            When enabled, replaces compute_batched_algorithm_component's
            per-update, per-shape-bucket jax.vmap loop with exactly one
            jax.vmap call spanning every (subject, update) pair at once --
            fewer, bigger dispatches instead of many small ones,
            numerically identical either way. None (the default) = AUTO:
            resolved here (immediately before build_update_layer_precompute)
            via batched_weighted_estimating_function_stack.resolve_combine_updates_into_one_vmap
            -- enabled exactly when eligible (alg_update_func_args_mask_index
            >= 0 AND alg_update_func_args_previous_betas_index < 0),
            otherwise silently left on the default loop; if a structural
            invariant violation only detectable during
            build_update_layer_precompute's combined-block work is hit, auto
            mode falls back to the default loop with a WARNING log instead
            of erroring (see that function's combine_is_required argument).
            True = force on, keeping every loud error: requires
            alg_update_func_args_mask_index >= 0 (raises ValueError
            otherwise) and is incompatible with
            alg_update_func_args_previous_betas_index >= 0 (raises
            NotImplementedError otherwise -- see
            build_update_layer_precompute's own docstring for both). False =
            force off, even when eligible. Ignored when precomputed_layers
            is given, exactly like the four *_mask_index/*_ragged_indices
            arguments above -- the caller already decided how those layers
            were built.
    Returns:
        jnp.ndarray:
            A 2D JAX NumPy array holding the average weighted estimating function stack.

        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            A tuple containing
            1. the average weighted estimating function stack
            2. the adjusted meat matrix, pre-summed across subjects
               (stacks.T @ stacks -- an exact linear-algebra identity for
               sum_i outer(stack_i, stack_i), never materializing the
               (num_subjects, out_dim, out_dim) per-subject tensor, which
               dominates memory at real (oralytics-scale) problem sizes --
               see docs/adr/0001-adaptive-sandwich-performance-plan.md)
            3. the classical meat matrix, pre-summed across subjects the
               same way (inference_component.T @ inference_component)
            4. the subject-level inverse classical bread matrix contributions
            5. raw per-subject weighted estimating function
            stacks.
        or jnp.ndarray:
            A 1-D JAX NumPy array representing the subject's weighted estimating function
            stack.
        depending on the value of include_auxiliary_outputs.
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
    subject_ids_np = np.asarray(subject_ids.tolist())
    need_pi_beta_grid = (
        alg_update_func_args_action_prob_index >= 0
        or inference_func_args_action_prob_index >= 0
    )

    if precomputed_layers is not None:
        # See this parameter's own docstring above: skip rebuilding the
        # structural precompute from the raw arguments -- the caller has
        # already built (or, under jax.jit, reconstructed from traced
        # arrays) equivalent action_prob_layer/update_layer/inference_layer
        # objects.
        action_prob_layer, update_layer, inference_layer = precomputed_layers
    else:
        # 2. One-time, plain-numpy structural precompute (never touches betas/theta
        # values -- only which cells are active and which policy/action applied).
        # This replaces the ragged per-subject/per-update Python loops with a
        # small, fixed number of jax.vmap calls below. See
        # docs/adr/0001-adaptive-sandwich-performance-plan.md, Step 4.
        action_prob_layer = build_action_prob_layer_precompute(
            subject_ids_np,
            action_prob_func_args_by_subject_id_by_decision_time,
            action_by_decision_time_by_subject_id,
            policy_num_by_decision_time_by_subject_id,
            beta_index_by_policy_num,
            initial_policy_num,
            action_prob_func_args_beta_index,
        )
        # Resolve the tri-state combine request (None = auto) at this single
        # choke point both real entry paths flow through -- the
        # precomputed_layers/diagnostic path above bypasses it, exactly as
        # documented. combine_is_required=True only for an explicit True, so
        # auto-resolved combining falls back (warn-logged) on structural
        # invariant violations instead of erroring -- see
        # build_update_layer_precompute's docstring.
        resolved_combine_updates_into_one_vmap = resolve_combine_updates_into_one_vmap(
            combine_updates_into_one_vmap,
            alg_update_func_args_mask_index,
            alg_update_func_args_previous_betas_index,
        )
        update_layer = build_update_layer_precompute(
            subject_ids_np,
            update_func_args_by_by_subject_id_by_policy_num,
            beta_index_by_policy_num,
            alg_update_func_args_action_prob_times_index,
            action_prob_layer,
            alg_update_func_args_mask_index,
            alg_update_func_args_ragged_indices,
            combine_updates_into_one_vmap=resolved_combine_updates_into_one_vmap,
            alg_update_func_args_beta_index=alg_update_func_args_beta_index,
            alg_update_func_args_action_prob_index=alg_update_func_args_action_prob_index,
            # Truthiness, not `is True`: any explicit truthy request (True, 1,
            # np.True_, ...) must keep the documented fail-loud contract --
            # only auto (None) gets the warn-and-fall-back semantics.
            combine_is_required=combine_updates_into_one_vmap is not None
            and bool(combine_updates_into_one_vmap),
        )
        inference_layer = build_inference_layer_precompute(
            inference_func_args_by_subject_id,
            inference_func_args_action_prob_index,
            inference_action_prob_decision_times_by_subject_id,
            action_prob_layer,
            inference_func_args_mask_index,
            inference_func_args_ragged_indices,
        )
    # Cheap visibility into shape-bucket fan-out: each distinct shape bucket
    # at a given update becomes its own jax.vmap dispatch in
    # compute_batched_algorithm_component below (see
    # docs/adr/0001-adaptive-sandwich-performance-plan.md, Step 4 -- padding
    # instead of bucketing this axis would silently corrupt any
    # alg_update_func that reduces over accumulated per-subject rows, e.g.
    # under incremental/staggered recruitment). Logged unconditionally since
    # it's O(updates), not O(subjects); watch this if it ever grows close to
    # the subject count, since that means bucketing isn't actually reducing
    # dispatch count much for this study's enrollment pattern.
    bucket_counts_by_update = [len(b) for b in update_layer.buckets_by_update_index]
    logger.info(
        "Algorithm shape-bucket fan-out: %d update(s), %d subject(s), "
        "%d total bucket(s) (max %d in a single update).",
        len(bucket_counts_by_update),
        len(subject_ids_np),
        sum(bucket_counts_by_update),
        max(bucket_counts_by_update, default=0),
    )

    # 3. Batched forward computation: one jax.vmap call spanning every
    # subject and every global decision time for the Radon-Nikodym weights,
    # plus O(updates * shape buckets) jax.vmap calls for the algorithm and
    # inference estimating functions, instead of one Python-dispatched call
    # per subject (or per subject per update). Computed before the data
    # checks below so pi_beta_grid is available for them to reuse.
    raw_weight_grid, pi_beta_grid = compute_action_prob_layer_outputs(
        action_prob_func,
        action_prob_func_args_beta_index,
        action_prob_layer,
        betas,
        need_pi_beta_grid,
    )
    rl_weight_products, inference_weight_products = compute_windowed_weight_products(
        raw_weight_grid,
        action_prob_layer.active_mask,
        action_prob_layer.lo_idx,
        update_layer.hi_idx,
        action_prob_layer.subject_end_idx,
    )

    # 4. Batched algorithm/inference component computation, run BEFORE the
    # data checks below (step 5) so they can reuse its per-bucket "threaded"
    # (reconstructed-action-prob) results instead of recomputing an
    # identical jax.vmap pass just for validation -- see
    # check_batched_algorithm_estimating_function_args_equivalent's
    # precomputed_threaded_results docstring.
    with log_phase_duration("batched_components.algorithm"):
        algorithm_component, algorithm_bucket_outputs = (
            compute_batched_algorithm_component(
                betas,
                beta_dim,
                algorithm_estimating_func,
                alg_update_func_args_beta_index,
                alg_update_func_args_previous_betas_index,
                alg_update_func_args_action_prob_index,
                action_prob_layer,
                update_layer,
                pi_beta_grid,
                rl_weight_products,
            )
        )
    with log_phase_duration("batched_components.inference"):
        inference_component, inference_hessians, inference_bucket_outputs = (
            compute_batched_inference_outputs(
                theta,
                theta_dim,
                inference_estimating_func,
                inference_func_args_theta_index,
                inference_func_args_action_prob_index,
                action_prob_layer,
                inference_layer,
                pi_beta_grid,
                inference_weight_products,
                need_hessians=include_auxiliary_outputs,
            )
        )

    # 5. Data checks: if action probabilities are used in the algorithm or
    # inference estimating functions, make sure that substituting in the
    # reconstructed action probabilities (as the batched computation above
    # does) is approximately equivalent to using the original action
    # probabilities. Reuses pi_beta_grid (already computed above) instead of
    # re-deriving reconstructed action probabilities via a second,
    # per-subject/per-update dispatched pass through
    # arg_threading_helpers.thread_update_func_args/thread_inference_func_args
    # -- opt-out via suppress_all_data_checks.
    if not suppress_all_data_checks:
        if alg_update_func_args_action_prob_index >= 0:
            logger.info(
                "Checking that reconstructed action probabilities are consistent with "
                "recorded ones in the algorithm update function args for all subjects."
            )
        with log_phase_duration("data_checks.algorithm"):
            check_batched_algorithm_estimating_function_args_equivalent(
                algorithm_estimating_func,
                betas,
                alg_update_func_args_beta_index,
                alg_update_func_args_previous_betas_index,
                alg_update_func_args_action_prob_index,
                action_prob_layer,
                update_layer,
                pi_beta_grid,
                algorithm_bucket_outputs,
            )
        if inference_func_args_action_prob_index >= 0:
            logger.info(
                "Checking that reconstructed action probabilities are consistent with "
                "recorded ones in the inference function args for all subjects."
            )
        with log_phase_duration("data_checks.inference"):
            check_batched_inference_estimating_function_args_equivalent(
                inference_estimating_func,
                theta,
                inference_func_args_theta_index,
                inference_func_args_action_prob_index,
                action_prob_layer,
                inference_layer,
                pi_beta_grid,
                inference_bucket_outputs,
            )

    stacks = jnp.concatenate([algorithm_component, inference_component], axis=1)

    if not include_auxiliary_outputs:
        return jnp.mean(stacks, axis=0)

    # mean_i(outer(stack_i, stack_i)) == (stacks.T @ stacks) / N exactly
    # (sum_i outer(x_i, x_i) == X.T @ X). Computing stacks.T @ stacks
    # directly never materializes the (N, out_dim, out_dim) per-subject
    # tensor jax.vmap(jnp.outer) would -- that tensor was the dominant
    # memory cost at real problem sizes (out_dim in the thousands). See
    # docs/adr/0001-adaptive-sandwich-performance-plan.md.
    outer_products = stacks.T @ stacks
    inference_only_outer_products = inference_component.T @ inference_component

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


def resolve_jacobian_row_chunk_size(
    requested: int | None,
    out_dim: int,
) -> int | None:
    """
    Resolves the jacobian_row_chunk_size request into a concrete decision for
    construct_classical_and_adjusted_sandwiches's backward pass: returns None
    for a single unchunked eager jax.vmap(pullback) over the whole cotangent
    basis, or a positive chunk size for the chunked, jitted backward path.

    - requested > 0: honored verbatim (the pre-auto explicit behavior).
    - requested == 0: force the unchunked single eager vmap -- the pre-auto
      DEFAULT behavior -- even when the auto heuristic would chunk. Fastest
      when it fits in memory.
    - requested < 0: ValueError.
    - requested is None (the default, "auto"): unchunked when
      out_dim <= JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM (small problems, where
      chunking's one-time jit compile was measured ~2.4x SLOWER than the
      plain eager call); otherwise
      max(1, min(JACOBIAN_AUTO_MAX_CHUNK, JACOBIAN_AUTO_ROW_BUDGET // out_dim)),
      which keeps chunk_size * out_dim (the empirically-usable memory proxy)
      at or below every budget verified safe at real oralytics scale and at
      roughly one-third of the one observed crash-while-chunked budget --
      see the constants' own comment in lifejacket/constants.py for the full
      empirical calibration table.

    HONESTY NOTE on the heuristic: out_dim (= num_updates * beta_dim +
    theta_dim) is a single-variable PROXY for the true per-cotangent
    backward-graph footprint, which also grows with num_subjects x
    num_updates x per-subject history length; the thresholds were calibrated
    on ONE study shape (oralytics, beta_dim=135) on ONE 24GB machine. The
    explicit parameter is the escape hatch in both directions: to use LESS
    memory (smaller machine, bigger study, or a crash under auto), pass a
    smaller explicit chunk size (e.g. 8 or 4); to go FASTER when memory is
    known-plentiful, pass a larger explicit chunk size, or 0 to force the
    single unchunked vmap.
    """
    if requested is not None:
        if requested < 0:
            raise ValueError(
                "jacobian_row_chunk_size must be a non-negative int or None "
                "(None = auto, 0 = force a single unchunked backward vmap, "
                f"positive = explicit chunk size), got {requested!r}."
            )
        if requested == 0:
            logger.info(
                "jacobian_row_chunk_size=0: forcing the single unchunked "
                "eager jax.vmap(pullback) backward pass (out_dim=%d).",
                out_dim,
            )
            return None
        logger.info(
            "jacobian_row_chunk_size=%d was passed explicitly; using it as-is "
            "(out_dim=%d).",
            requested,
            out_dim,
        )
        return requested
    if out_dim <= JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM:
        logger.info(
            "jacobian_row_chunk_size=None (auto) resolved to unchunked: "
            "out_dim=%d <= %d. Pass an explicit positive chunk size to chunk "
            "anyway (e.g. if this crashes from memory pressure).",
            out_dim,
            JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM,
        )
        return None
    chunk_size = max(
        1, min(JACOBIAN_AUTO_MAX_CHUNK, JACOBIAN_AUTO_ROW_BUDGET // out_dim)
    )
    logger.info(
        "jacobian_row_chunk_size=None (auto) resolved to chunk size %d: "
        "out_dim=%d > %d, targeting chunk_size * out_dim <= %d (a memory "
        "proxy calibrated on one 24GB-machine study -- pass a smaller "
        "explicit chunk size if memory is tighter, a larger one or 0 for the "
        "unchunked vmap if memory is plentiful).",
        chunk_size,
        out_dim,
        JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM,
        JACOBIAN_AUTO_ROW_BUDGET,
    )
    return chunk_size


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
    form_adjusted_meat_adjustments_explicitly: bool,
    analysis_df: pd.DataFrame | None,
    active_col_name: str | None,
    action_col_name: str | None,
    calendar_t_col_name: str | None,
    subject_id_col_name: str | None,
    action_prob_func_args: tuple | None,
    action_prob_col_name: str | None,
    alg_update_func_args_mask_index: int = -1,
    alg_update_func_args_ragged_indices: tuple[int, ...] = (),
    inference_func_args_mask_index: int = -1,
    inference_func_args_ragged_indices: tuple[int, ...] = (),
    jacobian_row_chunk_size: int | None = None,
    combine_updates_into_one_vmap: bool | None = None,
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
        form_adjusted_meat_adjustments_explicitly (bool):
            If True, explicitly forms the per-subject meat adjustments that differentiate the adjusted
            sandwich from the classical sandwich. This is for diagnostic purposes, as the
            adjusted sandwich is formed without doing this. WARNING: this ends by dropping into an
            interactive debugger (breakpoint()) -- only enable it in an interactive session; see
            form_adjusted_meat_adjustments_directly's own docstring.
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
        alg_update_func_args_mask_index, alg_update_func_args_ragged_indices,
        inference_func_args_mask_index, inference_func_args_ragged_indices:
            Passed straight through to the jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values)
            call this function makes; see that function's own docstring for what they do.
            (compute_local_linearization_error_ratio is called separately, by analyze_dataset
            directly, not from here -- it receives its own copies of these same four arguments.)
        jacobian_row_chunk_size (int | None):
            None (the default) = AUTO: resolved by
            resolve_jacobian_row_chunk_size once out_dim is known -- a
            single, plain eager jax.vmap(pullback) call over the full output
            basis (the original behavior) when out_dim <=
            JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM, else a heuristic chunk size
            max(1, min(JACOBIAN_AUTO_MAX_CHUNK, JACOBIAN_AUTO_ROW_BUDGET //
            out_dim)); see that resolver's docstring, including the honesty
            note that its out_dim threshold is a proxy calibrated on one
            study shape/machine and how to override in both directions.
            0 = force the single unchunked eager vmap even when auto would
            chunk (fastest when it fits in memory). A positive int is an
            explicit chunk size, honored verbatim (the pre-auto opt-in
            behavior, unchanged): the
            jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values)
            call below is instead computed as one jax.vjp call (the
            expensive forward pass, paid exactly once either way) followed
            by ceil(out_dim / jacobian_row_chunk_size) separate backward
            calls, each over at most jacobian_row_chunk_size rows of the
            output-basis identity matrix, concatenated back together --
            mathematically and numerically IDENTICAL to the unchunked call
            (see docs/adr/0001-adaptive-sandwich-performance-plan.md's
            "chunked jacrev" section) regardless of chunk size. When
            chunking is requested this way, each backward call is made
            through a jax.jit-compiled function that takes the forward
            pass's pullback residuals as explicit arguments rather than
            closing over them (compiled once per distinct chunk shape,
            reused for every chunk of that shape -- see the inline comment
            at the call site for the mechanism, why it avoids both
            re-running the forward pass and embedding residuals as XLA
            constants, AND an important measured caveat: on this repo's
            own "medium" benchmark fixture, this jit-compiled path was
            measured to cost MORE wall-clock time than the plain eager
            per-chunk loop it replaces, with no peak-memory improvement,
            because compiling this hot path's large per-update/shape-bucket
            unrolled graph is itself expensive -- so do not assume a
            smaller jacobian_row_chunk_size is a free memory win without
            re-measuring both time and memory at your own scale). Tune this
            per-study/per-machine when the auto heuristic's proxy doesn't
            match your machine or study shape.
        combine_updates_into_one_vmap (bool | None):
            Passed straight through -- UNRESOLVED -- to
            get_avg_weighted_estimating_function_stacks_and_aux_values's own
            argument of the same name, which is where None (the default,
            "auto") is resolved via
            batched_weighted_estimating_function_stack.resolve_combine_updates_into_one_vmap;
            see both docstrings for the semantics (None = enable when
            eligible, with a warn-logged fallback to the default loop on
            mid-precompute structural violations; True = force on with loud
            errors when ineligible; False = force off). NOT threaded into
            the separate local-linearization diagnostic
            (compute_local_linearization_error_ratio, called directly by
            analyze_dataset, not from here) -- that diagnostic continues to
            use the original per-update/per-bucket loop regardless of this
            argument.
    Returns:
        tuple[jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32], jnp.ndarray[jnp.float32]]:
            A tuple containing:
            - The raw joint bread matrix.
            - The joint meat matrix.
            - The joint sandwich matrix.
            - The classical bread matrix.
            - The classical meat matrix.
            - The classical sandwich matrix.
            - The average weighted estimating function stack.
            - All per-subject weighted estimating function stacks.
            - A trivial all-ones array of length num_subjects, kept only for
              output-shape stability (previously the per-subject adjusted
              meat small-sample corrections; there is no small-sample
              correction feature any more).
            - A trivial all-ones array of length num_subjects, kept only for
              output-shape stability (previously the per-subject classical
              meat small-sample corrections).
            - The per-subject adjusted meat contributions, if form_adjusted_meat_adjustments_explicitly
              is True, otherwise an array of NaNs.
    """
    logger.info(
        "Differentiating average weighted estimating function stack and collecting auxiliary values."
    )
    theta_dim = theta_est.shape[0]
    beta_dim = all_post_update_betas.shape[1]
    # NOTE: wrapping this call in jax.jit was tried and reverted -- twice now.
    # See docs/adr/0001-adaptive-sandwich-performance-plan.md's "Step 3" section
    # for both attempts. The first (before ADS-139 Step 4) failed because the
    # per-subject/per-update graph was fully Python-unrolled, giving XLA
    # nothing batched to compile once. The second (after Step 4's jax.vmap
    # batching) compiled fine on its own, but wrapping THIS call specifically
    # in jax.jit was a net regression at medium scale even so: this call runs
    # only once per analyze_dataset invocation (unlike the local-linearization
    # diagnostic below, which now IS jitted and calls the same underlying
    # function 15 times -- see that closure's own comment), so there is no
    # compile-once-reuse-many-times amortization here, and compiling this
    # larger, differentiated graph measurably slowed down even the separate,
    # already-jitted diagnostic closure running later in the same process
    # (0.45s -> 3.6s, reproducible) -- some interaction between the two
    # concurrent jax.jit compilations, not fully root-caused. Net effect at
    # n=100: ~7.1s -> ~16.3-16.5s total wall-clock. Reverted; do not re-add
    # jax.jit here without re-measuring at both benchmark scales.
    # Fail fast on a nonsensical value, before the expensive forward pass;
    # resolve_jacobian_row_chunk_size re-validates and handles the None/0
    # semantics once out_dim is known below.
    if jacobian_row_chunk_size is not None and jacobian_row_chunk_size < 0:
        raise ValueError(
            "jacobian_row_chunk_size must be a non-negative int or None "
            "(None = auto, 0 = force a single unchunked backward vmap, "
            f"positive = explicit chunk size), got {jacobian_row_chunk_size!r}."
        )

    with log_phase_duration(
        "jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values)"
    ):
        # While JAX can technically differentiate with respect to a list of JAX arrays,
        # it is apparently more efficient to flatten them into a single array. This is done
        # here to improve performance. We can simply unflatten them inside the function.
        flattened_betas_and_theta = flatten_params(all_post_update_betas, theta_est)

        def _avg_stack_fn(flattened_x):
            return get_avg_weighted_estimating_function_stacks_and_aux_values(
                flattened_x,
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
                alg_update_func_args_mask_index=alg_update_func_args_mask_index,
                alg_update_func_args_ragged_indices=alg_update_func_args_ragged_indices,
                inference_func_args_mask_index=inference_func_args_mask_index,
                inference_func_args_ragged_indices=inference_func_args_ragged_indices,
                combine_updates_into_one_vmap=combine_updates_into_one_vmap,
            )

        # This is jax.jacrev(f, has_aux=True)(x) manually inlined -- see
        # jax/_src/api.py's jacrev/_std_basis/_vjp (jax 0.4.30): jacrev IS
        # exactly one jax.vjp call (the expensive forward pass -- structural
        # precompute + all batched per-subject/per-update estimating-function
        # evaluations -- paid ONCE, unchanged from today) followed by one
        # jax.vmap(pullback) call over jnp.eye(out_dim). Chunking splits ONLY
        # that second step, which is what was confirmed (real repro, 24GB
        # machine, num_users=30) to exhaust memory at real scale -- see
        # docs/adr/0001-adaptive-sandwich-performance-plan.md's "chunked
        # jacrev" section.
        with log_phase_duration(
            "jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values).forward_vjp"
        ):
            (
                avg_estimating_function_stack,
                pullback,
                (
                    _avg_estimating_function_stack_aux_copy,
                    per_subject_joint_adjusted_meat_contributions,
                    per_subject_classical_meat_contributions,
                    per_subject_classical_bread_contributions,
                    per_subject_estimating_function_stacks,
                ),
            ) = jax.vjp(_avg_stack_fn, flattened_betas_and_theta, has_aux=True)
            # JAX dispatches asynchronously -- without forcing the forward
            # pass's outputs to materialize here, this phase's timer would
            # only measure dispatch overhead, and the real device time would
            # leak into whichever phase next reads these values (see
            # docs/adr/0001-adaptive-sandwich-performance-plan.md's
            # forward/backward split measurement).
            jax.block_until_ready(
                (
                    avg_estimating_function_stack,
                    _avg_estimating_function_stack_aux_copy,
                    per_subject_joint_adjusted_meat_contributions,
                    per_subject_classical_meat_contributions,
                    per_subject_classical_bread_contributions,
                    per_subject_estimating_function_stacks,
                )
            )
            # ru_maxrss is a monotonically-increasing process watermark (never
            # decreases), so logging it here and again after the backward
            # phase below gives the ADDITIONAL peak memory the backward
            # phase caused, isolated from whatever the forward pass (this
            # jax.vjp call's own saved residuals/linearization tape) already
            # used.
            logger.info(
                "Peak RSS after forward_vjp: %.1f MB",
                _peak_rss_mb(),
            )

        out_dim = avg_estimating_function_stack.shape[0]

        # out_dim is first known exactly here (right before the backward
        # branch), so this is where the None="auto" default gets resolved to
        # either "unchunked" (None) or a concrete chunk size -- see
        # resolve_jacobian_row_chunk_size's docstring for the heuristic, its
        # empirical calibration, and its honesty caveats.
        resolved_jacobian_row_chunk_size = resolve_jacobian_row_chunk_size(
            jacobian_row_chunk_size, out_dim
        )

        with log_phase_duration(
            "jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values).backward_vmap_chunks"
        ):
            cotangent_basis = jnp.eye(
                out_dim, dtype=avg_estimating_function_stack.dtype
            )

            if resolved_jacobian_row_chunk_size is None:
                # No chunking (an explicit 0, or auto resolving small
                # out_dim to unchunked): the exact original behavior (a
                # single eager jax.vmap(pullback) call over the whole
                # output basis, unchanged since before this file's own
                # forward/backward split). Deliberately NOT run through the
                # jit machinery below -- see that branch's comment for why:
                # measured on this repo's own "medium" benchmark fixture,
                # jax.jit-compiling this backward step, even via the
                # explicit-residual mechanism that avoids the
                # closure-embedding trap, costs MORE wall-clock time (one
                # XLA compile of the whole per-update/shape-bucket unrolled
                # graph, ~4.8s at that fixture's scale, vs. ~2.0s for the
                # plain eager call) and did not show a peak-memory win
                # either, for a single, un-chunked call that has no
                # repeated-call compile-cost amortization to earn it back.
                raw_joint_bread_matrix = jax.vmap(pullback)(cotangent_basis)[0]
            else:
                effective_chunk_size = min(resolved_jacobian_row_chunk_size, out_dim)

                # Chunking was explicitly requested (trading time for lower
                # peak memory during this phase, per this parameter's own
                # docstring) -- so, unlike the no-chunking branch above,
                # there IS a real compile-once-reuse-many opportunity here
                # if jacobian_row_chunk_size divides out_dim into several
                # same-shaped chunks. This branch uses that opportunity via
                # an explicit-residual jax.jit mechanism rather than the
                # plain eager per-chunk jax.vmap(pullback) loop an earlier
                # version of this code used:
                #
                # A naive "jax.jit(lambda c: jax.vmap(pullback)(c))" closes
                # over `pullback`, and jax.jit embeds a closed-over
                # jax.vjp pullback's captured forward-pass residuals as XLA
                # literal constants in the compiled program -- confirmed
                # (via compiled-HLO constant counts, on a toy problem) to
                # make peak memory WORSE than the plain eager loop, with no
                # speed benefit. That is the same closure-embedding
                # pathology already diagnosed and fixed for
                # _eval_avg_stack_jit elsewhere in this file (see
                # _extract_diagnostic_jit_arrays/
                # _rebuild_precomputes_from_jit_arrays).
                #
                # The mechanism below avoids that specific trap by getting
                # pullback's captured residuals OUT of the closure and into
                # explicit jit arguments, using only public jax.tree_util
                # API. jax.vjp's returned pullback is deliberately built
                # (jax/_src/interpreters/ad.py's `vjp`, jax 0.4.30: "Ensure
                # that vjp_ is a PyTree so that we can pass it from the
                # forward to the backward pass in a custom VJP") as a
                # jax.tree_util.Partial chain with the forward-pass
                # residuals inside `.args` -- a registered pytree, not an
                # opaque closure. So jax.tree_util.tree_flatten(pullback)
                # yields those residual arrays as genuine flat leaves (with
                # the jaxpr/pvals/avals structure captured as a static
                # treedef, not as data), and a function written to take
                # (cotangent_chunk, *residual_leaves) as REAL arguments --
                # jax.jit-compiled ONCE below -- gets those residuals
                # passed as true runtime inputs on every one of the
                # ceil(out_dim / effective_chunk_size) calls, not baked in
                # as literals. jax.jit compiles once per distinct chunk
                # shape it is called with (the same "compile once per
                # shape" pattern already relied on elsewhere in this file
                # for _eval_avg_stack_jit) -- at most twice here, since
                # every chunk has effective_chunk_size rows except possibly
                # a shorter final remainder chunk.
                #
                # IMPORTANT, measured finding (not just a toy-problem
                # extrapolation -- see this session's own toy benchmarking
                # for that separately) -- this mechanism was ALSO measured
                # directly against the real
                # get_avg_weighted_estimating_function_stacks_and_aux_values
                # hot path, using this repo's own "medium" benchmark
                # fixture (n=100, T=10, out_dim=56): there, jax.jit
                # compiling this backward step -- even via this
                # closure-avoiding mechanism -- did NOT reduce peak memory
                # relative to the plain eager per-chunk jax.vmap(pullback)
                # loop, and cost substantially MORE wall-clock time (each
                # distinct chunk shape's XLA compile took several seconds
                # by itself, dwarfing the eager loop's total backward-phase
                # time at this fixture's scale). This directly contradicts
                # the toy-problem prediction that motivated trying this
                # mechanism, and reinforces -- with a second, independent,
                # real-hot-path data point -- this same function's
                # historical note (above, near the top of
                # construct_classical_and_adjusted_sandwiches) that
                # jax.jit-compiling this specific hot path has already
                # once been tried and reverted for a similar compile-cost
                # reason. The mechanism is kept here, gated behind the
                # explicit jacobian_row_chunk_size opt-in, because: (a) it
                # is still the only verified way to jit this step at all
                # without the (worse) closure-embedding memory regression;
                # (b) this repo's own fixtures are far smaller
                # (beta_dim/out_dim) than the real oralytics-scale crash
                # this chunking knob exists for, so a real-scale verdict on
                # whether the compile cost is worth it cannot be reached
                # here (see this function's own module-level ADR notes);
                # and (c) it is correctness-preserving either way. Anyone
                # tuning jacobian_row_chunk_size for a real memory crisis
                # should re-measure both wall-clock time AND peak memory at
                # their own scale before assuming this mechanism helps --
                # do not assume it is a free win.
                #
                # This also relies on an internal-but-deliberate JAX
                # implementation detail (pullback being a flatten-able
                # Partial chain with residuals in .args), not a documented
                # part of jax.vjp's public contract, so we defensively fall
                # back to the plain eager loop if tree_flatten ever yields
                # zero leaves (e.g. a future jax release restructuring
                # pullback) -- correctness is unaffected either way; only
                # the (already-uncertain, per above) memory/speed tradeoff
                # would be affected, silently, without this fallback ever
                # being exercised in CI today.
                residual_leaves, pullback_treedef = jax.tree_util.tree_flatten(pullback)

                if not residual_leaves:
                    raw_joint_bread_matrix = jnp.concatenate(
                        [
                            jax.vmap(pullback)(
                                cotangent_basis[start : start + effective_chunk_size]
                            )[0]
                            for start in range(0, out_dim, effective_chunk_size)
                        ],
                        axis=0,
                    )
                else:

                    def _backward_chunk(cotangent_chunk, *leaves):
                        chunk_pullback = jax.tree_util.tree_unflatten(
                            pullback_treedef, list(leaves)
                        )
                        return jax.vmap(chunk_pullback)(cotangent_chunk)[0]

                    jitted_backward_chunk = jax.jit(_backward_chunk)

                    with log_phase_duration(
                        "jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values)"
                        ".backward_vmap_chunks.jit_compile"
                    ):
                        first_chunk_size = min(effective_chunk_size, out_dim)
                        compiled_backward_chunk = jitted_backward_chunk.lower(
                            cotangent_basis[0:first_chunk_size], *residual_leaves
                        ).compile()

                    with log_phase_duration(
                        "jax.jacrev(get_avg_weighted_estimating_function_stacks_and_aux_values)"
                        ".backward_vmap_chunks.chunk_calls"
                    ):
                        chunk_parts = []
                        for start in range(0, out_dim, effective_chunk_size):
                            chunk = cotangent_basis[
                                start : start + effective_chunk_size
                            ]
                            if chunk.shape[0] == first_chunk_size:
                                chunk_parts.append(
                                    compiled_backward_chunk(chunk, *residual_leaves)
                                )
                            else:
                                # A shorter final remainder chunk -- a
                                # differently-shaped input, so jax.jit
                                # compiles once more for this one distinct
                                # shape (and would reuse that compilation
                                # if this shape recurred, though it never
                                # does here since there is at most one
                                # remainder chunk).
                                chunk_parts.append(
                                    jitted_backward_chunk(chunk, *residual_leaves)
                                )
                        raw_joint_bread_matrix = jnp.concatenate(chunk_parts, axis=0)

            jax.block_until_ready(raw_joint_bread_matrix)
            logger.info(
                "Peak RSS after backward_vmap_chunks: %.1f MB",
                _peak_rss_mb(),
            )

    num_subjects = len(subject_ids)

    with log_phase_duration("compute_meat_matrices"):
        # per_subject_joint_adjusted_meat_contributions and
        # per_subject_classical_meat_contributions are already pre-summed
        # across subjects (stacks.T @ stacks and inference_component.T @
        # inference_component -- see
        # get_avg_weighted_estimating_function_stacks_and_aux_values), so
        # the meat matrices are just their per-subject average.
        # per_subject_adjusted_corrections/per_subject_classical_corrections
        # are kept as trivial all-ones arrays purely to preserve this
        # function's output shape (there is no longer any small-sample
        # correction to report here).
        joint_adjusted_meat_matrix = (
            per_subject_joint_adjusted_meat_contributions / num_subjects
        )
        classical_meat_matrix = per_subject_classical_meat_contributions / num_subjects
        per_subject_adjusted_corrections = jnp.ones(num_subjects)
        per_subject_classical_corrections = jnp.ones(num_subjects)
        # This is the phase this session's real oralytics-scale run was
        # observed to crash inside, back when this step materialized a
        # per-subject (N, out_dim, out_dim) tensor (see
        # docs/adr/0001-adaptive-sandwich-performance-plan.md); force
        # materialization and log peak RSS here, same pattern as the
        # forward_vjp/backward_vmap_chunks phases above, to confirm that
        # cost is gone.
        jax.block_until_ready((joint_adjusted_meat_matrix, classical_meat_matrix))
        logger.info(
            "Peak RSS after compute_meat_matrices: %.1f MB",
            _peak_rss_mb(),
        )

    # Now stably (no explicit inversion) form our sandwiches.
    with log_phase_duration("form_sandwich_from_bread_and_meat (joint)"):
        joint_sandwich = form_sandwich_from_bread_and_meat(
            raw_joint_bread_matrix,
            joint_adjusted_meat_matrix,
            num_subjects,
            method=SandwichFormationMethods.BREAD_T_QR,
        )
    with log_phase_duration("form_sandwich_from_bread_and_meat (classical)"):
        classical_bread_matrix = jnp.mean(
            per_subject_classical_bread_contributions, axis=0
        )
        classical_sandwich = form_sandwich_from_bread_and_meat(
            classical_bread_matrix,
            classical_meat_matrix,
            num_subjects,
            method=SandwichFormationMethods.BREAD_T_QR,
        )
    jax.block_until_ready((joint_sandwich, classical_bread_matrix, classical_sandwich))
    logger.info(
        "Peak RSS after form_sandwich_from_bread_and_meat: %.1f MB",
        _peak_rss_mb(),
    )

    per_subject_adjusted_meat_contributions = jnp.full(
        (len(subject_ids), theta_dim, theta_dim), jnp.nan
    )
    if form_adjusted_meat_adjustments_explicitly:
        with log_phase_duration("form_adjusted_meat_adjustments_directly (diagnostic)"):
            per_subject_adjusted_meat_contributions = (
                form_adjusted_meat_adjustments_directly(
                    theta_dim,
                    all_post_update_betas.shape[1],
                    raw_joint_bread_matrix,
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
            # per_subject_adjusted_meat_contributions is a genuine
            # per-subject (num_subjects, theta_dim, theta_dim) tensor, freshly
            # computed by form_adjusted_meat_adjustments_directly just above
            # (unlike joint_adjusted_meat_matrix/classical_meat_matrix above,
            # which are pre-summed), so its meat matrix is just its
            # per-subject mean.
            theta_only_adjusted_meat_matrix_v2 = jnp.mean(
                per_subject_adjusted_meat_contributions, axis=0
            )
            theta_only_adjusted_sandwich_from_adjustments = (
                form_sandwich_from_bread_and_meat(
                    classical_bread_matrix,
                    theta_only_adjusted_meat_matrix_v2,
                    num_subjects,
                    method=SandwichFormationMethods.BREAD_T_QR,
                )
            )
            theta_only_adjusted_sandwich = joint_sandwich[-theta_dim:, -theta_dim:]

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
        raw_joint_bread_matrix,
        joint_adjusted_meat_matrix,
        joint_sandwich,
        classical_bread_matrix,
        classical_meat_matrix,
        classical_sandwich,
        avg_estimating_function_stack,
        per_subject_estimating_function_stacks,
        per_subject_adjusted_corrections,
        per_subject_classical_corrections,
        per_subject_adjusted_meat_contributions,
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


def compute_eigenvalue_and_condition_diagnostics(
    raw_joint_bread_matrix: jnp.ndarray,
    joint_sandwich_matrix: jnp.ndarray,
    adjusted_sandwich_var_estimate: jnp.ndarray,
) -> tuple[float, float, np.ndarray, float, float, np.ndarray, float]:
    """
    Computes condition-number and eigenvalue-spread diagnostics for the joint
    bread matrix and both the joint and theta-only adjusted sandwich matrices.

    Args:
        raw_joint_bread_matrix (jnp.ndarray):
            The (unstabilized) joint bread matrix.
        joint_sandwich_matrix (jnp.ndarray):
            The joint (betas and theta) sandwich variance matrix.
        adjusted_sandwich_var_estimate (jnp.ndarray):
            The theta-only slice of the adjusted sandwich variance matrix.

    Returns:
        tuple:
            - joint_bread_cond (float): condition number of the raw joint bread matrix.
            - max_eig_joint (float): max eigenvalue of the joint sandwich matrix.
            - eigvals_joint_sandwich (np.ndarray): all eigenvalues of the joint
              sandwich matrix.
            - max_to_median_ratio_joint_sandwich (float): max/median eigenvalue ratio
              for the joint sandwich matrix, over eigenvalues >= 1e-8 * max.
            - max_eig_theta (float): max eigenvalue of the theta-only adjusted
              sandwich matrix.
            - eigvals_theta_only_adjusted_sandwich (np.ndarray): all eigenvalues of
              the theta-only adjusted sandwich matrix.
            - max_to_median_ratio_theta_only_adjusted_sandwich (float): max/median
              eigenvalue ratio for the theta-only adjusted sandwich matrix, over
              eigenvalues >= 1e-8 * max.
    """
    joint_bread_cond = jnp.linalg.cond(raw_joint_bread_matrix)
    logger.info(
        "Joint bread condition number: %f",
        joint_bread_cond,
    )

    # calculate the max eigenvalue of the theta-only adjusted sandwich
    eigvals_theta_only_adjusted_sandwich = scipy.linalg.eigvalsh(
        adjusted_sandwich_var_estimate
    )
    max_eig_theta = float(eigvals_theta_only_adjusted_sandwich.max())
    logger.info(
        "Max eigenvalue of theta-only adjusted sandwich matrix: %f",
        max_eig_theta,
    )

    # Compute ratios: max eigenvalue / median eigenvalue among those >= 1e-8 * max.
    eigvals_joint_sandwich = scipy.linalg.eigvalsh(joint_sandwich_matrix)
    max_eig_joint = float(eigvals_joint_sandwich.max())
    logger.info(
        "Max eigenvalue of joint adjusted sandwich matrix: %f",
        max_eig_joint,
    )

    joint_keep = eigvals_joint_sandwich >= (1e-8 * max_eig_joint)
    joint_median_kept = (
        float(np.median(eigvals_joint_sandwich[joint_keep]))
        if np.any(joint_keep)
        else math.nan
    )
    max_to_median_ratio_joint_sandwich = (
        (max_eig_joint / joint_median_kept)
        if (not math.isnan(joint_median_kept) and joint_median_kept > 0)
        else (
            math.inf
            if (not math.isnan(joint_median_kept) and joint_median_kept == 0)
            else math.nan
        )
    )
    logger.info(
        "Max/median eigenvalue ratio (joint sandwich; median over eigvals >= 1e-8*max): %f",
        max_to_median_ratio_joint_sandwich,
    )

    theta_keep = eigvals_theta_only_adjusted_sandwich >= (1e-8 * max_eig_theta)
    theta_median_kept = (
        float(np.median(eigvals_theta_only_adjusted_sandwich[theta_keep]))
        if np.any(theta_keep)
        else math.nan
    )
    max_to_median_ratio_theta_only_adjusted_sandwich = (
        (max_eig_theta / theta_median_kept)
        if (not math.isnan(theta_median_kept) and theta_median_kept > 0)
        else (
            math.inf
            if (not math.isnan(theta_median_kept) and theta_median_kept == 0)
            else math.nan
        )
    )
    logger.info(
        "Max/median eigenvalue ratio (theta-only adjusted sandwich; median over eigvals >= 1e-8*max): %f",
        max_to_median_ratio_theta_only_adjusted_sandwich,
    )

    return (
        joint_bread_cond,
        max_eig_joint,
        eigvals_joint_sandwich,
        max_to_median_ratio_joint_sandwich,
        max_eig_theta,
        eigvals_theta_only_adjusted_sandwich,
        max_to_median_ratio_theta_only_adjusted_sandwich,
    )


class _BucketJitArrays(typing.NamedTuple):
    """
    Traced-argument counterpart of one UpdateArgBucket: subject_positions
    (read as a scatter index by compute_batched_algorithm_component /
    compute_batched_inference_outputs' `.at[bucket.subject_positions,
    ...].set(...)`) and the pre-stacked tensor for every "stackable"
    raw_arg_lists position (every position that is not overridden by
    _build_algorithm_bucket_overrides/_build_inference_bucket_overrides and
    not all-None -- see _bucket_stackable_positions). A plain
    typing.NamedTuple is a first-class JAX pytree with no registration
    needed (see Investigation A/B discussion above this module's
    compute_local_linearization_error_ratio).

    Deliberately does NOT carry the position indices themselves (which
    positions were stacked, in what order) -- those are static Python ints,
    recomputed identically by _bucket_stackable_positions from the
    (closed-over, never-traced) base bucket + override_positions at both
    extract and rebuild time, so they never need to flow through this
    traced pytree at all.
    """

    subject_positions: jnp.ndarray  # (bucket_size,) int
    stacked_tensors: tuple[jnp.ndarray, ...]  # one per stackable position


class _ActionProbLayerJitArrays(typing.NamedTuple):
    """
    Traced-argument counterpart of ActionProbLayerPrecompute's (N, T, ...)
    / (N,)-shaped fields that compute_action_prob_layer_outputs /
    compute_windowed_weight_products read as real array VALUES (not merely
    a `.shape`). ActionProbLayerPrecompute's other fields (subject_ids,
    time_values, time_to_col, fill_col_index, subject_start_idx,
    min_time_by_policy_num, subject_id_to_pos, T) are either never read by
    those two functions, or read only for small/scalar Python-level
    bookkeeping -- never as data flowing into a jnp op -- so they stay
    closed-over Python/numpy constants on the base object (see
    _rebuild_precomputes_from_jit_arrays); embedding those as XLA constants
    is not the pathology this module exists to avoid, since their size
    doesn't scale with N*T the way the fields below do.
    """

    active_mask: jnp.ndarray  # (N, T) bool
    beta_row_index: jnp.ndarray  # (N, T) int
    actions_grid: jnp.ndarray  # (N, T) int
    raw_arg_tensors: tuple[jnp.ndarray, ...]  # each (N, T, *shape_k)
    lo_idx: jnp.ndarray  # (N,) int
    subject_end_idx: jnp.ndarray  # (N,) int


class _UpdateLayerJitArrays(typing.NamedTuple):
    hi_idx: jnp.ndarray  # (N, U) int
    valid_update: jnp.ndarray  # (N, U) bool
    buckets_by_update_index: tuple[tuple[_BucketJitArrays, ...], ...]


class _InferenceLayerJitArrays(typing.NamedTuple):
    buckets: tuple[_BucketJitArrays, ...]


class _DiagnosticJitArrays(typing.NamedTuple):
    """
    The flat pytree that becomes _eval_avg_stack_jit's second, genuinely
    traced argument -- see that closure's own comment for why this (as
    opposed to closing over the precompute objects directly) is the actual
    fix for the compile-time-scales-with-N pathology.
    """

    action_prob: _ActionProbLayerJitArrays
    update: _UpdateLayerJitArrays
    inference: _InferenceLayerJitArrays


def _algorithm_override_positions(
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
) -> set[int]:
    """
    The raw_arg_lists positions _build_algorithm_bucket_overrides always
    (beta) or conditionally (previous-betas, action-prob, each only if its
    *_index is >= 0) overrides for every bucket at every update -- these
    depend only on this static config, never on which bucket. A bucket's
    raw_arg_lists value at one of these positions is never actually read as
    the estimating function's real input (see _assemble_call_args_and_in_axes:
    override_position_values entries replace it entirely before the
    jax.vmap call), so it never needs to become a traced array -- mirrors
    _assemble_call_args_and_in_axes's own "remaining_positions = not in
    override_position_values" split.
    """
    positions = {alg_update_func_args_beta_index}
    if alg_update_func_args_previous_betas_index >= 0:
        positions.add(alg_update_func_args_previous_betas_index)
    if alg_update_func_args_action_prob_index >= 0:
        positions.add(alg_update_func_args_action_prob_index)
    return positions


def _inference_override_positions(
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
) -> set[int]:
    """Same idea as _algorithm_override_positions, for inference buckets."""
    positions = {inference_func_args_theta_index}
    if inference_func_args_action_prob_index >= 0:
        positions.add(inference_func_args_action_prob_index)
    return positions


def _bucket_stackable_positions(
    bucket: UpdateArgBucket, override_positions: set[int]
) -> list[int]:
    """
    Which of bucket.raw_arg_lists' positions get pre-stacked into a traced
    tensor by _extract_bucket_jit_arrays/_rebuild_bucket_from_jit_arrays --
    every position that is neither overridden nor all-None, exactly
    mirroring _assemble_call_args_and_in_axes's own
    remaining_positions + _stackable_positions computation. Keeping this as
    one small, shared helper (rather than two separately-written position
    lists in extract vs. rebuild) is what guarantees extraction and rebuild
    stay in lockstep.
    """
    num_args = len(bucket.raw_arg_lists)
    remaining_positions = [k for k in range(num_args) if k not in override_positions]
    return _stackable_positions(bucket.raw_arg_lists, remaining_positions)


def _extract_bucket_jit_arrays(
    bucket: UpdateArgBucket, override_positions: set[int]
) -> _BucketJitArrays:
    stackable = _bucket_stackable_positions(bucket, override_positions)
    stacked_tensors, _ = stack_batched_arg_lists_into_tensors(
        [bucket.raw_arg_lists[pos] for pos in stackable]
    )
    return _BucketJitArrays(
        subject_positions=jnp.asarray(bucket.subject_positions),
        stacked_tensors=tuple(stacked_tensors),
    )


def _rebuild_bucket_from_jit_arrays(
    base_bucket: UpdateArgBucket,
    bucket_jit_arrays: _BucketJitArrays,
    override_positions: set[int],
) -> UpdateArgBucket:
    stackable = _bucket_stackable_positions(base_bucket, override_positions)
    new_raw_arg_lists = list(base_bucket.raw_arg_lists)
    for pos, tensor in zip(stackable, bucket_jit_arrays.stacked_tensors, strict=True):
        # Store the traced (bucket_size, ...) tensor directly, rather than
        # unstacking it into a Python list of per-subject slices via
        # list(tensor): stack_batched_arg_lists_into_tensors
        # (vmap_helpers.py) recognizes an already-array-shaped position and
        # passes it through as-is, instead of re-deriving the identical
        # tensor via a slice-then-restack round trip that would otherwise
        # add two graph ops per subject in the bucket for no benefit --
        # get_avg_weighted_estimating_function_stacks_and_aux_values still
        # runs completely unmodified once precomputed_layers is substituted
        # in; only the internal representation of an already-stacked
        # position changed, not the numeric result.
        new_raw_arg_lists[pos] = tensor
    return dataclasses.replace(
        base_bucket,
        subject_positions=bucket_jit_arrays.subject_positions,
        raw_arg_lists=new_raw_arg_lists,
    )


def _extract_diagnostic_jit_arrays(
    action_prob_layer: ActionProbLayerPrecompute,
    update_layer: UpdateLayerPrecompute,
    inference_layer: InferenceLayerPrecompute,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
) -> _DiagnosticJitArrays:
    """
    Pulls every array-valued field that compute_action_prob_layer_outputs /
    compute_windowed_weight_products / compute_batched_algorithm_component /
    compute_batched_inference_outputs read as real array DATA (as opposed to
    merely consulting a `.shape`, or reading small/scalar Python-level
    metadata) out of the three (concrete, numpy-valued) structural
    precompute objects, as a flat pytree of jnp arrays. Called once,
    eagerly (outside any jit trace), in compute_local_linearization_error_ratio.

    This flat pytree is what becomes _eval_avg_stack_jit's second,
    genuinely traced argument: every leaf here is data that scales with the
    number of subjects (N) and/or decision times (T) -- exactly the data
    that must NOT be left as a Python closure over the base precompute
    objects, since a jax.jit trace bakes any closed-over concrete array
    into the compiled program as an XLA literal constant, and
    large-constant embedding/optimization is what made compile time balloon
    with N before this fix (see
    docs/adr/0001-adaptive-sandwich-performance-plan.md and this function's
    call site's own comments). See _ActionProbLayerJitArrays/_BucketJitArrays
    for exactly which fields, and why the rest are left as closed-over
    Python constants instead.
    """
    algo_overrides = _algorithm_override_positions(
        alg_update_func_args_beta_index,
        alg_update_func_args_previous_betas_index,
        alg_update_func_args_action_prob_index,
    )
    inference_overrides = _inference_override_positions(
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
    )

    action_prob_jit_arrays = _ActionProbLayerJitArrays(
        active_mask=jnp.asarray(action_prob_layer.active_mask),
        beta_row_index=jnp.asarray(action_prob_layer.beta_row_index),
        actions_grid=jnp.asarray(action_prob_layer.actions_grid),
        raw_arg_tensors=tuple(
            jnp.asarray(t) for t in action_prob_layer.raw_arg_tensors
        ),
        lo_idx=jnp.asarray(action_prob_layer.lo_idx),
        subject_end_idx=jnp.asarray(action_prob_layer.subject_end_idx),
    )
    update_jit_arrays = _UpdateLayerJitArrays(
        hi_idx=jnp.asarray(update_layer.hi_idx),
        valid_update=jnp.asarray(update_layer.valid_update),
        buckets_by_update_index=tuple(
            tuple(
                _extract_bucket_jit_arrays(bucket, algo_overrides) for bucket in buckets
            )
            for buckets in update_layer.buckets_by_update_index
        ),
    )
    inference_jit_arrays = _InferenceLayerJitArrays(
        buckets=tuple(
            _extract_bucket_jit_arrays(bucket, inference_overrides)
            for bucket in inference_layer.buckets
        )
    )
    return _DiagnosticJitArrays(
        action_prob=action_prob_jit_arrays,
        update=update_jit_arrays,
        inference=inference_jit_arrays,
    )


def _rebuild_precomputes_from_jit_arrays(
    base_action_prob_layer: ActionProbLayerPrecompute,
    base_update_layer: UpdateLayerPrecompute,
    base_inference_layer: InferenceLayerPrecompute,
    jit_arrays: _DiagnosticJitArrays,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_previous_betas_index: int,
    alg_update_func_args_action_prob_index: int,
    inference_func_args_theta_index: int,
    inference_func_args_action_prob_index: int,
) -> tuple[ActionProbLayerPrecompute, UpdateLayerPrecompute, InferenceLayerPrecompute]:
    """
    Inverse of _extract_diagnostic_jit_arrays -- called INSIDE the jax.jit
    trace (from _eval_avg_stack_jit's body): reconstructs three precompute
    objects equivalent to the base ones, but with every extracted field's
    value coming from jit_arrays' traced leaves instead of the base
    objects' closed-over concrete arrays.

    dataclasses.replace(base_object, **traced_fields) works directly here
    with no pytree registration needed at all: ActionProbLayerPrecompute /
    UpdateLayerPrecompute / InferenceLayerPrecompute / UpdateArgBucket are
    all plain frozen dataclasses, so replace() just builds a fresh instance
    with the given fields overridden by tracers, copying every
    non-overridden field (time_to_col, subject_id_to_pos,
    min_time_by_policy_num, subject_ids_in_order,
    action_prob_times_by_subject, policy_nums_by_update_index, etc.)
    through unchanged as ordinary closed-over Python objects -- this is
    Investigation B's "manual extract/rebuild via dataclasses.replace"
    design.
    """
    algo_overrides = _algorithm_override_positions(
        alg_update_func_args_beta_index,
        alg_update_func_args_previous_betas_index,
        alg_update_func_args_action_prob_index,
    )
    inference_overrides = _inference_override_positions(
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
    )

    action_prob_layer = dataclasses.replace(
        base_action_prob_layer,
        active_mask=jit_arrays.action_prob.active_mask,
        beta_row_index=jit_arrays.action_prob.beta_row_index,
        actions_grid=jit_arrays.action_prob.actions_grid,
        raw_arg_tensors=jit_arrays.action_prob.raw_arg_tensors,
        lo_idx=jit_arrays.action_prob.lo_idx,
        subject_end_idx=jit_arrays.action_prob.subject_end_idx,
    )
    update_layer = dataclasses.replace(
        base_update_layer,
        hi_idx=jit_arrays.update.hi_idx,
        valid_update=jit_arrays.update.valid_update,
        buckets_by_update_index=[
            [
                _rebuild_bucket_from_jit_arrays(
                    base_bucket, bucket_jit_arrays, algo_overrides
                )
                for base_bucket, bucket_jit_arrays in zip(
                    base_buckets, jit_buckets, strict=True
                )
            ]
            for base_buckets, jit_buckets in zip(
                base_update_layer.buckets_by_update_index,
                jit_arrays.update.buckets_by_update_index,
                strict=True,
            )
        ],
    )
    inference_layer = dataclasses.replace(
        base_inference_layer,
        buckets=[
            _rebuild_bucket_from_jit_arrays(
                base_bucket, bucket_jit_arrays, inference_overrides
            )
            for base_bucket, bucket_jit_arrays in zip(
                base_inference_layer.buckets,
                jit_arrays.inference.buckets,
                strict=True,
            )
        ],
    )
    return action_prob_layer, update_layer, inference_layer


def compute_local_linearization_error_ratio(
    joint_bread_matrix: jnp.ndarray,
    avg_estimating_function_stack: jnp.ndarray,
    per_subject_estimating_function_stacks: jnp.ndarray,
    all_post_update_betas: jnp.ndarray,
    theta_est: jnp.ndarray,
    beta_dim: int,
    theta_dim: int,
    subject_ids: jnp.ndarray,
    action_prob_func: Callable,
    action_prob_func_args_beta_index: int,
    alg_update_func: Callable,
    alg_update_func_type: str,
    alg_update_func_args_beta_index: int,
    alg_update_func_args_action_prob_index: int,
    alg_update_func_args_action_prob_times_index: int,
    alg_update_func_args_previous_betas_index: int,
    inference_func: Callable,
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
    alg_update_func_args_mask_index: int = -1,
    alg_update_func_args_ragged_indices: tuple[int, ...] = (),
    inference_func_args_mask_index: int = -1,
    inference_func_args_ragged_indices: tuple[int, ...] = (),
) -> tuple[float, float, float]:
    """
    Compares the nonlinear Taylor remainder of the joint estimating-function map to
    the retained linear term, at perturbations on the O(1/sqrt(n)) scale, as a
    necessary/sanity diagnostic that the first-order linearization the sandwich
    variance relies on is locally accurate at the estimation scale.

    Define r(delta) = || g(eta+delta) - g(eta) - B delta ||_2 / || B delta ||_2,
    where g(eta) is the avg per-subject weighted estimating-function stack and B is
    the joint bread (Jacobian of g w.r.t. flattened betas+theta). This ratio is
    dimensionless.

    Args:
        joint_bread_matrix (jnp.ndarray):
            The joint bread matrix, B above.
        avg_estimating_function_stack (jnp.ndarray):
            g(eta) above, evaluated at the final estimate.
        per_subject_estimating_function_stacks (jnp.ndarray):
            Per-subject estimating function stacks, used to align perturbation draws
            with the empirical joint estimating-function stack covariance.
        all_post_update_betas (jnp.ndarray):
            All parameter estimates for the algorithm updates.
        theta_est (jnp.ndarray):
            The parameter estimate for inference.
        beta_dim, theta_dim, subject_ids, action_prob_func,
        action_prob_func_args_beta_index, alg_update_func, alg_update_func_type,
        alg_update_func_args_beta_index, alg_update_func_args_action_prob_index,
        alg_update_func_args_action_prob_times_index,
        alg_update_func_args_previous_betas_index, inference_func,
        inference_func_type, inference_func_args_theta_index,
        inference_func_args_action_prob_index,
        action_prob_func_args_by_subject_id_by_decision_time,
        policy_num_by_decision_time_by_subject_id, initial_policy_num,
        beta_index_by_policy_num, inference_func_args_by_subject_id,
        inference_action_prob_decision_times_by_subject_id,
        update_func_args_by_by_subject_id_by_policy_num,
        action_by_decision_time_by_subject_id,
        alg_update_func_args_mask_index, alg_update_func_args_ragged_indices,
        inference_func_args_mask_index, inference_func_args_ragged_indices:
            Passed straight through to get_avg_weighted_estimating_function_stacks_and_aux_values
            (via this function's own build_update_layer_precompute/build_inference_layer_precompute
            calls, since this diagnostic builds its structural precompute directly rather than
            through that function); see that function's own docstring for what the four
            mask/ragged-indices arguments do.

    Returns:
        tuple[float, float, float]:
            The median, 90th percentile, and max local linearization error ratio
            over the sampled perturbations.
    """
    joint_bread = joint_bread_matrix
    g_hat = avg_estimating_function_stack
    stacks = per_subject_estimating_function_stacks

    # Add a small ridge to improve numerical stability for ill-conditioned bread.
    ridge_scale = 1e-8

    # Only add a ridge when the (possibly-already-stabilized) bread is still numerically problematic.
    cond_threshold = 1e12
    bread_cond = float(jnp.linalg.cond(joint_bread))

    if (not math.isfinite(bread_cond)) or (bread_cond > cond_threshold):
        diag_scale = jnp.max(jnp.abs(jnp.diag(joint_bread)))
        ridge = ridge_scale * jnp.where(diag_scale > 0, diag_scale, 1.0)
        joint_bread = joint_bread + ridge * jnp.eye(joint_bread.shape[0])
        logger.info(
            "Added ridge %.3e to joint bread for diagnostic solve (cond=%.3e, threshold=%.3e).",
            float(ridge),
            bread_cond,
            cond_threshold,
        )

    num_subjects = stacks.shape[0]

    # One-time structural precompute, built exactly the way
    # get_avg_weighted_estimating_function_stacks_and_aux_values would build
    # it itself (same build_*_precompute calls, same raw arguments) -- but
    # built HERE, once, in plain eager numpy, so it can be extracted into a
    # traced jit argument below instead of being closed over by
    # _eval_avg_stack_jit. See _extract_diagnostic_jit_arrays's own
    # docstring for why closing over these objects directly (the previous
    # version of this fix) is exactly what caused compile time to balloon
    # with the number of subjects.
    subject_ids_np = np.asarray(subject_ids.tolist())
    base_action_prob_layer = build_action_prob_layer_precompute(
        subject_ids_np,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )
    base_update_layer = build_update_layer_precompute(
        subject_ids_np,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        alg_update_func_args_action_prob_times_index,
        base_action_prob_layer,
        alg_update_func_args_mask_index,
        alg_update_func_args_ragged_indices,
    )
    base_inference_layer = build_inference_layer_precompute(
        inference_func_args_by_subject_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_subject_id,
        base_action_prob_layer,
        inference_func_args_mask_index,
        inference_func_args_ragged_indices,
    )
    jit_arrays = _extract_diagnostic_jit_arrays(
        base_action_prob_layer,
        base_update_layer,
        base_inference_layer,
        alg_update_func_args_beta_index,
        alg_update_func_args_previous_betas_index,
        alg_update_func_args_action_prob_index,
        inference_func_args_theta_index,
        inference_func_args_action_prob_index,
    )

    # This closure is jit-friendly by construction: suppress_all_data_checks and
    # include_auxiliary_outputs are hardcoded below (never traced), so the
    # np.testing.assert_allclose/.tolist()/int() concretization hazards that
    # broke the earlier, ADR-documented jax.jit attempt on the differentiated
    # call never enter this trace. Since ADS-139 Step 4 replaced the ragged
    # per-subject/per-update Python loops with jax.vmap batching, this now
    # traces to a small, compile-once graph -- worth jitting since this
    # closure is called J=15 times below (this diagnostic previously
    # dominated wall-clock, per this ADR's Step 0 benchmark).
    #
    # LANDMINE: do not change suppress_all_data_checks to a real,
    # non-hardcoded value here. check_batched_algorithm_estimating_function_args_equivalent
    # / check_batched_inference_estimating_function_args_equivalent (called
    # when suppress_all_data_checks is False) call np.asarray() on their
    # results -- fine under jax.jacrev's concrete-valued autodiff tracing
    # (which is all they run under elsewhere), but under jax.jit's abstract
    # tracing every value in the trace becomes a non-concrete tracer
    # (confirmed empirically: even a jax.lax.stop_gradient'd value raises
    # TracerArrayConversionError here, unlike under jacrev alone), so those
    # checks would hard-crash the moment this closure ever traced with
    # checks enabled.
    #
    # This closure must stay nested here (rather than hoisted to module level like
    # the outer function around it) because it closes over the static config above
    # and below: jax.jit needs those values fixed at trace time, and several of
    # them (the dict-typed args) are unhashable, so they can't be passed as jit
    # static_argnums/static_argnames to a module-level function instead.
    #
    # jit_arrays is this closure's SECOND argument (not closed over) --
    # deliberately, so its leaves are real jit trace parameters rather than
    # Python constants XLA would otherwise have to embed as literals (see
    # docs/adr/0001-adaptive-sandwich-performance-plan.md's "Real-scale
    # (n=10000) findings" section, and _extract_diagnostic_jit_arrays'
    # docstring). base_action_prob_layer/base_update_layer/base_inference_layer
    # ARE still closed over, but only to supply the small, never-traced
    # metadata dataclasses.replace(...) needs to fill in every field this
    # closure's jit_arrays argument does not carry (time_to_col,
    # subject_id_to_pos, subject_ids_in_order, action_prob_times_by_subject,
    # etc.) -- see _rebuild_precomputes_from_jit_arrays.
    @jax.jit
    def _eval_avg_stack_jit(
        flattened_betas_and_theta: jnp.ndarray,
        jit_arrays: _DiagnosticJitArrays,
    ) -> jnp.ndarray:
        action_prob_layer, update_layer, inference_layer = (
            _rebuild_precomputes_from_jit_arrays(
                base_action_prob_layer,
                base_update_layer,
                base_inference_layer,
                jit_arrays,
                alg_update_func_args_beta_index,
                alg_update_func_args_previous_betas_index,
                alg_update_func_args_action_prob_index,
                inference_func_args_theta_index,
                inference_func_args_action_prob_index,
            )
        )
        return get_avg_weighted_estimating_function_stacks_and_aux_values(
            flattened_betas_and_theta,
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
            True,  # suppress_all_data_checks
            True,  # suppress_interactive_data_checks
            False,  # include_auxiliary_outputs
            precomputed_layers=(action_prob_layer, update_layer, inference_layer),
        )

    # Evaluate at the final estimate.
    eta_hat = flatten_params(all_post_update_betas, theta_est)

    # Draw perturbations delta_j on the O(1/sqrt(n)) scale, aligned with the empirical
    # joint estimating function stack covariance, without forming a d_joint x d_joint matrix
    # square-root. If G is the (n x d) matrix of per-subject stacks, then (1/n) G^T G is the
    # empirical covariance in joint estimating function stack space. Sampling u = (G^T w)/sqrt(n) with w~N(0, I_n) gives
    # u ~ N(0, empirical joint estimating function stack covariance G^T G/n ) in joint estimating function stack space.
    key = jax.random.PRNGKey(0)

    # The number of perturbations we will probe
    J = 15
    # Chunk size to reduce peak memory
    chunk_size = 1

    ratios_list = []
    chunk_wall_times = []
    num_chunks = (J + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, J)
        cur_size = end - start
        if cur_size <= 0:
            continue

        chunk_start = time.perf_counter()

        subkey = jax.random.fold_in(key, chunk_idx)
        W = jax.random.normal(subkey, shape=(cur_size, num_subjects))

        U = (W @ stacks) / jnp.sqrt(num_subjects)

        c = 1.0
        # TODO: Consider QR decomposition
        delta = (c / jnp.sqrt(num_subjects)) * jnp.linalg.solve(joint_bread, U.T).T

        B_delta = (joint_bread @ delta.T).T
        g_plus = jax.vmap(lambda d: _eval_avg_stack_jit(eta_hat + d, jit_arrays))(delta)
        remainder = g_plus - g_hat - B_delta

        denom = jnp.linalg.norm(B_delta, axis=1)
        numer = jnp.linalg.norm(remainder, axis=1)
        ratios = jnp.where(denom > 0, numer / denom, jnp.inf)

        # block_until_ready so each chunk's wall time reflects its own actual
        # device work (JAX dispatch is async otherwise) -- separates one-time
        # jit compile cost (chunk 0) from steady-state per-chunk cost.
        jax.block_until_ready(ratios)
        chunk_wall_times.append(time.perf_counter() - chunk_start)

        ratios_list.append(ratios)

    logger.info(
        "Local linearization diagnostic per-chunk wall times (seconds): %s",
        [round(t, 3) for t in chunk_wall_times],
    )

    ratios = jnp.concatenate(ratios_list, axis=0)

    local_error_ratio_median = float(jnp.median(ratios))
    local_error_ratio_p90 = float(jnp.quantile(ratios, 0.9))
    local_error_ratio_max = float(jnp.max(ratios))

    logger.info(
        "Local linearization error ratio (median over %d draws): %.6f",
        J,
        local_error_ratio_median,
    )
    logger.info(
        "Local linearization error ratio (90th pct over %d draws): %.6f",
        J,
        local_error_ratio_p90,
    )

    logger.info(
        "Local linearization error ratio (max over %d draws): %.6f",
        J,
        local_error_ratio_max,
    )

    return local_error_ratio_median, local_error_ratio_p90, local_error_ratio_max


if __name__ == "__main__":
    cli()
