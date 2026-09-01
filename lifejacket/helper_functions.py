from __future__ import annotations

import collections
import contextlib
import importlib.machinery
import importlib.util
import logging
import math
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .constants import (
    JACOBIAN_AUTO_MAX_CHUNK,
    JACOBIAN_AUTO_ROW_BUDGET,
    JACOBIAN_AUTO_UNCHUNKED_MAX_OUT_DIM,
)
from .vmap_helpers import stack_batched_arg_lists_into_tensors

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def log_phase_duration(phase_name: str):
    """
    Context manager that logs the wall-clock duration of the wrapped block at
    INFO level, tagged with phase_name.

    This exists to give a coarse, always-on breakdown of where analyze_dataset
    (and similar entry points) spend their time, without requiring a separate
    profiler to be attached. See docs/adr/0001-adaptive-sandwich-performance-plan.md
    for how this is used as the first step of the performance work.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("Phase '%s' took %.3f seconds.", phase_name, elapsed)


def conditional_x_or_one_minus_x(x, condition):
    return (1 - condition) + (2 * condition - 1) * x


def invert_matrix_and_check_conditioning(
    matrix: np.ndarray,
    condition_num_threshold: float = 10**4,
):
    """
    Check a matrix's condition number and invert it. If the condition number is
    above a threshold, apply stabilization methods to improve conditioning.
    Parameters
    """
    inverse = None
    condition_number = np.linalg.cond(matrix)
    if condition_number > condition_num_threshold:
        logger.warning(
            "You are inverting a matrix with a potentially large condition number: %s",
            condition_number,
        )
    if inverse is None:
        inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    return inverse, condition_number


def zero_small_off_diagonal_blocks(
    matrix: jnp.ndarray,
    block_sizes: list[int],
    frobenius_norm_threshold_fraction: float = 1e-3,
):
    """
    Zero off-diagonal blocks whose Frobenius norm is < frobenius_norm_threshold_fraction x
    Frobenius norm of the diagonal block in the same ROW. One could compare to
    the same column or both the row and column, but we choose row here since
    rows correspond to a single RL update or inference step in the bread
    inverse matrices this method is designed for.

    Args:
        matrix (jnp.ndarray):
            2-D ndarray, square (q_total x q_total)
        block_sizes (list[int]):
            list like [p1, p2, ..., pT]
        frobenius_norm_threshold_fraction (float):
            frobenius norm fraction relative to same-row diagonal block under which we zero a block

    Returns
        ndarray with selected off-blocks zeroed
    """

    bounds = np.cumsum([0] + list(block_sizes))
    num_block_rows_cols = len(block_sizes)
    J_trim = matrix.copy()

    # 1. collect Frobenius norms of every diagonal block in one pass
    diag_norm = np.empty(num_block_rows_cols)
    for t in range(num_block_rows_cols):
        sl = slice(bounds[t], bounds[t + 1])
        diag_norm[t] = np.linalg.norm(matrix[sl, sl], ord="fro")

    # 2. Zero all sufficiently small off-diagonal blocks
    for t in range(num_block_rows_cols):
        source_norm = diag_norm[t]
        r0, r1 = bounds[t], bounds[t + 1]  # rows belonging to block t

        # rows BELOW the diagonal (lower-triangular part)
        for tau in range(t + 1, num_block_rows_cols):
            c0, c1 = bounds[tau], bounds[tau + 1]
            block = J_trim[r0:r1, c0:c1]
            block_norm = np.linalg.norm(block, ord="fro")
            if (
                block_norm
                and block_norm < frobenius_norm_threshold_fraction * source_norm
            ):
                logger.info(
                    "Zeroing out block [%s:%s, %s:%s] with Frobenius norm %s < %s * %s",
                    r0,
                    r1,
                    c0,
                    c1,
                    block_norm,
                    frobenius_norm_threshold_fraction,
                    source_norm,
                )
                J_trim = J_trim.at[r0:r1, c0:c1].set(0.0)

    return J_trim


def append_new_block_row_to_block_lower_triangular_matrix(
    cached_block: jnp.ndarray, new_block_row: jnp.ndarray
) -> jnp.ndarray:
    """
    Augments a square, block lower-triangular matrix with a new block row,
    exploiting the fact that a block lower-triangular structure means the new
    row's only nonzero entries are in its own columns and the columns already
    present in cached_block -- there are no new rows required above the
    diagonal, since earlier outputs cannot depend on parameters introduced
    after them.

    Args:
        cached_block (jnp.ndarray): The existing square block matrix, shape
            (previous_dim, previous_dim).
        new_block_row (jnp.ndarray): The new bottom block row, shape
            (new_dim, previous_dim + new_dim) -- typically a Jacobian of
            new_dim new outputs with respect to all previous_dim + new_dim
            parameters (previous ones plus the new_dim newly introduced
            ones).

    Returns:
        jnp.ndarray: The augmented (previous_dim + new_dim, previous_dim +
            new_dim) square matrix, with cached_block and a new all-zero
            block in the top row, and new_block_row spanning the full width
            of the bottom row.

    IMPORTANT: jnp.block requires a NESTED list to build a 2D block matrix --
    passing a flat list instead concatenates every element along the last
    axis only (equivalent to hstack), which silently produces a non-square,
    wrong-shaped result as soon as cached_block has more than new_dim rows.
    This function exists specifically so that mistake can't be made twice;
    see docs/adr/0001-adaptive-sandwich-performance-plan.md for the bug this
    was extracted from.
    """
    previous_dim = cached_block.shape[0]
    new_dim = new_block_row.shape[0]
    return jnp.block(
        [
            [cached_block, jnp.zeros((previous_dim, new_dim))],
            [new_block_row],
        ]
    )


def invert_bread_matrix(
    bread,
    beta_dim,
    theta_dim,
):
    """
    Invert the bread matrix to get the inverse bread matrix.  This is a special
    function in order to take advantage of the block lower triangular structure.

    The procedure is as follows:
    1. Initialize the matrix B = A^{-1} as a block lower triangular matrix
       with the same block structure as A.

    2. Compute the diagonal blocks B_{ii}:
       For each diagonal block A_{ii}, calculate:
           B_{ii} = A_{ii}^{-1}

    3. Compute the off-diagonal blocks B_{ij} for i > j:
       For each off-diagonal block B_{ij} (where i > j), compute:
           B_{ij} = -A_{ii}^{-1} * sum(A_{ik} * B_{kj} for k in range(j, i))
    """
    blocks = []
    num_beta_block_rows = (bread.shape[0] - theta_dim) // beta_dim

    # Create upper rows of block of bread (just the beta portion)
    for i in range(0, num_beta_block_rows):
        beta_block_row = []
        beta_diag_inverse = invert_matrix_and_check_conditioning(
            bread[
                beta_dim * i : beta_dim * (i + 1),
                beta_dim * i : beta_dim * (i + 1),
            ],
        )[0]
        for j in range(0, num_beta_block_rows):
            if i > j:
                beta_block_row.append(
                    -beta_diag_inverse
                    @ sum(
                        bread[
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
        bread[
            -theta_dim:,
            -theta_dim:,
        ],
    )[0]
    for k in range(0, num_beta_block_rows):
        theta_block_row.append(
            -theta_diag_inverse
            @ sum(
                bread[
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


def matrix_inv_sqrt(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return (mat)^{-1/2} with eigenvalues clipped at `eps`."""
    eigval, eigvec = np.linalg.eigh(mat)
    eigval = np.clip(eigval, eps, None)  # ensure strictly positive
    return eigvec @ np.diag(eigval**-0.5) @ eigvec.T


def resolve_jacobian_row_chunk_size(
    requested: int | None,
    out_dim: int,
) -> int | None:
    """
    Resolves the jacobian_row_chunk_size request into a concrete decision for
    construct_classical_and_adjusted_sandwiches's backward pass: returns None
    for a single unchunked eager jax.vmap(pullback) over the whole cotangent
    basis, or a positive chunk size for the chunked, jitted backward path.

    This lives here, rather than next to that backward pass, because it is the
    ONE memory-bounding policy shared by every reverse-mode Jacobian in the
    package -- diagnostics.py cannot import post_deployment_analysis (that
    module imports diagnostics, so the reverse direction is circular), and the
    policy must not fork per call site. Call sites that do not implement their
    own chunked backward pass pass this function's result straight to
    compute_row_chunked_jacobian below, whose None/0 case is the plain
    unchunked jax.jacrev rather than the eager vmap(pullback) described here
    (same numbers, same memory profile, different plumbing).

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


def compute_row_chunked_jacobian(
    fn: collections.abc.Callable[[jnp.ndarray], jnp.ndarray],
    x: jnp.ndarray,
    chunk_size: int | None = None,
) -> jnp.ndarray:
    """
    Memory-bounded reverse-mode Jacobian: the same quantity as
    jax.jacrev(fn)(x) (same shape, same forward pass, the same independent
    per-row vjp), but with the output cotangent basis walked in chunks of at
    most chunk_size rows so that peak memory scales with chunk_size rather
    than with the full output dimension. Chunking is exact -- no row is
    accumulated across chunks, so no chunk size is "more approximate" than
    another; measured agreement with jax.jacrev is ~1 ulp (float64) and
    ~5e-7 absolute (float32), identical at every chunk size, which is XLA
    fusing the compiled scan body differently from eager op-by-op dispatch
    rather than anything the chunking does. This is the general-purpose
    counterpart to
    construct_classical_and_adjusted_sandwiches's own hand-rolled chunked
    backward pass; use this one anywhere that does not need that function's
    explicit-residual jit machinery.

    Args:
        fn: a function from one array to one array (differentiated w.r.t. its
            single argument; close over anything else).
        x: the point to differentiate at.
        chunk_size: None or 0 = no chunking, i.e. a plain jax.jacrev(fn)(x)
            (jacrev's own single vmap over the whole basis -- fastest when it
            fits in memory). A positive int caps how many output-basis rows
            are pulled back at once. Negative values raise ValueError. Pass
            resolve_jacobian_row_chunk_size(requested, out_dim) to get the
            package-wide auto policy.

    Returns:
        The Jacobian, of shape fn(x).shape + x.shape -- exactly jax.jacrev's.
        fn's argument and result must each be a single array (not a pytree),
        which is what makes that shape, and the flat output basis below,
        well-defined.

    The chunk loop is jax.lax.map (a scan under the hood), NOT a Python for
    loop, because this is called from inside jax.jit traces: a Python loop
    unrolls at trace time into num_chunks copies of the backward graph, whose
    intermediates XLA is then free to keep live simultaneously -- which
    reinstates exactly the peak-memory blowup the chunking exists to prevent
    (real 24GB OOM crashes at oralytics scale; see
    docs/adr/0001-adaptive-sandwich-performance-plan.md). lax.map keeps one
    chunk's backward pass live at a time under jit and eagerly alike.

    A final short chunk is handled by PADDING the basis up to a whole number
    of chunks with all-zero cotangent rows (lax.map needs one uniform chunk
    shape) and dropping the corresponding all-zero Jacobian rows from the
    result. The padding costs at most chunk_size - 1 extra pullbacks of the
    zero cotangent; it never changes the returned rows.
    """
    if chunk_size is not None and chunk_size < 0:
        raise ValueError(
            "chunk_size must be a non-negative int or None (None or 0 = no "
            f"chunking, positive = max rows per chunk), got {chunk_size!r}."
        )
    if chunk_size is None or chunk_size == 0:
        return jax.jacrev(fn)(x)
    if not hasattr(x, "shape"):
        raise TypeError(
            "compute_row_chunked_jacobian differentiates w.r.t. a single "
            f"array, not a pytree; got {type(x).__name__}."
        )

    outputs, pullback = jax.vjp(fn, x)
    if not hasattr(outputs, "shape"):
        raise TypeError(
            "compute_row_chunked_jacobian needs fn to return a single array "
            "(the output basis it chunks is that array's flattened identity); "
            f"got {type(outputs).__name__}."
        )
    out_dim = math.prod(outputs.shape)
    effective_chunk_size = min(chunk_size, out_dim)
    if effective_chunk_size < 1:
        # Empty output: no basis to chunk, and the arithmetic below would
        # divide by zero.
        return jax.jacrev(fn)(x)

    num_chunks = math.ceil(out_dim / effective_chunk_size)
    padded_out_dim = num_chunks * effective_chunk_size
    # Rows out_dim..padded_out_dim-1 of a non-square jnp.eye are all zero --
    # that is the padding described above, at no extra construction cost.
    cotangent_chunks = jnp.eye(padded_out_dim, out_dim, dtype=outputs.dtype).reshape(
        num_chunks, effective_chunk_size, *outputs.shape
    )

    def pull_back_chunk(cotangent_chunk):
        return jax.vmap(lambda cotangent: pullback(cotangent)[0])(cotangent_chunk)

    jacobian_rows = jax.lax.map(pull_back_chunk, cotangent_chunks)
    return jacobian_rows.reshape(padded_out_dim, *x.shape)[:out_dim].reshape(
        *outputs.shape, *x.shape
    )


def load_module_from_source_file(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module


def load_function_from_same_named_file(filename):
    module = load_module_from_source_file(filename, filename)
    try:
        return module.__dict__[os.path.basename(filename).split(".")[0]]
    except AttributeError as e:
        raise ValueError(
            f"Unable to import function from {filename}.  Please verify the file has the same name as the function of interest (ignoring the extension)."
        ) from e
    except KeyError as e:
        raise ValueError(
            f"Unable to import function from {filename}.  Please verify the file has the same name as the function of interest (ignoring the extension)."
        ) from e


def confirm_input_check_result(message, suppress_interaction, error=None):

    if suppress_interaction:
        logger.info(
            "Skipping the following interactive data check, as requested:\n%s", message
        )
        return
    answer = None
    while answer != "y":
        answer = input(message).lower()
        if answer == "y":
            print("\nOk, proceeding.\n")
        elif answer == "n":
            if error:
                raise SystemExit from error
            raise SystemExit
        else:
            print("\nPlease enter 'y' or 'n'.\n")


def get_active_df_column(analysis_df, col_name, active_col_name):
    return jnp.array(
        analysis_df.loc[analysis_df[active_col_name] == 1, col_name]
        .to_numpy()
        .reshape(-1, 1)
    )


def flatten_params(betas: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate(list(betas) + [theta])


def unflatten_params(
    flat: jnp.ndarray, beta_dim: int, theta_dim: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    theta = flat[-theta_dim:]
    betas = jnp.array(
        [
            flat[i * beta_dim : (i + 1) * beta_dim]
            for i in range((len(flat) - theta_dim) // beta_dim)
        ]
    )
    return betas, theta


def get_radon_nikodym_weight(
    beta_target: jnp.ndarray[jnp.float32],
    action_prob_func: callable,
    action_prob_func_args_beta_index: int,
    action: int,
    *action_prob_func_args_single_subject: tuple[Any, ...],
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
            a particular subject given their state and the algorithm parameters.
        action_prob_func_args_beta_index (int):
            The index of the beta argument in the action probability function's arguments.
        action (int):
            The actual taken action at the relevant decision time.
        *action_prob_func_args_single_subject (tuple[Any, ...]):
            The arguments to the action probability function for the relevant subject at this time.

    Returns:
        jnp.float32: The Radon-Nikodym weight.

    """

    # numerator
    pi_beta = action_prob_func(*action_prob_func_args_single_subject)

    # denominator, where we thread in beta_target so that differentiation with respect to the
    # original beta in the arguments leaves this alone.
    beta_target_action_prob_func_args_single_subject = [
        *action_prob_func_args_single_subject
    ]
    beta_target_action_prob_func_args_single_subject[
        action_prob_func_args_beta_index
    ] = beta_target
    pi_beta_target = action_prob_func(*beta_target_action_prob_func_args_single_subject)

    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


def get_min_time_by_policy_num(
    single_subject_policy_num_by_decision_time, beta_index_by_policy_num
):
    """
    Returns a dictionary mapping each policy number to the first time it was applicable,
    and the first time after the first update.
    """
    min_time_by_policy_num = {}
    first_time_after_first_update = None
    for decision_time, policy_num in single_subject_policy_num_by_decision_time.items():
        if policy_num not in min_time_by_policy_num:
            min_time_by_policy_num[policy_num] = decision_time

        # Grab the first time where a non-initial, non-fallback policy is used.
        # Assumes single_subject_policy_num_by_decision_time is sorted.
        if (
            policy_num in beta_index_by_policy_num
            and first_time_after_first_update is None
        ):
            first_time_after_first_update = decision_time

    return min_time_by_policy_num, first_time_after_first_update


def compute_subject_radon_nikodym_weights(
    action_prob_func,
    action_prob_func_args_beta_index,
    action_prob_func_args_by_decision_time,
    threaded_action_prob_func_args_by_decision_time,
    policy_num_by_decision_time,
    action_by_decision_time,
    beta_index_by_policy_num,
):
    """
    Computes, for a single subject, the full-window vector of Radon-Nikodym
    weights (get_radon_nikodym_weight, jax.vmap'd over this subject's active
    decision times, padded with the product-identity 1.0 at any decision time
    with no real action-probability-function args) plus the auxiliary values
    needed to select the correct sub-window of that vector for a given
    algorithm update or for inference.

    This was extracted from what used to be three near-identical, hand-copied
    implementations (one each in post_deployment_analysis.py,
    deployment_conditioning_monitor.py, and
    get_datum_for_blowup_supervised_learning.py) differing only in local
    variable-name prefixes -- see
    docs/adr/0001-adaptive-sandwich-performance-plan.md.

    Args:
        action_prob_func (callable): The action probability function.
        action_prob_func_args_beta_index (int): The index of beta in the
            action probability function's arguments.
        action_prob_func_args_by_decision_time (dict[int, tuple[Any, ...]]):
            This subject's action probability function arguments by decision
            time (all decision times; empty tuple if not active then). NOTE:
            these do NOT contain the shared betas, so they're impervious to
            differentiation.
        threaded_action_prob_func_args_by_decision_time (dict[int, tuple[Any, ...]]):
            Same, but with the shared betas threaded in for differentiation.
        policy_num_by_decision_time (dict[int, int | float]): The policy
            number in use at each of this subject's active decision times.
        action_by_decision_time (dict[int, int]): The action taken at each of
            this subject's active decision times.
        beta_index_by_policy_num (dict[int | float, int]): Maps non-initial,
            non-fallback policy numbers to their index in all_post_update_betas.

    Returns:
        all_weights (jnp.ndarray): 1-D array of weights, one per decision
            time in [subject_start_time, subject_end_time], in order.
        decision_time_to_all_weights_index_offset (int): subtract this from a
            decision time to get its index into all_weights (== subject_start_time).
        first_time_after_first_update (int | None): passed through from
            get_min_time_by_policy_num.
        min_time_by_policy_num (dict[int | float, int]): passed through from
            get_min_time_by_policy_num.
        subject_start_time (int): this subject's earliest active decision time.
        subject_end_time (int): this subject's latest active decision time.
    """
    min_time_by_policy_num, first_time_after_first_update = get_min_time_by_policy_num(
        policy_num_by_decision_time,
        beta_index_by_policy_num,
    )

    subject_start_time = math.inf
    subject_end_time = -math.inf
    for decision_time in action_by_decision_time:
        subject_start_time = min(subject_start_time, decision_time)
        subject_end_time = max(subject_end_time, decision_time)

    # Fail loudly here rather than letting active_index (below) silently
    # misalign or overrun active_weights on a genuine intra-window gap (this
    # subject active, then inactive, then active again, within their own
    # [subject_start_time, subject_end_time]) -- see
    # docs/adr/0001-adaptive-sandwich-performance-plan.md, Step 4. This
    # protects every caller of this shared function identically.
    gap_times = [
        decision_time
        for decision_time in range(int(subject_start_time), int(subject_end_time) + 1)
        if not action_prob_func_args_by_decision_time.get(decision_time)
    ]
    if gap_times:
        raise ValueError(
            f"Subject has an intra-window gap (inactive at decision time(s) "
            f"{gap_times}, strictly between their own first active time "
            f"{subject_start_time} and last active time {subject_end_time}), "
            "which violates the 'once active, stays active with no re-entry' "
            "invariant this Radon-Nikodym weight-window logic assumes."
        )

    # Sort the threaded args by decision time to be cautious. We check if the
    # subject id is present in the subject args dict because we may call this
    # on a subset of the subject arg dict when we are batching arguments by
    # shape.
    sorted_threaded_action_prob_args_by_decision_time = {
        decision_time: threaded_action_prob_func_args_by_decision_time[decision_time]
        for decision_time in range(subject_start_time, subject_end_time + 1)
        if decision_time in threaded_action_prob_func_args_by_decision_time
    }

    # Build the beta_target/action vectors in the SAME order as the threaded
    # args we stack below, so the vmap arguments stay aligned even if
    # action_prob_func_args_by_decision_time/action_by_decision_time were not
    # insertion-sorted by decision_time (their order is whatever row order
    # analysis_df had, which is not guaranteed to match).
    active_decision_times = [
        decision_time
        for decision_time, args in sorted_threaded_action_prob_args_by_decision_time.items()
        if args
    ]
    active_betas_list_by_decision_time_index = jnp.array(
        [
            action_prob_func_args_by_decision_time[decision_time][
                action_prob_func_args_beta_index
            ]
            for decision_time in active_decision_times
        ]
    )
    active_actions_list_by_decision_time_index = jnp.array(
        [
            action_by_decision_time[decision_time]
            for decision_time in active_decision_times
        ]
    )

    num_args = None
    for args in sorted_threaded_action_prob_args_by_decision_time.values():
        if args:
            num_args = len(args)
            break

    # NOTE: Cannot do [[]] * num_args here! Then all lists point same object...
    batched_threaded_arg_lists = [[] for _ in range(num_args)]
    for (
        _decision_time,
        args,
    ) in sorted_threaded_action_prob_args_by_decision_time.items():
        if not args:
            continue
        for idx, arg in enumerate(args):
            batched_threaded_arg_lists[idx].append(arg)

    batched_threaded_arg_tensors, batch_axes = stack_batched_arg_lists_into_tensors(
        batched_threaded_arg_lists
    )

    # Note that we do NOT use the shared betas in the first arg to the weight
    # function, since we don't want differentiation to happen with respect to
    # them. Just grab the original beta from the update function arguments.
    # This is the same value, but impervious to differentiation with respect
    # to all_post_update_betas. The args, on the other hand, are a function
    # of all_post_update_betas.
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
        _decision_time,
        args,
    ) in sorted_threaded_action_prob_args_by_decision_time.items():
        all_weights_raw.append(active_weights[active_index] if args else 1.0)
        active_index += 1
    all_weights = jnp.array(all_weights_raw)

    return (
        all_weights,
        decision_time_to_all_weights_index_offset,
        first_time_after_first_update,
        min_time_by_policy_num,
        subject_start_time,
        subject_end_time,
    )


def calculate_beta_dim(
    action_prob_func_args: dict[int, dict[collections.abc.Hashable, tuple[Any, ...]]],
    action_prob_func_args_beta_index: int,
) -> int:
    """
    Calculates the dimension of the beta vector based on the action probability function arguments.

    Args:
        action_prob_func_args (dict): Dictionary containing the action probability function arguments.
        action_prob_func_args_beta_index (int): Index of the beta parameter in the action probability function arguments.

    Returns:
        int: The dimension of the beta vector.
    """
    for decision_time in action_prob_func_args:
        for subject_id in action_prob_func_args[decision_time]:
            if action_prob_func_args[decision_time][subject_id]:
                return len(
                    action_prob_func_args[decision_time][subject_id][
                        action_prob_func_args_beta_index
                    ]
                )
    raise ValueError(
        "No valid beta vector found in action probability function arguments. Please check the input data."
    )


def construct_beta_index_by_policy_num_map(
    analysis_df: pd.DataFrame, policy_num_col_name: str, active_col_name: str
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
        analysis_df[
            (analysis_df[policy_num_col_name] >= 0)
            & (analysis_df[active_col_name] == 1)
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
        for subject_id in alg_update_func_args[policy_num]:
            if alg_update_func_args[policy_num][subject_id]:
                all_post_update_betas.append(
                    alg_update_func_args[policy_num][subject_id][
                        alg_update_func_args_beta_index
                    ]
                )
                break
    return jnp.array(all_post_update_betas)


def extract_action_and_policy_by_decision_time_by_subject_id(
    analysis_df,
    subject_id_col_name,
    active_col_name,
    calendar_t_col_name,
    action_col_name,
    policy_num_col_name,
):
    action_by_decision_time_by_subject_id = {}
    policy_num_by_decision_time_by_subject_id = {}
    for subject_id, subject_df in analysis_df.groupby(subject_id_col_name):
        active_subject_df = subject_df[subject_df[active_col_name] == 1]
        action_by_decision_time_by_subject_id[subject_id] = dict(
            zip(
                active_subject_df[calendar_t_col_name],
                active_subject_df[action_col_name],
                strict=False,
            )
        )
        policy_num_by_decision_time_by_subject_id[subject_id] = dict(
            zip(
                active_subject_df[calendar_t_col_name],
                active_subject_df[policy_num_col_name],
                strict=False,
            )
        )
    return (
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
    )
