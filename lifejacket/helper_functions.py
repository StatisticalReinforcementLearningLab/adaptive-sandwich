from __future__ import annotations

import collections
import os
import importlib.util
import importlib.machinery
import logging
from typing import Any

import numpy as np
import jax.numpy as jnp
import pandas as pd

from .constants import InverseStabilizationMethods

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def conditional_x_or_one_minus_x(x, condition):
    return (1 - condition) + (2 * condition - 1) * x


def invert_matrix_and_check_conditioning(
    matrix: np.ndarray,
    inverse_stabilization_method: str = InverseStabilizationMethods.NONE,
    condition_num_threshold: float = 10**4,
    ridge_median_singular_value_fraction: str = 0.01,
    beta_dim: int = None,
    theta_dim: int = None,
):
    """
    Check a matrix's condition number and invert it. If the condition number is
    above a threshold, apply stabilization methods to improve conditioning.
    Parameters
    """
    inverse = None
    pre_inversion_condition_number = np.linalg.cond(matrix)
    if pre_inversion_condition_number > condition_num_threshold:
        logger.warning(
            "You are inverting a matrix with a large condition number: %s",
            pre_inversion_condition_number,
        )
        if (
            inverse_stabilization_method
            == InverseStabilizationMethods.TRIM_SMALL_SINGULAR_VALUES
        ):
            logger.info("Trimming small singular values to improve conditioning.")
            u, s, vT = np.linalg.svd(matrix, full_matrices=False)
            logger.info(
                " Sorted singular values: %s",
                s,
            )
            sing_values_above_threshold_cond = s > s.max() / condition_num_threshold
            if not np.any(sing_values_above_threshold_cond):
                raise RuntimeError(
                    f"All singular values are below the threshold of {s.max() / condition_num_threshold}. Singular value trimming will not work.",
                )
            trimmed_pseudoinverse = (
                vT.T[:, sing_values_above_threshold_cond]
                / s[sing_values_above_threshold_cond]
            ) @ u[:, sing_values_above_threshold_cond].T
            inverse = trimmed_pseudoinverse
            pre_inversion_condition_number = (
                s[sing_values_above_threshold_cond].max()
                / s[sing_values_above_threshold_cond].min()
            )

            logger.info(
                "Kept %s out of %s singular values. Condition number of resulting lower-rank-approximation before inversion: %s",
                sum(sing_values_above_threshold_cond),
                len(s),
                pre_inversion_condition_number,
            )
        elif (
            inverse_stabilization_method
            == InverseStabilizationMethods.ADD_RIDGE_FIXED_CONDITION_NUMBER
        ):
            logger.info("Adding ridge/Tikhonov regularization to improve conditioning.")
            _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
            logger.info(
                "Using fixed condition number threshold of %s to determine lambda.",
                condition_num_threshold,
            )
            lambda_ = (
                singular_values.max() / condition_num_threshold - singular_values.min()
            )
            logger.info("Lambda for ridge regularization: %s", lambda_)
            new_matrix = matrix + lambda_ * np.eye(matrix.shape[0])
            pre_inversion_condition_number = np.linalg.cond(new_matrix)
            logger.info(
                "Condition number of matrix after ridge regularization: %s",
                pre_inversion_condition_number,
            )
            inverse = np.linalg.solve(new_matrix, np.eye(matrix.shape[0]))
        elif (
            inverse_stabilization_method
            == InverseStabilizationMethods.ADD_RIDGE_MEDIAN_SINGULAR_VALUE_FRACTION
        ):
            logger.info("Adding ridge/Tikhonov regularization to improve conditioning.")
            _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
            logger.info(
                "Using median singular value times %s as lambda.",
                ridge_median_singular_value_fraction,
            )
            lambda_ = ridge_median_singular_value_fraction * np.median(singular_values)
            logger.info("Lambda for ridge regularization: %s", lambda_)
            new_matrix = matrix + lambda_ * np.eye(matrix.shape[0])
            pre_inversion_condition_number = np.linalg.cond(new_matrix)
            logger.info(
                "Condition number of matrix after ridge regularization: %s",
                pre_inversion_condition_number,
            )
            inverse = np.linalg.solve(new_matrix, np.eye(matrix.shape[0]))
        elif (
            inverse_stabilization_method
            == InverseStabilizationMethods.INVERSE_BREAD_STRUCTURE_AWARE_INVERSION
        ):
            if not beta_dim or not theta_dim:
                raise ValueError(
                    "When using structure-aware inversion, beta_dim and theta_dim must be provided."
                )
            logger.info(
                "Using inverse bread's block lower triangular structure to invert only diagonal blocks."
            )
            pre_inversion_condition_number = np.linalg.cond(matrix)
            inverse = invert_inverse_bread_matrix(
                matrix,
                beta_dim,
                theta_dim,
                InverseStabilizationMethods.ADD_RIDGE_FIXED_CONDITION_NUMBER,
            )
        elif (
            inverse_stabilization_method
            == InverseStabilizationMethods.ZERO_OUT_SMALL_OFF_DIAGONALS
        ):
            if not beta_dim or not theta_dim:
                raise ValueError(
                    "When zeroing out small off diagonals, beta_dim and theta_dim must be provided."
                )
            logger.info(
                "Zeroing out small off-diagonal blocks to improve conditioning."
            )
            zeroed_matrix = zero_small_off_diagonal_blocks(
                matrix,
                ([beta_dim] * (matrix.shape[0] // beta_dim)) + [theta_dim],
            )
            pre_inversion_condition_number = np.linalg.cond(zeroed_matrix)
            logger.info(
                "Condition number of matrix after zeroing out small off-diagonal blocks: %s",
                pre_inversion_condition_number,
            )
            inverse = np.linalg.solve(zeroed_matrix, np.eye(zeroed_matrix.shape[0]))
        elif (
            inverse_stabilization_method
            == InverseStabilizationMethods.ALL_METHODS_COMPETITION
        ):
            # TODO: Choose right metric for competition... identity diff might not be it.
            raise NotImplementedError(
                "All methods competition is not implemented yet. Please choose a specific method."
            )
        elif inverse_stabilization_method == InverseStabilizationMethods.NONE:
            logger.info("No inverse stabilization method applied. Inverting directly.")
        else:
            raise ValueError(
                f"Unknown inverse stabilization method: {inverse_stabilization_method}"
            )
    if inverse is None:
        inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    return inverse, pre_inversion_condition_number


def zero_small_off_diagonal_blocks(
    matrix: jnp.ndarray,
    block_sizes: list[int],
    frobenius_norm_threshold_fraction: float = 1e-3,
):
    """
    Zero off-diagonal blocks whose Frobenius norm is < frobenius_norm_threshold_fraction x
    Frobenius norm of the diagonal block in the same ROW. One could compare to
    the same column or both the row and column, but we choose row here since
    rows correspond to a single RL update or inference step in the adaptive bread
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


def invert_inverse_bread_matrix(
    inverse_bread,
    beta_dim,
    theta_dim,
    diag_inverse_stabilization_method=InverseStabilizationMethods.TRIM_SMALL_SINGULAR_VALUES,
):
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
            diag_inverse_stabilization_method,
        )[0]
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
        ],
        diag_inverse_stabilization_method,
    )[0]
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


def matrix_inv_sqrt(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return (mat)^{-1/2} with eigenvalues clipped at `eps`."""
    eigval, eigvec = np.linalg.eigh(mat)
    eigval = np.clip(eigval, eps, None)  # ensure strictly positive
    return eigvec @ np.diag(eigval**-0.5) @ eigvec.T


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
        # pylint: disable=bad-builtin
        answer = input(message).lower()
        # pylint: enable=bad-builtin
        if answer == "y":
            print("\nOk, proceeding.\n")
        elif answer == "n":
            if error:
                raise SystemExit from error
            raise SystemExit
        else:
            print("\nPlease enter 'y' or 'n'.\n")


def get_in_study_df_column(study_df, col_name, in_study_col_name):
    return jnp.array(
        study_df.loc[study_df[in_study_col_name] == 1, col_name]
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
        for user_id in action_prob_func_args[decision_time]:
            if action_prob_func_args[decision_time][user_id]:
                return len(
                    action_prob_func_args[decision_time][user_id][
                        action_prob_func_args_beta_index
                    ]
                )
    raise ValueError(
        "No valid beta vector found in action probability function arguments. Please check the input data."
    )


def construct_beta_index_by_policy_num_map(
    study_df: pd.DataFrame, policy_num_col_name: str, in_study_col_name: str
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
