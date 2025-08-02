import os
import importlib.util
import importlib.machinery
import logging

import numpy as np
import jax.numpy as jnp

from constants import InverseStabilizationMethods

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def conditional_x_or_one_minus_x(x, condition):
    return (1 - condition) + (2 * condition - 1) * x


def clip(lower_clip, upper_clip, vals):
    lower_clipped = np.maximum(vals, lower_clip)
    clipped = np.minimum(lower_clipped, upper_clip)
    return clipped


def invert_matrix_and_check_conditioning(
    matrix: np.ndarray,
    inverse_stabilization_method: str,
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


def confirm_input_check_result(message, suppress_interactive_data_checks, error=None):

    if suppress_interactive_data_checks:
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


def replace_tuple_index(tupl, index, value):
    return tupl[:index] + (value,) + tupl[index + 1 :]


def get_action_1_fraction(study_df, in_study_col_name, action_col_name):
    """
    Get the fraction of action 1 in the study_df.
    """
    action_1_count = np.sum(
        (study_df[in_study_col_name] == 1) & (study_df[action_col_name] == 1)
    )
    total_count = len(study_df)
    if total_count == 0:
        return 0.0
    return action_1_count / total_count


def get_action_prob_variance(study_df, in_study_col_name, action_prob_col_name):
    """
    Get the variance of the action probabilities in the study_df.
    """
    action_probs = study_df.loc[study_df[in_study_col_name] == 1, action_prob_col_name]
    return np.var(action_probs)
