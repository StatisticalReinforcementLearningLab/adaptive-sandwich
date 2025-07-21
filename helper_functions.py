import os
import importlib.util
import importlib.machinery
import logging

import numpy as np
import jax.numpy as jnp

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
    matrix,
    trim_small_singular_values=False,
    try_tikhonov_if_poorly_conditioned=False,
    condition_num_threshold=10**4,
):
    inverse = None
    condition_number = np.linalg.cond(matrix)
    if condition_number > condition_num_threshold:
        logger.warning(
            "You are inverting a matrix with a large condition number: %s",
            condition_number,
        )
        # TODO: could try largest condition number to trim to such that original
        # times trimmed pseudoinverse is close to identity (or could minimize identity diff)
        if trim_small_singular_values:
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
            condition_number = np.linalg.cond(trimmed_pseudoinverse)
            logger.info(
                "Kept %s out of %s singular values. Condition number of resulting lower-rank-approximation: %s",
                sum(sing_values_above_threshold_cond),
                len(s),
                s[sing_values_above_threshold_cond].max()
                / s[sing_values_above_threshold_cond].min(),
            )
        if try_tikhonov_if_poorly_conditioned:
            inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
            supposed_identity = inverse * matrix
            min_distance_from_identity = np.sum(
                (supposed_identity - np.eye(matrix.shape[0])) ** 2
            )
            for exponent in range(1, 7):
                lambd = 10 ** (-exponent)
                new_matrix_to_invert = matrix.T @ matrix + lambd * np.eye(
                    matrix.shape[1]
                )
                inverse_candidate = np.linalg.inv(new_matrix_to_invert) @ matrix.T
                condition_number = np.linalg.cond(new_matrix_to_invert)
                logger.info(
                    "Trying Tikhonov regularization with lambda = %s to improve conditioning. New condition number of matrix to be inverted: %s",
                    lambd,
                    condition_number,
                )

                supposed_identity = inverse_candidate * matrix
                distance_from_identity = np.sum(
                    (supposed_identity - np.eye(matrix.shape[0])) ** 2
                )
                if distance_from_identity < min_distance_from_identity:
                    logger.info(
                        "Tikhonov regularization with lambda = %s got us an improved inverse.",
                        lambd,
                    )
                    min_distance_from_identity = distance_from_identity
                    inverse = inverse_candidate
                else:
                    logger.info(
                        "Tikhonov regularization with lambda = %s did not improve the inverse.",
                        lambd,
                    )
    if inverse is None:
        inverse = np.linalg.solve(matrix, np.eye(matrix.shape[0]))
    return inverse, condition_number


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
        ]
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
