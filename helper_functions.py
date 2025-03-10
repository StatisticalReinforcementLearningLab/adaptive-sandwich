import os
import warnings
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
    matrix, try_tikhonov_if_poorly_conditioned=False, condition_num_threshold=10**3
):
    condition_number = np.linalg.cond(matrix)
    inverse = np.linalg.inv(matrix)
    if condition_number > condition_num_threshold:
        warnings.warn(
            f"You are inverting a matrix with a large condition number: {condition_number}"
        )
        if try_tikhonov_if_poorly_conditioned:
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
    return inverse


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


def confirm_input_check_result(message, error=None):
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
