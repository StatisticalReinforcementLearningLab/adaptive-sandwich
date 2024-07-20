import warnings
import importlib.util
import importlib.machinery

import numpy as np


def conditional_x_or_one_minus_x(x, condition):
    return (1 - condition) + (2 * condition - 1) * x


def clip(lower_clip, upper_clip, vals):
    lower_clipped = np.maximum(vals, lower_clip)
    clipped = np.minimum(lower_clipped, upper_clip)
    return clipped


### study df manipulation functions


def get_user_column(study_df, user_id, column_name=None):
    """
    Extract just the supplied column for the given user in the given study_df as a
    numpy (column) vector.
    """
    if not column_name:
        raise ValueError("Please provide a column to extract")

    return (
        study_df.loc[study_df.user_id == user_id][column_name].to_numpy().reshape(-1, 1)
    )


def get_user_actions(study_df, user_id):
    return get_user_column(study_df, user_id, column_name="action")


def get_user_rewards(study_df, user_id):
    return get_user_column(study_df, user_id, column_name="reward")


def get_user_action1probs(study_df, user_id):
    return get_user_column(study_df, user_id, column_name="action1prob")


def invert_matrix_and_check_conditioning(matrix, condition_num_threshold=10**3):
    condition_number = np.linalg.cond(matrix)
    if condition_number > condition_num_threshold:
        warnings.warn(
            f"You are inverting a matrix with a large condition number: {condition_number}"
        )
    return np.linalg.inv(matrix)


def load_module_from_source_file(modname, filename):
    loader = importlib.machinery.SourceFileLoader(modname, filename)
    spec = importlib.util.spec_from_file_location(modname, filename, loader=loader)
    module = importlib.util.module_from_spec(spec)
    # The module is always executed and not cached in sys.modules.
    # Uncomment the following line to cache the module.
    # sys.modules[module.__name__] = module
    loader.exec_module(module)
    return module
