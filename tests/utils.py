import os
import pickle

import numpy as np
import pandas as pd

def get_abs_path(code_path, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(code_path), relative_path))


def assert_real_run_output_as_expected(test_file_path, relative_path_to_output_dir):
    # Load the observed and expected pickle files
    with open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/study_df.pkl",
        ),
        "rb",
    ) as observed_study_df_pickle, open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/analysis.pkl",
        ),
        "rb",
    ) as observed_analysis_pickle, open(
        get_abs_path(
            test_file_path,
            f"{relative_path_to_output_dir}/debug_pieces.pkl",
        ),
        "rb",
    ) as observed_debug_pieces_pickle, open(
        get_abs_path(test_file_path, "expected_study_df.pkl"),
        "rb",
    ) as expected_study_df_pickle, open(
        get_abs_path(test_file_path, "expected_analysis.pkl"),
        "rb",
    ) as expected_analysis_pickle, open(
        get_abs_path(test_file_path, "expected_debug_pieces.pkl"),
        "rb",
    ) as expected_debug_pieces_pickle:
        observed_study_df = pickle.load(observed_study_df_pickle)
        observed_analysis_dict = pickle.load(observed_analysis_pickle)
        observed_debug_pieces_dict = pickle.load(observed_debug_pieces_pickle)

        # The expected df is generated from a time when these were set as Int64.
        # I don't remember why we had to change to float64; I believe it was
        # necessary on the analysis side.
        expected_study_df = pickle.load(expected_study_df_pickle).astype(
            {"policy_num": "float64", "action": "float64"}
        )
        expected_analysis_dict = pickle.load(expected_analysis_pickle)
        expected_debug_pieces_dict = pickle.load(expected_debug_pieces_pickle)

        ### Check base study dataframes equal. This is important so that
        # we are even in the game, trying to produce the right inference results.
        pd.testing.assert_frame_equal(observed_study_df, expected_study_df)

        # Check that we have the same theta estimate in both cases.
        np.testing.assert_allclose(
            observed_analysis_dict["theta_est"],
            expected_analysis_dict["theta_est"],
            rtol=1e-6,
        )

        # Too hard to go back in time and add expected values for all the keys here,
        # but we can at least check they're present in the observed dict now,
        # and then still compare the most important ones to observed.
        expected_debug_keys = [
            "theta_est",
            "adaptive_sandwich_var_estimate",
            "classical_sandwich_var_estimate",
            "joint_bread_inverse_matrix",
            "joint_bread_matrix",
            "joint_meat_matrix",
            "classical_bread_inverse_matrix",
            "classical_bread_matrix",
            "classical_meat_matrix",
            "all_estimating_function_stacks",
        ]
        assert list(observed_debug_pieces_dict.keys()) == expected_debug_keys

        ### Check joint meat and bread inverse, uniting RL and inference
        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_meat_matrix"],
            expected_debug_pieces_dict["joint_meat_matrix"],
            atol=1e-5,
        )

        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_bread_inverse_matrix"],
            expected_debug_pieces_dict["joint_bread_inverse_matrix"],
            atol=1e-5,
        )

        ### Check final results
        np.testing.assert_allclose(
            observed_analysis_dict["adaptive_sandwich_var_estimate"],
            expected_analysis_dict["adaptive_sandwich_var_estimate"],
            atol=1e-8,
        )
        np.testing.assert_allclose(
            observed_analysis_dict["classical_sandwich_var_estimate"],
            expected_analysis_dict["classical_sandwich_var_estimate"],
            atol=1e-8,
        )

import numpy as np


def finite_difference_gradient(func, param, h=1e-5):
    """
    Compute the finite difference approximation of the gradient of a function.

    Parameters:
    func (callable): The function for which the gradient is to be approximated.
                     It should take a single argument which is a numpy array.
    param (numpy.ndarray): The point at which the gradient is to be approximated.
    h (float, optional): The step size for the finite difference approximation.
                         Default is 1e-5.

    Returns:
    numpy.ndarray: The approximated gradient of the function at the given point.
    """
    grad_approx = np.zeros_like(param)
    for i in range(len(param)):
        param_plus = param.copy()
        param_minus = param.copy()
        param_plus[i] += h
        param_minus[i] -= h
        grad_approx[i] = (func(param_plus) - func(param_minus)) / (2 * h)
    return grad_approx


def finite_difference_jacobian(f, x, h=None):
    """
    Computes the Jacobian matrix of a vector-valued function f at x using central differences.

    Parameters:
    f  : function R^n -> R^m
    x  : numpy array (point where Jacobian is computed)
    h  : finite difference step size

    Returns:
    J  : Jacobian matrix (m x n)
    """
    if h is None:
        h = np.linalg.norm(x) * 1.66e-2
    n = len(x)
    m = len(f(x))
    J = np.zeros((m, n))
    I = np.eye(n)  # Identity matrix to get unit vectors

    for i in range(n):
        # This handles the cases in which x is a row OR column vector
        perturbation = h * I[i].reshape(x.shape)
        J[:, i] = (f(x + perturbation) - f(x - perturbation)) / (2 * h)
    return J


def finite_difference_hessian(f, x, h=None):
    """
    Computes the Hessian matrix of f at x using central differences.

    Parameters:
    f  : function R^n -> R
    x  : numpy array (point where Hessian is computed)
    h  : finite difference step size

    Returns:
    H  : Hessian matrix (n x n)
    """

    if h is None:
        h = np.linalg.norm(x) * 1.66e-2
    n = len(x)
    H = np.zeros((n, n))
    I = np.eye(n)  # Identity matrix to get unit vectors

    # Handles the cases where x is a row OR column vector
    for i in range(n):
        perturbation_i = h * I[i].reshape(x.shape)
        for j in range(i, n):  # Compute only upper triangle, exploit symmetry
            perturbation_j = h * I[j].reshape(x.shape)
            if i == j:
                # Second derivative wrt x_i^2
                H[i, i] = (
                    f(x + perturbation_i) - 2 * f(x) + f(x - perturbation_i)
                ) / h**2
            else:
                # Mixed derivative wrt x_i, x_j
                H[i, j] = H[j, i] = (
                    f(x + perturbation_i + perturbation_j)
                    - f(x + perturbation_i - perturbation_j)
                    - f(x - perturbation_i + perturbation_j)
                    + f(x - perturbation_i - perturbation_j)
                ) / (4 * h**2)

    return H


def finite_difference_mixed_derivative(f, x, y, h=1e-5):
    """
    Computes the mixed derivative matrix of f at (x, y) using central differences.

    Parameters:
    f  : function R^n x R^m -> R
    x  : numpy array (first vector input)
    y  : numpy array (second vector input)
    h  : finite difference step size

    Returns:
    H  : Mixed derivative matrix (n x m)
    """
    n = len(x)
    m = len(y)
    D = np.zeros((n, m))
    I_x = np.eye(n)  # Identity matrix for x
    I_y = np.eye(m)  # Identity matrix for y

    # Handles the cases where x is a row OR column vector
    for i in range(n):
        perturbation_i = h * I_x[i].reshape(x.shape)
        for j in range(m):
            perturbation_j = h * I_y[j].reshape(y.shape)
            D[i, j] = (
                f(x + perturbation_i, y + perturbation_j)
                - f(x + perturbation_i, y - perturbation_j)
                - f(x - perturbation_i, y + perturbation_j)
                + f(x - perturbation_i, y - perturbation_j)
            ) / (4 * h**2)

    return D
