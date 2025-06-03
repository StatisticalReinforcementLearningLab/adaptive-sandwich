import os
import pickle

import numpy as np
import pandas as pd


def get_abs_path(code_path, relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(code_path), relative_path))


def find_sim_root(starting_dir):
    """
    Walk upward from starting_dir until we see a “simulated_data” subfolder.
    Return the absolute path to that simulated_data folder (not just its parent).
    Raise if never found.
    """
    current = os.path.abspath(starting_dir)
    while True:
        candidate = os.path.join(current, "simulated_data")
        if os.path.isdir(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("Could not locate simulated_data directory")
        current = parent


def assert_real_run_output_as_expected(test_file_path, relative_path_to_output_dir):
    # 1) Locate the real “simulated_data” folder by climbing up from the test’s directory
    test_dir = os.path.dirname(test_file_path)
    sim_root = find_sim_root(test_dir)
    #    Now sim_root == "/…/adaptive-sandwich/simulated_data"
    #
    # 2) The caller passed something like
    #    "simulated_data/synthetic_mode=delayed_1_dosage…"
    #    so strip off the leading "simulated_data/" before joining:
    suffix = relative_path_to_output_dir.split("simulated_data/")[-1]
    observed_dir = os.path.join(sim_root, suffix)

    with open(os.path.join(observed_dir, "study_df.pkl"), "rb") as obs_study, open(
        os.path.join(observed_dir, "analysis.pkl"), "rb"
    ) as obs_analysis, open(
        os.path.join(observed_dir, "debug_pieces.pkl"), "rb"
    ) as obs_debug, open(
        get_abs_path(test_file_path, "expected_study_df.pkl"), "rb"
    ) as exp_study, open(
        get_abs_path(test_file_path, "expected_analysis.pkl"), "rb"
    ) as exp_analysis, open(
        get_abs_path(test_file_path, "expected_debug_pieces.pkl"), "rb"
    ) as exp_debug:

        observed_study_df = pickle.load(obs_study)
        observed_analysis_dict = pickle.load(obs_analysis)
        observed_debug_pieces_dict = pickle.load(obs_debug)

        expected_study_df = pickle.load(exp_study).astype(
            {"policy_num": "float64", "action": "float64"}
        )
        expected_analysis_dict = pickle.load(exp_analysis)
        expected_debug_pieces_dict = pickle.load(exp_debug)

        pd.testing.assert_frame_equal(observed_study_df, expected_study_df)

        np.testing.assert_allclose(
            observed_analysis_dict["theta_est"],
            expected_analysis_dict["theta_est"],
            rtol=1e-6,
        )

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

        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_meat_matrix"],
            expected_debug_pieces_dict["joint_meat_matrix"],
            rtol=6e-4,
        )
        np.testing.assert_allclose(
            observed_debug_pieces_dict["joint_bread_inverse_matrix"],
            expected_debug_pieces_dict["joint_bread_inverse_matrix"],
            atol=1e-5,
        )

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
