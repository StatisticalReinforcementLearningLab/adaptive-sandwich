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
