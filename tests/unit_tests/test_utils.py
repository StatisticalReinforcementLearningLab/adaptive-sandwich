import numpy as np
import pytest

from tests.unit_tests.utils import finite_difference_hessian


def test_finite_difference_hessian_quadratic():
    # Define a quadratic function f(x) = x^T A x + b^T x + c
    A = np.array([[3, 2], [2, 6]])
    b = np.array([1, 1])
    c = 0

    def f(x):
        return x.T @ A @ x + b.T @ x + c

    x = np.array([1.0, 2.0])
    expected_hessian = 2 * A
    computed_hessian = finite_difference_hessian(f, x)

    np.testing.assert_allclose(computed_hessian, expected_hessian, atol=1e-4)


def test_finite_difference_hessian_linear():
    # Define a linear function f(x) = b^T x + c
    b = np.array([1, 1])
    c = 0

    def f(x):
        return b.T @ x + c

    x = np.array([1.0, 2.0])
    expected_hessian = np.zeros((2, 2))
    computed_hessian = finite_difference_hessian(f, x)

    np.testing.assert_allclose(computed_hessian, expected_hessian, atol=1e-5)


def test_finite_difference_hessian_constant():
    # Define a constant function f(x) = c
    c = 5

    def f(x):
        return c

    x = np.array([1.0, 2.0])
    expected_hessian = np.zeros((2, 2))
    computed_hessian = finite_difference_hessian(f, x)

    np.testing.assert_allclose(computed_hessian, expected_hessian, atol=1e-5)


def test_finite_difference_hessian_non_square():
    # Define a non-square quadratic function f(x) = x1^2 + 2*x1*x2 + 3*x2^2
    def f(x):
        return x[0] ** 2 + 2 * x[0] * x[1] + 3 * x[1] ** 2

    x = np.array([1.0, 2.0])
    expected_hessian = np.array([[2, 2], [2, 6]])
    computed_hessian = finite_difference_hessian(f, x)

    np.testing.assert_allclose(computed_hessian, expected_hessian, atol=1e-5)


def perform_bayesian_linear_regression(
    prior_mean, prior_variance, features, target, noise_variance
):
    """
    Perform Bayesian linear regression using a conjugate prior.

    Args:
        prior_mean (np.ndarray): The mean of the prior distribution.
        prior_variance (np.ndarray): The variance of the prior distribution.
        data (np.ndarray): The observed data.

    Returns:
        np.ndarray: The posterior mean.
    """
    # Assuming data is a 2D array with shape (n_samples, n_features)
    X = features
    y = target

    # Compute posterior parameters
    posterior_variance = np.linalg.inv(
        np.linalg.inv(prior_variance) + X.T @ X / noise_variance
    )
    posterior_mean = posterior_variance @ (
        np.linalg.inv(prior_variance) @ prior_mean + X.T @ y / noise_variance
    )

    return posterior_mean, posterior_variance


if __name__ == "__main__":
    pytest.main()
