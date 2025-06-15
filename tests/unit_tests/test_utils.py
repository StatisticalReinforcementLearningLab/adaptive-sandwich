import numpy as np
from jax import numpy as jnp

from tests.utils import finite_difference_hessian


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

    def f(x):  # pylint: disable=unused-argument
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
        prior_mean (np.ndarray):
            The mean of the prior distribution.
        prior_variance (np.ndarray):
            The variance of the prior distribution.
        features (np.ndarray):
            The observed data (2D array)
        target (np.ndarray):
            The target variable.
        noise_variance (float):
            The noise variance in the linear model.

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
        np.linalg.inv(prior_variance) @ prior_mean.reshape(-1, 1)
        + X.T @ y / noise_variance
    )

    return posterior_mean.flatten(), posterior_variance


def assert_dict_with_arrays_equal(d1, d2):
    """
    Recursively compare two dictionaries (or nested structures) that may contain numpy arrays.
    """

    try:
        assert type(d1) == type(d2), f"Type mismatch: {type(d1)} != {type(d2)}"
    except AssertionError as e:
        breakpoint()
    if isinstance(d1, dict):
        assert d1.keys() == d2.keys(), f"Dict keys mismatch: {d1.keys()} != {d2.keys()}"
        for k in d1:
            assert_dict_with_arrays_equal(d1[k], d2[k])
    elif isinstance(d1, (list, tuple)):
        assert len(d1) == len(d2), f"Length mismatch: {len(d1)} != {len(d2)}"
        for v1, v2 in zip(d1, d2):
            assert_dict_with_arrays_equal(v1, v2)
    elif isinstance(d1, (np.ndarray, jnp.ndarray)):
        np.testing.assert_array_equal(d1, d2)
    else:
        assert d1 == d2, f"Value mismatch: {d1} != {d2}"
