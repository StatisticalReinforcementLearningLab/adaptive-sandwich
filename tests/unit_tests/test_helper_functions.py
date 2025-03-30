import numpy as np

import helper_functions


def test_invert_inverse_bread_matrix_2x2_block_diagonal():
    # Test case 1: Simple 2x2 block matrix
    inverse_bread = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]]
    )
    np.testing.assert_allclose(
        helper_functions.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_4x4_block_diagonal():
    inverse_bread = np.array([[4, 1, 0, 0], [1, 4, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [
            [0.26666667, -0.06666667, 0, 0],
            [-0.06666667, 0.26666667, 0, 0],
            [0, 0, 0.66666667, -0.33333333],
            [0, 0, -0.33333333, 0.66666667],
        ]
    )
    np.testing.assert_allclose(
        helper_functions.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_6x6_block_diagonal():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0],
            [0, 0, 2, 1, 0, 0],
            [0, 0, 1, 2, 0, 0],
            [0, 0, 0, 0, 3, 1],
            [0, 0, 0, 0, 1, 3],
        ]
    )
    beta_dim = 2
    theta_dim = 2
    expected_bread = np.array(
        [
            [0.26666667, -0.06666667, 0, 0, 0, 0],
            [-0.06666667, 0.26666667, 0, 0, 0, 0],
            [0, 0, 0.66666667, -0.33333333, 0, 0],
            [0, 0, -0.33333333, 0.66666667, 0, 0],
            [0, 0, 0, 0, 0.375, -0.125],
            [0, 0, 0, 0, -0.125, 0.375],
        ]
    )
    np.testing.assert_allclose(
        helper_functions.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        rtol=1e-05,
    )


def test_invert_inverse_bread_matrix_6x6_block_lower_triangular():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0],
            [7, 1, 2, 1, 0, 0],
            [1, 7, 1, 2, 0, 0],
            [5, 1, 6, 1, 3, 1],
            [1, 5, 1, 6, 1, 3],
        ]
    )
    beta_dim = 2
    theta_dim = 2

    expected_bread = np.linalg.inv(inverse_bread)

    np.testing.assert_allclose(
        helper_functions.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        atol=1e-12,
    )


def test_invert_inverse_bread_matrix_different_beta_theta_block_lower_triangular():
    inverse_bread = np.array(
        [
            [4, 1, 0, 0, 0, 0, 0],
            [1, 4, 0, 0, 0, 0, 0],
            [7, 1, 2, 1, 0, 0, 0],
            [1, 7, 1, 2, 0, 0, 0],
            [5, 1, 6, 1, 3, 1, 4],
            [1, 5, 1, 6, 1, 3, 4],
            [8, 7, 6, 5, 9, 3, 5],
        ]
    )
    beta_dim = 2
    theta_dim = 3

    expected_bread = np.linalg.inv(inverse_bread)

    np.testing.assert_allclose(
        helper_functions.invert_inverse_bread_matrix(
            inverse_bread, beta_dim, theta_dim
        ),
        expected_bread,
        atol=1e-12,
    )
