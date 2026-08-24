import numpy as np

from lifejacket import helper_functions


def test_invert_bread_matrix_2x2_block_diagonal():
    # Test case 1: Simple 2x2 block matrix
    bread = np.array([[4, 0, 0, 0], [0, 4, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread_inverse = np.array(
        [[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]]
    )
    np.testing.assert_allclose(
        helper_functions.invert_bread_matrix(bread, beta_dim, theta_dim),
        expected_bread_inverse,
        rtol=1e-05,
    )


def test_invert_bread_matrix_4x4_block_diagonal():
    bread = np.array([[4, 1, 0, 0], [1, 4, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]])
    beta_dim = 2
    theta_dim = 2
    expected_bread_inverse = np.array(
        [
            [0.26666667, -0.06666667, 0, 0],
            [-0.06666667, 0.26666667, 0, 0],
            [0, 0, 0.66666667, -0.33333333],
            [0, 0, -0.33333333, 0.66666667],
        ]
    )
    np.testing.assert_allclose(
        helper_functions.invert_bread_matrix(bread, beta_dim, theta_dim),
        expected_bread_inverse,
        rtol=1e-05,
    )


def test_invert_bread_matrix_6x6_block_diagonal():
    bread = np.array(
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
    expected_bread_inverse = np.array(
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
        helper_functions.invert_bread_matrix(bread, beta_dim, theta_dim),
        expected_bread_inverse,
        rtol=1e-05,
    )


def test_invert_bread_matrix_6x6_block_lower_triangular():
    bread = np.array(
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

    expected_bread_inverse = np.linalg.inv(bread)

    np.testing.assert_allclose(
        helper_functions.invert_bread_matrix(bread, beta_dim, theta_dim),
        expected_bread_inverse,
        atol=1e-12,
    )


def test_invert_bread_matrix_different_beta_theta_block_lower_triangular():
    bread = np.array(
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

    expected_bread_inverse = np.linalg.inv(bread)

    np.testing.assert_allclose(
        helper_functions.invert_bread_matrix(bread, beta_dim, theta_dim),
        expected_bread_inverse,
        atol=1e-12,
    )


def test_append_new_block_row_to_block_lower_triangular_matrix_equal_dims():
    # cached_block and the new block are the same size (previous_dim ==
    # new_dim == 2) -- the case that happened to look correct under the
    # original jnp.block flat-list bug, since a flat hstack of matching-row-
    # count pieces at least produces *some* rectangular result.
    cached_block = np.array([[1.0, 2.0], [3.0, 4.0]])
    new_block_row = np.array([[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])

    result = helper_functions.append_new_block_row_to_block_lower_triangular_matrix(
        cached_block, new_block_row
    )

    assert result.shape == (4, 4)
    expected = np.array(
        [
            [1.0, 2.0, 0.0, 0.0],
            [3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_append_new_block_row_to_block_lower_triangular_matrix_unequal_dims():
    # cached_block is larger than the newly appended block (previous_dim=4 >
    # new_dim=2) -- exactly the shape under which the original bug produced a
    # non-square (2, 8) result instead of the required square (6, 6) matrix.
    cached_block = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]
    )
    new_block_row = np.array(
        [
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            [23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
        ]
    )

    result = helper_functions.append_new_block_row_to_block_lower_triangular_matrix(
        cached_block, new_block_row
    )

    assert result.shape == (6, 6)
    expected = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            [5.0, 6.0, 7.0, 8.0, 0.0, 0.0],
            [9.0, 10.0, 11.0, 12.0, 0.0, 0.0],
            [13.0, 14.0, 15.0, 16.0, 0.0, 0.0],
            [17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
            [23.0, 24.0, 25.0, 26.0, 27.0, 28.0],
        ]
    )
    np.testing.assert_allclose(result, expected)
