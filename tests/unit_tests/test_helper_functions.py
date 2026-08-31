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


# ---------------------------------------------------------------------------
# array_scale_absolute_tolerance: the shared scale-aware atol floor for comparing two float
# computations of the same quantity (see docs/adr/0002's correction section for the raw-units
# bug class it exists to prevent).
# ---------------------------------------------------------------------------


def test_array_scale_absolute_tolerance_tracks_the_reference_scale():
    import numpy as np

    from lifejacket.helper_functions import array_scale_absolute_tolerance

    # A near-zero component with noise proportional to the array's magnitude: the exact
    # situation the fixed atol=1e-7 mishandled. The scale-aware floor must accept it at ANY
    # reward scale, because the noise and the tolerance grow together.
    for scale in (1.0, 1e4, 1e8):
        reference = np.array([scale, 0.0])
        other = np.array([scale, 3e-7 * scale])
        np.testing.assert_allclose(
            reference,
            other,
            atol=array_scale_absolute_tolerance(reference),
            rtol=1e-3,
        )

    # And the same comparison fails under the legacy fixed tolerance once the scale is large --
    # the false alarm this replaces.
    import pytest

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.array([1e8, 0.0]),
            np.array([1e8, 3e-7 * 1e8]),
            atol=1e-7,
            rtol=1e-3,
        )


def test_array_scale_absolute_tolerance_degenerate_references():
    import numpy as np

    from lifejacket.helper_functions import array_scale_absolute_tolerance

    # All-zero reference: exact agreement is the right demand (atol 0), and identical zeros
    # still compare equal.
    assert array_scale_absolute_tolerance(np.zeros(4)) == 0.0
    np.testing.assert_allclose(np.zeros(4), np.zeros(4), atol=0.0, rtol=1e-3)
    # Empty reference: nothing to compare, no crash.
    assert array_scale_absolute_tolerance(np.array([])) == 0.0
