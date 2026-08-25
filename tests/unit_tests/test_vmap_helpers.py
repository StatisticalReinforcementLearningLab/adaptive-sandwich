import jax.numpy as jnp
import numpy as np

from lifejacket.vmap_helpers import stack_batched_arg_lists_into_tensors


def test_stack_batched_arg_lists_into_tensors_plain_python_scalars():
    # A list of plain Python floats -- never isinstance(..., jnp.ndarray) -- takes the
    # final "list of scalars" branch.
    batched_arg_lists = [[0.1, 0.2, 0.3]]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0]
    np.testing.assert_allclose(tensors[0], jnp.array([0.1, 0.2, 0.3]))


def test_stack_batched_arg_lists_into_tensors_0d_array_scalars():
    # A list of 0-D jnp arrays: isinstance(x, jnp.ndarray) is True (unlike a plain Python
    # float), which is exactly what a scalar argument becomes once it crosses a jax.jit
    # boundary. Before the fix, this hit the "vector (1D array)" branch (since a 0-D array
    # isn't > 2 or == 2 dimensional, the old code assumed it must be 1-D) and failed the
    # `assert batched_arg_list[0].ndim == 1` inside that branch. It must be treated the same
    # as a list of plain Python scalars.
    batched_arg_lists = [[jnp.array(0.1), jnp.array(0.2), jnp.array(0.3)]]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0]
    assert tensors[0].shape == (3,)
    np.testing.assert_allclose(tensors[0], jnp.array([0.1, 0.2, 0.3]))


def test_stack_batched_arg_lists_into_tensors_1d_arrays():
    batched_arg_lists = [[jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0]
    assert tensors[0].shape == (2, 2)
    np.testing.assert_allclose(tensors[0], jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_stack_batched_arg_lists_into_tensors_plain_python_sequences():
    # A list of plain Python lists (not yet jnp arrays) should still be cast and vstacked,
    # same as real 1-D arrays.
    batched_arg_lists = [[[1.0, 2.0], [3.0, 4.0]]]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0]
    assert tensors[0].shape == (2, 2)
    np.testing.assert_allclose(tensors[0], jnp.array([[1.0, 2.0], [3.0, 4.0]]))


def test_stack_batched_arg_lists_into_tensors_2d_arrays():
    batched_arg_lists = [
        [jnp.array([[1.0, 2.0], [3.0, 4.0]]), jnp.array([[5.0, 6.0], [7.0, 8.0]])]
    ]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0]
    assert tensors[0].shape == (2, 2, 2)


def test_stack_batched_arg_lists_into_tensors_multiple_arg_positions():
    # Mixes a 0-D-scalar arg position with a 1-D-vector arg position, matching how a real
    # args tuple has several positions of different kinds.
    batched_arg_lists = [
        [jnp.array(1.0), jnp.array(2.0)],
        [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])],
    ]

    tensors, batch_axes = stack_batched_arg_lists_into_tensors(batched_arg_lists)

    assert batch_axes == [0, 0]
    assert tensors[0].shape == (2,)
    assert tensors[1].shape == (2, 2)


def test_stack_batched_arg_lists_into_tensors_already_stacked_array_passthrough():
    # A position that is ALREADY a single (bucket_size, ...) array -- rather than a Python
    # list of bucket_size per-subject values -- must be used as-is, not re-derived via
    # list(...)/jnp.stack. This is the shape
    # post_deployment_analysis._rebuild_bucket_from_jit_arrays produces when reconstructing
    # an UpdateArgBucket from a jax.jit-traced argument instead of a plain per-subject list
    # (see that function's own comment for why avoiding the round trip matters).
    already_stacked = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    tensors, batch_axes = stack_batched_arg_lists_into_tensors([already_stacked])

    assert batch_axes == [0]
    assert tensors[0].shape == (3, 2)
    np.testing.assert_allclose(tensors[0], already_stacked)
    # Passed straight through, not rebuilt into a new array via stack/vstack.
    assert tensors[0] is already_stacked
