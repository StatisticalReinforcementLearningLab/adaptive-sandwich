import jax.numpy as jnp
import numpy as np
import pytest

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


def test_stack_batched_arg_lists_into_tensors_rejects_bare_0d_array():
    # A bare 0-D array at a position is a caller contract violation (a
    # scalar-per-subject position must be a plain Python LIST of scalars):
    # it has no axis-0 batch dimension, so the passthrough branch accepting
    # it would only fail later inside jax.vmap with a confusing non-local
    # error. Must fail loudly and locally instead.
    with pytest.raises(TypeError, match="0-D"):
        stack_batched_arg_lists_into_tensors([jnp.array(3.0)])
    with pytest.raises(TypeError, match="0-D"):
        stack_batched_arg_lists_into_tensors([np.array(3.0)])


# Every error below must name the offending argument POSITION: the underlying
# jnp.stack/vstack/array errors do not, and by the time this function runs the
# caller has pivoted the per-subject tuples into per-position lists, so the
# position is the one coordinate the user can act on.


def test_stack_batched_arg_lists_into_tensors_rejects_3d_arrays_naming_position():
    batched_arg_lists = [
        [jnp.array(1.0), jnp.array(2.0)],
        [jnp.ones((2, 2, 2)), jnp.ones((2, 2, 2))],
    ]

    with pytest.raises(TypeError, match=r"position 1.*more than 2"):
        stack_batched_arg_lists_into_tensors(batched_arg_lists)


def test_stack_batched_arg_lists_into_tensors_mismatched_2d_shapes_error_names_position():
    # Shape-bucketing normally guarantees identical shapes within a batch, but direct
    # callers exist; a mismatch must not surface as jax's bare "All input arrays must
    # have the same shape."
    batched_arg_lists = [[jnp.ones((2, 2)), jnp.ones((3, 2))]]

    with pytest.raises(ValueError, match=r"position 0.*same array shape"):
        stack_batched_arg_lists_into_tensors(batched_arg_lists)


def test_stack_batched_arg_lists_into_tensors_mismatched_1d_lengths_error_names_position():
    batched_arg_lists = [[jnp.array([1.0, 2.0]), jnp.array([1.0, 2.0, 3.0])]]

    with pytest.raises(ValueError, match=r"position 0.*same vector length"):
        stack_batched_arg_lists_into_tensors(batched_arg_lists)


def test_stack_batched_arg_lists_into_tensors_rejects_nested_sequence_naming_position():
    # A plain-sequence position must be FLAT: a nested list casts to a 2-D array, which
    # previously died on a bare `assert ... ndim == 1` with no message at all.
    batched_arg_lists = [[[[1.0, 2.0], [3.0, 4.0]]]]

    with pytest.raises(TypeError, match=r"position 0.*FLAT"):
        stack_batched_arg_lists_into_tensors(batched_arg_lists)


def test_stack_batched_arg_lists_into_tensors_rejects_strings_naming_position():
    # Strings dodge the vector branch (deliberately) and land in the scalar branch,
    # where jnp.array raises a jax-internal TypeError; it must be wrapped with the
    # position and the list of supported kinds. (For user-supplied argument tuples,
    # input_checks.require_supplied_arg_types_supported already rejects this earlier
    # with the key and subject id attached.)
    batched_arg_lists = [["logistic", "logistic"]]

    with pytest.raises(TypeError, match=r"position 0.*'str'"):
        stack_batched_arg_lists_into_tensors(batched_arg_lists)
