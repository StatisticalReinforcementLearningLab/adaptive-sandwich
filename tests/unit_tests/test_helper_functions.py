import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
# compute_row_chunked_jacobian: the package-wide memory-bounded reverse-mode Jacobian. Same
# quantity as jax.jacrev, with the output cotangent basis walked in chunks so peak memory
# scales with chunk_size rather than out_dim. Two call sites depend on it (the bootstrap's
# per-iterate Newton Jacobian inside a jax.jit trace, and the diagnostic suite's Jacobian-drift
# check), so the properties pinned here are: exactness at every chunk size, the padding path
# for a chunk size that does not divide out_dim, and -- the one most easily lost in a refactor
# -- that the chunk loop does NOT unroll under jit.
# ---------------------------------------------------------------------------


def _nonlinear_map(v):
    # R^5 -> R^7, deliberately non-square and with every output row depending on a different
    # mix of inputs, so a mis-sliced or mis-ordered chunk cannot pass by symmetry.
    return jnp.stack(
        [
            v[0] ** 2 + jnp.sin(v[1]),
            v[1] * v[2],
            jnp.exp(0.3 * v[0]) + v[3],
            jnp.tanh(v[2]) - v[4],
            v[0] * v[1] * v[2],
            jnp.log1p(v[3] ** 2),
            v[4] ** 3,
        ]
    )


_NONLINEAR_MAP_POINT = jnp.array([0.3, -0.7, 0.5, 1.1, -0.2])


# out_dim is 7, so 2/3/4/5/6 all leave a short final chunk (the padding path), 7 is exactly one
# chunk, and 8/100 exceed the output dimension entirely.
@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4, 5, 6, 7, 8, 100])
def test_compute_row_chunked_jacobian_matches_jacrev_at_every_chunk_size(chunk_size):
    expected = jax.jacrev(_nonlinear_map)(_NONLINEAR_MAP_POINT)
    chunked = helper_functions.compute_row_chunked_jacobian(
        _nonlinear_map, _NONLINEAR_MAP_POINT, chunk_size
    )

    assert chunked.shape == expected.shape
    # Not exact equality: lax.map's body is compiled while eager jacrev is not, so XLA is free
    # to fuse the two differently (measured ~1 ulp in float64, ~5e-7 in the float32 jax runs by
    # default here). The tolerance is float32 noise, several orders below any real chunking bug,
    # which would misplace whole rows.
    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("chunk_size", [None, 0])
def test_compute_row_chunked_jacobian_unchunked_requests_take_the_jacrev_path(
    chunk_size,
):
    # None and 0 are documented to return jax.jacrev's result verbatim (no vjp/lax.map
    # machinery at all), so here -- and only here -- bitwise agreement is the contract.
    np.testing.assert_array_equal(
        np.asarray(
            helper_functions.compute_row_chunked_jacobian(
                _nonlinear_map, _NONLINEAR_MAP_POINT, chunk_size
            )
        ),
        np.asarray(jax.jacrev(_nonlinear_map)(_NONLINEAR_MAP_POINT)),
    )


def test_compute_row_chunked_jacobian_final_short_chunk_keeps_its_real_rows():
    # A short final chunk is padded with all-zero cotangent rows whose all-zero Jacobian rows
    # are then dropped. Off-by-one in that drop returns the PADDING instead of the last real
    # rows -- which is silent, because zeros are a plausible-looking Jacobian block. out_dim=7
    # with chunk_size=4 puts rows 4..6 in the padded chunk.
    expected = np.asarray(jax.jacrev(_nonlinear_map)(_NONLINEAR_MAP_POINT))
    chunked = np.asarray(
        helper_functions.compute_row_chunked_jacobian(
            _nonlinear_map, _NONLINEAR_MAP_POINT, 4
        )
    )

    assert np.all(np.any(expected[4:] != 0.0, axis=1)), (
        "fixture no longer has nonzero trailing Jacobian rows, so it cannot detect padding "
        "leaking into the result"
    )
    assert np.all(np.any(chunked[4:] != 0.0, axis=1))
    np.testing.assert_allclose(chunked[4:], expected[4:], rtol=1e-6, atol=1e-6)


def test_compute_row_chunked_jacobian_is_correct_inside_a_jit_trace():
    # The bootstrap's Newton Jacobian calls this from inside @jax.jit, so tracing it must work
    # and give the same answer. Compared against a jitted jacrev, not an eager one: comparing a
    # compiled result against an uncompiled one folds XLA fusion differences into the tolerance
    # for no reason.
    expected = jax.jit(jax.jacrev(_nonlinear_map))(_NONLINEAR_MAP_POINT)
    chunked = jax.jit(
        lambda v: helper_functions.compute_row_chunked_jacobian(_nonlinear_map, v, 3)
    )(_NONLINEAR_MAP_POINT)

    np.testing.assert_allclose(
        np.asarray(chunked), np.asarray(expected), rtol=1e-6, atol=1e-6
    )


def test_compute_row_chunked_jacobian_does_not_unroll_the_chunk_loop():
    # THE regression this helper exists for. A Python `for` loop over the chunks passes every
    # numerical test above and fails only this one: under jax.jit it unrolls into num_chunks
    # copies of the backward graph, whose intermediates XLA may keep live simultaneously --
    # reinstating exactly the peak-memory blowup the chunking was added to prevent (real 24GB
    # OOM crashes at oralytics scale; docs/adr/0001).
    def wide_map(v):
        return jnp.concatenate(
            [jnp.sin(v * (i + 1.0)) + v[0] * v[1] for i in range(15)]
        )

    point = jnp.array([0.4, -0.6, 0.9, 0.1])
    out_dim = int(wide_map(point).shape[0])

    jaxprs = {
        chunk_size: jax.make_jaxpr(
            lambda v, c=chunk_size: helper_functions.compute_row_chunked_jacobian(
                wide_map, v, c
            )
        )(point)
        for chunk_size in (1, out_dim // 4, out_dim)
    }

    for chunk_size, jaxpr in jaxprs.items():
        assert "scan[" in str(jaxpr), (
            f"chunk_size={chunk_size} produced no scan primitive -- the chunk loop is not "
            "jax.lax.map any more"
        )
    # The count of TOP-LEVEL equations, not the printed length: the chunk shapes appear in the
    # jaxpr's type annotations, so the text differs harmlessly between chunk sizes while the
    # graph structure does not. 60 chunks of 1 row and 1 chunk of 60 rows must trace to the
    # same number of equations, the scan body holding the one backward pass either way; a
    # Python loop over the chunks traces to one copy per chunk (measured here: 350 equations
    # at 1 chunk against 12746 at 60).
    equation_counts = {
        chunk_size: len(jaxpr.jaxpr.eqns) for chunk_size, jaxpr in jaxprs.items()
    }
    assert len(set(equation_counts.values())) == 1, equation_counts


def test_compute_row_chunked_jacobian_rejects_inputs_it_cannot_chunk():
    with pytest.raises(ValueError, match="non-negative"):
        helper_functions.compute_row_chunked_jacobian(
            _nonlinear_map, _NONLINEAR_MAP_POINT, -1
        )
    # The flat output basis is only well-defined for a single array in and a single array out;
    # a pytree must say so rather than failing somewhere inside jnp.eye.
    with pytest.raises(TypeError, match="pytree"):
        helper_functions.compute_row_chunked_jacobian(
            lambda p: p["a"] * 2.0, {"a": jnp.ones(3)}, 2
        )
    with pytest.raises(TypeError, match="single array"):
        helper_functions.compute_row_chunked_jacobian(
            lambda v: {"a": v * 2.0}, _NONLINEAR_MAP_POINT, 2
        )


def test_resolve_jacobian_row_chunk_size_is_one_shared_policy_object():
    # The auto policy moved here from post_deployment_analysis so diagnostics.py could import it
    # without a cycle; that module keeps a re-export for its own call site and for the existing
    # tests in test_post_deployment_analysis.py. If the re-export silently disappears those
    # would break far from the cause, and, worse, a second copy of the policy could be
    # reintroduced there without anything noticing.
    from lifejacket import post_deployment_analysis

    assert (
        post_deployment_analysis.resolve_jacobian_row_chunk_size
        is helper_functions.resolve_jacobian_row_chunk_size
    )


# ---------------------------------------------------------------------------
# clopper_pearson_upper_bound: moved here from the deleted simulator_calibration
# module, which required a caller-supplied simulator and had no in-package callers.
# This helper does: diagnostics.py reads it at six sites for the root-failure,
# branch-change and domain-failure bounds.
# ---------------------------------------------------------------------------


def test_clopper_pearson_zero_failures_matches_the_closed_form():
    import math

    from lifejacket.helper_functions import clopper_pearson_upper_bound

    # With no observed failures the bound reduces exactly to 1 - alpha**(1/n).
    assert math.isclose(
        clopper_pearson_upper_bound(0, 59, 0.95), 1 - 0.05 ** (1 / 59), rel_tol=1e-9
    )
    assert math.isclose(
        clopper_pearson_upper_bound(0, 299, 0.95), 1 - 0.05 ** (1 / 299), rel_tol=1e-9
    )
    # More trials at the same confidence must tighten the bound.
    assert clopper_pearson_upper_bound(0, 299, 0.95) < clopper_pearson_upper_bound(
        0, 59, 0.95
    )


def test_clopper_pearson_legitimate_boundaries():
    import math

    from lifejacket.helper_functions import clopper_pearson_upper_bound

    # Nothing observed: the bound is genuinely undefined, not 0 and not 1.
    assert math.isnan(clopper_pearson_upper_bound(0, 0, 0.95))
    # Everything failed: the bound really is 1.
    assert clopper_pearson_upper_bound(10, 10, 0.95) == 1.0


def test_clopper_pearson_rejects_inputs_that_would_read_as_good_news():
    import pytest

    from lifejacket.helper_functions import clopper_pearson_upper_bound

    for bad_confidence in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="confidence_level"):
            clopper_pearson_upper_bound(0, 10, bad_confidence)
    # More failures than trials is an impossible observation; it used to return a
    # legitimate-looking 1.0 and hide whatever miscounted upstream.
    with pytest.raises(ValueError, match="num_failures"):
        clopper_pearson_upper_bound(11, 10, 0.95)
    with pytest.raises(ValueError, match="num_failures"):
        clopper_pearson_upper_bound(-1, 10, 0.95)
    # A negative trial count used to return nan, which callers propagate into a metric.
    with pytest.raises(ValueError, match="num_trials"):
        clopper_pearson_upper_bound(0, -5, 0.95)


def test_clopper_pearson_warns_on_a_sub_50_percent_confidence_level(caplog):
    import logging

    from lifejacket.helper_functions import clopper_pearson_upper_bound

    # 0.05 is a LEGAL confidence level, so this cannot be rejected -- but passing alpha where a
    # confidence level belongs is the realistic mix-up, and it is dangerous because it returns a
    # ~50x TIGHTER bound (0.0051 vs 0.2589 for 0 of 10) that reads as much stronger evidence.
    # The value is still computed; the caller is told why it looks too good.
    with caplog.at_level(logging.WARNING, logger="lifejacket.helper_functions"):
        bound = clopper_pearson_upper_bound(0, 10, 0.05)
    assert bound < clopper_pearson_upper_bound(0, 10, 0.95)
    assert "confidence_level" in caplog.text
    assert "alpha" in caplog.text

    # A conventional level must not warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="lifejacket.helper_functions"):
        clopper_pearson_upper_bound(0, 10, 0.95)
    assert caplog.text == ""
