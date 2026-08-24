import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lifejacket.batched_weighted_estimating_function_stack import (
    build_action_prob_layer_precompute,
    compute_action_prob_layer_outputs,
)
from lifejacket.helper_functions import compute_subject_radon_nikodym_weights


def _build_incremental_recruitment_precompute():
    """
    Two subjects, three decision times (1, 2, 3). Subject 1 is active the
    whole time; subject 2 is recruited at t=2 (inactive, not gapped, at
    t=1) -- this is what actually produces a self-padded cell in practice
    (staggered recruitment), as opposed to an invalid intra-window gap.
    """
    subject_ids = np.array([1, 2])
    beta_index_by_policy_num = {}
    initial_policy_num = 1
    action_prob_func_args_beta_index = 0

    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {1: (jnp.array([1.0]), 1.0), 2: ()},
        2: {1: (jnp.array([1.0]), 2.0), 2: (jnp.array([1.0]), -1.0)},
        3: {1: (jnp.array([1.0]), 3.0), 2: (jnp.array([1.0]), -2.0)},
    }
    action_by_decision_time_by_subject_id = {
        1: {1: 0, 2: 1, 3: 0},
        2: {2: 1, 3: 0},
    }
    policy_num_by_decision_time_by_subject_id = {
        1: {1: 1, 2: 1, 3: 1},
        2: {2: 1, 3: 1},
    }

    precompute = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )
    return precompute, action_prob_func_args_beta_index


def test_self_padded_singular_function_produces_finite_gradients():
    """
    Regression guard using an action_prob_func with a genuine internal
    singularity (division by treat_state) at exactly the value (0.0) a
    naive zero-fill of an invalid/padded cell would use. Confirms the
    self-padded computation (every invalid cell filled with that subject's
    own real, non-zero treat_state) produces finite weights and finite
    gradients wrt betas, for both compute_action_prob_layer_outputs'
    raw_weight_grid and its optional pi_beta_grid.

    Note: empirically (see the investigation in the ADR / PR discussion for
    this test), an equivalent naive zero-fill at the same invalid cell does
    NOT actually produce a NaN gradient wrt betas in this architecture --
    build_threaded_action_prob_beta_tensor's jnp.where gate on the beta
    argument has a select-based (not multiplicative) backward rule that
    routes any NaN produced by treat_state=0 to the non-differentiated raw
    beta branch, never to betas itself. Self-padding is kept anyway as
    defense-in-depth (it keeps forward values at invalid cells meaningful
    rather than function-specific garbage, and doesn't rely on every future
    caller of these arrays routing through that same where-gated
    substitution), but this test intentionally only asserts the positive
    property that actually holds, not a stronger claim about what would
    happen without it.
    """

    def reciprocal_singularity_action_prob_func(beta, treat_state):
        return jnp.clip(beta[0] / treat_state, 0.01, 0.99)

    precompute, beta_index = _build_incremental_recruitment_precompute()
    betas = jnp.array([[1.0]])

    def total_weight(b):
        raw_weight_grid, pi_beta_grid = compute_action_prob_layer_outputs(
            reciprocal_singularity_action_prob_func, beta_index, precompute, b, True
        )
        return jnp.sum(raw_weight_grid) + jnp.sum(pi_beta_grid)

    weight_grid, pi_beta_grid = compute_action_prob_layer_outputs(
        reciprocal_singularity_action_prob_func, beta_index, precompute, betas, True
    )
    assert jnp.all(jnp.isfinite(weight_grid))
    assert jnp.all(jnp.isfinite(pi_beta_grid))

    grad = jax.grad(total_weight)(betas)
    assert jnp.all(jnp.isfinite(grad))


def _build_intra_window_gap_fixture():
    """
    Subject 2 is active at t=1 and t=3 but NOT t=2 -- an intra-window gap,
    which violates the "once active, stays active" invariant the
    Radon-Nikodym weight-window logic assumes (as opposed to staggered
    recruitment, which is not a gap since it's outside the subject's own
    [start, end] window).
    """
    subject_ids = np.array([1, 2])
    beta_index_by_policy_num = {}
    initial_policy_num = 1
    action_prob_func_args_beta_index = 0

    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {1: (jnp.array([1.0]), 1.0), 2: (jnp.array([1.0]), 1.0)},
        2: {1: (jnp.array([1.0]), 2.0), 2: ()},
        3: {1: (jnp.array([1.0]), 3.0), 2: (jnp.array([1.0]), -2.0)},
    }
    action_by_decision_time_by_subject_id = {
        1: {1: 0, 2: 1, 3: 0},
        2: {1: 1, 3: 0},
    }
    policy_num_by_decision_time_by_subject_id = {
        1: {1: 1, 2: 1, 3: 1},
        2: {1: 1, 3: 1},
    }
    return (
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )


def test_build_action_prob_layer_precompute_accepts_contiguous_fixtures():
    # The staggered-recruitment fixture used above has no intra-window gap
    # (subject 2's inactivity at t=1 is before their own window starts) and
    # should build cleanly.
    precompute, _ = _build_incremental_recruitment_precompute()
    assert precompute.active_mask.shape == (2, 3)


def test_build_action_prob_layer_precompute_raises_on_intra_window_gap():
    with pytest.raises(ValueError, match="intra-window gap"):
        build_action_prob_layer_precompute(*_build_intra_window_gap_fixture())


def _dummy_action_prob_func(beta):
    return jnp.clip(beta[0], 0.01, 0.99)


def test_compute_subject_radon_nikodym_weights_raises_on_intra_window_gap():
    beta = jnp.array([0.5])
    # Subject active at 1, 2, 4 but not 3 -- a gap strictly within their own
    # [1, 4] window.
    args_by_decision_time = {1: (beta,), 2: (beta,), 3: (), 4: (beta,)}
    action_by_decision_time = {1: 0, 2: 1, 4: 0}
    policy_num_by_decision_time = {1: 1, 2: 1, 4: 1}

    with pytest.raises(ValueError, match="intra-window gap"):
        compute_subject_radon_nikodym_weights(
            _dummy_action_prob_func,
            0,
            args_by_decision_time,
            args_by_decision_time,
            policy_num_by_decision_time,
            action_by_decision_time,
            {},
        )


def test_compute_subject_radon_nikodym_weights_accepts_contiguous_data():
    beta = jnp.array([0.5])
    args_by_decision_time = {1: (beta,), 2: (beta,), 3: (beta,)}
    action_by_decision_time = {1: 0, 2: 1, 3: 0}
    policy_num_by_decision_time = {1: 1, 2: 1, 3: 1}

    compute_subject_radon_nikodym_weights(
        _dummy_action_prob_func,
        0,
        args_by_decision_time,
        args_by_decision_time,
        policy_num_by_decision_time,
        action_by_decision_time,
        {},
    )
