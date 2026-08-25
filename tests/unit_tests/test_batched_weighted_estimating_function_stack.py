import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lifejacket.batched_weighted_estimating_function_stack import (
    build_action_prob_layer_precompute,
    build_inference_layer_precompute,
    build_update_layer_precompute,
    check_batched_algorithm_estimating_function_args_equivalent,
    check_batched_inference_estimating_function_args_equivalent,
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


def test_build_action_prob_layer_precompute_raises_clearly_on_none_arg_value():
    """
    action_prob_func_args positions are always required (they always flow
    into a real call to action_prob_func), unlike alg_update_func_args'/
    inference_func_args' override-only positions, which may legitimately be
    None. A None value here must fail loudly and clearly at precompute time,
    not silently become a dtype=object array that later crashes confusingly
    deep inside jax.vmap.
    """
    subject_ids = np.array([1, 2])
    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {1: (jnp.array([1.0]), None), 2: (jnp.array([1.0]), 1.0)},
        2: {1: (jnp.array([1.0]), 2.0), 2: (jnp.array([1.0]), -1.0)},
    }
    action_by_decision_time_by_subject_id = {1: {1: 0, 2: 1}, 2: {1: 1, 2: 0}}
    policy_num_by_decision_time_by_subject_id = {1: {1: 1, 2: 1}, 2: {1: 1, 2: 1}}

    with pytest.raises(ValueError, match="None value"):
        build_action_prob_layer_precompute(
            subject_ids,
            action_prob_func_args_by_subject_id_by_decision_time,
            action_by_decision_time_by_subject_id,
            policy_num_by_decision_time_by_subject_id,
            {},
            1,
            0,
        )


def _linear_action_prob_func(beta, treat_state):
    return jnp.clip(beta[0] * treat_state, 0.01, 0.99)


def _algorithm_estimating_func_using_action_prob(beta, action_prob, action_prob_times):
    del action_prob_times  # only needed for reconstruction indexing, not the math
    return beta * action_prob


def test_check_batched_algorithm_estimating_function_args_equivalent_passes_on_consistent_data():
    """
    Positive counterpart to
    test_batched_algorithm_data_check_detects_the_same_inconsistency_as_the_original
    (tests/unit_tests/test_post_deployment_analysis.py), which only exercises
    the raise path (that fixture is deliberately inconsistent for other
    testing purposes). This one hand-builds a single-subject, single-update
    fixture where the recorded action_prob_func_args beta and the recorded
    "original" action probability in update_func_args are BOTH self-
    consistent with all_post_update_betas and action_prob_func -- i.e. real,
    well-formed data -- and confirms the check passes silently.
    """
    subject_ids = np.array([1])
    beta_index_by_policy_num = {2: 0}
    initial_policy_num = 1
    action_prob_func_args_beta_index = 0
    all_post_update_betas = jnp.array([[2.0]])

    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {1: (jnp.array([1.0]), 0.5)},
        # Decision time 2 is under the update (policy_num=2); its recorded
        # beta already matches all_post_update_betas[0] -- real,
        # self-consistent data, not a fabricated placeholder.
        2: {1: (jnp.array([2.0]), 0.5)},
    }
    action_by_decision_time_by_subject_id = {1: {1: 0, 2: 1}}
    policy_num_by_decision_time_by_subject_id = {1: {1: 1, 2: 2}}

    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )

    # The "original" recorded action probability must be computed the same
    # way action_prob_func would compute it, for this fixture to actually be
    # self-consistent (this is what a real, correctly-collected dataset
    # looks like -- not something the check itself should ever have to
    # fabricate).
    recorded_action_prob = jnp.array([_linear_action_prob_func(jnp.array([2.0]), 0.5)])
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {1: (jnp.array([2.0]), recorded_action_prob, jnp.array([2]))}
    }
    update_layer = build_update_layer_precompute(
        subject_ids,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        2,  # alg_update_func_args_action_prob_times_index
        action_prob_layer,
    )

    betas = all_post_update_betas
    _, pi_beta_grid = compute_action_prob_layer_outputs(
        _linear_action_prob_func,
        action_prob_func_args_beta_index,
        action_prob_layer,
        betas,
        True,
    )

    # Must not raise.
    check_batched_algorithm_estimating_function_args_equivalent(
        _algorithm_estimating_func_using_action_prob,
        betas,
        0,  # alg_update_func_args_beta_index
        -1,  # alg_update_func_args_previous_betas_index
        1,  # alg_update_func_args_action_prob_index
        action_prob_layer,
        update_layer,
        pi_beta_grid,
    )


def _inference_estimating_func_using_action_prob(theta, action_prob):
    return theta * action_prob


def test_check_batched_inference_estimating_function_args_equivalent_passes_on_consistent_data():
    """
    check_batched_inference_estimating_function_args_equivalent had no test
    anywhere in the suite before this: the one fixture that exercises it
    end-to-end (setup_data_two_loss_functions_use_action_probs_both_sides in
    test_post_deployment_analysis.py) is deliberately inconsistent on the
    ALGORITHM side, so that check raises first and the inference check never
    actually runs. This hand-builds a minimal, genuinely self-consistent
    fixture (no policy updates at all, to isolate the inference side) and
    confirms the check passes silently on well-formed data -- the inference
    counterpart to
    test_check_batched_algorithm_estimating_function_args_equivalent_passes_on_consistent_data
    above.
    """
    subject_ids = np.array([1])
    action_prob_func_args_beta_index = 0

    # No policy updates at all (everything stays on the initial policy) --
    # isolates the inference side from the algorithm-side override logic
    # exercised by the test above.
    action_prob_func_args_by_subject_id_by_decision_time = {1: {1: (jnp.array([2.0]), 0.5)}}
    action_by_decision_time_by_subject_id = {1: {1: 0}}
    policy_num_by_decision_time_by_subject_id = {1: {1: 1}}

    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        {},  # beta_index_by_policy_num -- no updates
        1,  # initial_policy_num
        action_prob_func_args_beta_index,
    )
    inference_action_prob_decision_times_by_subject_id = {1: np.array([1])}
    inference_func_args_action_prob_index = 1
    # Self-consistent: computed the same way pi_beta_grid will compute it
    # (beta_row_index is -1 everywhere here since there are no updates, so
    # the raw recorded beta [2.0] is what's actually used either way).
    recorded_action_prob = jnp.array([_linear_action_prob_func(jnp.array([2.0]), 0.5)])
    theta = jnp.array([3.0])
    inference_func_args_by_subject_id = {1: (theta, recorded_action_prob)}

    inference_layer = build_inference_layer_precompute(
        inference_func_args_by_subject_id,
        inference_func_args_action_prob_index,
        inference_action_prob_decision_times_by_subject_id,
        action_prob_layer,
    )

    # A single, never-substituted row: beta_row_index is -1 everywhere in
    # this fixture (no updates), so this value is never actually gathered --
    # it just needs to exist with a valid shape (a genuinely empty (0, 1)
    # betas array hits an unrelated, pre-existing edge case in
    # build_threaded_action_prob_beta_tensor's clamping for zero-update
    # studies, out of scope for this test).
    betas = jnp.zeros((1, 1))
    _, pi_beta_grid = compute_action_prob_layer_outputs(
        _linear_action_prob_func,
        action_prob_func_args_beta_index,
        action_prob_layer,
        betas,
        True,
    )

    # Must not raise.
    check_batched_inference_estimating_function_args_equivalent(
        _inference_estimating_func_using_action_prob,
        theta,
        0,  # inference_func_args_theta_index
        inference_func_args_action_prob_index,
        action_prob_layer,
        inference_layer,
        pi_beta_grid,
    )
