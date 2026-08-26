import logging

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
    compute_batched_algorithm_component,
    compute_batched_inference_outputs,
    resolve_combine_updates_into_one_vmap,
    self_pad_ragged_args_and_build_mask,
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
    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {1: (jnp.array([2.0]), 0.5)}
    }
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


def test_self_pad_ragged_args_and_build_mask_pads_with_repeated_last_row():
    """
    Three subjects with different per-decision-time history lengths (the
    real-world shape produced by staggered/incremental recruitment). Every
    subject's ragged positions get padded up to the max length (3) by
    repeating that subject's own last row -- never a fabricated zero -- and
    the mask marks exactly which rows are real.
    """
    args_by_subject_id = {
        1: (np.array([1.0, 2.0]), np.array([10, 20])),  # 2 real rows
        2: (np.array([3.0, 4.0, 5.0]), np.array([30, 40, 50])),  # 3 real rows (the max)
        3: (np.array([6.0]), np.array([60])),  # 1 real row
    }

    result = self_pad_ragged_args_and_build_mask(
        args_by_subject_id, ragged_indices=(0, 1), mask_index=2
    )

    np.testing.assert_array_equal(result[1][0], [1.0, 2.0, 2.0])
    np.testing.assert_array_equal(result[1][1], [10, 20, 20])
    np.testing.assert_array_equal(result[1][2], [1.0, 1.0, 0.0])

    np.testing.assert_array_equal(result[2][0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(result[2][2], [1.0, 1.0, 1.0])

    np.testing.assert_array_equal(result[3][0], [6.0, 6.0, 6.0])
    np.testing.assert_array_equal(result[3][1], [60, 60, 60])
    np.testing.assert_array_equal(result[3][2], [1.0, 0.0, 0.0])


def test_self_pad_ragged_args_and_build_mask_rejects_disagreeing_ragged_lengths():
    """
    If two positions meant to represent the same "how much history" concept
    disagree, per subject, on row count, that's a caller error (they aren't
    actually the same ragged axis) and must raise, not silently pick one.
    """
    args_by_subject_id = {
        1: (np.array([1.0, 2.0]), np.array([10, 20, 30])),  # lengths 2 vs 3
    }
    with pytest.raises(ValueError, match="disagreeing row counts"):
        self_pad_ragged_args_and_build_mask(
            args_by_subject_id, ragged_indices=(0, 1), mask_index=2
        )


def test_self_pad_ragged_args_and_build_mask_requires_mask_index_to_be_appended_last():
    args_by_subject_id = {1: (np.array([1.0, 2.0]),)}
    with pytest.raises(ValueError, match="mask_index=0"):
        self_pad_ragged_args_and_build_mask(
            args_by_subject_id, ragged_indices=(0,), mask_index=0
        )


def test_build_update_layer_precompute_mask_index_consolidates_shape_buckets():
    """
    Same three-subject staggered-history-length scenario, run through the
    real build_update_layer_precompute entry point. Without opting in
    (default alg_update_func_args_mask_index=-1), exact-shape bucketing
    produces one bucket per distinct history length (3, matching real
    incremental-recruitment studies producing far more buckets than
    subject-count alone would suggest). Opting in collapses this to exactly
    one bucket per update, regardless of how many distinct history lengths
    exist.
    """
    subject_ids = np.array([1, 2, 3])
    action_prob_func_args_beta_index = 0
    beta_index_by_policy_num = {2: 0}
    initial_policy_num = 1

    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {
            1: (jnp.array([1.0]), 1.0),
            2: (jnp.array([1.0]), 1.0),
            3: (jnp.array([1.0]), 1.0),
        },
        2: {
            1: (jnp.array([1.0]), 1.0),
            2: (jnp.array([1.0]), 1.0),
            3: (jnp.array([1.0]), 1.0),
        },
    }
    action_by_decision_time_by_subject_id = {
        1: {1: 0, 2: 0},
        2: {1: 0, 2: 0},
        3: {1: 0, 2: 0},
    }
    policy_num_by_decision_time_by_subject_id = {
        1: {1: 1, 2: 1},
        2: {1: 1, 2: 1},
        3: {1: 1, 2: 1},
    }
    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )

    # Staggered per-subject history lengths at the one update -- the actual
    # shape produced by incremental recruitment. Args are (beta_placeholder,
    # state): beta is an override position (its raw value is discarded), state
    # is the ragged one.
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            1: (jnp.array([0.0]), jnp.array([1.0, 2.0])),
            2: (jnp.array([0.0]), jnp.array([3.0, 4.0, 5.0])),
            3: (jnp.array([0.0]), jnp.array([6.0])),
        }
    }

    unpadded_layer = build_update_layer_precompute(
        subject_ids,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        -1,  # alg_update_func_args_action_prob_times_index
        action_prob_layer,
    )
    assert len(unpadded_layer.buckets_by_update_index[0]) == 3

    padded_layer = build_update_layer_precompute(
        subject_ids,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        -1,  # alg_update_func_args_action_prob_times_index
        action_prob_layer,
        alg_update_func_args_mask_index=2,
        alg_update_func_args_ragged_indices=(1,),
    )
    buckets = padded_layer.buckets_by_update_index[0]
    assert len(buckets) == 1
    (bucket,) = buckets
    assert sorted(bucket.subject_ids_in_order) == [1, 2, 3]
    # raw_arg_lists has 3 positions now: beta placeholder, padded state, mask.
    assert len(bucket.raw_arg_lists) == 3


def test_compute_batched_algorithm_component_with_mask_matches_unpadded_per_subject_sums():
    """
    End-to-end numeric equivalence: a mask-aware algorithm_estimating_func
    that sums state weighted by the mask must reproduce each subject's own
    (unpadded) sum exactly, regardless of how much padding was needed to
    consolidate the bucket -- proving padding+masking is mathematically a
    no-op on the answer, only on dispatch count.
    """
    subject_ids = np.array([1, 2, 3])
    action_prob_func_args_beta_index = 0
    beta_index_by_policy_num = {2: 0}
    initial_policy_num = 1

    action_prob_func_args_by_subject_id_by_decision_time = {
        1: {
            1: (jnp.array([1.0]), 1.0),
            2: (jnp.array([1.0]), 1.0),
            3: (jnp.array([1.0]), 1.0),
        },
        2: {
            1: (jnp.array([1.0]), 1.0),
            2: (jnp.array([1.0]), 1.0),
            3: (jnp.array([1.0]), 1.0),
        },
    }
    action_by_decision_time_by_subject_id = {
        1: {1: 0, 2: 0},
        2: {1: 0, 2: 0},
        3: {1: 0, 2: 0},
    }
    policy_num_by_decision_time_by_subject_id = {
        1: {1: 1, 2: 1},
        2: {1: 1, 2: 1},
        3: {1: 1, 2: 1},
    }
    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )

    per_subject_state = {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0], 3: [6.0]}
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            sid: (jnp.array([0.0]), jnp.array(state))
            for sid, state in per_subject_state.items()
        }
    }
    update_layer = build_update_layer_precompute(
        subject_ids,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        -1,  # alg_update_func_args_action_prob_times_index
        action_prob_layer,
        alg_update_func_args_mask_index=2,
        alg_update_func_args_ragged_indices=(1,),
    )

    def mask_aware_sum_estimating_func(beta, state, mask):
        return jnp.sum(mask * state).reshape(1)

    betas = jnp.array([[0.0]])
    algorithm_component, bucket_outputs = compute_batched_algorithm_component(
        betas,
        1,  # beta_dim
        mask_aware_sum_estimating_func,
        0,  # alg_update_func_args_beta_index
        -1,  # alg_update_func_args_previous_betas_index
        -1,  # alg_update_func_args_action_prob_index
        action_prob_layer,
        update_layer,
        None,  # pi_beta_grid
        jnp.ones((3, 1)),  # rl_weight_products
    )

    assert len(bucket_outputs) == 1
    by_subject_pos = {
        int(pos): float(value[0])
        for pos, value in zip(
            update_layer.buckets_by_update_index[0][0].subject_positions,
            bucket_outputs[0],
            strict=True,
        )
    }
    for sid, state in per_subject_state.items():
        pos = action_prob_layer.subject_id_to_pos[sid]
        np.testing.assert_allclose(by_subject_pos[pos], sum(state))


def test_build_inference_layer_precompute_mask_index_consolidates_and_matches_unpadded_sums():
    """
    Inference-side counterpart to
    test_compute_batched_algorithm_component_with_mask_matches_unpadded_per_subject_sums.
    Three subjects with staggered per-subject history lengths collapse from
    3 buckets to 1 when opted in, and a mask-aware inference_estimating_func
    reproduces each subject's own unpadded sum exactly.
    """
    subject_ids = np.array([1, 2, 3])
    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        {
            1: {
                1: (jnp.array([1.0]), 1.0),
                2: (jnp.array([1.0]), 1.0),
                3: (jnp.array([1.0]), 1.0),
            }
        },
        {1: {1: 0}, 2: {1: 0}, 3: {1: 0}},
        {1: {1: 1}, 2: {1: 1}, 3: {1: 1}},
        {},
        1,
        0,
    )

    per_subject_reward = {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0], 3: [6.0]}
    inference_func_args_by_subject_id = {
        sid: (jnp.array([0.0]), jnp.array(rewards))
        for sid, rewards in per_subject_reward.items()
    }

    unpadded_layer = build_inference_layer_precompute(
        inference_func_args_by_subject_id,
        -1,  # inference_func_args_action_prob_index
        {},
        action_prob_layer,
    )
    assert len(unpadded_layer.buckets) == 3

    padded_layer = build_inference_layer_precompute(
        inference_func_args_by_subject_id,
        -1,  # inference_func_args_action_prob_index
        {},
        action_prob_layer,
        inference_func_args_mask_index=2,
        inference_func_args_ragged_indices=(1,),
    )
    assert len(padded_layer.buckets) == 1

    def mask_aware_sum_estimating_func(theta, rewards, mask):
        return jnp.sum(mask * rewards).reshape(1)

    theta = jnp.array([0.0])
    weighted_component, _, bucket_outputs = compute_batched_inference_outputs(
        theta,
        1,  # theta_dim
        mask_aware_sum_estimating_func,
        0,  # inference_func_args_theta_index
        -1,  # inference_func_args_action_prob_index
        action_prob_layer,
        padded_layer,
        None,  # pi_beta_grid
        jnp.ones(3),  # inference_weight_products
        need_hessians=False,
    )

    assert len(bucket_outputs) == 1
    for sid, rewards in per_subject_reward.items():
        pos = action_prob_layer.subject_id_to_pos[sid]
        np.testing.assert_allclose(float(weighted_component[pos, 0]), sum(rewards))


def test_self_pad_ragged_args_and_build_mask_target_max_length_pads_further():
    """
    combine_updates_into_one_vmap's mechanism relies on target_max_length
    pushing every group's own padding past its local max, up to one shared
    global max -- confirm that in isolation before trusting the full
    build_update_layer_precompute integration below.
    """
    args_by_subject_id = {
        1: (np.array([1.0, 2.0]),),  # local max would be 2
    }
    result = self_pad_ragged_args_and_build_mask(
        args_by_subject_id, ragged_indices=(0,), mask_index=1, target_max_length=4
    )
    np.testing.assert_array_equal(result[1][0], [1.0, 2.0, 2.0, 2.0])
    np.testing.assert_array_equal(result[1][1], [1.0, 1.0, 0.0, 0.0])


def test_self_pad_ragged_args_and_build_mask_target_max_length_rejects_too_small():
    args_by_subject_id = {1: (np.array([1.0, 2.0, 3.0]),)}
    with pytest.raises(ValueError, match="target_max_length"):
        self_pad_ragged_args_and_build_mask(
            args_by_subject_id, ragged_indices=(0,), mask_index=1, target_max_length=2
        )


def _build_two_update_staggered_recruitment_layers(alg_update_func_args_mask_index=2):
    """
    Shared fixture for the combine_updates_into_one_vmap tests below: 3
    subjects, 2 updates. At update 1 (policy_num=2), only subjects 1 and 2
    are valid, with staggered history lengths (2 vs 3) -- exercising the
    global (cross-update) padding step. At update 2 (policy_num=3), all
    three subjects are valid, with a third, different set of staggered
    lengths (4, 5, 1) -- the global max across both updates is 5, wider than
    either update's own local max (3 and 5 respectively -- so update 1 alone
    would need re-padding from 3 up to 5). Subject 3's invalidity at update 1
    exercises update_fill_index's backward-fill (self-padded from its own
    real row at update 2).
    """
    subject_ids = np.array([1, 2, 3])
    beta_index_by_policy_num = {2: 0, 3: 1}
    initial_policy_num = 1
    action_prob_func_args_beta_index = 0
    times = [1, 2, 3, 4]

    action_prob_func_args_by_subject_id_by_decision_time = {
        t: {sid: (jnp.array([1.0]), 1.0) for sid in [1, 2, 3]} for t in times
    }
    action_by_decision_time_by_subject_id = {
        sid: {t: 0 for t in times} for sid in [1, 2, 3]
    }
    policy_num_by_decision_time_by_subject_id = {
        sid: {t: 1 for t in times} for sid in [1, 2, 3]
    }
    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )

    per_subject_state_u1 = {1: [1.0, 2.0], 2: [3.0, 4.0, 5.0]}
    per_subject_state_u2 = {
        1: [1.0, 2.0, 3.0, 4.0],
        2: [5.0, 6.0, 7.0, 8.0, 9.0],
        3: [10.0],
    }
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            sid: (jnp.array([0.0]), jnp.array(state))
            for sid, state in per_subject_state_u1.items()
        },
        3: {
            sid: (jnp.array([0.0]), jnp.array(state))
            for sid, state in per_subject_state_u2.items()
        },
    }

    common_kwargs = dict(
        subject_ids=subject_ids,
        update_func_args_by_by_subject_id_by_policy_num=update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num=beta_index_by_policy_num,
        alg_update_func_args_action_prob_times_index=-1,
        action_prob_layer=action_prob_layer,
        alg_update_func_args_mask_index=alg_update_func_args_mask_index,
        alg_update_func_args_ragged_indices=(1,),
    )
    uncombined_layer = build_update_layer_precompute(**common_kwargs)
    combined_layer = build_update_layer_precompute(
        **common_kwargs,
        combine_updates_into_one_vmap=True,
        alg_update_func_args_beta_index=0,
    )
    return (
        action_prob_layer,
        uncombined_layer,
        combined_layer,
        per_subject_state_u1,
        per_subject_state_u2,
    )


def _mask_aware_sum_estimating_func(beta, state, mask):
    return jnp.sum(mask * state).reshape(1)


def test_combine_updates_into_one_vmap_matches_uncombined_path_and_ground_truth():
    """
    End-to-end numeric equivalence for the combine_updates_into_one_vmap
    opt-in: with staggered per-subject history lengths AND staggered
    per-update validity (subject 3 only valid at the second update),
    compute_batched_algorithm_component's combined path must reproduce
    exactly what the existing (uncombined, per-update/per-bucket) path
    produces, and both must match each subject's own unpadded ground-truth
    sum -- proving the global padding + cross-update self-padding is
    mathematically a no-op on the answer, only on dispatch count.
    """
    (
        action_prob_layer,
        uncombined_layer,
        combined_layer,
        per_subject_state_u1,
        per_subject_state_u2,
    ) = _build_two_update_staggered_recruitment_layers()

    assert combined_layer.combined_arg_tensors is not None
    assert uncombined_layer.combined_arg_tensors is None

    betas = jnp.array([[0.0], [0.0]])  # (U=2, beta_dim=1)
    common_call_args = (
        betas,
        1,  # beta_dim
        _mask_aware_sum_estimating_func,
        0,  # alg_update_func_args_beta_index
        -1,  # alg_update_func_args_previous_betas_index
        -1,  # alg_update_func_args_action_prob_index
        action_prob_layer,
    )

    uncombined_component, _ = compute_batched_algorithm_component(
        *common_call_args, uncombined_layer, None, jnp.ones((3, 2))
    )
    combined_component, combined_bucket_outputs = compute_batched_algorithm_component(
        *common_call_args, combined_layer, None, jnp.ones((3, 2))
    )

    np.testing.assert_allclose(
        np.asarray(uncombined_component), np.asarray(combined_component), atol=1e-6
    )

    expected = np.zeros((3, 2))
    for sid, state in per_subject_state_u1.items():
        expected[action_prob_layer.subject_id_to_pos[sid], 0] = sum(state)
    for sid, state in per_subject_state_u2.items():
        expected[action_prob_layer.subject_id_to_pos[sid], 1] = sum(state)
    np.testing.assert_allclose(
        np.asarray(combined_component).reshape(3, 2), expected, atol=1e-6
    )

    # Subject 3 is invalid at update 1 -- its combined-path contribution
    # there must still be exactly zero (via the outer valid_update gate),
    # even though update_fill_index self-padded its row with real data from
    # update 2 to keep jax.vmap's input in-domain.
    subject_3_pos = action_prob_layer.subject_id_to_pos[3]
    assert float(combined_component[subject_3_pos, 0]) == 0.0

    # bucket_outputs must still be usable by
    # check_batched_algorithm_estimating_function_args_equivalent exactly as
    # today: one entry per non-empty (update, bucket) pair.
    total_nonempty_buckets = sum(
        1
        for buckets in combined_layer.buckets_by_update_index
        for b in buckets
        if b.subject_ids_in_order
    )
    assert len(combined_bucket_outputs) == total_nonempty_buckets


def test_combine_updates_into_one_vmap_requires_mask_index():
    subject_ids = np.array([1])
    with pytest.raises(ValueError, match="alg_update_func_args_mask_index"):
        build_update_layer_precompute(
            subject_ids,
            {1: {1: (jnp.array([0.0]), jnp.array([1.0]))}},
            {1: 0},
            -1,
            build_action_prob_layer_precompute(
                subject_ids,
                {1: {1: (jnp.array([1.0]), 1.0)}},
                {1: {1: 0}},
                {1: {1: 1}},
                {1: 0},
                0,
                0,
            ),
            combine_updates_into_one_vmap=True,
        )


def test_compute_batched_algorithm_component_combined_rejects_previous_betas():
    """
    The combined path's fixed-shape (N, U, beta_dim) tensor layout cannot
    express previous_betas' genuine variable-length-per-update slice (see
    _build_algorithm_bucket_overrides) -- must raise loudly rather than
    silently produce a wrong answer.
    """
    (
        action_prob_layer,
        _uncombined_layer,
        combined_layer,
        _u1,
        _u2,
    ) = _build_two_update_staggered_recruitment_layers()

    betas = jnp.array([[0.0], [0.0]])
    with pytest.raises(NotImplementedError, match="previous_betas"):
        compute_batched_algorithm_component(
            betas,
            1,
            _mask_aware_sum_estimating_func,
            0,  # alg_update_func_args_beta_index
            0,  # alg_update_func_args_previous_betas_index >= 0
            -1,
            action_prob_layer,
            combined_layer,
            None,
            jnp.ones((3, 2)),
        )


def test_combine_updates_into_one_vmap_passes_through_all_none_argument_position():
    """
    An argument position holding a literal None for every subject (an
    unused, override-only slot -- e.g. an unused previous_betas/action_prob
    position when the corresponding *_index is -1; see _stackable_positions'
    own docstring) must be passed straight through as None, exactly like the
    per-bucket path already does via _stackable_positions -- NOT stacked
    into a real (N, U, *shape) tensor, which would either crash (nothing
    real to self-pad an all-None position with) or silently build a garbage
    object-dtype array.
    """
    subject_ids = np.array([1, 2])
    beta_index_by_policy_num = {2: 0}
    initial_policy_num = 1
    action_prob_func_args_beta_index = 0
    times = [1, 2]

    action_prob_func_args_by_subject_id_by_decision_time = {
        t: {sid: (jnp.array([1.0]), 1.0) for sid in [1, 2]} for t in times
    }
    action_by_decision_time_by_subject_id = {
        sid: {t: 0 for t in times} for sid in [1, 2]
    }
    policy_num_by_decision_time_by_subject_id = {
        sid: {t: 1 for t in times} for sid in [1, 2]
    }
    action_prob_layer = build_action_prob_layer_precompute(
        subject_ids,
        action_prob_func_args_by_subject_id_by_decision_time,
        action_by_decision_time_by_subject_id,
        policy_num_by_decision_time_by_subject_id,
        beta_index_by_policy_num,
        initial_policy_num,
        action_prob_func_args_beta_index,
    )

    # args: (beta_placeholder, state, unused_none_slot) -- unused_none_slot
    # is None for every subject, e.g. an unused previous_betas position.
    per_subject_state = {1: [1.0, 2.0], 2: [3.0]}
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            sid: (jnp.array([0.0]), jnp.array(state), None)
            for sid, state in per_subject_state.items()
        }
    }

    combined_layer = build_update_layer_precompute(
        subject_ids,
        update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num,
        -1,  # alg_update_func_args_action_prob_times_index
        action_prob_layer,
        alg_update_func_args_mask_index=3,
        alg_update_func_args_ragged_indices=(1,),
        combine_updates_into_one_vmap=True,
        alg_update_func_args_beta_index=0,
    )
    # position 2 (the all-None slot) must not have been stacked.
    assert 2 not in combined_layer.combined_arg_positions

    def mask_aware_sum_with_unused_slot(beta, state, unused, mask):
        assert unused is None
        return jnp.sum(mask * state).reshape(1)

    betas = jnp.array([[0.0]])
    component, _ = compute_batched_algorithm_component(
        betas,
        1,  # beta_dim
        mask_aware_sum_with_unused_slot,
        0,  # alg_update_func_args_beta_index
        -1,  # alg_update_func_args_previous_betas_index
        -1,  # alg_update_func_args_action_prob_index
        action_prob_layer,
        combined_layer,
        None,
        jnp.ones((2, 1)),
    )
    for sid, state in per_subject_state.items():
        pos = action_prob_layer.subject_id_to_pos[sid]
        np.testing.assert_allclose(float(component[pos, 0]), sum(state))


def test_resolve_combine_updates_into_one_vmap_eligibility_matrix():
    """
    The pure auto-resolution helper: None (auto) enables combining exactly
    when eligible (mask_index >= 0 AND previous_betas_index < 0); an explicit
    True/False passes straight through regardless of eligibility.
    """
    # Auto: eligible.
    assert resolve_combine_updates_into_one_vmap(None, 2, -1) is True
    # Auto: no mask opt-in -> off (this is what every unmasked caller,
    # including the golden benchmark fixtures, resolves to).
    assert resolve_combine_updates_into_one_vmap(None, -1, -1) is False
    # Auto: previous_betas in use -> off.
    assert resolve_combine_updates_into_one_vmap(None, 2, 3) is False
    assert resolve_combine_updates_into_one_vmap(None, -1, 3) is False
    # Explicit values pass through, even when eligibility says otherwise.
    assert resolve_combine_updates_into_one_vmap(True, -1, 3) is True
    assert resolve_combine_updates_into_one_vmap(False, 2, -1) is False


def test_combine_auto_resolution_numerically_identical_to_explicit_true():
    """
    Constraint check for the auto default: a layer built from the
    auto-RESOLVED decision (None -> True on an eligible masked fixture) must
    produce numerically identical results to a layer built from an explicit
    combine_updates_into_one_vmap=True.
    """
    (
        action_prob_layer,
        _uncombined_layer,
        explicit_layer,
        _u1,
        _u2,
    ) = _build_two_update_staggered_recruitment_layers()

    resolved = resolve_combine_updates_into_one_vmap(
        None,
        alg_update_func_args_mask_index=2,
        alg_update_func_args_previous_betas_index=-1,
    )
    assert resolved is True

    # Rebuild the same fixture layer, but driving combine from the resolver
    # (and with combine_is_required=False, exactly as
    # get_avg_weighted_estimating_function_stacks_and_aux_values passes it
    # for an auto resolution).
    (
        _action_prob_layer_again,
        _uncombined_again,
        _explicit_again,
        per_subject_state_u1,
        per_subject_state_u2,
    ) = _build_two_update_staggered_recruitment_layers()

    betas = jnp.array([[0.0], [0.0]])
    common_call_args = (
        betas,
        1,
        _mask_aware_sum_estimating_func,
        0,
        -1,
        -1,
        action_prob_layer,
    )
    explicit_component, _ = compute_batched_algorithm_component(
        *common_call_args, explicit_layer, None, jnp.ones((3, 2))
    )

    subject_ids = np.array([1, 2, 3])
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            sid: (jnp.array([0.0]), jnp.array(state))
            for sid, state in per_subject_state_u1.items()
        },
        3: {
            sid: (jnp.array([0.0]), jnp.array(state))
            for sid, state in per_subject_state_u2.items()
        },
    }
    auto_layer = build_update_layer_precompute(
        subject_ids=subject_ids,
        update_func_args_by_by_subject_id_by_policy_num=update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num={2: 0, 3: 1},
        alg_update_func_args_action_prob_times_index=-1,
        action_prob_layer=action_prob_layer,
        alg_update_func_args_mask_index=2,
        alg_update_func_args_ragged_indices=(1,),
        combine_updates_into_one_vmap=resolved,
        alg_update_func_args_beta_index=0,
        combine_is_required=False,
    )
    assert auto_layer.combined_arg_tensors is not None
    auto_component, _ = compute_batched_algorithm_component(
        *common_call_args, auto_layer, None, jnp.ones((3, 2))
    )
    np.testing.assert_array_equal(
        np.asarray(auto_component), np.asarray(explicit_component)
    )


def _build_action_prob_layer_for_subjects(subject_ids, beta_index_by_policy_num):
    times = [1, 2]
    sids = subject_ids.tolist()
    return build_action_prob_layer_precompute(
        subject_ids,
        {t: {sid: (jnp.array([1.0]), 1.0) for sid in sids} for t in times},
        {sid: {t: 0 for t in times} for sid in sids},
        {sid: {t: 1 for t in times} for sid in sids},
        beta_index_by_policy_num,
        1,  # initial_policy_num
        0,  # action_prob_func_args_beta_index
    )


def _mask_aware_sum_with_extra(beta, state, extra, mask):
    return (jnp.sum(mask * state) + jnp.sum(extra)).reshape(1)


def test_combine_auto_mode_falls_back_on_within_update_shape_violation(caplog):
    """
    A NON-ragged, non-overridden argument position whose shape differs across
    subjects at the same update leaves multiple shape-buckets even after
    global padding -- a structural invariant combining requires, detectable
    only during build_update_layer_precompute's combined-block work. With
    combine_is_required=False (an AUTO-resolved combine), the build must fall
    back to the default per-update/per-bucket loop (combined_* left None),
    warn-logging the violation, and produce results identical to an ordinary
    uncombined masked layer. With combine_is_required=True (an explicit
    combine_updates_into_one_vmap=True), the same violation must still raise
    the loud ValueError it always has.
    """
    subject_ids = np.array([1, 2])
    beta_index_by_policy_num = {2: 0}
    action_prob_layer = _build_action_prob_layer_for_subjects(
        subject_ids, beta_index_by_policy_num
    )

    # args: (beta, ragged_state, extra, mask). extra's shape differs across
    # subjects at the same update: (2,) vs (3,).
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {
            1: (jnp.array([0.0]), jnp.array([1.0, 2.0]), jnp.array([10.0, 20.0])),
            2: (
                jnp.array([0.0]),
                jnp.array([3.0]),
                jnp.array([30.0, 40.0, 50.0]),
            ),
        }
    }
    common_kwargs = dict(
        subject_ids=subject_ids,
        update_func_args_by_by_subject_id_by_policy_num=update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num=beta_index_by_policy_num,
        alg_update_func_args_action_prob_times_index=-1,
        action_prob_layer=action_prob_layer,
        alg_update_func_args_mask_index=3,
        alg_update_func_args_ragged_indices=(1,),
        alg_update_func_args_beta_index=0,
    )

    # Explicit True: still the loud error, verbatim.
    with pytest.raises(ValueError, match="distinct argument shapes"):
        build_update_layer_precompute(
            **common_kwargs,
            combine_updates_into_one_vmap=True,
            combine_is_required=True,
        )

    # Auto-resolved True: warn-logged fallback, combined_* left None.
    with caplog.at_level(
        logging.WARNING,
        logger="lifejacket.batched_weighted_estimating_function_stack",
    ):
        fallback_layer = build_update_layer_precompute(
            **common_kwargs,
            combine_updates_into_one_vmap=True,
            combine_is_required=False,
        )
    assert fallback_layer.combined_arg_tensors is None
    assert fallback_layer.combined_arg_positions is None
    assert fallback_layer.combined_num_args is None
    assert fallback_layer.combined_action_prob_col_idx is None
    assert fallback_layer.update_fill_index is None
    assert any(
        "auto-enabled" in record.message and record.levelno == logging.WARNING
        for record in caplog.records
    )

    # The fallback layer's numbers are the default loop's numbers: identical
    # to a layer that never attempted combining at all.
    plain_kwargs = {
        k: v for k, v in common_kwargs.items() if k != "alg_update_func_args_beta_index"
    }
    plain_layer = build_update_layer_precompute(**plain_kwargs)
    betas = jnp.array([[0.0]])
    call_args = (
        betas,
        1,
        _mask_aware_sum_with_extra,
        0,
        -1,
        -1,
        action_prob_layer,
    )
    fallback_component, _ = compute_batched_algorithm_component(
        *call_args, fallback_layer, None, jnp.ones((2, 1))
    )
    plain_component, _ = compute_batched_algorithm_component(
        *call_args, plain_layer, None, jnp.ones((2, 1))
    )
    np.testing.assert_array_equal(
        np.asarray(fallback_component), np.asarray(plain_component)
    )
    # And both match the unpadded ground truth.
    expected = {
        1: 1.0 + 2.0 + 10.0 + 20.0,
        2: 3.0 + 30.0 + 40.0 + 50.0,
    }
    for sid, value in expected.items():
        pos = action_prob_layer.subject_id_to_pos[sid]
        assert float(fallback_component[pos, 0]) == pytest.approx(value)


def test_combine_auto_mode_falls_back_on_cross_update_shape_violation():
    """
    Same as above, for the OTHER mid-precompute invariant: a non-ragged
    argument position whose shape agrees within each update but differs
    ACROSS updates.
    """
    subject_ids = np.array([1])
    beta_index_by_policy_num = {2: 0, 3: 1}
    action_prob_layer = _build_action_prob_layer_for_subjects(
        subject_ids, beta_index_by_policy_num
    )

    # extra is (1,) at update 1 but (2,) at update 2 -- consistent within
    # each update, inconsistent across them.
    update_func_args_by_by_subject_id_by_policy_num = {
        2: {1: (jnp.array([0.0]), jnp.array([1.0]), jnp.array([10.0]))},
        3: {1: (jnp.array([0.0]), jnp.array([2.0, 3.0]), jnp.array([20.0, 30.0]))},
    }
    common_kwargs = dict(
        subject_ids=subject_ids,
        update_func_args_by_by_subject_id_by_policy_num=update_func_args_by_by_subject_id_by_policy_num,
        beta_index_by_policy_num=beta_index_by_policy_num,
        alg_update_func_args_action_prob_times_index=-1,
        action_prob_layer=action_prob_layer,
        alg_update_func_args_mask_index=3,
        alg_update_func_args_ragged_indices=(1,),
        alg_update_func_args_beta_index=0,
    )

    with pytest.raises(ValueError, match="identical shape across every"):
        build_update_layer_precompute(
            **common_kwargs,
            combine_updates_into_one_vmap=True,
            combine_is_required=True,
        )

    fallback_layer = build_update_layer_precompute(
        **common_kwargs,
        combine_updates_into_one_vmap=True,
        combine_is_required=False,
    )
    assert fallback_layer.combined_arg_tensors is None
    assert fallback_layer.update_fill_index is None
