"""
Coverage for require_recorded_action_prob_betas_match_update_betas_for_their_policy.

This check is the action-probability function's original-vs-threaded equivalence check.
arg_threading_helpers.thread_action_prob_func_args discards the beta recorded in
action_prob_func_args and substitutes
all_post_update_betas[beta_index_by_policy_num[policy_num]], where all_post_update_betas comes
from helper_functions.collect_all_post_update_betas (which harvests alg_update_func_args, taking
the FIRST NON-BLANK subject's beta per policy) and beta_index_by_policy_num comes from
helper_functions.construct_beta_index_by_policy_num_map (which excludes fallback/negative
policies and the initial policy, the minimum non-negative active policy number). So the invariant
under test -- recorded action-prob beta == its policy's update beta, for non-initial,
non-fallback, active, non-blank cells only -- is exactly the set of substitutions the threading
code performs.
"""

import re

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

_SUBJECT_IDS_BETAPOLICY = (0, 1)
_BETA_DIM_BETAPOLICY = 4
_ACTION_PROB_BETA_INDEX_BETAPOLICY = 1
_ALG_UPDATE_BETA_INDEX_BETAPOLICY = 0
# The initial policy number is 1, NOT 0 -- matching the real fixtures, where policy_num is
# float64 running 1.0..7.0 and alg_update_func_args is keyed by the ints 2..7.
_POLICY_NUM_BY_DECISION_TIME_BETAPOLICY = {0: 1.0, 1: 2.0, 2: 3.0}

# Sentinel: a "blank"/not-applicable argument tuple, which nearly every check skips.
_BLANK_BETAPOLICY = ()


def _beta_betapolicy(policy_num):
    """A distinct, deterministic float64 beta per policy number.

    Deliberately not exactly representable in binary, so that the float32-vs-float64
    storage-tolerance tests exercise a genuine representation difference.
    """
    return (np.arange(_BETA_DIM_BETAPOLICY, dtype=np.float64) + 1.0) * 0.1 + float(
        policy_num
    )


def _features_betapolicy(decision_time, subject_id):
    return np.full(
        _BETA_DIM_BETAPOLICY, 1.0 + decision_time + 0.5 * subject_id, dtype=np.float64
    )


def _build_study_betapolicy(
    *,
    policy_num_by_decision_time=None,
    policy_num_dtype=np.float64,
    recorded_beta_overrides=None,
    update_beta_overrides=None,
    inactive_cells=(),
    drop_update_policy_nums=(),
    extra_update_policy_nums=(),
    recorded_betas_as_float32=False,
):
    """Build a small consistent study, optionally with one specific defect introduced.

    Default shape: subjects 0 and 1, decision times 0/1/2, active policies 1.0/2.0/3.0 (so the
    initial policy number is 1, not 0), updates recorded for policies 2 and 3 keyed by int.
    action_prob_func_args tuples are (features, beta) -- beta at index 1 -- while
    alg_update_func_args tuples are (beta, previous_betas) -- beta at index 0 -- so the two
    supplied beta indices are distinct and cannot be silently swapped.

    Keyword flags:
      policy_num_by_decision_time: remap decision times onto policy numbers (e.g. to make the
        initial policy 0, or to make a decision time use a negative fallback policy).
      policy_num_dtype: dtype of the analysis_df policy_num column.
      recorded_beta_overrides: {(decision_time, subject_id): beta or ()} -- replace what
        action_prob_func_args records for that cell, or blank the whole tuple.
      update_beta_overrides: {(policy_num_key, subject_id): beta or ()} -- replace what
        alg_update_func_args records, or blank that subject's whole tuple.
      inactive_cells: [(decision_time, subject_id)] marked in_study=0 (action NaN).
      drop_update_policy_nums: policy numbers removed from alg_update_func_args entirely.
      extra_update_policy_nums: policy numbers ADDED to alg_update_func_args that the threading
        code never reads (the initial policy, or a fallback policy), each carrying a beta that
        deliberately matches nothing. Needed to make the initial-policy and fallback skips
        load-bearing: without an entry present, those cells would be skipped by the
        "policy absent from alg_update_func_args" rule instead, and the test would be vacuous.
      recorded_betas_as_float32: store every recorded action-prob beta as a jnp float32 array.
    """
    policy_num_by_decision_time = (
        _POLICY_NUM_BY_DECISION_TIME_BETAPOLICY
        if policy_num_by_decision_time is None
        else policy_num_by_decision_time
    )
    recorded_beta_overrides = recorded_beta_overrides or {}
    update_beta_overrides = update_beta_overrides or {}
    inactive_cells = set(inactive_cells)

    rows = []
    action_prob_func_args = {}
    for decision_time, policy_num in policy_num_by_decision_time.items():
        action_prob_func_args[decision_time] = {}
        for subject_id in _SUBJECT_IDS_BETAPOLICY:
            is_active = (decision_time, subject_id) not in inactive_cells
            rows.append(
                {
                    "calendar_t": decision_time,
                    "user_id": subject_id,
                    "in_study": 1 if is_active else 0,
                    "policy_num": policy_num,
                    "action": 1.0 if is_active else np.nan,
                }
            )
            recorded_beta = recorded_beta_overrides.get(
                (decision_time, subject_id), _beta_betapolicy(policy_num)
            )
            if recorded_beta is _BLANK_BETAPOLICY:
                action_prob_func_args[decision_time][subject_id] = ()
                continue
            if recorded_betas_as_float32:
                recorded_beta = jnp.asarray(recorded_beta, dtype=jnp.float32)
            action_prob_func_args[decision_time][subject_id] = (
                _features_betapolicy(decision_time, subject_id),
                recorded_beta,
            )

    analysis_df = pd.DataFrame(rows)
    analysis_df["policy_num"] = analysis_df["policy_num"].astype(policy_num_dtype)
    analysis_df["in_study"] = analysis_df["in_study"].astype(np.int64)

    active_nonnegative_policy_nums = sorted(
        {
            policy_num
            for decision_time, policy_num in policy_num_by_decision_time.items()
            for subject_id in _SUBJECT_IDS_BETAPOLICY
            if policy_num >= 0 and (decision_time, subject_id) not in inactive_cells
        }
    )
    # Every non-initial, non-fallback policy gets an update recorded for it, keyed by int the
    # way the real alg_update_func_args is (ints 2..7 against a float64 policy_num column).
    update_policy_nums = [
        int(policy_num) for policy_num in active_nonnegative_policy_nums[1:]
    ]
    alg_update_func_args = {}
    for policy_num_key in update_policy_nums:
        if policy_num_key in drop_update_policy_nums:
            continue
        alg_update_func_args[policy_num_key] = {}
        for subject_id in _SUBJECT_IDS_BETAPOLICY:
            update_beta = update_beta_overrides.get(
                (policy_num_key, subject_id), _beta_betapolicy(policy_num_key)
            )
            if update_beta is _BLANK_BETAPOLICY:
                alg_update_func_args[policy_num_key][subject_id] = ()
                continue
            # previous_betas is not read by the check under test; it is present only so the
            # update tuples have the realistic (beta, previous_betas) shape.
            lower_policy_nums = [
                p for p in active_nonnegative_policy_nums if p < policy_num_key
            ]
            previous_betas = (
                np.stack([_beta_betapolicy(p) for p in lower_policy_nums])
                if lower_policy_nums
                else np.zeros((0, _BETA_DIM_BETAPOLICY), dtype=np.float64)
            )
            alg_update_func_args[policy_num_key][subject_id] = (
                update_beta,
                previous_betas,
            )

    for policy_num_key in extra_update_policy_nums:
        unmatchable_beta = _beta_betapolicy(policy_num_key) + 100.0
        alg_update_func_args[policy_num_key] = {
            subject_id: (
                unmatchable_beta,
                np.zeros((0, _BETA_DIM_BETAPOLICY), dtype=np.float64),
            )
            for subject_id in _SUBJECT_IDS_BETAPOLICY
        }

    return analysis_df, action_prob_func_args, alg_update_func_args


def _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args):
    """Invoke the check under test with this module's fixture column names/indices."""
    input_checks.require_recorded_action_prob_betas_match_update_betas_for_their_policy(
        analysis_df,
        "in_study",
        "calendar_t",
        "user_id",
        "policy_num",
        action_prob_func_args,
        _ACTION_PROB_BETA_INDEX_BETAPOLICY,
        alg_update_func_args,
        _ALG_UPDATE_BETA_INDEX_BETAPOLICY,
    )


def test_recorded_action_prob_betas_match_update_betas_passes_on_consistent_study_betapolicy():
    """Baseline: every recorded action-prob beta equals its policy's update beta, so the
    threading substitution is a no-op and the check must pass."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy()

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_mismatch_names_the_offending_cell_betapolicy():
    """One cell's recorded beta disagrees with its policy's update beta. The estimator would
    differentiate that decision's action probability at a beta the recorded probability did not
    come from, so the check must raise and name the (decision_time, subject_id, policy_num)
    cell plus a max |difference|."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(1, 0): _beta_betapolicy(2.0) + 0.5}
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("(1, 0, 2.0, 'max |difference| 0.5')"),
    ):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_mismatch_reports_cell_count_betapolicy():
    """Both subjects at one decision time disagree: the message must report 2 cells, not 1,
    so a reader can tell a single stray cell from a whole decision time being wrong."""
    wrong_beta = _beta_betapolicy(3.0) - 2.0
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(2, 0): wrong_beta, (2, 1): wrong_beta}
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("at 2 (decision_time, subject_id) cell(s)"),
    ):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_mismatch_reports_max_across_components_betapolicy():
    """The reported difference is the MAX over beta components, not the first or the last one:
    a beta perturbed by 0.25 in one slot and 0.5 in another reports 0.5."""
    perturbed = _beta_betapolicy(2.0) + np.array([0.0, 0.25, 0.5, 0.0])
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(1, 1): perturbed}
    )

    with pytest.raises(AssertionError, match=re.escape("max |difference| 0.5")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_cells_on_nonzero_initial_policy_betapolicy():
    """The initial policy number here is 1, not 0. Its beta was produced by no update and
    thread_action_prob_func_args passes the recorded tuple through untouched, so a wildly
    different recorded beta on the initial policy is not this check's business."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        extra_update_policy_nums=(1,),
        recorded_beta_overrides={
            (0, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (0, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        },
    )
    # The initial policy is 1.0, is present in analysis_df's active rows, AND has an entry in
    # alg_update_func_args carrying a beta that matches nothing -- so only the
    # "policy_num == initial_policy_num" rule can keep this study clean. (Drop that rule and
    # this fixture reports two cells.)
    assert 1.0 in set(analysis_df.loc[analysis_df["in_study"] == 1, "policy_num"])
    assert 1 in alg_update_func_args

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_initial_policy_when_it_is_zero_betapolicy():
    """The initial policy number is derived as the minimum non-negative active policy number
    rather than hardcoded, so a 0-based study skips policy 0 and still compares policies 1/2."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        policy_num_by_decision_time={0: 0.0, 1: 1.0, 2: 2.0},
        extra_update_policy_nums=(0,),
        recorded_beta_overrides={
            (0, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (0, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        },
    )
    assert sorted(alg_update_func_args.keys()) == [0, 1, 2]

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_still_caught_when_initial_policy_is_zero_betapolicy():
    """Companion to the previous test: skipping the 0 initial policy must not disable the
    comparison for the policies that updates did produce."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        policy_num_by_decision_time={0: 0.0, 1: 1.0, 2: 2.0},
        recorded_beta_overrides={(2, 0): _beta_betapolicy(2.0) + 1.0},
    )

    with pytest.raises(AssertionError, match=re.escape("(2, 0, 2.0,")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_fallback_policy_cells_betapolicy():
    """Fallback (negative) policies have no update-produced beta to compare against by design
    and are expected to carry blank args, so a non-blank mismatched recorded beta on one is
    skipped here rather than double-reported."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        policy_num_by_decision_time={0: 1.0, 1: 2.0, 2: -1.0},
        extra_update_policy_nums=(-1,),
        recorded_beta_overrides={
            (2, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (2, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        },
    )
    # -1 deliberately HAS an entry in alg_update_func_args with an unmatchable beta, so the
    # "policy_num < 0" rule is the only thing keeping this study clean -- not the separate
    # "policy absent from alg_update_func_args" rule.
    assert -1 in alg_update_func_args

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_blank_action_prob_arg_tuples_betapolicy():
    """A blank () action-prob tuple at a cell analysis_df marks ACTIVE on an updated policy is
    require_action_probabilities_in_analysis_df_can_be_reconstructed' finding; this check must
    skip it, because indexing () for a beta raises IndexError. That combination is what makes
    the `if not args: continue` guard load-bearing rather than shadowed by the inactive-cell
    rule."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(2, 1): _BLANK_BETAPOLICY},
    )
    assert action_prob_func_args[2][1] == ()
    # The cell is active and its policy IS in alg_update_func_args, so no other skip applies.
    assert (
        analysis_df.loc[
            (analysis_df["calendar_t"] == 2) & (analysis_df["user_id"] == 1), "in_study"
        ].item()
        == 1
    )
    assert 3 in alg_update_func_args

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_passes_with_blank_args_at_out_of_study_cells_betapolicy():
    """The realistic legal shape of a blank tuple: blank args exactly where analysis_df marks
    the subject out of study (action NaN). Both skip rules apply and the check passes."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(2, 1): _BLANK_BETAPOLICY},
        inactive_cells=[(2, 1)],
    )
    assert action_prob_func_args[2][1] == ()
    assert np.isnan(
        analysis_df.loc[
            (analysis_df["calendar_t"] == 2) & (analysis_df["user_id"] == 1), "action"
        ].item()
    )

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_policy_absent_from_alg_update_args_betapolicy():
    """A policy missing from alg_update_func_args entirely is
    require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args'
    finding; this check must skip it instead of raising a KeyError or a confusing message."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        drop_update_policy_nums=(3,),
        recorded_beta_overrides={
            (2, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (2, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        },
    )
    assert 3 not in alg_update_func_args

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_skips_cells_analysis_df_does_not_mark_active_betapolicy():
    """A non-blank tuple at a cell analysis_df marks inactive belongs to
    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times; here it has no
    policy in force, so it is skipped rather than reported under a misleading policy number."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        inactive_cells=[(1, 1)],
        recorded_beta_overrides={(1, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0)},
    )
    assert action_prob_func_args[1][1] != ()

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_catches_shape_mismatch_instead_of_broadcasting_betapolicy():
    """A recorded beta of shape (1,) against a (4,) update beta would BROADCAST under
    np.array_equal/np.allclose semantics if values happened to line up; shapes are therefore
    compared first and reported as a shape disagreement."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(1, 0): np.array([2.1])}
    )

    with pytest.raises(AssertionError, match=re.escape("shape (1,) vs (4,)")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_catches_broadcastable_equal_valued_shape_mismatch_betapolicy():
    """The sharpest version of the shape rule: a recorded (1,) beta whose single value equals
    every component of a constant (4,) update beta IS elementwise-equal after broadcasting, and
    must still be reported."""
    constant_beta = np.full(_BETA_DIM_BETAPOLICY, 7.0)
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        update_beta_overrides={(2, 0): constant_beta, (2, 1): constant_beta},
        recorded_beta_overrides={(1, 0): np.array([7.0]), (1, 1): constant_beta},
    )
    # Sanity: numpy really would call these equal if shapes were ignored.
    assert np.array_equal(np.array([7.0]) == constant_beta, np.ones(4, dtype=bool))

    with pytest.raises(AssertionError, match=re.escape("shape (1,) vs (4,)")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_tolerates_float32_vs_float64_storage_betapolicy():
    """jnp arrays default to float32 while alg_update_func_args here holds numpy float64. The
    SAME recorded value stored at the two precisions must pass -- that is a storage artifact,
    not a wiring error."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_betas_as_float32=True
    )
    # The tolerance is genuinely load-bearing here: the two dtypes are not bit-identical.
    assert (
        action_prob_func_args[1][0][_ACTION_PROB_BETA_INDEX_BETAPOLICY].dtype
        == jnp.float32
    )
    assert not np.array_equal(
        np.asarray(action_prob_func_args[1][0][_ACTION_PROB_BETA_INDEX_BETAPOLICY]),
        alg_update_func_args[2][0][_ALG_UPDATE_BETA_INDEX_BETAPOLICY],
    )

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_float32_storage_still_catches_real_difference_betapolicy():
    """The float32 tolerance absorbs storage precision only: a genuinely different beta, stored
    as float32, must still be reported."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_betas_as_float32=True,
        recorded_beta_overrides={(1, 0): _beta_betapolicy(2.0) + 1e-3},
    )

    with pytest.raises(AssertionError, match=re.escape("(1, 0, 2.0,")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_representative_update_beta_is_first_non_blank_subject_betapolicy():
    """collect_all_post_update_betas takes the first NON-BLANK subject's beta per policy, so a
    policy whose first subject has blank update args resolves to the next subject's beta -- and
    a recorded beta equal to that one must pass."""
    alternate_beta = _beta_betapolicy(2.0) + 5.0
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        update_beta_overrides={(2, 0): _BLANK_BETAPOLICY, (2, 1): alternate_beta},
        recorded_beta_overrides={(1, 0): alternate_beta, (1, 1): alternate_beta},
    )
    assert alg_update_func_args[2][0] == ()

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_blank_first_subject_does_not_excuse_mismatch_betapolicy():
    """Companion to the previous test: with subject 0's update args blank, the representative
    is subject 1's beta, so a recorded beta matching neither is still reported. This pins that
    the blank tuple is skipped over rather than making the whole policy unverifiable."""
    alternate_beta = _beta_betapolicy(2.0) + 5.0
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        update_beta_overrides={(2, 0): _BLANK_BETAPOLICY, (2, 1): alternate_beta},
    )

    with pytest.raises(AssertionError, match=re.escape("max |difference| 5")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_returns_early_when_every_active_row_is_fallback_betapolicy():
    """With no non-negative active policy number there is no initial policy to derive, so the
    check returns rather than crashing on an empty .min()."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={
            (1, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (1, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        }
    )
    analysis_df["policy_num"] = -1.0
    assert alg_update_func_args, (
        "alg_update_func_args must be non-empty to be a real pin"
    )

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_returns_early_on_empty_analysis_df_betapolicy():
    """An analysis_df with no rows at all must return quietly rather than raise from the
    initial-policy derivation."""
    _, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={
            (1, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (1, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        }
    )
    empty_df = pd.DataFrame(
        {
            "calendar_t": pd.Series(dtype="int64"),
            "user_id": pd.Series(dtype="int64"),
            "in_study": pd.Series(dtype="int64"),
            "policy_num": pd.Series(dtype="float64"),
            "action": pd.Series(dtype="float64"),
        }
    )

    _call_betapolicy(empty_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_handles_int_policy_num_column_dtype_betapolicy():
    """policy_num is float64 in the real fixtures but alg_update_func_args is keyed by ints; an
    int64 policy_num column must resolve to the same update betas (and still catch a mismatch)
    rather than silently missing every policy through a dtype-mismatched dict lookup."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        policy_num_dtype=np.int64,
        recorded_beta_overrides={(2, 1): _beta_betapolicy(3.0) + 0.75},
    )
    assert analysis_df["policy_num"].dtype == np.int64

    with pytest.raises(AssertionError, match=re.escape("max |difference| 0.75")):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_passes_with_int_policy_num_column_dtype_betapolicy():
    """Companion to the previous test: the int64 policy_num column also passes when everything
    agrees, so the dtype crossing is not merely making every comparison fail."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        policy_num_dtype=np.int64
    )

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_empty_action_prob_func_args_passes_betapolicy():
    """No recorded action-prob args at all means nothing to compare; the check must not
    manufacture a failure from the absent side."""
    analysis_df, _, alg_update_func_args = _build_study_betapolicy()

    _call_betapolicy(analysis_df, {}, alg_update_func_args)


def test_recorded_action_prob_betas_all_blank_update_args_for_a_policy_is_skipped_betapolicy():
    """Documenting the boundary: when EVERY subject's update args for a policy are blank, that
    policy has no representative beta, so this check skips it and reports nothing -- even though
    the recorded beta disagrees with the rest of the study."""
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        update_beta_overrides={
            (2, 0): _BLANK_BETAPOLICY,
            (2, 1): _BLANK_BETAPOLICY,
        },
        recorded_beta_overrides={
            (1, 0): np.full(_BETA_DIM_BETAPOLICY, -999.0),
            (1, 1): np.full(_BETA_DIM_BETAPOLICY, -999.0),
        },
    )
    assert alg_update_func_args[2] == {0: (), 1: ()}

    _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)


def test_recorded_action_prob_betas_nan_recorded_beta_is_reported_betapolicy():
    """A NaN recorded beta is neither array_equal nor allclose to a real one, so it must be
    reported (with a nan difference) rather than slipping through a comparison that quietly
    treats NaN as a match."""
    nan_beta = _beta_betapolicy(2.0).copy()
    nan_beta[0] = np.nan
    analysis_df, action_prob_func_args, alg_update_func_args = _build_study_betapolicy(
        recorded_beta_overrides={(1, 0): nan_beta}
    )

    with pytest.raises(
        AssertionError, match=re.escape("(1, 0, 2.0, 'max |difference|")
    ):
        _call_betapolicy(analysis_df, action_prob_func_args, alg_update_func_args)
