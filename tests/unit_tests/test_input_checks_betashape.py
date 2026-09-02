"""
Coverage for the beta/previous-beta SHAPE and cross-subject AGREEMENT checks, plus the
action-prob-times wiring checks:

  require_beta_is_1D_array_in_alg_update_args
  require_previous_betas_is_2D_array_in_alg_update_args
  require_beta_is_1D_array_in_action_prob_args
  require_betas_match_in_alg_update_args_each_update
  require_previous_betas_match_in_alg_update_args_each_update
  require_betas_match_in_action_prob_func_args_each_policy
  require_action_prob_times_given_if_index_supplied
  require_action_prob_index_given_if_times_supplied
  require_valid_action_prob_times_given_if_index_supplied

Two invariants these checks exist to protect, and which the tests below pin down:

* SHAPE. beta is threaded in as a (beta_dim,) vector and previous betas as a
  (n_updates, beta_dim) matrix. A 2D beta or a 1D previous-betas matrix does not fail
  where it is supplied -- it fails much later as an inscrutable vmap/stacking shape
  error, or not at all (it silently broadcasts).
* AGREEMENT. beta is SHARED: across subjects at a given update in the algorithm update
  args, and across every cell in force under one POLICY in the action prob args. The
  threading code keeps one representative per policy (the first non-blank subject's, see
  helper_functions.collect_all_post_update_betas) and discards the rest, so any
  disagreement is a silently-dropped input, not an error. The action-prob side was keyed
  on the decision time until recently; the section on
  require_betas_match_in_action_prob_func_args_each_policy below records what the
  re-keying changed, in both directions.

None of these nine functions calls helper_functions.confirm_input_check_result, so none of
them can prompt on builtins.input; every test here is non-interactive by construction.
"""

import re

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

_SUBJECT_IDS_BETASHAPE = (0, 1)
_BETA_DIM_BETASHAPE = 3

# Argument-tuple layout used throughout this module:
#   (beta, previous_betas, action_probs, action_prob_times)
_BETA_INDEX_BETASHAPE = 0
_PREVIOUS_BETAS_INDEX_BETASHAPE = 1
_ACTION_PROB_INDEX_BETASHAPE = 2
_ACTION_PROB_TIMES_INDEX_BETASHAPE = 3

# A negative index means "this optional argument is absent". -1 is the conventional
# spelling, but nothing in the code privileges it, so the early-return tests also use a
# more-negative value.
_ABSENT_INDEX_BETASHAPE = -1

# The "blank"/not-applicable argument tuple that nearly every check skips with
# `if not args: continue`.
_BLANK_BETASHAPE = ()

# alg_update_func_args is keyed by the policy numbers an UPDATE produced, so the initial
# policy (1 in the real fixtures, NOT 0) is absent and the keys start at 2. They are ints
# even though analysis_df's policy_num column is float64.
_POLICY_NUMS_BETASHAPE = (2, 3)

# The calendar times present in the analysis_df built below: 0..4 inclusive.
_CALENDAR_TIMES_BETASHAPE = (0, 1, 2, 3, 4)

# The calendar times of the per-policy beta fixture: 0..3, so that each of its two policies
# is in force at exactly two of them.
_POLICY_BETA_TIMES_BETASHAPE = (0, 1, 2, 3)


def _beta_betashape(seed_value):
    """A distinct, deterministic 1D beta per seed value, exactly representable in float32."""
    return np.arange(_BETA_DIM_BETASHAPE, dtype=np.float64) + float(seed_value)


def _previous_betas_betashape(seed_value):
    """A deterministic 2D (2, beta_dim) previous-betas matrix per seed value."""
    return np.stack([_beta_betashape(seed_value), _beta_betashape(seed_value) + 10.0])


def _build_alg_update_args_betashape(
    *,
    beta_2d=False,
    previous_betas_1d=False,
    beta_value_mismatch=False,
    beta_shape_mismatch=False,
    previous_betas_value_mismatch=False,
    beta_dtype_mismatch=False,
    beta_nan=False,
    blank_subject_id=None,
    all_blank=False,
    empty_subject_dicts=False,
):
    """
    A valid alg_update_func_args: two update-produced policies (2 and 3), two subjects,
    every subject at a policy carrying the SAME 1D beta and the SAME 2D previous-betas
    matrix for that policy, with betas differing BETWEEN policies.

    Each keyword introduces exactly one defect, always in subject 1's tuple at policy 3 so
    that the per-policy scoping of the agreement checks is exercised (policy 2 stays clean):

    beta_2d                      -- beta stored as a (1, beta_dim) matrix instead of a vector.
    previous_betas_1d            -- previous betas stored as a flat vector instead of a matrix.
    beta_value_mismatch          -- a different beta of the SAME shape (cross-subject disagreement).
    beta_shape_mismatch          -- a beta of a different LENGTH (which numpy would broadcast).
    previous_betas_value_mismatch-- a different previous-betas matrix of the same shape.
    beta_dtype_mismatch          -- the same beta VALUES stored as float32 rather than float64.
    beta_nan                     -- every subject's beta at policy 3 carries a NaN component.
    blank_subject_id             -- that subject's tuple is blank () at every policy.
    all_blank                    -- every tuple is blank ().
    empty_subject_dicts          -- each policy maps to an empty per-subject dict.
    """
    if empty_subject_dicts:
        return {policy_num: {} for policy_num in _POLICY_NUMS_BETASHAPE}

    alg_update_func_args = {}
    for policy_num in _POLICY_NUMS_BETASHAPE:
        args_by_subject_id = {}
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            if all_blank or subject_id == blank_subject_id:
                args_by_subject_id[subject_id] = _BLANK_BETASHAPE
                continue

            beta = _beta_betashape(policy_num)
            previous_betas = _previous_betas_betashape(policy_num)
            if beta_nan and policy_num == 3:
                beta = beta.copy()
                beta[0] = np.nan
            if policy_num == 3 and subject_id == 1:
                if beta_2d:
                    beta = beta.reshape(1, -1)
                if beta_value_mismatch:
                    beta = _beta_betashape(policy_num) + 100.0
                if beta_shape_mismatch:
                    beta = np.arange(_BETA_DIM_BETASHAPE + 1, dtype=np.float64) + float(
                        policy_num
                    )
                if beta_dtype_mismatch:
                    beta = beta.astype(np.float32)
                if previous_betas_1d:
                    previous_betas = previous_betas.reshape(-1)
                if previous_betas_value_mismatch:
                    previous_betas = _previous_betas_betashape(policy_num) + 100.0

            args_by_subject_id[subject_id] = (
                beta,
                previous_betas,
                np.array([0.5, 0.5]),
                np.array([0, 1]),
            )
        alg_update_func_args[policy_num] = args_by_subject_id
    return alg_update_func_args


def _build_action_prob_args_betashape(
    *,
    beta_2d=False,
    beta_value_mismatch=False,
    beta_shape_mismatch=False,
    blank_subject_id=None,
    all_blank=False,
):
    """
    A valid action_prob_func_args: {decision_time: {subject_id: (beta, features)}} over the
    calendar times 0..4, every subject at a decision time sharing that time's beta, with
    betas differing BETWEEN decision times.

    Defects are introduced in subject 1's tuple at decision time 2 only, so the per-decision
    scoping of the agreement check is exercised.
    """
    action_prob_func_args = {}
    for decision_time in _CALENDAR_TIMES_BETASHAPE:
        args_by_subject_id = {}
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            if all_blank or subject_id == blank_subject_id:
                args_by_subject_id[subject_id] = _BLANK_BETASHAPE
                continue

            beta = _beta_betashape(decision_time)
            if decision_time == 2 and subject_id == 1:
                if beta_2d:
                    beta = beta.reshape(-1, 1)
                if beta_value_mismatch:
                    beta = _beta_betashape(decision_time) + 100.0
                if beta_shape_mismatch:
                    beta = np.arange(_BETA_DIM_BETASHAPE + 1, dtype=np.float64)

            args_by_subject_id[subject_id] = (
                beta,
                np.full(_BETA_DIM_BETASHAPE, 1.0 + subject_id),
            )
        action_prob_func_args[decision_time] = args_by_subject_id
    return action_prob_func_args


def _build_times_study_betashape(
    *,
    times_by_policy=None,
    calendar_t_dtype="int64",
    all_blank=False,
    blank_subject_id=None,
    subject_1_out_of_study_at_the_extremes=False,
):
    """
    An (analysis_df, alg_update_func_args) pair for the action-prob-times check.

    analysis_df: calendar times 0..4 for both subjects, float64 policy_num whose initial
    value is 1.0 (NOT 0), int64 0/1 in_study, float64 action holding NaN out of study.
    Only the calendar time column matters to the check under test -- it takes the study's
    time window from that column's min()/max() over EVERY row, in-study or not.

    alg_update_func_args carries, per policy, strictly increasing supplied times inside
    that window: policy 2 -> [0, 1], policy 3 -> [2, 3].

    times_by_policy overrides the supplied times per policy number, which is how the
    non-strictly-increasing and out-of-window branches are reached.
    """
    default_times_by_policy = {2: [0, 1], 3: [2, 3]}
    times_by_policy = {**default_times_by_policy, **(times_by_policy or {})}

    rows = []
    for calendar_t in _CALENDAR_TIMES_BETASHAPE:
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            is_in_study = not (
                subject_1_out_of_study_at_the_extremes
                and subject_id == 1
                and calendar_t
                in (min(_CALENDAR_TIMES_BETASHAPE), max(_CALENDAR_TIMES_BETASHAPE))
            )
            rows.append(
                {
                    "calendar_t": calendar_t,
                    "user_id": subject_id,
                    "in_study": 1 if is_in_study else 0,
                    "policy_num": 1.0 + float(calendar_t // 2),
                    "action": float(calendar_t % 2) if is_in_study else np.nan,
                }
            )
    analysis_df = pd.DataFrame(rows)
    analysis_df["calendar_t"] = analysis_df["calendar_t"].astype(calendar_t_dtype)
    analysis_df["in_study"] = analysis_df["in_study"].astype("int64")

    alg_update_func_args = {}
    for policy_num in _POLICY_NUMS_BETASHAPE:
        args_by_subject_id = {}
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            if all_blank or subject_id == blank_subject_id:
                args_by_subject_id[subject_id] = _BLANK_BETASHAPE
                continue
            times = np.asarray(times_by_policy[policy_num])
            args_by_subject_id[subject_id] = (
                _beta_betashape(policy_num),
                _previous_betas_betashape(policy_num),
                np.full(len(times), 0.5),
                times,
            )
        alg_update_func_args[policy_num] = args_by_subject_id
    return analysis_df, alg_update_func_args


def _build_policy_betas_study_betashape(
    *,
    policy_num_overrides=None,
    beta_overrides=None,
    inactive_cells=(),
    blank_cells=(),
):
    """
    An (analysis_df, action_prob_func_args) pair for the per-POLICY beta agreement check.

    analysis_df: the calendar times 0..3 for both subjects, every cell active, float64
    policy_num -- policy 2.0 in force at times 0 and 1, policy 3.0 at times 2 and 3. One
    policy spanning SEVERAL decision times is the whole point of the fixture: it is the
    configuration the per-policy invariant constrains and the decision-time check it
    replaced could not see at all.

    action_prob_func_args: {decision_time: {subject_id: (beta, features)}} over the same
    times, each cell recording the beta of the policy in force there, so every cell under
    one policy agrees and the betas differ BETWEEN policies.

    policy_num_overrides -- {(calendar_t, subject_id): policy_num} rewriting analysis_df's
        policy column for those cells; a negative number spells a fallback policy.
    beta_overrides       -- {(calendar_t, subject_id): beta} replacing the beta recorded in
        the args for those cells, which is how a disagreement is introduced.
    inactive_cells       -- cells whose analysis_df row carries in_study 0 (and a NaN
        action, as a real out-of-study row does).
    blank_cells          -- cells whose args tuple is the blank () instead of
        (beta, features); a beta override for such a cell never reaches the args.
    """
    policy_num_overrides = policy_num_overrides or {}
    beta_overrides = beta_overrides or {}

    policy_num_by_cell = {}
    for calendar_t in _POLICY_BETA_TIMES_BETASHAPE:
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            cell = (calendar_t, subject_id)
            policy_num_by_cell[cell] = policy_num_overrides.get(
                cell, 2.0 + float(calendar_t // 2)
            )

    rows = []
    for calendar_t in _POLICY_BETA_TIMES_BETASHAPE:
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            cell = (calendar_t, subject_id)
            is_active = cell not in inactive_cells
            rows.append(
                {
                    "calendar_t": calendar_t,
                    "user_id": subject_id,
                    "in_study": 1 if is_active else 0,
                    "policy_num": policy_num_by_cell[cell],
                    "action": float(calendar_t % 2) if is_active else np.nan,
                }
            )
    analysis_df = pd.DataFrame(rows)
    analysis_df["calendar_t"] = analysis_df["calendar_t"].astype("int64")
    analysis_df["in_study"] = analysis_df["in_study"].astype("int64")
    analysis_df["policy_num"] = analysis_df["policy_num"].astype("float64")

    action_prob_func_args = {}
    for calendar_t in _POLICY_BETA_TIMES_BETASHAPE:
        args_by_subject_id = {}
        for subject_id in _SUBJECT_IDS_BETASHAPE:
            cell = (calendar_t, subject_id)
            if cell in blank_cells:
                args_by_subject_id[subject_id] = _BLANK_BETASHAPE
                continue
            beta = beta_overrides.get(cell, _beta_betashape(policy_num_by_cell[cell]))
            args_by_subject_id[subject_id] = (
                beta,
                np.full(_BETA_DIM_BETASHAPE, 1.0 + subject_id),
            )
        action_prob_func_args[calendar_t] = args_by_subject_id
    return analysis_df, action_prob_func_args


def _check_policy_betas_betashape(analysis_df, action_prob_func_args):
    """
    Call the per-policy beta agreement check with this module's column names, pinning the
    positional order of its signature in one place:
    (analysis_df, active, calendar_t, subject_id, policy_num, args, beta_index).
    """
    input_checks.require_betas_match_in_action_prob_func_args_each_policy(
        analysis_df,
        "in_study",
        "calendar_t",
        "user_id",
        "policy_num",
        action_prob_func_args,
        _BETA_INDEX_BETASHAPE,
    )


def _recorded_policy_num_betashape(analysis_df, calendar_t, subject_id):
    """
    The policy number as the check will see it: read back out of analysis_df, so the value
    that lands in the assertion message carries the column's numpy dtype rather than the
    Python float the fixture was built from.
    """
    matching_rows = analysis_df.loc[
        (analysis_df["calendar_t"] == calendar_t)
        & (analysis_df["user_id"] == subject_id),
        "policy_num",
    ]
    return matching_rows.to_numpy()[0]


def _expected_policy_beta_message_betashape(mismatches):
    """
    The check's full assertion text for a given list of
    (policy_num, first cell, disagreeing cell) triples.
    """
    shown = sorted(mismatches, key=str)[:5]
    return (
        "The action prob args record different betas for cells sharing a policy number, "
        f"at {len(mismatches)} cell(s). A policy is one parameter vector, and the "
        "estimator substitutes a single shared beta per policy, so every cell in force "
        "under a given policy must record the same beta. Up to 5 shown as (policy_num, "
        f"first cell, disagreeing cell): {shown}. Please see the contract for details."
    )


### require_beta_is_1D_array_in_alg_update_args


def test_require_beta_is_1D_array_in_alg_update_args_passes_on_1d_betas_betashape():
    """The happy path: a (beta_dim,) vector at the beta position for every subject."""
    input_checks.require_beta_is_1D_array_in_alg_update_args(
        _build_alg_update_args_betashape(), _BETA_INDEX_BETASHAPE
    )


def test_require_beta_is_1D_array_in_alg_update_args_rejects_2d_beta_betashape():
    """A (1, beta_dim) beta broadcasts silently downstream instead of failing here."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Beta is not a 1D array in the algorithm update function args."
        ),
    ):
        input_checks.require_beta_is_1D_array_in_alg_update_args(
            _build_alg_update_args_betashape(beta_2d=True), _BETA_INDEX_BETASHAPE
        )


def test_require_beta_is_1D_array_in_alg_update_args_skips_blank_arg_tuples_betashape():
    """
    Blank () tuples must be skipped, not indexed. Every tuple is blank here, so without
    `if not args: continue` this would die with IndexError on ()[0] rather than passing.
    """
    input_checks.require_beta_is_1D_array_in_alg_update_args(
        _build_alg_update_args_betashape(all_blank=True), _BETA_INDEX_BETASHAPE
    )


def test_require_beta_is_1D_array_in_alg_update_args_accepts_jnp_betas_betashape():
    """jnp arrays (what a real study supplies) expose .ndim the same way numpy does."""
    alg_update_func_args = {
        policy_num: {
            subject_id: (jnp.array([1.0, 2.0, 3.0]), jnp.ones((2, 3)))
            for subject_id in _SUBJECT_IDS_BETASHAPE
        }
        for policy_num in _POLICY_NUMS_BETASHAPE
    }
    input_checks.require_beta_is_1D_array_in_alg_update_args(
        alg_update_func_args, _BETA_INDEX_BETASHAPE
    )


def test_require_beta_is_1D_array_in_alg_update_args_passes_on_empty_args_betashape():
    """An empty args dict, and policies with empty per-subject dicts, are vacuous passes."""
    input_checks.require_beta_is_1D_array_in_alg_update_args({}, _BETA_INDEX_BETASHAPE)
    input_checks.require_beta_is_1D_array_in_alg_update_args(
        _build_alg_update_args_betashape(empty_subject_dicts=True),
        _BETA_INDEX_BETASHAPE,
    )


### require_previous_betas_is_2D_array_in_alg_update_args


def test_require_previous_betas_is_2D_array_in_alg_update_args_passes_on_2d_betashape():
    """The happy path: an (n_updates, beta_dim) matrix at the previous-betas position."""
    input_checks.require_previous_betas_is_2D_array_in_alg_update_args(
        _build_alg_update_args_betashape(), _PREVIOUS_BETAS_INDEX_BETASHAPE
    )


def test_require_previous_betas_is_2D_array_in_alg_update_args_rejects_1d_betashape():
    """A flattened previous-betas matrix loses the per-update rows the update function indexes."""
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Previous betas is not a 2D array in the algorithm update function args."
        ),
    ):
        input_checks.require_previous_betas_is_2D_array_in_alg_update_args(
            _build_alg_update_args_betashape(previous_betas_1d=True),
            _PREVIOUS_BETAS_INDEX_BETASHAPE,
        )


def test_require_previous_betas_is_2D_array_in_alg_update_args_returns_early_when_absent_betashape():
    """
    Previous betas are optional: a negative index means the argument is absent, so the
    check must return before touching the tuples. The defect that WOULD be flagged (a 1D
    previous-betas matrix at position 1) is present here, and must not be reported.
    """
    alg_update_func_args = _build_alg_update_args_betashape(previous_betas_1d=True)
    input_checks.require_previous_betas_is_2D_array_in_alg_update_args(
        alg_update_func_args, _ABSENT_INDEX_BETASHAPE
    )
    # Nothing privileges -1 as the "absent" spelling; any negative index returns early.
    input_checks.require_previous_betas_is_2D_array_in_alg_update_args(
        alg_update_func_args, -7
    )


def test_require_previous_betas_is_2D_array_in_alg_update_args_skips_blank_arg_tuples_betashape():
    """Blank () tuples are skipped; without the skip, indexing () raises IndexError."""
    input_checks.require_previous_betas_is_2D_array_in_alg_update_args(
        _build_alg_update_args_betashape(all_blank=True),
        _PREVIOUS_BETAS_INDEX_BETASHAPE,
    )


### require_beta_is_1D_array_in_action_prob_args


def test_require_beta_is_1D_array_in_action_prob_args_passes_on_1d_betas_betashape():
    """The happy path over {decision_time: {subject_id: args}}."""
    input_checks.require_beta_is_1D_array_in_action_prob_args(
        _build_action_prob_args_betashape(), _BETA_INDEX_BETASHAPE
    )


def test_require_beta_is_1D_array_in_action_prob_args_rejects_2d_beta_betashape():
    """
    A (beta_dim, 1) column-vector beta. jnp.dot would still produce a number from it, so
    nothing downstream necessarily complains -- this check is what rejects it, and its
    message must name the ACTION PROBABILITY args, not the algorithm update args.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Beta is not a 1D array in the action probability function args."
        ),
    ):
        input_checks.require_beta_is_1D_array_in_action_prob_args(
            _build_action_prob_args_betashape(beta_2d=True), _BETA_INDEX_BETASHAPE
        )


def test_require_beta_is_1D_array_in_action_prob_args_skips_blank_arg_tuples_betashape():
    """
    Out-of-study cells carry blank () tuples (that is exactly what
    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times
    enforces), so this check must skip them rather than index them.
    """
    input_checks.require_beta_is_1D_array_in_action_prob_args(
        _build_action_prob_args_betashape(all_blank=True), _BETA_INDEX_BETASHAPE
    )
    input_checks.require_beta_is_1D_array_in_action_prob_args({}, _BETA_INDEX_BETASHAPE)


### require_betas_match_in_alg_update_args_each_update


def test_require_betas_match_in_alg_update_args_each_update_passes_betashape():
    """
    Agreement is required WITHIN a policy only: the fixture's betas differ between policy
    2 and policy 3, which must not be reported.
    """
    input_checks.require_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(), _BETA_INDEX_BETASHAPE
    )


def test_require_betas_match_in_alg_update_args_each_update_rejects_value_mismatch_betashape():
    """
    Subject 1's beta at policy 3 differs in value. Only the first non-blank subject's beta
    survives threading, so this disagreement would otherwise be silently discarded. The
    message must name the offending policy (3), not the clean one (2).
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Betas do not match across subjects in the algorithm update function args for "
            "policy number 3. Please see the contract for details."
        ),
    ):
        input_checks.require_betas_match_in_alg_update_args_each_update(
            _build_alg_update_args_betashape(beta_value_mismatch=True),
            _BETA_INDEX_BETASHAPE,
        )


def test_require_betas_match_in_alg_update_args_each_update_rejects_shape_mismatch_betashape():
    """
    A beta of a different LENGTH must be caught too. np.array_equal compares shapes first
    (unlike ==, which would broadcast a (1,) beta against a (3,) one and find them equal).
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Betas do not match across subjects in the algorithm update function args for "
            "policy number 3."
        ),
    ):
        input_checks.require_betas_match_in_alg_update_args_each_update(
            _build_alg_update_args_betashape(beta_shape_mismatch=True),
            _BETA_INDEX_BETASHAPE,
        )


def test_require_betas_match_in_alg_update_args_each_update_ignores_float32_vs_float64_betashape():
    """
    The same beta VALUES stored at different precisions are equal to np.array_equal, so a
    float32/float64 storage difference (jnp defaults to float32, numpy to float64) is not
    reported as a cross-subject disagreement.
    """
    input_checks.require_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(beta_dtype_mismatch=True),
        _BETA_INDEX_BETASHAPE,
    )


def test_require_betas_match_in_alg_update_args_each_update_ignores_int_vs_float_dtype_betashape():
    """An integer-dtype beta and a float-dtype beta with identical values also compare equal."""
    alg_update_func_args = {
        2: {
            0: (np.array([1, 2, 3]), _previous_betas_betashape(2)),
            1: (np.array([1.0, 2.0, 3.0]), _previous_betas_betashape(2)),
        }
    }
    input_checks.require_betas_match_in_alg_update_args_each_update(
        alg_update_func_args, _BETA_INDEX_BETASHAPE
    )


def test_require_betas_match_in_alg_update_args_each_update_skips_blank_arg_tuples_betashape():
    """
    The FIRST subject's tuple is blank, so the representative beta has to come from the
    second one. With only one non-blank subject per policy there is nothing to compare
    against and the check passes -- but it must get there without indexing ().
    """
    input_checks.require_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(blank_subject_id=0), _BETA_INDEX_BETASHAPE
    )
    input_checks.require_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(all_blank=True), _BETA_INDEX_BETASHAPE
    )


def test_require_betas_match_in_alg_update_args_each_update_reports_nan_betas_as_mismatched_betashape():
    """
    Documents a sharp edge rather than an intended feature: np.array_equal defaults to
    equal_nan=False, so two BIT-IDENTICAL betas that carry a NaN component are reported as
    a cross-subject mismatch. The check still fails loudly on a blown-up algorithm, but
    with a message that misattributes the cause.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Betas do not match across subjects in the algorithm update function args for "
            "policy number 3."
        ),
    ):
        input_checks.require_betas_match_in_alg_update_args_each_update(
            _build_alg_update_args_betashape(beta_nan=True), _BETA_INDEX_BETASHAPE
        )


### require_previous_betas_match_in_alg_update_args_each_update


def test_require_previous_betas_match_in_alg_update_args_each_update_passes_betashape():
    """The happy path: one shared previous-betas matrix per policy."""
    input_checks.require_previous_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(), _PREVIOUS_BETAS_INDEX_BETASHAPE
    )


def test_require_previous_betas_match_in_alg_update_args_each_update_rejects_mismatch_betashape():
    """
    Previous betas are shared across subjects exactly like beta, so a per-subject
    previous-betas matrix is a wiring error. The message must say PREVIOUS betas.
    """
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Previous betas do not match across subjects in the algorithm update function "
            "args for policy number 3. Please see the contract for details."
        ),
    ):
        input_checks.require_previous_betas_match_in_alg_update_args_each_update(
            _build_alg_update_args_betashape(previous_betas_value_mismatch=True),
            _PREVIOUS_BETAS_INDEX_BETASHAPE,
        )


def test_require_previous_betas_match_in_alg_update_args_each_update_returns_early_when_absent_betashape():
    """
    A negative previous-betas index means the argument is absent, so a genuine
    previous-betas disagreement at position 1 must NOT be reported -- position 1 holds
    something else entirely in that configuration.
    """
    alg_update_func_args = _build_alg_update_args_betashape(
        previous_betas_value_mismatch=True
    )
    input_checks.require_previous_betas_match_in_alg_update_args_each_update(
        alg_update_func_args, _ABSENT_INDEX_BETASHAPE
    )
    input_checks.require_previous_betas_match_in_alg_update_args_each_update(
        alg_update_func_args, -4
    )


def test_require_previous_betas_match_in_alg_update_args_each_update_skips_blank_arg_tuples_betashape():
    """Blank () tuples are skipped when choosing and comparing the representative matrix."""
    input_checks.require_previous_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(blank_subject_id=0),
        _PREVIOUS_BETAS_INDEX_BETASHAPE,
    )
    input_checks.require_previous_betas_match_in_alg_update_args_each_update(
        _build_alg_update_args_betashape(all_blank=True),
        _PREVIOUS_BETAS_INDEX_BETASHAPE,
    )


### require_betas_match_in_action_prob_func_args_each_policy
#
# This check replaced require_betas_match_in_action_prob_func_args_each_decision, which
# required every subject at one calendar time to record the same beta. The invariant is now
# keyed on the POLICY NUMBER in force at each cell, which is why the check takes analysis_df
# and four column names: the args dict is keyed by decision time and records no policy at
# all, so the policy in force has to be looked up per (decision time, subject) cell.
#
# The re-keying moved the boundary in BOTH directions, and the tests below pin each side:
# stricter across time (one policy spans several decision times and must present one beta at
# all of them, which the decision-time check never compared), and more permissive within a
# time (two policies at one decision time are a supported configuration, which the
# decision-time check rejected). Fallback (negative) policy cells are skipped outright.


def test_require_betas_match_in_action_prob_func_args_each_policy_passes_betashape():
    """
    The happy path: policy 2.0 is in force at times 0 and 1 and records one beta at both,
    policy 3.0 is in force at times 2 and 3 and records a different beta at both.

    Agreement is required WITHIN a policy only, so the beta CHANGING from time 1 to time 2
    -- which is what an algorithm update does -- must not be reported.
    """
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape()
    _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_rejects_value_mismatch_within_a_policy_betashape():
    """
    INVERTED from the decision-time check: subject 0's recorded beta at time 3 differs from
    the beta the same policy (3.0) recorded at time 2. The old
    require_betas_match_in_action_prob_func_args_each_decision compared subjects within one
    calendar time only, so it passed this input silently -- yet the estimator substitutes a
    single shared beta per policy, so one of the two recorded betas is being discarded.

    The message must name the policy number and both cells, not a decision time.
    """
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        beta_overrides={(3, 0): _beta_betashape(3) + 100.0}
    )
    policy_num = _recorded_policy_num_betashape(analysis_df, 2, 0)
    expected_message = _expected_policy_beta_message_betashape(
        [(policy_num, (2, 0), (3, 0))]
    )
    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_rejects_shape_mismatch_within_a_policy_betashape():
    """
    A beta of a different LENGTH under the same policy is a mismatch, not a broadcast: the
    check compares .shape before np.array_equal, so a (beta_dim + 1,) beta is reported
    rather than compared elementwise against the (beta_dim,) reference.
    """
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        beta_overrides={(3, 1): np.arange(_BETA_DIM_BETASHAPE + 1, dtype=np.float64)}
    )
    policy_num = _recorded_policy_num_betashape(analysis_df, 2, 0)
    expected_message = _expected_policy_beta_message_betashape(
        [(policy_num, (2, 0), (3, 1))]
    )
    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_allows_two_policies_at_one_decision_time_betashape():
    """
    INVERTED from the decision-time check: subject 1's update lags, so at time 2 subject 0
    is on policy 3.0 while subject 1 is still on policy 2.0, and the two subjects therefore
    record DIFFERENT betas at the same decision time.

    require_betas_match_in_action_prob_func_args_each_decision rejected exactly this input.
    It is a supported configuration -- fallback policies interleave with the current one,
    and verify_analysis_df_summary_satisfactory reports the number of such times as an
    ordinary statistic -- so it must now pass. Each policy still records one beta.
    """
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        policy_num_overrides={(2, 1): 2.0}
    )
    beta_subject_0 = action_prob_func_args[2][0][_BETA_INDEX_BETASHAPE]
    beta_subject_1 = action_prob_func_args[2][1][_BETA_INDEX_BETASHAPE]
    assert not np.array_equal(beta_subject_0, beta_subject_1)
    _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_skips_fallback_policy_cells_betashape():
    """
    Both subjects fall back at time 1 and record different betas there. A negative policy
    number means "the algorithm was bypassed" rather than naming a parameter vector, and
    the estimator never substitutes a beta for such a cell, so fallback cells are skipped
    outright rather than grouped together under -1.0.

    The second half is the control: the very same pair of betas under a NON-negative policy
    number is reported, which is what makes the first half a real skip rather than a
    vacuous pass.
    """
    divergent_betas = {(1, 0): _beta_betashape(50), (1, 1): _beta_betashape(60)}
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        policy_num_overrides={(1, 0): -1.0, (1, 1): -1.0},
        beta_overrides=divergent_betas,
    )
    _check_policy_betas_betashape(analysis_df, action_prob_func_args)

    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        beta_overrides=divergent_betas
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The action prob args record different betas for cells sharing a policy "
            "number, at 2 cell(s)."
        ),
    ):
        _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_skips_blank_arg_tuples_betashape():
    """
    Subjects out of study at a decision time carry blank () tuples; they must neither be
    indexed (()[0] is an IndexError) nor become a policy's representative beta.
    """
    every_cell = tuple(
        (calendar_t, subject_id)
        for calendar_t in _POLICY_BETA_TIMES_BETASHAPE
        for subject_id in _SUBJECT_IDS_BETASHAPE
    )
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        blank_cells=every_cell
    )
    _check_policy_betas_betashape(analysis_df, action_prob_func_args)

    analysis_df, _ = _build_policy_betas_study_betashape()
    _check_policy_betas_betashape(analysis_df, {})


def test_require_betas_match_in_action_prob_func_args_each_policy_takes_reference_beta_from_first_nonblank_cell_betashape():
    """
    Policy 3.0's first cell in iteration order, (2, 0), is blank, so the reference beta has
    to come from the next non-blank cell, (2, 1) -- and a later disagreement is still
    reported against it. A blank tuple skips one cell, it does not stop the policy from
    being checked.
    """
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        blank_cells=((2, 0),),
        beta_overrides={(3, 0): _beta_betashape(3) + 100.0},
    )
    policy_num = _recorded_policy_num_betashape(analysis_df, 2, 1)
    expected_message = _expected_policy_beta_message_betashape(
        [(policy_num, (2, 1), (3, 0))]
    )
    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _check_policy_betas_betashape(analysis_df, action_prob_func_args)


def test_require_betas_match_in_action_prob_func_args_each_policy_skips_cells_not_marked_active_betashape():
    """
    The policy lookup is built from ACTIVE rows only, so a cell analysis_df does not mark
    active has no policy in force and is skipped -- exactly like a cell with no analysis_df
    row at all. Its recorded beta diverges here and must not be reported.

    The second half is the control: the same divergent beta at the same cell IS reported
    once that cell is active, so the skip is doing the work rather than the betas agreeing.
    """
    beta_overrides = {(3, 0): _beta_betashape(3) + 100.0}
    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        inactive_cells=((3, 0),), beta_overrides=beta_overrides
    )
    _check_policy_betas_betashape(analysis_df, action_prob_func_args)

    analysis_df, action_prob_func_args = _build_policy_betas_study_betashape(
        beta_overrides=beta_overrides
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "different betas for cells sharing a policy number, at 1 cell(s)"
        ),
    ):
        _check_policy_betas_betashape(analysis_df, action_prob_func_args)


### require_action_prob_times_given_if_index_supplied
#
# Bare asserts with NO message, so these tests match on the exception type only.


def test_require_action_prob_times_given_if_index_supplied_passes_when_both_supplied_betashape():
    """Action probabilities at one position and their times at a DIFFERENT one: valid."""
    input_checks.require_action_prob_times_given_if_index_supplied(
        _ACTION_PROB_INDEX_BETASHAPE, _ACTION_PROB_TIMES_INDEX_BETASHAPE
    )


def test_require_action_prob_times_given_if_index_supplied_passes_when_index_absent_betashape():
    """
    When action probabilities are not supplied the times index is unconstrained: neither
    "also absent" nor "supplied anyway" is this check's business.
    """
    input_checks.require_action_prob_times_given_if_index_supplied(
        _ABSENT_INDEX_BETASHAPE, _ABSENT_INDEX_BETASHAPE
    )
    input_checks.require_action_prob_times_given_if_index_supplied(
        -3, _ACTION_PROB_TIMES_INDEX_BETASHAPE
    )


def test_require_action_prob_times_given_if_index_supplied_rejects_missing_times_betashape():
    """
    Action probabilities without their times cannot be checked against analysis_df at all
    (require_action_prob_args_in_alg_update_func_correspond_to_analysis_df indexes the
    times to find the rows to compare), so this combination is rejected up front.
    """
    with pytest.raises(AssertionError):
        input_checks.require_action_prob_times_given_if_index_supplied(
            _ACTION_PROB_INDEX_BETASHAPE, _ABSENT_INDEX_BETASHAPE
        )


def test_require_action_prob_times_given_if_index_supplied_rejects_colliding_indices_betashape():
    """
    Zero, the index most likely to be left at its default, must not serve as both the
    action-probability and the times position: one argument cannot be both.
    """
    with pytest.raises(AssertionError):
        input_checks.require_action_prob_times_given_if_index_supplied(0, 0)


### require_action_prob_index_given_if_times_supplied


def test_require_action_prob_index_given_if_times_supplied_passes_when_both_supplied_betashape():
    """The mirror-image happy path of the previous function."""
    input_checks.require_action_prob_index_given_if_times_supplied(
        _ACTION_PROB_INDEX_BETASHAPE, _ACTION_PROB_TIMES_INDEX_BETASHAPE
    )


def test_require_action_prob_index_given_if_times_supplied_passes_when_times_absent_betashape():
    """With no times supplied this check is a no-op, whatever the action-prob index is."""
    input_checks.require_action_prob_index_given_if_times_supplied(
        _ABSENT_INDEX_BETASHAPE, _ABSENT_INDEX_BETASHAPE
    )
    input_checks.require_action_prob_index_given_if_times_supplied(
        _ACTION_PROB_INDEX_BETASHAPE, -2
    )


def test_require_action_prob_index_given_if_times_supplied_rejects_missing_index_betashape():
    """
    Times without the action probabilities they timestamp: the times argument would never
    be read, so the two must be supplied together.
    """
    with pytest.raises(AssertionError):
        input_checks.require_action_prob_index_given_if_times_supplied(
            _ABSENT_INDEX_BETASHAPE, _ACTION_PROB_TIMES_INDEX_BETASHAPE
        )


def test_require_action_prob_index_given_if_times_supplied_rejects_colliding_indices_betashape():
    """The collision branch is duplicated in both directions; both must reject it."""
    with pytest.raises(AssertionError):
        input_checks.require_action_prob_index_given_if_times_supplied(
            _ACTION_PROB_TIMES_INDEX_BETASHAPE, _ACTION_PROB_TIMES_INDEX_BETASHAPE
        )


### require_valid_action_prob_times_given_if_index_supplied


def test_require_valid_action_prob_times_given_if_index_supplied_passes_betashape():
    """Strictly increasing times, all inside the study's calendar-time window."""
    analysis_df, alg_update_func_args = _build_times_study_betashape()
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_returns_early_when_absent_betashape():
    """
    A negative times index means no times were supplied, so the check must return before
    reading position 3 -- even though the times sitting there are both out of order and
    out of the study window.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={2: [9, 4, 4], 3: [-11, -12]}
    )
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df, "calendar_t", alg_update_func_args, _ABSENT_INDEX_BETASHAPE
    )
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df, "calendar_t", alg_update_func_args, -6
    )


def test_require_valid_action_prob_times_given_if_index_supplied_rejects_repeated_times_betashape():
    """
    STRICTLY increasing, not merely non-decreasing: a repeated time means the same
    decision was counted twice in the update's action-probability vector.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={3: [2, 2, 3]}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Non-strictly-increasing times were given for action probabilities in the "
            "algorithm update function args for subject 0 and policy 3."
        ),
    ):
        input_checks.require_valid_action_prob_times_given_if_index_supplied(
            analysis_df,
            "calendar_t",
            alg_update_func_args,
            _ACTION_PROB_TIMES_INDEX_BETASHAPE,
        )


def test_require_valid_action_prob_times_given_if_index_supplied_rejects_out_of_order_times_betashape():
    """
    Descending times are the same branch but the likelier real error (an unsorted append).
    Order matters because the window check below only inspects the first and last entries.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={3: [3, 2]}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Non-strictly-increasing times were given for action probabilities in the "
            "algorithm update function args"
        ),
    ):
        input_checks.require_valid_action_prob_times_given_if_index_supplied(
            analysis_df,
            "calendar_t",
            alg_update_func_args,
            _ACTION_PROB_TIMES_INDEX_BETASHAPE,
        )


def test_require_valid_action_prob_times_given_if_index_supplied_rejects_times_before_the_study_betashape():
    """
    A time earlier than any row in analysis_df -- typically an off-by-one or a
    wrong-origin clock. The message must quote the study's actual window (0 and 4).
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={2: [-1, 0, 1]}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Times not present in the study were given for action probabilities in the "
            "algorithm update function args. The min and max times in the analysis "
            "DataFrame are 0 and 4"
        ),
    ):
        input_checks.require_valid_action_prob_times_given_if_index_supplied(
            analysis_df,
            "calendar_t",
            alg_update_func_args,
            _ACTION_PROB_TIMES_INDEX_BETASHAPE,
        )


def test_require_valid_action_prob_times_given_if_index_supplied_rejects_times_after_the_study_betashape():
    """
    The upper end of the same window check: a time past the last row of analysis_df would
    have no action probability to correspond to.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={3: [2, 3, 5]}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Times not present in the study were given for action probabilities"
        ),
    ):
        input_checks.require_valid_action_prob_times_given_if_index_supplied(
            analysis_df,
            "calendar_t",
            alg_update_func_args,
            _ACTION_PROB_TIMES_INDEX_BETASHAPE,
        )


def test_require_valid_action_prob_times_given_if_index_supplied_accepts_boundary_times_betashape():
    """
    The window is inclusive at both ends: the study's very first and very last calendar
    times are valid supplied times.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={3: [0, 4]}
    )
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_accepts_single_time_betashape():
    """
    A one-element times array: the strictly-increasing loop is empty (range(1, 1)) and the
    first and last entries are the same value, so a subject with a single decision under a
    policy passes.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={2: [1], 3: [2]}
    )
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_accepts_float_times_betashape():
    """
    float64 times against an int64 calendar-time column. The real fixtures store times as
    float arrays, so the comparison must be numeric, not dtype-sensitive.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={2: [0.0, 1.0], 3: [2.0, 3.0]}
    )
    assert analysis_df["calendar_t"].dtype == np.dtype("int64")
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_accepts_float_calendar_column_betashape():
    """
    The mirror case: a float64 calendar-time column (min/max come back as float64) with
    integer supplied times still compares cleanly.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        calendar_t_dtype="float64"
    )
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_accepts_jnp_times_betashape():
    """
    jnp arrays at the times position, as a real study supplies. len() and element
    comparison work, and the jnp bool each comparison yields is truthy-evaluable.
    """
    analysis_df, _ = _build_times_study_betashape()
    alg_update_func_args = {
        2: {
            subject_id: (
                jnp.array([1.0, 2.0, 3.0]),
                jnp.ones((2, 3)),
                jnp.array([0.5, 0.5]),
                jnp.array([0.0, 1.0]),
            )
            for subject_id in _SUBJECT_IDS_BETASHAPE
        }
    }
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_skips_blank_arg_tuples_betashape():
    """
    Blank () tuples are skipped: an all-blank args dict passes (indexing () would raise
    IndexError), and a blank first subject does not stop the second from being validated.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(all_blank=True)
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )

    analysis_df, alg_update_func_args = _build_times_study_betashape(
        blank_subject_id=0, times_by_policy={3: [3, 2]}
    )
    with pytest.raises(
        AssertionError,
        match=re.escape("for subject 1 and policy 3"),
    ):
        input_checks.require_valid_action_prob_times_given_if_index_supplied(
            analysis_df,
            "calendar_t",
            alg_update_func_args,
            _ACTION_PROB_TIMES_INDEX_BETASHAPE,
        )


def test_require_valid_action_prob_times_given_if_index_supplied_window_ignores_the_active_column_betashape():
    """
    Documents the SCOPE of "present in the study": the window is the calendar-time
    column's min/max over EVERY row, in-study or not. Subject 1 is out of study at times
    0 and 4 here, yet times [0, 4] supplied for subject 1 pass -- the check does not
    consult the in-study column or any per-subject window.
    """
    analysis_df, alg_update_func_args = _build_times_study_betashape(
        times_by_policy={3: [0, 4]}, subject_1_out_of_study_at_the_extremes=True
    )
    assert analysis_df.loc[
        (analysis_df["user_id"] == 1) & (analysis_df["calendar_t"] == 0), "in_study"
    ].tolist() == [0]
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df,
        "calendar_t",
        alg_update_func_args,
        _ACTION_PROB_TIMES_INDEX_BETASHAPE,
    )


def test_require_valid_action_prob_times_given_if_index_supplied_passes_on_empty_args_betashape():
    """An empty args dict has no times to validate; the check is a vacuous pass."""
    analysis_df, _ = _build_times_study_betashape()
    input_checks.require_valid_action_prob_times_given_if_index_supplied(
        analysis_df, "calendar_t", {}, _ACTION_PROB_TIMES_INDEX_BETASHAPE
    )
