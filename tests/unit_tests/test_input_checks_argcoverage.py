"""
First-wave input checks that police the coverage of the two argument dictionaries:
that alg_update_func_args and action_prob_func_args carry an entry for every
subject at every update / decision time, that blank (empty-tuple) entries line up
exactly with the out-of-study cells of the analysis DataFrame, and that the action
probabilities carried inside the update args agree with the ones recorded in the
analysis DataFrame.

Fixture shape mirrors tests/benchmarks/fixtures/small: policy_num is float64 and
the INITIAL policy number is 1.0 (not 0.0), the in-study column is int64 0/1, the
action column is float64 and holds NaN on out-of-study rows, and
alg_update_func_args is keyed by int policy numbers starting at 2 (the initial
policy has no update args).
"""

import collections
import re

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp

from lifejacket import input_checks

_ACTIVE_COL_ARGCOVERAGE = "in_study"
_ACTION_COL_ARGCOVERAGE = "action"
_POLICY_NUM_COL_ARGCOVERAGE = "policy_num"
_CALENDAR_T_COL_ARGCOVERAGE = "calendar_t"
_SUBJECT_ID_COL_ARGCOVERAGE = "subject_id"
_ACTION_PROB_COL_ARGCOVERAGE = "action_prob"

_SUBJECT_IDS_ARGCOVERAGE = (1, 2)
_DECISION_TIMES_ARGCOVERAGE = (0, 1, 2, 3, 4, 5)
_POLICY_NUM_BY_TIME_ARGCOVERAGE = {0: 1.0, 1: 1.0, 2: 2.0, 3: 2.0, 4: 3.0, 5: 3.0}
# Policy 1 is the initial policy, so only policies 2 and 3 have update args. Each
# update carries the action probabilities of every decision time it learned from.
_UPDATE_TIMES_BY_POLICY_ARGCOVERAGE = {2: (0, 1), 3: (0, 1, 2, 3)}
# Subject 2 drops out for the final decision time only, so every time carried in
# the update args is still an in-study time for both subjects.
_INACTIVE_CELL_ARGCOVERAGE = (5, 2)
_BETA_ARGCOVERAGE = jnp.array([0.5, -0.25])
# Arg tuple layout used throughout: (beta, action_probs, action_prob_times).
_ACTION_PROB_INDEX_ARGCOVERAGE = 1
_ACTION_PROB_TIMES_INDEX_ARGCOVERAGE = 2
_ABSENT_INDEX_ARGCOVERAGE = -1


def _recorded_action_prob_argcoverage(decision_time, subject_id):
    """Deterministic, in-(0, 1) action probability for one (time, subject) cell."""
    return 0.3 + 0.05 * decision_time + 0.1 * (subject_id - 1)


def _build_study_argcoverage(
    *,
    all_subjects_always_active=False,
    drop_subject_from_update_args=False,
    extra_subject_in_update_args=False,
    blank_update_args_for_subject=False,
    corrupt_update_arg_action_prob=False,
    perturb_update_arg_action_prob_within_tolerance=False,
    float_action_prob_times=False,
    drop_subject_from_action_prob_args=False,
    extra_subject_in_action_prob_args=False,
    drop_decision_time_from_action_prob_args=False,
    extra_decision_time_in_action_prob_args=False,
    blank_action_prob_args_at_active_cell=False,
    nonblank_action_prob_args_at_inactive_cell=False,
):
    """
    Build a small but realistically shaped study: 2 subjects, 6 decision times,
    3 policies (initial policy 1.0 plus updates at 2 and 3). Subject 2 is out of
    study at the final decision time, so its action is NaN there, its action
    probability is NaN there, and its action_prob_func_args entry there is the
    blank tuple (). With no keyword flags every check in this family passes.

    Each flag introduces exactly one defect:
      all_subjects_always_active            -- nobody ever leaves the study, so
                                               there are no blank args at all
      drop_subject_from_update_args         -- policy 3 has no entry for subject 2
      extra_subject_in_update_args          -- policy 3 has an entry for a subject
                                               that is not in the analysis DataFrame
      blank_update_args_for_subject         -- policy 3's entry for subject 2 is ()
      corrupt_update_arg_action_prob        -- policy 3's probabilities for subject 1
                                               are off by 1e-3 (outside rtol/atol)
      perturb_..._within_tolerance          -- ... off by 1e-9 (inside rtol/atol)
      float_action_prob_times               -- the times arrays are float64, not int64
      drop_subject_from_action_prob_args    -- decision time 2 has no entry for subject 2
      extra_subject_in_action_prob_args     -- decision time 2 has an entry for a subject
                                               that is not in the analysis DataFrame
      drop_decision_time_from_action_prob_args -- the final decision time key is missing
      extra_decision_time_in_action_prob_args  -- a decision time key is present that the
                                               analysis DataFrame does not have
      blank_action_prob_args_at_active_cell -- () at an IN-study cell
      nonblank_action_prob_args_at_inactive_cell -- real args at an OUT-of-study cell
    """
    unknown_subject_id = 99
    inactive_time, inactive_subject_id = _INACTIVE_CELL_ARGCOVERAGE

    def _is_active(decision_time, subject_id):
        if all_subjects_always_active:
            return True
        return (decision_time, subject_id) != _INACTIVE_CELL_ARGCOVERAGE

    rows = []
    action_prob_func_args = {}
    for decision_time in _DECISION_TIMES_ARGCOVERAGE:
        args_by_subject = {}
        for subject_id in _SUBJECT_IDS_ARGCOVERAGE:
            active = _is_active(decision_time, subject_id)
            rows.append(
                {
                    _CALENDAR_T_COL_ARGCOVERAGE: decision_time,
                    _SUBJECT_ID_COL_ARGCOVERAGE: subject_id,
                    _POLICY_NUM_COL_ARGCOVERAGE: _POLICY_NUM_BY_TIME_ARGCOVERAGE[
                        decision_time
                    ],
                    _ACTIVE_COL_ARGCOVERAGE: 1 if active else 0,
                    _ACTION_COL_ARGCOVERAGE: (
                        float((decision_time + subject_id) % 2) if active else np.nan
                    ),
                    _ACTION_PROB_COL_ARGCOVERAGE: (
                        _recorded_action_prob_argcoverage(decision_time, subject_id)
                        if active
                        else np.nan
                    ),
                }
            )
            features = jnp.array([float(decision_time), float(subject_id)])
            args_by_subject[subject_id] = (
                (_BETA_ARGCOVERAGE, features) if active else ()
            )
        action_prob_func_args[decision_time] = args_by_subject

    if nonblank_action_prob_args_at_inactive_cell:
        action_prob_func_args[inactive_time][inactive_subject_id] = (
            _BETA_ARGCOVERAGE,
            jnp.array([float(inactive_time), float(inactive_subject_id)]),
        )
    if blank_action_prob_args_at_active_cell:
        # (t=2, subject 2) is in study, yet its args are blank.
        action_prob_func_args[2][2] = ()
    if drop_subject_from_action_prob_args:
        del action_prob_func_args[2][2]
    if extra_subject_in_action_prob_args:
        action_prob_func_args[2][unknown_subject_id] = (
            _BETA_ARGCOVERAGE,
            jnp.array([2.0, float(unknown_subject_id)]),
        )
    if drop_decision_time_from_action_prob_args:
        del action_prob_func_args[_DECISION_TIMES_ARGCOVERAGE[-1]]
    if extra_decision_time_in_action_prob_args:
        extra_time = _DECISION_TIMES_ARGCOVERAGE[-1] + 1
        action_prob_func_args[extra_time] = {
            subject_id: (
                _BETA_ARGCOVERAGE,
                jnp.array([float(extra_time), float(subject_id)]),
            )
            for subject_id in _SUBJECT_IDS_ARGCOVERAGE
        }

    analysis_df = pd.DataFrame(rows)

    times_dtype = np.float64 if float_action_prob_times else np.int64
    alg_update_func_args = {}
    for policy_num, update_times in _UPDATE_TIMES_BY_POLICY_ARGCOVERAGE.items():
        args_by_subject = {}
        for subject_id in _SUBJECT_IDS_ARGCOVERAGE:
            action_probs = np.array(
                [
                    [_recorded_action_prob_argcoverage(update_time, subject_id)]
                    for update_time in update_times
                ],
                dtype=np.float64,
            )
            action_prob_times = np.array(
                [[update_time] for update_time in update_times], dtype=times_dtype
            )
            args_by_subject[subject_id] = (
                _BETA_ARGCOVERAGE,
                action_probs,
                action_prob_times,
            )
        alg_update_func_args[policy_num] = args_by_subject

    beta, probs, times = alg_update_func_args[3][1]
    if corrupt_update_arg_action_prob:
        alg_update_func_args[3][1] = (beta, probs + 1e-3, times)
    if perturb_update_arg_action_prob_within_tolerance:
        alg_update_func_args[3][1] = (beta, probs + 1e-9, times)
    if blank_update_args_for_subject:
        alg_update_func_args[3][2] = ()
    if drop_subject_from_update_args:
        del alg_update_func_args[3][2]
    if extra_subject_in_update_args:
        alg_update_func_args[3][unknown_subject_id] = alg_update_func_args[3][1]

    return analysis_df, action_prob_func_args, alg_update_func_args


def _build_empty_analysis_df_argcoverage():
    """An analysis DataFrame with the right columns and dtypes but zero rows."""
    return pd.DataFrame(
        {
            _CALENDAR_T_COL_ARGCOVERAGE: pd.Series([], dtype="int64"),
            _SUBJECT_ID_COL_ARGCOVERAGE: pd.Series([], dtype="int64"),
            _POLICY_NUM_COL_ARGCOVERAGE: pd.Series([], dtype="float64"),
            _ACTIVE_COL_ARGCOVERAGE: pd.Series([], dtype="int64"),
            _ACTION_COL_ARGCOVERAGE: pd.Series([], dtype="float64"),
            _ACTION_PROB_COL_ARGCOVERAGE: pd.Series([], dtype="float64"),
        }
    )


def _call_alg_update_args_for_all_subjects_argcoverage(
    analysis_df, alg_update_func_args
):
    input_checks.require_alg_update_args_given_for_all_subjects_at_each_update(
        analysis_df, _SUBJECT_ID_COL_ARGCOVERAGE, alg_update_func_args
    )


def _call_action_prob_args_for_all_subjects_argcoverage(
    analysis_df, action_prob_func_args
):
    input_checks.require_action_prob_func_args_given_for_all_subjects_at_each_decision(
        analysis_df, _SUBJECT_ID_COL_ARGCOVERAGE, action_prob_func_args
    )


def _call_action_prob_args_for_all_times_argcoverage(
    analysis_df, action_prob_func_args
):
    input_checks.require_action_prob_func_args_given_for_all_decision_times(
        analysis_df, _CALENDAR_T_COL_ARGCOVERAGE, action_prob_func_args
    )


def _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args):
    input_checks.require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times(
        analysis_df,
        _CALENDAR_T_COL_ARGCOVERAGE,
        action_prob_func_args,
        _ACTIVE_COL_ARGCOVERAGE,
        _SUBJECT_ID_COL_ARGCOVERAGE,
    )


def _call_update_args_action_probs_correspond_argcoverage(
    analysis_df,
    alg_update_func_args,
    action_prob_index=_ACTION_PROB_INDEX_ARGCOVERAGE,
    action_prob_times_index=_ACTION_PROB_TIMES_INDEX_ARGCOVERAGE,
):
    input_checks.require_action_prob_args_in_alg_update_func_correspond_to_analysis_df(
        analysis_df,
        _ACTION_PROB_COL_ARGCOVERAGE,
        _CALENDAR_T_COL_ARGCOVERAGE,
        _SUBJECT_ID_COL_ARGCOVERAGE,
        alg_update_func_args,
        action_prob_index,
        action_prob_times_index,
    )


# ---------------------------------------------------------------------------
# require_alg_update_args_given_for_all_subjects_at_each_update: every policy that
# has update args must have an entry for EXACTLY the set of subjects in the
# analysis DataFrame (set equality, so extras are as bad as omissions).
# ---------------------------------------------------------------------------


def test_require_alg_update_args_given_for_all_subjects_at_each_update_passes_argcoverage():
    # A well-formed study, with float64 policy_num starting at 1.0 and int update-arg
    # keys 2 and 3, must satisfy the check without complaint.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage()

    _call_alg_update_args_for_all_subjects_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_alg_update_args_given_for_all_subjects_at_each_update_missing_subject_fails_argcoverage():
    # Subject 2 has no update args at policy 3 -- the omission this check exists to
    # catch, reported with the offending policy number.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        drop_subject_from_update_args=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Not all subjects present in algorithm update function args for policy"
            " number 3."
        ),
    ):
        _call_alg_update_args_for_all_subjects_argcoverage(
            analysis_df, alg_update_func_args
        )


def test_require_alg_update_args_given_for_all_subjects_at_each_update_extra_subject_fails_argcoverage():
    # The check asserts SET EQUALITY, so a subject present in the update args but
    # absent from the analysis DataFrame must fail too, not just a missing one.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        extra_subject_in_update_args=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Not all subjects present in algorithm update function args for policy"
            " number 3."
        ),
    ):
        _call_alg_update_args_for_all_subjects_argcoverage(
            analysis_df, alg_update_func_args
        )


def test_require_alg_update_args_given_for_all_subjects_at_each_update_passes_with_blank_args_argcoverage():
    # A blank () entry still counts as the subject being PRESENT: this check looks at
    # keys only, so a subject with no applicable args at an update must not trip it.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        blank_update_args_for_subject=True
    )

    _call_alg_update_args_for_all_subjects_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_alg_update_args_given_for_all_subjects_at_each_update_passes_with_no_updates_argcoverage():
    # An empty alg_update_func_args (a study whose algorithm never updated) has no
    # policies to iterate, so the check must pass vacuously rather than blow up.
    analysis_df, _, _ = _build_study_argcoverage()

    _call_alg_update_args_for_all_subjects_argcoverage(analysis_df, {})


def test_require_alg_update_args_given_for_all_subjects_at_each_update_empty_analysis_df_fails_argcoverage():
    # No subjects in the analysis DataFrame at all, but update args name two of them:
    # set equality must fail rather than silently accept the orphaned args.
    _, _, alg_update_func_args = _build_study_argcoverage()

    with pytest.raises(
        AssertionError,
        match="Not all subjects present in algorithm update function args",
    ):
        _call_alg_update_args_for_all_subjects_argcoverage(
            _build_empty_analysis_df_argcoverage(), alg_update_func_args
        )


# ---------------------------------------------------------------------------
# require_action_prob_func_args_given_for_all_subjects_at_each_decision: the same
# set-equality requirement, one level of nesting over decision times instead of
# policy numbers.
# ---------------------------------------------------------------------------


def test_require_action_prob_func_args_given_for_all_subjects_at_each_decision_passes_argcoverage():
    # Every decision time carries an entry for every subject (blank for the
    # out-of-study cell), which is what this check demands.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()

    _call_action_prob_args_for_all_subjects_argcoverage(
        analysis_df, action_prob_func_args
    )


def test_require_action_prob_func_args_given_for_all_subjects_at_each_decision_missing_subject_fails_argcoverage():
    # Subject 2 is missing from decision time 2 entirely. The message quotes the
    # decision time (its wording says "algorithm update function args", which is a
    # copy-paste artifact in the implementation, so pin only the stable part).
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        drop_subject_from_action_prob_args=True
    )

    with pytest.raises(
        AssertionError, match=re.escape("for decision time 2. Please see the contract")
    ):
        _call_action_prob_args_for_all_subjects_argcoverage(
            analysis_df, action_prob_func_args
        )


def test_require_action_prob_func_args_given_for_all_subjects_at_each_decision_extra_subject_fails_argcoverage():
    # A subject in the args that the analysis DataFrame never mentions must fail the
    # set-equality assertion as well.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        extra_subject_in_action_prob_args=True
    )

    with pytest.raises(
        AssertionError, match=re.escape("for decision time 2. Please see the contract")
    ):
        _call_action_prob_args_for_all_subjects_argcoverage(
            analysis_df, action_prob_func_args
        )


def test_require_action_prob_func_args_given_for_all_subjects_at_each_decision_passes_with_blank_args_argcoverage():
    # A blank () at an out-of-study cell keeps the subject's KEY present, so this
    # check must pass -- blankness is policed by the out-of-study check instead.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()
    inactive_time, inactive_subject_id = _INACTIVE_CELL_ARGCOVERAGE
    assert action_prob_func_args[inactive_time][inactive_subject_id] == ()

    _call_action_prob_args_for_all_subjects_argcoverage(
        analysis_df, action_prob_func_args
    )


# ---------------------------------------------------------------------------
# require_action_prob_func_args_given_for_all_decision_times: the top-level keys of
# action_prob_func_args must be exactly the unique calendar times.
# ---------------------------------------------------------------------------


def test_require_action_prob_func_args_given_for_all_decision_times_passes_argcoverage():
    # int keys 0..5 against an int64 calendar_t column: the sets compare equal.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()

    _call_action_prob_args_for_all_times_argcoverage(analysis_df, action_prob_func_args)


def test_require_action_prob_func_args_given_for_all_decision_times_missing_time_fails_argcoverage():
    # The final decision time has no args at all -- a silently truncated args dict.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        drop_decision_time_from_action_prob_args=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("Not all decision times present in action prob function args."),
    ):
        _call_action_prob_args_for_all_times_argcoverage(
            analysis_df, action_prob_func_args
        )


def test_require_action_prob_func_args_given_for_all_decision_times_extra_time_fails_argcoverage():
    # An args entry for a decision time the analysis DataFrame does not contain --
    # the other direction of the same set equality.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        extra_decision_time_in_action_prob_args=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("Not all decision times present in action prob function args."),
    ):
        _call_action_prob_args_for_all_times_argcoverage(
            analysis_df, action_prob_func_args
        )


def test_require_action_prob_func_args_given_for_all_decision_times_passes_on_empty_frame_with_empty_args_argcoverage():
    # Degenerate edge case: no rows means no decision times, which an empty args
    # dict matches exactly. Must pass, not raise on the empty unique() array.
    _call_action_prob_args_for_all_times_argcoverage(
        _build_empty_analysis_df_argcoverage(), {}
    )


def test_require_action_prob_func_args_given_for_all_decision_times_fails_on_empty_frame_with_nonempty_args_argcoverage():
    # An empty analysis DataFrame with populated args is a wiring error, so the
    # empty-vs-nonempty set comparison must fail.
    _, action_prob_func_args, _ = _build_study_argcoverage()

    with pytest.raises(
        AssertionError,
        match="Not all decision times present in action prob function args",
    ):
        _call_action_prob_args_for_all_times_argcoverage(
            _build_empty_analysis_df_argcoverage(), action_prob_func_args
        )


# ---------------------------------------------------------------------------
# require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times: the
# per-subject set of blank-args times must equal, exactly, the per-subject set of
# times the analysis DataFrame marks out of study.
# ---------------------------------------------------------------------------


def test_require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times_passes_argcoverage():
    # Subject 2's only blank cell is its only out-of-study time, so the two
    # per-subject mappings agree.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()

    _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args)


def test_require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times_blank_at_active_cell_fails_argcoverage():
    # A blank () at an IN-study cell (t=2, subject 2) gives that subject an extra
    # blank time the analysis DataFrame does not consider out of study.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        blank_action_prob_args_at_active_cell=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Inactive decision times according to the analysis DataFrame do not match up"
        ),
    ):
        _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args)


def test_require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times_nonblank_at_inactive_cell_fails_argcoverage():
    # Real (non-blank) args at an OUT-of-study cell: the args side then reports no
    # blank times for subject 2 at all while the DataFrame reports one -- the
    # opposite direction of the same set equality.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        nonblank_action_prob_args_at_inactive_cell=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Inactive decision times according to the analysis DataFrame do not match up"
        ),
    ):
        _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args)


def test_require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times_passes_when_nobody_leaves_argcoverage():
    # Nobody ever leaves the study and nothing is blank, so both sides are empty
    # mappings. Pins that a study with no dropout is accepted (the analysis-DataFrame
    # side is a plain dict from groupby while the args side is a defaultdict, and
    # those two must still compare equal when empty).
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage(
        all_subjects_always_active=True
    )
    assert not analysis_df[analysis_df[_ACTIVE_COL_ARGCOVERAGE] == 0].shape[0]

    _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args)


def test_require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times_missing_subject_key_fails_argcoverage():
    # Dropping subject 2's out-of-study entry altogether (rather than blanking it)
    # also leaves the args side with no blank time for that subject.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()
    inactive_time, inactive_subject_id = _INACTIVE_CELL_ARGCOVERAGE
    del action_prob_func_args[inactive_time][inactive_subject_id]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Inactive decision times according to the analysis DataFrame do not match up"
        ),
    ):
        _call_out_of_study_blank_args_argcoverage(analysis_df, action_prob_func_args)


# ---------------------------------------------------------------------------
# require_action_prob_args_in_alg_update_func_correspond_to_analysis_df: when the
# update args carry action probabilities, they must match the analysis DataFrame's
# recorded ones at the times the args also carry, within rtol=1e-5 / atol=1e-8.
# ---------------------------------------------------------------------------


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_passes_argcoverage():
    # Update args carrying exactly the recorded probabilities for the times they name.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage()

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_returns_early_when_index_absent_argcoverage():
    # A negative "absent" action-prob index means the update args carry no action
    # probabilities, so the check must return BEFORE touching the args -- proved here
    # by handing it args whose probabilities are deliberately wrong (and would fail
    # loudly at index 1) and expecting no error at all.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        corrupt_update_arg_action_prob=True
    )

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df,
        alg_update_func_args,
        action_prob_index=_ABSENT_INDEX_ARGCOVERAGE,
        action_prob_times_index=_ABSENT_INDEX_ARGCOVERAGE,
    )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_mismatch_fails_argcoverage():
    # Subject 1's probabilities at policy 3 are off by 1e-3, well outside
    # rtol=1e-5 / atol=1e-8 for probabilities of this size.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        corrupt_update_arg_action_prob=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "There is a mismatch for subject 1 between the action probabilities supplied"
            " in the args to the algorithm update function at policy 3"
        ),
    ):
        _call_update_args_action_probs_correspond_argcoverage(
            analysis_df, alg_update_func_args
        )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_within_tolerance_passes_argcoverage():
    # A 1e-9 discrepancy (float round-trip noise) is inside the explicit
    # rtol=1e-5 / atol=1e-8 tolerances and must NOT be reported as a mismatch.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        perturb_update_arg_action_prob_within_tolerance=True
    )

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_skips_blank_args_argcoverage():
    # A blank () entry means "not applicable at this update"; the check must skip it
    # via `if not args: continue` rather than index into an empty tuple.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        blank_update_args_for_subject=True
    )
    assert alg_update_func_args[3][2] == ()

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_passes_with_float_times_argcoverage():
    # Real studies carry float64 time arrays (policy_num and friends are float64 in
    # the fixtures). The check does `decision_time.item()` to look up an int64
    # calendar_t, and 5.0 must still find time 5.
    analysis_df, _, alg_update_func_args = _build_study_argcoverage(
        float_action_prob_times=True
    )
    assert alg_update_func_args[3][1][_ACTION_PROB_TIMES_INDEX_ARGCOVERAGE].dtype == (
        np.float64
    )

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_passes_with_no_update_args_argcoverage():
    # No updates at all: nothing to compare, so the check must pass vacuously even
    # with a real action-prob index supplied.
    analysis_df, _, _ = _build_study_argcoverage()

    _call_update_args_action_probs_correspond_argcoverage(analysis_df, {})


def test_require_action_prob_args_in_alg_update_func_correspond_to_analysis_df_uses_recorded_probs_not_nan_rows_argcoverage():
    # Guard on the fixture's own premise for this check: every time carried in the
    # update args is an IN-study time for the subject carrying it, so the recorded
    # action probability looked up is a real number rather than the NaN stored on
    # out-of-study rows (np.allclose against NaN would fail unconditionally).
    analysis_df, _, alg_update_func_args = _build_study_argcoverage()
    lookup = analysis_df.set_index(
        [_CALENDAR_T_COL_ARGCOVERAGE, _SUBJECT_ID_COL_ARGCOVERAGE]
    )[_ACTION_PROB_COL_ARGCOVERAGE].to_dict()
    for _policy_num, args_by_subject in alg_update_func_args.items():
        for subject_id, args in args_by_subject.items():
            for decision_time in args[_ACTION_PROB_TIMES_INDEX_ARGCOVERAGE].flatten():
                assert not np.isnan(lookup[(decision_time.item(), subject_id)])

    _call_update_args_action_probs_correspond_argcoverage(
        analysis_df, alg_update_func_args
    )


def test_require_out_of_study_rows_carry_nan_action_and_blank_args_argcoverage():
    # Fixture-shape guard shared by this whole family: the out-of-study row really
    # does hold NaN in the float64 action column and int64 0 in the in-study column,
    # and its args entry really is the blank tuple -- the combination the blank-args
    # check is written against.
    analysis_df, action_prob_func_args, _ = _build_study_argcoverage()
    inactive_time, inactive_subject_id = _INACTIVE_CELL_ARGCOVERAGE
    row = analysis_df[
        (analysis_df[_CALENDAR_T_COL_ARGCOVERAGE] == inactive_time)
        & (analysis_df[_SUBJECT_ID_COL_ARGCOVERAGE] == inactive_subject_id)
    ].iloc[0]

    assert analysis_df[_ACTION_COL_ARGCOVERAGE].dtype == np.float64
    assert analysis_df[_ACTIVE_COL_ARGCOVERAGE].dtype == np.int64
    assert analysis_df[_POLICY_NUM_COL_ARGCOVERAGE].dtype == np.float64
    assert analysis_df[_POLICY_NUM_COL_ARGCOVERAGE].min() == 1.0
    assert row[_ACTIVE_COL_ARGCOVERAGE] == 0
    assert np.isnan(row[_ACTION_COL_ARGCOVERAGE])
    assert action_prob_func_args[inactive_time][inactive_subject_id] == ()
    assert isinstance(
        collections.defaultdict(set), dict
    )  # the args side of the blank-times comparison is a defaultdict
