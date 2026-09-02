"""
Tests for require_output_dir_ready, and for the subject-id column contract after the
subject-id column was exempted from require_all_named_columns_not_object_type_in_analysis_df
on 2026-09-02 so that STRING subject ids are allowed.

The exemption moved the subject-id column's only structural guard onto
require_hashable_subject_ids, so the tests below pin both halves of that guard (hashability and
mutual comparability) as well as the exemption itself.
"""

import os
import re

import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

_ACTIVE_COL_SID = "in_study"
_ACTION_COL_SID = "action"
_POLICY_NUM_COL_SID = "policy_num"
_CALENDAR_T_COL_SID = "calendar_t"
_SUBJECT_ID_COL_SID = "user_id"
_ACTION_PROB_COL_SID = "action1prob"
_REWARD_COL_SID = "reward"


def _build_study_sid(subject_ids=("s1", "s2"), object_dtype_column=None):
    """
    Two subjects x two calendar times, every row active. subject_ids supplies the two ids and
    may be any type, which is the point of these tests. object_dtype_column, when given, names
    a NON-subject-id column to force to object dtype.
    """
    rows = []
    for subject_id in subject_ids:
        for calendar_t in (0, 1):
            rows.append(
                {
                    _SUBJECT_ID_COL_SID: subject_id,
                    _CALENDAR_T_COL_SID: calendar_t,
                    _ACTIVE_COL_SID: 1,
                    _ACTION_COL_SID: 0,
                    _POLICY_NUM_COL_SID: 1,
                    _ACTION_PROB_COL_SID: 0.5,
                    _REWARD_COL_SID: 1.0,
                }
            )
    analysis_df = pd.DataFrame(rows)
    if object_dtype_column is not None:
        analysis_df[object_dtype_column] = analysis_df[object_dtype_column].astype(
            object
        )
    return analysis_df


def _check_object_dtype_sid(analysis_df):
    # reward_col_name became a required 8th positional parameter on 2026-09-02, and the reward
    # column is one of the columns this check now inspects.
    return input_checks.require_all_named_columns_not_object_type_in_analysis_df(
        analysis_df,
        _ACTIVE_COL_SID,
        _ACTION_COL_SID,
        _POLICY_NUM_COL_SID,
        _CALENDAR_T_COL_SID,
        _SUBJECT_ID_COL_SID,
        _ACTION_PROB_COL_SID,
        _REWARD_COL_SID,
    )


def _check_columns_present_sid(analysis_df, reward_col_name=_REWARD_COL_SID):
    # Same new 8th positional parameter as above.
    return input_checks.require_all_named_columns_present_in_analysis_df(
        analysis_df,
        _ACTIVE_COL_SID,
        _ACTION_COL_SID,
        _POLICY_NUM_COL_SID,
        _CALENDAR_T_COL_SID,
        _SUBJECT_ID_COL_SID,
        _ACTION_PROB_COL_SID,
        reward_col_name,
    )


def _check_subject_ids_sid(analysis_df):
    return input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_SID, _SUBJECT_ID_COL_SID
    )


# ---------------------------------------------------------------------------
# require_output_dir_ready
# ---------------------------------------------------------------------------


def test_require_output_dir_ready_passes_on_a_writable_directory(tmp_path):
    input_checks.require_output_dir_ready(tmp_path)


def test_require_output_dir_ready_accepts_a_string_path(tmp_path):
    # analyze_dataset's own signature is pathlib.Path | str.
    input_checks.require_output_dir_ready(str(tmp_path))


def test_require_output_dir_ready_leaves_no_probe_file_behind(tmp_path):
    # The writability test writes a real file; it must clean up after itself rather than
    # littering the user's results directory.
    before = set(os.listdir(tmp_path))
    input_checks.require_output_dir_ready(tmp_path)
    assert set(os.listdir(tmp_path)) == before


def test_require_output_dir_ready_raises_on_a_missing_directory(tmp_path):
    # The whole point: this must fail BEFORE the analysis runs, not after it has computed
    # everything and tries to write analysis.pkl.
    missing = tmp_path / "does_not_exist"
    with pytest.raises(AssertionError, match="does not exist"):
        input_checks.require_output_dir_ready(missing)


def test_require_output_dir_ready_raises_when_path_is_a_file(tmp_path):
    not_a_dir = tmp_path / "results.txt"
    not_a_dir.write_text("not a directory")
    with pytest.raises(AssertionError, match="is not a directory"):
        input_checks.require_output_dir_ready(not_a_dir)


def test_require_output_dir_ready_raises_on_a_read_only_directory(tmp_path):
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        with pytest.raises(AssertionError, match="could not be written to"):
            input_checks.require_output_dir_ready(read_only)
    finally:
        # Restore so tmp_path teardown can remove it.
        read_only.chmod(0o700)


# ---------------------------------------------------------------------------
# The object-dtype exemption for the subject-id column
# ---------------------------------------------------------------------------


def test_object_dtype_check_allows_object_subject_ids():
    # REGRESSION: until 2026-09-02 this check included subject_id_col_name, so a string
    # subject-id column -- which pandas stores as object dtype -- was rejected outright, and no
    # study could use string ids at all.
    analysis_df = _build_study_sid(subject_ids=("s1", "s2"))
    assert analysis_df[_SUBJECT_ID_COL_SID].dtype == object
    _check_object_dtype_sid(analysis_df)


def test_object_dtype_check_still_allows_numeric_subject_ids():
    analysis_df = _build_study_sid(subject_ids=(1, 2))
    _check_object_dtype_sid(analysis_df)


@pytest.mark.parametrize(
    "colname",
    [
        _ACTIVE_COL_SID,
        _ACTION_COL_SID,
        _POLICY_NUM_COL_SID,
        _CALENDAR_T_COL_SID,
        _ACTION_PROB_COL_SID,
        # reward joined the inspected columns on 2026-09-02, at the same time the subject-id
        # column was exempted from them.
        _REWARD_COL_SID,
    ],
)
def test_object_dtype_check_still_rejects_every_other_column(colname):
    # The exemption must be surgical: the other six columns are all consumed numerically, so
    # object dtype genuinely breaks them and must still be rejected.
    analysis_df = _build_study_sid(subject_ids=(1, 2), object_dtype_column=colname)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These analysis DataFrame columns are of object type, but are consumed "
            f"numerically: {[colname]}"
        ),
    ):
        _check_object_dtype_sid(analysis_df)


# ---------------------------------------------------------------------------
# The reward column's arrival in require_all_named_columns_present_in_analysis_df
# ---------------------------------------------------------------------------


def test_columns_present_check_passes_on_the_fixture_sid():
    _check_columns_present_sid(_build_study_sid(subject_ids=(1, 2)))


def test_columns_present_check_now_catches_a_misnamed_reward_column_sid():
    # REGRESSION: reward_col_name was not checked at all until 2026-09-02, so a typo in it
    # surfaced as a bare KeyError from inside a plotting routine.
    analysis_df = _build_study_sid(subject_ids=(1, 2))
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These named columns are not in the analysis DataFrame: ['rewrad']"
        ),
    ):
        _check_columns_present_sid(analysis_df, reward_col_name="rewrad")


def test_columns_present_check_reports_every_missing_column_at_once_sid():
    # INVERTED on 2026-09-02: this check used to assert once per column, so it reported only the
    # FIRST missing name ("{col_name} not in analysis DataFrame."). It now collects them all into
    # one message, along with the columns that are present.
    analysis_df = _build_study_sid(subject_ids=(1, 2)).drop(
        columns=[_ACTION_COL_SID, _ACTION_PROB_COL_SID]
    )
    with pytest.raises(AssertionError) as excinfo:
        _check_columns_present_sid(analysis_df, reward_col_name="rewrad")
    message = str(excinfo.value)
    # Hoisted out of the f-strings below: ruff's formatter splits a long expression inside an
    # f-string across lines, which is a SyntaxError on Python 3.10.
    expected_missing = [_ACTION_COL_SID, _ACTION_PROB_COL_SID, "rewrad"]
    columns_present = sorted(map(str, analysis_df.columns))
    assert (
        f"These named columns are not in the analysis DataFrame: {expected_missing}."
        in message
    )
    assert f"Columns present: {columns_present}." in message


# ---------------------------------------------------------------------------
# require_hashable_subject_ids -- now the subject-id column's only structural guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "subject_ids",
    [
        ("s1", "s2"),
        (1, 2),
        (1.5, 2.5),
        (np.int64(1), np.int64(2)),
        # Ids at or above 2**31, which the old jnp.array(...) conversion truncated.
        (1700000000123, 1700000000456),
    ],
)
def test_require_hashable_subject_ids_passes_on_homogeneous_hashable_ids(subject_ids):
    _check_subject_ids_sid(_build_study_sid(subject_ids=subject_ids))


@pytest.mark.parametrize("unhashable", [[1], {1: 2}, {1, 2}])
def test_require_hashable_subject_ids_raises_on_unhashable_ids(unhashable):
    # This is the case the object-dtype check used to catch incidentally. Now that the
    # subject-id column is exempt from that check, this guard is the only thing standing
    # between an unhashable id and a KeyError deep in the per-subject dictionaries.
    #
    # Built through the row records rather than assigned with .loc afterwards: pandas rejects
    # setting a dict or set into a single cell ("Must have equal len keys and value when
    # setting with an iterable"), so .loc cannot express this fixture at all.
    analysis_df = _build_study_sid(subject_ids=(unhashable, "s2"))
    with pytest.raises(AssertionError, match="must be hashable"):
        _check_subject_ids_sid(analysis_df)


def test_require_hashable_subject_ids_raises_on_mixed_string_and_numeric_ids():
    # Newly reachable because of the object-dtype exemption: a mixed column is perfectly
    # hashable, so it keys dictionaries fine and would instead blow up much later as a bare
    # "'<' not supported between instances of 'str' and 'int'" from one of the sorted(...)
    # calls over subject ids.
    analysis_df = _build_study_sid(subject_ids=("s1", 2))
    with pytest.raises(AssertionError, match=re.escape("mix ['number', 'string']")):
        _check_subject_ids_sid(analysis_df)


def test_require_hashable_subject_ids_treats_int_and_float_ids_as_comparable():
    # int and float are mutually orderable, so a column mixing them sorts fine and must NOT be
    # rejected -- the comparability check groups them as one "number" type on purpose.
    analysis_df = _build_study_sid(subject_ids=(1, 2.5))
    _check_subject_ids_sid(analysis_df)


def test_require_hashable_subject_ids_covers_out_of_study_rows():
    # INVERTED on 2026-09-02: this check reads the WHOLE subject-id column now, not just the
    # active rows, because every consumer of subject ids (.unique(), groupby(), duplicated())
    # hashes the whole column and so breaks on an unhashable id wherever it sits.
    #
    # The unhashable id is placed via the row records, not .loc: pandas unwraps a length-1 list
    # to its scalar on .loc assignment, so `df.loc[0, col] = [999]` stores the INT 999 and the
    # test would pass while proving nothing. Assert the cell really holds a list first.
    analysis_df = _build_study_sid(subject_ids=([999], "s2"))
    assert isinstance(analysis_df.loc[0, _SUBJECT_ID_COL_SID], list)
    # Rows 0 and 1 are the unhashable subject's two decision times; mark both out of study.
    analysis_df.loc[0:1, _ACTIVE_COL_SID] = 0

    with pytest.raises(AssertionError, match="must be hashable"):
        _check_subject_ids_sid(analysis_df)


def test_require_hashable_subject_ids_passes_when_no_row_is_active():
    analysis_df = _build_study_sid(subject_ids=("s1", "s2"))
    analysis_df[_ACTIVE_COL_SID] = 0
    _check_subject_ids_sid(analysis_df)


def test_mixed_ids_would_otherwise_fail_only_at_sort_time():
    # Pins the justification for the comparability check: sorting really is what breaks, and it
    # breaks with a message that names neither the column nor the values.
    with pytest.raises(TypeError, match="not supported between instances"):
        sorted(["s1", 2])
