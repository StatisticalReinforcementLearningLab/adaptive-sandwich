"""
Coverage for the five input checks whose long-standing bugs were fixed on 2026-09-02,
plus the shared helper introduced with them:

  - require_binary_actions
  - require_binary_active_indicators
  - require_consecutive_integer_policy_numbers
  - require_consecutive_integer_calendar_times
  - require_hashable_subject_ids
  - _unique_sorted_integer_values

Every "regression" test below is an input the OLD implementation accepted (or crashed on
with the wrong exception) and the current one correctly rejects.
"""

import re

import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

# Column names match the real fixture shape in tests/benchmarks/fixtures/small.
_ACTIVE_COL_FIXES = "in_study"
_ACTION_COL_FIXES = "action"
_POLICY_NUM_COL_FIXES = "policy_num"
_CALENDAR_T_COL_FIXES = "calendar_t"
_SUBJECT_ID_COL_FIXES = "subject_id"

# Row labels of the baseline study built by _build_study_fixes, for reference:
#   0..3 -> subject "s1" at calendar_t 0..3, all in study
#   4..6 -> subject "s2" at calendar_t 0..2, in study
#   7    -> subject "s2" at calendar_t 3,    OUT of study (NaN action)
_S1_T1_LABEL_FIXES = 1
_S1_T2_LABEL_FIXES = 2
_S1_T3_LABEL_FIXES = 3
_S2_T1_LABEL_FIXES = 5
_S2_T2_LABEL_FIXES = 6
_OUT_OF_STUDY_LABEL_FIXES = 7
# A deliberately non-monotonic, deterministic row permutation (positional).
_SHUFFLE_ORDER_FIXES = [3, 5, 2, 7, 0, 6, 1, 4]


def _build_study_fixes(
    *,
    empty=False,
    shuffle_rows=False,
    integer_policy_nums=False,
    zero_based_policy_nums=False,
    float_calendar_times=False,
    integer_subject_ids=False,
    all_rows_out_of_study=False,
    row_labeled_zero_out_of_study=False,
    active_indicator_two=False,
    active_indicator_half=False,
    nan_active_indicator=False,
    negative_active_indicator=False,
    fractional_action=False,
    action_of_two=False,
    nan_action_on_active_row=False,
    gapped_policy_nums=False,
    fractional_policy_num=False,
    all_active_rows_fallback=False,
    some_fallback_policy_nums=False,
    out_of_study_row_wild_policy_num=False,
    gapped_calendar_times=False,
    fractional_calendar_time=False,
    out_of_study_row_gaps_calendar_times=False,
    list_subject_id_on_active_row=False,
    set_subject_id_on_active_row=False,
    list_subject_id_on_out_of_study_row=False,
):
    """
    A small valid study shaped like the real fixtures: two subjects over calendar times
    0..3, in_study as int64 0/1, action as float64 (NaN on the out-of-study row),
    policy_num as float64 whose INITIAL POLICY NUMBER IS 1 (not 0), subject ids as
    strings. Subject "s2" leaves the study at calendar_t 3.

    With no keyword flags the frame passes all five checks. Each flag introduces exactly
    one defect (or one benign variation); see the flag name and the test that uses it.
    """
    policy_num_by_time = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}
    rows = []
    for subject_id in ("s1", "s2"):
        for calendar_t in (0, 1, 2, 3):
            is_active = not (subject_id == "s2" and calendar_t == 3)
            policy_num = policy_num_by_time[calendar_t]
            if zero_based_policy_nums:
                policy_num -= 1.0
            rows.append(
                {
                    _CALENDAR_T_COL_FIXES: calendar_t,
                    _SUBJECT_ID_COL_FIXES: subject_id,
                    _ACTIVE_COL_FIXES: 1 if is_active else 0,
                    _POLICY_NUM_COL_FIXES: policy_num,
                    _ACTION_COL_FIXES: float(calendar_t % 2) if is_active else np.nan,
                }
            )
    analysis_df = pd.DataFrame(rows)

    if empty:
        # Keep the columns and their dtypes, drop every row.
        return analysis_df.iloc[0:0]

    # --- dtype variations -------------------------------------------------------
    if integer_policy_nums:
        analysis_df[_POLICY_NUM_COL_FIXES] = analysis_df[_POLICY_NUM_COL_FIXES].astype(
            "int64"
        )
    if float_calendar_times or fractional_calendar_time:
        analysis_df[_CALENDAR_T_COL_FIXES] = analysis_df[_CALENDAR_T_COL_FIXES].astype(
            "float64"
        )
    if integer_subject_ids:
        analysis_df[_SUBJECT_ID_COL_FIXES] = (
            analysis_df[_SUBJECT_ID_COL_FIXES].map({"s1": 11, "s2": 12}).astype("int64")
        )

    # --- active-indicator column ------------------------------------------------
    if all_rows_out_of_study:
        analysis_df[_ACTIVE_COL_FIXES] = 0
    if row_labeled_zero_out_of_study:
        analysis_df.at[0, _ACTIVE_COL_FIXES] = 0
        analysis_df.at[0, _ACTION_COL_FIXES] = np.nan
    if active_indicator_two:
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTIVE_COL_FIXES] = 2
    if negative_active_indicator:
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTIVE_COL_FIXES] = -1
    if active_indicator_half:
        analysis_df[_ACTIVE_COL_FIXES] = analysis_df[_ACTIVE_COL_FIXES].astype(
            "float64"
        )
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTIVE_COL_FIXES] = 0.5
    if nan_active_indicator:
        analysis_df[_ACTIVE_COL_FIXES] = analysis_df[_ACTIVE_COL_FIXES].astype(
            "float64"
        )
        analysis_df.at[_OUT_OF_STUDY_LABEL_FIXES, _ACTIVE_COL_FIXES] = np.nan

    # --- action column ----------------------------------------------------------
    if fractional_action:
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTION_COL_FIXES] = 0.5
    if action_of_two:
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTION_COL_FIXES] = 2.0
    if nan_action_on_active_row:
        analysis_df.at[_S1_T1_LABEL_FIXES, _ACTION_COL_FIXES] = np.nan

    # --- policy_num column ------------------------------------------------------
    if gapped_policy_nums:
        analysis_df.at[_S1_T3_LABEL_FIXES, _POLICY_NUM_COL_FIXES] = 5.0
    if fractional_policy_num:
        # Deliberately a row whose policy number is 1.0, so truncating 1.5 -> 1 leaves
        # the untouched, apparently consecutive {1, 2, 3}.
        analysis_df.at[_S1_T1_LABEL_FIXES, _POLICY_NUM_COL_FIXES] = 1.5
    if all_active_rows_fallback:
        analysis_df.loc[
            analysis_df[_ACTIVE_COL_FIXES] == 1, _POLICY_NUM_COL_FIXES
        ] = -1.0
    if some_fallback_policy_nums:
        analysis_df.loc[
            [_S1_T1_LABEL_FIXES, _S2_T1_LABEL_FIXES], _POLICY_NUM_COL_FIXES
        ] = -1.0
    if out_of_study_row_wild_policy_num:
        analysis_df.at[_OUT_OF_STUDY_LABEL_FIXES, _POLICY_NUM_COL_FIXES] = 99.0

    # --- calendar_t column ------------------------------------------------------
    if gapped_calendar_times:
        analysis_df.loc[
            analysis_df[_CALENDAR_T_COL_FIXES] == 2, _CALENDAR_T_COL_FIXES
        ] = 5
    if fractional_calendar_time:
        # Only subject s2's t=2 row, so truncating 1.5 -> 1 leaves s1's 2 in place and
        # the truncated sequence {0, 1, 2, 3} gap-free.
        analysis_df.at[_S2_T2_LABEL_FIXES, _CALENDAR_T_COL_FIXES] = 1.5
    if out_of_study_row_gaps_calendar_times:
        analysis_df.at[_OUT_OF_STUDY_LABEL_FIXES, _CALENDAR_T_COL_FIXES] = 9

    # --- subject_id column ------------------------------------------------------
    unhashable_by_label = {}
    if list_subject_id_on_active_row:
        unhashable_by_label[_S1_T1_LABEL_FIXES] = ["s1"]
    if set_subject_id_on_active_row:
        unhashable_by_label[_S1_T1_LABEL_FIXES] = {"s1"}
    if list_subject_id_on_out_of_study_row:
        unhashable_by_label[_OUT_OF_STUDY_LABEL_FIXES] = ["s2"]
    if unhashable_by_label:
        # Element-wise assignment, because .at/.loc would try to broadcast a container.
        analysis_df[_SUBJECT_ID_COL_FIXES] = [
            unhashable_by_label.get(label, value)
            for label, value in analysis_df[_SUBJECT_ID_COL_FIXES].items()
        ]

    if shuffle_rows:
        # Row order (and therefore index-label order) is deliberately not sorted by
        # calendar_t or policy_num.
        analysis_df = analysis_df.iloc[_SHUFFLE_ORDER_FIXES]

    return analysis_df


# ---------------------------------------------------------------------------
# require_binary_actions
# ---------------------------------------------------------------------------


def test_require_binary_actions_passes_on_valid_float_actions_with_nan_out_of_study():
    # Real fixture shape: float64 action column holding NaN on out-of-study rows. The
    # active filter must keep those NaNs from being treated as non-binary actions.
    analysis_df = _build_study_fixes()

    input_checks.require_binary_actions(
        analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
    )


def test_require_binary_actions_passes_on_integer_dtype_actions():
    # int64 0/1 actions (no out-of-study rows to hold NaN) are equally valid.
    analysis_df = _build_study_fixes()
    analysis_df = analysis_df[analysis_df[_ACTIVE_COL_FIXES] == 1].copy()
    analysis_df[_ACTION_COL_FIXES] = analysis_df[_ACTION_COL_FIXES].astype("int64")

    input_checks.require_binary_actions(
        analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
    )


def test_require_binary_actions_raises_on_fractional_action_on_active_row():
    # REGRESSION: the old implementation cast the action column with .astype("int64")
    # first, truncating 0.5 -> 0, so a fractional action sailed through.
    analysis_df = _build_study_fixes(fractional_action=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("Actions are not binary. action takes the value(s) [0.5]"),
    ):
        input_checks.require_binary_actions(
            analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
        )


def test_require_binary_actions_raises_on_action_of_two():
    # An out-of-range integer action was caught before the fix too; keep it pinned.
    analysis_df = _build_study_fixes(action_of_two=True)

    with pytest.raises(
        AssertionError, match=re.escape("action takes the value(s) [2.0]")
    ):
        input_checks.require_binary_actions(
            analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
        )


def test_require_binary_actions_raises_on_nan_action_on_active_row():
    # NaN is legitimate only while a subject is out of study; on an active row it means
    # a missing action and must be rejected (the old int cast raised on NaN instead of
    # reporting it).
    analysis_df = _build_study_fixes(nan_action_on_active_row=True)

    with pytest.raises(
        AssertionError, match=re.escape("action takes the value(s) [nan]")
    ):
        input_checks.require_binary_actions(
            analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
        )


def test_require_binary_actions_tolerates_non_binary_actions_on_out_of_study_rows():
    # The active filter is deliberately retained: actions are only meaningful in study,
    # so garbage on out-of-study rows must not fail this particular check.
    analysis_df = _build_study_fixes(all_rows_out_of_study=True, fractional_action=True)

    input_checks.require_binary_actions(
        analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
    )


def test_require_binary_actions_passes_on_empty_analysis_df():
    # .isin([0, 1]).all() on an empty Series is vacuously True -- no crash, no failure.
    analysis_df = _build_study_fixes(empty=True)

    input_checks.require_binary_actions(
        analysis_df, _ACTIVE_COL_FIXES, _ACTION_COL_FIXES
    )


# ---------------------------------------------------------------------------
# require_binary_active_indicators
# ---------------------------------------------------------------------------


def test_require_binary_active_indicators_passes_on_int64_zero_one_column():
    # Real fixture shape: in_study is int64 holding 0/1.
    analysis_df = _build_study_fixes()
    assert analysis_df[_ACTIVE_COL_FIXES].dtype == "int64"

    input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_passes_on_float64_zero_one_column():
    # A float64 in-study column of 0.0/1.0 is still binary and must pass.
    analysis_df = _build_study_fixes()
    analysis_df[_ACTIVE_COL_FIXES] = analysis_df[_ACTIVE_COL_FIXES].astype("float64")

    input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_raises_on_indicator_value_two():
    # REGRESSION: the old check filtered to active == 1 and then asserted those values
    # were in {0, 1} -- vacuously true -- so an in-study column of {1, 2} passed.
    analysis_df = _build_study_fixes(active_indicator_two=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "In-study indicators are not binary. in_study takes the value(s) [2]"
        ),
    ):
        input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_raises_on_fractional_indicator_half():
    # REGRESSION: 0.5 was both hidden by the active == 1 pre-filter and truncated to 0
    # by the old .astype("int64").
    analysis_df = _build_study_fixes(active_indicator_half=True)

    with pytest.raises(
        AssertionError, match=re.escape("in_study takes the value(s) [0.5]")
    ):
        input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_raises_on_nan_indicator():
    # REGRESSION: a NaN on an out-of-study row was never looked at, because the old
    # check only inspected rows where active == 1. Now the whole column is validated.
    analysis_df = _build_study_fixes(nan_active_indicator=True)

    with pytest.raises(
        AssertionError, match=re.escape("in_study takes the value(s) [nan]")
    ):
        input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_raises_on_negative_indicator():
    # -1 is not a valid in-study flag; it would silently exclude the row everywhere.
    analysis_df = _build_study_fixes(negative_active_indicator=True)

    with pytest.raises(
        AssertionError, match=re.escape("in_study takes the value(s) [-1]")
    ):
        input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


def test_require_binary_active_indicators_passes_on_empty_analysis_df():
    # An empty frame has no offending values.
    analysis_df = _build_study_fixes(empty=True)

    input_checks.require_binary_active_indicators(analysis_df, _ACTIVE_COL_FIXES)


# ---------------------------------------------------------------------------
# _unique_sorted_integer_values
# ---------------------------------------------------------------------------


def test_unique_sorted_integer_values_sorts_regardless_of_appearance_order():
    # REGRESSION: pandas' Series.unique() returns order of first APPEARANCE, which is
    # what made both consecutive-integer checks depend on row order.
    values = pd.Series([3.0, 1.0, 2.0, 1.0])
    assert list(values.unique()) == [3.0, 1.0, 2.0]

    unique_sorted = input_checks._unique_sorted_integer_values(values, "Some values")

    assert unique_sorted.tolist() == [1, 2, 3]


def test_unique_sorted_integer_values_returns_empty_for_empty_series():
    # Empty in, empty out -- and no .min() on an empty array.
    unique_sorted = input_checks._unique_sorted_integer_values(
        pd.Series([], dtype="float64"), "Some values"
    )

    assert unique_sorted.size == 0


def test_unique_sorted_integer_values_passes_through_integer_dtype():
    # An already-integer column skips the integrality assertions entirely.
    unique_sorted = input_checks._unique_sorted_integer_values(
        pd.Series([7, 5, 6], dtype="int64"), "Some values"
    )

    assert unique_sorted.tolist() == [5, 6, 7]
    assert np.issubdtype(unique_sorted.dtype, np.integer)


def test_unique_sorted_integer_values_casts_integral_floats_to_int64():
    # float64 columns are supported for legacy reasons, but only when every value is
    # exactly integral; the result is handed back as int64.
    unique_sorted = input_checks._unique_sorted_integer_values(
        pd.Series([2.0, 1.0, 3.0]), "Some values"
    )

    assert unique_sorted.dtype == np.dtype("int64")
    assert unique_sorted.tolist() == [1, 2, 3]


def test_unique_sorted_integer_values_raises_on_fractional_float():
    # REGRESSION: the old .astype("int64") truncated 1.5 -> 1, letting a fractional
    # value masquerade as part of a perfectly consecutive sequence.
    with pytest.raises(
        AssertionError,
        match=re.escape("Some values contains non-integer values: [1.5]"),
    ):
        input_checks._unique_sorted_integer_values(
            pd.Series([1.0, 1.5, 2.0]), "Some values"
        )


def test_unique_sorted_integer_values_raises_on_nan():
    # NaN is reported as non-finite rather than crashing the int cast.
    with pytest.raises(
        AssertionError,
        match=re.escape("Some values contains non-finite values: [nan]"),
    ):
        input_checks._unique_sorted_integer_values(
            pd.Series([1.0, np.nan]), "Some values"
        )


def test_unique_sorted_integer_values_raises_on_infinity():
    # Infinity needs its own guard: np.rint(inf) == inf, so the integrality test alone
    # would happily accept it and then overflow the int64 cast.
    with pytest.raises(
        AssertionError,
        match=re.escape("Some values contains non-finite values: [inf]"),
    ):
        input_checks._unique_sorted_integer_values(
            pd.Series([1.0, np.inf]), "Some values"
        )


# ---------------------------------------------------------------------------
# require_consecutive_integer_policy_numbers
# ---------------------------------------------------------------------------


def test_require_consecutive_integer_policy_numbers_passes_on_study_starting_at_one():
    # Real fixture shape: float64 policy_num whose INITIAL POLICY NUMBER IS 1, not 0.
    analysis_df = _build_study_fixes()
    assert analysis_df[_POLICY_NUM_COL_FIXES].min() == 1.0

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_passes_on_zero_based_numbering():
    # A 0-based study is equally consecutive; the check compares against
    # range(min, max + 1), not range(0, max + 1).
    analysis_df = _build_study_fixes(zero_based_policy_nums=True)

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_passes_on_integer_dtype_column():
    # int64 policy numbers take the early dtype branch of the shared helper.
    analysis_df = _build_study_fixes(integer_policy_nums=True)

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_passes_on_shuffled_row_order():
    # REGRESSION: with rows not ordered by policy_num, Series.unique() yields
    # [3, 1, 2], which the old implementation compared to range(1, 4) and spuriously
    # rejected. The values are consecutive, so this must pass.
    analysis_df = _build_study_fixes(shuffle_rows=True)
    active_policy_nums = analysis_df.loc[
        analysis_df[_ACTIVE_COL_FIXES] == 1, _POLICY_NUM_COL_FIXES
    ]
    assert list(active_policy_nums.unique()) == [3.0, 1.0, 2.0]

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_ignores_negative_fallback_values():
    # Negative policy numbers mark "fallback policy used"; they are excluded before the
    # consecutiveness comparison and must not read as a gap below the minimum.
    analysis_df = _build_study_fixes(some_fallback_policy_nums=True)
    assert (analysis_df[_POLICY_NUM_COL_FIXES] < 0).any()

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_returns_early_when_all_fallback():
    # Documented edge case: every active row used a fallback policy, so there is no
    # sequence to check. Must return quietly rather than call .min() on an empty array.
    analysis_df = _build_study_fixes(all_active_rows_fallback=True)
    assert (
        analysis_df.loc[analysis_df[_ACTIVE_COL_FIXES] == 1, _POLICY_NUM_COL_FIXES] < 0
    ).all()

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_passes_on_empty_analysis_df():
    # No active rows at all -> nothing to check, and no crash.
    analysis_df = _build_study_fixes(empty=True)

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_ignores_out_of_study_rows():
    # Policy numbers are only meaningful in study, so a wild value on an out-of-study
    # row must not manufacture a gap.
    analysis_df = _build_study_fixes(out_of_study_row_wild_policy_num=True)

    input_checks.require_consecutive_integer_policy_numbers(
        analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
    )


def test_require_consecutive_integer_policy_numbers_raises_on_gapped_sequence():
    # A genuinely gapped sequence {1, 2, 5} must be rejected, and the message must
    # report the sorted values actually present.
    analysis_df = _build_study_fixes(gapped_policy_nums=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Policy numbers are not consecutive integers. Non-fallback, in-study "
            "policy_num values present: [1, 2, 5]."
        ),
    ):
        input_checks.require_consecutive_integer_policy_numbers(
            analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
        )


def test_require_consecutive_integer_policy_numbers_raises_on_fractional_policy_number():
    # REGRESSION: the old .astype("int64") truncated 1.5 -> 1, so a fractional policy
    # number masqueraded as the perfectly consecutive 1. The frame is built so the
    # truncated values are {1, 2, 3} in sorted order of appearance -- exactly what the
    # old check compared against range(1, 4), so it passed.
    analysis_df = _build_study_fixes(fractional_policy_num=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Non-fallback, in-study policy_num contains non-integer values: [1.5]."
        ),
    ):
        input_checks.require_consecutive_integer_policy_numbers(
            analysis_df, _ACTIVE_COL_FIXES, _POLICY_NUM_COL_FIXES
        )


# ---------------------------------------------------------------------------
# require_consecutive_integer_calendar_times
# ---------------------------------------------------------------------------


def test_require_consecutive_integer_calendar_times_passes_on_valid_study():
    analysis_df = _build_study_fixes()

    input_checks.require_consecutive_integer_calendar_times(
        analysis_df, _CALENDAR_T_COL_FIXES
    )


def test_require_consecutive_integer_calendar_times_passes_on_float_dtype_column():
    # float64 calendar times with integral values go through the cast branch and pass.
    analysis_df = _build_study_fixes(float_calendar_times=True)
    assert analysis_df[_CALENDAR_T_COL_FIXES].dtype == "float64"

    input_checks.require_consecutive_integer_calendar_times(
        analysis_df, _CALENDAR_T_COL_FIXES
    )


def test_require_consecutive_integer_calendar_times_passes_on_shuffled_row_order():
    # REGRESSION: Series.unique() on this frame yields [3, 1, 2, 0], which the old
    # implementation compared to range(0, 4) and spuriously rejected purely because
    # analysis_df was not sorted by calendar_t.
    analysis_df = _build_study_fixes(shuffle_rows=True)
    assert list(analysis_df[_CALENDAR_T_COL_FIXES].unique()) == [3, 1, 2, 0]

    input_checks.require_consecutive_integer_calendar_times(
        analysis_df, _CALENDAR_T_COL_FIXES
    )


def test_require_consecutive_integer_calendar_times_passes_on_empty_analysis_df():
    # Documented edge case: an empty frame returns early, before .min() on no values.
    analysis_df = _build_study_fixes(empty=True)

    input_checks.require_consecutive_integer_calendar_times(
        analysis_df, _CALENDAR_T_COL_FIXES
    )


def test_require_consecutive_integer_calendar_times_raises_on_gapped_sequence():
    # {0, 1, 3, 5} skips 2 and 4 -- a real gap in the study timeline.
    analysis_df = _build_study_fixes(gapped_calendar_times=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Calendar times are not consecutive integers. calendar_t values "
            "present: [0, 1, 3, 5]."
        ),
    ):
        input_checks.require_consecutive_integer_calendar_times(
            analysis_df, _CALENDAR_T_COL_FIXES
        )


def test_require_consecutive_integer_calendar_times_raises_on_fractional_time():
    # REGRESSION: 1.5 used to be truncated to 1 by .astype("int64"). Only one
    # subject's t=2 row is fractional here, so the truncated times were still
    # {0, 1, 2, 3} in order of appearance and the old check passed outright.
    analysis_df = _build_study_fixes(fractional_calendar_time=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("calendar_t contains non-integer values: [1.5]."),
    ):
        input_checks.require_consecutive_integer_calendar_times(
            analysis_df, _CALENDAR_T_COL_FIXES
        )


def test_require_consecutive_integer_calendar_times_includes_out_of_study_rows():
    # Deliberately unlike the policy-number check: calendar times must be well formed
    # even on rows where the subject is not in the study, so this check does NOT filter
    # by the active column.
    analysis_df = _build_study_fixes(out_of_study_row_gaps_calendar_times=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("calendar_t values present: [0, 1, 2, 3, 9]."),
    ):
        input_checks.require_consecutive_integer_calendar_times(
            analysis_df, _CALENDAR_T_COL_FIXES
        )


# ---------------------------------------------------------------------------
# require_hashable_subject_ids
# ---------------------------------------------------------------------------


def test_require_hashable_subject_ids_passes_on_string_ids():
    analysis_df = _build_study_fixes()

    input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
    )


def test_require_hashable_subject_ids_passes_on_integer_ids():
    # int64 subject ids are hashable too.
    analysis_df = _build_study_fixes(integer_subject_ids=True)

    input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
    )


def test_require_hashable_subject_ids_passes_when_row_labeled_zero_is_not_active():
    # REGRESSION: the old implementation sampled a single id with label-based [0], which
    # raised KeyError whenever the row labeled 0 was not active.
    analysis_df = _build_study_fixes(row_labeled_zero_out_of_study=True)
    assert analysis_df.at[0, _ACTIVE_COL_FIXES] == 0

    input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
    )


def test_require_hashable_subject_ids_passes_on_shuffled_index_labels():
    # Row order must not matter here either: every active id is inspected, not one
    # positionally- or label-sampled row.
    analysis_df = _build_study_fixes(shuffle_rows=True)
    assert analysis_df.index.tolist() != sorted(analysis_df.index.tolist())

    input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
    )


def test_require_hashable_subject_ids_raises_on_unhashable_id_even_with_no_active_rows():
    # INVERTED on 2026-09-02, when this check's scope widened from the active rows to the WHOLE
    # subject-id column. It previously asserted that nothing active meant nothing to validate.
    # But every consumer of subject ids hashes the whole column -- .unique(), groupby(), and
    # duplicated(subset=[subject_id, calendar_t]) -- so an unhashable id breaks them regardless
    # of whether its rows are active, and the active-rows-only scope let it through.
    analysis_df = _build_study_fixes(
        all_rows_out_of_study=True, list_subject_id_on_active_row=True
    )

    with pytest.raises(AssertionError, match="must be hashable"):
        input_checks.require_hashable_subject_ids(
            analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
        )


def test_require_hashable_subject_ids_passes_on_an_all_inactive_but_well_formed_column():
    # The other half of the widened scope: no active rows is still fine when the ids themselves
    # are well formed, so the check has not simply become "reject inactive studies".
    analysis_df = _build_study_fixes(all_rows_out_of_study=True)

    input_checks.require_hashable_subject_ids(
        analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
    )


def test_require_hashable_subject_ids_raises_on_list_valued_active_id():
    # REGRESSION: the old implementation called isinstance(...) and threw the result
    # away, so it enforced nothing at all and an unhashable list id passed.
    analysis_df = _build_study_fixes(list_subject_id_on_active_row=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Subject IDs must be hashable; found 1 unhashable value(s), e.g. [['s1']]."
        ),
    ):
        input_checks.require_hashable_subject_ids(
            analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
        )


def test_require_hashable_subject_ids_raises_on_set_valued_active_id():
    # A set is the other common unhashable id. The check must report it rather than let
    # pandas raise a bare TypeError from an internal hash (hence no .unique() call).
    analysis_df = _build_study_fixes(set_subject_id_on_active_row=True)

    with pytest.raises(AssertionError, match=re.escape("found 1 unhashable value(s)")):
        input_checks.require_hashable_subject_ids(
            analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
        )


def test_require_hashable_subject_ids_raises_on_unhashable_id_on_out_of_study_row():
    # INVERTED on 2026-09-02 along with the scope widening described above. The old comment here
    # reasoned that only ACTIVE subject ids key the per-subject argument dictionaries -- true,
    # but incomplete: pandas hashes the whole column in duplicated()/groupby()/unique(), so an
    # unhashable id on an out-of-study row still raises a bare "TypeError: unhashable type:
    # 'list'" from one of the checks that run before this one.
    analysis_df = _build_study_fixes(list_subject_id_on_out_of_study_row=True)

    with pytest.raises(AssertionError, match="must be hashable"):
        input_checks.require_hashable_subject_ids(
            analysis_df, _ACTIVE_COL_FIXES, _SUBJECT_ID_COL_FIXES
        )
