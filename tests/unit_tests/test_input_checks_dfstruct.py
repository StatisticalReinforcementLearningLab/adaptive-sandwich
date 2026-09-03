"""
Coverage for the analysis_df structural checks and the theta shape check in
lifejacket.input_checks:

  - require_all_subjects_have_all_times_in_analysis_df
  - require_all_named_columns_present_in_analysis_df
  - require_all_named_columns_not_object_type_in_analysis_df
  - require_theta_is_1D_array

Both column checks gained a reward_col_name parameter (their 8th, positional) on 2026-09-02, and
the presence check now reports EVERY missing column in one combined message instead of raising
one assert per column, so the expected text is built by
_expected_missing_columns_message_dfstruct below.

The builder below mirrors the real data shape (tests/benchmarks/fixtures/small): policy_num is
float64 whose INITIAL value is 1.0 (not 0.0) and which holds NaN on out-of-study rows, the
in-study indicator is int64 0/1, and the action/action-probability/reward columns are float64
holding NaN while a subject is out of study.
"""

import re

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

_ACTIVE_COL_DFSTRUCT = "in_study"
_ACTION_COL_DFSTRUCT = "action"
_POLICY_NUM_COL_DFSTRUCT = "policy_num"
_CALENDAR_T_COL_DFSTRUCT = "calendar_t"
_SUBJECT_ID_COL_DFSTRUCT = "user_id"
_ACTION_PROB_COL_DFSTRUCT = "action1prob"
_REWARD_COL_DFSTRUCT = "reward"

# The columns require_all_named_columns_not_object_type_in_analysis_df actually inspects, in the
# order it iterates them.
# _SUBJECT_ID_COL_DFSTRUCT is deliberately absent: it was exempted on 2026-09-02 so that string
# subject ids are allowed, and its contract moved to require_hashable_subject_ids. Every column
# below is consumed numerically, so object dtype genuinely breaks it.
# _REWARD_COL_DFSTRUCT was ADDED on 2026-09-02, when the check gained its reward_col_name
# parameter: the reward column is now dtype-inspected too, appended at the END of the tuple.
_CHECKED_COLS_DFSTRUCT = (
    _ACTIVE_COL_DFSTRUCT,
    _ACTION_COL_DFSTRUCT,
    _POLICY_NUM_COL_DFSTRUCT,
    _CALENDAR_T_COL_DFSTRUCT,
    _ACTION_PROB_COL_DFSTRUCT,
    _REWARD_COL_DFSTRUCT,
)

# The columns require_all_named_columns_present_in_analysis_df demands, in the order it declares
# them as parameters -- which is also the order its single combined message lists the missing
# ones in. A SUPERSET of the dtype-inspected tuple above: presence is required of the subject-id
# column as well, only its dtype is left unconstrained.
_PRESENCE_COLS_DFSTRUCT = (
    _ACTIVE_COL_DFSTRUCT,
    _ACTION_COL_DFSTRUCT,
    _POLICY_NUM_COL_DFSTRUCT,
    _CALENDAR_T_COL_DFSTRUCT,
    _SUBJECT_ID_COL_DFSTRUCT,
    _ACTION_PROB_COL_DFSTRUCT,
    _REWARD_COL_DFSTRUCT,
)

_SUBJECT_IDS_DFSTRUCT = (1, 2, 3)
_CALENDAR_TIMES_DFSTRUCT = (0, 1, 2)


def _build_analysis_df_dfstruct(
    *,
    omit_subject_time=None,
    duplicate_subject_time=None,
    extra_time_for_subject=None,
    object_dtype_columns=(),
    drop_columns=(),
    string_subject_ids=False,
    pandas_string_subject_ids=False,
    float_calendar_times=False,
):
    """
    A small valid analysis_df: 3 subjects (ids 1, 2, 3) x 3 calendar times (0, 1, 2), a
    complete rectangular grid. Subject 3 is out of study at t=2, and that row carries
    in_study=0 with NaN action/action_prob/policy_num/reward -- i.e. the out-of-study rows are
    PRESENT in the frame, which is what require_all_subjects_have_all_times... needs, since it
    never filters on the active column.

    Keyword flags each introduce exactly one defect:
      omit_subject_time=(subject, t)      drop that row entirely
      duplicate_subject_time=(subject, t) append a second copy of that row
      extra_time_for_subject=subject      give that ONE subject a t=3 row nobody else has
      object_dtype_columns=(name, ...)    cast those columns to object dtype
      drop_columns=(name, ...)            drop those columns
      string_subject_ids=True             user_id as plain Python str (-> object dtype)
      pandas_string_subject_ids=True      user_id as pandas "string" dtype
      float_calendar_times=True           calendar_t as float64 rather than int64
    """
    rows = []
    for subject_id in _SUBJECT_IDS_DFSTRUCT:
        for calendar_t in _CALENDAR_TIMES_DFSTRUCT:
            if omit_subject_time == (subject_id, calendar_t):
                continue
            rows.append(_row_dfstruct(subject_id, calendar_t))

    if duplicate_subject_time is not None:
        rows.append(_row_dfstruct(*duplicate_subject_time))
    if extra_time_for_subject is not None:
        rows.append(
            _row_dfstruct(extra_time_for_subject, max(_CALENDAR_TIMES_DFSTRUCT) + 1)
        )

    analysis_df = pd.DataFrame(rows)
    analysis_df = analysis_df.astype(
        {
            _SUBJECT_ID_COL_DFSTRUCT: "int64",
            _CALENDAR_T_COL_DFSTRUCT: "int64",
            _ACTIVE_COL_DFSTRUCT: "int64",
            _POLICY_NUM_COL_DFSTRUCT: "float64",
            _ACTION_COL_DFSTRUCT: "float64",
            _ACTION_PROB_COL_DFSTRUCT: "float64",
            _REWARD_COL_DFSTRUCT: "float64",
        }
    )

    if string_subject_ids:
        analysis_df[_SUBJECT_ID_COL_DFSTRUCT] = [
            f"user_{subject_id}" for subject_id in analysis_df[_SUBJECT_ID_COL_DFSTRUCT]
        ]
    if pandas_string_subject_ids:
        analysis_df[_SUBJECT_ID_COL_DFSTRUCT] = pd.array(
            [
                f"user_{subject_id}"
                for subject_id in analysis_df[_SUBJECT_ID_COL_DFSTRUCT]
            ],
            dtype="string",
        )
    if float_calendar_times:
        analysis_df[_CALENDAR_T_COL_DFSTRUCT] = analysis_df[
            _CALENDAR_T_COL_DFSTRUCT
        ].astype("float64")
    for colname in object_dtype_columns:
        analysis_df[colname] = analysis_df[colname].astype(object)
    if drop_columns:
        analysis_df = analysis_df.drop(columns=list(drop_columns))

    return analysis_df


def _row_dfstruct(subject_id, calendar_t):
    """One analysis_df row. Subject 3 at t=2 is the single out-of-study cell."""
    in_study = 0 if (subject_id == 3 and calendar_t == 2) else 1
    if not in_study:
        return {
            _SUBJECT_ID_COL_DFSTRUCT: subject_id,
            _CALENDAR_T_COL_DFSTRUCT: calendar_t,
            # Real studies leave policy_num NaN on out-of-study rows; keeping the column
            # float64 (never object) is exactly what the dtype check cares about.
            _POLICY_NUM_COL_DFSTRUCT: np.nan,
            _ACTIVE_COL_DFSTRUCT: 0,
            _ACTION_COL_DFSTRUCT: np.nan,
            _ACTION_PROB_COL_DFSTRUCT: np.nan,
            _REWARD_COL_DFSTRUCT: np.nan,
        }
    return {
        _SUBJECT_ID_COL_DFSTRUCT: subject_id,
        _CALENDAR_T_COL_DFSTRUCT: calendar_t,
        # Initial policy number is 1, not 0.
        _POLICY_NUM_COL_DFSTRUCT: float(calendar_t + 1),
        _ACTIVE_COL_DFSTRUCT: 1,
        _ACTION_COL_DFSTRUCT: float(calendar_t % 2),
        _ACTION_PROB_COL_DFSTRUCT: 0.3 + 0.1 * calendar_t,
        _REWARD_COL_DFSTRUCT: float(subject_id),
    }


def _call_all_times_check_dfstruct(analysis_df):
    input_checks.require_all_subjects_have_all_times_in_analysis_df(
        analysis_df, _CALENDAR_T_COL_DFSTRUCT, _SUBJECT_ID_COL_DFSTRUCT
    )


def _column_names_dfstruct(**overrides):
    """
    The builder's column names, keyed by the parameter each is passed to. reward_col_name is the
    8th (and last) positional parameter of both column checks as of 2026-09-02.
    """
    names = {
        "active_col_name": _ACTIVE_COL_DFSTRUCT,
        "action_col_name": _ACTION_COL_DFSTRUCT,
        "policy_num_col_name": _POLICY_NUM_COL_DFSTRUCT,
        "calendar_t_col_name": _CALENDAR_T_COL_DFSTRUCT,
        "subject_id_col_name": _SUBJECT_ID_COL_DFSTRUCT,
        "action_prob_col_name": _ACTION_PROB_COL_DFSTRUCT,
        "reward_col_name": _REWARD_COL_DFSTRUCT,
    }
    names.update(overrides)
    return names


def _call_columns_present_check_dfstruct(analysis_df, **overrides):
    """
    Calls the presence check with the builder's column names, positionally, in the order the
    function declares them. `overrides` swaps in a different name for one parameter.
    """
    names = _column_names_dfstruct(**overrides)
    input_checks.require_all_named_columns_present_in_analysis_df(
        analysis_df,
        names["active_col_name"],
        names["action_col_name"],
        names["policy_num_col_name"],
        names["calendar_t_col_name"],
        names["subject_id_col_name"],
        names["action_prob_col_name"],
        names["reward_col_name"],
    )


def _call_columns_not_object_check_dfstruct(analysis_df, **overrides):
    """Same, for the object-dtype check (identical parameter list and order)."""
    names = _column_names_dfstruct(**overrides)
    input_checks.require_all_named_columns_not_object_type_in_analysis_df(
        analysis_df,
        names["active_col_name"],
        names["action_col_name"],
        names["policy_num_col_name"],
        names["calendar_t_col_name"],
        names["subject_id_col_name"],
        names["action_prob_col_name"],
        names["reward_col_name"],
    )


def _expected_object_dtype_message_dfstruct(object_col_names):
    """
    The leading, column-naming half of the object-dtype check's message. Rebuilt from the same
    list the implementation reports so these tests pin the CONTENT, not the prose around it.
    """
    return (
        "These analysis DataFrame columns are of object type, but are consumed numerically: "
        f"{list(object_col_names)}"
    )


def _expected_missing_columns_message_dfstruct(analysis_df, missing_col_names):
    """
    The presence check's one combined message, rebuilt from its two halves.

    Until 2026-09-02 the check raised one assert per column, with the message
    "{col_name} not in analysis DataFrame."; it now collects every missing name and reports them
    together with the columns that ARE present, so every expectation in this file is built here
    rather than hand-written per test. The "columns present" half is computed into a local rather
    than called inline in the f-string, which ruff's formatter would be free to split across
    lines -- a SyntaxError on Python 3.10.
    """
    columns_present = sorted(map(str, analysis_df.columns))
    return (
        f"These named columns are not in the analysis DataFrame: {list(missing_col_names)}. "
        f"Columns present: {columns_present}."
    )


### require_all_subjects_have_all_times_in_analysis_df


def test_require_all_subjects_have_all_times_passes_on_complete_grid_dfstruct():
    # The happy path, and specifically that a subject who LEAVES the study mid-way still
    # passes as long as its out-of-study rows are present: this check never filters on the
    # active column, so presence -- not activity -- is what it demands.
    _call_all_times_check_dfstruct(_build_analysis_df_dfstruct())


def test_require_all_subjects_have_all_times_raises_when_a_subject_is_missing_a_time_dfstruct():
    # Subject 3's t=2 row is dropped instead of being marked out of study -- the most common
    # real shape of this defect (rows deleted rather than flagged inactive).
    analysis_df = _build_analysis_df_dfstruct(omit_subject_time=(3, 2))

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Not all subjects have all calendar times in the analysis DataFrame."
        ),
    ):
        _call_all_times_check_dfstruct(analysis_df)


def test_require_all_subjects_have_all_times_raises_when_one_subject_has_an_extra_time_dfstruct():
    # A time only ONE subject has still fails, because the reference set is the union over the
    # whole frame: subject 1 gets a t=3 row, so subjects 2 and 3 are now short a time.
    analysis_df = _build_analysis_df_dfstruct(extra_time_for_subject=1)

    with pytest.raises(
        AssertionError, match=re.escape("Not all subjects have all calendar times")
    ):
        _call_all_times_check_dfstruct(analysis_df)


def test_require_all_subjects_have_all_times_raises_for_a_single_missing_middle_time_dfstruct():
    # An interior hole (subject 2 missing t=1) rather than a truncated tail. The check compares
    # SETS, so it catches gaps a "same number of rows per subject" heuristic would miss --
    # here subject 2 also gets a duplicate t=0 row, so its ROW COUNT still matches everyone
    # else's while its time set does not.
    analysis_df = _build_analysis_df_dfstruct(
        omit_subject_time=(2, 1), duplicate_subject_time=(2, 0)
    )
    assert analysis_df.groupby(_SUBJECT_ID_COL_DFSTRUCT).size().nunique() == 1, (
        "precondition: every subject has the same number of rows"
    )

    with pytest.raises(
        AssertionError, match=re.escape("Not all subjects have all calendar times")
    ):
        _call_all_times_check_dfstruct(analysis_df)


def test_require_all_subjects_have_all_times_passes_on_duplicate_rows_dfstruct():
    # Documents a real blind spot rather than an aspiration: because the comparison is on sets
    # of times, a duplicated (subject, time) row passes this check untouched. Duplicate active
    # cells are rejected elsewhere (require_action_probabilities_..._can_be_reconstructed), not
    # here.
    _call_all_times_check_dfstruct(
        _build_analysis_df_dfstruct(duplicate_subject_time=(1, 0))
    )


def test_require_all_subjects_have_all_times_passes_on_float_calendar_times_dfstruct():
    # float64 calendar times (0.0, 1.0, 2.0) compare equal inside the per-subject sets, so the
    # check is dtype-agnostic between int64 and float64 time columns.
    _call_all_times_check_dfstruct(
        _build_analysis_df_dfstruct(float_calendar_times=True)
    )


def test_require_all_subjects_have_all_times_passes_on_empty_frame_dfstruct():
    # Edge case: with no rows there are no subjects to disagree, so the check is vacuously
    # satisfied instead of raising. Pinned so that a future rewrite (e.g. one that indexes
    # into the groupby result) does not start raising on an empty frame by accident.
    empty_df = _build_analysis_df_dfstruct().iloc[0:0]
    assert empty_df.empty

    _call_all_times_check_dfstruct(empty_df)


def test_require_all_subjects_have_all_times_passes_for_a_single_subject_dfstruct():
    # One subject can never disagree with the union of its own times.
    single_subject_df = _build_analysis_df_dfstruct()
    single_subject_df = single_subject_df[
        single_subject_df[_SUBJECT_ID_COL_DFSTRUCT] == 1
    ]

    _call_all_times_check_dfstruct(single_subject_df)


### require_all_named_columns_present_in_analysis_df


def test_require_all_named_columns_present_passes_on_valid_frame_dfstruct():
    _call_columns_present_check_dfstruct(_build_analysis_df_dfstruct())


@pytest.mark.parametrize("missing_col", _PRESENCE_COLS_DFSTRUCT)
def test_require_all_named_columns_present_raises_for_each_missing_column_dfstruct(
    missing_col,
):
    # One case per column the check requires -- now all SEVEN, reward included -- so that
    # dropping any single name from its tuple shows up as a failure here rather than as a
    # KeyError from deep in the estimator. The message must name the column that is missing.
    # Retargeted 2026-09-02: this used to expect the per-column message
    # "{missing_col} not in analysis DataFrame.", and parametrized only over the dtype-inspected
    # columns (which excluded subject_id, and reward, which nothing checked at all).
    analysis_df = _build_analysis_df_dfstruct(drop_columns=(missing_col,))
    expected_message = _expected_missing_columns_message_dfstruct(
        analysis_df, [missing_col]
    )

    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _call_columns_present_check_dfstruct(analysis_df)


def test_require_all_named_columns_present_reports_every_missing_column_at_once_dfstruct():
    # The point of the 2026-09-02 message change: the check used to raise on the FIRST missing
    # column it reached (one assert per column), so a caller who had mis-wired two names had to
    # fix one and re-run to discover the other. Both names must now appear in a single message,
    # in the check's own parameter order (action before reward), alongside the columns present.
    analysis_df = _build_analysis_df_dfstruct(
        drop_columns=(_ACTION_COL_DFSTRUCT, _REWARD_COL_DFSTRUCT)
    )
    expected_message = _expected_missing_columns_message_dfstruct(
        analysis_df, [_ACTION_COL_DFSTRUCT, _REWARD_COL_DFSTRUCT]
    )

    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _call_columns_present_check_dfstruct(analysis_df)


def test_require_all_named_columns_present_raises_for_a_misspelled_column_name_dfstruct():
    # The failure the check exists for in practice is a mis-typed *argument*, not a missing
    # column: the frame is complete but the caller named a column that is not in it.
    # The "Columns present:" half of the message added on 2026-09-02 is what makes this
    # failure self-service -- the real column name (action1prob) is listed right there.
    analysis_df = _build_analysis_df_dfstruct()
    expected_message = _expected_missing_columns_message_dfstruct(
        analysis_df, ["action_1_prob"]
    )
    assert _ACTION_PROB_COL_DFSTRUCT in expected_message

    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _call_columns_present_check_dfstruct(
            analysis_df, action_prob_col_name="action_1_prob"
        )


def test_require_all_named_columns_present_now_checks_the_reward_column_dfstruct():
    # INVERTED on 2026-09-02. This test previously pinned the opposite -- that reward_col_name
    # was not a parameter of EITHER column check, so a frame with no reward column at all sailed
    # through both -- and flagged that as a regression, because reward_col_name is a required
    # argument of analyze_dataset and is read by verify_analysis_df_summary_satisfactory: a typo
    # in it surfaced as a bare KeyError from inside a plotting routine, naming nothing. Both
    # checks now take reward_col_name as their 8th positional parameter, so a missing reward
    # column is reported here by name.
    analysis_df = _build_analysis_df_dfstruct(drop_columns=(_REWARD_COL_DFSTRUCT,))
    assert _REWARD_COL_DFSTRUCT not in analysis_df.columns
    expected_message = _expected_missing_columns_message_dfstruct(
        analysis_df, [_REWARD_COL_DFSTRUCT]
    )

    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _call_columns_present_check_dfstruct(analysis_df)

    # And the dtype check, which indexes the frame directly, now reaches the reward column too,
    # so on this same frame it raises a bare KeyError -- one more reason the presence check runs
    # first (and, since 2026-09-02, runs at the very top of perform_first_wave_input_checks).
    with pytest.raises(KeyError, match=re.escape(_REWARD_COL_DFSTRUCT)):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_present_catches_a_misspelled_reward_column_dfstruct():
    # The regression the reward parameter was added for, in the shape it actually took: the
    # frame HAS a reward column, but the caller mis-typed reward_col_name. Until 2026-09-02
    # neither column check looked at reward_col_name at all, so the typo travelled all the way
    # to verify_analysis_df_summary_satisfactory and came out as a bare KeyError from inside a
    # plotting routine, after the rest of the first wave had already run.
    analysis_df = _build_analysis_df_dfstruct()
    assert _REWARD_COL_DFSTRUCT in analysis_df.columns
    expected_message = _expected_missing_columns_message_dfstruct(
        analysis_df, ["rewards"]
    )

    with pytest.raises(AssertionError, match=re.escape(expected_message)):
        _call_columns_present_check_dfstruct(analysis_df, reward_col_name="rewards")


def test_require_all_named_columns_present_passes_on_empty_frame_with_the_columns_dfstruct():
    # Presence is about columns, not rows: a zero-row frame that still declares the seven
    # required columns (reward included as of 2026-09-02) passes.
    _call_columns_present_check_dfstruct(_build_analysis_df_dfstruct().iloc[0:0])


### require_all_named_columns_not_object_type_in_analysis_df


def test_require_all_named_columns_not_object_type_passes_on_real_world_dtypes_dfstruct():
    # int64 in_study, int64 user_id/calendar_t, float64 action/action1prob/reward and a
    # float64 policy_num that starts at 1.0 and holds NaN on the out-of-study row -- the fixture
    # shape. NaNs do not make a numeric column object-typed, so this passes.
    analysis_df = _build_analysis_df_dfstruct()
    assert analysis_df[_POLICY_NUM_COL_DFSTRUCT].isna().any()
    assert analysis_df[_ACTION_COL_DFSTRUCT].isna().any()
    assert analysis_df[_REWARD_COL_DFSTRUCT].isna().any()

    _call_columns_not_object_check_dfstruct(analysis_df)


@pytest.mark.parametrize("object_col", _CHECKED_COLS_DFSTRUCT)
def test_require_all_named_columns_not_object_type_raises_for_each_object_column_dfstruct(
    object_col,
):
    # One case per column in the loop: every one of the six inspected columns -- reward
    # included since 2026-09-02 -- is actually looked at, and the message names the offending
    # one. subject_id is not in this tuple; its exemption is pinned by the string-id tests below.
    analysis_df = _build_analysis_df_dfstruct(object_dtype_columns=(object_col,))
    assert analysis_df[object_col].dtype == object

    with pytest.raises(
        AssertionError,
        match=re.escape(_expected_object_dtype_message_dfstruct([object_col])),
    ):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_reports_columns_in_parameter_order_dfstruct():
    # RETARGETED on 2026-09-02: the check now reports EVERY object-typed column in one message,
    # rather than asserting inside its loop and naming only the first (the old "At least ..."
    # hedge). This test used to pin that only `action` was named; what still matters, and is
    # still pinned, is the ORDER -- the report follows the CHECK's parameter order (active,
    # action, policy_num, calendar_t, action_prob, reward; no subject_id since 2026-09-02), NOT
    # the frame's column order. The frame deliberately places calendar_t before action, yet
    # action is reported first.
    analysis_df = _build_analysis_df_dfstruct(
        object_dtype_columns=(_CALENDAR_T_COL_DFSTRUCT, _ACTION_COL_DFSTRUCT)
    )
    frame_order = list(analysis_df.columns)
    assert frame_order.index(_CALENDAR_T_COL_DFSTRUCT) < frame_order.index(
        _ACTION_COL_DFSTRUCT
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            _expected_object_dtype_message_dfstruct(
                [_ACTION_COL_DFSTRUCT, _CALENDAR_T_COL_DFSTRUCT]
            )
        ),
    ):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_now_inspects_the_reward_column_dfstruct():
    # The dtype half of the same 2026-09-02 fix, called out explicitly because it closes a
    # REGRESSION: the reward column was checked neither for presence nor for dtype, yet it is
    # consumed numerically (verify_analysis_df_summary_satisfactory's summary statistics, and the
    # inference-func argument built from the column of that name), so an object-typed reward
    # column used to sail straight through the first wave.
    analysis_df = _build_analysis_df_dfstruct(
        object_dtype_columns=(_REWARD_COL_DFSTRUCT,)
    )
    assert analysis_df[_REWARD_COL_DFSTRUCT].dtype == object

    with pytest.raises(
        AssertionError,
        match=re.escape(
            _expected_object_dtype_message_dfstruct([_REWARD_COL_DFSTRUCT])
        ),
    ):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_reports_reward_last_dfstruct():
    # RETARGETED on 2026-09-02, when the check began reporting every offender at once. It used
    # to pin that reward, appended to the END of the iteration tuple, was NOT the column named
    # when an earlier column was also object-typed. Now both are named -- and reward still
    # comes last, which is the part worth keeping: the reported order is the check's parameter
    # order, so it stays a deliberate choice rather than an accident of how the new parameter
    # was slotted in.
    analysis_df = _build_analysis_df_dfstruct(
        object_dtype_columns=(_REWARD_COL_DFSTRUCT, _ACTION_PROB_COL_DFSTRUCT)
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            _expected_object_dtype_message_dfstruct(
                [_ACTION_PROB_COL_DFSTRUCT, _REWARD_COL_DFSTRUCT]
            )
        ),
    ):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_accepts_plain_str_subject_ids_dfstruct():
    # INVERTED on 2026-09-02. This test previously pinned the opposite: a plain-str subject_id
    # column is object-typed in pandas, so this check rejected it, which meant no study could
    # use string subject ids at all -- even though require_hashable_subject_ids accepts any
    # hashable id. The subject-id column is now exempt from this check (its contract lives in
    # require_hashable_subject_ids), and analyze_dataset was verified end to end to return
    # bit-identical estimates under string ids.
    analysis_df = _build_analysis_df_dfstruct(string_subject_ids=True)
    assert analysis_df[_SUBJECT_ID_COL_DFSTRUCT].dtype == object

    _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_ignores_unhashable_subject_ids_dfstruct():
    # The exemption is total: this check no longer looks at the subject-id column at all, so
    # even an unhashable id passes HERE. That is not a hole -- require_hashable_subject_ids is
    # what rejects it (see test_input_checks_output_and_subject_ids.py) -- but it is worth
    # pinning, because this check used to be what caught unhashable ids incidentally.
    analysis_df = _build_analysis_df_dfstruct()
    analysis_df[_SUBJECT_ID_COL_DFSTRUCT] = pd.Series(
        [[i] for i in range(len(analysis_df))], dtype=object
    )

    _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_accepts_pandas_string_subject_ids_dfstruct():
    # The same string ids stored in pandas' extension "string" dtype pass, because the check
    # tests dtype != "object" rather than "is numeric". Documented so the asymmetry with the
    # plain-str case above is visible.
    analysis_df = _build_analysis_df_dfstruct(pandas_string_subject_ids=True)
    assert analysis_df[_SUBJECT_ID_COL_DFSTRUCT].dtype != object

    _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_raises_key_error_for_absent_column_dfstruct():
    # This check indexes the frame directly, so an absent column surfaces as a bare KeyError,
    # NOT as the friendly assertion from the presence check. That is why
    # perform_first_wave_input_checks calls the presence check first; calling this one alone on
    # an incomplete frame gives an unhelpful error.
    analysis_df = _build_analysis_df_dfstruct(drop_columns=(_ACTION_PROB_COL_DFSTRUCT,))

    with pytest.raises(KeyError, match=re.escape(_ACTION_PROB_COL_DFSTRUCT)):
        _call_columns_not_object_check_dfstruct(analysis_df)


def test_require_all_named_columns_not_object_type_raises_on_default_dtype_empty_frame_dfstruct():
    # Edge case worth knowing: an empty frame built by naming columns only gets object dtype
    # for every column, so it FAILS this check (while passing the presence check). Empty
    # analysis frames therefore have to declare dtypes to get through the first wave.
    # All SEVEN required columns, not just the inspected ones: the presence check below wants
    # the subject-id column too, even though the dtype check no longer inspects it.
    empty_df = pd.DataFrame(columns=list(_PRESENCE_COLS_DFSTRUCT))
    assert all(empty_df[col].dtype == object for col in _CHECKED_COLS_DFSTRUCT)

    _call_columns_present_check_dfstruct(empty_df)
    with pytest.raises(
        AssertionError,
        match=re.escape(
            _expected_object_dtype_message_dfstruct(
                [
                    _ACTIVE_COL_DFSTRUCT,
                    _ACTION_COL_DFSTRUCT,
                    _POLICY_NUM_COL_DFSTRUCT,
                    _CALENDAR_T_COL_DFSTRUCT,
                    _ACTION_PROB_COL_DFSTRUCT,
                    _REWARD_COL_DFSTRUCT,
                ]
            )
        ),
    ):
        _call_columns_not_object_check_dfstruct(empty_df)


def test_require_all_named_columns_not_object_type_passes_on_typed_empty_frame_dfstruct():
    # The other side of the previous test: a zero-row frame that keeps its int64/float64
    # dtypes passes.
    _call_columns_not_object_check_dfstruct(_build_analysis_df_dfstruct().iloc[0:0])


### require_theta_is_1D_array


def test_require_theta_is_1D_array_passes_on_1d_jnp_array_dfstruct():
    input_checks.require_theta_is_1D_array(jnp.array([0.1, -0.2, 3.0]))


def test_require_theta_is_1D_array_passes_on_1d_numpy_array_dfstruct():
    # numpy and jax arrays both expose .ndim, and the check does not care which it got.
    input_checks.require_theta_is_1D_array(np.array([0.1, -0.2, 3.0]))


def test_require_theta_is_1D_array_passes_on_length_one_array_dfstruct():
    # A single-parameter theta is 1D, not scalar; it must not be confused with the 0-d case.
    input_checks.require_theta_is_1D_array(jnp.array([2.5]))


def test_require_theta_is_1D_array_passes_on_empty_array_dfstruct():
    # Edge case: an empty 1D theta satisfies this check. Nothing here enforces a non-zero
    # parameter count, so a mis-wired theta of length 0 gets past the first wave.
    theta_est = jnp.array([])
    assert theta_est.ndim == 1 and theta_est.size == 0

    input_checks.require_theta_is_1D_array(theta_est)


def test_require_theta_is_1D_array_raises_on_column_vector_dfstruct():
    # The realistic defect: theta estimated as a (k, 1) column vector rather than flattened.
    # Broadcasting would otherwise silently turn downstream (k,) arithmetic into (k, k).
    with pytest.raises(AssertionError, match=re.escape("Theta is not a 1D array.")):
        input_checks.require_theta_is_1D_array(jnp.array([[0.1], [-0.2], [3.0]]))


def test_require_theta_is_1D_array_raises_on_row_vector_dfstruct():
    # A (1, k) row vector is just as wrong as (k, 1), and its .size matches a valid theta's,
    # so only the ndim test catches it.
    with pytest.raises(AssertionError, match=re.escape("Theta is not a 1D array.")):
        input_checks.require_theta_is_1D_array(jnp.array([[0.1, -0.2, 3.0]]))


def test_require_theta_is_1D_array_raises_on_0d_scalar_array_dfstruct():
    # A 0-d array (a scalar theta from a 1-parameter model that was not wrapped) has ndim 0.
    with pytest.raises(AssertionError, match=re.escape("Theta is not a 1D array.")):
        input_checks.require_theta_is_1D_array(jnp.array(2.5))


def test_require_theta_is_1D_array_raises_attribute_error_on_python_list_dfstruct():
    # Documents current behavior: a plain list has no .ndim, so this check raises
    # AttributeError rather than a message telling the caller to pass an array.
    with pytest.raises(AttributeError, match="ndim"):
        input_checks.require_theta_is_1D_array([0.1, -0.2, 3.0])
