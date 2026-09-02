"""
Dedicated unit coverage for the always-on data-integrity input checks added on 2026-09-02:

  - require_no_duplicate_subject_time_rows
  - require_contiguous_participation
  - require_nondecreasing_policy_numbers_over_time
  - require_analysis_df_values_finite
  - require_supplied_args_finite
  - require_beta_dimensions_consistent
  - require_theta_estimate_is_finite_and_nonempty
  - require_valid_percentile_bootstrap_settings

Each check is exercised DIRECTLY here (the orchestrator path is covered in
test_input_checks_orchestrator.py). Because these checks are always on and cannot be
suppressed, roughly half of the tests below are FALSE-ALARM guards: they pin inputs that
are legitimate in real studies and must NOT be rejected -- NaNs on out-of-study rows,
staggered recruitment, interleaved fallback (negative) policies, blank argument tuples,
non-numeric argument entries, and the first update policy's empty previous-betas block.
"""

import ast
import re

import numpy as np
import pandas as pd
import pytest
from jax import numpy as jnp

from lifejacket import input_checks

# Column names match the real fixture shape in tests/benchmarks/fixtures/small.
_ACTIVE_COL_INTEGRITY = "in_study"
_ACTION_COL_INTEGRITY = "action"
_POLICY_NUM_COL_INTEGRITY = "policy_num"
_CALENDAR_T_COL_INTEGRITY = "calendar_t"
_SUBJECT_ID_COL_INTEGRITY = "subject_id"
_ACTION_PROB_COL_INTEGRITY = "action_prob"
_REWARD_COL_INTEGRITY = "reward"

# The four columns require_analysis_df_values_finite inspects.
_FINITE_CHECKED_COLS_INTEGRITY = (
    _ACTION_COL_INTEGRITY,
    _POLICY_NUM_COL_INTEGRITY,
    _ACTION_PROB_COL_INTEGRITY,
    _REWARD_COL_INTEGRITY,
)

# Row labels of the baseline study built by _build_study_integrity, for reference:
#   0..3  -> subject "s1" at calendar_t 0..3, in study throughout
#   4..7  -> subject "s2" at calendar_t 0..3, OUT of study at 3 (leaves early)
#   8..11 -> subject "s3" at calendar_t 0..3, OUT of study at 0 (recruited late)
_S1_T0_LABEL_INTEGRITY = 0
_S1_T1_LABEL_INTEGRITY = 1
_S1_T2_LABEL_INTEGRITY = 2
_S1_T3_LABEL_INTEGRITY = 3
_S2_T0_LABEL_INTEGRITY = 4
_S2_T1_LABEL_INTEGRITY = 5
_S2_T2_LABEL_INTEGRITY = 6
_S2_T3_LABEL_INTEGRITY = 7
_S3_T0_LABEL_INTEGRITY = 8
_S3_T1_LABEL_INTEGRITY = 9
_S3_T2_LABEL_INTEGRITY = 10
_S3_T3_LABEL_INTEGRITY = 11

# A deliberately non-monotonic, deterministic row permutation (positional), used to prove
# these checks do not depend on the analysis DataFrame's row order.
_SHUFFLE_ORDER_INTEGRITY = [7, 0, 10, 3, 5, 11, 1, 8, 4, 9, 2, 6]

# Beta/argument-shape fixtures.
_BETA_DIM_INTEGRITY = 2
_ACTION_PROB_BETA_INDEX_INTEGRITY = 0
_UPDATE_BETA_INDEX_INTEGRITY = 0
_UPDATE_PREVIOUS_BETAS_INDEX_INTEGRITY = 1
# Policy 1 is the initial policy (not produced by an update), so the update-produced
# policies are 2 and 3 and their beta-history indices are 0 and 1 -- exactly what
# helper_functions.construct_beta_index_by_policy_num_map builds for this study.
_BETA_INDEX_BY_POLICY_NUM_INTEGRITY = {2: 0, 3: 1}


def _build_study_integrity(
    *,
    empty=False,
    shuffle_rows=False,
    duplicate_active_row=False,
    duplicate_inactive_row=False,
    duplicate_row_of_other_subject_too=False,
    leave_and_return=False,
    never_active_subject=False,
    all_rows_out_of_study=False,
    single_active_decision_time=False,
    fallback_policies_interleaved=False,
    mixed_policies_at_one_decision_time=False,
    fallback_hiding_policy_decrease=False,
    policy_decrease_on_active_rows=False,
    policy_decrease_for_two_subjects=False,
    policy_decrease_only_on_inactive_rows=False,
    nan_action_on_active_row=False,
    nan_policy_num_on_active_row=False,
    nan_action_prob_on_active_row=False,
    nan_reward_on_active_row=False,
    inf_reward_on_active_row=False,
    nan_calendar_t_on_active_row=False,
    object_dtype_reward=False,
):
    """
    A small valid study shaped like the real fixtures: three subjects over calendar times
    0..3 with STAGGERED participation ("s2" leaves early, "s3" is recruited late),
    in_study as int64 0/1, string subject ids, and action / policy_num / action_prob /
    reward as float64 columns that hold NaN on every out-of-study row -- so all four are
    non-finite frame-wide and finite over the active rows, which is the shape
    require_analysis_df_values_finite exists to tolerate.

    The initial policy number is 1 (not 0); policy 1 is in force at calendar times 0 and
    1, policy 2 at time 2 and policy 3 at time 3.

    With no keyword flags the frame passes all four analysis_df integrity checks. Each
    flag introduces exactly one defect (or one benign variation); see the flag name and
    the test that uses it.
    """
    policy_num_by_time = {0: 1.0, 1: 1.0, 2: 2.0, 3: 3.0}
    active_times_by_subject_id = {
        "s1": (0, 1, 2, 3),
        "s2": (0, 1, 2),
        "s3": (1, 2, 3),
    }
    rows = []
    for subject_id, active_times in active_times_by_subject_id.items():
        for calendar_t in (0, 1, 2, 3):
            is_active = calendar_t in active_times
            rows.append(
                {
                    _CALENDAR_T_COL_INTEGRITY: calendar_t,
                    _SUBJECT_ID_COL_INTEGRITY: subject_id,
                    _ACTIVE_COL_INTEGRITY: 1 if is_active else 0,
                    _POLICY_NUM_COL_INTEGRITY: (
                        policy_num_by_time[calendar_t] if is_active else np.nan
                    ),
                    _ACTION_COL_INTEGRITY: (
                        float(calendar_t % 2) if is_active else np.nan
                    ),
                    _ACTION_PROB_COL_INTEGRITY: (
                        (0.3 if calendar_t % 2 == 0 else 0.7) if is_active else np.nan
                    ),
                    _REWARD_COL_INTEGRITY: (
                        float(calendar_t) - 1.5 if is_active else np.nan
                    ),
                }
            )
    analysis_df = pd.DataFrame(rows)

    if empty:
        # Keep the columns and their dtypes, drop every row.
        return analysis_df.iloc[0:0]

    # --- participation / active-indicator variations ----------------------------
    if all_rows_out_of_study:
        analysis_df[_ACTIVE_COL_INTEGRITY] = 0
    if never_active_subject:
        # "s3" has rows at every decision time but is active at none of them.
        for label in (
            _S3_T0_LABEL_INTEGRITY,
            _S3_T1_LABEL_INTEGRITY,
            _S3_T2_LABEL_INTEGRITY,
            _S3_T3_LABEL_INTEGRITY,
        ):
            analysis_df.at[label, _ACTIVE_COL_INTEGRITY] = 0
    if leave_and_return:
        # "s1" is active at times 0, 2 and 3 but not at 1: a gap in the middle.
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _ACTIVE_COL_INTEGRITY] = 0
    if single_active_decision_time:
        # "s3" is active at exactly one decision time (2), which is contiguous.
        analysis_df.at[_S3_T1_LABEL_INTEGRITY, _ACTIVE_COL_INTEGRITY] = 0
        analysis_df.at[_S3_T3_LABEL_INTEGRITY, _ACTIVE_COL_INTEGRITY] = 0

    # --- policy-number variations -----------------------------------------------
    if fallback_policies_interleaved:
        # Every subject falls back to a negative policy at calendar time 1, so each
        # non-negative subsequence is still 1, 2, 3 (or 2, 3 for late-recruited "s3").
        analysis_df.loc[
            analysis_df[_CALENDAR_T_COL_INTEGRITY] == 1, _POLICY_NUM_COL_INTEGRITY
        ] = -1.0
    if mixed_policies_at_one_decision_time:
        # Two different policies in force at the SAME decision time: "s2" is on a
        # fallback policy at time 2 while "s1" and "s3" are on policy 2.
        analysis_df.at[_S2_T2_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = -2.0
    if fallback_hiding_policy_decrease:
        # "s1" only: 2, (fallback), 1, 1 -- the fallback must not hide the 2 -> 1 drop.
        analysis_df.at[_S1_T0_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 2.0
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = -1.0
        analysis_df.at[_S1_T2_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 1.0
        analysis_df.at[_S1_T3_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 1.0
    if policy_decrease_on_active_rows:
        # "s1" goes 1, 1, 2, 1: back to an earlier policy at the last decision time.
        analysis_df.at[_S1_T3_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 1.0
    if policy_decrease_for_two_subjects:
        analysis_df.at[_S1_T3_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 1.0
        analysis_df.at[_S3_T3_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 1.0
    if policy_decrease_only_on_inactive_rows:
        # Out-of-study rows carry nonsense policy numbers, which is nobody's business:
        # every policy-ordering question is about the rows the subject was in study for.
        analysis_df.at[_S2_T3_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 0.0
        analysis_df.at[_S3_T0_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = 99.0

    # --- non-finite values on ACTIVE rows ---------------------------------------
    if nan_action_on_active_row:
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _ACTION_COL_INTEGRITY] = np.nan
    if nan_policy_num_on_active_row:
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _POLICY_NUM_COL_INTEGRITY] = np.nan
    if nan_action_prob_on_active_row:
        analysis_df.at[_S2_T1_LABEL_INTEGRITY, _ACTION_PROB_COL_INTEGRITY] = np.nan
    if nan_reward_on_active_row:
        analysis_df.at[_S3_T1_LABEL_INTEGRITY, _REWARD_COL_INTEGRITY] = np.nan
    if inf_reward_on_active_row:
        analysis_df.at[_S3_T1_LABEL_INTEGRITY, _REWARD_COL_INTEGRITY] = np.inf
    if nan_calendar_t_on_active_row:
        analysis_df[_CALENDAR_T_COL_INTEGRITY] = analysis_df[
            _CALENDAR_T_COL_INTEGRITY
        ].astype("float64")
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _CALENDAR_T_COL_INTEGRITY] = np.nan
    if object_dtype_reward:
        analysis_df[_REWARD_COL_INTEGRITY] = analysis_df[_REWARD_COL_INTEGRITY].astype(
            "object"
        )
        analysis_df.at[_S1_T1_LABEL_INTEGRITY, _REWARD_COL_INTEGRITY] = "not a number"

    # --- duplicate rows ---------------------------------------------------------
    duplicated_labels = []
    if duplicate_active_row:
        duplicated_labels.append(_S1_T1_LABEL_INTEGRITY)
    if duplicate_inactive_row:
        duplicated_labels.append(_S2_T3_LABEL_INTEGRITY)
    if duplicate_row_of_other_subject_too:
        duplicated_labels.append(_S3_T2_LABEL_INTEGRITY)
    if duplicated_labels:
        analysis_df = pd.concat(
            [analysis_df, analysis_df.loc[duplicated_labels]], ignore_index=False
        )

    if shuffle_rows:
        analysis_df = analysis_df.iloc[_SHUFFLE_ORDER_INTEGRITY]

    return analysis_df


def _build_supplied_args_integrity(
    *,
    empty=False,
    all_blank=False,
    nan_and_inf_in_array=False,
    inf_in_float_scalar=False,
    nan_in_jnp_array=False,
    string_entry=False,
    none_entry=False,
    bool_array_entry=False,
    six_offending_positions=False,
):
    """
    An argument dictionary shaped the way the package's arg dictionaries are: keyed by a
    decision time / policy number, then by subject id, with a tuple per subject that
    carries a beta array plus non-array scalars, and a BLANK () tuple for a subject who
    is out of study at that key.

    With no keyword flags every numeric entry is finite, so require_supplied_args_finite
    passes. Each flag introduces exactly one defect (or one benign variation).
    """
    args_by_subject_id_by_key = {}
    for key in (0, 1):
        args_by_subject_id = {}
        for subject_id in ("s1", "s2", "s3"):
            if key == 1 and subject_id == "s3":
                # Out of study here: a blank tuple, which has nothing to check.
                args_by_subject_id[subject_id] = ()
                continue
            args_by_subject_id[subject_id] = (
                np.array([0.1, 0.2]),  # beta
                0.35,  # a plain float scalar, e.g. lower_clip
                4,  # a plain int scalar, e.g. n
            )
        args_by_subject_id_by_key[key] = args_by_subject_id

    def _replace(key, subject_id, position, value):
        args = list(args_by_subject_id_by_key[key][subject_id])
        args[position] = value
        args_by_subject_id_by_key[key][subject_id] = tuple(args)

    if empty:
        return {}
    if all_blank:
        return {
            key: {subject_id: () for subject_id in args_by_subject_id}
            for key, args_by_subject_id in args_by_subject_id_by_key.items()
        }
    if nan_and_inf_in_array:
        _replace(0, "s1", 0, np.array([np.nan, np.inf]))
    if inf_in_float_scalar:
        _replace(1, "s2", 1, float("inf"))
    if nan_in_jnp_array:
        _replace(0, "s2", 0, jnp.array([0.1, jnp.nan]))
    if string_entry:
        # Argument tuples legitimately carry non-numeric entries; a string has no
        # finiteness to test and must be skipped rather than rejected.
        for subject_id in ("s1", "s2", "s3"):
            if args_by_subject_id_by_key[0][subject_id]:
                _replace(0, subject_id, 1, "logistic")
    if none_entry:
        _replace(0, "s1", 2, None)
    if bool_array_entry:
        _replace(0, "s1", 2, np.array([True, False]))
    if six_offending_positions:
        for subject_id in ("s1", "s2", "s3"):
            for position in (0, 1):
                _replace(
                    0,
                    subject_id,
                    position,
                    np.nan if position == 1 else np.array([np.nan, 0.2]),
                )
    return args_by_subject_id_by_key


def _build_beta_args_integrity(
    *,
    jnp_betas=False,
    list_betas=False,
    wrong_action_prob_beta_length=False,
    action_prob_beta_two_dimensional=False,
    wrong_update_beta_length=False,
    previous_betas_row_count_too_small=False,
    previous_betas_row_count_too_large=False,
    previous_betas_wrong_width=False,
    fallback_policy_in_update_args=False,
    initial_policy_in_update_args=False,
):
    """
    (action_prob_func_args, alg_update_func_args) shaped for a study whose beta dimension
    is _BETA_DIM_INTEGRITY, whose initial policy is 1, and whose update-produced policies
    are 2 (beta-history index 0) and 3 (beta-history index 1).

    The previous-betas convention the implementation documents is
    shape == (that policy's index in beta_index_by_policy_num, beta_dim), so the FIRST
    update policy's block is legitimately empty: (0, beta_dim).
    """

    def _beta(*values):
        if jnp_betas:
            return jnp.array(values)
        if list_betas:
            return list(values)
        return np.array(values)

    action_prob_func_args = {}
    for decision_time in (0, 1, 2, 3):
        action_prob_func_args[decision_time] = {
            "s1": (_beta(0.1, 0.2), 0.35),
            "s2": (_beta(0.1, 0.2), 0.35),
            # Blank tuple: out of study, nothing to check.
            "s3": (),
        }
    if wrong_action_prob_beta_length:
        action_prob_func_args[0]["s1"] = (np.array([0.1, 0.2, 0.3]), 0.35)
    if action_prob_beta_two_dimensional:
        action_prob_func_args[0]["s1"] = (np.array([[0.1, 0.2]]), 0.35)

    previous_betas_by_policy_num = {
        2: np.zeros((0, _BETA_DIM_INTEGRITY)),
        3: np.zeros((1, _BETA_DIM_INTEGRITY)),
    }
    if previous_betas_row_count_too_small:
        previous_betas_by_policy_num[3] = np.zeros((0, _BETA_DIM_INTEGRITY))
    if previous_betas_row_count_too_large:
        previous_betas_by_policy_num[2] = np.zeros((1, _BETA_DIM_INTEGRITY))
    if previous_betas_wrong_width:
        previous_betas_by_policy_num[3] = np.zeros((1, _BETA_DIM_INTEGRITY + 1))

    alg_update_func_args = {}
    for policy_num, previous_betas in previous_betas_by_policy_num.items():
        alg_update_func_args[policy_num] = {
            "s1": (_beta(0.3, 0.4), previous_betas, 4),
            "s2": (_beta(0.3, 0.4), previous_betas, 4),
            "s3": (),
        }
    if wrong_update_beta_length:
        alg_update_func_args[3]["s2"] = (
            np.array([0.3]),
            previous_betas_by_policy_num[3],
            4,
        )
    if fallback_policy_in_update_args:
        # A fallback policy has no position in the update sequence, so no expected
        # previous-betas row count -- deliberately wildly shaped here.
        alg_update_func_args[-1] = {
            "s1": (_beta(0.3, 0.4), np.zeros((7, _BETA_DIM_INTEGRITY)), 4),
            "s2": (),
            "s3": (),
        }
    if initial_policy_in_update_args:
        # Likewise the initial policy, which no update produced.
        alg_update_func_args[1] = {
            "s1": (_beta(0.3, 0.4), np.zeros((5, _BETA_DIM_INTEGRITY)), 4),
            "s2": (),
            "s3": (),
        }
    return action_prob_func_args, alg_update_func_args


def _call_beta_dimensions_integrity(
    action_prob_func_args,
    alg_update_func_args,
    *,
    previous_betas_index=_UPDATE_PREVIOUS_BETAS_INDEX_INTEGRITY,
    beta_dim=_BETA_DIM_INTEGRITY,
    beta_index_by_policy_num=None,
):
    input_checks.require_beta_dimensions_consistent(
        action_prob_func_args,
        _ACTION_PROB_BETA_INDEX_INTEGRITY,
        alg_update_func_args,
        _UPDATE_BETA_INDEX_INTEGRITY,
        previous_betas_index,
        _BETA_INDEX_BY_POLICY_NUM_INTEGRITY
        if beta_index_by_policy_num is None
        else beta_index_by_policy_num,
        beta_dim,
    )


def _shown_examples_integrity(message):
    """
    The `{...}` mapping a check printed as its "up to 5 shown" examples, parsed back into
    a Python dict so a test can assert on how MANY examples were shown without depending
    on which ones sorted first.
    """
    match = re.search(r"up to 5 shown: (\{.*\})\. Please see", message)
    assert match is not None, f"no examples mapping found in message: {message}"
    return ast.literal_eval(match.group(1))


# ---------------------------------------------------------------------------
# require_no_duplicate_subject_time_rows
# ---------------------------------------------------------------------------


def test_require_no_duplicate_subject_time_rows_passes_on_valid_study():
    analysis_df = _build_study_integrity()

    input_checks.require_no_duplicate_subject_time_rows(
        analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
    )


def test_require_no_duplicate_subject_time_rows_passes_on_repeated_times_across_subjects():
    # Every decision time appears once per subject, which is the normal panel shape: it is
    # the (subject_id, calendar_t) PAIR that must be unique, not either column alone.
    analysis_df = _build_study_integrity()
    assert analysis_df[_CALENDAR_T_COL_INTEGRITY].duplicated().any()
    assert analysis_df[_SUBJECT_ID_COL_INTEGRITY].duplicated().any()

    input_checks.require_no_duplicate_subject_time_rows(
        analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
    )


def test_require_no_duplicate_subject_time_rows_passes_on_shuffled_row_order():
    analysis_df = _build_study_integrity(shuffle_rows=True)

    input_checks.require_no_duplicate_subject_time_rows(
        analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
    )


def test_require_no_duplicate_subject_time_rows_passes_on_empty_analysis_df():
    analysis_df = _build_study_integrity(empty=True)

    input_checks.require_no_duplicate_subject_time_rows(
        analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
    )


def test_require_no_duplicate_subject_time_rows_raises_on_duplicate_active_row():
    analysis_df = _build_study_integrity(duplicate_active_row=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The analysis DataFrame has 2 row(s) sharing a (subject_id, calendar_t) key, "
            "e.g. [{'subject_id': 's1', 'calendar_t': 1}]"
        ),
    ):
        input_checks.require_no_duplicate_subject_time_rows(
            analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
        )


def test_require_no_duplicate_subject_time_rows_raises_on_duplicate_inactive_row():
    # The whole point of this check over the duplicate detection inside
    # require_action_probabilities_in_analysis_df_can_be_reconstructed: that one looks at
    # ACTIVE rows only, so a duplicated out-of-study row sails through it while still
    # breaking every (subject_id, calendar_t)-keyed structure in the package.
    analysis_df = _build_study_integrity(duplicate_inactive_row=True)
    duplicated_row = analysis_df.loc[_S2_T3_LABEL_INTEGRITY]
    assert (duplicated_row[_ACTIVE_COL_INTEGRITY] == 0).all()

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The analysis DataFrame has 2 row(s) sharing a (subject_id, calendar_t) key, "
            "e.g. [{'subject_id': 's2', 'calendar_t': 3}]"
        ),
    ):
        input_checks.require_no_duplicate_subject_time_rows(
            analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
        )


def test_require_no_duplicate_subject_time_rows_reports_every_duplicated_row_and_key():
    analysis_df = _build_study_integrity(
        duplicate_active_row=True, duplicate_row_of_other_subject_too=True
    )

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_no_duplicate_subject_time_rows(
            analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
        )

    message = str(excinfo.value)
    # Four rows are involved: two originals and their two copies.
    assert "The analysis DataFrame has 4 row(s) sharing a" in message
    assert "{'subject_id': 's1', 'calendar_t': 1}" in message
    assert "{'subject_id': 's3', 'calendar_t': 2}" in message


def test_require_no_duplicate_subject_time_rows_finds_duplicates_regardless_of_row_order():
    # The duplicated rows are not adjacent after the shuffle.
    analysis_df = _build_study_integrity(duplicate_active_row=True).iloc[
        [12, 0, 5, 1, 9, 3, 11, 2, 8, 4, 10, 6, 7]
    ]

    with pytest.raises(
        AssertionError,
        match=re.escape("2 row(s) sharing a (subject_id, calendar_t) key"),
    ):
        input_checks.require_no_duplicate_subject_time_rows(
            analysis_df, _CALENDAR_T_COL_INTEGRITY, _SUBJECT_ID_COL_INTEGRITY
        )


# ---------------------------------------------------------------------------
# require_contiguous_participation
# ---------------------------------------------------------------------------


def test_require_contiguous_participation_passes_on_staggered_recruitment():
    # FALSE-ALARM GUARD: subjects starting late ("s3") and leaving early ("s2") are the
    # normal shape of a micro-randomized trial, not a defect.
    analysis_df = _build_study_integrity()

    input_checks.require_contiguous_participation(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
    )


def test_require_contiguous_participation_passes_on_single_active_decision_time():
    # One active decision time is one unbroken stretch.
    analysis_df = _build_study_integrity(single_active_decision_time=True)

    input_checks.require_contiguous_participation(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
    )


def test_require_contiguous_participation_passes_on_shuffled_row_order():
    # Contiguity is judged after sorting each subject's rows by calendar time, so a frame
    # that is not already time-ordered must not be reported as a leave-and-return.
    analysis_df = _build_study_integrity(shuffle_rows=True)

    input_checks.require_contiguous_participation(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
    )


def test_require_contiguous_participation_passes_on_empty_analysis_df():
    # No subjects, so no subject can be non-contiguous or never active.
    analysis_df = _build_study_integrity(empty=True)

    input_checks.require_contiguous_participation(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
    )


def test_require_contiguous_participation_raises_on_leave_and_return():
    analysis_df = _build_study_integrity(leave_and_return=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) leave the study and return, e.g. ['s1']"),
    ):
        input_checks.require_contiguous_participation(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
        )


def test_require_contiguous_participation_raises_on_never_active_subject():
    analysis_df = _build_study_integrity(never_active_subject=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 subject(s) are never active at any decision time, e.g. ['s3']"
        ),
    ):
        input_checks.require_contiguous_participation(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
        )


def test_require_contiguous_participation_reports_all_never_active_subjects():
    analysis_df = _build_study_integrity(all_rows_out_of_study=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "3 subject(s) are never active at any decision time, e.g. "
            "['s1', 's2', 's3']"
        ),
    ):
        input_checks.require_contiguous_participation(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
        )


def test_require_contiguous_participation_reports_never_active_before_noncontiguous():
    # Both defects at once: the never-active assertion is evaluated first, so that is the
    # message the caller sees.
    analysis_df = _build_study_integrity(
        never_active_subject=True, leave_and_return=True
    )

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_contiguous_participation(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
        )

    message = str(excinfo.value)
    assert "never active at any decision time" in message
    assert "leave the study and return" not in message


def test_require_contiguous_participation_raises_on_leave_and_return_in_shuffled_frame():
    # The gap must be detected from calendar time, not from row adjacency.
    analysis_df = _build_study_integrity(leave_and_return=True, shuffle_rows=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) leave the study and return, e.g. ['s1']"),
    ):
        input_checks.require_contiguous_participation(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
        )


# ---------------------------------------------------------------------------
# require_nondecreasing_policy_numbers_over_time
# ---------------------------------------------------------------------------


def test_require_nondecreasing_policy_numbers_over_time_passes_on_valid_study():
    # Repeated policy numbers (1 at times 0 and 1) are non-DECREASING, not increasing.
    analysis_df = _build_study_integrity()

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_passes_on_interleaved_fallbacks():
    # FALSE-ALARM GUARD: policies 1, -1, 2, 3 over four decision times. Fallback
    # (negative) policies mark "the algorithm was bypassed here", carry no position in
    # the update sequence, and must neither break monotonicity nor be ordered themselves.
    analysis_df = _build_study_integrity(fallback_policies_interleaved=True)
    fallback_policy_nums = analysis_df.loc[
        analysis_df[_POLICY_NUM_COL_INTEGRITY] < 0, _POLICY_NUM_COL_INTEGRITY
    ]
    assert not fallback_policy_nums.empty

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_passes_on_two_policies_at_one_time():
    # FALSE-ALARM GUARD: multiple policies in force at the SAME decision time is a
    # supported configuration -- "s2" is on a fallback at time 2 while the others are on
    # policy 2. This check is per SUBJECT over time, never a cross-subject comparison.
    analysis_df = _build_study_integrity(mixed_policies_at_one_decision_time=True)
    policies_at_time_two = set(
        analysis_df.loc[
            (analysis_df[_CALENDAR_T_COL_INTEGRITY] == 2)
            & (analysis_df[_ACTIVE_COL_INTEGRITY] == 1),
            _POLICY_NUM_COL_INTEGRITY,
        ]
    )
    assert len(policies_at_time_two) > 1

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_passes_on_out_of_study_nonsense():
    # Out-of-study rows carry NaN policy numbers in the baseline and outright nonsense
    # (0.0 after policy 2, 99.0 before policy 1) with this flag; the active filter is
    # what makes both harmless.
    analysis_df = _build_study_integrity(policy_decrease_only_on_inactive_rows=True)

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_passes_on_shuffled_row_order():
    analysis_df = _build_study_integrity(shuffle_rows=True)

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_passes_on_empty_analysis_df():
    analysis_df = _build_study_integrity(empty=True)

    input_checks.require_nondecreasing_policy_numbers_over_time(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _CALENDAR_T_COL_INTEGRITY,
        _SUBJECT_ID_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
    )


def test_require_nondecreasing_policy_numbers_over_time_raises_on_genuine_decrease():
    analysis_df = _build_study_integrity(policy_decrease_on_active_rows=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 subject(s) have non-fallback policy numbers that DECREASE as calendar "
            "time advances, e.g. ['s1']"
        ),
    ):
        input_checks.require_nondecreasing_policy_numbers_over_time(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
        )


def test_require_nondecreasing_policy_numbers_over_time_raises_when_fallback_hides_decrease():
    # A fallback sitting between the two offending non-negative policies must not hide
    # the regression: the check is applied to the non-negative SUBSEQUENCE (2, 1, 1).
    analysis_df = _build_study_integrity(fallback_hiding_policy_decrease=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) have non-fallback policy numbers that DECREASE"),
    ):
        input_checks.require_nondecreasing_policy_numbers_over_time(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
        )


def test_require_nondecreasing_policy_numbers_over_time_reports_every_offending_subject():
    analysis_df = _build_study_integrity(policy_decrease_for_two_subjects=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "2 subject(s) have non-fallback policy numbers that DECREASE as calendar "
            "time advances, e.g. ['s1', 's3']"
        ),
    ):
        input_checks.require_nondecreasing_policy_numbers_over_time(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
        )


def test_require_nondecreasing_policy_numbers_over_time_raises_on_decrease_in_shuffled_frame():
    # Without the internal sort by calendar time this frame's row order alone would make
    # the sequence look fine (or make a valid one look broken).
    analysis_df = _build_study_integrity(
        policy_decrease_on_active_rows=True, shuffle_rows=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) have non-fallback policy numbers that DECREASE"),
    ):
        input_checks.require_nondecreasing_policy_numbers_over_time(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _CALENDAR_T_COL_INTEGRITY,
            _SUBJECT_ID_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
        )


# ---------------------------------------------------------------------------
# require_analysis_df_values_finite
# ---------------------------------------------------------------------------


def test_require_analysis_df_values_finite_passes_when_nans_are_only_on_inactive_rows():
    # FALSE-ALARM GUARD, and the whole scope decision behind this check: in the real
    # fixture shape all four inspected columns are non-finite frame-wide and finite over
    # the active rows. Checking them frame-wide would reject every real study.
    analysis_df = _build_study_integrity()
    for col_name in _FINITE_CHECKED_COLS_INTEGRITY:
        column_values = analysis_df[col_name].to_numpy()
        assert not np.isfinite(column_values).all()
        active_values = analysis_df.loc[
            analysis_df[_ACTIVE_COL_INTEGRITY] == 1, col_name
        ].to_numpy()
        assert np.isfinite(active_values).all()

    input_checks.require_analysis_df_values_finite(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _ACTION_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
        _ACTION_PROB_COL_INTEGRITY,
        _REWARD_COL_INTEGRITY,
    )


def test_require_analysis_df_values_finite_passes_when_every_row_is_out_of_study():
    # Nothing active, so nothing to check -- vacuous rather than an error.
    analysis_df = _build_study_integrity(
        all_rows_out_of_study=True, nan_reward_on_active_row=True
    )

    input_checks.require_analysis_df_values_finite(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _ACTION_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
        _ACTION_PROB_COL_INTEGRITY,
        _REWARD_COL_INTEGRITY,
    )


def test_require_analysis_df_values_finite_passes_on_empty_analysis_df():
    analysis_df = _build_study_integrity(empty=True)

    input_checks.require_analysis_df_values_finite(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _ACTION_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
        _ACTION_PROB_COL_INTEGRITY,
        _REWARD_COL_INTEGRITY,
    )


def test_require_analysis_df_values_finite_ignores_non_numeric_columns():
    # A non-numeric column is require_all_named_columns_not_object_type_in_analysis_df's
    # finding; this check skips it rather than crashing inside np.isfinite.
    analysis_df = _build_study_integrity(object_dtype_reward=True)
    assert analysis_df[_REWARD_COL_INTEGRITY].dtype == "object"

    input_checks.require_analysis_df_values_finite(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _ACTION_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
        _ACTION_PROB_COL_INTEGRITY,
        _REWARD_COL_INTEGRITY,
    )


def test_require_analysis_df_values_finite_ignores_calendar_time_column():
    # Deliberately out of scope: calendar_t's finiteness and integrality are asserted
    # over the WHOLE column by require_consecutive_integer_calendar_times, since times
    # must be well-formed even out of study.
    analysis_df = _build_study_integrity(nan_calendar_t_on_active_row=True)

    input_checks.require_analysis_df_values_finite(
        analysis_df,
        _ACTIVE_COL_INTEGRITY,
        _ACTION_COL_INTEGRITY,
        _POLICY_NUM_COL_INTEGRITY,
        _ACTION_PROB_COL_INTEGRITY,
        _REWARD_COL_INTEGRITY,
    )


def test_require_analysis_df_values_finite_raises_on_nan_reward_on_active_row():
    analysis_df = _build_study_integrity(nan_reward_on_active_row=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These analysis DataFrame columns contain non-finite values (NaN or inf) on "
            "ACTIVE rows -- column -> count: {'reward': 1}"
        ),
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_raises_on_infinite_reward_on_active_row():
    # inf, not just NaN: it propagates into the estimates and the reported variance the
    # same way.
    analysis_df = _build_study_integrity(inf_reward_on_active_row=True)

    with pytest.raises(
        AssertionError, match=re.escape("column -> count: {'reward': 1}")
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_raises_on_nan_action_on_active_row():
    analysis_df = _build_study_integrity(nan_action_on_active_row=True)

    with pytest.raises(
        AssertionError, match=re.escape("column -> count: {'action': 1}")
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_raises_on_nan_policy_num_on_active_row():
    analysis_df = _build_study_integrity(nan_policy_num_on_active_row=True)

    with pytest.raises(
        AssertionError, match=re.escape("column -> count: {'policy_num': 1}")
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_raises_on_nan_action_prob_on_active_row():
    analysis_df = _build_study_integrity(nan_action_prob_on_active_row=True)

    with pytest.raises(
        AssertionError, match=re.escape("column -> count: {'action_prob': 1}")
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_reports_every_offending_column_at_once():
    # Column -> count for all offenders together, in the order the columns are inspected.
    analysis_df = _build_study_integrity(
        nan_action_prob_on_active_row=True, nan_reward_on_active_row=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("column -> count: {'action_prob': 1, 'reward': 1}"),
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


def test_require_analysis_df_values_finite_counts_every_offending_active_row():
    analysis_df = _build_study_integrity()
    analysis_df.loc[analysis_df[_ACTIVE_COL_INTEGRITY] == 1, _REWARD_COL_INTEGRITY] = (
        np.nan
    )

    with pytest.raises(
        AssertionError, match=re.escape("column -> count: {'reward': 10}")
    ):
        input_checks.require_analysis_df_values_finite(
            analysis_df,
            _ACTIVE_COL_INTEGRITY,
            _ACTION_COL_INTEGRITY,
            _POLICY_NUM_COL_INTEGRITY,
            _ACTION_PROB_COL_INTEGRITY,
            _REWARD_COL_INTEGRITY,
        )


# ---------------------------------------------------------------------------
# require_supplied_args_finite
# ---------------------------------------------------------------------------


def test_require_supplied_args_finite_passes_on_valid_args():
    args_by_subject_id_by_key = _build_supplied_args_integrity()

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "action_prob_func", "decision_time"
    )


def test_require_supplied_args_finite_skips_blank_arg_tuples():
    # FALSE-ALARM GUARD: a blank () tuple is how "this subject is out of study at this
    # key" is expressed, and it has nothing to test.
    args_by_subject_id_by_key = _build_supplied_args_integrity(all_blank=True)

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "action_prob_func", "decision_time"
    )


def test_require_supplied_args_finite_passes_on_empty_args_dictionary():
    args_by_subject_id_by_key = _build_supplied_args_integrity(empty=True)

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "alg_update_func", "policy_num"
    )


def test_require_supplied_args_finite_skips_non_numeric_string_entries():
    # FALSE-ALARM GUARD: argument tuples legitimately carry non-array, non-numeric
    # entries; anything that is not a number has no finiteness to test.
    args_by_subject_id_by_key = _build_supplied_args_integrity(string_entry=True)

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "action_prob_func", "decision_time"
    )


def test_require_supplied_args_finite_skips_none_entries():
    args_by_subject_id_by_key = _build_supplied_args_integrity(none_entry=True)

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "action_prob_func", "decision_time"
    )


def test_require_supplied_args_finite_skips_boolean_array_entries():
    # np.bool_ is not a subtype of np.number, so a mask-like boolean entry is skipped.
    args_by_subject_id_by_key = _build_supplied_args_integrity(bool_array_entry=True)

    input_checks.require_supplied_args_finite(
        args_by_subject_id_by_key, "alg_update_func", "policy_num"
    )


def test_require_supplied_args_finite_raises_on_nan_and_inf_in_an_array():
    args_by_subject_id_by_key = _build_supplied_args_integrity(
        nan_and_inf_in_array=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 supplied action_prob_func argument position(s) contain non-finite values "
            "(NaN or inf). Offending (decision_time, subject_id, arg position) -> count, "
            "up to 5 shown: {(0, 's1', 0): 2}"
        ),
    ):
        input_checks.require_supplied_args_finite(
            args_by_subject_id_by_key, "action_prob_func", "decision_time"
        )


def test_require_supplied_args_finite_raises_on_infinite_plain_float_scalar():
    # A non-array scalar is still numeric, so it is checked.
    args_by_subject_id_by_key = _build_supplied_args_integrity(inf_in_float_scalar=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 supplied alg_update_func argument position(s) contain non-finite values "
            "(NaN or inf). Offending (policy_num, subject_id, arg position) -> count, "
            "up to 5 shown: {(1, 's2', 1): 1}"
        ),
    ):
        input_checks.require_supplied_args_finite(
            args_by_subject_id_by_key, "alg_update_func", "policy_num"
        )


def test_require_supplied_args_finite_raises_on_nan_in_jax_array():
    # The real argument tuples hold jnp arrays, not numpy ones.
    args_by_subject_id_by_key = _build_supplied_args_integrity(nan_in_jnp_array=True)

    with pytest.raises(
        AssertionError, match=re.escape("up to 5 shown: {(0, 's2', 0): 1}")
    ):
        input_checks.require_supplied_args_finite(
            args_by_subject_id_by_key, "action_prob_func", "decision_time"
        )


def test_require_supplied_args_finite_counts_all_offenders_but_shows_only_five():
    args_by_subject_id_by_key = _build_supplied_args_integrity(
        six_offending_positions=True
    )

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_supplied_args_finite(
            args_by_subject_id_by_key, "action_prob_func", "decision_time"
        )

    message = str(excinfo.value)
    assert "6 supplied action_prob_func argument position(s)" in message
    assert len(_shown_examples_integrity(message)) == 5


# ---------------------------------------------------------------------------
# require_beta_dimensions_consistent
# ---------------------------------------------------------------------------


def test_require_beta_dimensions_consistent_passes_on_valid_args():
    # The first update policy (2, beta-history index 0) legitimately carries an EMPTY
    # (0, beta_dim) previous-betas block: no update precedes it.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity()
    assert alg_update_func_args[2]["s1"][
        _UPDATE_PREVIOUS_BETAS_INDEX_INTEGRITY
    ].shape == (0, _BETA_DIM_INTEGRITY)

    _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_passes_on_jax_array_betas():
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        jnp_betas=True
    )

    _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_passes_on_list_betas():
    # np.asarray is applied before the shape comparison, so a plain list of the right
    # length is accepted.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        list_betas=True
    )

    _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_passes_on_empty_arg_dictionaries():
    _call_beta_dimensions_integrity({}, {})


def test_require_beta_dimensions_consistent_raises_on_wrong_action_prob_beta_length():
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        wrong_action_prob_beta_length=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 action_prob_func_args beta(s) are not shaped (2,), the dimension taken "
            "from the first supplied beta. Offending (decision_time, subject_id) -> "
            "shape, up to 5 shown: {(0, 's1'): (3,)}"
        ),
    ):
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_raises_on_two_dimensional_action_prob_beta():
    # A (1, 2) beta has beta_dim entries but the wrong shape; the comparison is exact.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        action_prob_beta_two_dimensional=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape("up to 5 shown: {(0, 's1'): (1, 2)}"),
    ):
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_raises_on_wrong_update_beta_length():
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        wrong_update_beta_length=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 alg_update_func_args beta(s) are not shaped (2,). Offending (policy_num, "
            "subject_id) -> shape, up to 5 shown: {(3, 's2'): (1,)}"
        ),
    ):
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_raises_on_too_few_previous_betas_rows():
    # Policy 3 is the second update, so its previous-betas block must have exactly one
    # row: the threading code uses the RECORDED row count as the slice length into the
    # shared beta history, so an empty block would thread in no history at all.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        previous_betas_row_count_too_small=True
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "previous-betas block(s) have the wrong shape; each must be (number of "
            "updates before that policy, 2)"
        ),
    ):
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_raises_on_too_many_previous_betas_rows():
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        previous_betas_row_count_too_large=True
    )

    with pytest.raises(AssertionError) as excinfo:
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)

    message = str(excinfo.value)
    assert "previous-betas block(s) have the wrong shape" in message
    # Both subjects of policy 2 share the offending block.
    assert _shown_examples_integrity(message) == {
        (2, "s1"): "(1, 2) != expected (0, 2)",
        (2, "s2"): "(1, 2) != expected (0, 2)",
    }


def test_require_beta_dimensions_consistent_raises_on_wrong_previous_betas_width():
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        previous_betas_wrong_width=True
    )

    with pytest.raises(AssertionError) as excinfo:
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)

    message = str(excinfo.value)
    assert "previous-betas block(s) have the wrong shape" in message
    assert _shown_examples_integrity(message) == {
        (3, "s1"): "(1, 3) != expected (1, 2)",
        (3, "s2"): "(1, 3) != expected (1, 2)",
    }


def test_require_beta_dimensions_consistent_skips_previous_betas_when_index_negative():
    # Early return: with no previous-betas index supplied there is no previous-betas
    # argument to shape-check, so a wildly shaped one at position 1 is not this check's
    # business. The plain betas are still checked.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        previous_betas_wrong_width=True
    )

    _call_beta_dimensions_integrity(
        action_prob_func_args, alg_update_func_args, previous_betas_index=-1
    )


def test_require_beta_dimensions_consistent_skips_previous_betas_for_fallback_policy():
    # A fallback policy has no position in the update sequence, so there is no expected
    # previous-betas row count to check against.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        fallback_policy_in_update_args=True
    )
    assert -1 not in _BETA_INDEX_BY_POLICY_NUM_INTEGRITY

    _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_skips_previous_betas_for_initial_policy():
    # Likewise the initial policy, which no update produced.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        initial_policy_in_update_args=True
    )
    assert 1 not in _BETA_INDEX_BY_POLICY_NUM_INTEGRITY

    _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)


def test_require_beta_dimensions_consistent_raises_on_wrong_beta_dim_for_whole_study():
    # beta_dim is read off the FIRST non-blank action-prob tuple elsewhere; passing a
    # different one here means every recorded beta disagrees with it.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity()

    with pytest.raises(AssertionError) as excinfo:
        _call_beta_dimensions_integrity(
            action_prob_func_args, alg_update_func_args, beta_dim=3
        )

    message = str(excinfo.value)
    assert "action_prob_func_args beta(s) are not shaped (3,)" in message
    # Eight non-blank action-prob tuples (two subjects at four decision times), of which
    # only five are shown.
    assert message.startswith("8 action_prob_func_args beta(s)")
    assert len(_shown_examples_integrity(message)) == 5


def test_require_beta_dimensions_consistent_reports_action_prob_betas_before_update_betas():
    # Both defects at once: the action-prob assertion is evaluated first.
    action_prob_func_args, alg_update_func_args = _build_beta_args_integrity(
        wrong_action_prob_beta_length=True, wrong_update_beta_length=True
    )

    with pytest.raises(AssertionError) as excinfo:
        _call_beta_dimensions_integrity(action_prob_func_args, alg_update_func_args)

    message = str(excinfo.value)
    assert "action_prob_func_args beta(s) are not shaped" in message
    assert "alg_update_func_args beta(s) are not shaped" not in message


# ---------------------------------------------------------------------------
# require_theta_estimate_is_finite_and_nonempty
# ---------------------------------------------------------------------------


def test_require_theta_estimate_is_finite_and_nonempty_passes_on_valid_theta():
    input_checks.require_theta_estimate_is_finite_and_nonempty(
        jnp.array([0.5, -1.25, 3.0])
    )


def test_require_theta_estimate_is_finite_and_nonempty_passes_on_single_component():
    input_checks.require_theta_estimate_is_finite_and_nonempty(np.array([0.0]))


def test_require_theta_estimate_is_finite_and_nonempty_passes_on_integer_theta():
    input_checks.require_theta_estimate_is_finite_and_nonempty(np.array([1, 2, 3]))


def test_require_theta_estimate_is_finite_and_nonempty_ignores_dimensionality():
    # Deliberately out of scope: the number of dimensions is require_theta_is_1D_array's
    # finding, so a finite non-empty 2D array passes this one.
    input_checks.require_theta_estimate_is_finite_and_nonempty(
        np.array([[1.0, 2.0], [3.0, 4.0]])
    )


def test_require_theta_estimate_is_finite_and_nonempty_raises_on_empty_theta():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "theta_calculation_func returned an empty theta estimate; the inferential "
            "target must have at least one component."
        ),
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(jnp.array([]))


def test_require_theta_estimate_is_finite_and_nonempty_raises_on_nan_component():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "theta_calculation_func returned a theta estimate with non-finite values at "
            "component(s) [1]"
        ),
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(
            np.array([0.5, np.nan, 3.0])
        )


def test_require_theta_estimate_is_finite_and_nonempty_raises_on_infinite_component():
    with pytest.raises(
        AssertionError,
        match=re.escape("non-finite values at component(s) [2]"),
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(
            np.array([0.5, 1.0, -np.inf])
        )


def test_require_theta_estimate_is_finite_and_nonempty_reports_up_to_five_components():
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_theta_estimate_is_finite_and_nonempty(np.full(7, np.nan))

    assert "non-finite values at component(s) [0, 1, 2, 3, 4]" in str(excinfo.value)


def test_require_theta_estimate_is_finite_and_nonempty_raises_on_non_numeric_theta():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "theta_calculation_func returned a non-numeric theta estimate (dtype <U1)"
        ),
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(np.array(["a", "b"]))


def test_require_theta_estimate_is_finite_and_nonempty_raises_on_object_dtype_theta():
    # np.isfinite would raise a bare TypeError on an object array, so the numeric check
    # has to come first.
    with pytest.raises(
        AssertionError,
        match=re.escape("non-numeric theta estimate (dtype object)"),
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(
            np.array([None, 1.0], dtype="object")
        )


def test_require_theta_estimate_is_finite_and_nonempty_checks_emptiness_before_dtype():
    # An empty string-dtype theta is reported as empty, not as non-numeric.
    with pytest.raises(
        AssertionError, match=re.escape("returned an empty theta estimate")
    ):
        input_checks.require_theta_estimate_is_finite_and_nonempty(
            np.array([], dtype="<U1")
        )


# ---------------------------------------------------------------------------
# require_valid_percentile_bootstrap_settings
# ---------------------------------------------------------------------------


def test_require_valid_percentile_bootstrap_settings_passes_on_typical_settings():
    input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, 12345)


def test_require_valid_percentile_bootstrap_settings_passes_on_zero_draws():
    # 0 disables the bootstrap, and alpha is still validated in that case so a bad one is
    # caught the first time it is passed rather than the first time it is used.
    input_checks.require_valid_percentile_bootstrap_settings(0, 0.05, None)


def test_require_valid_percentile_bootstrap_settings_allows_seed_zero():
    # 0 is a perfectly good seed and must not be confused with "no seed supplied".
    input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, 0)


def test_require_valid_percentile_bootstrap_settings_allows_seed_none():
    input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, None)


def test_require_valid_percentile_bootstrap_settings_allows_numpy_integers():
    input_checks.require_valid_percentile_bootstrap_settings(
        np.int64(500), np.float64(0.1), np.int32(7)
    )


def test_require_valid_percentile_bootstrap_settings_allows_alpha_near_the_boundaries():
    input_checks.require_valid_percentile_bootstrap_settings(200, 1e-9, 0)
    input_checks.require_valid_percentile_bootstrap_settings(200, 1.0 - 1e-9, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_alpha_as_a_percentage():
    # The mistake this check exists for: alpha is used directly as a quantile level, so
    # 95 silently produces a meaningless interval that is then reported as
    # percentile_bootstrap_ci in analysis.pkl.
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_alpha must be strictly between 0 and 1 -- it is a "
            "PROPORTION, not a percentage, so pass 0.05 for a 95% interval -- got 95."
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(200, 95, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_alpha_of_exactly_zero():
    with pytest.raises(
        AssertionError,
        match=re.escape("percentile_bootstrap_alpha must be strictly between 0 and 1"),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(200, 0.0, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_alpha_of_exactly_one():
    with pytest.raises(AssertionError, match=re.escape("got 1.0.")):
        input_checks.require_valid_percentile_bootstrap_settings(200, 1.0, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_negative_alpha():
    with pytest.raises(AssertionError, match=re.escape("got -0.05.")):
        input_checks.require_valid_percentile_bootstrap_settings(200, -0.05, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_boolean_alpha():
    # True == 1, which is not strictly inside (0, 1).
    with pytest.raises(AssertionError, match=re.escape("got True.")):
        input_checks.require_valid_percentile_bootstrap_settings(200, True, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_negative_draws():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_draws must be a non-negative integer, got -1 (0 "
            "disables the bootstrap)."
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(-1, 0.05, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_float_draws():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_draws must be a non-negative integer, got 200.0"
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(200.0, 0.05, 0)


def test_require_valid_percentile_bootstrap_settings_rejects_boolean_draws():
    # bool is deliberately excluded even though it is an int subclass: True as a draw
    # count is a wiring mistake, not a request for one bootstrap draw.
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_draws must be a non-negative integer, got True"
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(True, 0.05, 0)


def test_require_valid_percentile_bootstrap_settings_rejects_boolean_false_draws():
    # False == 0 would otherwise read as "bootstrap disabled".
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_draws must be a non-negative integer, got False"
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(False, 0.05, 0)


def test_require_valid_percentile_bootstrap_settings_raises_on_negative_seed():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_seed must be None or a non-negative integer "
            "(np.random.default_rng rejects negative seeds), got -1."
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, -1)


def test_require_valid_percentile_bootstrap_settings_rejects_boolean_seed():
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "percentile_bootstrap_seed must be None or a non-negative integer"
        ),
    ):
        input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, True)


def test_require_valid_percentile_bootstrap_settings_raises_on_float_seed():
    with pytest.raises(AssertionError, match=re.escape("got 12345.0.")):
        input_checks.require_valid_percentile_bootstrap_settings(200, 0.05, 12345.0)


def test_require_valid_percentile_bootstrap_settings_checks_draws_before_alpha():
    # Both settings bad: draws is asserted first, so that is the reported finding.
    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_valid_percentile_bootstrap_settings(-1, 95, -1)

    message = str(excinfo.value)
    assert "percentile_bootstrap_draws must be a non-negative integer" in message
    assert "percentile_bootstrap_alpha" not in message
    assert "percentile_bootstrap_seed" not in message
