import re

import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

_SUBJECT_IDS_POLICYNUMS = (0, 1)


def _build_study_policynums(
    *,
    active_policy_nums=(1, 2, 3),
    inactive_policy_nums=(),
    policy_num_dtype="float64",
    update_arg_keys=(2, 3),
    blank_update_arg_keys=(),
):
    """
    A tiny study shaped like the real fixtures: two subjects, one decision time per entry in
    active_policy_nums (in_study=1, a real action) followed by one decision time per entry in
    inactive_policy_nums (in_study=0, action NaN). policy_num is float64 by default and the
    initial policy number is 1, NOT 0, exactly as in tests/benchmarks/fixtures/small.

    active_policy_nums: the policy number in force at each in-study decision time. A NEGATIVE
        entry is a fallback policy.
    inactive_policy_nums: policy numbers that appear only on out-of-study rows (np.nan is
        allowed with the default float64 dtype).
    update_arg_keys: the policy numbers keyed in alg_update_func_args -- ints, so that matching
        them against the float64 policy_num column is exercised by default.
    blank_update_arg_keys: those keys get an EMPTY TUPLE per subject ("blank"/not applicable)
        rather than a real argument tuple.
    """
    rows = []
    calendar_t = 0
    for policy_num, in_study in [
        (policy_num, 1) for policy_num in active_policy_nums
    ] + [(policy_num, 0) for policy_num in inactive_policy_nums]:
        for subject_id in _SUBJECT_IDS_POLICYNUMS:
            rows.append(
                {
                    "calendar_t": calendar_t,
                    "user_id": subject_id,
                    "in_study": in_study,
                    "policy_num": policy_num,
                    "action": 1.0 if in_study else np.nan,
                }
            )
        calendar_t += 1

    analysis_df = pd.DataFrame(
        rows, columns=["calendar_t", "user_id", "in_study", "policy_num", "action"]
    )
    analysis_df["policy_num"] = analysis_df["policy_num"].astype(policy_num_dtype)
    analysis_df["in_study"] = analysis_df["in_study"].astype("int64")

    alg_update_func_args = {
        policy_num: {
            subject_id: (
                ()
                if policy_num in blank_update_arg_keys
                else (np.array([0.5, -0.5]), np.array([1.0, 2.0]))
            )
            for subject_id in _SUBJECT_IDS_POLICYNUMS
        }
        for policy_num in update_arg_keys
    }
    return analysis_df, alg_update_func_args


def _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args):
    input_checks.require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args(
        analysis_df, "in_study", "policy_num", alg_update_func_args
    )


def _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args):
    input_checks.require_no_policy_numbers_present_in_alg_update_args_but_not_analysis_df(
        analysis_df, "policy_num", alg_update_func_args
    )


# ---------------------------------------------------------------------------
# require_all_policy_numbers_in_analysis_df_except_possibly_initial_and_fallback_present_in_alg_update_args
# ---------------------------------------------------------------------------


def test_all_policy_nums_present_passes_on_valid_study_policynums():
    # Baseline: initial policy 1 plus updates 2 and 3, all updates supplied. Also pins that INT
    # update-dict keys (2, 3) match the FLOAT64 policy_num values (2.0, 3.0) -- the real
    # fixtures pair int keys with a float64 column, so a check that compared them by identity
    # rather than by value would reject every real study.
    analysis_df, alg_update_func_args = _build_study_policynums()

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_raises_when_non_initial_policy_missing_policynums():
    # THE (a) REGRESSION. With no fallback rows present, the pre-fix code compared the policy
    # column against its own (identically labeled) self elementwise, so every comparison was
    # "value > itself" == False, the offending set came out empty, and issubset() was trivially
    # satisfied -- a genuinely missing update for policy 3 PASSED the check. It must raise.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), update_arg_keys=(2,)
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "are not in the update function args: [3.0] (the initial policy number is 1.0)"
        ),
    ):
        _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_passes_with_fallback_rows_present_policynums():
    # THE (b) REGRESSION, passing direction. Fallback (negative) policy rows make the filtered
    # nonnegative Series shorter than the column, so the pre-fix elementwise comparison blew up
    # with ValueError("Can only compare identically-labeled Series objects") instead of checking
    # anything. A valid study with fallback rows must simply pass.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, -1, 3), update_arg_keys=(2, 3)
    )

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_raises_assertion_error_not_value_error_with_fallback_rows_policynums():
    # THE (b) REGRESSION, failing direction: with fallback rows present the check must reach its
    # own assertion (AssertionError naming policy 3) rather than dying inside pandas with
    # ValueError about identically-labeled Series. pytest.raises(AssertionError) fails loudly if
    # a ValueError comes out instead.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, -1, 3), update_arg_keys=(2,)
    )

    with pytest.raises(AssertionError, match=re.escape("update function args: [3.0]")):
        _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_does_not_require_the_non_zero_initial_policy_policynums():
    # The initial policy is produced by no update, so with an initial policy number of 1 (the
    # real-fixture shape, not 0) alg_update_func_args must be allowed to key only 2 and 3. A
    # check that hardcoded "the initial policy is 0" would demand an update for policy 1 here.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), update_arg_keys=(2, 3)
    )
    assert 1 not in alg_update_func_args

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_message_excludes_the_non_zero_initial_policy_policynums():
    # The failure message used to recompute the offending set with a hardcoded "> 0". For a
    # study whose initial policy is 1 that wrongly named the initial policy 1.0 as missing
    # alongside the genuinely missing 2.0. The message must list ONLY 2.0 and must report the
    # inferred initial policy number as 1.0.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), update_arg_keys=(3,)
    )

    with pytest.raises(AssertionError) as excinfo:
        _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)

    message = str(excinfo.value)
    assert (
        "are not in the update function args: [2.0] (the initial policy number is 1.0)"
        in message
    )
    assert "1.0," not in message  # the initial policy is not reported as missing


def test_all_policy_nums_present_treats_zero_initial_policy_number_policynums():
    # The other end of the same inference: a study whose initial policy IS 0 must not require an
    # update for policy 0, and its message must report 0.0 as the initial policy number.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(0, 1, 2), update_arg_keys=(1, 2)
    )
    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)

    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(0, 1, 2), update_arg_keys=(1,)
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "are not in the update function args: [2.0] (the initial policy number is 0.0)"
        ),
    ):
        _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_handles_int_dtype_policy_numbers_policynums():
    # policy_num arrives as float64 in the real fixtures but int64 is legal too; neither the
    # comparison nor the message may depend on the dtype (int64 renders as [3], not [3.0]).
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), policy_num_dtype="int64", update_arg_keys=(2, 3)
    )
    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)

    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), policy_num_dtype="int64", update_arg_keys=(2,)
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "are not in the update function args: [3] (the initial policy number is 1)"
        ),
    ):
        _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_accepts_float_update_keys_against_int_policy_numbers_policynums():
    # The dtype mismatch in the other direction: float keys 2.0/3.0 against an int64 column.
    # Policy 2 and 2.0 are the same policy.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3),
        policy_num_dtype="int64",
        update_arg_keys=(2.0, 3.0),
    )

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_returns_early_when_every_active_row_is_fallback_policynums():
    # Documented early return: if every in-study row used a fallback (negative) policy there is
    # no update-produced policy to require, so even a completely empty alg_update_func_args is
    # acceptable rather than an error (and .min() of an empty Series must never be reached).
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(-1, -1), update_arg_keys=()
    )
    assert alg_update_func_args == {}

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_ignores_out_of_study_rows_policynums():
    # Only in-study rows can demand an update: policies 1, 2 and 3 here sit exclusively on
    # out-of-study rows while every active row used a fallback policy, so the check must still
    # take its early return instead of requiring updates for them.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(-1,), inactive_policy_nums=(1, 2, 3), update_arg_keys=()
    )

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_ignores_policy_seen_only_out_of_study_policynums():
    # Same rule with a live study around it: policy 4 appears only on out-of-study rows, so its
    # absence from alg_update_func_args is not a failure.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2), inactive_policy_nums=(4,), update_arg_keys=(2,)
    )

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_tolerates_nan_policy_numbers_on_out_of_study_rows_policynums():
    # Out-of-study rows in real frames carry NaN in the float64 columns. A NaN policy number on
    # an inactive row must not perturb the initial-policy inference or the required set.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3),
        inactive_policy_nums=(np.nan,),
        update_arg_keys=(2, 3),
    )

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


def test_all_policy_nums_present_passes_on_empty_analysis_df_policynums():
    # Empty frame: no active rows at all, so the early return applies and no exception (and no
    # empty-Series .min()) may escape.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(), update_arg_keys=()
    )
    assert analysis_df.empty

    _require_all_policy_nums_present_policynums(analysis_df, alg_update_func_args)


# ---------------------------------------------------------------------------
# require_no_policy_numbers_present_in_alg_update_args_but_not_analysis_df
# ---------------------------------------------------------------------------


def test_no_extra_policy_nums_passes_on_valid_study_policynums():
    # Baseline, and the int-vs-float64 pairing again: update-dict keys 2 and 3 must be
    # recognized as the analysis_df's 2.0 and 3.0 rather than reported as unknown policies.
    analysis_df, alg_update_func_args = _build_study_policynums()
    assert [type(key) for key in alg_update_func_args] == [int, int]
    assert analysis_df["policy_num"].dtype == np.dtype("float64")

    _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_raises_on_policy_absent_from_analysis_df_policynums():
    # A stale/typo'd update key (8) that names no policy in the frame at all: the update args
    # would silently contribute an estimating equation for a policy nobody ever ran.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), update_arg_keys=(2, 3, 8)
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "There are policy numbers present in algorithm update function args but not in "
            "the analysis DataFrame."
        ),
    ):
        _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_failure_message_lists_both_policy_sets_policynums():
    # The message has to name both sides for the mismatch to be actionable; each set is printed
    # on its own line, so pin the lines rather than one flat regex.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3), update_arg_keys=(2, 3, 8)
    )

    with pytest.raises(AssertionError) as excinfo:
        _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)

    message = str(excinfo.value)
    assert "alg_update_func_args policy numbers: [2, 3, 8]" in message
    assert "analysis_df policy numbers: [1.0, 2.0, 3.0]" in message


def test_no_extra_policy_nums_raises_even_when_the_extra_args_are_blank_policynums():
    # An extra key whose per-subject args are all EMPTY TUPLES is still an extra key: this check
    # has no `if not args: continue` skip, so a blank-but-present unknown policy must still be
    # reported instead of being quietly waved through as "not applicable".
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3),
        update_arg_keys=(2, 3, 8),
        blank_update_arg_keys=(8,),
    )
    assert alg_update_func_args[8] == {0: (), 1: ()}

    with pytest.raises(
        AssertionError,
        match=re.escape("alg_update_func_args policy numbers: [2, 3, 8]"),
    ):
        _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_accepts_policy_present_only_on_out_of_study_rows_policynums():
    # This check looks at the WHOLE frame, not just active rows: policy 4 exists in the study
    # (on out-of-study rows) so an update keyed at 4 is not an unknown policy.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2), inactive_policy_nums=(4,), update_arg_keys=(2, 4)
    )

    _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_accepts_fallback_policy_key_present_in_analysis_df_policynums():
    # A negative (fallback) policy number is a legal key as long as the frame contains it; the
    # check must not special-case the sign.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, -1), update_arg_keys=(2, -1)
    )

    _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_passes_on_empty_update_args_and_empty_analysis_df_policynums():
    # Degenerate both-empty input: the empty key set is a subset of the empty policy set, so
    # this must pass rather than raise or trip over an empty .unique().
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(), update_arg_keys=()
    )
    assert analysis_df.empty and alg_update_func_args == {}

    _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_raises_on_empty_analysis_df_with_update_args_policynums():
    # An empty frame with update args supplied is the pathological wiring this check exists to
    # catch; the empty analysis_df policy list must be reported, not crash the formatter.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(), update_arg_keys=(2,)
    )

    with pytest.raises(
        AssertionError, match=re.escape("analysis_df policy numbers: [].")
    ):
        _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)


def test_no_extra_policy_nums_accepts_float_keys_matching_int_policy_numbers_policynums():
    # Mirror of the baseline dtype pairing: float keys 2.0/3.0 against an int64 policy column.
    analysis_df, alg_update_func_args = _build_study_policynums(
        active_policy_nums=(1, 2, 3),
        policy_num_dtype="int64",
        update_arg_keys=(2.0, 3.0),
    )

    _require_no_extra_policy_nums_policynums(analysis_df, alg_update_func_args)
