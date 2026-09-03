"""
Coverage for the input checks that route through helper_functions.confirm_input_check_result --
the two purely interactive confirmations plus the legacy raw-units sum-to-zero check:

  * confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied
  * verify_analysis_df_summary_satisfactory
  * require_estimating_functions_sum_to_zero

confirm_input_check_result calls builtins.input() in a `while answer != "y"` loop whenever
suppress_interactive_data_checks is False, so EVERY non-suppressed call below monkeypatches
builtins.input. An unpatched one would block the test run forever, and one patched to return
anything other than "y"/"n" forever would spin in the re-prompt loop.
"""

import re

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks

# ---------------------------------------------------------------------------
# builtins.input harnesses. Every non-suppressed call in this file goes through one of these:
# _capture_input_interactive (records the prompt, answers from a finite script and then "y", so
# it can never spin) or _forbid_input_interactive (turns "this check prompted when it should
# not have" into a failure instead of a hang).
# ---------------------------------------------------------------------------


def _capture_input_interactive(monkeypatch, answers=("y",)):
    """
    Patch builtins.input to record each prompt and answer from `answers`, falling back to "y"
    once the script is exhausted so the confirm loop always terminates. Returns the (growing)
    list of prompts the check passed to confirm_input_check_result.
    """
    prompts = []
    remaining = list(answers)

    def _answer(prompt):
        prompts.append(prompt)
        return remaining.pop(0) if remaining else "y"

    monkeypatch.setattr("builtins.input", _answer)
    return prompts


def _forbid_input_interactive(monkeypatch):
    """Patch builtins.input to fail loudly: this check must not prompt at all."""

    def _fail(prompt):
        raise AssertionError(
            f"builtins.input was called when no interactive confirmation was expected: {prompt!r}"
        )

    monkeypatch.setattr("builtins.input", _fail)


def _pin_terminal_size_interactive(monkeypatch, columns=80, lines=24):
    """
    plotext sizes every figure from shutil.get_terminal_size() at plt.clear_figure() time, and
    shutil honours COLUMNS/LINES ahead of the real terminal -- so pinning them is what makes
    assertions about the rendered canvas independent of whoever's terminal runs pytest.
    """
    monkeypatch.setenv("COLUMNS", str(columns))
    monkeypatch.setenv("LINES", str(lines))


_ANSI_ESCAPE_INTERACTIVE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi_interactive(text):
    """plotext embeds SGR colour escapes in the plots it builds into the prompt."""
    return _ANSI_ESCAPE_INTERACTIVE.sub("", text)


def _summary_bullets_interactive(prompt):
    """
    The bullet list at the head of verify_analysis_df_summary_satisfactory's prompt, i.e.
    everything before the first embedded plotext canvas.
    """
    return prompt[: prompt.index("* The following trajectories")]


# ---------------------------------------------------------------------------
# confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied: a pure
# confirmation, gated on BOTH the action-prob index and the previous-betas index being absent.
# "Absent" is encoded as a negative index, so 0 is a supplied index, not a missing one.
# ---------------------------------------------------------------------------


def _confirm_no_action_probs_interactive(
    *, action_prob_index=-1, previous_betas_index=-1, suppress=True
):
    input_checks.confirm_action_probabilities_not_in_alg_update_args_if_index_not_supplied(
        action_prob_index,
        previous_betas_index,
        suppress,
    )


def test_confirm_no_action_probs_returns_without_prompting_when_suppressed_interactive(
    monkeypatch,
):
    # The suppressed path must never reach builtins.input, even though both indices are absent
    # and the confirmation therefore applies: this is the path every batch/cluster run takes.
    _forbid_input_interactive(monkeypatch)

    _confirm_no_action_probs_interactive(suppress=True)


def test_confirm_no_action_probs_prompt_names_both_absent_argument_kinds_interactive(
    monkeypatch,
):
    # The prompt is the entire value of this check, so it has to say WHICH claim the user is
    # being asked to confirm (no action probabilities AND no previous betas in the update args)
    # and end in the y/n question confirm_input_check_result loops on.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))

    _confirm_no_action_probs_interactive(suppress=False)

    assert len(prompts) == 1
    assert (
        "does not have action probabilities or previous betas in its arguments"
        in (prompts[0])
    )
    assert prompts[0].strip().endswith("Continue? (y/n)")


def test_confirm_no_action_probs_declining_raises_system_exit_interactive(monkeypatch):
    # Answering "n" aborts the analysis. No `error` is passed at this call site, so the
    # SystemExit is raised bare rather than chained from an AssertionError.
    _capture_input_interactive(monkeypatch, answers=("n",))

    with pytest.raises(SystemExit) as excinfo:
        _confirm_no_action_probs_interactive(suppress=False)

    assert excinfo.value.__cause__ is None


def test_confirm_no_action_probs_not_asked_when_action_prob_index_supplied_interactive(
    monkeypatch,
):
    # A supplied action-prob index means the args demonstrably DO carry action probabilities,
    # so there is nothing to confirm: input must not be reached even un-suppressed.
    _forbid_input_interactive(monkeypatch)

    _confirm_no_action_probs_interactive(
        action_prob_index=2, previous_betas_index=-1, suppress=False
    )


def test_confirm_no_action_probs_not_asked_when_only_previous_betas_supplied_interactive(
    monkeypatch,
):
    # The gate is an AND over both indices: supplying previous_betas alone also suppresses the
    # confirmation, even though the action-prob index is still absent. Pins the current
    # behavior (see implementation_concerns), not an endorsement of it.
    _forbid_input_interactive(monkeypatch)

    _confirm_no_action_probs_interactive(
        action_prob_index=-1, previous_betas_index=1, suppress=False
    )


def test_confirm_no_action_probs_index_zero_counts_as_supplied_interactive(monkeypatch):
    # Index 0 is a legitimate argument position; only a NEGATIVE index means "not supplied".
    # A truthiness test instead of `< 0` would treat 0 as absent and prompt here.
    _forbid_input_interactive(monkeypatch)

    _confirm_no_action_probs_interactive(
        action_prob_index=0, previous_betas_index=-1, suppress=False
    )
    _confirm_no_action_probs_interactive(
        action_prob_index=-1, previous_betas_index=0, suppress=False
    )


def test_confirm_no_action_probs_reprompts_on_unrecognized_answer_interactive(
    monkeypatch,
):
    # confirm_input_check_result loops until it sees "y" or "n", lower-casing each answer:
    # a typo re-prompts with the same message, and "Y" is accepted. This is exactly why every
    # patched input in this file eventually returns "y" -- anything else spins forever.
    prompts = _capture_input_interactive(monkeypatch, answers=("maybe", "Y"))

    _confirm_no_action_probs_interactive(suppress=False)

    assert len(prompts) == 2
    assert prompts[0] == prompts[1]


# ---------------------------------------------------------------------------
# verify_analysis_df_summary_satisfactory: computes a study summary over the ACTIVE rows,
# renders two plotext trajectories into the prompt, and asks the user to confirm it.
#
# The builder mirrors the real-world fixture shape: policy_num is float64 and its INITIAL value
# is 1 (not 0), the active column is int64 0/1 named in_study, and the action column is float64
# holding NaN on out-of-study rows.
# ---------------------------------------------------------------------------


def _build_summary_study_interactive(
    *,
    num_subjects=3,
    num_decision_times=24,
    decision_times_per_policy=2,
    initial_policy_num=1,
    fallback_decision_times=(),
    out_of_study_cells=(),
    poison_out_of_study_rows=False,
    duplicate_policy_at_time=None,
    rotate_action_probs=False,
    integer_policy_dtype=False,
):
    """
    A rectangular study: `num_subjects` subjects x `num_decision_times` decision times, with a
    new policy every `decision_times_per_policy` times starting at `initial_policy_num`.

    fallback_decision_times marks whole decision times as fallback (policy_num -1).
    out_of_study_cells is (decision_time, subject_index) pairs marked in_study=0; their action /
    action_prob / reward are NaN, or -- with poison_out_of_study_rows=True -- wild finite values,
    which is what distinguishes "filtered by the active column" from "filtered by NaN".
    duplicate_policy_at_time bumps subject 0's policy at that time so the decision time carries
    two distinct non-fallback policies.
    """
    out_of_study_cells = set(out_of_study_cells)
    rows = []
    for decision_time in range(num_decision_times):
        for subject_index in range(num_subjects):
            if decision_time in fallback_decision_times:
                policy_num = -1
            else:
                policy_num = (
                    initial_policy_num + decision_time // decision_times_per_policy
                )
                if duplicate_policy_at_time == decision_time and subject_index == 0:
                    policy_num += 1
            active = 0 if (decision_time, subject_index) in out_of_study_cells else 1
            if rotate_action_probs:
                action_prob = 0.1 + 0.1 * ((subject_index + decision_time) % 5)
            else:
                action_prob = 0.3 + 0.1 * subject_index
            reward = float(subject_index) + 0.25 * (decision_time % 4)
            if not active:
                action_prob = 99.0 if poison_out_of_study_rows else np.nan
                reward = -99.0 if poison_out_of_study_rows else np.nan
            rows.append(
                {
                    "user_id": f"s{subject_index}",
                    "calendar_t": decision_time,
                    "policy_num": policy_num
                    if integer_policy_dtype
                    else float(policy_num),
                    "in_study": active,
                    "action": float(subject_index % 2) if active else np.nan,
                    "action_prob": action_prob,
                    "reward": reward,
                }
            )

    analysis_df = pd.DataFrame(rows)
    analysis_df["in_study"] = analysis_df["in_study"].astype("int64")
    analysis_df["policy_num"] = analysis_df["policy_num"].astype(
        "int64" if integer_policy_dtype else "float64"
    )
    return analysis_df


def _verify_summary_interactive(analysis_df, *, beta_dim=3, theta_dim=2, suppress=True):
    input_checks.verify_analysis_df_summary_satisfactory(
        analysis_df,
        "user_id",
        "policy_num",
        "calendar_t",
        "in_study",
        "action_prob",
        "reward",
        beta_dim,
        theta_dim,
        suppress,
    )


def test_summary_check_returns_without_prompting_when_suppressed_interactive(
    monkeypatch,
):
    # Suppressed: the summary and both plots are still computed (so a crash in the plotting
    # math would surface here) but nothing is asked of the user.
    _forbid_input_interactive(monkeypatch)

    _verify_summary_interactive(_build_summary_study_interactive(), suppress=True)


def test_summary_prompt_reports_the_study_shape_and_both_trajectories_interactive(
    monkeypatch,
):
    # The prompt is the whole output of this check, so pin every line of the summary it asks
    # the user to sign off on. 3 subjects x 24 decision times, a new policy every 2 times
    # (policies 1..12), no fallback rows and no out-of-study rows.
    _pin_terminal_size_interactive(monkeypatch)
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))

    _verify_summary_interactive(
        _build_summary_study_interactive(), beta_dim=3, theta_dim=2, suppress=False
    )

    assert len(prompts) == 1
    bullets = _summary_bullets_interactive(prompts[0])
    assert "* 3 subjects" in bullets
    assert (
        "* 24 decision times, for an average of 24.0 decisions per subject" in bullets
    )
    assert "* RL parameters of dimension 3 per update" in bullets
    assert "* Inferential target of dimension 2" in bullets
    # Policy 1 spans decision times 0 and 1, i.e. 2 x 3 = 6 rows.
    assert "* 6 data points before the first update" in bullets
    assert "* 0 decision times (0.0%) for which fallback policies were used" in bullets
    assert (
        "* 0 decision times (0.0%) for which multiple non-fallback policies were used"
        in bullets
    )
    assert "* Minimum action probability 0.3" in bullets
    assert "* Maximum action probability 0.5" in bullets

    # Both plotext canvases are actually built into the prompt, with their axis labels and the
    # decimated x ticks (24 points // 10 -> every 2nd decision time).
    plain = _strip_ansi_interactive(prompts[0])
    assert "Action 1 Probability 25/50/75 Quantile Trajectories" in plain
    assert "Action 1 Probability Quantiles" in plain
    assert "Avg Reward Trajectory" in plain
    assert plain.count("Decision Time") == 2
    tick_row = [line for line in plain.splitlines() if line.strip().startswith("0  ")]
    assert tick_row, plain
    assert tick_row[0].split() == [
        "0",
        "2",
        "4",
        "6",
        "8",
        "10",
        "12",
        "14",
        "16",
        "18",
        "20",
        "22",
    ]
    assert plain.strip().endswith("Does this meet expectations? (y/n)")


def test_summary_declining_the_summary_raises_system_exit_interactive(monkeypatch):
    # "n" means the summary did not match what the user believes they ran: abort rather than
    # analyze the wrong dataset. No `error` is threaded here, so SystemExit is unchained.
    _capture_input_interactive(monkeypatch, answers=("n",))

    with pytest.raises(SystemExit) as excinfo:
        _verify_summary_interactive(
            _build_summary_study_interactive(num_decision_times=6), suppress=False
        )

    assert excinfo.value.__cause__ is None


def test_summary_counts_only_rows_the_active_column_marks_in_study_interactive(
    monkeypatch,
):
    # Every statistic is taken over active rows only. Subject s3 is never in study, and cell
    # (0, 0) drops out of an otherwise-complete grid; out-of-study rows carry FINITE poison
    # values (99.0 / -99.0) rather than NaN, so a check that leaned on NaN-skipping pandas
    # aggregations instead of the active column would report 99.0 as the maximum action
    # probability and 4 subjects.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_subjects=4,
        num_decision_times=24,
        out_of_study_cells=[(t, 3) for t in range(24)] + [(0, 0)],
        poison_out_of_study_rows=True,
    )

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert "* 3 subjects" in bullets
    # 24 * 3 active rows minus the one dropped cell, over 3 subjects.
    assert "* 24 decision times, for an average of 23.666666666666668" in bullets
    assert "* 5 data points before the first update" in bullets
    assert "* Minimum action probability 0.3" in bullets
    assert "* Maximum action probability 0.5" in bullets
    assert "99" not in bullets


def test_summary_reports_fallback_rows_not_fallback_decision_times_interactive(
    monkeypatch,
):
    # num_decision_times_with_fallback_policies is len(rows with policy_num < 0), a ROW count,
    # but it is reported as a count of decision times and divided by the number of decision
    # times. With 3 subjects fallback across 3 of 4 decision times that is 9 "decision times"
    # out of 4, i.e. 225%. Pins the arithmetic that is actually shipped (see
    # implementation_concerns) so a fix has to update this test deliberately.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_decision_times=4, fallback_decision_times=(0, 1, 2)
    )

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert (
        "* 9 decision times (225.0%) for which fallback policies were used" in bullets
    )
    # Only decision time 3 is non-fallback, and it is the first policy still visible there.
    assert "* 3 data points before the first update" in bullets


def test_summary_counts_the_non_zero_initial_policy_as_an_update_interactive(
    monkeypatch,
):
    # num_non_initial_or_fallback_policies filters on policy_num > 0, which only excludes the
    # initial policy when the initial policy is numbered 0. Real fixtures number it 1, so all
    # 12 distinct policies -- initial included -- are reported as "12 policy updates" when only
    # 11 updates happened. The 0-based study below is the case the filter was written for.
    prompts = _capture_input_interactive(monkeypatch, answers=("y", "y"))

    _verify_summary_interactive(
        _build_summary_study_interactive(initial_policy_num=1), suppress=False
    )
    assert "* 12 policy updates" in _summary_bullets_interactive(prompts[0])

    _verify_summary_interactive(
        _build_summary_study_interactive(initial_policy_num=0), suppress=False
    )
    assert "* 11 policy updates" in _summary_bullets_interactive(prompts[1])
    # Either way the first policy's rows are what "before the first update" counts, so that
    # line is unaffected by the numbering convention.
    for prompt in prompts:
        assert (
            "* 6 data points before the first update"
            in _summary_bullets_interactive(prompt)
        )


def test_summary_survives_a_single_decision_time_interactive(monkeypatch):
    # One decision time makes len(trajectory) // 10 == 0, and range(0, 1, 0) is a ValueError:
    # the max(1, ...) floor in both xticks calls is what keeps this from blowing up. plt.error
    # and plt.scatter also have to cope with a one-point series.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))

    _verify_summary_interactive(
        _build_summary_study_interactive(num_decision_times=1),
        beta_dim=1,
        theta_dim=1,
        suppress=False,
    )

    bullets = _summary_bullets_interactive(prompts[0])
    assert "* 1 decision times, for an average of 1.0 decisions per subject" in bullets
    assert "* 3 data points before the first update" in bullets


def test_summary_handles_a_study_that_is_entirely_fallback_interactive(monkeypatch):
    # With no non-fallback rows at all, min_non_fallback_policy_num is NaN and the
    # policy_num == NaN comparison matches nothing, so "data points before the first update"
    # silently reports 0 instead of flagging that there was no first update.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_decision_times=4, fallback_decision_times=(0, 1, 2, 3)
    )

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert "* 0 policy updates" in bullets
    assert "* 0 data points before the first update" in bullets
    assert (
        "* 12 decision times (300.0%) for which fallback policies were used" in bullets
    )
    assert (
        "* 0 decision times (0.0%) for which multiple non-fallback policies were used"
        in bullets
    )


def test_summary_flags_decision_times_serving_multiple_policies_interactive(
    monkeypatch,
):
    # A decision time at which subjects sit on different non-fallback policies (a staggered
    # update) is counted separately; fallback rows are excluded from that groupby, so a study
    # with fallback rows at other times must not inflate the count.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_decision_times=12, duplicate_policy_at_time=5, fallback_decision_times=(11,)
    )

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert (
        "* 1 decision times (8.333333333333334%) for which multiple non-fallback"
        " policies were used" in bullets
    )
    assert "* 3 decision times (25.0%) for which fallback policies were used" in bullets


def test_summary_raises_zero_division_when_no_row_is_active_interactive(monkeypatch):
    # An analysis_df whose active column is 0 everywhere (or whose active column name points at
    # the wrong column) leaves num_subjects == 0, and avg_decisions_per_subject divides by it.
    # The failure is a bare ZeroDivisionError from the summary arithmetic rather than an input
    # check message -- pinned here so the behavior is at least known.
    _forbid_input_interactive(monkeypatch)
    analysis_df = _build_summary_study_interactive(
        num_decision_times=4,
        out_of_study_cells=[(t, s) for t in range(4) for s in range(3)],
    )

    with pytest.raises(ZeroDivisionError):
        _verify_summary_interactive(analysis_df, suppress=True)


def test_summary_handles_integer_dtype_policy_numbers_interactive(monkeypatch):
    # Real fixtures store policy_num as float64, but an int64 column has to summarize the same
    # way: the > 0 / >= 0 / < 0 comparisons and the min-policy equality must not depend on the
    # column's dtype.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_decision_times=4, integer_policy_dtype=True
    )
    assert analysis_df["policy_num"].dtype == np.int64

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert "* 2 policy updates" in bullets
    assert "* 6 data points before the first update" in bullets


def test_summary_quantile_trajectory_spans_varying_action_probabilities_interactive(
    monkeypatch,
):
    # Action probabilities that rotate across subjects AND decision times give the 25/50/75
    # groupby-quantile a genuinely different answer at each decision time (and a non-constant
    # yerr for plt.error), unlike the flat default study.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))
    analysis_df = _build_summary_study_interactive(
        num_decision_times=20, rotate_action_probs=True
    )

    _verify_summary_interactive(analysis_df, suppress=False)

    bullets = _summary_bullets_interactive(prompts[0])
    assert "* Minimum action probability 0.1" in bullets
    assert "* Maximum action probability 0.5" in bullets
    # Sanity-check that the trajectory the plot was handed really does vary over time.
    active_df = analysis_df[analysis_df["in_study"] == 1]
    medians = active_df.groupby("calendar_t")["action_prob"].median().to_numpy()
    assert medians.min() < medians.max()


# ---------------------------------------------------------------------------
# require_estimating_functions_sum_to_zero (LEGACY, superseded by the SE-standardized version):
# two nested np.testing.assert_allclose gates against the zero vector, atol=1e-2 (hard, raises)
# then atol=5e-4 (soft, interactive). Both tolerances are absolute and in the estimating
# equations' own reward-scale units, which is exactly the non-portability docs/adr/0002 records.
# ---------------------------------------------------------------------------


def _require_legacy_sum_to_zero_interactive(
    residual, *, beta_dim=2, theta_dim=2, suppress=True
):
    input_checks.require_estimating_functions_sum_to_zero(
        jnp.asarray(residual), beta_dim, theta_dim, suppress
    )


def test_legacy_sum_to_zero_passes_on_a_clean_residual_interactive(monkeypatch):
    # Well inside both tolerances: no prompt on either the suppressed or the un-suppressed
    # path. Any prompt at all fails the test rather than blocking on stdin.
    _forbid_input_interactive(monkeypatch)

    _require_legacy_sum_to_zero_interactive(np.full(4, 1e-5), suppress=True)
    _require_legacy_sum_to_zero_interactive(np.full(4, 1e-5), suppress=False)


def test_legacy_sum_to_zero_hard_gate_raises_and_never_prompts_interactive(monkeypatch):
    # A residual past the loose atol=1e-2 is a hard failure: it re-raises the assert_allclose
    # AssertionError regardless of suppress_interactive_data_checks, so the user is never
    # offered the chance to wave it through.
    _forbid_input_interactive(monkeypatch)
    residual = np.array([0.0, 0.05, 0.0, 0.0])

    for suppress in (True, False):
        with pytest.raises(
            AssertionError,
            match=re.escape("Not equal to tolerance rtol=1e-07, atol=0.01"),
        ) as excinfo:
            _require_legacy_sum_to_zero_interactive(residual, suppress=suppress)
        assert "Max absolute difference: 0.05" in str(excinfo.value)


def test_legacy_sum_to_zero_soft_band_returns_when_suppressed_interactive(monkeypatch):
    # Between the two tolerances (5e-4 < 6e-4 <= 1e-2) the check only wants confirmation, so
    # suppressing interaction must log-and-continue rather than raise.
    _forbid_input_interactive(monkeypatch)

    _require_legacy_sum_to_zero_interactive(np.full(4, 6e-4), suppress=True)


def test_legacy_sum_to_zero_soft_band_prompt_carries_the_allclose_report_interactive(
    monkeypatch,
):
    # The per-update/inference breakdown only goes to the logger, and this package installs no
    # handler -- so what the user actually sees to judge is whatever the prompt embeds: the
    # tight tolerance and the offending values from the assert_allclose report.
    prompts = _capture_input_interactive(monkeypatch, answers=("y",))

    _require_legacy_sum_to_zero_interactive(np.full(4, 6e-4), suppress=False)

    assert len(prompts) == 1
    assert (
        "Estimating functions do not average to within default tolerance of zero vector"
        in prompts[0]
    )
    assert "Not equal to tolerance rtol=1e-07, atol=0.0005" in prompts[0]
    assert "Mismatched elements: 4 / 4" in prompts[0]
    assert "Max absolute difference: 0.0006" in prompts[0]
    assert prompts[0].strip().endswith("Continue? (y/n)")


def test_legacy_sum_to_zero_soft_band_declined_chains_the_assertion_interactive(
    monkeypatch,
):
    # This call site DOES pass the AssertionError to confirm_input_check_result, so declining
    # raises `SystemExit from error`: the numeric failure survives as __cause__ in the
    # traceback instead of being replaced by a bare exit.
    _capture_input_interactive(monkeypatch, answers=("n",))

    with pytest.raises(SystemExit) as excinfo:
        _require_legacy_sum_to_zero_interactive(np.full(4, 6e-4), suppress=False)

    assert isinstance(excinfo.value.__cause__, AssertionError)
    assert "atol=0.0005" in str(excinfo.value.__cause__)


def test_legacy_sum_to_zero_tolerances_are_absolute_and_not_scale_portable_interactive(
    monkeypatch,
):
    # The reason this check was superseded: both gates are fixed absolute numbers in the
    # equations' own units. The SAME residual pattern passes cleanly at one reward scale and
    # hard-fails at 100x, with nothing about the fit having changed.
    _forbid_input_interactive(monkeypatch)
    residual = np.full(4, 4.9e-4)

    _require_legacy_sum_to_zero_interactive(residual, suppress=True)

    with pytest.raises(AssertionError):
        _require_legacy_sum_to_zero_interactive(residual * 100.0, suppress=True)


def test_legacy_sum_to_zero_nonfinite_residual_hard_fails_interactive(monkeypatch):
    # A NaN component is not "within atol of zero": assert_allclose reports it as a nan
    # location mismatch, which lands on the hard-failure branch rather than the prompt.
    _forbid_input_interactive(monkeypatch)

    with pytest.raises(AssertionError, match="nan location mismatch"):
        _require_legacy_sum_to_zero_interactive(
            np.array([np.nan, 0.0, 0.0, 0.0]), suppress=True
        )

    with pytest.raises(AssertionError):
        _require_legacy_sum_to_zero_interactive(
            np.array([np.inf, 0.0, 0.0, 0.0]), suppress=True
        )


def test_legacy_sum_to_zero_stack_shape_is_only_validated_on_failure_interactive(
    monkeypatch,
):
    # The `(size - theta_dim) % beta_dim == 0` internal consistency assert lives inside both
    # failure branches, so a stack whose length cannot be split into beta blocks plus theta
    # passes silently on clean data...
    _forbid_input_interactive(monkeypatch)
    misshapen = np.full(5, 1e-5)

    _require_legacy_sum_to_zero_interactive(misshapen, beta_dim=2, theta_dim=2)

    # ...and when it does fire, the bare internal AssertionError REPLACES the numeric
    # assert_allclose report the user needs, on both the hard and the soft branch.
    for value in (0.05, 6e-4):
        with pytest.raises(AssertionError) as excinfo:
            _require_legacy_sum_to_zero_interactive(
                np.full(5, value), beta_dim=2, theta_dim=2
            )
        assert str(excinfo.value) == ""


def test_legacy_sum_to_zero_accepts_an_empty_stack_interactive(monkeypatch):
    # A degenerate stack with no components at all compares equal to jnp.zeros(0) and is
    # accepted without comment: there is nothing here to reject, but nothing was checked either.
    _forbid_input_interactive(monkeypatch)

    _require_legacy_sum_to_zero_interactive(
        np.array([]), beta_dim=2, theta_dim=0, suppress=True
    )


def test_legacy_sum_to_zero_accepts_float64_and_integer_stacks_interactive(monkeypatch):
    # Callers hand this either a jax float32 array or a plain float64/integer numpy array; the
    # comparison against jnp.zeros(size) has to cope with all of them. An exact integer zero
    # stack is the trivially-passing case.
    _forbid_input_interactive(monkeypatch)

    float64_residual = np.full(4, 1e-5, dtype=np.float64)
    assert float64_residual.dtype == np.float64
    input_checks.require_estimating_functions_sum_to_zero(float64_residual, 2, 2, True)

    integer_residual = np.zeros(4, dtype=np.int64)
    input_checks.require_estimating_functions_sum_to_zero(integer_residual, 2, 2, True)
