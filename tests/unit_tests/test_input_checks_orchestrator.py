"""
Integration coverage for the input-check ORCHESTRATORS,
input_checks.perform_first_wave_input_checks and, for the handful of dataframe-only checks that
moved ahead of the caller's theta_calculation_func,
input_checks.perform_conditional_zeroth_wave_dataframe_checks.

Every test here goes through an orchestrator rather than a single check function, because the
question these tests answer is not "is this check correct?" (the sibling modules cover that) but
"is this check actually REACHED, with the arguments the orchestrator threads into it?". A check
belongs to exactly ONE wave -- the zeroth wave's three are not repeated in the first -- so the
test for each goes through the wave that owns it. The
golden-path test is the anchor: one small study that satisfies every first-wave check at once,
with a NON-ZERO initial policy number (active policies 1, 2, 3 and alg_update_func_args keyed
2, 3, mirroring tests/benchmarks/fixtures/small where policy_num is float64 1.0..7.0). Each
other test perturbs exactly one input away from that study.

None of these tests may run with suppress_interactive_data_checks=False without monkeypatching
builtins.input: verify_analysis_df_summary_satisfactory always reaches
helper_functions.confirm_input_check_result, which calls input() and would hang forever.

The golden study's analysis DataFrame carries a `state` column that nothing in it is compared
against, and _inference_func_orch's non-theta parameters are named after that column and
`reward` on purpose: require_inference_func_parameter_names_are_analysis_df_columns requires
every inference_func parameter except the theta one to name a real analysis_df column, because
post_deployment_analysis.process_inference_func_args fills each parameter from the column of
that name. A study whose inference function declares a parameter no column matches cannot be
analyzed at all, so it is not a valid golden study.
"""

import re

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks
from lifejacket.constants import FunctionTypes

_SUBJECT_IDS_ORCH = (1, 2)

# The beta in force for each policy number. Policy 1 is the INITIAL policy (produced by no
# update, so it never appears in alg_update_func_args); policies 2 and 3 are produced by the
# two updates.
_BETA_BY_POLICY_NUM_ORCH = {
    1: jnp.array([0.4, -0.3]),
    2: jnp.array([0.9, 0.2]),
    3: jnp.array([-0.5, 0.7]),
}
_BETA_DIM_ORCH = 2


def _action_prob_func_orch(beta, features):
    """Declares exactly TWO parameters, so every supplied action-prob tuple must have two."""
    return jax.nn.sigmoid(jnp.dot(beta, features))


def _alg_update_func_orch(
    beta, previous_betas, action_probs, action_prob_times, rewards
):
    """
    Declares exactly FIVE parameters: beta at 0, previous betas at 1, action probabilities at
    2, their times at 3, rewards at 4. The first wave never calls this; it only inspects the
    signature, so the body just has to be plausible.
    """
    del previous_betas, action_prob_times
    return jnp.sum((rewards - action_probs * jnp.sum(beta)) ** 2)


def _masked_alg_update_func_orch(
    beta, previous_betas, action_probs, action_prob_times, rewards, mask
):
    """
    The mask-aware counterpart: SIX declared parameters for the same five supplied ones,
    because self_pad_ragged_args_and_build_mask appends the validity mask as a new last
    argument.
    """
    del previous_betas, action_prob_times
    return jnp.sum(mask * (rewards - action_probs * jnp.sum(beta)) ** 2)


def _inference_func_orch(theta, state, reward):
    """
    Declares THREE parameters, with theta at index 0.

    Its two NON-theta parameters are named `state` and `reward` because each must name an
    analysis DataFrame column: process_inference_func_args builds this function's argument
    tuples by looking up the column whose name equals each parameter's name, and
    require_inference_func_parameter_names_are_analysis_df_columns enforces exactly that. The
    theta position is exempt -- it is filled from theta_est, not from a column -- so its name
    is unconstrained.
    """
    features = jnp.array([1.0, state, state**2])
    return (reward - jnp.dot(theta, features)) * features


def _state_orch(decision_time, subject_id):
    """
    The scalar covariate the analysis DataFrame's `state` column carries, and the varying
    component of the action-prob features.
    """
    return 0.1 * decision_time + 0.5 * subject_id


def _features_orch(decision_time, subject_id):
    return jnp.array([1.0, _state_orch(decision_time, subject_id)])


def _reward_orch(decision_time, subject_id):
    return 1.0 + 0.1 * decision_time + 0.5 * subject_id


def _assert_first_wave_passes_orch(study):
    """
    RETARGETED on 2026-09-02: perform_first_wave_input_checks used to return None, and eleven
    tests pinned that. It now returns the measurements the diagnostic summary reports --
    currently the action-probability reconstruction's agreement -- so a passing run is pinned
    by the SHAPE of that return instead: the measurement must exist, be finite, and sit within
    the reconstruction's own tolerance, because a pass with a missing or out-of-tolerance
    measurement would mean the check and its report have drifted apart.
    """
    measurements = input_checks.perform_first_wave_input_checks(**study)
    reconstruction = measurements["action_prob_reconstruction"]
    assert set(reconstruction) == {"max_abs_difference", "num_cells", "atol"}
    assert reconstruction["num_cells"] > 0
    assert 0.0 <= reconstruction["max_abs_difference"] <= reconstruction["atol"]


def _zeroth_wave_kwargs_orch(study):
    """
    The subset of a study's keyword arguments that
    input_checks.perform_conditional_zeroth_wave_dataframe_checks takes.

    That wave runs BEFORE the caller's theta_calculation_func, so it sees only the frame and
    the column names -- none of the functions, argument tuples or indices the first wave
    needs. Its checks are no longer repeated inside the first wave, so the tests that pin them
    go through this instead of through _build_study_orch's full kwargs.
    """
    return {
        key: study[key]
        for key in (
            "analysis_df",
            "active_col_name",
            "action_col_name",
            "policy_num_col_name",
            "calendar_t_col_name",
            "subject_id_col_name",
            "action_prob_col_name",
            "reward_col_name",
        )
    }


def _build_study_orch(
    *,
    stale_action_prob_beta=False,
    extra_action_prob_arg=False,
    short_alg_update_arg=False,
    blank_args_at_active_cell=False,
    nonblank_args_at_inactive_cell=False,
    gap_in_calendar_times=False,
    drop_last_update=False,
    integer_policy_num_dtype=False,
    mask_aware=False,
    mismatched_shape_action_prob_beta=False,
    policy_number_regresses=False,
    beta_disagrees_within_policy=False,
    mixed_policies_at_one_time=False,
):
    """
    Build the keyword arguments for one call to perform_first_wave_input_checks.

    The valid study: subjects 1 and 2, consecutive calendar times 1..4, policies 1 (initial),
    2, 3 in force at times 1, 2, 3, 4 respectively (policy 3 stays in force for the last two
    times), and subject 2 out of study at the last time. So alg_update_func_args is keyed by
    policy numbers 2 and 3 -- the two update-produced policies -- and THE INITIAL POLICY NUMBER
    IS 1, NOT 0. Recorded action probabilities are computed with the very same
    action_prob_func/args pairs that are handed to the checks, and the action probabilities
    inside alg_update_func_args are read back out of those recorded values, so the
    reconstruction and correspondence checks agree by construction.

    Scalar wiring parameters (indices, function types, the mask index) are NOT flags here --
    tests override those directly on the returned dict, which is exactly the kind of
    misconfiguration they model.

    Every subject is active for one unbroken stretch, no (subject_id, calendar_t) pair repeats,
    each subject's non-fallback policy numbers only increase, every value on an active row is
    finite, every supplied argument is finite, every beta is shaped (beta_dim,) and each
    previous-betas block is shaped (that policy's index in beta_index_by_policy_num, beta_dim)
    -- so the study satisfies the checks added on 2026-09-02 as well as the older ones.

    Flags, each introducing exactly one defect:
      stale_action_prob_beta: record the PRE-update beta (policy 2's) for EVERY decision time
        where policy 3 is in force, and derive those times' recorded probabilities from it, so
        reconstruction still passes but the recorded beta disagrees with update 3's beta.
        Applied at every policy-3 time rather than just one of them so that the betas still
        AGREE across the cells sharing policy 3: otherwise
        require_betas_match_in_action_prob_func_args_each_policy, which runs first, is the
        check that fires instead of the recorded-vs-update comparison.
      extra_action_prob_arg: ONE action-prob tuple carries three values instead of two, so the
        supplied widths disagree with each other (not merely with the signature).
      short_alg_update_arg: ONE alg-update tuple carries four values instead of five, likewise
        making the supplied widths mutually inconsistent. Narrowing EVERY tuple instead --
        _narrow_all_alg_update_tuples_orch -- is the separate, consistent-but-uncallable case.
      blank_args_at_active_cell: an empty action-prob tuple at a cell marked in study.
      nonblank_args_at_inactive_cell: a real action-prob tuple at the out-of-study cell.
      gap_in_calendar_times: calendar times 1, 2, 3, 5 instead of 1, 2, 3, 4.
      drop_last_update: alg_update_func_args omits policy 3 entirely.
      integer_policy_num_dtype: policy_num is int64 (no NaN on out-of-study rows) instead of
        the float64 the repo's own fixtures use.
      mask_aware: use the six-parameter mask-aware update function, with the mask index at the
        supplied tuple length and the genuinely ragged positions self-padded.
      mismatched_shape_action_prob_beta: record a one-component beta (with matching
        one-component features, so the recorded probabilities still reconstruct) for the
        decision times where policy 3 is in force, so the recorded beta disagrees with update
        3's two-component beta in SHAPE rather than in value.
      policy_number_regresses: the last decision time runs under policy 2 again, after policy
        3 was in force at the time before it. Every cell under policy 2 records policy 2's
        beta, so the per-policy beta invariant still holds and only the temporal ordering is
        wrong.
      beta_disagrees_within_policy: the last decision time (policy 3, same as the time before
        it) records a DIFFERENT two-component beta, with its recorded probabilities derived
        from that beta so reconstruction still passes. Two cells sharing a policy number then
        disagree about that policy's beta.
      mixed_policies_at_one_time: at the third decision time the second subject stays on policy
        2 while the first moves to policy 3, each recording its own policy's beta. This is a
        VALID study -- two different policies at one decision time is supported -- and is the
        configuration the old per-decision-time beta check wrongly rejected.
    """
    calendar_times = [1, 2, 3, 5] if gap_in_calendar_times else [1, 2, 3, 4]
    policy_num_by_time = {
        calendar_times[0]: 1,
        calendar_times[1]: 2,
        calendar_times[2]: 3,
        calendar_times[3]: 2 if policy_number_regresses else 3,
    }
    # Exactly one out-of-study cell: the last subject at the last decision time.
    inactive_cell = (calendar_times[3], _SUBJECT_IDS_ORCH[-1])

    action_prob_beta_by_time = {
        decision_time: _BETA_BY_POLICY_NUM_ORCH[policy_num]
        for decision_time, policy_num in policy_num_by_time.items()
    }
    # Every time policy 3 is in force, not just one of them, so that the only thing wrong with
    # the resulting study is the one thing each flag names.
    policy_3_times = [
        decision_time
        for decision_time, policy_num in policy_num_by_time.items()
        if policy_num == 3
    ]
    if stale_action_prob_beta:
        for decision_time in policy_3_times:
            action_prob_beta_by_time[decision_time] = _BETA_BY_POLICY_NUM_ORCH[2]
    if mismatched_shape_action_prob_beta:
        for decision_time in policy_3_times:
            action_prob_beta_by_time[decision_time] = jnp.array([0.5])
    if beta_disagrees_within_policy:
        action_prob_beta_by_time[calendar_times[3]] = jnp.array([0.15, -0.25])

    # Per CELL rather than per decision time, because two subjects at one decision time may
    # legitimately be on different policies (mixed_policies_at_one_time).
    policy_num_by_cell = {
        (decision_time, subject_id): policy_num_by_time[decision_time]
        for decision_time in calendar_times
        for subject_id in _SUBJECT_IDS_ORCH
    }
    action_prob_beta_by_cell = {
        cell: action_prob_beta_by_time[cell[0]] for cell in policy_num_by_cell
    }
    if mixed_policies_at_one_time:
        lagging_cell = (calendar_times[2], _SUBJECT_IDS_ORCH[-1])
        policy_num_by_cell[lagging_cell] = 2
        action_prob_beta_by_cell[lagging_cell] = _BETA_BY_POLICY_NUM_ORCH[2]

    rows = []
    action_prob_func_args = {decision_time: {} for decision_time in calendar_times}
    recorded_action_prob_by_cell = {}
    for subject_id in _SUBJECT_IDS_ORCH:
        for decision_time in calendar_times:
            cell = (decision_time, subject_id)
            is_active = cell != inactive_cell
            features = _features_orch(decision_time, subject_id)
            beta = action_prob_beta_by_cell[cell]
            policy_num = policy_num_by_cell[cell]
            if mismatched_shape_action_prob_beta and policy_num == 3:
                # One component, to match the one-component beta recorded at this time.
                features = jnp.array([0.75])
            if is_active:
                action_prob = float(_action_prob_func_orch(beta, features))
                recorded_action_prob_by_cell[cell] = action_prob
                rows.append(
                    {
                        "user_id": subject_id,
                        "calendar_t": decision_time,
                        "policy_num": float(policy_num),
                        "in_study": 1,
                        "action": float((decision_time + subject_id) % 2),
                        "action1prob": action_prob,
                        "reward": _reward_orch(decision_time, subject_id),
                        # Named after _inference_func_orch's second parameter, which is what
                        # makes this a study the inference function can actually be wired to.
                        "state": _state_orch(decision_time, subject_id),
                    }
                )
                args = (beta, features)
                if extra_action_prob_arg and cell == (
                    calendar_times[0],
                    _SUBJECT_IDS_ORCH[0],
                ):
                    args = (beta, features, features)
                action_prob_func_args[decision_time][subject_id] = args
            else:
                rows.append(
                    {
                        "user_id": subject_id,
                        "calendar_t": decision_time,
                        # NaN on out-of-study rows, exactly like the repo's own fixtures.
                        "policy_num": (
                            float(policy_num) if integer_policy_num_dtype else np.nan
                        ),
                        "in_study": 0,
                        "action": np.nan,
                        "action1prob": np.nan,
                        # NaN on an out-of-study row is legitimate and must stay legitimate:
                        # require_analysis_df_values_finite looks at ACTIVE rows only, exactly
                        # so that studies like this one are not rejected.
                        "reward": np.nan,
                        "state": _state_orch(decision_time, subject_id),
                    }
                )
                action_prob_func_args[decision_time][subject_id] = (
                    (beta, features) if nonblank_args_at_inactive_cell else ()
                )

    if blank_args_at_active_cell:
        action_prob_func_args[calendar_times[2]][_SUBJECT_IDS_ORCH[0]] = ()

    analysis_df = pd.DataFrame(rows)
    if integer_policy_num_dtype:
        analysis_df["policy_num"] = analysis_df["policy_num"].astype("int64")

    # Update 2 was computed from the first decision time's data; update 3 from the first two.
    times_by_update_policy_num = {2: calendar_times[:1], 3: calendar_times[:2]}
    # One row per update PRECEDING this policy -- which is exactly that policy's index in
    # helper_functions.construct_beta_index_by_policy_num_map, since the threading code slices
    # the shared post-update beta history by the RECORDED ROW COUNT, and
    # require_beta_dimensions_consistent asserts that shape exactly. Policy 2 is the first
    # update, so nothing precedes it and its block is EMPTY; policy 3's block holds policy 2's
    # beta alone. The INITIAL policy's beta is deliberately absent from both: it was produced
    # by no update, so it never enters all_post_update_betas.
    previous_betas_by_update_policy_num = {
        2: jnp.zeros((0, _BETA_DIM_ORCH)),
        3: jnp.stack([_BETA_BY_POLICY_NUM_ORCH[2]]),
    }
    alg_update_func_args = {}
    for update_policy_num in (2, 3):
        if drop_last_update and update_policy_num == 3:
            continue
        update_times = times_by_update_policy_num[update_policy_num]
        args_by_subject_id = {}
        for subject_id in _SUBJECT_IDS_ORCH:
            args = (
                _BETA_BY_POLICY_NUM_ORCH[update_policy_num],
                previous_betas_by_update_policy_num[update_policy_num],
                jnp.array(
                    [
                        recorded_action_prob_by_cell[(decision_time, subject_id)]
                        for decision_time in update_times
                    ]
                ),
                jnp.array([float(decision_time) for decision_time in update_times]),
                jnp.array(
                    [
                        _reward_orch(decision_time, subject_id)
                        for decision_time in update_times
                    ]
                ),
            )
            if (
                short_alg_update_arg
                and update_policy_num == 2
                and subject_id == _SUBJECT_IDS_ORCH[0]
            ):
                args = args[:-1]
            args_by_subject_id[subject_id] = args
        alg_update_func_args[update_policy_num] = args_by_subject_id

    return {
        "analysis_df": analysis_df,
        "active_col_name": "in_study",
        "action_col_name": "action",
        "policy_num_col_name": "policy_num",
        "calendar_t_col_name": "calendar_t",
        "subject_id_col_name": "user_id",
        "action_prob_col_name": "action1prob",
        "reward_col_name": "reward",
        "action_prob_func": _action_prob_func_orch,
        "action_prob_func_args": action_prob_func_args,
        "action_prob_func_args_beta_index": 0,
        "alg_update_func_args": alg_update_func_args,
        "alg_update_func_args_beta_index": 0,
        "alg_update_func_args_action_prob_index": 2,
        "alg_update_func_args_action_prob_times_index": 3,
        "alg_update_func_args_previous_betas_index": 1,
        "theta_est": jnp.array([0.25, -0.5, 1.5]),
        "beta_dim": _BETA_DIM_ORCH,
        "suppress_interactive_data_checks": True,
        "alg_update_func": (
            _masked_alg_update_func_orch if mask_aware else _alg_update_func_orch
        ),
        "alg_update_func_type": FunctionTypes.LOSS,
        "inference_func": _inference_func_orch,
        "inference_func_type": FunctionTypes.ESTIMATING,
        "inference_func_args_theta_index": 0,
        "alg_update_func_args_mask_index": 5 if mask_aware else -1,
        "alg_update_func_args_ragged_indices": (2, 3, 4) if mask_aware else (),
    }


def _narrow_all_alg_update_tuples_orch(study, width):
    """
    Narrow EVERY non-blank alg-update tuple to its first `width` values, in place, and return
    the study.

    Deliberately uniform: the resulting widths are mutually CONSISTENT, so what a study built
    this way exercises is the callability half of
    require_arg_tuple_lengths_consistent_and_callable rather than the consistency half. Blank
    tuples stay blank -- they mark out-of-study cells and are skipped by every width check.
    """
    study["alg_update_func_args"] = {
        policy_num: {
            subject_id: (args[:width] if args else args)
            for subject_id, args in args_by_subject_id.items()
        }
        for policy_num, args_by_subject_id in study["alg_update_func_args"].items()
    }
    return study


def _blank_all_alg_update_tuples_for_policy_orch(study, policy_num):
    """
    Replace every subject's tuple for one update policy with the empty tuple, keeping the
    policy as a KEY of alg_update_func_args, and return the study.

    That combination is what
    require_every_update_policy_has_at_least_one_nonblank_arg_tuple exists for: the key-set
    checks are satisfied, yet the policy contributes no beta at all.
    """
    study["alg_update_func_args"][policy_num] = {
        subject_id: () for subject_id in study["alg_update_func_args"][policy_num]
    }
    return study


def _deactivate_cells_orch(study, cells):
    """
    Mark the given (decision_time, subject_id) cells out of study and blank their action-prob
    argument tuples, in place, and return the study.

    Both halves are required to change nothing but the participation pattern:
    require_out_of_study_decision_times_are_exactly_blank_action_prob_args_times runs long
    before the participation check and demands that the blank tuples line up exactly with the
    out-of-study cells, so flipping the indicator alone would fire THAT check instead.

    The deactivated rows KEEP their recorded values rather than being NaN'd out. Those values
    are exempt from require_analysis_df_values_finite (active rows only), and the update args
    still carry copies of the recorded action probabilities from the first two decision times,
    which require_action_prob_args_in_alg_update_func_correspond_to_analysis_df looks up over
    the whole frame regardless of participation.
    """
    analysis_df = study["analysis_df"]
    for decision_time, subject_id in cells:
        row_mask = (analysis_df["calendar_t"] == decision_time) & (
            analysis_df["user_id"] == subject_id
        )
        assert row_mask.sum() == 1
        analysis_df.loc[row_mask, "in_study"] = 0
        study["action_prob_func_args"][decision_time][subject_id] = ()
    return study


def test_perform_first_wave_input_checks_passes_on_valid_study_orch():
    """
    The golden path: a self-consistent study with a NON-ZERO initial policy number passes
    every first-wave check. Every other test in this module is this study with one thing
    broken, so a failure here invalidates all of them.
    """
    study = _build_study_orch()

    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_passes_with_integer_policy_num_column_orch():
    """
    The same study with policy_num stored as int64 rather than the float64 the repo's fixtures
    use. The policy-number checks compare dict keys (python ints) against column values, so
    they must not depend on the column's dtype.
    """
    study = _build_study_orch(integer_policy_num_dtype=True)

    assert study["analysis_df"]["policy_num"].dtype == "int64"
    _assert_first_wave_passes_orch(study)


def test_perform_conditional_zeroth_wave_dataframe_checks_pass_on_valid_study_orch():
    """
    The golden study satisfies the zeroth wave too.

    The anchor for the three tests below, which each break one thing the zeroth wave -- and,
    since the duplication was removed, ONLY the zeroth wave -- is responsible for. Nothing to
    assert but that it returns: this wave reports no measurements.
    """
    study = _build_study_orch()

    input_checks.perform_conditional_zeroth_wave_dataframe_checks(
        **_zeroth_wave_kwargs_orch(study)
    )


def test_perform_conditional_zeroth_wave_dataframe_checks_on_empty_study_report_it_clearly_orch():
    """
    An empty analysis_df.

    RETARGETED THREE TIMES. It first asserted a ZeroDivisionError, from
    verify_analysis_df_summary_satisfactory dividing the active row count by a subject count of
    zero; then a bare IndexError, once the orchestrator began deriving beta_index_by_policy_num
    ahead of the summary. Both were unnamed downstream failures. require_analysis_df_nonempty
    then ran at the top of the first wave and said what was actually wrong, which is the whole
    point of the check -- so this asserts the MESSAGE, not merely that something raised. It now
    lives in the zeroth wave, ahead of theta_calculation_func, and is not repeated in the
    first wave, so the call under test moved with it.
    """
    study = _build_study_orch()
    study["analysis_df"] = study["analysis_df"].iloc[0:0]

    with pytest.raises(AssertionError, match="analysis DataFrame is empty"):
        input_checks.perform_conditional_zeroth_wave_dataframe_checks(
            **_zeroth_wave_kwargs_orch(study)
        )


def test_perform_first_wave_input_checks_interactive_confirmation_accepted_orch(
    monkeypatch,
):
    """
    With suppress_interactive_data_checks=False the orchestrator reaches
    verify_analysis_df_summary_satisfactory's interactive prompt. builtins.input MUST be
    patched here -- an unpatched call blocks the test run forever. The prompt has to describe
    the study that was actually supplied (2 subjects, 4 decision times).
    """
    study = _build_study_orch()
    study["suppress_interactive_data_checks"] = False
    prompts = []

    def _answer_yes(prompt):
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", _answer_yes)

    _assert_first_wave_passes_orch(study)
    assert len(prompts) == 1
    assert "2 subjects" in prompts[0]
    assert "4 decision times" in prompts[0]
    assert f"RL parameters of dimension {_BETA_DIM_ORCH}" in prompts[0]


def test_perform_first_wave_input_checks_interactive_rejection_exits_orch(monkeypatch):
    """Answering "n" to the summary prompt aborts the whole run with SystemExit."""
    study = _build_study_orch()
    study["suppress_interactive_data_checks"] = False
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    with pytest.raises(SystemExit):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_unrecognized_alg_update_func_type_orch():
    """A typo'd alg_update_func_type is caught up front, not as a bare
    "Unknown update function type." from inside the derivative precompute."""
    study = _build_study_orch()
    study["alg_update_func_type"] = "Loss"

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func_type='Loss' is not a recognized function type"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_unrecognized_inference_func_type_orch():
    """The same, for the inference function type, which is validated by the same check."""
    study = _build_study_orch()
    study["inference_func_type"] = "estimating_function"

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "inference_func_type='estimating_function' is not a recognized function type"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_inconsistent_action_prob_arg_tuple_lengths_orch():
    """
    ONE action-prob tuple carries three values while every other carries two: the CONSISTENCY
    half of require_arg_tuple_lengths_consistent_and_callable. Unchecked, this overruns the
    batching code's per-position lists with a bare IndexError, because
    vmap_helpers.build_batched_arg_lists_by_subject takes its argument count from the first
    subject in each bucket and indexes every other subject's tuple by that range.

    The message must name both lengths and one example (decision_time, subject_id) per length,
    since "some tuple somewhere is the wrong width" is unactionable on a real study.
    """
    study = _build_study_orch(extra_action_prob_arg=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Supplied action_prob_func argument tuples do not all have the same length; "
            "lengths [2, 3] are all present. One example (decision_time, subject_id) per "
            "length: {2: (1, 2), 3: (1, 1)}."
        ),
    ) as excinfo:
        input_checks.perform_first_wave_input_checks(**study)

    # The three-value tuple was planted at the first decision time for the first subject, so
    # that is the example the message must offer for the odd length.
    assert "3: (1, 1)" in str(excinfo.value)


def test_perform_first_wave_input_checks_rejects_uncallable_action_prob_arg_tuple_length_orch():
    """
    EVERY action-prob tuple carries three values, so the widths are mutually consistent and it
    is the CALLABILITY half that must object: _action_prob_func_orch declares exactly two
    positional parameters, so the reconstruction check would later call it with three.

    This is the other assertion in the same check, with a different message, so it needs its
    own test.
    """
    study = _build_study_orch()
    study["action_prob_func_args"] = {
        decision_time: {
            # Repeat the last value rather than inventing one, so nothing but the WIDTH
            # differs from the golden study.
            subject_id: (args + (args[-1],)) if args else args
            for subject_id, args in args_by_subject_id.items()
        }
        for decision_time, args_by_subject_id in study["action_prob_func_args"].items()
    }

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "action_prob_func will be called with 3 positional argument(s), but its signature "
            "accepts exactly 2."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_inconsistent_alg_update_arg_tuple_lengths_orch():
    """
    ONE alg-update tuple is missing its last value while the other three carry five: the
    CONSISTENCY half again, on the update-function side, where the key in the message is a
    policy number rather than a decision time.
    """
    study = _build_study_orch(short_alg_update_arg=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Supplied alg_update_func argument tuples do not all have the same length; "
            "lengths [4, 5] are all present. One example (policy_num, subject_id) per length: "
            "{4: (2, 1), 5: (2, 2)}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_widths_differing_by_policy_orch():
    """
    Every tuple for update policy 2 is four values wide and every tuple for policy 3 is five:
    consistent WITHIN each policy, inconsistent across them. The check is stated over all
    supplied tuples at once, not per policy, because a single argument-position layout has to
    serve every update.
    """
    study = _build_study_orch()
    study["alg_update_func_args"][2] = {
        subject_id: args[:4]
        for subject_id, args in study["alg_update_func_args"][2].items()
    }

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Supplied alg_update_func argument tuples do not all have the same length; "
            "lengths [4, 5] are all present. One example (policy_num, subject_id) per length: "
            "{4: (2, 1), 5: (3, 1)}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_uncallable_alg_update_arg_tuple_length_orch():
    """
    EVERY alg-update tuple is narrowed to four values, so the widths agree and only the
    CALLABILITY half is left to object: _alg_update_func_orch requires all five of its
    positional parameters, so four can never satisfy it no matter how the batching code counts.
    """
    study = _narrow_all_alg_update_tuples_orch(_build_study_orch(), 4)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 4 positional argument(s), but its signature "
            "accepts exactly 5."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_out_of_range_action_prob_index_orch():
    """
    An alg_update_func_args_action_prob_index that addresses no position in the supplied
    five-value tuples (valid positions are 0..4).
    """
    study = _build_study_orch()
    study["alg_update_func_args_action_prob_index"] = 5

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func argument indices do not address a position in argument "
            "tuples of length 5 (valid positions are 0 through 4)"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_colliding_alg_update_indices_orch():
    """
    beta and the action probabilities pointed at the same position: the threading code would
    write the reconstructed action probabilities over the beta position, so the beta being
    differentiated never reaches the update function.
    """
    study = _build_study_orch()
    study["alg_update_func_args_beta_index"] = study[
        "alg_update_func_args_action_prob_index"
    ]

    with pytest.raises(
        AssertionError,
        match=re.escape("These alg_update_func argument indices collide"),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_out_of_range_theta_index_orch():
    """
    An inference_func_args_theta_index past the inference function's declared parameter count
    (three parameters, so 3 addresses nothing).
    """
    study = _build_study_orch()
    study["inference_func_args_theta_index"] = 3

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These inference_func argument indices do not address a position in argument "
            "tuples of length 3 (valid positions are 0 through 2)"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_negative_theta_index_orch():
    """
    A negative theta index means "absent", and process_inference_func_args would then never
    substitute theta at all -- inference differentiated with respect to a theta that was never
    inserted, with NO error anywhere downstream. This is the silent failure the check exists
    for, so the orchestrator must reject it.
    """
    study = _build_study_orch()
    study["inference_func_args_theta_index"] = -1

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These inference_func argument indices are required but were not supplied"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_negative_alg_update_beta_index_orch():
    """The same "absent" rejection for the algorithm update function's beta index."""
    study = _build_study_orch()
    study["alg_update_func_args_beta_index"] = -1

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func argument indices are required but were not supplied"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_stale_recorded_action_prob_beta_orch():
    """
    The recorded action-prob beta for every decision time where policy 3 is in force is policy
    2's beta -- the classic off-by-one in policy numbering. The recorded probabilities were
    derived from that same stale beta, so reconstruction still passes; only the
    recorded-vs-update beta comparison can catch it.

    All THREE cells in force under policy 3 are reported (both subjects at the third decision
    time and the one remaining subject at the fourth). It used to be two: the flag staled a
    single decision time's beta, which the new
    require_betas_match_in_action_prob_func_args_each_policy would now reject first for
    disagreeing with the other policy-3 cell, so the defect is planted consistently across the
    policy instead.
    """
    study = _build_study_orch(stale_action_prob_beta=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The beta recorded in action_prob_func_args disagrees with the beta recorded in "
            "alg_update_func_args for the policy in force, at 3 (decision_time, subject_id) "
            "cell(s)."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_wrong_shaped_recorded_beta_orch():
    """
    The recorded action-prob beta has ONE component where beta_dim (and update 3's beta) has
    two.

    This test used to assert the "shape (1,) vs (2,)" wording of
    require_recorded_action_prob_betas_match_update_betas_for_their_policy, which compares
    shapes before values so a (1,) beta cannot broadcast against a (2,) one and be called
    equal. That check no longer gets the chance: require_beta_dimensions_consistent is new and
    runs much earlier, and it rejects ANY recorded beta not shaped (beta_dim,) outright -- so
    no shape disagreement between a recorded beta and an update beta can reach the later check
    through the orchestrator at all. The assertion is retargeted to the message that actually
    fires; the later check's own shape branch is covered directly, with the function.
    """
    study = _build_study_orch(mismatched_shape_action_prob_beta=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "3 action_prob_func_args beta(s) are not shaped (2,), the dimension taken from the "
            "first supplied beta. Offending (decision_time, subject_id) -> shape, up to 5 "
            "shown: {(3, 1): (1,), (3, 2): (1,), (4, 1): (1,)}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_blank_args_at_active_cell_orch():
    """
    An empty action-prob tuple at a cell analysis_df marks in study: nothing can reconstruct
    that row's recorded probability, and this must be a clear ValueError rather than a silently
    narrowed check.
    """
    study = _build_study_orch(blank_args_at_active_cell=True)

    with pytest.raises(ValueError, match="could not reconstruct a prediction for 1"):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_nonblank_args_at_inactive_cell_orch():
    """
    The mirror image: a real (non-empty) tuple at the one out-of-study cell. Blank tuples must
    line up exactly with the out-of-study cells.
    """
    study = _build_study_orch(nonblank_args_at_inactive_cell=True)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "found non-blank action_prob_func_args for 1 (decision_time, subject_id) pair(s) "
            "the analysis DataFrame does not mark active"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_corrupted_recorded_action_probability_orch():
    """
    A recorded action probability that action_prob_func cannot reproduce from its args. The
    corrupted cell is at the LAST decision time on purpose: the two updates' args carry copies
    of the recorded probabilities from the first two times, so corrupting one of those trips
    require_action_prob_args_in_alg_update_func_correspond_to_analysis_df (which runs earlier)
    instead of the reconstruction check this test is about.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    analysis_df.loc[
        (analysis_df["calendar_t"] == 4) & (analysis_df["user_id"] == 1), "action1prob"
    ] = 0.123456

    with pytest.raises(AssertionError, match="Not equal to tolerance"):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_missing_non_initial_policy_update_orch():
    """
    alg_update_func_args omits policy 3, which active rows do use. The initial policy number
    here is 1, not 0, so this pins the fixed version of the check: it derives the initial
    policy number as the minimum non-negative active policy number instead of hardcoding 0
    (and it must actually compare, rather than comparing the policy column against itself).
    """
    study = _build_study_orch(drop_last_update=True)

    with pytest.raises(
        AssertionError,
        match=re.escape("(the initial policy number is 1.0)"),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_all_blank_args_for_one_update_policy_orch():
    """
    Update policy 2 is still a KEY of alg_update_func_args but every subject's tuple under it
    is empty, so no beta is recorded for it anywhere.

    The two key-set checks either side of this one are both satisfied by that dict:
    require_alg_update_args_given_for_all_subjects_at_each_update compares only subject-id key
    sets, and require_all_policy_numbers_..._present_in_alg_update_args only asks whether the
    policy is a key. What breaks is downstream and silent --
    helper_functions.collect_all_post_update_betas appends one beta per policy that HAS a
    non-blank tuple, while construct_beta_index_by_policy_num_map indexes policies by their
    position in the same sorted order, so policy 2 would resolve to policy 3's beta and policy
    3 to whatever follows it. This test proves the orchestrator reaches the check that stops
    that.
    """
    study = _blank_all_alg_update_tuples_for_policy_orch(_build_study_orch(), 2)

    assert 2 in study["alg_update_func_args"]
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These update policies are present in alg_update_func_args but have a blank "
            "(empty-tuple) argument tuple for every subject: [2.0]."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_all_blank_args_for_last_update_policy_orch():
    """
    The same defect on the LAST update policy, which shifts no other policy's beta down and so
    is the case a positional argument-counting implementation would most easily miss. It is
    still wrong -- policy 3's beta comes from nowhere -- so it must still be rejected.
    """
    study = _blank_all_alg_update_tuples_for_policy_orch(_build_study_orch(), 3)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These update policies are present in alg_update_func_args but have a blank "
            "(empty-tuple) argument tuple for every subject: [3.0]."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_wholly_blank_alg_update_args_orch():
    """
    EVERY tuple of EVERY update policy is blank. Nothing non-blank is supplied at all, so
    require_arg_tuple_lengths_consistent_and_callable has no common width to report and returns
    None rather than an int.

    This is the case that pins the orchestrator's None guards: the index-in-range, mask-append
    and ragged-position checks all take that width as an argument, and running them on None
    would raise a bare TypeError from a comparison against None instead of the message that
    actually names the problem. Reaching the all-blank-policy check at all proves the guards
    held.
    """
    study = _build_study_orch()
    for policy_num in tuple(study["alg_update_func_args"]):
        _blank_all_alg_update_tuples_for_policy_orch(study, policy_num)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These update policies are present in alg_update_func_args but have a blank "
            "(empty-tuple) argument tuple for every subject: [2.0, 3.0]."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_policy_num_absent_from_analysis_df_orch():
    """An update keyed by a policy number the analysis DataFrame has never heard of."""
    study = _build_study_orch()
    study["alg_update_func_args"][9] = study["alg_update_func_args"][3]

    with pytest.raises(
        AssertionError,
        match="policy numbers present in algorithm update function args but not in the analysis DataFrame",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_calendar_time_gap_orch():
    """
    Calendar times 1, 2, 3, 5. Every subject still has every time and every action-prob arg
    key still lines up, so only the consecutive-integer check can catch the gap. It reaches
    that check through the orchestrator with the float-free int64 column the frame really has.
    """
    study = _build_study_orch(gap_in_calendar_times=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Calendar times are not consecutive integers. calendar_t values present: "
            "[1, 2, 3, 5]"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_non_binary_in_study_indicator_orch():
    """
    An in_study value of 2 does not get through the orchestrator.

    It is NOT require_binary_active_indicators that catches it here, even though that is the
    check written for it: a row whose indicator is neither 0 nor 1 is counted as neither
    active (in_study == 1) nor out of study (in_study == 0), so the args-consistency checks
    that run much earlier fire first -- this cell now has non-blank action-prob args while the
    analysis DataFrame no longer marks it active. Pinning the message that actually fires is
    the point: the orchestrator's ORDER is what this module tests, and that check's own
    direct coverage belongs with the function.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    analysis_df.loc[
        (analysis_df["calendar_t"] == 1) & (analysis_df["user_id"] == 1), "in_study"
    ] = 2

    with pytest.raises(
        ValueError,
        match=re.escape(
            "found non-blank action_prob_func_args for 1 (decision_time, subject_id) pair(s) "
            "the analysis DataFrame does not mark active"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_fractional_action_orch():
    """
    A fractional action of 0.5 on an active row. The pre-fix check cast to int64 first, which
    truncated 0.5 to 0 and passed; the action column is float64 here (it holds NaN out of
    study), which is exactly the dtype that made the truncation possible.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    assert analysis_df["action"].dtype == "float64"
    analysis_df.loc[
        (analysis_df["calendar_t"] == 1) & (analysis_df["user_id"] == 1), "action"
    ] = 0.5

    with pytest.raises(AssertionError, match="Actions are not binary"):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_two_dimensional_theta_orch():
    """theta_est must be 1D; the estimate is a column-shaped array here."""
    study = _build_study_orch()
    study["theta_est"] = jnp.array([[0.25], [-0.5], [1.5]])

    with pytest.raises(AssertionError, match="Theta is not a 1D array"):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_passes_masked_configuration_orch():
    """
    The mask-padding configuration: an alg_update_func declaring one more parameter than the
    supplied tuples carry, alg_update_func_args_mask_index equal to the supplied tuple length
    (the mask is APPENDED, never inserted), and the genuinely ragged positions -- the action
    probabilities, their times and the rewards -- listed as self-padding.
    """
    study = _build_study_orch(mask_aware=True)

    assert study["alg_update_func_args_mask_index"] == 5
    assert study["alg_update_func_args_ragged_indices"] == (2, 3, 4)
    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_rejects_shared_beta_as_ragged_index_orch():
    """
    beta listed as a self-padding position. Padding repeats a position's last row, so it would
    append copies of beta's last component and change its dimension rather than merely adding
    padding rows.
    """
    study = _build_study_orch(mask_aware=True)
    study["alg_update_func_args_ragged_indices"] = (
        study["alg_update_func_args_beta_index"],
        2,
        3,
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These alg_update_func parameters are shared across subjects but were listed as "
            "ragged (self-padding) positions"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_empty_ragged_indices_with_mask_orch():
    """Mask padding requested with nothing to pad -- a blank ragged-index tuple."""
    study = _build_study_orch(mask_aware=True)
    study["alg_update_func_args_ragged_indices"] = ()

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "requests mask padding (mask index 5) but supplied no ragged argument positions"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_inserted_rather_than_appended_mask_orch():
    """
    A mask index that does not equal the supplied tuple length. The mask is appended as a new
    last argument, so 3 would mean overwriting the action-prob times instead.
    """
    study = _build_study_orch(mask_aware=True)
    study["alg_update_func_args_mask_index"] = 3

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func mask index 3 must equal the supplied argument tuple length (5)"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_mask_index_without_mask_aware_func_orch():
    """
    Requesting mask padding while passing the five-parameter (mask-UNAWARE) update function.
    The five supplied values are a perfectly legal width on their own; what makes this
    unanalyzable is that self_pad_ragged_args_and_build_mask will APPEND the validity mask, so
    the function is actually called with six -- one more than its signature can take.

    So the message must speak about the CALL length rather than the supplied length, and say
    where the extra argument comes from; otherwise "5 supplied, 5 declared, rejected" reads as
    a bug in the check.
    """
    study = _build_study_orch()
    study["alg_update_func_args_mask_index"] = 5
    study["alg_update_func_args_ragged_indices"] = (2, 3, 4)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 6 positional argument(s) (the 5 supplied, "
            "plus the validity mask that will be appended at mask index 5), but its signature "
            "accepts exactly 5."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_accepts_variadic_alg_update_func_orch():
    """
    A *args update function is ACCEPTED. This test is the inverse of the one it replaces, and
    it is here to keep the old false alarm from coming back.

    The old check rejected any function declaring *args outright, on the theory that the
    package could not honor its arity. It can: the production batching code takes the argument
    count from the DATA, not from the signature (vmap_helpers.build_batched_arg_lists_by_subject
    says so in its own docstring, because introspection is wrong for a wrapped function). A
    *args function accepts the five supplied values, so there is nothing to object to, and an
    always-on check that rejects data the estimator handles correctly is worse than no check.
    """
    study = _build_study_orch()

    def _variadic_alg_update_func(beta, *args):
        return jnp.sum(beta)

    study["alg_update_func"] = _variadic_alg_update_func

    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_accepts_variadic_forwarding_shim_orch():
    """
    The wrapper shape a caller actually writes: a `*rest` shim that forwards straight to the
    real update function. Verified on this repo's own fixture to produce estimates
    BIT-IDENTICAL to the unwrapped function, so the first wave must let it through.
    """
    study = _build_study_orch()

    def _forwarding_shim(beta_est, *rest):
        return _alg_update_func_orch(beta_est, *rest)

    study["alg_update_func"] = _forwarding_shim

    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_accepts_variadic_forwarding_shim_with_mask_orch():
    """
    The same forwarding shim in the mask-padding configuration, where the check compares
    against six (the five supplied plus the appended mask) rather than five. *args swallows the
    mask exactly as it swallows the rest, so this must pass too -- the appended mask is not a
    reason to demand a fixed arity.
    """
    study = _build_study_orch(mask_aware=True)

    def _forwarding_shim(beta_est, *rest):
        return _masked_alg_update_func_orch(beta_est, *rest)

    study["alg_update_func"] = _forwarding_shim

    assert study["alg_update_func_args_mask_index"] == 5
    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_accepts_extra_defaulted_parameter_orch():
    """
    The other legitimate wrapper: one MORE declared parameter than the supplied tuples carry,
    defaulted so the extra never has to be supplied (a ridge hyperparameter here). Also
    verified bit-identical on this repo's fixture, and also rejected by the old equality check.
    """
    study = _build_study_orch()

    def _ridge_penalized_alg_update_func(
        beta, previous_betas, action_probs, action_prob_times, rewards, ridge=0.25
    ):
        return _alg_update_func_orch(
            beta, previous_betas, action_probs, action_prob_times, rewards
        ) + ridge * jnp.sum(beta**2)

    study["alg_update_func"] = _ridge_penalized_alg_update_func

    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_rejects_too_few_args_for_defaulted_signature_orch():
    """
    Accepting the extra-defaulted-parameter wrapper does not mean accepting anything: with the
    six-parameter (five required, one defaulted) wrapper, four supplied values still cannot
    fill the five required ones.

    Pins the "N to M" rendering of the accepted range, which only a signature with an optional
    parameter can produce.
    """
    study = _build_study_orch()

    def _ridge_penalized_alg_update_func(
        beta, previous_betas, action_probs, action_prob_times, rewards, ridge=0.25
    ):
        return _alg_update_func_orch(
            beta, previous_betas, action_probs, action_prob_times, rewards
        ) + ridge * jnp.sum(beta**2)

    study["alg_update_func"] = _ridge_penalized_alg_update_func
    _narrow_all_alg_update_tuples_orch(study, 4)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 4 positional argument(s), but its signature "
            "accepts 5 to 6."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_too_few_args_for_variadic_signature_orch():
    """
    Nor does accepting *args mean accepting any width: a shim declaring two REQUIRED parameters
    before its *args cannot be called with one argument, however variadic its tail.

    Pins the "at least N" rendering of the accepted range.
    """
    study = _build_study_orch()

    def _two_required_then_variadic(beta_est, previous_betas, *rest):
        return _alg_update_func_orch(beta_est, previous_betas, *rest)

    study["alg_update_func"] = _two_required_then_variadic
    _narrow_all_alg_update_tuples_orch(study, 1)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "alg_update_func will be called with 1 positional argument(s), but its signature "
            "accepts at least 2."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_mismatched_action_prob_in_update_args_orch():
    """
    An action probability recorded in the update args that disagrees with the analysis
    DataFrame's value at the time it claims to come from.
    """
    study = _build_study_orch()
    args_by_subject_id = study["alg_update_func_args"][3]
    original_args = args_by_subject_id[_SUBJECT_IDS_ORCH[0]]
    args_by_subject_id[_SUBJECT_IDS_ORCH[0]] = (
        original_args[:2] + (jnp.array([0.11, 0.22]),) + original_args[3:]
    )

    with pytest.raises(
        AssertionError,
        match="mismatch for subject 1 between the action probabilities supplied",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_non_strictly_increasing_action_prob_times_orch():
    """
    Update 3's action-prob times, and the probabilities they index, both reversed. The
    correspondence check still passes (the pairs still agree with the analysis DataFrame), so
    the only thing left to reject the reversal is the strictly-increasing assertion in
    require_valid_action_prob_times_given_if_index_supplied.
    """
    study = _build_study_orch()
    args_by_subject_id = study["alg_update_func_args"][3]
    for subject_id in _SUBJECT_IDS_ORCH:
        original_args = args_by_subject_id[subject_id]
        args_by_subject_id[subject_id] = (
            original_args[:2]
            + (jnp.flip(original_args[2]), jnp.flip(original_args[3]))
            + original_args[4:]
        )

    with pytest.raises(
        AssertionError,
        match="Non-strictly-increasing times were given for action probabilities",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_action_prob_time_outside_study_raises_key_error_orch():
    """
    An action-prob time in the update args that the study never had (9, against calendar times
    1..4).

    require_valid_action_prob_times_given_if_index_supplied has a purpose-built message for
    exactly this ("Times not present in the study were given..."), but the orchestrator runs
    require_action_prob_args_in_alg_update_func_correspond_to_analysis_df first, and that check
    looks the time up in a dict built from the analysis DataFrame -- so what a caller actually
    sees today is a bare KeyError naming the missing (time, subject) key and nothing else. This
    test pins that CURRENT behavior; it is asserting the raise, not endorsing the message.
    """
    study = _build_study_orch()
    args_by_subject_id = study["alg_update_func_args"][3]
    for subject_id in _SUBJECT_IDS_ORCH:
        original_args = args_by_subject_id[subject_id]
        args_by_subject_id[subject_id] = (
            original_args[:3] + (jnp.array([1.0, 9.0]),) + original_args[4:]
        )

    with pytest.raises(KeyError, match=re.escape("(9.0, 1)")):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_betas_disagreeing_across_subjects_orch():
    """
    Two subjects reporting different betas for the SAME update. This is what makes "the update
    beta for this policy" well defined, and it runs before the recorded-vs-update comparison.
    """
    study = _build_study_orch()
    args_by_subject_id = study["alg_update_func_args"][3]
    original_args = args_by_subject_id[_SUBJECT_IDS_ORCH[-1]]
    args_by_subject_id[_SUBJECT_IDS_ORCH[-1]] = (
        jnp.array([0.0, 0.0]),
    ) + original_args[1:]

    with pytest.raises(
        AssertionError,
        match="Betas do not match across subjects in the algorithm update function args for policy number 3",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_missing_subject_in_update_args_orch():
    """
    An update whose args omit a subject entirely. Blank tuples, not missing keys, are how a
    subject with nothing to contribute is expressed.
    """
    study = _build_study_orch()
    del study["alg_update_func_args"][3][_SUBJECT_IDS_ORCH[-1]]

    with pytest.raises(
        AssertionError,
        match="Not all subjects present in algorithm update function args for policy number 3",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_two_dimensional_beta_in_update_args_orch():
    """beta must be 1D in the update args; a (1, 2) column beta is the common mistake."""
    study = _build_study_orch()
    for subject_id in _SUBJECT_IDS_ORCH:
        original_args = study["alg_update_func_args"][3][subject_id]
        study["alg_update_func_args"][3][subject_id] = (
            jnp.atleast_2d(original_args[0]),
        ) + original_args[1:]

    with pytest.raises(
        AssertionError,
        match="Beta is not a 1D array in the algorithm update function args",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_one_dimensional_previous_betas_orch():
    """
    Previous betas must be 2D (one row per previous update).

    Flattened on update policy 3 rather than policy 2, which it used to use: policy 2's block
    is now legitimately EMPTY, shaped (0, beta_dim), and flattening that yields the degenerate
    shape (0,). Policy 3's one-row block flattens to a genuine (beta_dim,) vector -- the shape
    a caller who forgot the block is a stack of rows would actually pass.
    """
    study = _build_study_orch()
    for subject_id in _SUBJECT_IDS_ORCH:
        original_args = study["alg_update_func_args"][3][subject_id]
        study["alg_update_func_args"][3][subject_id] = (
            original_args[:1] + (jnp.ravel(original_args[1]),) + original_args[2:]
        )

    with pytest.raises(
        AssertionError,
        match="Previous betas is not a 2D array in the algorithm update function args",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_missing_decision_time_in_action_prob_args_orch():
    """
    action_prob_func_args must cover every decision time in the analysis DataFrame -- an
    omitted time would silently drop that time from the reconstruction check's coverage.
    """
    study = _build_study_orch()
    del study["action_prob_func_args"][3]

    with pytest.raises(
        AssertionError,
        match="Not all decision times present in action prob function args",
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_action_prob_times_index_without_probs_orch():
    """
    Action-prob times supplied with no action-prob index (a negative index means "absent"), so
    the times address values the threading code would never substitute.

    No match= here, deliberately: require_action_prob_index_given_if_times_supplied is a bare
    assert with NO message, so the AssertionError this raises carries no text to match on.
    """
    study = _build_study_orch()
    study["alg_update_func_args_action_prob_index"] = -1

    with pytest.raises(AssertionError) as exception_info:
        input_checks.perform_first_wave_input_checks(**study)
    assert str(exception_info.value) == ""


### Coverage for the checks the orchestrator gained on 2026-09-02. Each test below perturbs
### exactly ONE thing away from the golden study, and each exists to prove the orchestrator
### actually REACHES the check named in its docstring, with the arguments it threads into it --
### the checks' own edge cases belong with the functions.


def test_perform_conditional_zeroth_wave_dataframe_checks_reject_missing_reward_column_orch():
    """
    The reward column is not in the analysis DataFrame.

    reward_col_name is a parameter of require_all_named_columns_present_in_analysis_df, and the
    presence check runs in the ZEROTH wave, ahead of everything. Before that, a wrong reward
    column name reached verify_analysis_df_summary_satisfactory, which reads that column to
    build its average-reward plot, and surfaced as a bare KeyError from inside a plotting
    routine.
    """
    study = _build_study_orch()
    study["analysis_df"] = study["analysis_df"].drop(columns=["reward"])

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These named columns are not in the analysis DataFrame: ['reward']. Columns "
            "present: ['action', 'action1prob', 'calendar_t', 'in_study', 'policy_num', "
            "'state', 'user_id']."
        ),
    ):
        input_checks.perform_conditional_zeroth_wave_dataframe_checks(
            **_zeroth_wave_kwargs_orch(study)
        )


def test_perform_conditional_zeroth_wave_dataframe_checks_report_every_missing_column_at_once_orch():
    """
    Two column names are wrong, and BOTH are named in one message.

    This is the behavior change in require_all_named_columns_present_in_analysis_df: it used to
    assert once per column ("{col_name} not in analysis DataFrame."), so a caller who had
    mis-wired several names -- the usual case -- learned about them one rerun at a time.
    """
    study = _build_study_orch()
    study["action_prob_col_name"] = "action_prob"
    study["reward_col_name"] = "rewrd"

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These named columns are not in the analysis DataFrame: "
            "['action_prob', 'rewrd']."
        ),
    ):
        input_checks.perform_conditional_zeroth_wave_dataframe_checks(
            **_zeroth_wave_kwargs_orch(study)
        )


def test_perform_first_wave_input_checks_rejects_object_dtype_reward_column_orch():
    """
    An object-dtype reward column.

    reward_col_name is also a new parameter of
    require_all_named_columns_not_object_type_in_analysis_df, so the reward column is now
    INSPECTED rather than merely accepted: it is consumed numerically (averaged per decision
    time for the summary plot, and fed to the estimator), so object dtype genuinely breaks it.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    analysis_df["reward"] = analysis_df["reward"].astype(object)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These analysis DataFrame columns are of object type, but are consumed "
            "numerically: ['reward']"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_inference_func_parameter_that_is_no_column_orch():
    """
    An inference function declaring a parameter no analysis DataFrame column matches.

    process_inference_func_args builds this function's argument tuples by filling each declared
    parameter from the column whose NAME equals it, so a parameter naming no column cannot be
    supplied at all -- and without this check it surfaces as a bare KeyError from
    helper_functions.get_active_df_column, with nothing to say a parameter name was the problem.
    """
    study = _build_study_orch()

    def _inference_func_with_unknown_parameter(theta, state, engagement):
        return (engagement - jnp.dot(theta[:2], jnp.array([1.0, state]))) * state

    study["inference_func"] = _inference_func_with_unknown_parameter

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These inference_func parameters are not analysis DataFrame columns (position -> "
            "parameter name): {2: 'engagement'}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_exempts_theta_parameter_name_from_column_rule_orch():
    """
    The theta parameter's NAME is unconstrained, and the exemption follows the theta INDEX
    rather than assuming position 0: this inference function takes theta second, under a name
    ("rl_theta") that is no column, and its other two parameters name real columns.

    theta is supplied from theta_est, not from a column, so requiring its name to be a column
    would reject perfectly valid wiring.
    """
    study = _build_study_orch()

    def _inference_func_with_theta_second(state, rl_theta, reward):
        return (reward - jnp.dot(rl_theta[:2], jnp.array([1.0, state]))) * state

    study["inference_func"] = _inference_func_with_theta_second
    study["inference_func_args_theta_index"] = 1

    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_rejects_duplicate_subject_time_rows_orch():
    """
    Two rows share a (user_id, calendar_t) key, so every per-subject-per-time dictionary in the
    package would silently keep whichever row was built last.

    The duplicate is planted on the OUT-OF-STUDY row deliberately. The only pre-existing
    duplicate detection lives inside
    require_action_probabilities_in_analysis_df_can_be_reconstructed, looks at ACTIVE rows only,
    and runs earlier -- so an active duplicate would never reach the new whole-frame check. And
    require_all_subjects_have_all_times_in_analysis_df compares SETS of times, which duplicates
    sail through.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    inactive_row = analysis_df[
        (analysis_df["calendar_t"] == 4) & (analysis_df["user_id"] == 2)
    ]
    assert len(inactive_row) == 1
    assert int(inactive_row["in_study"].iloc[0]) == 0
    study["analysis_df"] = pd.concat([analysis_df, inactive_row], ignore_index=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The analysis DataFrame has 2 row(s) sharing a (subject_id, calendar_t) key, e.g. "
            "[{'user_id': 2, 'calendar_t': 4}]."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_subject_who_leaves_and_returns_orch():
    """
    The second subject is out of study at the SECOND decision time and back in at the third,
    leaving a gap in the middle of its participation.

    Staggered entry and early exit are legitimate -- the golden study has one subject leaving
    early -- but a gap changes what "this subject's history so far" means, and the padding path
    assumes a single active window per subject.
    """
    study = _deactivate_cells_orch(_build_study_orch(), [(2, _SUBJECT_IDS_ORCH[-1])])

    analysis_df = study["analysis_df"]
    active_times = sorted(
        analysis_df.loc[
            (analysis_df["user_id"] == _SUBJECT_IDS_ORCH[-1])
            & (analysis_df["in_study"] == 1),
            "calendar_t",
        ]
    )
    assert active_times == [1, 3]
    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) leave the study and return"),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_subject_never_active_orch():
    """
    The second subject is out of study at EVERY decision time. It contributes nothing but still
    counts in the denominator of every per-subject average, so it must be dropped from the
    analysis DataFrame rather than carried as a row of zeros.

    Asserted separately from the leaving-and-returning case because it is a separate assertion
    in require_contiguous_participation, and it is the one that runs first.
    """
    study = _build_study_orch()
    _deactivate_cells_orch(
        study,
        [(decision_time, _SUBJECT_IDS_ORCH[-1]) for decision_time in (1, 2, 3, 4)],
    )

    analysis_df = study["analysis_df"]
    assert not (
        (analysis_df["user_id"] == _SUBJECT_IDS_ORCH[-1])
        & (analysis_df["in_study"] == 1)
    ).any()
    with pytest.raises(
        AssertionError,
        match=re.escape("1 subject(s) are never active at any decision time"),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_decreasing_policy_numbers_over_time_orch():
    """
    The last decision time runs under policy 2 again, after policy 3 was in force at the time
    before it, so the first subject's non-fallback policy numbers go 1, 2, 3, 2.

    Every cell under policy 2 records policy 2's beta, so the per-policy beta invariant and the
    recorded-vs-update comparison are both satisfied and the SET of policy numbers is still the
    gapless {1, 2, 3} -- which is all
    require_consecutive_integer_policy_numbers ever checked. Only the temporal ordering is
    wrong, and only the new check looks at it.
    """
    study = _build_study_orch(policy_number_regresses=True)

    analysis_df = study["analysis_df"]
    policy_nums_over_time = analysis_df.loc[
        analysis_df["user_id"] == _SUBJECT_IDS_ORCH[0]
    ].sort_values("calendar_t")["policy_num"]
    assert policy_nums_over_time.tolist() == [1.0, 2.0, 3.0, 2.0]
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 subject(s) have non-fallback policy numbers that DECREASE as calendar time "
            "advances"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_nan_reward_on_active_row_orch():
    """
    A NaN reward on an ACTIVE row. It propagates silently through theta estimation into the
    bread, the meat and the reported variance, and until this check existed it surfaced only as
    a diagnostic finding after the whole computation had run.
    """
    study = _build_study_orch()
    analysis_df = study["analysis_df"]
    analysis_df.loc[
        (analysis_df["calendar_t"] == 3) & (analysis_df["user_id"] == 1), "reward"
    ] = np.nan

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "These analysis DataFrame columns contain non-finite values (NaN or inf) on ACTIVE "
            "rows -- column -> count: {'reward': 1}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_accepts_nan_values_on_inactive_rows_orch():
    """
    The other half of require_analysis_df_values_finite's scope: the golden study's one
    out-of-study row holds NaN in the action, action-probability, policy-number AND reward
    columns, and still passes.

    That is not an oversight but the whole scope decision -- those four columns legitimately
    hold NaN out of study in this repo's own fixtures, so a frame-wide finiteness check would
    reject every real study.
    """
    study = _build_study_orch()
    inactive_rows = study["analysis_df"][study["analysis_df"]["in_study"] == 0]

    assert len(inactive_rows) == 1
    assert (
        inactive_rows[["action", "action1prob", "policy_num", "reward"]]
        .isna()
        .all()
        .all()
    )
    _assert_first_wave_passes_orch(study)


def test_perform_first_wave_input_checks_rejects_nonfinite_action_prob_func_arg_orch():
    """
    A NaN inside one supplied action-prob feature vector. Nothing else would catch it as such:
    the reconstruction check would compare a NaN prediction against the recorded probability
    and report a tolerance failure that says nothing about where the NaN came from.
    """
    study = _build_study_orch()
    original_args = study["action_prob_func_args"][1][_SUBJECT_IDS_ORCH[0]]
    study["action_prob_func_args"][1][_SUBJECT_IDS_ORCH[0]] = (
        original_args[0],
        jnp.array([1.0, jnp.nan]),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 supplied action_prob_func argument position(s) contain non-finite values (NaN "
            "or inf). Offending (decision_time, subject_id, arg position) -> count, up to 5 "
            "shown: {(1, 1, 1): 1}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_nonfinite_alg_update_func_arg_orch():
    """
    The same check at its OTHER call site: a NaN in a supplied reward vector inside
    alg_update_func_args. The orchestrator calls require_supplied_args_finite once per argument
    dictionary, and the message has to name which dictionary and key kind it is talking about
    (policy_num here rather than decision_time).
    """
    study = _build_study_orch()
    original_args = study["alg_update_func_args"][3][_SUBJECT_IDS_ORCH[0]]
    study["alg_update_func_args"][3][_SUBJECT_IDS_ORCH[0]] = original_args[:4] + (
        jnp.array([jnp.nan, 2.2]),
    )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 supplied alg_update_func argument position(s) contain non-finite values (NaN or "
            "inf). Offending (policy_num, subject_id, arg position) -> count, up to 5 shown: "
            "{(3, 1, 4): 1}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_wrong_length_action_prob_beta_orch():
    """
    One recorded action-prob beta has three components where beta_dim is two. beta_dim is used
    to slice and index the stacked system everywhere, so a beta of another length silently
    changes which components of the joint bread and meat matrices mean what.

    The reconstruction check would also object to this tuple -- a three-component beta cannot be
    dotted with two-component features -- but it runs later, and its message would be about a
    dot-product shape rather than about beta_dim.
    """
    study = _build_study_orch()
    original_args = study["action_prob_func_args"][1][_SUBJECT_IDS_ORCH[0]]
    study["action_prob_func_args"][1][_SUBJECT_IDS_ORCH[0]] = (
        jnp.append(original_args[0], 0.0),
    ) + original_args[1:]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 action_prob_func_args beta(s) are not shaped (2,), the dimension taken from the "
            "first supplied beta. Offending (decision_time, subject_id) -> shape, up to 5 "
            "shown: {(1, 1): (3,)}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_wrong_length_update_beta_orch():
    """
    The same defect on the update side: one alg-update beta with three components. It is still
    1D, so require_beta_is_1D_array_in_alg_update_args passes it, and
    helper_functions.collect_all_post_update_betas would eventually fail on the ragged
    jnp.array() with a message naming no policy.
    """
    study = _build_study_orch()
    original_args = study["alg_update_func_args"][3][_SUBJECT_IDS_ORCH[0]]
    study["alg_update_func_args"][3][_SUBJECT_IDS_ORCH[0]] = (
        jnp.append(original_args[0], 0.0),
    ) + original_args[1:]

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "1 alg_update_func_args beta(s) are not shaped (2,). Offending (policy_num, "
            "subject_id) -> shape, up to 5 shown: {(3, 1): (3,)}."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_wrong_row_count_previous_betas_orch():
    """
    Update policy 2's previous-betas block carries ONE row -- the initial policy's beta -- where
    the check requires zero.

    This is precisely the shape this module's own fixture used before
    require_beta_dimensions_consistent existed, and it is wrong because
    arg_threading_helpers.thread_update_func_args slices the shared post-update beta history by
    the RECORDED ROW COUNT, and that history contains only update-produced betas: nothing
    precedes the first update. Both subjects are given the bad block so that
    require_previous_betas_match_in_alg_update_args_each_update is not the check that fires.
    """
    study = _build_study_orch()
    for subject_id in _SUBJECT_IDS_ORCH:
        original_args = study["alg_update_func_args"][2][subject_id]
        study["alg_update_func_args"][2][subject_id] = (
            original_args[:1]
            + (jnp.stack([_BETA_BY_POLICY_NUM_ORCH[1]]),)
            + original_args[2:]
        )

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "2 alg_update_func_args previous-betas block(s) have the wrong shape; each must be "
            "(number of updates before that policy, 2)"
        ),
    ) as exception_info:
        input_checks.perform_first_wave_input_checks(**study)

    assert "(1, 2) != expected (0, 2)" in str(exception_info.value)


def test_perform_first_wave_input_checks_rejects_nonfinite_theta_estimate_orch():
    """
    A theta estimate with a NaN component. theta seeds the joint estimating system, so this
    makes every downstream quantity NaN; require_theta_is_1D_array, which runs first, only
    looks at the number of dimensions.
    """
    study = _build_study_orch()
    study["theta_est"] = jnp.array([0.25, jnp.nan, 1.5])

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "theta_calculation_func returned a theta estimate with non-finite values at "
            "component(s) [1]"
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_empty_theta_estimate_orch():
    """
    An empty theta estimate. It is 1D, so the older shape check accepts it, and a theta_dim of
    zero turns the stacked-system arithmetic into a silently degenerate no-op rather than an
    error.
    """
    study = _build_study_orch()
    study["theta_est"] = jnp.array([])

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "theta_calculation_func returned an empty theta estimate; the inferential target "
            "must have at least one component."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_rejects_betas_disagreeing_within_one_policy_orch():
    """
    Policy 3 is in force at the last TWO decision times, and the recorded action-prob beta
    differs between them -- with the last time's recorded probabilities derived from its own
    beta, so reconstruction still passes.

    This is the new invariant in require_betas_match_in_action_prob_func_args_each_policy: a
    policy is one parameter vector, and the threading code substitutes a single shared beta per
    policy, so every cell in force under a policy must record the same beta. The
    recorded-vs-update comparison would also object to the odd beta, but it runs afterwards.
    """
    study = _build_study_orch(beta_disagrees_within_policy=True)

    with pytest.raises(
        AssertionError,
        match=re.escape(
            "The action prob args record different betas for cells sharing a policy number, at "
            "1 cell(s)."
        ),
    ):
        input_checks.perform_first_wave_input_checks(**study)


def test_perform_first_wave_input_checks_accepts_two_policies_at_one_decision_time_orch():
    """
    Two DIFFERENT policies in force at the SAME decision time, recording different betas, is
    ACCEPTED. This test is the inverse of the one it replaces.

    require_betas_match_in_action_prob_func_args_each_decision required every subject at a given
    calendar time to share one beta, which fired on exactly this configuration -- and multiple
    policies at one decision time is supported and expected (fallback policies interleave with
    the current one, and verify_analysis_df_summary_satisfactory reports the count of such times
    as an ordinary statistic). Its replacement keys on the POLICY number instead, so it is more
    permissive here and stricter where it matters.
    """
    study = _build_study_orch(mixed_policies_at_one_time=True)

    analysis_df = study["analysis_df"]
    policy_nums_at_third_time = set(
        analysis_df.loc[
            (analysis_df["calendar_t"] == 3) & (analysis_df["in_study"] == 1),
            "policy_num",
        ]
    )
    assert policy_nums_at_third_time == {2.0, 3.0}
    first_subject_beta = study["action_prob_func_args"][3][_SUBJECT_IDS_ORCH[0]][0]
    second_subject_beta = study["action_prob_func_args"][3][_SUBJECT_IDS_ORCH[-1]][0]
    assert not np.array_equal(first_subject_beta, second_subject_beta)

    _assert_first_wave_passes_orch(study)
