import re

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks


def _action_prob_func(beta, features):
    return jnp.dot(beta, features)


def _build_reconstruction_fixture(
    *, blank_active_cell=False, nonblank_inactive_cell=False
):
    """
    Two decision times (0, 1), two subjects (0, 1). analysis_df marks every
    (t, subject) active except (1, 1). action_prob_func_args are exactly the
    values that reproduce analysis_df's action_prob column for active cells (and
    an empty tuple for the inactive one), so the check should pass by default.

    blank_active_cell=True makes action_prob_func_args[1][0] an empty tuple even
    though analysis_df marks (t=1, subject=0) active -- the "missing_keys" error path.
    nonblank_inactive_cell=True gives action_prob_func_args[1][1] a real (non-empty)
    tuple even though analysis_df still marks (t=1, subject=1) inactive -- the
    "unexpected_keys" error path.
    """
    beta = jnp.array([1.0, 2.0])
    # (t, subject_id) -> (features, is_active_in_analysis_df)
    cells = {
        (0, 0): (jnp.array([0.1, 0.2]), True),
        (0, 1): (jnp.array([0.3, 0.4]), True),
        (1, 0): (jnp.array([0.5, 0.6]), True),
        (1, 1): (jnp.array([0.7, 0.8]), False),
    }

    rows = []
    action_prob_func_args = {0: {}, 1: {}}
    for (t, subject_id), (features, is_active) in cells.items():
        if is_active:
            rows.append(
                {
                    "calendar_t": t,
                    "user_id": subject_id,
                    "active": 1,
                    "action_prob": float(_action_prob_func(beta, features)),
                }
            )
            action_prob_func_args[t][subject_id] = (beta, features)
        else:
            rows.append(
                {
                    "calendar_t": t,
                    "user_id": subject_id,
                    "active": 0,
                    "action_prob": np.nan,
                }
            )
            action_prob_func_args[t][subject_id] = (
                (beta, features) if nonblank_inactive_cell else ()
            )

    if blank_active_cell:
        action_prob_func_args[1][0] = ()

    analysis_df = pd.DataFrame(rows)
    return analysis_df, action_prob_func_args


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_passes():
    analysis_df, action_prob_func_args = _build_reconstruction_fixture()

    input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
        analysis_df,
        "action_prob",
        "calendar_t",
        "user_id",
        "active",
        action_prob_func_args,
        _action_prob_func,
    )


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_passes_jit_wrapped_func():
    # Regression test: a jax.jit-wrapped action_prob_func is a PjitFunction,
    # which has no __code__, unlike a plain function or a jax.grad-wrapped
    # one. The check's argument batching must not rely on
    # func.__code__.co_argcount to find the arg count (batch_args_by_subject
    # derives it from the supplied data, so any wrapper works).
    analysis_df, action_prob_func_args = _build_reconstruction_fixture()

    input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
        analysis_df,
        "action_prob",
        "calendar_t",
        "user_id",
        "active",
        action_prob_func_args,
        jax.jit(_action_prob_func),
    )


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_mismatch_fails():
    analysis_df, action_prob_func_args = _build_reconstruction_fixture()
    # Corrupt one recorded action probability so it no longer matches what
    # action_prob_func actually produces from its args.
    analysis_df.loc[
        (analysis_df["calendar_t"] == 0) & (analysis_df["user_id"] == 0), "action_prob"
    ] = 999.0

    with pytest.raises(AssertionError):
        input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
            analysis_df,
            "action_prob",
            "calendar_t",
            "user_id",
            "active",
            action_prob_func_args,
            _action_prob_func,
        )


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_missing_key_fails_clearly():
    # An active (decision_time, subject_id) with no corresponding non-blank args --
    # must fail with a clear, specific error, not an opaque crash.
    analysis_df, action_prob_func_args = _build_reconstruction_fixture(
        blank_active_cell=True
    )

    with pytest.raises(ValueError, match="could not reconstruct"):
        input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
            analysis_df,
            "action_prob",
            "calendar_t",
            "user_id",
            "active",
            action_prob_func_args,
            _action_prob_func,
        )


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_unexpected_key_fails_clearly():
    # A non-blank args entry for a (decision_time, subject_id) analysis_df marks
    # inactive -- must fail with a clear, specific error (not a bare KeyError) since
    # this scenario has no corresponding "actual" row to compare against at all.
    analysis_df, action_prob_func_args = _build_reconstruction_fixture(
        nonblank_inactive_cell=True
    )

    with pytest.raises(ValueError, match="does not mark"):
        input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
            analysis_df,
            "action_prob",
            "calendar_t",
            "user_id",
            "active",
            action_prob_func_args,
            _action_prob_func,
        )


def test_require_action_probabilities_in_analysis_df_can_be_reconstructed_duplicate_active_rows_fails_clearly():
    # Two active rows for the same (calendar_t, subject_id) -- must fail loudly
    # instead of silently keeping only one via dict-key collision, which would
    # quietly drop the duplicate from this check's coverage.
    analysis_df, action_prob_func_args = _build_reconstruction_fixture()
    duplicate_row = analysis_df[
        (analysis_df["calendar_t"] == 0) & (analysis_df["user_id"] == 0)
    ]
    analysis_df = pd.concat([analysis_df, duplicate_row], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate active rows"):
        input_checks.require_action_probabilities_in_analysis_df_can_be_reconstructed(
            analysis_df,
            "action_prob",
            "calendar_t",
            "user_id",
            "active",
            action_prob_func_args,
            _action_prob_func,
        )


def _algorithm_estimating_func(beta, features):
    return beta * features


def test_require_threaded_algorithm_estimating_function_args_equivalent_passes():
    beta = jnp.array([1.0, 2.0])
    features_by_subject = {0: jnp.array([0.1, 0.2]), 1: jnp.array([0.3, 0.4])}
    update_func_args_by_by_subject_id_by_policy_num = {
        1: {
            subject_id: (beta, features)
            for subject_id, features in features_by_subject.items()
        }
    }
    # "Threaded" args carry the exact same values, just as a separate object --
    # matching how threading re-derives an equal (not necessarily identical) beta.
    threaded_update_func_args_by_policy_num_by_subject_id = {
        subject_id: {1: (jnp.array(beta), jnp.array(features))}
        for subject_id, features in features_by_subject.items()
    }

    input_checks.require_threaded_algorithm_estimating_function_args_equivalent(
        _algorithm_estimating_func,
        update_func_args_by_by_subject_id_by_policy_num,
        threaded_update_func_args_by_policy_num_by_subject_id,
        suppress_interactive_data_checks=True,
    )


def test_require_threaded_algorithm_estimating_function_args_equivalent_mismatch_fails():
    beta = jnp.array([1.0, 2.0])
    features_by_subject = {0: jnp.array([0.1, 0.2]), 1: jnp.array([0.3, 0.4])}
    update_func_args_by_by_subject_id_by_policy_num = {
        1: {
            subject_id: (beta, features)
            for subject_id, features in features_by_subject.items()
        }
    }
    threaded_update_func_args_by_policy_num_by_subject_id = {
        0: {1: (jnp.array(beta), jnp.array([0.1, 0.2]))},
        # Subject 1's threaded features deliberately don't match its original ones.
        1: {1: (jnp.array(beta), jnp.array([9.0, 9.0]))},
    }

    with pytest.raises(AssertionError):
        input_checks.require_threaded_algorithm_estimating_function_args_equivalent(
            _algorithm_estimating_func,
            update_func_args_by_by_subject_id_by_policy_num,
            threaded_update_func_args_by_policy_num_by_subject_id,
            suppress_interactive_data_checks=True,
        )


def _inference_estimating_func(theta, features):
    return theta * features


def test_require_threaded_inference_estimating_function_args_equivalent_passes():
    theta = jnp.array([1.0, 2.0])
    inference_func_args_by_subject_id = {
        0: (theta, jnp.array([0.1, 0.2])),
        1: (theta, jnp.array([0.3, 0.4])),
    }
    threaded_inference_func_args_by_subject_id = {
        subject_id: (jnp.array(theta), jnp.array(features))
        for subject_id, (_, features) in inference_func_args_by_subject_id.items()
    }

    input_checks.require_threaded_inference_estimating_function_args_equivalent(
        _inference_estimating_func,
        inference_func_args_by_subject_id,
        threaded_inference_func_args_by_subject_id,
        suppress_interactive_data_checks=True,
    )


def test_require_threaded_inference_estimating_function_args_equivalent_mismatch_fails():
    theta = jnp.array([1.0, 2.0])
    inference_func_args_by_subject_id = {
        0: (theta, jnp.array([0.1, 0.2])),
        1: (theta, jnp.array([0.3, 0.4])),
    }
    threaded_inference_func_args_by_subject_id = {
        0: (jnp.array(theta), jnp.array([0.1, 0.2])),
        # Subject 1's threaded features deliberately don't match its original ones.
        1: (jnp.array(theta), jnp.array([9.0, 9.0])),
    }

    with pytest.raises(AssertionError):
        input_checks.require_threaded_inference_estimating_function_args_equivalent(
            _inference_estimating_func,
            inference_func_args_by_subject_id,
            threaded_inference_func_args_by_subject_id,
            suppress_interactive_data_checks=True,
        )


# ---------------------------------------------------------------------------
# require_estimating_functions_sum_to_zero_se_standardized: each component of the residual (the
# subject-mean estimating function) is judged against its own standard error, taken directly
# from the per-subject values that were averaged (a_j = |mean psi_j| / (rms psi_j / sqrt(n))),
# so the check is portable across reward scales and consults neither the bread nor the sandwich
# (its earlier displacement form inherited both matrices' degeneracies -- see the regression
# tests at the end of this block). Fixtures: _stacks_with_residuals builds per-subject stacks
# whose component-j statistic is EXACTLY the requested value -- half the subjects at +d, half
# at -d (rms pinned to `scale` exactly) plus the mean offset that produces the statistic -- so
# every expected number is hand-derivable and each block's residual is isolated.
# ---------------------------------------------------------------------------


def _stacks_with_residuals(
    residuals_in_se, beta_dim=2, theta_dim=2, scale=1.0, num_subjects=100
):
    residuals_in_se = np.asarray(residuals_in_se, dtype=np.float64)
    dim = residuals_in_se.size
    assert (dim - theta_dim) % beta_dim == 0
    m = residuals_in_se * scale / np.sqrt(num_subjects)  # per-component mean
    d = np.sqrt(
        scale**2 - m**2
    )  # alternating spread pinning the rms to exactly `scale`
    signs = np.tile([1.0, -1.0], num_subjects // 2)[:, None]
    stacks = m + signs * d
    return jnp.asarray(stacks), beta_dim, theta_dim


def _per_block_residuals(message):
    """
    The max SE-standardized residual the failure text reports for each block, keyed by block
    label ("update 1", ..., "inference").
    """
    return {
        label: float(value)
        for label, value in re.findall(
            r"^(update \d+|inference): max residual (\S+) SE", message, re.MULTILINE
        )
    }


def test_se_standardized_sum_to_zero_passes_where_raw_units_would_hard_fail():
    # High reward scale (per-subject terms of size 1000): a residual of 0.002 SE corresponds to
    # a raw mean of 0.002 * 1000 / sqrt(100) = 0.2 -- past the legacy hard gate of 1e-2. The
    # legacy check raises on exactly this input; the SE-standardized one must pass it.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(
        np.full(4, 0.002), scale=1000.0
    )

    with pytest.raises(AssertionError):
        input_checks.require_estimating_functions_sum_to_zero(
            jnp.mean(stacks, axis=0),
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        stacks, beta_dim, theta_dim, suppress_interactive_data_checks=True
    )


@pytest.mark.parametrize(
    ("offending_component", "offending_block"),
    # A 2-update stack (beta_dim=2) plus theta_dim=2 gives flat components 0-1 = update 1,
    # 2-3 = update 2, 4-5 = inference. Index 2 sits exactly on a block boundary and index 5 at
    # the end of the stack, so an off-by-one in the block slicing shows up as a wrong label
    # rather than cancelling out.
    [(0, "update 1"), (2, "update 2"), (5, "inference")],
)
def test_se_standardized_sum_to_zero_hard_failure_attributes_to_the_offending_block(
    offending_component, offending_block
):
    # Residual engineered to 0.2 of ONE component's SE (past the 0.1 hard tolerance) while
    # leaving every other component exactly at zero; the error must say which block. Asserting
    # only that "update 1" appears somewhere would test nothing: the failure text ends in a
    # breakdown that prints one line per block on ANY hard failure, so an off-by-one that
    # pointed users at the wrong estimating equations would still raise, still match, and still
    # pass. What is pinned here is the attribution itself -- the named offender, the
    # "<-- largest" marker, and the numbers on the other blocks' lines.
    residuals = np.zeros(6)
    residuals[offending_component] = 0.2
    stacks, beta_dim, theta_dim = _stacks_with_residuals(residuals)

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            stacks, beta_dim, theta_dim, suppress_interactive_data_checks=True
        )

    message = str(excinfo.value)
    assert re.search(
        rf"residual for {offending_block} component \d+ is 0\.2 ", message
    ), message
    for other_block in {"update 1", "update 2", "inference"} - {offending_block}:
        assert f"residual for {other_block} " not in message

    per_block = _per_block_residuals(message)
    assert set(per_block) == {"update 1", "update 2", "inference"}
    assert per_block[offending_block] == pytest.approx(0.2, rel=1e-3)
    for other_block, value in per_block.items():
        if other_block != offending_block:
            assert value == pytest.approx(0.0, abs=1e-9)
    # Exactly one line is marked as the largest, and it is the offending block's.
    assert message.count("<-- largest") == 1
    assert re.search(rf"^{offending_block}: .*<-- largest$", message, re.MULTILINE)


@pytest.mark.parametrize(
    ("offending_component", "offending_block"),
    # beta_dim=2 with theta_dim=5 and ONE update: flat components 0-1 = update 1, 2-6 =
    # inference. The case above cannot detect a missing update/inference boundary guard, because
    # at beta_dim == theta_dim == 2 the quotient component // beta_dim happens to equal
    # num_updates at every inference index anyway -- deleting the guard leaves it green. Here
    # index 2 is the first inference component and index 4 lands mid-inference, where the bare
    # quotient would mislabel them "update 2" and "update 3".
    [(0, "update 1"), (2, "inference"), (4, "inference"), (6, "inference")],
)
def test_se_standardized_sum_to_zero_attribution_holds_across_the_inference_boundary(
    offending_component, offending_block
):
    residuals = np.zeros(7)
    residuals[offending_component] = 0.2
    stacks, beta_dim, theta_dim = _stacks_with_residuals(
        residuals, beta_dim=2, theta_dim=5
    )

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            stacks, beta_dim, theta_dim, suppress_interactive_data_checks=True
        )

    message = str(excinfo.value)
    assert re.search(
        rf"residual for {offending_block} component \d+ is 0\.2 ", message
    ), message
    per_block = _per_block_residuals(message)
    assert set(per_block) == {"update 1", "inference"}
    assert per_block[offending_block] == pytest.approx(0.2, rel=1e-3)
    assert message.count("<-- largest") == 1
    assert re.search(rf"^{offending_block}: .*<-- largest$", message, re.MULTILINE)


def test_se_standardized_sum_to_zero_soft_band_prompt_contains_the_breakdown(
    monkeypatch,
):
    # The interactive prompt has to be self-contained: this package installs no logging
    # handler, so a prompt that says "see the per-block breakdown above" shows the user nothing
    # unless they configured logging themselves.
    residuals = np.zeros(6)
    residuals[2] = 0.05  # between the 0.01 soft and 0.1 hard tolerances
    stacks, beta_dim, theta_dim = _stacks_with_residuals(residuals)

    prompts = []

    def _capture_prompt(message):
        prompts.append(message)
        return "y"  # anything but y/n re-prompts forever

    monkeypatch.setattr("builtins.input", _capture_prompt)

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        stacks, beta_dim, theta_dim, suppress_interactive_data_checks=False
    )

    assert len(prompts) == 1
    assert "residual for update 2 component 0" in prompts[0]
    assert _per_block_residuals(prompts[0]) == {
        "update 1": pytest.approx(0.0, abs=1e-9),
        "update 2": pytest.approx(0.05, rel=1e-3),
        "inference": pytest.approx(0.0, abs=1e-9),
    }


def test_se_standardized_sum_to_zero_soft_band_confirms_instead_of_raising():
    # Residual of 0.05 SE sits between soft (0.01) and hard (0.1): with interaction suppressed
    # this must log-and-continue, never raise.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(np.full(4, 0.05))

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        stacks, beta_dim, theta_dim, suppress_interactive_data_checks=True
    )


def test_se_standardized_sum_to_zero_treats_identically_zero_components_as_rooted():
    # A component identically zero across all subjects has s_j == 0 AND r_j == 0: a trivially
    # rooted equation, not a 0/0 or a masked failure. The displacement form "excluded" such
    # components -- and silently PASSED a stack whose every SE had collapsed to zero. Here the
    # zero-denominator case is provably benign (s_j == 0 forces r_j == 0), so the check must
    # pass without warning or division error, and other components must still be judged.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(np.zeros(4))
    stacks = np.array(stacks)  # jnp arrays view as read-only; copy to mutate
    stacks[:, 0] = 0.0

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(stacks), beta_dim, theta_dim, suppress_interactive_data_checks=True
    )

    # An all-zero stack -- every component trivially rooted -- must also pass, not blow up.
    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.zeros((100, 4)), beta_dim, theta_dim, suppress_interactive_data_checks=True
    )


def test_se_standardized_sum_to_zero_skips_components_below_the_noise_floor():
    # Reproduces the RL smoke fixture failure that motivated the floor: a component whose only
    # nonzero per-subject values are two identical 2-ulp float32 rounding residues (2.4e-7,
    # non-cancelling) among otherwise O(1) components. Its dispersion IS the float noise, so
    # the unfloored statistic reads exactly sqrt(2) SE by construction -- not a rooting
    # failure. Below the floor (relative_noise_floor * max_k s_k) the component must be
    # trivially rooted; the skip is provably safe because |r_j| <= s_j bounds the skipped
    # residual by the floor itself.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(np.zeros(4))
    stacks = np.array(stacks)  # jnp arrays view as read-only; copy to mutate
    stacks[:, 1] = 0.0
    stacks[3, 1] = 2.4e-7
    stacks[6, 1] = 2.4e-7

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(stacks), beta_dim, theta_dim, suppress_interactive_data_checks=True
    )

    # The same two-subject non-cancelling shape ABOVE the floor is a genuine violation and
    # must still hard-fail (at 0.5 against unit-scale neighbors, a = sqrt(2) > 0.1): the floor
    # must excuse only float noise, not real non-cancellation.
    stacks[3, 1] = 0.5
    stacks[6, 1] = 0.5
    with pytest.raises(AssertionError, match="update 1 component 1"):
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            jnp.asarray(stacks),
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )


@pytest.mark.parametrize("scale", [1.0, 1e6])
def test_se_standardized_sum_to_zero_detection_is_reward_scale_free(scale):
    # REGRESSION AGAINST THE DISPLACEMENT FORM: detection must not depend on the size of the
    # per-subject terms. The displacement form divided by the sandwich SE, so a meat-driven
    # blow-up (U5/B_influence: SE spread ~2e5x) masked genuinely unrooted equations -- a
    # residual displacing estimates by 1000 raw units passed. Here numerator and denominator
    # share the reward units, so the same 0.2 SE violation raises identically at term scale 1
    # and term scale 1e6.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(np.full(4, 0.2), scale=scale)

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            stacks, beta_dim, theta_dim, suppress_interactive_data_checks=True
        )

    assert "is 0.2 of its own standard error" in str(excinfo.value)


def test_se_standardized_sum_to_zero_raises_on_nonfinite_stacks():
    # A nonfinite subject poisons r and s into nan, and nan > 0 is False -- without an explicit
    # guard the where-mask would leave a == 0 everywhere and the check would silently PASS on
    # garbage input.
    stacks, beta_dim, theta_dim = _stacks_with_residuals(np.zeros(4))
    stacks = np.array(stacks)  # jnp arrays view as read-only; copy to mutate
    stacks[0, 0] = np.nan

    with pytest.raises(AssertionError, match="nonfinite"):
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            jnp.asarray(stacks),
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )


# ---------------------------------------------------------------------------
# componentwise_absolute_tolerance / require_original_and_threaded_results_agree: the one
# original-vs-threaded comparison rule, shared by the two unbatched checks above and by
# batched_weighted_estimating_function_stack's bucket comparison. Both degenerate cases below
# are REGRESSIONS AGAINST MAIN: main compared with np.testing.assert_allclose under
# helper_functions.array_scale_absolute_tolerance, whose single scalar floor is derived from the
# array's global maximum -- which both masks small components and demands bit-exactness of an
# all-zero reference. Each test asserts the correct behavior and, alongside it, what the scalar
# floor did, so a revert cannot pass quietly.
# ---------------------------------------------------------------------------


def test_componentwise_absolute_tolerance_confines_each_component_to_its_own_scale():
    # First axis is the subject/batch axis every jax.vmap in this package produces; the second
    # indexes components, which carry genuinely different units (an intercept score vs. a
    # reward-scaled score).
    reference = np.array([[1e4, 0.5], [0.9e4, 0.4]])

    atol = input_checks.componentwise_absolute_tolerance(reference)

    floor = input_checks.ORIGINAL_VS_THREADED_RELATIVE_FLOOR
    assert atol.shape == (1, 2)
    np.testing.assert_allclose(atol, [[1e4 * floor, 0.5 * floor]])


def test_original_and_threaded_agreement_catches_a_small_component_mismatch():
    # A 1% inconsistency in an O(1) component, sitting in the same array as a ~1e4 one -- the
    # realistic reconstruction failure the global-max floor swallowed whole (its atol there is
    # 1e-6 * 1e4 = 1e-2, i.e. 2x the error itself).
    original = np.array([[1e4, 0.5], [1e4, 0.5]])
    threaded = original.copy()
    threaded[1, 1] = 0.505

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_original_and_threaded_results_agree(
            original, threaded, rtol=1e-3, context="update 3, subjects [7, 9]"
        )
    message = str(excinfo.value)
    assert "update 3, subjects [7, 9]" in message
    assert "(1, 1)" in message

    # What main did with the same inputs, and the reason this is a regression test and not just
    # a unit test: under one global-max atol the comparison passed.
    np.testing.assert_allclose(
        original,
        threaded,
        rtol=1e-3,
        atol=1e-6 * np.max(np.abs(original)),
    )

    # The scale-awareness that floor existed for is retained: float noise proportional to the
    # LARGE component's own magnitude still passes.
    noisy_large = original.copy()
    noisy_large[0, 0] = 1e4 + 5e-3
    input_checks.require_original_and_threaded_results_agree(
        original, noisy_large, rtol=1e-3, context="ctx"
    )


def test_original_and_threaded_agreement_tolerates_noise_against_an_all_zero_reference():
    # A degenerate/burn-in bucket whose original output cancels to exactly zero: two float32
    # computations of a cancelling sum do not agree bitwise, and the global-max floor's 0.0
    # atol demanded exactly that, hard-failing an always-on input check on healthy data.
    original = np.zeros((3, 2))

    input_checks.require_original_and_threaded_results_agree(
        original, np.full((3, 2), 1e-12), rtol=1e-3, context="ctx"
    )
    # Still not a licence to accept anything: 1e-3 against an exact zero is a real
    # disagreement, not float noise.
    with pytest.raises(AssertionError):
        input_checks.require_original_and_threaded_results_agree(
            original, np.full((3, 2), 1e-3), rtol=1e-3, context="ctx"
        )

    # Main's floor for that same reference, i.e. bit-exactness.
    assert 1e-6 * np.max(np.abs(original)) == 0.0


def test_componentwise_absolute_tolerance_degenerate_references():
    floor = input_checks.ORIGINAL_VS_THREADED_RELATIVE_FLOOR
    # A component identically zero across subjects has no scale of its own, so it borrows the
    # SMALLEST nonzero component scale -- not the array's overall scale, which would reopen the
    # cross-component leak this function exists to close. With a 1e4-vs-0.5 spread the array
    # scale would hand the zero component the large component's slack, 2e4x the smallest
    # scale's own floor, so a mismatch anywhere in that band would pass: looser than the fixed
    # 1e-7 this replaced, i.e. a regression against main.
    np.testing.assert_allclose(
        input_checks.componentwise_absolute_tolerance(
            np.array([[1e4, 0.5, 0.0], [1e4, 0.5, 0.0]])
        ),
        [[1e4 * floor, 0.5 * floor, 0.5 * floor]],
    )
    # The same rule with only one nonzero component: it is both the max and the min.
    np.testing.assert_allclose(
        input_checks.componentwise_absolute_tolerance(
            np.array([[1.0, 0.0], [2.0, 0.0]])
        ),
        [[2.0 * floor, 2.0 * floor]],
    )
    # And the consequence that matters: an error injected into the all-zero component beside a
    # 1e4 component is caught across the whole band the array-scale fallback used to swallow.
    original = np.array([[1e4, 0.0, 0.5], [1e4, 0.0, 0.5]])
    for injected_error in (1e-5, 1e-3, 5e-3, 2e-2):
        threaded = original.copy()
        threaded[0, 1] = injected_error
        with pytest.raises(AssertionError):
            input_checks.require_original_and_threaded_results_agree(
                original, threaded, rtol=1e-3, context="ctx"
            )
    # Nothing observable anywhere: fall back to a unit scale, i.e. relative_floor itself.
    np.testing.assert_allclose(
        input_checks.componentwise_absolute_tolerance(np.zeros((3, 2))),
        [[floor, floor]],
    )
    # Empty: nothing to compare in the first place.
    assert input_checks.componentwise_absolute_tolerance(np.array([])).size == 0


def test_original_and_threaded_agreement_preserves_nonfinite_semantics():
    # assert_allclose's own rules, which the explicit comparison had to reimplement: matching
    # nans and matching signed infinities agree, and a mismatch against one does not.
    original = np.array([[np.nan, np.inf, 1.0], [-np.inf, 2.0, 3.0]])
    input_checks.require_original_and_threaded_results_agree(
        original, original.copy(), rtol=1e-3, context="ctx"
    )

    # A nonfinite entry must not poison the whole array's scale into nan and reject every
    # other component along with it.
    tolerable_noise = original.copy()
    tolerable_noise[1, 2] = 3.0 + 1e-9
    input_checks.require_original_and_threaded_results_agree(
        original, tolerable_noise, rtol=1e-3, context="ctx"
    )

    nan_against_number = original.copy()
    nan_against_number[0, 0] = 1.0
    with pytest.raises(AssertionError):
        input_checks.require_original_and_threaded_results_agree(
            original, nan_against_number, rtol=1e-3, context="ctx"
        )

    # Every INFINITY mismatch, in both directions. These are the cases the comment above claims
    # and the reason the tolerance branch is gated on both sides being finite: rtol * |threaded|
    # is itself infinite at an infinite threaded component, so an ungated `difference <= allowed`
    # accepts any original there -- silently blessing a reconstructed action probability that
    # blew the threaded estimating function up, which is precisely what this check exists to
    # catch. Each of these passed before that gate was added.
    for row, column, original_value, threaded_value in [
        (0, 1, np.inf, -np.inf),
        (1, 0, -np.inf, np.inf),
        (1, 1, 2.0, np.inf),
        (1, 1, 2.0, -np.inf),
        (0, 1, np.inf, 2.0),
    ]:
        mismatched_original = original.copy()
        mismatched_threaded = original.copy()
        mismatched_original[row, column] = original_value
        mismatched_threaded[row, column] = threaded_value
        with pytest.raises(AssertionError):
            input_checks.require_original_and_threaded_results_agree(
                mismatched_original, mismatched_threaded, rtol=1e-3, context="ctx"
            )


def test_original_and_threaded_agreement_handles_degenerate_shapes():
    # A scalar-valued estimating function (0-d per subject, 1-D batched) and an empty group
    # must compare without error rather than tripping over the subject-axis reduction.
    for original, threaded in (
        (np.float64(1.0), np.float64(1.0)),
        (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
        (np.array([]), np.array([])),
    ):
        input_checks.require_original_and_threaded_results_agree(
            original, threaded, rtol=1e-3, context="ctx"
        )

    with pytest.raises(AssertionError, match="different shapes"):
        input_checks.require_original_and_threaded_results_agree(
            np.zeros((2, 3)), np.zeros((2, 4)), rtol=1e-3, context="ctx"
        )


def test_original_and_threaded_agreement_tolerates_float32_cancellation_noise():
    # The seed-0 local oralytics run that calibrated ORIGINAL_VS_THREADED_RELATIVE_FLOOR
    # (2026-09-02): a near-fully-cancelling component whose two float32 evaluations differ by
    # ~14 ulps of the component's own scale (values ~4e-4 of that scale), while every
    # same-scale value matches to machine precision. Under the previous 1e-6 floor (~8 ulps)
    # this healthy run hard-failed the always-on equivalence check by a factor of 1.22.
    # Component scale 1.6561e-3 is the run's own: atol was reported as 1.6561057e-09.
    original = np.array([[1.6561057e-03, 0.5], [6.3050538e-07, 0.5]])
    threaded = original.copy()
    threaded[1, 0] = (
        6.3329935e-07  # |difference| 2.79e-9, ~1.7e-6 of the component scale
    )

    input_checks.require_original_and_threaded_results_agree(
        original, threaded, rtol=1e-3, context="ctx"
    )

    # The loosened floor is still a floor, not a hole: the same component at ~10x that noise
    # (well past 100 ulps of its scale) fails.
    threaded[1, 0] = original[1, 0] + 3e-8
    with pytest.raises(AssertionError):
        input_checks.require_original_and_threaded_results_agree(
            original, threaded, rtol=1e-3, context="ctx"
        )


def test_original_and_threaded_agreement_failure_message_is_bounded():
    # The failure message names the worst offenders and SUMMARIZES the arrays instead of
    # dumping them: the module-level np.set_printoptions(threshold=np.inf) otherwise prints
    # every value of both arrays -- observed at ~4,400 terminal lines for a 65x135 bucket,
    # burying the handful of lines that localize the failure.
    original = np.full((80, 40), 0.5)
    threaded = original.copy()
    threaded[3, 7] = 0.6
    threaded[5, 1] = 0.7

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_original_and_threaded_results_agree(
            original, threaded, rtol=1e-3, context="ctx"
        )
    message = str(excinfo.value)
    # Both offenders are named, worst (largest multiple of its own tolerance) first.
    assert message.index("index (5, 1)") < message.index("index (3, 7)")
    assert "2 of 3200 values" in message
    # The 3,200-value arrays appear only in summarized form.
    assert "..." in message
    assert len(message.splitlines()) < 40
