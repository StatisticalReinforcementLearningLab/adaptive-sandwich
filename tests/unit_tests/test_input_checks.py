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
    # one. get_batched_arg_lists_and_involved_user_ids must not rely on
    # func.__code__.co_argcount to find the arg count.
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
# require_estimating_functions_sum_to_zero_se_standardized: the residual is judged by how far
# it displaces each stacked estimate in units of that estimate's own SE (portable across reward
# scales), not by a raw-units absolute tolerance. Fixtures: num_updates updates (beta_dim=2)
# plus theta_dim=2, so the stack has 2*num_updates + 2 components; B and V are diagonal so the
# expected statistic is hand-derivable and each block's displacement is isolated.
# ---------------------------------------------------------------------------


def _sum_to_zero_fixture(se=0.1, bread_scale=100.0, num_updates=1):
    beta_dim, theta_dim = 2, 2
    dim = num_updates * beta_dim + theta_dim
    B = np.eye(dim) * bread_scale
    V = np.eye(dim) * se**2
    return B, V, beta_dim, theta_dim


def _per_block_displacements(message):
    """
    The max SE-standardized displacement the failure text reports for each block, keyed by
    block label ("update 1", ..., "inference").
    """
    return {
        label: float(value)
        for label, value in re.findall(
            r"^(update \d+|inference): max displacement (\S+) SE", message, re.MULTILINE
        )
    }


def test_se_standardized_sum_to_zero_passes_where_raw_units_would_hard_fail():
    # Steep equations (bread scale 100): a raw residual of 0.02 -- past the legacy hard gate of
    # 1e-2 -- displaces each estimate by only 0.02/100 = 2e-4, i.e. 2e-3 of its SE. The legacy
    # check raises on exactly this input; the SE-standardized one must pass it.
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture()
    r = np.full(4, 0.02)

    with pytest.raises(AssertionError):
        input_checks.require_estimating_functions_sum_to_zero(
            jnp.asarray(r), beta_dim, theta_dim, suppress_interactive_data_checks=True
        )

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(r), B, V, beta_dim, theta_dim, suppress_interactive_data_checks=True
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
    # Residual engineered to displace ONE block's estimate by 0.2 of its SE (past the 0.1 hard
    # tolerance) while leaving every other block exactly at zero; the error must say which
    # block. Asserting only that "update 1" appears somewhere would test nothing: the failure
    # text ends in a breakdown that prints one line per block on ANY hard failure, so an
    # off-by-one that pointed users at the wrong estimating equations would still raise, still
    # match, and still pass. What is pinned here is the attribution itself -- the named
    # offender, the "<-- largest" marker, and the numbers on the other blocks' lines.
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture(num_updates=2)
    displacement = np.zeros(6)
    displacement[offending_component] = 0.2 * 0.1
    r = B @ displacement

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            jnp.asarray(r),
            B,
            V,
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )

    message = str(excinfo.value)
    assert re.search(rf"displaces {offending_block} component \d+ by 0\.2 ", message), (
        message
    )
    for other_block in {"update 1", "update 2", "inference"} - {offending_block}:
        assert f"displaces {other_block} " not in message

    per_block = _per_block_displacements(message)
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
    beta_dim, theta_dim, num_updates = 2, 5, 1
    dim = num_updates * beta_dim + theta_dim
    B = np.eye(dim) * 100.0
    V = np.eye(dim) * 0.1**2
    displacement = np.zeros(dim)
    displacement[offending_component] = 0.2 * 0.1
    r = B @ displacement

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            jnp.asarray(r),
            B,
            V,
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )

    message = str(excinfo.value)
    assert re.search(rf"displaces {offending_block} component \d+ by 0\.2 ", message), (
        message
    )
    per_block = _per_block_displacements(message)
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
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture(num_updates=2)
    displacement = np.zeros(6)
    displacement[2] = 0.05 * 0.1  # between the 0.01 soft and 0.1 hard tolerances
    r = B @ displacement

    prompts = []

    def _capture_prompt(message):
        prompts.append(message)
        return "y"  # anything but y/n re-prompts forever

    monkeypatch.setattr("builtins.input", _capture_prompt)

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(r),
        B,
        V,
        beta_dim,
        theta_dim,
        suppress_interactive_data_checks=False,
    )

    assert len(prompts) == 1
    assert "displaces update 2 component 0" in prompts[0]
    assert _per_block_displacements(prompts[0]) == {
        "update 1": pytest.approx(0.0, abs=1e-9),
        "update 2": pytest.approx(0.05, rel=1e-3),
        "inference": pytest.approx(0.0, abs=1e-9),
    }


def test_se_standardized_sum_to_zero_raises_assertion_error_on_exactly_singular_bread():
    # An exactly singular joint bread makes np.linalg.solve RAISE rather than return inf/nan,
    # so the nonfinite-displacement guard never sees it: without an explicit catch the check
    # escapes as a bare numpy.linalg.LinAlgError, past every caller that handles the designed
    # AssertionError, and with none of the guidance the designed message carries.
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture()
    B = B.copy()
    B[1, 1] = 0.0

    with pytest.raises(AssertionError) as excinfo:
        input_checks.require_estimating_functions_sum_to_zero_se_standardized(
            jnp.asarray(np.ones(4)),
            B,
            V,
            beta_dim,
            theta_dim,
            suppress_interactive_data_checks=True,
        )

    assert "bread_stability" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, np.linalg.LinAlgError)


def test_se_standardized_sum_to_zero_soft_band_confirms_instead_of_raising():
    # Displacement of 0.05 SE sits between soft (0.01) and hard (0.1): with interaction
    # suppressed this must log-and-continue, never raise.
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture()
    displacement = np.full(4, 0.05 * 0.1)
    r = B @ displacement

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(r), B, V, beta_dim, theta_dim, suppress_interactive_data_checks=True
    )


def test_se_standardized_sum_to_zero_excludes_zero_variance_components():
    # A component with (numerically) zero variance is a rank/identification finding for
    # bread_stability, not a sum-to-zero failure: a huge displacement confined to that
    # component must not raise here.
    B, V, beta_dim, theta_dim = _sum_to_zero_fixture()
    V = V.copy()
    V[0, 0] = 0.0
    displacement = np.array([5.0, 0.0, 0.0, 0.0])
    r = B @ displacement

    input_checks.require_estimating_functions_sum_to_zero_se_standardized(
        jnp.asarray(r), B, V, beta_dim, theta_dim, suppress_interactive_data_checks=True
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

    assert atol.shape == (1, 2)
    np.testing.assert_allclose(atol, [[1e-2, 5e-7]])


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
    # A component identically zero across subjects has no scale of its own, so it borrows the
    # SMALLEST nonzero component scale -- not the array's overall scale, which would reopen the
    # cross-component leak this function exists to close. With a 1e4-vs-0.5 spread the array
    # scale would hand the zero component atol=1e-2, so a mismatch anywhere in 1e-7..1e-2 there
    # would pass: looser than the fixed 1e-7 this replaced, i.e. a regression against main.
    np.testing.assert_allclose(
        input_checks.componentwise_absolute_tolerance(
            np.array([[1e4, 0.5, 0.0], [1e4, 0.5, 0.0]])
        ),
        [[1e-2, 5e-7, 5e-7]],
    )
    # The same rule with only one nonzero component: it is both the max and the min.
    np.testing.assert_allclose(
        input_checks.componentwise_absolute_tolerance(
            np.array([[1.0, 0.0], [2.0, 0.0]])
        ),
        [[2e-6, 2e-6]],
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
        [[1e-6, 1e-6]],
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
