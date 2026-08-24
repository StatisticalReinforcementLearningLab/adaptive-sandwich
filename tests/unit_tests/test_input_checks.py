import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from lifejacket import input_checks


def _action_prob_func(beta, features):
    return jnp.dot(beta, features)


def _build_reconstruction_fixture(*, blank_active_cell=False, nonblank_inactive_cell=False):
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
                {"calendar_t": t, "user_id": subject_id, "active": 0, "action_prob": np.nan}
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


def _algorithm_estimating_func(beta, features):
    return beta * features


def test_require_threaded_algorithm_estimating_function_args_equivalent_passes():
    beta = jnp.array([1.0, 2.0])
    features_by_subject = {0: jnp.array([0.1, 0.2]), 1: jnp.array([0.3, 0.4])}
    update_func_args_by_by_subject_id_by_policy_num = {
        1: {subject_id: (beta, features) for subject_id, features in features_by_subject.items()}
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
        1: {subject_id: (beta, features) for subject_id, features in features_by_subject.items()}
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
