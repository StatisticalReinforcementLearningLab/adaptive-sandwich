"""
End-to-end analyze_dataset runs proving the subject-id column may hold values that are not
small integers. Both cases below raised before 2026-09-02, when
post_deployment_analysis derived `subject_ids` with jnp.array(...):

  - STRING ids raised TypeError("Only arrays of numeric types are supported by JAX") -- and,
    upstream of that, require_all_named_columns_not_object_type_in_analysis_df rejected the
    column for being object dtype.
  - Integer ids at or above 2**31 were silently TRUNCATED by JAX's default 32-bit mode
    (1700000000123 -> -807049093) and then matched no key in any per-subject argument
    dictionary, surfacing as KeyError(-807049215) from deep inside the precompute -- loud, but
    naming a number that appears nowhere in the user's data.

These drive analyze_dataset directly on the benchmark fixture rather than through
run_local_pipeline, deliberately: that fixture wipes the shared simulated_data tree on every
invocation, and nothing here needs the simulator. The assertion is EQUALITY against the
integer-id baseline -- subject ids are opaque keys, so relabelling them must not move a single
digit of the estimates.
"""

import pickle

import numpy as np
import pytest
from tests.utils import get_abs_path

from lifejacket.helper_functions import load_function_from_same_named_file
from lifejacket.post_deployment_analysis import analyze_dataset

FIXTURE_DIR = get_abs_path(__file__, "../../benchmarks/fixtures/small")
FUNCTIONS_DIR = get_abs_path(
    __file__, "../../simulators_and_runners/functions_to_pass_to_analysis"
)


def _run_with_remapped_subject_ids(output_dir, id_mapper):
    """
    Run the real estimator on the benchmark fixture with every subject id passed through
    id_mapper (applied to the analysis DataFrame and to both argument dictionaries' keys, so
    the study stays self-consistent). Returns (theta_est, adjusted_sandwich_var_estimate).
    """
    with open(f"{FIXTURE_DIR}/study_df.pkl", "rb") as f:
        analysis_df = pickle.load(f)
    with open(f"{FIXTURE_DIR}/pi_args.pkl", "rb") as f:
        action_prob_func_args = pickle.load(f)
    with open(f"{FIXTURE_DIR}/rl_update_args.pkl", "rb") as f:
        alg_update_func_args = pickle.load(f)

    if id_mapper is not None:
        analysis_df = analysis_df.copy()
        analysis_df["user_id"] = analysis_df["user_id"].map(id_mapper)
        action_prob_func_args = {
            decision_time: {
                id_mapper[subject_id]: args for subject_id, args in by_subject.items()
            }
            for decision_time, by_subject in action_prob_func_args.items()
        }
        alg_update_func_args = {
            policy_num: {
                id_mapper[subject_id]: args for subject_id, args in by_subject.items()
            }
            for policy_num, by_subject in alg_update_func_args.items()
        }

    analyze_dataset(
        output_dir=output_dir,
        analysis_df=analysis_df,
        action_prob_func=load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/synthetic_get_action_1_prob_generalized_logistic.py"
        ),
        action_prob_func_args=action_prob_func_args,
        action_prob_func_args_beta_index=0,
        alg_update_func=load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/RL_least_squares_loss_regularized.py"
        ),
        alg_update_func_type="loss",
        alg_update_func_args=alg_update_func_args,
        alg_update_func_args_beta_index=0,
        alg_update_func_args_action_prob_index=5,
        alg_update_func_args_action_prob_times_index=6,
        alg_update_func_args_previous_betas_index=-1,
        inference_func=load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/synthetic_get_least_squares_loss_inference_no_action_centering.py"
        ),
        inference_func_type="loss",
        inference_func_args_theta_index=0,
        theta_calculation_func=load_function_from_same_named_file(
            f"{FUNCTIONS_DIR}/synthetic_estimate_theta_least_squares_no_action_centering.py"
        ),
        active_col_name="in_study",
        action_col_name="action",
        policy_num_col_name="policy_num",
        calendar_t_col_name="calendar_t",
        subject_id_col_name="user_id",
        action_prob_col_name="action1prob",
        reward_col_name="reward",
        suppress_interactive_data_checks=True,
        # Checks ON: the point is that the input checks accept these ids too.
        suppress_all_data_checks=False,
        form_adjusted_meat_adjustments_explicitly=False,
        run_diagnostics=False,
    )

    with open(f"{output_dir}/analysis.pkl", "rb") as f:
        results = pickle.load(f)
    return (
        np.asarray(results["theta_est"], dtype=np.float64),
        np.asarray(results["adjusted_sandwich_var_estimate"], dtype=np.float64),
    )


def _fixture_subject_ids():
    with open(f"{FIXTURE_DIR}/study_df.pkl", "rb") as f:
        return sorted(pickle.load(f)["user_id"].unique())


@pytest.mark.parametrize(
    "label,make_mapper",
    [
        ("strings", lambda ids: {i: f"subj-{i:03d}" for i in ids}),
        # Above 2**31, so JAX's 32-bit default would truncate these.
        ("large_ints", lambda ids: {i: 1700000000000 + i for i in ids}),
    ],
)
def test_analyze_dataset_is_invariant_to_subject_id_relabelling(
    tmp_path, label, make_mapper
):
    subject_ids = _fixture_subject_ids()

    # Created explicitly: analyze_dataset requires output_dir to exist up front (it writes
    # every result there only at the end of the run) and does not create it.
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    relabelled_dir = tmp_path / label
    relabelled_dir.mkdir()

    baseline_theta, baseline_var = _run_with_remapped_subject_ids(baseline_dir, None)
    relabelled_theta, relabelled_var = _run_with_remapped_subject_ids(
        relabelled_dir, make_mapper(subject_ids)
    )

    # EXACT equality, not allclose: relabelling opaque keys changes nothing the estimator
    # computes, so any difference at all would mean the ids are leaking into the arithmetic.
    np.testing.assert_array_equal(relabelled_theta, baseline_theta)
    np.testing.assert_array_equal(relabelled_var, baseline_var)


def test_analyze_dataset_fails_fast_on_a_missing_output_dir(tmp_path):
    # require_output_dir_ready runs before theta estimation, so this must raise immediately
    # rather than after the whole analysis has been computed. Pytest gives no direct timing
    # assertion, so this pins the message; the ordering is pinned by the check being called
    # unconditionally at the top of analyze_dataset.
    with pytest.raises(AssertionError, match="does not exist"):
        _run_with_remapped_subject_ids(tmp_path / "never_created", None)
