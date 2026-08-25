"""
Lightweight, always-runnable performance/regression benchmark for
analyze_dataset. This is Step 0 of the performance work described in
docs/adr/0001-adaptive-sandwich-performance-plan.md: before optimizing
anything, we need (a) a real, empirical phase-by-phase timing breakdown
instead of a plausible-but-unverified hypothesis about where the time goes,
and (b) a fast, numerically-tight regression gate so every later change to
this hot path can be checked for both a speed win and "did it change the
answer."

Unlike tests/unit_tests and tests/integration_tests, this suite is not about
correctness of a specific scenario -- it runs analyze_dataset on one small,
committed synthetic fixture and:
  1. Prints a per-phase wall-clock timing breakdown, using the
     log_phase_duration instrumentation added to lifejacket/helper_functions.py
     and wired into lifejacket/post_deployment_analysis.py.
  2. Asserts the returned theta_est / adjusted_sandwich_var_estimate /
     classical_sandwich_var_estimate match a committed golden fixture at a
     tight tolerance (rtol=1e-6) -- any optimization that changes these
     beyond ordinary floating-point noise should fail this test.

Run with:
    python -m pytest tests/benchmarks -v -s
to see the phase timing breakdown printed for each parametrization. The
"medium" scale (n=100, T=10, matching tests/integration_tests) is marked
`slow` -- exclude it for a quick check with `-m "not slow"`, or run only it
with `-m slow`. "small" (n=20, T=6) is fast enough to always run.

To (re)generate a fixture and its golden output (e.g. after intentionally and
independently verifying a change to the numerical results, or if the
synthetic simulation parameters change), see generate_fixture.py in this
directory.
"""

from __future__ import annotations

import logging
import pickle
import re
import time

import numpy as np
import pytest
from tests.utils import get_abs_path

from lifejacket.helper_functions import load_function_from_same_named_file
from lifejacket.post_deployment_analysis import analyze_dataset

FUNCTIONS_DIR = get_abs_path(
    __file__, "../simulators_and_runners/functions_to_pass_to_analysis"
)

PHASE_LOG_PATTERN = re.compile(
    r"^Phase '(?P<phase>.+)' took (?P<seconds>[\d.]+) seconds\.$"
)


def _fixture_dir(scale):
    return get_abs_path(__file__, f"fixtures/{scale}")


def _load_fixture_inputs(scale):
    fixture_dir = _fixture_dir(scale)
    with open(f"{fixture_dir}/study_df.pkl", "rb") as f:
        study_df = pickle.load(f)
    with open(f"{fixture_dir}/pi_args.pkl", "rb") as f:
        action_prob_func_args = pickle.load(f)
    with open(f"{fixture_dir}/rl_update_args.pkl", "rb") as f:
        alg_update_func_args = pickle.load(f)
    return study_df, action_prob_func_args, alg_update_func_args


def _run_analyze_dataset(tmp_path, scale, suppress_all_data_checks, caplog):
    study_df, action_prob_func_args, alg_update_func_args = _load_fixture_inputs(scale)

    action_prob_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_get_action_1_prob_generalized_logistic.py"
    )
    alg_update_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/RL_least_squares_loss_regularized.py"
    )
    inference_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_get_least_squares_loss_inference_no_action_centering.py"
    )
    theta_calculation_func = load_function_from_same_named_file(
        f"{FUNCTIONS_DIR}/synthetic_estimate_theta_least_squares_no_action_centering.py"
    )

    caplog.clear()
    with caplog.at_level(logging.INFO):
        start = time.perf_counter()
        result = analyze_dataset(
            output_dir=tmp_path,
            analysis_df=study_df,
            action_prob_func=action_prob_func,
            action_prob_func_args=action_prob_func_args,
            action_prob_func_args_beta_index=0,
            alg_update_func=alg_update_func,
            alg_update_func_type="loss",
            alg_update_func_args=alg_update_func_args,
            alg_update_func_args_beta_index=0,
            alg_update_func_args_action_prob_index=5,
            alg_update_func_args_action_prob_times_index=6,
            alg_update_func_args_previous_betas_index=-1,
            inference_func=inference_func,
            inference_func_type="loss",
            inference_func_args_theta_index=0,
            theta_calculation_func=theta_calculation_func,
            active_col_name="in_study",
            action_col_name="action",
            policy_num_col_name="policy_num",
            calendar_t_col_name="calendar_t",
            subject_id_col_name="user_id",
            action_prob_col_name="action1prob",
            reward_col_name="reward",
            suppress_interactive_data_checks=True,
            suppress_all_data_checks=suppress_all_data_checks,
            small_sample_correction="none",
            collect_data_for_blowup_supervised_learning=False,
            form_adjusted_meat_adjustments_explicitly=False,
            stabilize_joint_bread=True,
        )
        total_seconds = time.perf_counter() - start

    phase_timings = [
        (match.group("phase"), float(match.group("seconds")))
        for record in caplog.records
        for match in [PHASE_LOG_PATTERN.match(record.message)]
        if match
    ]
    return result, total_seconds, phase_timings


def _print_timing_report(label, total_seconds, phase_timings):
    print(f"\n--- analyze_dataset benchmark: {label} ---")
    print(f"Total wall-clock time: {total_seconds:.3f}s")
    for phase, seconds in sorted(phase_timings, key=lambda p: -p[1]):
        pct = 100 * seconds / total_seconds if total_seconds else float("nan")
        print(f"  {seconds:8.3f}s  ({pct:5.1f}%)  {phase}")


@pytest.mark.parametrize(
    "scale",
    ["small", pytest.param("medium", marks=pytest.mark.slow)],
)
@pytest.mark.parametrize("suppress_all_data_checks", [True, False])
def test_analyze_dataset_benchmark(tmp_path, caplog, scale, suppress_all_data_checks):
    """
    Times analyze_dataset on a committed synthetic fixture, both with and
    without data-consistency checks (input_checks.py runs by default in real
    usage -- see the ADR -- so both cases matter), and checks the numerical
    result against a committed golden fixture.

    "small" (n=20, T=6) is fast to always run. "medium" (n=100, T=10) matches
    tests/integration_tests' scale and is what actually shows whether a
    change to something that scales with subjects x decision_times (the
    input_checks.py functions, the jax.jacrev hot path) made a real
    difference -- "small" alone can understate or overstate a phase's
    relative cost.

    This is not a substitute for profiling at production scale (see the ADR's
    recommended cProfile/py-spy steps), but it gives a fast, always-available
    signal for whether a change made this hot path faster or slower, and
    whether it changed the answer.
    """
    result, total_seconds, phase_timings = _run_analyze_dataset(
        tmp_path, scale, suppress_all_data_checks, caplog
    )
    _print_timing_report(
        f"scale={scale}, suppress_all_data_checks={suppress_all_data_checks}",
        total_seconds,
        phase_timings,
    )

    assert phase_timings, (
        "No 'Phase ... took ... seconds' log lines were captured. Either "
        "log_phase_duration logging broke, or caplog isn't capturing at the "
        "right level/logger -- fix this before trusting the timing report."
    )

    with open(f"{_fixture_dir(scale)}/golden_analysis.pkl", "rb") as f:
        golden = pickle.load(f)

    np.testing.assert_allclose(
        np.asarray(result["theta_est"]),
        np.asarray(golden["theta_est"]),
        rtol=1e-6,
        err_msg="theta_est diverged from the golden fixture.",
    )
    np.testing.assert_allclose(
        np.asarray(result["adjusted_sandwich_var_estimate"]),
        np.asarray(golden["adjusted_sandwich_var_estimate"]),
        rtol=1e-6,
        err_msg="adjusted_sandwich_var_estimate diverged from the golden fixture.",
    )
    np.testing.assert_allclose(
        np.asarray(result["classical_sandwich_var_estimate"]),
        np.asarray(golden["classical_sandwich_var_estimate"]),
        rtol=1e-6,
        err_msg="classical_sandwich_var_estimate diverged from the golden fixture.",
    )
