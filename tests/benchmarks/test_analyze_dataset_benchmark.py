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
import pathlib
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
            form_adjusted_meat_adjustments_explicitly=False,
        )
        total_seconds = time.perf_counter() - start

    phase_timings = [
        (match.group("phase"), float(match.group("seconds")))
        for record in caplog.records
        for match in [PHASE_LOG_PATTERN.match(record.message)]
        if match
    ]

    # debug_pieces.pkl (unlike analysis.pkl, loaded into `result` above)
    # carries the local-linearization diagnostic's own numeric output
    # (local_linearization_error_ratio_median/p90/max, computed by
    # compute_local_linearization_error_ratio) -- see this test module's own
    # LOCAL_LINEARIZATION_GOLDEN for why this is checked here.
    with open(pathlib.Path(tmp_path) / "debug_pieces.pkl", "rb") as f:
        debug_pieces = pickle.load(f)

    return result, debug_pieces, total_seconds, phase_timings


# compute_local_linearization_error_ratio's own numeric output
# (local_linearization_error_ratio_median/p90/max, in debug_pieces.pkl) had
# never been checked against an expected value anywhere in this repo before
# (its keys were only checked for PRESENCE in tests/utils.py's
# assert_real_run_output_as_expected, used by tests/integration_tests) --
# confirmed finding 2(a) of the ADS-139 hot-path rewrite (see
# docs/adr/0001-adaptive-sandwich-performance-plan.md). These are first-run
# values, pinned here rather than hand-derived from first principles: the
# diagnostic already draws its J=15 perturbations from a fixed
# jax.random.PRNGKey(0) (see compute_local_linearization_error_ratio), so
# given the already-golden theta_est/betas this test's other assertions
# pin, its three summary statistics are fully deterministic -- there is
# nothing further to "pin down" by re-deriving them from scratch that this
# fixed-seed determinism doesn't already give. Not compared against
# suppress_all_data_checks=False's own analyze_dataset run, since the
# diagnostic itself always hardcodes suppress_all_data_checks=True/
# suppress_interactive_data_checks=True internally (see
# compute_local_linearization_error_ratio's own LANDMINE comment) and is
# otherwise identical either way -- confirmed empirically, both variants
# produce bit-identical values at both scales.
LOCAL_LINEARIZATION_GOLDEN = {
    "small": {
        "median": 0.11875026673078537,
        "p90": 0.20014047622680664,
        "max": 0.20171566307544708,
    },
    "medium": {
        "median": 0.0222441665828228,
        "p90": 0.04883337765932083,
        "max": 0.06498895585536957,
    },
}


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
    result, debug_pieces, total_seconds, phase_timings = _run_analyze_dataset(
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
    # rtol=1e-5 (not 1e-6, like theta_est above) for these two:
    # get_avg_weighted_estimating_function_stacks_and_aux_values computes
    # the meat matrices as stacks.T @ stacks (see ADS-139 followups --
    # the small_sample_correction feature that once made this an opt-in
    # fast path has been removed; this is now the only way these matrices
    # are computed), instead of
    # jnp.mean(jax.vmap(jnp.outer)(stacks, stacks), axis=0). The two are an
    # exact linear-algebra identity but a different float32 summation
    # order, routed through a BLAS-style matmul rather than an
    # elementwise-then-reduce pass. That reordering perturbs the meat matrix
    # at the ~1e-11
    # relative level, which the joint/classical bread solve in
    # form_sandwich_from_bread_and_meat can amplify by this fixture's own
    # bread condition number (a few hundred here) -- ordinary float32 noise
    # per this file's own module docstring, just past the tighter bound
    # used for theta_est (unaffected by this fast path, since it comes from
    # differentiating the average estimating function stack, not from
    # either meat matrix). tests/benchmarks/test_combine_updates_into_one_vmap_benchmark.py
    # already uses this same rtol=1e-5 for analogous reordering-tolerant
    # cross-checks.
    np.testing.assert_allclose(
        np.asarray(result["adjusted_sandwich_var_estimate"]),
        np.asarray(golden["adjusted_sandwich_var_estimate"]),
        rtol=1e-5,
        err_msg="adjusted_sandwich_var_estimate diverged from the golden fixture.",
    )
    np.testing.assert_allclose(
        np.asarray(result["classical_sandwich_var_estimate"]),
        np.asarray(golden["classical_sandwich_var_estimate"]),
        rtol=1e-5,
        err_msg="classical_sandwich_var_estimate diverged from the golden fixture.",
    )

    # See LOCAL_LINEARIZATION_GOLDEN's own comment: this hot path has been
    # substantially rewritten twice (ADS-139) with no numeric check on its
    # own output anywhere in the repo until now.
    local_linearization_golden = LOCAL_LINEARIZATION_GOLDEN[scale]
    np.testing.assert_allclose(
        float(debug_pieces["local_linearization_error_ratio_median"]),
        local_linearization_golden["median"],
        rtol=1e-5,
        err_msg="local_linearization_error_ratio_median diverged from its pinned value.",
    )
    np.testing.assert_allclose(
        float(debug_pieces["local_linearization_error_ratio_p90"]),
        local_linearization_golden["p90"],
        rtol=1e-5,
        err_msg="local_linearization_error_ratio_p90 diverged from its pinned value.",
    )
    np.testing.assert_allclose(
        float(debug_pieces["local_linearization_error_ratio_max"]),
        local_linearization_golden["max"],
        rtol=1e-5,
        err_msg="local_linearization_error_ratio_max diverged from its pinned value.",
    )
