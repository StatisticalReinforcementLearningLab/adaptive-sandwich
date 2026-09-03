import os
import pickle
from unittest import mock

import numpy as np
import pytest
from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.utils import get_abs_path

from lifejacket import post_deployment_analysis
from lifejacket.constants import (
    CheckStatuses,
    DiagnosticClassifications,
    DiagnosticVerdicts,
)
from lifejacket.diagnostics import DiagnosticReport
from lifejacket.helper_functions import load_function_from_same_named_file
from lifejacket.post_deployment_analysis import analyze_dataset


def test_RL_diagnostics_smoke(run_local_pipeline):  # pylint: disable=redefined-outer-name
    # A small/fast configuration -- this test is about exercising the new run_diagnostics=True
    # path end-to-end against the real simulator and estimator, not about a specific numeric
    # result (see tests/unit_tests/test_diagnostics.py for the closed-form correctness checks).
    # fail_on_flagged_diagnostics=0 because this config's verdict is allowed to be flagged
    # (the classification assertion below accepts failed) and the CLI would then exit 3 --
    # this test is the documented read-the-report-from-disk consumer of that flag.
    result = run_local_pipeline(
        T="6",
        n="20",
        recruit_n="10",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
        run_diagnostics="1",
        fail_on_flagged_diagnostics="0",
    )

    # The end-of-run diagnostic summary and verdict must be printed whenever the suite runs.
    stdout_text = str(result)
    assert "DIAGNOSTIC SUMMARY" in stdout_text
    assert "VERDICT:" in stdout_text

    output_dir = get_abs_path(
        __file__,
        "../../simulators_and_runners/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sigmoid_LS_smooth_clip_T=6_n=20_recruitN=10_decisionsBtwnUpdates=1_steepness=0.5_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0_lambda=0.0_lowerclip=0.1_upperclip=0.9/exp=1",
    )

    with open(f"{output_dir}/diagnostic_report.pkl", "rb") as f:
        report = pickle.load(f)

    assert isinstance(report, DiagnosticReport)
    # The input-check rows must carry their measurements, not just statuses. Pinned here, at
    # the only place that exercises analyze_dataset's report-building end to end, because two
    # bugs already slipped past the unit suite in exactly this spot on 2026-09-02: a helper
    # placed inside analyze's click decorator chain (which BROKE every direct analyze_dataset
    # call while every unit test stayed green), and a late None re-initialization that
    # silently discarded the first wave's measurements before this report was built.
    reconstruction_row = report.input_check_results[
        "action_probabilities_reconstructed"
    ]
    (reconstruction_criterion,) = reconstruction_row.criteria
    assert "agree to within 1e-06" in reconstruction_criterion.description
    assert "max difference" in reconstruction_criterion.value
    assert reconstruction_criterion.ok is True
    assert "first_wave_input_checks" in report.input_check_results
    sum_to_zero_row = report.input_check_results["estimating_functions_sum_to_zero"]
    assert any(
        "solve their estimating equations" in criterion.description
        and "worst residual" in criterion.value
        and " SE at " in criterion.value
        for criterion in sum_to_zero_row.criteria
    )
    assert report.classification in (
        DiagnosticClassifications.LOCALLY_SUPPORTED,
        DiagnosticClassifications.FAILED,
        DiagnosticClassifications.INDETERMINATE,
    )
    # run_diagnostic_suite can never certify "supported" on its own (only
    assert "root_and_implementation" in report.check_results

    # The existing analysis.pkl/debug_pieces.pkl outputs must be unaffected by run_diagnostics.
    with open(f"{output_dir}/analysis.pkl", "rb") as f:
        analysis_dict = pickle.load(f)
    assert "theta_est" in analysis_dict


def test_RL_diagnostics_multiplier_bootstrap_smoke(run_local_pipeline, tmp_path):  # pylint: disable=redefined-outer-name
    # End-to-end pass with the frozen-score multiplier bootstrap forced on via a
    # diagnostic_config_pickle -- the exact wiring the ADS-142 bootstrap-validation cluster runs
    # use (docs/adr/0002). Few draws: this guards the plumbing and the check's output shape
    # against the real simulator/estimator, not its statistical power.
    from lifejacket.diagnostics import DiagnosticConfig

    config_path = tmp_path / "bootstrap_config.pkl"
    with open(config_path, "wb") as f:
        pickle.dump(
            DiagnosticConfig(multiplier_bootstrap="always", num_bootstrap_draws=25), f
        )

    run_local_pipeline(
        T="6",
        n="20",
        recruit_n="10",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
        run_diagnostics="1",
        diagnostic_config_pickle=str(config_path),
    )

    output_dir = get_abs_path(
        __file__,
        "../../simulators_and_runners/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sigmoid_LS_smooth_clip_T=6_n=20_recruitN=10_decisionsBtwnUpdates=1_steepness=0.5_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0_lambda=0.0_lowerclip=0.1_upperclip=0.9/exp=1",
    )
    with open(f"{output_dir}/diagnostic_report.pkl", "rb") as f:
        report = pickle.load(f)

    boot = report.check_results["multiplier_bootstrap"]
    assert boot.metrics["num_trials"] == 50  # 25 draws, paired +/-
    assert set(boot.metrics["se_ratio_by_target"]) == set(report.target_labels)
    assert boot.metrics["multiplier_distribution"] == "rademacher"
    # The verdict itself is data-dependent; what must hold is that it is one of the defined
    # statuses and the band/ratio metrics are populated for downstream aggregation.
    assert boot.status in ("passed", "warning", "failed", "indeterminate")
    assert len(boot.metrics["se_ratio_null_band"]) == 2


def test_RL_percentile_bootstrap_smoke(run_local_pipeline):  # pylint: disable=redefined-outer-name
    # End-to-end pass of the refit percentile bootstrap (docs/adr/0003) against the real
    # simulator and estimator: the RN-weighted joint system is re-solved under Poisson
    # multiplicities and the percentile interval lands in analysis.pkl. Few draws -- this
    # guards the wiring and output shapes; the statistical validation is the ADR's own
    # externally-run coverage grids and the independent-reference comparison.
    run_local_pipeline(
        T="6",
        n="20",
        recruit_n="10",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
        run_diagnostics="0",
        percentile_bootstrap_draws="20",
        percentile_bootstrap_seed="11",
    )

    output_dir = get_abs_path(
        __file__,
        "../../simulators_and_runners/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sigmoid_LS_smooth_clip_T=6_n=20_recruitN=10_decisionsBtwnUpdates=1_steepness=0.5_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0_lambda=0.0_lowerclip=0.1_upperclip=0.9/exp=1",
    )
    with open(f"{output_dir}/analysis.pkl", "rb") as f:
        analysis = pickle.load(f)

    theta_est = np.asarray(analysis["theta_est"]).reshape(-1)
    ci = np.asarray(analysis["percentile_bootstrap_ci"])
    assert ci.shape == (theta_est.shape[0], 2)
    assert analysis["bootstrap_num_draws"] == 20
    # At n=20 with recruit_n=10, a Poisson draw occasionally zeroes out most of the first
    # recruitment wave, leaving that update's block genuinely unidentified -- those draws
    # correctly fail as singular_jacobian and are dropped and counted (docs/adr/0003's
    # degenerate-systems guardrail). What must hold: a large majority of draws converge, and
    # every failure is of the degenerate kind rather than a solver bug.
    #
    # The allowance is 4, not a round fraction of the draws: the multiplicities come from a
    # fixed numpy seed, so WHICH draws are degenerate is deterministic, and 3 of 20 fail here.
    # The one spare draw is headroom for a marginal draw flipping on cross-platform float32
    # differences (the same noise tests/utils.py documents for CI-vs-macOS), not slack for a
    # convergence regression -- 5 of 20 was loose enough to absorb one silently.
    assert analysis["bootstrap_num_failed_draws"] <= 4
    # And the taxonomy, not just the count: _newton_refit distinguishes a genuinely
    # unidentified draw (singular_jacobian) from nonfinite_residual / nonfinite_jacobian /
    # nonfinite_iterate / max_iterations, all of which are solver or wiring failures rather
    # than degenerate data and none of which this allowance covers.
    assert set(analysis["bootstrap_failure_reasons"]) <= {"singular_jacobian"}
    assert (
        sum(analysis["bootstrap_failure_reasons"].values())
        == analysis["bootstrap_num_failed_draws"]
    )
    # The bootstrap block is best-effort and records its own failures rather than aborting the
    # analysis, so a completed run has to be distinguished from a recorded failure explicitly.
    assert analysis["bootstrap_error"] is None
    assert np.all(np.isfinite(ci))
    assert np.all(ci[:, 0] < ci[:, 1])
    assert np.all(ci[:, 0] <= theta_est) and np.all(theta_est <= ci[:, 1])

    with open(f"{output_dir}/debug_pieces.pkl", "rb") as f:
        debug_pieces = pickle.load(f)
    draws = np.asarray(debug_pieces["percentile_bootstrap_theta_draws"])
    assert draws.shape == (
        20 - analysis["bootstrap_num_failed_draws"],
        theta_est.shape[0],
    )
    assert np.all(np.isfinite(draws))


# ---------------------------------------------------------------------------
# analyze_dataset end-to-end on a MASK-AWARE study (alg_update_func_args_mask_index >= 0), the
# shape-bucket-consolidation configuration real deployments use. The simulator pipeline above
# cannot produce one -- run_local_synthetic.sh has no mask wiring -- so these drive
# analyze_dataset directly on tests/benchmarks' own real, genuinely-ragged fixture data with
# the mask-aware alg_update_func the combine_updates_into_one_vmap benchmark already uses.
# Everything below runs the real estimator; nothing here is a simulator run.
# ---------------------------------------------------------------------------

BENCHMARK_FUNCTIONS_DIR = get_abs_path(
    __file__, "../../simulators_and_runners/functions_to_pass_to_analysis"
)
BENCHMARK_FIXTURE_DIR = get_abs_path(__file__, "../../benchmarks/fixtures/small")
# RL_least_squares_loss_regularized_masked's argument positions: the per-decision-time history
# args are ragged, and the validity mask is appended as the 11th (index 10) argument.
MASKED_RAGGED_INDICES = (1, 2, 3, 4, 5, 6)
MASKED_MASK_INDEX = 10


def _run_masked_analyze_dataset(output_dir, **overrides):
    with open(f"{BENCHMARK_FIXTURE_DIR}/study_df.pkl", "rb") as f:
        study_df = pickle.load(f)
    with open(f"{BENCHMARK_FIXTURE_DIR}/pi_args.pkl", "rb") as f:
        action_prob_func_args = pickle.load(f)
    with open(f"{BENCHMARK_FIXTURE_DIR}/rl_update_args.pkl", "rb") as f:
        alg_update_func_args = pickle.load(f)

    kwargs = {
        "output_dir": output_dir,
        "analysis_df": study_df,
        "action_prob_func": load_function_from_same_named_file(
            f"{BENCHMARK_FUNCTIONS_DIR}/synthetic_get_action_1_prob_generalized_logistic.py"
        ),
        "action_prob_func_args": action_prob_func_args,
        "action_prob_func_args_beta_index": 0,
        "alg_update_func": load_function_from_same_named_file(
            f"{BENCHMARK_FUNCTIONS_DIR}/RL_least_squares_loss_regularized_masked.py"
        ),
        "alg_update_func_type": "loss",
        "alg_update_func_args": alg_update_func_args,
        "alg_update_func_args_beta_index": 0,
        "alg_update_func_args_action_prob_index": 5,
        "alg_update_func_args_action_prob_times_index": 6,
        "alg_update_func_args_previous_betas_index": -1,
        "alg_update_func_args_mask_index": MASKED_MASK_INDEX,
        "alg_update_func_args_ragged_indices": MASKED_RAGGED_INDICES,
        "inference_func": load_function_from_same_named_file(
            f"{BENCHMARK_FUNCTIONS_DIR}/synthetic_get_least_squares_loss_inference_no_action_centering.py"
        ),
        "inference_func_type": "loss",
        "inference_func_args_theta_index": 0,
        "theta_calculation_func": load_function_from_same_named_file(
            f"{BENCHMARK_FUNCTIONS_DIR}/synthetic_estimate_theta_least_squares_no_action_centering.py"
        ),
        "active_col_name": "in_study",
        "action_col_name": "action",
        "policy_num_col_name": "policy_num",
        "calendar_t_col_name": "calendar_t",
        "subject_id_col_name": "user_id",
        "action_prob_col_name": "action1prob",
        "reward_col_name": "reward",
        "suppress_interactive_data_checks": True,
        "suppress_all_data_checks": True,
        "form_adjusted_meat_adjustments_explicitly": False,
        "run_diagnostics": False,
    }
    kwargs.update(overrides)
    return analyze_dataset(**kwargs)


def _load_pickle(output_dir, name):
    with open(f"{output_dir}/{name}", "rb") as f:
        return pickle.load(f)


@pytest.mark.parametrize("suppress_all_data_checks", [True, False])
def test_diagnostic_suite_runs_on_a_mask_aware_study(
    tmp_path, suppress_all_data_checks
):
    """
    Regression: the diagnostic suite silently produced NO report at all for every study
    analyzed with alg_update_func_args_mask_index >= 0. The suite's g_tilde re-evaluated the
    estimating-function stack WITHOUT the mask/ragged indices, so the mask-aware
    alg_update_func was called one argument short inside jax.vmap; the resulting TypeError hit
    the suite's blanket best-effort except and was logged as a warning.

    That is why this asserts the REPORT EXISTS rather than that nothing raised: on the buggy
    code analyze_dataset returned perfectly normally and wrote analysis.pkl and
    debug_pieces.pkl -- only diagnostic_report.pkl was missing, which is exactly what made the
    defect survive.

    Parametrized over suppress_all_data_checks because the second finding fixed here lives on
    the same path: with checks suppressed the suite must not re-run (and hard-fail on) the
    reconstruction input check the caller explicitly turned off, and a suppressed check must
    not be indistinguishable from a passing one in the report.
    """
    _run_masked_analyze_dataset(
        tmp_path,
        run_diagnostics=True,
        suppress_all_data_checks=suppress_all_data_checks,
    )

    report = _load_pickle(tmp_path, "diagnostic_report.pkl")
    assert isinstance(report, DiagnosticReport)
    assert report.classification != DiagnosticClassifications.FAILED
    # The suite actually ran its statistical checks rather than short-circuiting on a failed
    # input check (which forces FAILED and skips every one of them).
    assert "root_and_implementation" in report.check_results
    assert (
        report.check_results["root_and_implementation"].status == CheckStatuses.PASSED
    )
    assert len(report.check_results) >= 5

    reconstruction_check = report.input_check_results[
        "action_probabilities_reconstructed"
    ]
    if suppress_all_data_checks:
        # Recorded as not-run, not omitted and not silently "passed".
        assert reconstruction_check.status == CheckStatuses.INDETERMINATE
        assert "suppress_all_data_checks" in reconstruction_check.message
        # And it caps the verdict: the input rows are passed INTO the suite before the
        # verdict is derived, so unvalidated inputs can never read CERTIFIED (a Copilot
        # review finding -- they used to be grafted onto the finished report, leaving a
        # quiet suppressed run certified with exit status 0).
        assert report.verdict == DiagnosticVerdicts.NOT_CERTIFIED
    else:
        assert reconstruction_check.status == CheckStatuses.PASSED
        # An unsuppressed run may still be honestly uncertified for statistical reasons;
        # what it must never carry is a not-run input row.
        assert all(
            row.status != CheckStatuses.INDETERMINATE
            for row in report.input_check_results.values()
        )


def test_percentile_bootstrap_failure_does_not_destroy_the_analysis(tmp_path):
    """
    The bootstrap runs after the adjusted sandwich is computed but BEFORE anything is written,
    so an unguarded failure there threw away a completed analysis over an added interval. The
    guard's contract is that ANY exception escaping the bootstrap block is recorded and the
    surrounding analysis still lands on disk, so the failure is injected at the block's entry
    point directly: patch refit_percentile_bootstrap to raise (on post_deployment_analysis,
    per the NOTE above). This test used to reach the same guard by passing
    percentile_bootstrap_seed=-1 through the library entry point, but
    require_valid_percentile_bootstrap_settings now rejects that upfront, before the sandwich
    is ever computed -- and any replacement sneak-a-bad-value route would be one validation
    tightening away from breaking the same way.
    """
    with mock.patch.object(
        post_deployment_analysis,
        "refit_percentile_bootstrap",
        side_effect=ValueError("injected bootstrap failure"),
    ):
        # 12, not a smaller number: require_valid_percentile_bootstrap_settings rejects 1-9
        # up front (they cannot meet the quantile step's survivor minimum), and this test needs
        # to get PAST validation to the mocked refit.
        result = _run_masked_analyze_dataset(tmp_path, percentile_bootstrap_draws=12)

    assert "theta_est" in result
    analysis = _load_pickle(tmp_path, "analysis.pkl")
    assert "theta_est" in analysis
    assert np.all(np.isfinite(np.asarray(analysis["theta_est"])))
    assert np.all(np.isfinite(np.asarray(analysis["adjusted_sandwich_var_estimate"])))
    _load_pickle(tmp_path, "debug_pieces.pkl")

    # The failure is RECORDED, not silently swallowed: a NaN interval alone would be
    # indistinguishable from a bootstrap whose draws all failed to converge.
    assert analysis["bootstrap_error"] is not None
    assert "ValueError" in analysis["bootstrap_error"]
    assert np.all(np.isnan(np.asarray(analysis["percentile_bootstrap_ci"])))
    assert analysis["bootstrap_num_draws"] == 12
    assert analysis["bootstrap_num_failed_draws"] == 12


# ---------------------------------------------------------------------------
# Performance-regression guards, expressed as CALL COUNTS rather than wall-clock.
#
# tests/benchmarks measures timings but asserts nothing about them, so a run that
# got 10x slower still passes there. Timing thresholds are not the fix: on a shared
# CI runner the noise exceeds the regressions worth catching (this branch measured
# the same change at 1.10x and 0.77x under load). Counting invocations of an
# expensive operation is deterministic, machine-independent, and cannot flake.
#
# What these pin is a design CONTRACT, not an incidental number: the structural
# precompute (the O(N*T) per-subject/per-update bucket construction) is built ONCE
# per feature, not once per evaluation. The regression they exist to catch is real
# and was shipped: the diagnostic suite's g_tilde closure rebuilt it on every call,
# ~127 times per analysis, which is what made a 5-minute analysis take 3.6 hours.
# Deltas rather than absolute counts, so a legitimate refactor of the base path
# does not have to touch these.
#
# NOTE for anyone extending these: patch the name on post_deployment_analysis, not
# on batched_weighted_estimating_function_stack. The former imports these builders
# by name at module load, so patching the definition site intercepts nothing and
# the test silently measures a call count of zero.
# ---------------------------------------------------------------------------

_MAX_EXTRA_PRECOMPUTE_BUILDS = 2


def _count_update_layer_builds(output_dir, **overrides):
    # analyze_dataset writes its pickles into output_dir but does not create it.
    os.makedirs(output_dir, exist_ok=True)
    with mock.patch.object(
        post_deployment_analysis,
        "build_update_layer_precompute",
        wraps=post_deployment_analysis.build_update_layer_precompute,
    ) as builder:
        _run_masked_analyze_dataset(output_dir, **overrides)
    return builder.call_count


def test_diagnostic_suite_does_not_rebuild_the_precompute_per_evaluation(tmp_path):
    # The suite evaluates g_tilde ~127 times at default settings. Turning it on must cost
    # a CONSTANT number of extra precompute builds (measured: exactly one), not one per
    # evaluation -- the difference between seconds and hours.
    without = _count_update_layer_builds(
        str(tmp_path / "without"), run_diagnostics=False
    )
    with_diagnostics = _count_update_layer_builds(
        str(tmp_path / "with"), run_diagnostics=True
    )
    assert with_diagnostics - without <= _MAX_EXTRA_PRECOMPUTE_BUILDS, (
        f"Running diagnostics added {with_diagnostics - without} structural precompute "
        f"builds ({without} -> {with_diagnostics}). It must add a constant number, not one "
        "per g_tilde evaluation; see _diagnostics_g_tilde's precomputed_layers wiring."
    )


def test_refit_bootstrap_precompute_does_not_scale_with_draw_count(tmp_path):
    # Same contract on the other re-solving consumer: the interval re-solves the full
    # system once per draw, and each of those must reuse one precompute rather than
    # building its own.
    no_draws = _count_update_layer_builds(
        str(tmp_path / "none"), run_diagnostics=False, percentile_bootstrap_draws=0
    )
    with_draws = _count_update_layer_builds(
        str(tmp_path / "twelve"), run_diagnostics=False, percentile_bootstrap_draws=12
    )
    assert with_draws - no_draws <= _MAX_EXTRA_PRECOMPUTE_BUILDS, (
        f"12 bootstrap draws added {with_draws - no_draws} structural precompute builds "
        f"({no_draws} -> {with_draws}); the count must not scale with the draw count."
    )
