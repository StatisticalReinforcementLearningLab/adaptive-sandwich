import pickle

from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.utils import get_abs_path

from lifejacket.constants import DiagnosticClassifications
from lifejacket.diagnostics import DiagnosticReport


def test_RL_diagnostics_smoke(run_local_pipeline):  # pylint: disable=redefined-outer-name
    # A small/fast configuration -- this test is about exercising the new run_diagnostics=True
    # path end-to-end against the real simulator and estimator, not about a specific numeric
    # result (see tests/unit_tests/test_diagnostics.py for the closed-form correctness checks).
    run_local_pipeline(
        T="6",
        n="20",
        recruit_n="10",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
        run_diagnostics="1",
    )

    output_dir = get_abs_path(
        __file__,
        "../../simulators_and_runners/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sigmoid_LS_smooth_clip_T=6_n=20_recruitN=10_decisionsBtwnUpdates=1_steepness=0.5_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0_lambda=0.0_lowerclip=0.1_upperclip=0.9/exp=1",
    )

    with open(f"{output_dir}/diagnostic_report.pkl", "rb") as f:
        report = pickle.load(f)

    assert isinstance(report, DiagnosticReport)
    assert report.classification in (
        DiagnosticClassifications.LOCALLY_SUPPORTED,
        DiagnosticClassifications.FAILED,
        DiagnosticClassifications.INDETERMINATE,
    )
    # run_diagnostic_suite can never certify "supported" on its own (only
    # simulator_calibration.calibrate_and_classify can, after a held-out simulator pass).
    assert report.classification != DiagnosticClassifications.SUPPORTED
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
