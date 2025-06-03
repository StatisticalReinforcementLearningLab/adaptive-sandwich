import os
import pytest
from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.utils import assert_real_run_output_as_expected


def test_RL_center_0_inf_center_1_steep_3_incremental(
    run_local_pipeline,
):  # pylint: disable=redefined-outer-name
    run_local_pipeline(
        T="10",
        n="100",
        steepness="3.0",
        alg_state_feats="intercept,past_reward",
        action_centering_RL="0",
        recruit_n="20",
        theta_calculation_func_filename="functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_action_centering.py",
        inference_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_action_centering.py",
        alg_update_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_rl.py",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
    )

    # Find where “simulated_data” actually landed
    cwd = os.getcwd()
    sim_path = None
    for root, dirs, _ in os.walk(cwd):
        if "simulated_data" in dirs:
            sim_path = os.path.join(root, "simulated_data")
            break

    if sim_path is None:
        pytest.skip("run_local_pipeline did not create a simulated_data/ folder")

    print(f">>> Found simulated_data at: {sim_path!r}")
    print(">>> Listing everything inside simulated_data recursively:\n")
    for root, dirs, files in os.walk(sim_path):
        # Compute a relative path to make the printout shorter
        rel = os.path.relpath(root, sim_path)
        prefix = "" if rel == "." else rel + os.sep
        for d in dirs:
            print(f"{prefix}{d}/")
        for f in files:
            print(f"{prefix}{f}")

    assert_real_run_output_as_expected(
        test_file_path=__file__,
        relative_path_to_output_dir="../../../simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=10_n=100_recruitN=20_decisionsBtwnUpdates=1_steepness=3.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1",
    )
