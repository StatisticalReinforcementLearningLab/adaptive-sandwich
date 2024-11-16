from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.integration_tests.utils import assert_real_run_output_as_expected


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
        theta_calculation_func_filename="functions_to_pass_to_analysis/estimate_theta_least_squares_action_centering.py",
        inference_loss_func_filename="functions_to_pass_to_analysis/get_least_squares_loss_inference_action_centering.py",
        rl_update_func_filename="functions_to_pass_to_analysis/get_least_squares_estimating_function_rl.py",
        rl_update_func_type="estimating",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        suppress_interactive_data_checks="1",
    )

    assert_real_run_output_as_expected(
        test_file_path=__file__,
        relative_path_to_output_dir="../../../simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=10_n=100_recruitN=20_decisionsBtwnUpdates=1_steepness=3.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1",
    )
