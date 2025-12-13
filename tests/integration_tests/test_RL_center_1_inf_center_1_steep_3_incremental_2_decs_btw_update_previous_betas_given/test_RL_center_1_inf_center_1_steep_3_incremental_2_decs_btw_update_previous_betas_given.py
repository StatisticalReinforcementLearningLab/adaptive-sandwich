from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.utils import assert_real_run_output_as_expected


def test_RL_center_1_inf_center_1_steep_3_incremental_previous_betas_given(
    run_local_pipeline,
):  # pylint: disable=redefined-outer-name
    run_local_pipeline(
        T="10",
        n="100",
        steepness="3.0",
        alg_state_feats="intercept,past_reward",
        RL_alg="sigmoid_LS_hard_clip",
        action_centering_RL="1",
        recruit_n="20",
        theta_calculation_func_filename="functions_to_pass_to_analysis/synthetic_estimate_theta_least_squares_action_centering.py",
        inference_func_filename="functions_to_pass_to_analysis/synthetic_get_least_squares_loss_inference_action_centering.py",
        alg_update_func_filename="functions_to_pass_to_analysis/RL_least_squares_loss_regularized_previous_betas_as_args_hard_clipping.py",
        action_prob_func_filename="functions_to_pass_to_analysis/synthetic_get_action_1_prob_pure.py",
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
        decisions_between_updates="2",
        suppress_interactive_data_checks="1",
        collect_args_to_reconstruct_action_probs="1",
        alg_update_func_args_previous_betas_index="7",
        alg_update_func_args_action_prob_index="-1",
        alg_update_func_args_action_prob_times_index="-1",
        lclip="0.1",
        uclip="0.9",
        lambda_="0.0",
        err_corr="time_corr",
        update_cadence_offset="0",
        min_update_time="0",
        recruit_t="1",
    )

    assert_real_run_output_as_expected(
        test_file_path=__file__,
        relative_path_to_output_dir="../../../simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sigmoid_LS_hard_clip_T=10_n=100_recruitN=20_decisionsBtwnUpdates=2_steepness=3.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=1_lambda=0.0_lowerclip=0.1_upperclip=0.9/exp=1",
    )
