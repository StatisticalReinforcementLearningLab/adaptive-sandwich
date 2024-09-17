import pickle

import pandas as pd
import numpy as np

from tests.integration_tests.fixtures import (  # pylint: disable=unused-import
    run_local_pipeline,
)
from tests.integration_tests.utils import get_abs_path


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
        env_seed_override="1726458459",
        alg_seed_override="1726463458",
    )

    # Load the observed and expected pickle files

    with open(
        get_abs_path(
            __file__,
            "../../../simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=10_n=100_recruitN=20_decisionsBtwnUpdates=1_steepness=3.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1/study_df.pkl",
        ),
        "rb",
    ) as observed_study_df_pickle, open(
        get_abs_path(
            __file__,
            "../../../simulated_data/synthetic_mode=delayed_1_dosage_alg=sigmoid_LS_T=10_n=100_recruitN=20_decisionsBtwnUpdates=1_steepness=3.0_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1/analysis.pkl",
        ),
        "rb",
    ) as observed_analysis_pickle, open(
        get_abs_path(__file__, "expected_study_df.pkl"),
        "rb",
    ) as expected_study_df_pickle, open(
        get_abs_path(__file__, "expected_analysis.pkl"),
        "rb",
    ) as expected_analysis_pickle:
        observed_study_df = pickle.load(observed_study_df_pickle)
        observed_analysis_dict = pickle.load(observed_analysis_pickle)
        # The expected df is generated from a time when these were set as Int64.
        # I don't remember why we had to change to float64; I believe it was
        # necessary on the analysis side.
        expected_study_df = pickle.load(expected_study_df_pickle).astype(
            {"policy_num": "float64", "action": "float64"}
        )
        expected_analysis_dict = pickle.load(expected_analysis_pickle)

    try:
        pd.testing.assert_frame_equal(observed_study_df, expected_study_df)
        np.testing.assert_allclose(
            observed_analysis_dict["theta_est"], expected_analysis_dict["theta_est"]
        )
        np.testing.assert_allclose(
            observed_analysis_dict["adaptive_sandwich_var_estimate"],
            expected_analysis_dict["adaptive_sandwich_var_estimate"],
        )
        np.testing.assert_allclose(
            observed_analysis_dict["classical_sandwich_var_estimate"],
            expected_analysis_dict["classical_sandwich_var_estimate"],
        )
    except:
        breakpoint()
