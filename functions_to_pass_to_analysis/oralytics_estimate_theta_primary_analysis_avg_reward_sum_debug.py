import numpy as np


def oralytics_estimate_theta_primary_analysis_avg_reward_sum_debug(study_df):

    num_users = study_df["user_idx"].nunique()
    in_study_bool = study_df["in_study_indicator"] == 1
    return np.array([study_df.loc[in_study_bool].oscb.sum() / num_users])
