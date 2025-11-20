import numpy as np


def estimate_theta_avg_reward_sum(study_df):

    num_users = study_df["user_id"].nunique()
    # Sort of assumes no incremental recruitment, though it doesn't really
    # matter
    num_decision_times = study_df["calendar_t"].nunique()
    in_study_bool = study_df["in_study"] == 1

    return np.array(
        [study_df.loc[in_study_bool].reward.sum() / num_decision_times / num_users]
    )



