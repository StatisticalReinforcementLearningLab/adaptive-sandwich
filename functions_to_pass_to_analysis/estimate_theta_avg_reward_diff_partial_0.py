import numpy as np
from sklearn.linear_model import LinearRegression


###### point estimate of theta in the inference function, without further use twice
def estimate_theta_avg_reward_diff_partial_0(study_df):
    """
    Partial linear regression with linear nuisance function in numpy version
    """
    
    ####### simplified linear model without pretreat_features
    n = study_df["user_id"].nunique()
    T = study_df["calendar_t"].nunique()
    reward = study_df["reward"].values.reshape(-1, 1)
    theta_hat = reward.reshape(n, T).mean()
    return np.array(theta_hat.flatten()) # .flatten() at least 1D array

