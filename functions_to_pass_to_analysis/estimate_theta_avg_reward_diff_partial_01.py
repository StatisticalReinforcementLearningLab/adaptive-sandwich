import numpy as np
from sklearn.linear_model import LinearRegression


###### point estimate of theta in the inference function, without further use twice
def estimate_theta_avg_reward_diff_partial_01(study_df):
    """
    Partial linear regression with linear nuisance function in numpy version
    """
    
    ####### simplified linear model without pretreat_features
    n = study_df["user_id"].nunique()
    T = study_df["calendar_t"].nunique()
    intercept = study_df["intercept"].values.reshape(n, T)[:,0:1]
    reward = study_df["reward"].values.reshape(-1, 1)
    Z_id = study_df["Z_id"].values.reshape(n, T)[:,0:1] # constant to time
    C_design = np.hstack((intercept, Z_id)) # [n, 4]
    ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    theta_hat = np.linalg.pinv(C_design) @ ave_reward # [4, 1], pinv is equivalent to the close-form solution, but it is more stable
    return np.array(theta_hat.flatten()) # .flatten() at least 1D array


    