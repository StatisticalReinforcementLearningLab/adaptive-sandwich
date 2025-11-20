import numpy as np
from sklearn.linear_model import LinearRegression


###### point estimate of theta in the inference function, without further use twice
def estimate_theta_avg_reward_diff_partial_01feature(study_df):
    """
    Partial linear regression with linear nuisance function in numpy version
    """
    
    ####### simplified linear model without pretreat_features
    # n = study_df["user_id"].nunique()
    # T = study_df["calendar_t"].nunique()
    # intercept = study_df["intercept"].values.reshape(n, T)[:,0:1]
    # reward = study_df["reward"].values.reshape(-1, 1)
    # Z_id = study_df["Z_id"].values.reshape(n, T)[:,0:1] # constant to time
    # C_design = np.hstack((intercept, Z_id)) # [n, 4]
    # ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    # theta_hat = np.linalg.pinv(C_design) @ ave_reward # [4, 1], pinv is equivalent to the close-form solution, but it is more stable
    # return np.array(theta_hat[-1].flatten()) # .flatten() at least 1D array


    ###### fully linear estimation: equivalent to partial linear regression by FML theorem
    n = study_df["user_id"].nunique()
    T = study_df["calendar_t"].nunique()
    intercept = study_df["intercept"].values.reshape(n, T)[:,0:1]
    reward = study_df["reward"].values.reshape(-1, 1)
    pretreat_features1 = study_df["pretreat_feature1"].values.reshape(n, T)[:,0:1] # pre-treatment features
    pretreat_features2 = study_df["pretreat_feature2"].values.reshape(n, T)[:,0:1] # pre-treatment features
    Z_id = study_df["Z_id"].values.reshape(n, T)[:,0:1] # constant to time

    C_design = np.hstack((intercept, Z_id, pretreat_features1, pretreat_features2)) # [n, 4]
    ave_reward = reward.reshape(n, T).mean(axis=1, keepdims=True) # [n, 1]
    theta_hat = np.linalg.pinv(C_design) @ ave_reward # [4, 1], pinv is equivalent to the close-form solution, but it is more stable
    return np.array(theta_hat.flatten()) # .flatten() at least 1D array

    
    

    
# def estimate_theta_avg_reward_sum(study_df):

#     num_users = study_df["user_id"].nunique()
#     # Sort of assumes no incremental recruitment, though it doesn't really
#     # matter
#     num_decision_times = study_df["calendar_t"].nunique()
#     in_study_bool = study_df["in_study"] == 1

#     return np.array(
#         [study_df.loc[in_study_bool].reward.sum() / num_decision_times / num_users]
#     )



# def synthetic_estimate_theta_least_squares_no_action_centering(study_df):

#     covariate_names = ["intercept", "past_reward"]
#     # Note that the intercept is included in the features already (col of 1s)
#     # in the way we typically run this
#     linear_model = LinearRegression(fit_intercept=False)

#     in_study_bool = study_df["in_study"] == 1
#     trimmed_df = study_df.loc[in_study_bool, covariate_names].copy()
#     in_study_df = study_df[in_study_bool]
#     for feat in covariate_names:
#         trimmed_df[f"action:{feat}"] = in_study_df[feat] * (in_study_df["action"])

#     linear_model.fit(trimmed_df, in_study_df["reward"])

#     return linear_model.coef_

