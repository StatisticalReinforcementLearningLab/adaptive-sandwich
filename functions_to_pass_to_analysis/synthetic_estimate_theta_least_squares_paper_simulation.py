from sklearn.linear_model import LinearRegression


def synthetic_estimate_theta_least_squares_no_action_centering(study_df):

    covariate_names = ["intercept", "past_reward", "action"]
    # Note that the intercept is included in the features already (col of 1s)
    # in the way we typically run this
    linear_model = LinearRegression(fit_intercept=False)

    in_study_bool = study_df["in_study"] == 1
    trimmed_df = study_df.loc[in_study_bool, covariate_names].copy()
    in_study_df = study_df[in_study_bool]

    linear_model.fit(trimmed_df, in_study_df["reward"])

    return linear_model.coef_
