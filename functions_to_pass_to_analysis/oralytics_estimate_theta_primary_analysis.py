from sklearn.linear_model import LinearRegression


def oralytics_estimate_theta_primary_analysis(study_df):
    covariate_names = [
        "tod",
        "bbar",
        "abar",
        "appengage",
        "bias",  # this is the intercept
    ]
    # Note that the intercept is included in the features already (col of 1s)
    linear_model = LinearRegression(fit_intercept=False)

    in_study_bool = study_df["in_study_indicator"] == 1
    # TODO: Figure out indicator(s)
    trimmed_df = study_df.loc[in_study_bool, covariate_names + ["act_prob"]].copy()

    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = in_study_df["action"] - in_study_df["act_prob"]

    linear_model.fit(
        trimmed_df,
        in_study_df["oscb"],
        sample_weight=(1 / (trimmed_df["act_prob"] * (1 - trimmed_df["act_prob"]))),
    )

    return linear_model.coef_
