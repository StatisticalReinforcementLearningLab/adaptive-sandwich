from sklearn.linear_model import LinearRegression
import pandas as pd


# TODO: reward should be replaced with oscb
def miwaves_estimate_theta_primary_analysis(study_df):
    # Note that the intercept is included in the features already (col of 1s)
    linear_model = LinearRegression(fit_intercept=False)

    in_study_bool = study_df["in_study_indicator"] == 1
    trimmed_df = pd.DataFrame(study_df.loc[in_study_bool, "action_probability"])

    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = (
        in_study_df["action"] - in_study_df["action_probability"]
    )

    linear_model.fit(
        trimmed_df,
        in_study_df["outcome"],
        sample_weight=(
            1
            / (
                trimmed_df["action_probability"]
                * (1 - trimmed_df["action_probability"])
            )
        ),
    )

    return linear_model.coef_
