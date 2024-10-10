import numpy as np


def oralytics_mancl_derouen_bias_correction(meat, study_df):
    """
    Applies the correction from Mancl + Derouen 2001 (https://pubmed.ncbi.nlm.nih.gov/11252587/),
    the rationale being that we can put our estimating function in the form of
    a generalized estimating equation (GEE) and then apply this bias
    correction.

    This is similar to how it is applied in Liao + Klasjna + Tewari + Murphy 2016
    (https://pubmed.ncbi.nlm.nih.gov/26707831/), except our weights here make the
    V matrices in the above formulation not simply the identity matrix, and our
    meat matrix is more complicated.


    """

    # First organize all the data we need.
    covariate_names = [
        "tod",
        "bbar",
        "abar",
        "appengage",
        "bias",  # this is the intercept
    ]

    in_study_bool = study_df["in_study_indicator"] == 1
    # TODO: Figure out indicator(s)
    trimmed_df = study_df.loc[in_study_bool, covariate_names + ["act_prob", "user_idx"]].copy()

    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = in_study_df["action"] - in_study_df["act_prob"]

    # Now we need to

    # This matrix is the key to the bias correction.  We sandwich
    # the meat between I - H and (I - H).T to get the adjusted meat.
    H =

    I = np.eye(meat.shape[0])
    return (I - H) @ meat @ (I - H).T
