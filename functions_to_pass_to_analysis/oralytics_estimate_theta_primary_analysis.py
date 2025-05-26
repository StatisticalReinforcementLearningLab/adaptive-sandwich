import logging

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression

from helper_functions import load_function_from_same_named_file

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def oralytics_estimate_theta_primary_analysis(study_df):
    covariate_names = [
        "tod",
        "bbar",
        "abar",
        "appengage",
        "bias",  # this is the intercept
    ]
    # Note that the intercept is included by the features already (col of 1s)
    linear_model = LinearRegression(fit_intercept=False)

    in_study_bool = study_df["in_study_indicator"] == 1
    # TODO: Figure out other indicator(s)
    trimmed_df = study_df.loc[in_study_bool, covariate_names + ["act_prob"]].copy()

    in_study_df = study_df[in_study_bool]
    trimmed_df["centered_action"] = in_study_df["action"] - in_study_df["act_prob"]

    condition_number = np.linalg.cond(trimmed_df.to_numpy())
    if condition_number > 10**3:
        logger.warning(
            "Design matrix is ill-conditioned, with condition number %s. Consider removing some covariates.",
            condition_number,
        )
        logger.info(
            "Here is the correlation matrix of the columns of the design matrix:\n%s",
            trimmed_df.corr(),
        )
    linear_model.fit(
        trimmed_df,
        in_study_df["oscb"],
        sample_weight=(1 / (trimmed_df["act_prob"] * (1 - trimmed_df["act_prob"]))),
    )

    inference_loss_func_filename = (
        "functions_to_pass_to_analysis/oralytics_primary_analysis_loss.py"
    )
    inference_loss_func = load_function_from_same_named_file(
        inference_loss_func_filename
    )
    inference_estimating_function = jax.grad(inference_loss_func)
    total_grad = jnp.sum(
        jax.vmap(inference_estimating_function, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0))(
            jnp.array(linear_model.coef_),
            jnp.array(trimmed_df["tod"]),
            jnp.array(trimmed_df["bbar"]),
            jnp.array(trimmed_df["abar"]),
            jnp.array(trimmed_df["appengage"]),
            jnp.array(trimmed_df["bias"]),
            jnp.array(in_study_df["action"]),
            jnp.array(in_study_df["oscb"]),
            jnp.array(trimmed_df["act_prob"]),
        ),
        axis=0,
    )

    return linear_model.coef_
