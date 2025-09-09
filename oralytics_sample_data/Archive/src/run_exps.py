from collections.abc import Hashable
import os
import logging
import pickle as pkl
from typing import Any

import click
import numpy as np
import pandas as pd
import jax.numpy as jnp

import rl_experiments
import rl_algorithm
import sim_env_v1
import sim_env_v2
import sim_env_v3
import smoothing_function
import read_write_info

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX

SIM_ENV_NAMES = ["base_env_type", "delayed_effect_scale", "effect_size_scale"]
ALG_CAND_NAMES = ["b_logistic", "num_decision_times_between_updates", "cluster_size"]
OUTPUT_PATH_NAMES = SIM_ENV_NAMES + ALG_CAND_NAMES

EXP_SETTINGS = {
    "sim_env_version": 3,
    "base_env_type": "NON_STAT",  # This indicates non-stationarity in the environment.
    "effect_size_scale": "None",  # Don't be alarmed, this is not a parameter for the V3 algorithm
    "delayed_effect_scale": "LOW_R",
    "alg_type": "BLR_AC_V3",
    "noise_var": "None",  # Don't be alarmed, this is not a parameter for the V3 algorithm
    "clipping_vals": [0.2, 0.8],
    "b_logistic": 0.515,
    "num_decision_times_between_updates": 14,
    "cluster_size": "full_pooling",
    "cost_params": [80, 40],
    "per_user_weeks_in_study": 10,
    "num_decision_times_per_day_per_user": 2,
    "weeks_between_recruitments": 2,
}


def get_algorithm(
    alg_type,
    smoothing_func,
    num_decision_times_between_updates,
    cost_params,
    noise_var,
    use_monte_carlo_expectation,
):
    algorithm = None
    if alg_type == "BLR_AC":
        algorithm = rl_algorithm.BlrActionCentering(
            cost_params,
            num_decision_times_between_updates,
            smoothing_func,
            noise_var,
            use_monte_carlo_expectation,
        )
    elif alg_type == "BLR_NO_AC":
        algorithm = rl_algorithm.BlrNoActionCentering(
            cost_params,
            num_decision_times_between_updates,
            smoothing_func,
            noise_var,
            use_monte_carlo_expectation,
        )
    elif alg_type == "BLR_AC_V2":
        algorithm = rl_algorithm.BlrACV2(
            cost_params,
            num_decision_times_between_updates,
            smoothing_func,
            use_monte_carlo_expectation,
        )
    elif alg_type == "BLR_AC_V3":
        algorithm = rl_algorithm.BlrACV3(
            cost_params,
            num_decision_times_between_updates,
            smoothing_func,
            use_monte_carlo_expectation,
        )
    else:
        raise ValueError(f"Algorithm type {alg_type} not recognized.")
    logger.info("ALG TYPE: %s", alg_type)
    return algorithm


def get_sim_env(
    sim_env_version,
    base_env_type,
    delayed_effect_scale,
    seed,
    num_users,
):

    sim_env_module = [sim_env_v1, sim_env_v2, sim_env_v3][sim_env_version - 1]
    sim_env_constructor = getattr(
        sim_env_module, f"SimulationEnvironmentV{sim_env_version}"
    )

    # Randomly select a set of user ids WITH REPLACEMENT from the set of
    # possible user ids from the simulation environment.
    logger.info("SEED: %d", seed)
    np.random.seed(seed)
    users_list = np.random.choice(sim_env_module.SIM_ENV_USERS, size=num_users)
    logger.info(users_list)

    # Instantiate the simulation environment with the selected user ids and other
    # settings.
    sim_env = sim_env_constructor(users_list, base_env_type, delayed_effect_scale)
    logger.info(
        "PROCESSED ENV_TYPE: %s, DELAYED EFFECT SIZE SCALE: %s",
        base_env_type,
        delayed_effect_scale,
    )

    return users_list, sim_env


def run(
    exp_path: str,
    seed: int,
    num_users: int,
    users_per_recruitment: int,
    num_users_before_update: int,
    per_user_weeks_in_study: int,
    ignore_variance_for_rl_parameter_definition: bool,
    use_monte_carlo_expectation: bool,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[int, dict[Hashable, tuple[Any, ...]]],
    dict[int, dict[Hashable, tuple[Any, ...]]],
]:
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Set up the RL algorithm
    L_min, L_max = EXP_SETTINGS["clipping_vals"]
    logger.info("CLIPPING VALUES: [%s, %s]", L_min, L_max)
    smoothing_func = smoothing_function.generalized_logistic_func_wrapper(
        L_min,
        L_max,
        EXP_SETTINGS["b_logistic"],
    )
    logger.info("PROCESSED CANDIDATE VALS %s", EXP_SETTINGS["cost_params"])
    algorithm = get_algorithm(
        EXP_SETTINGS["alg_type"],
        smoothing_func,
        EXP_SETTINGS["num_decision_times_between_updates"],
        EXP_SETTINGS["cost_params"],
        EXP_SETTINGS["noise_var"],
        use_monte_carlo_expectation,
    )

    # Sample the users and set up the data-generating environment.
    template_users_list, environment_module = get_sim_env(
        EXP_SETTINGS["sim_env_version"],
        EXP_SETTINGS["base_env_type"],
        EXP_SETTINGS["delayed_effect_scale"],
        seed,
        num_users,
    )

    # Carry out the experiment.  This is the critical piece of code.
    # The data dataframe holds all study data generated, whereas the update
    # dataframe holds the updates to the algorithm's parameters.
    (
        data_df,
        update_df,
        study_df,
        alg_update_function_args,
        action_prob_function_args,
    ) = rl_experiments.run_incremental_recruitment_exp(
        template_users_list,
        users_per_recruitment,
        algorithm,
        environment_module,
        per_user_weeks_in_study,
        num_users_before_update,
        EXP_SETTINGS["num_decision_times_per_day_per_user"],
        EXP_SETTINGS["weeks_between_recruitments"],
        ignore_variance_for_rl_parameter_definition,
    )

    return (
        data_df,
        update_df,
        study_df,
        alg_update_function_args,
        action_prob_function_args,
    )


@click.command()
@click.option("--seed", default=0, help="Random seed for the experiment.")
@click.option(
    "--exp_dir",
    default=WRITE_PATH_PREFIX,
    help="Directory in which to save the experiment results.",
)
@click.option(
    "--num_users",
    default=70,
    help="The number of users in the trial. Should be a multiple of 5.",
)
@click.option(
    "--users_per_recruitment",
    default=5,
    help="The number of users recruited per recruitment.",
)
@click.option(
    "--num_users_before_update",
    default=15,
    help="The number of users required before the first update.",
)
@click.option(
    "--per_user_weeks_in_study",
    default=10,
    help="The number of weeks each user is in the study.",
)
@click.option(
    "--ignore_variance_for_rl_parameter_definition",
    default=0,
    type=click.Choice(["0", "1"]),
    help="If set, we will package arguments for analysis as if the RL parameters are simply the elements of the posterior mean, and the elements of the posterior variance are covariates.",
)
@click.option(
    "--use_numerical_expectation",
    is_flag=True,
    default=False,
    help="Use numerical integration instead of Monte Carlo expectation for action probabilities.",
)
def main(
    seed,
    exp_dir,
    num_users,
    users_per_recruitment,
    num_users_before_update,
    per_user_weeks_in_study,
    ignore_variance_for_rl_parameter_definition,
    use_numerical_expectation,
):
    """
    Run the main experiment with the given parameters.

    Parameters:
        seed (int):
            The seed for random number generation to ensure reproducibility.
        exp_dir (str):
            The directory where the experiment results will be stored.
        num_users (int):
            The number of users to simulate in the experiment.
        users_per_recruitment (float):
            The number of users recruited per recruitment.
        num_users_before_update (int):
            The number of users required before the first update.
        per_user_weeks_in_study (int):
            The number of weeks each user is in the study.
        ignore_variance_for_rl_parameter_definition (bool):
            If set, the RL parameters are treated as the posterior mean, and the posterior variance is treated as covariates.
        use_numerical_expectation (bool):
            Whether to use numerical integration for action probabilities instead of Monte Carlo expectation.

    Returns:
    None
    """

    # Really want this to be a boolean, but it's easiest to pass as a 0 or
    # 1 instead of a flag from the driver shell script.
    ignore_variance_for_rl_parameter_definition = int(
        ignore_variance_for_rl_parameter_definition
    )

    # Make exp_name from experiment settings
    exp_name = "_".join([str(EXP_SETTINGS[key]) for key in OUTPUT_PATH_NAMES])
    exp_path = os.path.join(exp_dir, exp_name, str(seed))

    logger.info("Running experiment: %s", exp_name)
    # The data df collects the data generated by the experiment. The update df
    # records the updates to the algorithm's parameters. The study df is a reformatting
    # of the data df for the analysis package. The alg_update_function_args and
    # action_prob_function_args are dictionaries that hold the arguments for the
    # algorithm update function and the action probability function for each user at the relevant
    # times.
    (
        data_df,
        update_df,
        study_df,
        alg_update_function_args,
        action_prob_function_args,
    ) = run(
        exp_path,
        seed,
        num_users,
        users_per_recruitment,
        num_users_before_update,
        per_user_weeks_in_study,
        ignore_variance_for_rl_parameter_definition,
        not use_numerical_expectation,
    )

    # Write the pickled results to file.
    pd.to_pickle(data_df, exp_path + "/data_df.pkl")
    pd.to_pickle(update_df, exp_path + "/update_df.pkl")
    pd.to_pickle(study_df, exp_path + "/study_df.pkl")
    with open(exp_path + "/loss_fn_data.pkl", "wb") as f:
        pkl.dump(alg_update_function_args, f)
    with open(exp_path + "/action_data.pkl", "wb") as f:
        pkl.dump(action_prob_function_args, f)

    logger.info("Experiment and post-processing complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
