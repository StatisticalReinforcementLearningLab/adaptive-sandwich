import os
import logging
import pickle as pkl

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
    "base_env_type": "NON_STAT",  # This indicated non-stationarity in the environment.
    "effect_size_scale": "None",  # Don't be alarmed, this is not a paramter for the V3 algorithm
    "delayed_effect_scale": "LOW_R",
    "alg_type": "BLR_AC_V3",
    "noise_var": "None",  # Don't be alarmed, this is not a paramter for the V3 algorithm
    "clipping_vals": [0.2, 0.8],
    "b_logistic": 0.515,
    "num_decision_times_between_updates": 14,
    "cluster_size": "full_pooling",
    "cost_params": [80, 40],
    "per_user_weeks_in_study": 10,
}


def get_algorithm(
    alg_type, smoothing_func, num_decision_times_between_updates, cost_params, noise_var
):
    algorithm = None
    if alg_type == "BLR_AC":
        algorithm = rl_algorithm.BlrActionCentering(
            cost_params, num_decision_times_between_updates, smoothing_func, noise_var
        )
    elif alg_type == "BLR_NO_AC":
        algorithm = rl_algorithm.BlrNoActionCentering(
            cost_params, num_decision_times_between_updates, smoothing_func, noise_var
        )
    elif alg_type == "BLR_AC_V2":
        algorithm = rl_algorithm.BlrACV2(
            cost_params, num_decision_times_between_updates, smoothing_func
        )
    elif alg_type == "BLR_AC_V3":
        algorithm = rl_algorithm.BlrACV3(
            cost_params, num_decision_times_between_updates, smoothing_func
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


def run(exp_path, seed, num_users, users_per_recruitment):
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
    )

    # Sample the users and set up the data-generating environment.
    template_users_list, environment_module = get_sim_env(
        EXP_SETTINGS["sim_env_version"],
        EXP_SETTINGS["base_env_type"],
        EXP_SETTINGS["delayed_effect_scale"],
        seed,
        num_users,
    )

    # Carry out the experiment.  This is the critical code.
    # The data dataframe holds all study data generated, whereas the update
    # dataframe holds the updates to the algorithm's parameters.
    data_df, update_df = rl_experiments.run_incremental_recruitment_exp(
        template_users_list,
        users_per_recruitment,
        algorithm,
        environment_module,
        EXP_SETTINGS["per_user_weeks_in_study"],
    )

    # Write the picked results to file.
    logger.info("Trial simulation complete. Writing results to file.")
    pd.to_pickle(data_df, exp_path + f"/{seed}_data_df.p")
    pd.to_pickle(update_df, exp_path + f"/{seed}_update_df.p")

    return data_df, update_df


def collect_alg_update_function_args(data_df, update_df):
    df = pd.DataFrame(
        columns=[
            "update_idx",
            "user_idx",
            "betas",
            "n_users",
            "states",
            "actions",
            "act_probs",
            "decision_times",
            "rewards",
            "prior_mu",
            "prior_sigma_inv",
            "init_noise_var",
        ]
    )
    update_func_args_dict = {}

    prior_mu = None
    prior_sigma_inv = None

    # TODO: Adjust this logic if we stop doing extra updates. Actually, it
    # needs to adjust based on dynamic recruitment rate anyway.
    starting_policy = 5

    for i in range(update_df["update_idx"].max()):
        utime_dict = {}

        temp = update_df[update_df["update_idx"] == i]

        mu = []
        # TODO: Potentially adjust the 15 to be dynamic based on settings
        for j in range(15):
            mu.append(temp[f"posterior_mu.{j}"].values[0])

        sigma = []

        # TODO: Potentially adjust the 15 to be dynamic based on settings
        for j in range(15):
            t = []
            for k in range(15):
                t.append(temp[f"posterior_var.{j}.{k}"].values[0])
            sigma.append(t)

        mu = jnp.array(mu)
        sigma = jnp.array(sigma)
        sigma_inv = jnp.linalg.inv(sigma)
        utsigma_inv = sigma_inv[jnp.triu_indices(sigma_inv.shape[0])]
        Vt = utsigma_inv.flatten()

        if i == 0:
            prior_mu = mu
            prior_sigma_inv = jnp.linalg.inv(sigma)
            continue
        elif i < starting_policy:
            continue
        else:
            # flatten and combine both mu and sigma into betas
            betas = jnp.concatenate([mu, Vt])

            num_users_entered_already = data_df[data_df["policy_idx"] < i][
                "user_idx"
            ].nunique()
            for user in data_df["user_idx"].unique():
                # Create the data dataframe
                temp = data_df[
                    (data_df["policy_idx"] < i) & (data_df["user_idx"] == user)
                ].reset_index(drop=True)

                # Check if the user has any data
                if temp.shape[0] != 0:
                    # Sort by calendar_decision_t
                    temp = temp.sort_values(by="calendar_decision_t")

                    # Phi = []
                    states = []
                    actions = []
                    act_probs = []
                    decision_times = []
                    rewards = jnp.array(temp["reward"].values)
                    for j in range(temp.shape[0]):
                        state = np.array(
                            temp.loc[j][
                                [
                                    "state.tod",
                                    "state.b.bar",
                                    "state.a.bar",
                                    "state.app.engage",
                                    "state.bias",
                                ]
                            ].values,
                            dtype=np.float32,
                        )
                        action = temp.loc[j]["action"]
                        act_prob = temp.loc[j]["prob"]
                        # TODO: Potentially adjust the 13 to be dynamic based on settings
                        decision_time = temp.loc[j]["calendar_decision_t"] - 13

                        states.append(state)
                        actions.append(action)
                        act_probs.append(act_prob)
                        decision_times.append(decision_time)

                    states = jnp.array(states)
                    actions = jnp.array(actions).reshape(-1, 1)
                    act_probs = jnp.array(act_probs).reshape(-1, 1)
                    decision_times = jnp.array(decision_times).reshape(-1, 1)

                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                [
                                    [
                                        i,
                                        user,
                                        betas,
                                        num_users_entered_already,
                                        states,
                                        actions,
                                        act_probs,
                                        decision_times,
                                        rewards,
                                        prior_mu,
                                        prior_sigma_inv,
                                        3396.449,
                                    ]
                                ],
                                columns=[
                                    "update_idx",
                                    "user_idx",
                                    "betas",
                                    "n_users",
                                    "states",
                                    "actions",
                                    "act_probs",
                                    "decision_times",
                                    "rewards",
                                    "prior_mu",
                                    "prior_sigma_inv",
                                    "init_noise_var",
                                ],
                            ),
                        ]
                    )
                    utime_dict[user] = (
                        betas,
                        num_users_entered_already,
                        states,
                        actions,
                        act_probs,
                        decision_times,
                        rewards,
                        prior_mu,
                        prior_sigma_inv,
                        3396.449,
                    )

                # Otherwise add a row with no data
                else:
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(
                                [
                                    [
                                        i,
                                        user,
                                        betas,
                                        num_users_entered_already,
                                        [],
                                        [],
                                        [],
                                        [],
                                        [],
                                        prior_mu,
                                        prior_sigma_inv,
                                        3396.449,
                                    ]
                                ],
                                columns=[
                                    "update_idx",
                                    "user_idx",
                                    "betas",
                                    "n_users",
                                    "states",
                                    "actions",
                                    "act_probs",
                                    "decision_times",
                                    "rewards",
                                    "prior_mu",
                                    "prior_sigma_inv",
                                    "init_noise_var",
                                ],
                            ),
                        ]
                    )
                    utime_dict[user] = ()

        update_func_args_dict[i] = utime_dict

    df.reset_index(drop=True, inplace=True)

    return df, update_func_args_dict


def collect_action_prob_function_args(data_df, update_df, update_func_args_dict):
    # Now create the dataframe for the action selection function
    df2 = pd.DataFrame(columns=["calendar_decision_t", "user_idx", "beta", "advantage"])

    act_prob_dict = {}

    # TODO: Adjust this logic if we stop doing extra updates. May need to
    # adjust based on dynamic recruitment rate anyway
    starting_policy = 5

    for i in range(
        data_df["calendar_decision_t"].min(), data_df["calendar_decision_t"].max() + 1
    ):
        temp = data_df[data_df["calendar_decision_t"] == i].reset_index(drop=True)

        utime_dict = {}

        for user in data_df["user_idx"].unique():
            record = temp[temp["user_idx"] == user].reset_index(drop=True)

            # Check if the user has any data
            if record.shape[0] != 0:
                # Fetch the policy number and associated beta
                policy = record["policy_idx"].values[0]
                if policy < starting_policy:
                    # Construct beta from update dataframe
                    update_0 = update_df[update_df["update_idx"] == 0].reset_index(
                        drop=True
                    )
                    # TODO: Perhaps refer to feature dim variable instead of hardcoding 15
                    mu = [
                        update_0[f"posterior_mu.{j}".format(j)].values[0]
                        for j in range(15)
                    ]
                    sigma = [
                        [
                            update_0[f"posterior_var.{j}.{k}"].values[0]
                            for k in range(15)
                        ]
                        for j in range(15)
                    ]
                    sigma_inv = jnp.linalg.inv(jnp.array(sigma))
                    utsigma_inv = np.array(sigma_inv)[
                        np.triu_indices(sigma_inv.shape[0])
                    ]
                    beta = jnp.concatenate(
                        [jnp.array(mu), jnp.array(utsigma_inv).flatten()]
                    )
                else:
                    # Get beta from update_func_args_dict first user first record
                    t1 = list(update_func_args_dict[policy].keys())[0]
                    beta = update_func_args_dict[policy][t1][0]

                # Fetch the state
                state = jnp.array(
                    record[
                        [
                            "state.tod",
                            "state.b.bar",
                            "state.a.bar",
                            "state.app.engage",
                            "state.bias",
                        ]
                    ].values[0],
                    dtype=np.float32,
                )

                df2 = pd.concat(
                    [
                        df2,
                        # TODO: Think about whether this 13 needs to be dynamic.
                        pd.DataFrame(
                            [[i - 13, user, beta, state]],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "beta",
                                "advantage",
                            ],
                        ),
                    ]
                )
                utime_dict[user] = (beta, state)
            else:
                df2 = pd.concat(
                    [
                        df2,
                        # TODO: Think about whether this 13 needs to be dynamic.
                        pd.DataFrame(
                            [[i - 13, user, [], []]],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "beta",
                                "advantage",
                            ],
                        ),
                    ]
                )
                utime_dict[user] = ()

        # TODO: Think about whether this 13 needs to be dynamic.
        act_prob_dict[i - 13] = utime_dict

    df2.reset_index(drop=True, inplace=True)

    return df2, act_prob_dict


def create_study_df(data):
    # Create the study dataframe
    # T x n pandas dataframe where T is the set of all calendar decision times for which at least
    # one user is active and n is the total number of users
    # in the study. The dataframe should have the following columns:
    # - calendar_decision_t: the calendar decision time
    # - user_idx: the user index
    # - in_study_indicator: a binary indicator for whether the user is in the study at the given
    #   calendar decision time
    # - action: the action taken by the user at the given calendar decision time
    # - policy_idx: the policy index used by the user at the given calendar decision time
    # - act_prob: the action selection probability for the user at the given calendar decision time
    # - reward: the reward received by the user at the given calendar decision time
    # - state: the state of the user at the given calendar decision time

    starting_policy = 5

    df3 = pd.DataFrame(
        columns=[
            "calendar_decision_t",
            "user_idx",
            "in_study_indicator",
            "action",
            "policy_idx",
            "act_prob",
            "reward",
            "oscb",
            "tod",
            "bbar",
            "abar",
            "appengage",
            "bias",
        ]
    )

    for i in range(
        data["calendar_decision_t"].min(), data["calendar_decision_t"].max() + 1
    ):
        temp = data[data["calendar_decision_t"] == i].reset_index(drop=True)

        for user in data["user_idx"].unique():
            record = temp[temp["user_idx"] == user].reset_index(drop=True)

            # Check if the user has any data
            if record.shape[0] != 0:
                # Fetch the policy number and associated beta
                policy = record["policy_idx"].values[0]

                # TODO: Adjust starting policy logic to be dynamic (and if we stop doing extra updates)
                if policy < starting_policy:
                    policy = 4

                # Fetch the state
                state = jnp.array(
                    record[
                        [
                            "state.tod",
                            "state.b.bar",
                            "state.a.bar",
                            "state.app.engage",
                            "state.bias",
                        ]
                    ].values[0],
                    dtype=np.float32,
                )

                df3 = pd.concat(
                    [
                        df3,
                        # TODO: Think about whether this 13 needs to be dynamic.
                        pd.DataFrame(
                            [
                                [
                                    i - 13,
                                    user,
                                    1,
                                    record["action"].values[0],
                                    policy,
                                    record["prob"].values[0],
                                    record["reward"].values[0],
                                    record["quality"].values[0],
                                    state[0].item(),
                                    state[1].item(),
                                    state[2].item(),
                                    state[3].item(),
                                    state[4].item(),
                                ]
                            ],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "in_study_indicator",
                                "action",
                                "policy_idx",
                                "act_prob",
                                "reward",
                                "oscb",
                                "tod",
                                "bbar",
                                "abar",
                                "appengage",
                                "bias",
                            ],
                        ),
                    ]
                )
            else:
                df3 = pd.concat(
                    [
                        df3,
                        # TODO: Think about whether this 13 needs to be dynamic.
                        pd.DataFrame(
                            [[i - 13, user, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                            columns=[
                                "calendar_decision_t",
                                "user_idx",
                                "in_study_indicator",
                                "action",
                                "policy_idx",
                                "act_prob",
                                "reward",
                                "oscb",
                                "tod",
                                "bbar",
                                "abar",
                                "appengage",
                                "bias",
                            ],
                        ),
                    ]
                )
    df3.reset_index(drop=True, inplace=True)
    df3 = df3.infer_objects()
    return df3


def process_results(data_df, update_df, exp_path, seed):
    _, loss_dict = collect_alg_update_function_args(data_df, update_df)
    _, act_prob_dict = collect_action_prob_function_args(data_df, update_df, loss_dict)
    study_df = create_study_df(data_df)

    with open(exp_path + f"/{seed}_loss_fn_data.pkl", "wb") as f:
        pkl.dump(loss_dict, f)
    with open(exp_path + f"/{seed}_action_data.pkl", "wb") as f:
        pkl.dump(act_prob_dict, f)
    with open(exp_path + f"/{seed}_study_data.pkl", "wb") as f:
        pkl.dump(study_df, f)


# TODO: Possibly parameterize the function to take in some or all of the
# experiment settings on the command line.
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
def main(seed, exp_dir, num_users, users_per_recruitment):
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
            "The number of users recruited per recruitment.

    Returns:
    None
    """

    # Make exp_name from experiment settings
    exp_name = "_".join([str(EXP_SETTINGS[key]) for key in OUTPUT_PATH_NAMES])
    exp_path = os.path.join(exp_dir, exp_name, str(seed))

    logger.info("Running experiment: %s", exp_name)
    # The data df collects the data generatd by the experiment. The update df
    # records the updates to the algorithm's parameters.
    data_df, update_df = run(exp_path, seed, num_users, users_per_recruitment)

    logger.info("Packaging results for after-study analysis.")
    process_results(data_df, update_df, exp_path, seed)

    logger.info("Experiment and post-processing complete.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
