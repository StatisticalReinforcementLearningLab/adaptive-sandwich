import sys
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
ALG_CAND_NAMES = ["b_logistic", "update_cadence", "cluster_size"]
OUTPUT_PATH_NAMES = SIM_ENV_NAMES + ALG_CAND_NAMES


def get_user_list(version, study_idxs):
    user_list = [version.USER_INDICES[idx] for idx in study_idxs]

    return user_list


def get_alg_candidate(alg_type, smoothing_func, update_cadence, cost_params, noise_var):
    alg_candidate = None
    if alg_type == "BLR_AC":
        alg_candidate = rl_algorithm.BlrActionCentering(
            cost_params, update_cadence, smoothing_func, noise_var
        )
    elif alg_type == "BLR_NO_AC":
        alg_candidate = rl_algorithm.BlrNoActionCentering(
            cost_params, update_cadence, smoothing_func, noise_var
        )
    elif alg_type == "BLR_AC_V2":
        alg_candidate = rl_algorithm.BlrACV2(
            cost_params, update_cadence, smoothing_func
        )
    elif alg_type == "BLR_AC_V3":
        alg_candidate = rl_algorithm.BlrACV3(
            cost_params, update_cadence, smoothing_func
        )
    else:
        print("ERROR: NO ALG_TYPE FOUND - ", alg_type)
    print(f"ALG TYPE: {alg_type}")
    return alg_candidate


def get_sim_env(
    sim_env_version,
    base_env_type,
    effect_size_scale,
    delayed_effect_scale,
    current_seed,
    num_users,
):
    version = sim_env = None
    if sim_env_version == "v3":
        version = sim_env_v3
        sim_env = sim_env_v3.SimulationEnvironmentV3
    elif sim_env_version == "v2":
        version = sim_env_v2
        sim_env = sim_env_v2.SimulationEnvironmentV2
    elif sim_env_version == "v1":
        version = sim_env_v1
        sim_env = sim_env_v1.SimulationEnvironmentV1
    else:
        print("ERROR: NO SIM ENV VERSION FOUND - ", sim_env_version)
    # draw different users per trial
    print("SEED: ", current_seed)
    np.random.seed(current_seed)
    study_idxs = np.random.choice(version.NUM_USER_MODELS, size=num_users)

    # get user ids corresponding to index
    users_list = get_user_list(version, study_idxs)
    print(users_list)
    ## HANDLING SIMULATION ENVIRONMENT ##
    environment_module = sim_env(
        users_list, base_env_type, effect_size_scale, delayed_effect_scale
    )

    print(
        f"PROCESSED ENV_TYPE: {base_env_type}, EFFECT SIZE SCALE: {effect_size_scale}"
    )

    return users_list, environment_module


def run(exp_dir, exp_name, exp_kwargs, current_seed, num_users, recruitment_rate):
    exp_path = os.path.join(exp_dir, exp_name, str(current_seed))

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    ## HANDLING RL ALGORITHM CANDIDATE ##
    L_min, L_max = exp_kwargs["clipping_vals"]
    b_logistic = exp_kwargs["b_logistic"]
    print(f"CLIPPING VALUES: [{L_min}, {L_max}]")
    smoothing_func = smoothing_function.generalized_logistic_func_wrapper(
        L_min, L_max, b_logistic
    )
    update_cadence = exp_kwargs["update_cadence"]
    cost_params = exp_kwargs["cost_params"]
    print(f"PROCESSED CANDIDATE VALS {cost_params}")
    noise_var = exp_kwargs["noise_var"]
    alg_type = exp_kwargs["alg_type"]
    alg_candidate = get_alg_candidate(
        alg_type, smoothing_func, update_cadence, cost_params, noise_var
    )

    data_pickle_template = exp_path + "/{}_data_df.p"
    update_pickle_template = exp_path + "/{}_update_df.p"

    sim_env_version = exp_kwargs["sim_env_version"]
    base_env_type = exp_kwargs["base_env_type"]
    effect_size_scale = exp_kwargs["effect_size_scale"]
    delayed_effect_scale = exp_kwargs["delayed_effect_scale"]

    users_list, environment_module = get_sim_env(
        sim_env_version,
        base_env_type,
        effect_size_scale,
        delayed_effect_scale,
        current_seed,
        num_users,
    )
    user_groups = rl_experiments.pre_process_users(users_list, recruitment_rate)
    data_df, update_df = rl_experiments.run_incremental_recruitment_exp(
        user_groups, recruitment_rate, alg_candidate, environment_module
    )
    data_df_pickle_location = data_pickle_template.format(current_seed)
    update_df_pickle_location = update_pickle_template.format(current_seed)

    print("TRIAL DONE, PICKLING NOW")
    pd.to_pickle(data_df, data_df_pickle_location)
    pd.to_pickle(update_df, update_df_pickle_location)


def create_loss_fn_dataframe(data, update):
    df = pd.DataFrame(
        columns=[
            "update_t",
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
    loss_dict = {}

    prior_mu = None
    prior_sigma_inv = None

    starting_policy = 5

    for i in range(update["update_t"].max()):
        utime_dict = {}

        temp = update[update["update_t"] == i]

        mu = []
        for j in range(15):
            mu.append(temp[f"posterior_mu.{j}"].values[0])

        sigma = []

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

            num_users_entered_already = data[data["policy_idx"] < i][
                "user_idx"
            ].nunique()
            for user in data["user_idx"].unique():
                # Create the data dataframe
                temp = data[
                    (data["policy_idx"] < i) & (data["user_idx"] == user)
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
                        decision_time = temp.loc[j]["calendar_decision_t"] - 13

                        states.append(state)
                        actions.append(action)
                        act_probs.append(act_prob)
                        decision_times.append(decision_time)

                        # phi = [*state, *(act_prob * state), *((action - act_prob) * state)]
                        # print(phi)
                        # Phi.append(phi)

                    # Phi = jnp.array(Phi)
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
                                    "update_t",
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
                                    "update_t",
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

        loss_dict[i] = utime_dict

    df.reset_index(drop=True, inplace=True)

    return df, loss_dict


def create_action_df(data, update, loss_dict):
    # Now create the dataframe for the action selection function
    df2 = pd.DataFrame(columns=["calendar_decision_t", "user_idx", "beta", "advantage"])

    act_prob_dict = {}

    starting_policy = 5

    for i in range(
        data["calendar_decision_t"].min(), data["calendar_decision_t"].max() + 1
    ):
        temp = data[data["calendar_decision_t"] == i].reset_index(drop=True)

        utime_dict = {}

        for user in data["user_idx"].unique():
            record = temp[temp["user_idx"] == user].reset_index(drop=True)

            # Check if the user has any data
            if record.shape[0] != 0:
                # Fetch the policy number and associated beta
                policy = record["policy_idx"].values[0]
                if policy < starting_policy:
                    # Construct beta from update dataframe
                    update_0 = update[update["update_t"] == 0].reset_index(drop=True)
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
                    # Get beta from loss_dict dictionary first user first record
                    t1 = list(loss_dict[policy].keys())[0]
                    beta = loss_dict[policy][t1][0]

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

        act_prob_dict[i - 13] = utime_dict

    df2.reset_index(drop=True, inplace=True)

    return df2, act_prob_dict


def create_study_df(data):
    # Create the study dataframe
    # T x n pandas dataframe where T is the set of all calendar decision times for which at least one user is active and n is the total number of users
    # in the study. The dataframe should have the following columns:
    # - calendar_decision_t: the calendar decision time
    # - user_idx: the user index
    # - in_study_indicator: a binary indicator for whether the user is in the study at the given calendar decision time
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

                # df3 = pd.concat([df3, pd.DataFrame([[i - 13, user, 1, record["action"].values[0], policy, record["prob"].values[0], record["reward"].values[0], state]],
                #                 columns=['calendar_decision_t', 'user_idx', 'in_study_indicator', 'action', 'policy_idx', 'act_prob', 'reward', 'state'])])
                df3 = pd.concat(
                    [
                        df3,
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
                # df3 = pd.concat([df3, pd.DataFrame([[i - 13, user, 0, [], [], [], [], []]],
                #                 columns=['calendar_decision_t', 'user_idx', 'in_study_indicator', 'action', 'policy_idx', 'act_prob', 'reward', 'state'])])
                df3 = pd.concat(
                    [
                        df3,
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


def process_results(exp_dir, exp_name, current_seed):
    exp_path = os.path.join(exp_dir, exp_name, str(current_seed))
    with open(exp_path + f"/{current_seed}_data_df.p", "rb") as data_file:
        data = pkl.load(data_file)
    with open(exp_path + f"/{current_seed}_update_df.p", "rb") as update_file:
        update = pkl.load(update_file)

    _, loss_dict = create_loss_fn_dataframe(data, update)
    _, act_prob_dict = create_action_df(data, update, loss_dict)
    study_df = create_study_df(data)

    with open(exp_path + f"/{current_seed}_loss_fn_data.pkl", "wb") as f:
        pkl.dump(loss_dict, f)
    with open(exp_path + f"/{current_seed}_action_data.pkl", "wb") as f:
        pkl.dump(act_prob_dict, f)
    with open(exp_path + f"/{current_seed}_study_data.pkl", "wb") as f:
        pkl.dump(study_df, f)


EXP_KWARGS = {
    "sim_env_version": "v3",
    "base_env_type": "NON_STAT",
    "effect_size_scale": "None",
    "delayed_effect_scale": "LOW_R",
    "alg_type": "BLR_AC_V3",
    "noise_var": "None",
    "clipping_vals": [0.2, 0.8],
    "b_logistic": 0.515,
    "update_cadence": 14,
    "cluster_size": "full_pooling",
    "cost_params": [80, 40],
}


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
    "--recruitment_rate",
    default=5,
    help="The number of users recruited per two week period",
)
def main(seed, exp_dir, num_users, recruitment_rate):
    exp_kwargs = EXP_KWARGS

    # Make exp_name from exp_kwargs
    exp_name = "_".join([str(exp_kwargs[key]) for key in OUTPUT_PATH_NAMES])

    print("Running experiment:", exp_name)
    run(exp_dir, exp_name, exp_kwargs, seed, num_users, recruitment_rate)

    print("Analyzing results")
    process_results(exp_dir, exp_name, seed)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
