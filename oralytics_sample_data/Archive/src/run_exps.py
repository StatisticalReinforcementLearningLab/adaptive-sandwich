import rl_experiments
import rl_algorithm
import sim_env_v1
import sim_env_v2
import sim_env_v3
import smoothing_function
import experiment_global_vars
import read_write_info

import numpy as np
import pandas as pd
import sys
import os

import pickle as pkl
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp

WRITE_PATH_PREFIX = read_write_info.WRITE_PATH_PREFIX
NUM_TRIAL_USERS = experiment_global_vars.NUM_TRIAL_USERS

SIM_ENV_NAMES = ["base_env_type", "delayed_effect_scale", "effect_size_scale"]
ALG_CAND_NAMES = ["b_logistic", "update_cadence", "cluster_size"]
OUTPUT_PATH_NAMES = SIM_ENV_NAMES + ALG_CAND_NAMES


def get_user_list(version, study_idxs):
    user_list = [version.USER_INDICES[idx] for idx in study_idxs]

    return user_list


def get_alg_candidate(
    alg_type, cluster_size, smoothing_func, update_cadence, cost_params, noise_var
):
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
    print("ALG TYPE: {}".format(alg_type))
    return alg_candidate


def get_sim_env(
    sim_env_version,
    base_env_type,
    effect_size_scale,
    delayed_effect_scale,
    current_seed,
):
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
    study_idxs = np.random.choice(version.NUM_USER_MODELS, size=NUM_TRIAL_USERS)

    # get user ids corresponding to index
    users_list = get_user_list(version, study_idxs)
    print(users_list)
    ## HANDLING SIMULATION ENVIRONMENT ##
    environment_module = sim_env(
        users_list, base_env_type, effect_size_scale, delayed_effect_scale
    )

    print(
        "PROCESSED ENV_TYPE: {}, EFFECT SIZE SCALE: {}".format(
            base_env_type, effect_size_scale
        )
    )

    return users_list, environment_module


def run(exp_dir, exp_name, exp_kwargs, current_seed):
    exp_path = os.path.join(exp_dir, exp_name, str(current_seed))

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    ## HANDLING RL ALGORITHM CANDIDATE ##
    cluster_size = NUM_TRIAL_USERS
    L_min, L_max = exp_kwargs["clipping_vals"]
    b_logistic = exp_kwargs["b_logistic"]
    print("CLIPPING VALUES: [{}, {}]".format(L_min, L_max))
    smoothing_func = smoothing_function.genearlized_logistic_func_wrapper(
        L_min, L_max, b_logistic
    )
    update_cadence = exp_kwargs["update_cadence"]
    cost_params = exp_kwargs["cost_params"]
    print("PROCESSED CANDIDATE VALS {}".format(cost_params))
    noise_var = exp_kwargs["noise_var"]
    alg_type = exp_kwargs["alg_type"]
    alg_candidate = get_alg_candidate(
        alg_type, cluster_size, smoothing_func, update_cadence, cost_params, noise_var
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
    )
    user_groups = rl_experiments.pre_process_users(users_list)
    data_df, update_df = rl_experiments.run_incremental_recruitment_exp(
        user_groups, alg_candidate, environment_module
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
    n_users = 70

    for i in range(update["update_t"].max()):
        utime_dict = {}

        temp = update[update["update_t"] == i]

        mu = []
        for j in range(15):
            mu.append(temp["posterior_mu.{0}".format(j)].values[0])

        sigma = []

        for j in range(15):
            t = []
            for k in range(15):
                t.append(temp["posterior_var.{0}.{1}".format(j, k)].values[0])
            sigma.append(t)

        mu = jnp.array(mu)
        sigma = jnp.array(sigma)
        sigma_inv = jnp.linalg.inv(sigma)
        utsigma_inv = sigma_inv[jnp.triu_indices(sigma_inv.shape[0])]
        Vt = utsigma_inv.flatten() / n_users

        if i == 0:
            prior_mu = mu
            prior_sigma_inv = jnp.linalg.inv(sigma)
            continue
        elif i < starting_policy:
            continue
        else:
            # flatten and combine both mu and sigma into betas
            betas = jnp.concatenate([mu, Vt])

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
                                        n_users,
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
                        n_users,
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
                                        n_users,
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
    n_users = 70

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
                        update_0["posterior_mu.{0}".format(j)].values[0]
                        for j in range(15)
                    ]
                    sigma = [
                        [
                            update_0["posterior_var.{0}.{1}".format(j, k)].values[0]
                            for k in range(15)
                        ]
                        for j in range(15)
                    ]
                    sigma_inv = jnp.linalg.inv(jnp.array(sigma))
                    utsigma_inv = np.array(sigma_inv)[
                        np.triu_indices(sigma_inv.shape[0])
                    ]
                    beta = jnp.concatenate(
                        [jnp.array(mu), (jnp.array(utsigma_inv).flatten()) / (n_users)]
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
                utime_dict[user] = (beta, state, n_users)
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


def process_results(exp_dir, exp_name, exp_kwargs, current_seed):
    exp_path = os.path.join(exp_dir, exp_name, str(current_seed))
    data = pkl.load(open(exp_path + "/{}_data_df.p".format(current_seed), "rb"))
    update = pkl.load(open(exp_path + "/{}_update_df.p".format(current_seed), "rb"))

    _, loss_dict = create_loss_fn_dataframe(data, update)
    _, act_prob_dict = create_action_df(data, update, loss_dict)
    study_df = create_study_df(data)

    pkl.dump(
        loss_dict, open(exp_path + "/{}_loss_fn_data.pkl".format(current_seed), "wb")
    )
    pkl.dump(
        act_prob_dict, open(exp_path + "/{}_action_data.pkl".format(current_seed), "wb")
    )
    pkl.dump(study_df, open(exp_path + "/{}_study_data.pkl".format(current_seed), "wb"))


def main():
    exp_dir = WRITE_PATH_PREFIX
    seed = int(sys.argv[1])
    exp_kwargs = read_write_info.exp_kwargs

    # Make exp_name from exp_kwargs
    exp_name = "_".join([str(exp_kwargs[key]) for key in OUTPUT_PATH_NAMES])

    print("Running experiment:", exp_name)
    run(exp_dir, exp_name, exp_kwargs, seed)

    print("Analyzing results")
    process_results(exp_dir, exp_name, exp_kwargs, seed)


if __name__ == "__main__":
    main()
