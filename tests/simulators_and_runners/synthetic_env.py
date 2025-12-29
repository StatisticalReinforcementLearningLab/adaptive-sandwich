import numpy as np
import pandas as pd


def get_entry_last_times(args):
    """Compute entry / last decision times compared to calendar for each user"""

    assert args.n % args.recruit_n == 0
    num_distinct_entry_times = args.n // args.recruit_n

    # recall recruit_t is the number of updates between recruitments
    distinct_entry_times = [
        i * args.decisions_between_updates * args.recruit_t + 1
        for i in range(num_distinct_entry_times)
    ]
    # Vector of entry decision times for all users
    entry_times = np.repeat(distinct_entry_times, args.recruit_n)
    # Vector of last decision times for all users
    last_times = entry_times + args.T - 1

    # We now form a chain of the indicator sequences for each recruitment class.
    # We will form the full indicators vector by repeating this by the number
    # of users in the recruitment classes.

    in_study_indicators = np.concatenate(
        [
            np.tile(
                [
                    int(entry_time <= decision_time <= (entry_time + args.T - 1))
                    for decision_time in range(1, max(last_times) + 1)
                ],
                args.recruit_n,
            )
            for entry_time in distinct_entry_times
        ]
    )
    return entry_times, last_times, in_study_indicators


def make_base_study_df(args, all_cols=None):
    """Create the pandas dataframe that will hold the study results"""

    # This avoids the classic Python gotcha involving mutable default args
    if all_cols is None:
        all_cols = [
            "user_id",
            "policy_num",
            "last_t",
            "entry_t",
            "in_study",
            "calendar_t",
        ]

    entry_times, last_times, in_study_indicators = get_entry_last_times(args)

    max_calendar_t = max(last_times)

    df_fill = np.zeros((args.n * max_calendar_t, len(all_cols)))
    df_fill.fill(np.nan)
    study_df = pd.DataFrame(df_fill, columns=all_cols)

    study_df["user_id"] = np.repeat(np.arange(1, args.n + 1), max_calendar_t)
    study_df["calendar_t"] = np.tile(np.arange(1, max_calendar_t + 1), args.n)
    study_df["in_study"] = np.array(in_study_indicators)

    # Used to index into reward noise in an incremental-recruitment-friendly way
    study_df["in_study_row_index"] = study_df["in_study"].cumsum() - 1
    study_df["policy_num"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["last_t"] = np.repeat(last_times, max_calendar_t)
    study_df["entry_t"] = np.repeat(entry_times, max_calendar_t)

    study_df = study_df.reset_index().drop(columns="index")
    return study_df


def load_synthetic_env_params(paramf_path="./synthetic_env_params/delayed_effects.txt"):
    """
    Goal:
        - Load synthetic model parameters
    Input:
        - `env_path`: String path to directory with environment parameters/data
    Output:
        - `param_vec`: Vector of parameters
    """

    # Load Model Parameters ###############################################
    with open(paramf_path, "r", encoding="utf-8") as f:
        params = f.readlines()

    all_env_params = []
    for row in params:
        tmp = np.array([float(x) for x in row.strip().split(",")])
        all_env_params.append(tmp)

    return np.vstack(all_env_params).squeeze()


class SyntheticEnv:
    def __init__(self, args, env_params, env_seed, gen_feats, err_corr="time_corr"):
        """
        - select users and output parameters for selected users
        - make df (dataframe) with all variables collected in study (many missing)
            including entry and last decision times for each user
        """
        self.env_params = env_params
        self.env_seed = env_seed
        self.rng = np.random.default_rng(env_seed)

        self.gen_feats = gen_feats
        self.err_corr = err_corr
        # Set during the run
        self.calendar_T = None

        # Generate noise for rewards
        if err_corr == "time_corr":
            cov_matrix = np.eye(args.T)
            for i in range(args.T):
                for j in range(args.T):
                    if i == j:
                        continue
                    cov_matrix[i][j] = pow(0.5, abs(i - j) / 2)
        elif err_corr == "independent":
            cov_matrix = np.eye(args.T)
        else:
            raise ValueError("Invalid noise correlation")

        # n x T
        noise = self.rng.multivariate_normal(
            mean=np.zeros(args.T), cov=cov_matrix, size=args.n
        )

        self.reward_noise = noise.flatten()

    def update_study_df(self, study_df, t):
        """
        Perform a running update of study_df for the given time.  This is about
        setting up the states for the next decision time if the user will be
        in the study, as the current decision time has already been carried out,
        with the resulting actions and rewards being collected below.
        """

        # Find users at current time who are in study and will continue on
        cont_user_current_bool = (
            (study_df["calendar_t"] == t)
            & (study_df["in_study"] == 1)
            & (study_df["last_t"] > t)
        )

        # Find users at following time who will be continuing users
        cont_user_next_bool = (
            (study_df["calendar_t"] == t + 1)
            & (study_df["in_study"] == 1)
            & (study_df["entry_t"] < t + 1)
        )

        actions = study_df[cont_user_current_bool]["action"].to_numpy()
        rewards = study_df[cont_user_current_bool]["reward"].to_numpy()

        # Form dosage update
        gamma = 0.95
        norm_gamma = 1 / (1 - gamma)
        prev_dosage = study_df[cont_user_current_bool]["dosage"].to_numpy()
        new_dosage = (actions + gamma * norm_gamma * prev_dosage) / norm_gamma
        new_dosage = new_dosage.reshape(-1, 1)

        # Make state features for next decision time
        gen_feats_action_names = [
            x
            for x in self.gen_feats
            if "reward" not in x and "dosage" not in x and "action" in x
        ]
        # slice gets rid of intercept
        get_past_actions_names = gen_feats_action_names[1:] + ["action"]
        past_actions = study_df[cont_user_current_bool][
            get_past_actions_names
        ].to_numpy()

        rewards_notflat = np.reshape(rewards, (-1, 1))
        state_vals = np.hstack(
            [past_actions, new_dosage, rewards_notflat, rewards_notflat * past_actions]
        )

        gen_feats_reward_action_names = [x + "_reward" for x in gen_feats_action_names]
        state_names = (
            gen_feats_action_names
            + ["dosage", "past_reward"]
            + gen_feats_reward_action_names
        )

        study_df.loc[cont_user_next_bool, state_names] = state_vals

        return study_df

    # TODO: This doesn't work with n = 1
    def sample_rewards(self, curr_timestep_data, actions, t):
        """Generate "random" rewards from saved noise"""
        main_covariates = curr_timestep_data[self.gen_feats]
        TE_covariates = main_covariates.to_numpy() * actions.reshape(-1, 1)
        gen_covariates = np.hstack([main_covariates, TE_covariates])

        if len(self.env_params.shape) == 2:
            params = self.env_params[t - 1]
        else:
            params = self.env_params

        reward_means = np.matmul(gen_covariates, params)
        rewards = (
            reward_means
            + self.reward_noise[curr_timestep_data["in_study_row_index"].to_numpy()]
        )

        return rewards

    def make_empty_study_df(self, args, user_df):
        base_cols = [
            "user_id",
            "policy_num",
            "last_t",
            "entry_t",
            "calendar_t",
            "action1prob",
            "intercept",
            "action",
            "reward",
        ]
        study_df = make_base_study_df(
            args, all_cols=base_cols + [x for x in self.gen_feats if x not in base_cols]
        )
        _, last_times, _ = get_entry_last_times(args)
        self.calendar_T = max(last_times)

        # initialize values
        study_df["intercept"] = 1
        first_entry_bool = study_df["calendar_t"] == study_df["entry_t"]

        zero_cols = [x for x in self.gen_feats if x not in base_cols]
        study_df.loc[first_entry_bool, zero_cols] = 0

        initial_past_rewards = self.rng.normal(0, 0.5, size=args.n)
        study_df.loc[first_entry_bool, "past_reward"] = initial_past_rewards
        study_df.loc[first_entry_bool, "past_action_1"] = 0
        study_df.loc[first_entry_bool, "past_action_1_reward"] = 0
        study_df.loc[first_entry_bool, "dosage"] = 0

        return study_df
