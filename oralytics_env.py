import numpy as np
import pandas as pd
import scipy


def get_entry_last_times(args):
    # Compute entry / last decision times compared to calendar for each user

    assert args.n % args.recruit_n == 0
    distinct_entry_times = int(args.n / args.recruit_n)
    all_entry_times = [
        i * args.decisions_between_updates * args.recruit_t
        for i in range(distinct_entry_times)
    ]
    entry_times = (
        np.repeat(all_entry_times, args.recruit_n) + 1
    )  # Vector of entry decision times for all users
    last_times = (
        np.repeat(all_entry_times, args.recruit_n) + args.T
    )  # Vector of last decision times for all
    return entry_times, last_times


def load_oralytics_env(
    paramf_path="./oralytics_env_params/non_stat_zero_infl_pois_model_params.csv",
):
    """
    Goal:
        - Load synethetic model parameters
    Input:
        - `env_path`: String path to directory with environment parameters/data
    Output:
        - `param_vec`: Vector of parameters
    """

    # Load Model Parameters ###############################################

    params_df = pd.read_csv(paramf_path)
    params_df.drop(columns=[params_df.columns[0], "User"], inplace=True)

    # Bernoulli Parameters
    bern_col_names = [
        "Intercept.Bern",
        "Time.of.Day.Bern",
        "Day.Type.Bern",
        "Day.in.Study.norm.Bern",
        "Prior.Day.Total.Brush.Time.norm.Bern",
        "Proportion.Brushed.In.Past.7.Days.Bern",
    ]
    bernoulli_params = params_df[bern_col_names].to_numpy()

    # Poisson Parameters
    poisson_col_names = [
        "Intercept.Poisson",
        "Time.of.Day.Poisson",
        "Day.Type.Poisson",
        "Day.in.Study.norm.Poisson",
        "Prior.Day.Total.Brush.Time.norm.Poisson",
        "Proportion.Brushed.In.Past.7.Days.Poisson",
    ]
    poisson_params = params_df[poisson_col_names].to_numpy()

    # Param names
    param_names = [
        "intercept",
        "time_of_day",
        "weekend",
        "day_in_study_norm",
        "prior_day_brush",
        "prop_brush_week",
    ]

    return param_names, bernoulli_params, poisson_params


class OralyticsEnv:
    def __init__(
        self,
        args,
        param_names,
        bernoulli_params,
        poisson_params,
        env_seed,
        errcorr="time_corr",
    ):
        """
        - select users and output parameters for selected users
        - make df (dataframe) with all variables collected in study (many missing)
            including entry and last decision times for each user
        """
        self.bernoulli_params = bernoulli_params
        self.poisson_params = poisson_params
        self.env_seed = env_seed
        self.rng = np.random.default_rng(env_seed)
        self.gen_feats = param_names
        self.treat_feats = [
            "intercept",
            "time_of_day",
            "weekend",
            "day_in_study_norm",
            "prior_day_brush",
        ]

    def update_study_df(self, study_df, t, rewards, actions, current_users):
        if t % 2 == 1:
            # check if it is an even decision time (means it is morning);
            # if so no need to update state
            return study_df

        # fill in other features
        new_features = []
        next_day_user_ids = study_df.loc[
            study_df["calendar_t"] == t + 1, "user_id"
        ].to_numpy()
        for user in next_day_user_ids:
            if user not in current_users.to_numpy():
                new_features.append([0, 0, 0, 0])
                continue

            user_study_df = study_df.loc[
                np.logical_and(study_df["user_id"] == user, study_df["calendar_t"] <= t)
            ]
            user_study_df_week = user_study_df[-14:]

            # prior_day_brush: Previous day brushing quality
            prevBrushQ_mean = 154
            prevBrushQ_std = 163
            norm_prevBrushQ = (
                np.sum(user_study_df_week[-2:]["reward"]) - prevBrushQ_mean
            ) / prevBrushQ_std

            # prop_brush_week: Proportion brushed in past week (check with Anna)
            prop_brush_week = np.mean(user_study_df_week["reward"] > 0)

            # prev_brush: Exponentially weighted average of past week brushing
            gamma = 13 / 14
            gamma_array = np.ones(13) * gamma
            exp_weights = np.flip(np.hstack([1, np.cumprod(gamma_array)]))
            c_gamma = (1 - gamma) / (1 - gamma**14)

            user_reward_week = user_study_df_week["reward"]
            prev_brush_raw = c_gamma * np.sum(
                exp_weights[-len(user_reward_week) :] * user_reward_week
            )
            prev_brush = (prev_brush_raw - (181 / 2)) / (179 / 2)

            # prev_message: Exponentially weighted average of messages sent in past week
            user_message_week = user_study_df_week["action"]
            prev_message = c_gamma * np.sum(
                exp_weights[-len(user_message_week) :] * user_message_week
            )

            new_features.append(
                [norm_prevBrushQ, prop_brush_week, prev_brush, prev_message]
            )

        col_vals = np.vstack(new_features)
        col_names = ["prior_day_brush", "prop_brush_week", "prev_brush", "prev_message"]

        study_df.loc[study_df["calendar_t"] == t + 1, col_names] = col_vals
        study_df.loc[study_df["calendar_t"] == t + 2, col_names] = col_vals

        return study_df

    def sample_rewards(self, curr_timestep_data, actions, t):
        current_users_idx = curr_timestep_data["user_id"].to_numpy() - 1

        bern_main = np.sum(
            self.study_bern_params[current_users_idx]
            * curr_timestep_data[self.gen_feats],
            axis=1,
        )
        poisson_main = np.sum(
            self.study_poisson_params[current_users_idx]
            * curr_timestep_data[self.gen_feats],
            axis=1,
        )

        bern_TE = np.sum(
            self.study_bern_TE[current_users_idx]
            * curr_timestep_data[self.treat_feats],
            axis=1,
        )
        poisson_TE = np.sum(
            self.study_poisson_TE[current_users_idx]
            * curr_timestep_data[self.treat_feats],
            axis=1,
        )

        # """
        # probability of intending to brush
        bern_p = 1 - scipy.special.expit(bern_main - actions * np.maximum(bern_TE, 0))
        poisson_lambda = np.exp(poisson_main + actions * np.maximum(poisson_TE, 0))
        # bern_p = 1-scipy.special.expit( bern_main - actions*bern_TE )
        # poisson_lambda = np.exp( poisson_main + actions*poisson_TE )

        brush_times = self.rng.binomial(1, bern_p) * self.rng.poisson(poisson_lambda)
        # rewards = np.minimum(brush_times, 180)
        rewards = np.sqrt(np.minimum(brush_times, 180))
        """
        rewards = self.rng.normal( poisson_main + poisson_TE )
        brush_times = rewards
        """

        return rewards, brush_times

    def make_empty_study_df(self, args, user_env_data):
        # Sample users with replacement
        n_robas = len(user_env_data["bern_params"])
        robas_id = self.rng.integers(0, high=n_robas, size=args.n)

        self.study_bern_params = user_env_data["bern_params"][robas_id]
        self.study_poisson_params = user_env_data["poisson_params"][robas_id]

        # Sample effect sizes ===================

        # mean and variance of effect sizes for 'time_of_day', 'weekend', 'day_in_study_norm', 'prior_day_brush'
        bern_abs_params = np.absolute(user_env_data["bern_params"][:, 1:-1]) / 100000
        poisson_abs_params = (
            np.absolute(user_env_data["poisson_params"][:, 1:-1]) / 100000
        )

        bern_mean = np.hstack([np.mean(bern_abs_params), np.mean(bern_abs_params, 0)])
        bern_std = np.hstack(
            [[np.std(np.mean(bern_abs_params, 0))], np.std(bern_abs_params, 0)]
        )
        poisson_mean = np.hstack(
            [np.mean(poisson_abs_params), np.mean(poisson_abs_params, 0)]
        )
        poisson_std = np.hstack(
            [[np.std(np.mean(poisson_abs_params, 0))], np.std(poisson_abs_params, 0)]
        )

        # changing signs of effect ('intercept', 'time_of_day', 'weekend', 'day_in_study_norm', 'prior_day_brush')
        bern_mean[3] *= -1
        poisson_mean[3] *= -1

        self.study_bern_TE = self.rng.multivariate_normal(
            bern_mean, np.diag(bern_std), size=args.n
        )
        self.study_poisson_TE = self.rng.multivariate_normal(
            poisson_mean, np.diag(poisson_std), size=args.n
        )

        # print( np.max(self.study_bern_TE), np.max(self.study_poisson_TE) )
        # import ipdb; ipdb.set_trace()

        # Get user entry and exit times ========================
        entry_times, last_times = get_entry_last_times(args)
        self.calendar_T = max(last_times)

        additional_cols = self.gen_feats + ["brush_time", "prev_brush", "prev_message"]
        df_fill = np.zeros((args.n * args.T, 3 + len(additional_cols)))
        df_fill.fill(np.nan)
        study_df = pd.DataFrame(
            df_fill, columns=["action", "reward", "action1prob"] + additional_cols
        )

        study_df.insert(0, "gen_id", np.repeat(robas_id, args.T))
        study_df.insert(0, "user_id", np.repeat(np.arange(1, args.n + 1), args.T))
        study_df.insert(1, "user_t", np.tile(np.arange(1, args.T + 1), args.n))
        study_df.insert(1, "policy_last_t", np.tile(np.nan, args.T * args.n))
        study_df.insert(1, "policy_num", np.tile(np.nan, args.T * args.n))
        study_df.insert(1, "last_t", np.repeat(last_times, args.T))
        study_df.insert(1, "entry_t", np.repeat(entry_times, args.T))
        study_df.insert(1, "calendar_t", study_df["entry_t"] + study_df["user_t"] - 1)

        study_df = study_df.reset_index().drop(columns="index")

        # initialize values known values
        assert args.T % 2 == 0  # morning and evening

        # assert args.T % 14 == 0  # some number of weeks
        num_weeks = np.ceil(args.T / 14)
        user_weeks = np.tile(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], int(num_weeks)
        )[: args.T]

        study_df["intercept"] = 1
        study_df["time_of_day"] = np.tile([0, 1], int(args.T / 2) * args.n)
        study_df["weekend"] = np.tile(user_weeks, args.n)
        study_df["day_in_study_norm"] = (study_df["user_t"] - (args.T + 1) / 2) / (
            (args.T - 1) / 2
        )
        # normalized betwen zero and one
        assert max(study_df["day_in_study_norm"]) == 1
        assert min(study_df["day_in_study_norm"]) == -1

        # initialize first state variables
        init_zero_vars = [
            "prior_day_brush",
            "prop_brush_week",
            "prev_brush",
            "prev_message",
        ]
        study_df.loc[study_df["user_t"] == 1, init_zero_vars] = 0
        study_df.loc[study_df["user_t"] == 2, init_zero_vars] = 0

        study_df.loc[study_df["user_t"] == 1, "prior_day_brush"] = self.rng.normal(
            size=args.n
        )
        study_df.loc[study_df["user_t"] == 2, "prior_day_brush"] = self.rng.normal(
            size=args.n
        )

        return study_df
