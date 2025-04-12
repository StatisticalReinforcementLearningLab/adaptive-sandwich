import numpy as np
import pandas as pd
import scipy

from least_squares_helper import get_utri, symmetric_fill_utri

def get_entry_last_times(args):
    # Compute entry / last decision times compared to calendar for each user

    assert args.n % args.recruit_n == 0
    distinct_entry_times = int(args.n / args.recruit_n)
    all_entry_times = [i*args.decisions_between_updates*args.recruit_t
                                        for i in range(distinct_entry_times)]
    entry_times = np.repeat( all_entry_times, args.recruit_n )+1
        # Vector of entry decision times for all users
    last_times = np.repeat( all_entry_times, args.recruit_n )+args.T
        # Vector of last decision times for all
    return entry_times, last_times


def make_base_study_df(args, all_cols=['user_id', 'user_t', 'policy_last_t', 'policy_num', 'last_t',
            'entry_t', 'calendar_t', 'in_study']):
    entry_times, last_times = get_entry_last_times(args)

    df_fill = np.zeros( (args.n*args.T, len(all_cols)) )
    df_fill.fill(np.nan)
    study_df = pd.DataFrame(df_fill, columns=all_cols)

    study_df['user_id'] = np.repeat( np.arange(1,args.n+1), args.T)
    study_df['user_t'] = np.tile( np.arange(1,args.T+1), args.n)
    study_df['policy_last_t'] = np.tile(np.nan, args.T*args.n)
    study_df['policy_num'] = np.tile(np.nan, args.T*args.n)
    study_df['last_t'] = np.repeat(last_times, args.T)
    study_df['entry_t'] = np.repeat(entry_times, args.T)
    study_df['calendar_t'] = study_df['entry_t']+study_df['user_t']-1

    # This is done only to interface with Nowell's package.
    # It suggests all users are always in the study, which
    # is all that is this branch of the code can fully handle,
    # even though the above has the start of support for# incremental
    # recruitment. I'll just add an assert that makes sure incremental
    # recruitment is not used.  If it is, this should be changed/removed.
    assert len(set(entry_times)) == 1, "Incremental recruitment not supported"
    assert len(set(last_times)) == 1, "Incremental recruitment not supported"
    study_df["in_study"] = 1

    study_df = study_df.reset_index().drop(columns='index')
    return study_df


def load_synthetic_env(paramf_path="./synthetic_env_params/delayed_effects.txt"):
    """
    Goal:
        - Load synethetic model parameters
    Input:
        - `env_path`: String path to directory with environment parameters/data
    Output:
        - `param_vec`: Vector of parameters
    """

    # Load Model Parameters ###############################################
    with open(paramf_path, 'r') as f:
        params = f.readlines()

    all_env_params = []
    for row in params:
        tmp = np.array([float(x) for x in row.strip().split(",")])
        all_env_params.append(tmp)

    #env_params = np.array([float(x) for x in params.strip().split(",")])
    return np.vstack(all_env_params).squeeze()


class SyntheticEnv:

    def __init__(self, args, env_params, env_seed, gen_feats, err_corr='time_corr'):
        """
        - select users and output parameters for selected users
        - make df (dataframe) with all variables collected in study (many missing)
            including entry and last decision times for each user
        """
        self.env_params = env_params
        self.env_seed = env_seed
        self.rng = np.random.default_rng(env_seed)
        self.args = args

        self.gen_feats = gen_feats
        self.err_corr = err_corr

        # Generate noise for rewards
        if err_corr == "time_corr":
            cov_matrix = np.eye(args.T)
            for i in range(args.T):
                for j in range(args.T):
                    if i == j:
                        continue
                    cov_matrix[i][j] = pow(0.5, abs(i-j)/2)
        elif err_corr == "independent":
            cov_matrix = np.eye(args.T)
        else:
            raise ValueError("Invalid noise correlation")

        # n x T
        noise = self.rng.multivariate_normal(mean = np.zeros(args.T),
                                                   cov=cov_matrix,
                                                   size=args.n)

        self.reward_noise = noise.flatten()


    #def update_study_df(self, study_df, t, rewards, actions, current_users):

    def update_study_df(self, study_df, t):
        # Find users who have already been in study and will continue on
        cont_user_current_bool = np.logical_and(study_df['calendar_t'] == t, \
                                                study_df['user_t'] < self.args.T)

        cont_user_next_bool = np.logical_and(study_df['calendar_t'] == t+1, \
                                             study_df['user_t'] > 1 ) # removes users who enter at t+1

        actions = study_df[cont_user_current_bool]['action'].to_numpy()
        rewards = study_df[cont_user_current_bool]['reward'].to_numpy()

        # Form dosage update
        gamma = 0.95
        norm_gamma = 1/(1-gamma)
        prev_dosage = study_df[ cont_user_current_bool ]['dosage'].to_numpy()
        new_dosage = ( actions + gamma*norm_gamma*prev_dosage ) / norm_gamma
        new_dosage = new_dosage.reshape(-1,1)

        # Make state features for next decision time
        gen_feats_action_names = [x for x in self.gen_feats \
            if 'reward' not in x and 'dosage' not in x and 'action' in x]
        get_past_actions_names = gen_feats_action_names[1:] + ['action']
        past_actions = study_df[ cont_user_current_bool ][get_past_actions_names].to_numpy()

        rewards_notflat = np.reshape(rewards, (-1,1))
        state_vals = np.hstack([past_actions, new_dosage, rewards_notflat, \
                rewards_notflat*past_actions])

        gen_feats_reward_action_names = [x+'_reward' for x in gen_feats_action_names ]
        state_names = gen_feats_action_names + ['dosage', 'past_reward'] + \
                gen_feats_reward_action_names


        study_df.loc[ cont_user_next_bool, state_names ] = state_vals

        return study_df


    def sample_rewards(self, curr_timestep_data, actions, t):
        main_covariates = curr_timestep_data[self.gen_feats]
        TE_covariates = main_covariates.to_numpy() * actions.reshape(-1,1)
        gen_covariates = np.hstack( [main_covariates, TE_covariates] )

        if len(self.env_params.shape) == 2:
            params = self.env_params[t-1]
        else:
            params = self.env_params

        reward_means = np.matmul(gen_covariates, params)
        rewards = reward_means + self.reward_noise[ curr_timestep_data.index.to_numpy() ]

        return rewards


    def make_empty_study_df(self, args, user_df):
        base_cols = ['user_id', 'user_t', 'policy_last_t', 'policy_num', 'last_t',
                'entry_t', 'calendar_t', 'action1prob', 'intercept', 'action', 'reward',]
        study_df = make_base_study_df(args,
                all_cols = base_cols + [ x for x in self.gen_feats if x not in base_cols ])
        entry_times, last_times = get_entry_last_times(args)
        self.calendar_T = max(last_times)

        # initialize values
        study_df['intercept'] = 1
        zero_col = [ x for x in self.gen_feats if x not in base_cols ]
        study_df.loc[ study_df['calendar_t'] == 1, zero_col ] = 0
        initial_past_rewards = self.rng.normal(0,0.5, size=args.n)
        study_df.loc[ study_df['user_t'] == 1, 'past_reward' ] = \
                initial_past_rewards
        study_df.loc[ study_df['user_t'] == 1, 'past_action_1' ] = 0
        study_df.loc[ study_df['user_t'] == 1, 'past_action_1_reward' ] = 0
        study_df.loc[ study_df['user_t'] == 1, 'dosage' ] = 0

        return study_df
