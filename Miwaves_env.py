import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle as pkl
import copy

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


def make_base_study_df(args, rng_seed=None, all_cols=None, df=None, sample_users=None):
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

    ### pre-treatment features for partial linear regression: pretreat_features1, pretreat_features2
    indices = np.arange(2)
    cov_matrix = 0.5 ** (np.abs(indices[:, None] - indices[None, :]))  # [2,2]
    mean = np.zeros(2)
    pre_treat_features = rng_seed.multivariate_normal(mean=mean, cov=cov_matrix, size=args.n)
    study_df['pretreat_feature1'] = np.repeat(pre_treat_features[:,0], max_calendar_t) # [1, 2, 3] -> [1, 1, 1, 2, 2, 2, 3, 3, 3]
    study_df['pretreat_feature2'] = np.repeat(pre_treat_features[:,1], max_calendar_t)

    ### Miwaves: state features for the study df
    study_df["user_id_SARA"] = np.repeat(sample_users, max_calendar_t) # u1, u1, u1, u2, u2, u2, u3, u3, u3 ...
    assert max_calendar_t % 2 == 0
    study_df["day"] = study_df["calendar_t"].apply(lambda x: (x - 1) // 2 + 1)
    study_df["time_of_day"] = study_df["calendar_t"].apply(lambda x: (x - 1) % 2) # 0: morning, 1: evening

    study_df["S1"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["S2"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["S3"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["S1_next"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["S2_next"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["S3_next"] = np.repeat(np.nan, max_calendar_t * args.n)
    study_df["dosage"] = np.repeat(np.nan, max_calendar_t * args.n)

    return study_df



def add_signal(
    user_models: dict, index: int, scale_factor: float = 1.0, simple_scale: bool = False
):
    """
    Adds a signal to the user models
    """
    for user in user_models.keys():
        real_weights = user_models[user].coef_[:, index]
        weights = np.sort(user_models[user].coef_[:, index])
        classes = user_models[user].classes_.astype(int)

        if simple_scale:
            user_models[user].coef_[:, index] = real_weights * scale_factor
            continue

        if 0 in classes:
            if len(classes) == 2:
                user_models[user].coef_[:, index] = np.absolute(weights) * scale_factor
            else:
                # First swap the weight of 0 with the minimum weight
                min_weight = weights[0]
                min_weight_index = np.where(real_weights == min_weight)[0][0]

                # Swap the weights
                real_weights[0], real_weights[min_weight_index] = (
                    real_weights[min_weight_index],
                    real_weights[0],
                )

                # Set the weights of 2 and 3 to be the average of both
                if 2 in classes and 3 in classes:
                    weight_2 = real_weights[np.where(classes == 2)[0][0]]
                    weight_3 = real_weights[np.where(classes == 3)[0][0]]
                    avg_weight = (weight_2 + weight_3) / 2
                    real_weights[np.where(classes == 2)[0][0]] = avg_weight
                    real_weights[np.where(classes == 3)[0][0]] = avg_weight

                # Update the weights
                user_models[user].coef_[:, index] = real_weights * scale_factor

        elif 2 in classes and 3 in classes:
            if len(classes) == 2:
                user_models[user].coef_[:, index] = 0.0
            else:
                # Set the weights of 2 and 3 to be the average of both
                weight_2 = real_weights[np.where(classes == 2)[0][0]]
                weight_3 = real_weights[np.where(classes == 3)[0][0]]
                avg_weight = (weight_2 + weight_3) / 2
                real_weights[np.where(classes == 2)[0][0]] = avg_weight
                real_weights[np.where(classes == 3)[0][0]] = avg_weight

                # Update the weights
                user_models[user].coef_[:, index] = real_weights * scale_factor
        else:
            user_models[user].coef_[:, index] = real_weights * scale_factor

    return user_models


def add_signal_user_specific(
    user_models: dict,
    index: int,
    scale_factor: list = None,
):
    """
    Adds user specific signals to the user models
    """
    for user in user_models.keys():
        real_weights = user_models[user].coef_[:, index]
        weights = np.sort(user_models[user].coef_[:, index])
        classes = user_models[user].classes_.astype(int)
        uid = 0

        if 0 in classes:
            if len(classes) == 2:
                user_models[user].coef_[:, index] = (
                    np.absolute(weights) * scale_factor[uid]
                )
            else:
                # First swap the weight of 0 with the minimum weight
                min_weight = weights[0]
                min_weight_index = np.where(real_weights == min_weight)[0][0]

                # Swap the weights
                real_weights[0], real_weights[min_weight_index] = (
                    real_weights[min_weight_index],
                    real_weights[0],
                )

                # Set the weights of 2 and 3 to be the average of both
                if 2 in classes and 3 in classes:
                    weight_2 = real_weights[np.where(classes == 2)[0][0]]
                    weight_3 = real_weights[np.where(classes == 3)[0][0]]
                    avg_weight = (weight_2 + weight_3) / 2
                    real_weights[np.where(classes == 2)[0][0]] = avg_weight
                    real_weights[np.where(classes == 3)[0][0]] = avg_weight

                # Update the weights
                user_models[user].coef_[:, index] = real_weights * scale_factor[uid]

        elif 2 in classes and 3 in classes:
            if len(classes) == 2:
                user_models[user].coef_[:, index] = 0.0
            else:
                # Set the weights of 2 and 3 to be the average of both
                weight_2 = real_weights[np.where(classes == 2)[0][0]]
                weight_3 = real_weights[np.where(classes == 3)[0][0]]
                avg_weight = (weight_2 + weight_3) / 2
                real_weights[np.where(classes == 2)[0][0]] = avg_weight
                real_weights[np.where(classes == 3)[0][0]] = avg_weight

                # Update the weights
                user_models[user].coef_[:, index] = real_weights * scale_factor[uid]
        else:
            user_models[user].coef_[:, index] = real_weights * scale_factor[uid]

        # Shrink the other coefficients as well in proportion
        if scale_factor[uid] < 1.0:
            user_models[user].coef_[:, index + 1 :] = (
                user_models[user].coef_[:, index + 1 :] * scale_factor[uid]
            )
        
        uid += 1

    return user_models



def format_data_for_prediction(data, action, day, start_day, dosage: float = None):
    """
    Formats data for prediction
    """
    X = pd.DataFrame(
        data[
            [
                "intercept",
                "engagement",
                "std_app_usage",
                "std_cannabis_use",
                "weekend",
                "std_day",
            ]
        ].astype(float)
    )
    X["weekend"] = 1 if (day + start_day) % 7 >= 5 else 0

    # Add columns to the data
    X["act_engagement"] = action * X["engagement"]
    X["act_std_app_usage"] = action * X["std_app_usage"]
    X["act_std_cannabis_use"] = action * X["std_cannabis_use"]
    X["act_std_day"] = action * X["std_day"]
    X["act_weekend"] = action * X["weekend"]
    X["act_intercept"] = action * X["intercept"]

    if dosage is not None:
        X["dosage"] = dosage

        # Format data again
        X = X[
            [
                "intercept",
                "engagement",
                "std_app_usage",
                "std_cannabis_use",
                "weekend",
                "std_day",
                "act_intercept",
                "act_engagement",
                "act_std_app_usage",
                "act_std_cannabis_use",
                "act_weekend",
                "act_std_day",
                "dosage"
            ]
        ].astype(float)
    else:
        # Format data again
        X = X[
            [
                "intercept",
                "engagement",
                "std_app_usage",
                "std_cannabis_use",
                "weekend",
                "std_day",
                "act_intercept",
                "act_engagement",
                "act_std_app_usage",
                "act_std_cannabis_use",
                "act_weekend",
                "act_std_day"
            ]
        ].astype(float)

    return X


def predict_probabilities(user_model, data, dropout_factor: float):
    """Predicts probabilities over reward distribution given some data"""
    """
    Reward function is defined by the user_model, i.e., weights
    """
    classes = user_model.classes_ # subset of [0, 1, 2, 3]

    flag = 0

    final_weights = []


    ##### construct a new weight: dosage_weight in the user model
    if len(classes) == 2:
        weights_list = list(user_model.coef_[0])
        weights = np.array(weights_list)
        dosage_weight = weights[:6].sum() / dropout_factor

        if dosage_weight > 0:
            dosage_weight = -dosage_weight
        weights_list.append(dosage_weight)
        final_weights = weights_list
    else:
        for i in range(len(classes)):
            weights_list = list(user_model.coef_[i])
            weights = np.array(weights_list)
            dosage_weight = weights[:6].sum() / dropout_factor

            if flag == 1:
                dosage_weight = -dosage_weight

            if i == 0 and dosage_weight < 0:
                dosage_weight = -dosage_weight
                flag = 1
            
            weights_list.append(dosage_weight) # 12 -> 13
            final_weights.append(weights_list) # num_classes x 13

    final_weights = np.array(final_weights)
    decision = np.dot(data, final_weights.T) + user_model.intercept_
    if len(classes) == 2:
        decision_vector = [[-decision[0], decision[0]]]
    else:
        decision_vector = decision
    probabilities = np.exp(decision_vector) / np.exp(decision_vector).sum() # data/state+action -> rewards distribution over actions
    return probabilities

class MiwavesEnv:
    def __init__(self, args, env_seed):
        """
        - select users and output parameters for selected users
        - make df (dataframe) with all variables collected in study (many missing)
            including entry and last decision times for each user
        """
        self.env_seed = env_seed
        self.rng = np.random.default_rng(env_seed)

        self.T = args.T # decision times
        self.n = args.n
        # Set during the run
        self.calendar_T = None

      
        ## highlighted hyper-parameters for Miwaves environment
        
        self.start_day = 0
        self.dosage_lookback = 6
        self.hyper_param_update_cadence = "weekly"
        self.posterior_update_cadence = "daily"
        self.dropout_percentage = 100.0
        self.act_cost_threshold = 0.1
        self.LOW_MULTIPLIER = 0.7
        self.HIGH_MULTIPLIER = 2.5
        self.FEATURE_ORDER = [
            "intercept",
            "engagement",
            "std_app_usage",
            "std_cannabis_use",
            "weekend",
            "std_day",
            "act_intercept",
            "act_engagement",
            "act_std_app_usage",
            "act_std_cannabis_use",
            "act_weekend",
            "act_std_day",
        ]
        self.decay_factor = 1.0

        ###### the following two are important
        
        self.dropout = args.Miwaves_habituation # -1: no habituation, 1: high habituation, 6: low habituation
        # args.Miwaves_treatmenteffect # 0: no treatment effect, 1: low treatment effect, 2: high treatment effect
        if args.Miwaves_treatmenteffect ==1:
            self.tx_effect_env = "overall_low"  # "none", "overall_low", "overall_high"
        elif args.Miwaves_treatmenteffect == 2:
            self.tx_effect_env = "overall_high"  # "none", "overall_low", "overall_high"
        else:
            self.tx_effect_env = "none"  # "none", "overall_low", "overall_high"

        # environment setup
        self.user_models = self.load_user_models() # user_models define the reward function of the user environment and then the next state
        
        index = self.FEATURE_ORDER.index("act_intercept")
        if self.tx_effect_env == "overall_low":
            self.user_models_ref = add_signal(
                self.user_models, index, scale_factor=self.LOW_MULTIPLIER
            )
        elif self.tx_effect_env == "overall_high":
            self.user_models_ref = add_signal(
                self.user_models, index, scale_factor=self.HIGH_MULTIPLIER
            )
        self.df = self.load_data()
        self.list_of_users = self.df["user_id_SARA"].unique()
        self.sampled_users = self.rng.choice(self.list_of_users, size=self.n, replace=True)
        num_dropout_users = int(self.n * (self.dropout_percentage / 100))
        self.dropout_users = self.rng.choice(list(range(self.n)), num_dropout_users, replace=False) # resample all users without 

        # self.dosage = np.zeros(self.n)  # initial dosage for all users
        discount_factor = float(self.dosage_lookback - 1) / self.dosage_lookback
        normalizing_factor = (1 - discount_factor) / (1 - discount_factor ** self.dosage_lookback)
        self.dosage_weights = np.array([normalizing_factor * (discount_factor ** i) for i in range(self.dosage_lookback)])    

        # self.full_df = pd.DataFrame()

        #### this dict is used to do the state transtion
        self.user_data = {}
        for i in range(self.n):
            self.user_data[i] = {
                "features": [{}],
                "state": [[0, 0, 1]],  # default initial starting state
                # "design_state": [],
                "action": [],
                "act_prob": [],
                "reward": [],
                "real_reward": [],
            }
    
    def create_state(self, user, time_of_day):
        """Create the state vector for the given features and decision point index
        time_of_day: 0 or 1
        """
        avg_reward = np.mean(self.user_data[user]["reward"][-3:])

        # create S1: recent engagement
        if avg_reward >= 2:
            S1 = 1
        else:
            S1 = 0

        # create S2 based on time of day
        S2 = time_of_day

        # create S3 based on last cannabis use
        # if user did not cannabis in the past decision point, S3 = 1
        if self.user_data[user]["features"][-1]["cannabis_use"] == 0:
            S3 = 1
        # if user used cannabis in the past decision point, S3 = 0
        # if the user did not report cannabis use in the past decision point, S3 = 0
        else:
            S3 = 0

        return [S1, S2, S3]


    def load_user_models(self) -> LogisticRegression:
        USERMODEL_PATH = "./Miwaves/MLR.pkl"
        # Load user models
        with open(USERMODEL_PATH, "rb") as f:
            user_models = pkl.load(f)
        return user_models


    def load_data(self):
        """
        The loaded df is used for the reward function 
        """
        CB_MEAN = 1.3
        CB_STD = 1.35
        APP_USE_MEAN = 350
        APP_USE_STD = 350
        DAY_MEAN = 15.5
        DAY_STD = 14.5
        def _normalize(df):
            df["std_cannabis_use"] = (df["cannabis_use"] - CB_MEAN) / CB_STD
            df["std_app_usage"] = (df["time_spent"] - APP_USE_MEAN) / APP_USE_STD
            df["std_day"] = (df["day"] - DAY_MEAN) / DAY_STD

            df["engagement"] = df["IsSurveyCompleted"].astype(float)
            df["std_day"] = df["std_day"].astype(float)
            df["std_cannabis_use"] = df["std_cannabis_use"].astype(float)
            df["std_app_usage"] = df["std_app_usage"].astype(float)
            df["weekend"] = df["weekend"].astype(float)
            df["intercept"] = 1.0
            return df
        
        # Load data
        PATH = "./Miwaves/combined_data.csv"
        df = pd.read_csv(PATH)
        df = _normalize(df)

        ###### change
        assert "user_id" == df.columns[0]
        df.rename(columns={df.columns[0]: "user_id_SARA"}, inplace=True)
        return df


    def update_study_df(self, study_df, t):
        """
        update next state [S1, S2, S3] in study_df
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

        # actions = study_df[cont_user_current_bool]["action"].to_numpy()
        # rewards = study_df[cont_user_current_bool]["reward"].to_numpy()

        # Form dosage update
        # gamma = 0.95
        # norm_gamma = 1 / (1 - gamma)
        # prev_dosage = study_df[cont_user_current_bool]["dosage"].to_numpy()
        # new_dosage = (actions + gamma * norm_gamma * prev_dosage) / norm_gamma
        # new_dosage = new_dosage.reshape(-1, 1)

        # # Make state features for next decision time
        # gen_feats_action_names = [ # past_action_1
        #     x
        #     for x in self.gen_feats #['intercept', 'past_action_1', 'past_reward', 'past_action_1_reward', 'dosage']
        #     if "reward" not in x and "dosage" not in x and "action" in x
        # ]
        # # slice gets rid of intercept
        # get_past_actions_names = gen_feats_action_names[1:] + ["action"] # ['action]
        # past_actions = study_df[cont_user_current_bool][
        #     get_past_actions_names
        # ].to_numpy()

        # rewards_notflat = np.reshape(rewards, (-1, 1))
        # state_vals = np.hstack(
        #     [past_actions, new_dosage, rewards_notflat, rewards_notflat * past_actions]
        # )

        # gen_feats_reward_action_names = [x + "_reward" for x in gen_feats_action_names]
        # state_names = (
        #     gen_feats_action_names
        #     + ["dosage", "past_reward"]
        #     + gen_feats_reward_action_names
        # ) #  ['past_action_1', 'dosage', 'past_reward', 'past_action_1_reward']

        # study_df.loc[cont_user_next_bool, state_names] = state_vals

        ## update S1, S2, S3
        temp_df = study_df.loc[cont_user_current_bool, :]
        next_state = []
        # breakpoint()
        for i in range(temp_df.shape[0]):
            assert temp_df.iloc[i]["time_of_day"] == ((t-1) % 2) # t=1, -> 0, 1 
            # next_state.append(self.create_state(i, (t+1) % 2))
            next_state.append(self.create_state(i, t % 2)) # need to flip the time of day
        next_state = np.vstack(next_state)
        study_df.loc[cont_user_next_bool, ["S1", "S2", "S3"]] = next_state
        study_df.loc[cont_user_current_bool, ["S1_next", "S2_next", "S3_next"]] = next_state
        
        
        ## update dosage: for each user at the next time step, compute dosage based on last 6 actions for the current time step
        dosage_bool = (
            (study_df["calendar_t"] <= t)
            & (study_df["in_study"] == 1)
            & (study_df["last_t"] > t)
        )
        user_ids = study_df[dosage_bool]["user_id"].unique()
        for user in user_ids:
            dosage_user_bool = dosage_bool & (study_df["user_id"] == user)
            last_6_actions = study_df.loc[dosage_user_bool].sort_values("calendar_t")["action"].values[-6:] # may less than 6
            study_df.loc[cont_user_next_bool & (study_df["user_id"] == user), "dosage"] = np.sum(last_6_actions * self.dosage_weights[: len(last_6_actions)])
        return study_df
    
        
    # def update_data(self, ):
    #     for i in range(self.n):
    #         self.user_data[i]["features"].append(data[i])

    ##### Miwave environment
    # 
    # def compute_reward(self, user):
    #     """Create the reward given the features for a given time index [return back from features to the reward]"""
    #     user_data = self.user_data[user]
    #     reward = 0
    #     features = user_data["features"][-1]
    #     if features["survey_completion"] == 1:
    #         if features["activity_question"] == 1:
    #             reward = 3
    #         else:
    #             reward = 2
    #     elif features["app_usage"] == 1:
    #         reward = 1
    #     return reward

    def action_cost(self, user):
        """Return the action cost for a given user"""
        user_data = self.user_data[user]
        # Check the last action
        last_action = user_data["action"][-1]

        # If the last action was 1, return the cost
        if last_action == 1:
            all_rewards = user_data["real_reward"]
            std_all_rewards = np.std(all_rewards, ddof=1)

            if np.isnan(std_all_rewards):
                std_all_rewards = 1.12

            return self.act_cost_threshold * std_all_rewards

        # If the last action was 0, return 0
        else:
            return 0
        
    def end_decision_point(self, t, rewards, action, action_prob):
        """
        update the self.user_data in the Miwaves environment
        """
        decision_point = t - 1
        time_of_day = decision_point % 2
        for i in range(self.n):
            self.user_data[i]["features"].append(self.new_data[i])
            self.user_data[i]["action"].append(action[i])
            self.user_data[i]["act_prob"].append(action_prob[i])
            self.user_data[i]["real_reward"].append(rewards[i]) # original reward
            self.user_data[i]["reward"].append(rewards[i] - self.action_cost(i)) # received rewards for the algorithm update
            self.user_data[i]["state"].append(self.create_state(i, 1-time_of_day)) # next state !!!!!!!!!! flip the time of day

    # TODO: This doesn't work with n = 1
    def _sample_rewards_base(self, curr_timestep_data, actions, t):

        """reward parameters determined by the user model"""
        self.user_models = copy.deepcopy(self.user_models_ref)
        # day = (t - 1) // 2 + 1
       
        num_users = curr_timestep_data['user_id'].nunique()
        reward_all = np.zeros(num_users)
        decision_point = t - 1 # 2*day + time_of_day

        self.new_data = [] # save features of all users and then save to the self.user_data in end_decision_point()

        for user in range(num_users):
            action = actions[user]
            
            curr_timestep_data_user = curr_timestep_data[curr_timestep_data['user_id'] == user + 1] # by index 0, 1, 2, ...
            time_of_day = curr_timestep_data_user['time_of_day'].values[0]
            day = curr_timestep_data_user['day'].values[0]
            user_id_SARA = curr_timestep_data_user['user_id_SARA'].values[0]
            assert self.sampled_users[user] == user_id_SARA
            dosage = curr_timestep_data_user['dosage'].values[0]
            user_model = self.user_models[user_id_SARA]
            dp_data = pd.DataFrame(
                    self.df[
                        (self.df["day"] == day)
                        & (self.df["time_of_day"] == time_of_day)
                        & (self.df["user_id_SARA"] == user_id_SARA)
                    ]
                )

            if self.dropout > 0 and user in self.dropout_users:
                X = format_data_for_prediction(dp_data, action, day, self.start_day, dosage)
                probabilities = predict_probabilities(user_model, X, self.dropout)[0]
            else: # no habituation
                X = format_data_for_prediction(dp_data, action, day, self.start_day)
                probabilities = user_model.predict_proba(X)[0]
                
            reward = self.rng.choice(user_model.classes_, p=probabilities)
            reward_all[user] = reward

            # update the data
            expected_reward = (
                np.array(probabilities)
                .ravel()
                .dot(np.array(user_model.classes_).ravel())
            )
            if reward == 3:
                activity_question = 1
                survey_completion = 1
                app_usage_flag = 1
            elif reward == 2:
                activity_question = 0
                survey_completion = 1
                app_usage_flag = 1
            elif reward == 1:
                activity_question = None
                survey_completion = 0
                app_usage_flag = 1
            else:
                activity_question = None
                survey_completion = 0
                app_usage_flag = 0
            new_data = {
                "user": user, # number: e.g., 0
                "decision_point": decision_point, # calendar_t - 1
                "day": day,
                "time_of_day": dp_data["time_of_day"].values[0],
                "app_usage": app_usage_flag,
                "cannabis_use": dp_data["cannabis_use"].values[0]
                if reward > 1
                else None,
                "survey_completion": survey_completion,
                "activity_question": activity_question,
                "action": action, # one scalar
                "expected_reward": expected_reward,
            }
            self.new_data.append(new_data)
            
             
        return reward_all
    
    def sample_rewards(self, curr_timestep_data, actions, t):
        reward_means = self._sample_rewards_base(curr_timestep_data, actions, t)
        rewards = (
            reward_means
        )
        return rewards

    def sample_rewards_prefeatures(self, curr_timestep_data, actions, t, alpha1, alpha2):
        reward_means = self._sample_rewards_base(curr_timestep_data, actions, t)
        rewards = (
            reward_means
            + alpha1 * curr_timestep_data['pretreat_feature1'].to_numpy()
            + alpha2 * curr_timestep_data['pretreat_feature2'].to_numpy()
        )
        return rewards


    def make_empty_study_df(self, args, user_df):
        base_cols = [
            "user_id",
            "user_id_SARA",
            "policy_num",
            "last_t",
            "entry_t",
            "calendar_t",
            "action1prob",
            "intercept",
            "action",
            "reward",
            ### new miwaves features
            "day",
            # "cannabis_use",
            # "time_spent",
            # "IsSurveyCompleted",
            # "weekend",
            "time_of_day",
            # "std_cannabis_use",
            # "std_app_usage",
            # "std_day",
            # "engagement",
            "S1",
            "S2",
            "S3",
            "S1_next",
            "S2_next",
            "S3_next",
            "dosage",
        ]
        study_df = make_base_study_df(
            args, self.rng,all_cols=base_cols, df=self.df, sample_users=self.sampled_users
        )
        _, last_times, _ = get_entry_last_times(args)
        self.calendar_T = max(last_times)

        # initialize values
        study_df["intercept"] = 1
        first_entry_bool = study_df["calendar_t"] == study_df["entry_t"]

        # zero_cols = [x for x in self.gen_feats if x not in base_cols]
        # study_df.loc[first_entry_bool, zero_cols] = 0

        # initial_past_rewards = self.rng.normal(0, 0.5, size=args.n)
        # study_df.loc[first_entry_bool, "past_reward"] = initial_past_rewards
        # study_df.loc[first_entry_bool, "past_action_1"] = 0
        # study_df.loc[first_entry_bool, "past_action_1_reward"] = 0
        # study_df.loc[first_entry_bool, "dosage"] = 0

        # state features: the same as Miwave simulator
        study_df.loc[first_entry_bool, "S1"] = 0
        study_df.loc[first_entry_bool, "S2"] = 0 # time_of_day = 0
        study_df.loc[first_entry_bool, "S3"] = 1  
        study_df.loc[first_entry_bool, "dosage"] = 0
        
        return study_df
