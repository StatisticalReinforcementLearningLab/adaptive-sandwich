#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import pandas as pd
import numpy as np
import pickle as pkl
import os
import scipy.stats as stats
import scipy.special as special
import copy
import argparse

# from BayesianLinearMixedEffects import BayesianLinearMixedEffectsRegression
from BayesianLinearMixedEffectsRD import BayesianLinearMixedEffectsRegressionRD
from BayesianLinearRegressionRD import BayesianLinearRegressionRD
from FlatProb import FlatProb
from priors import PRIOR_MEAN, PRIOR_VAR, INIT_COV, PRIOR_NOISE_VAR
#from priors_RD import PRIOR_MEAN, PRIOR_VAR, INIT_COV, PRIOR_NOISE_VAR

from typing import Callable

from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression

# In[3]:


# Paths
PATH = "./combined_data.csv"
USERMODEL_PATH = "./MLR.pkl"
SIMULATION_FOLDER = "./data/simulations/"
RANDOMVARS_PATH = './randomvars.pkl'
# SIMULATION_FOLDER = "./simulations/"

#PATH = "./../../../data/dataset/combined_data.csv"
#USERMODEL_PATH = "./../../../models/usermodels/MLR.pkl"
#SIMULATION_FOLDER = "./../../../data/simulations/"
#RANDOMVARS_PATH = "./../../../models/randomvars/randomvars.pkl"

# Normalization constants

CB_MEAN = 1.3
CB_STD = 1.35

APP_USE_MEAN = 350
APP_USE_STD = 350

DAY_MEAN = 15.5
DAY_STD = 14.5


# In[4]:


# These will later be inputs/parameter when using the file as script for running simulations
NUM_SIMS = 500
NUM_DAYS = 30
NUM_USERS = 120

L_MIN = 0.2
L_MAX = 0.8
LOGISTIC_B = 20
LOGISTIC_SIGMA = 0.95
LOGISTIC_C = 5

LOW_MULTIPLIER = 0.7
HIGH_MULTIPLIER = 2.5

START_DAY = 0  # 0 = monday, 1 = tuesday, 2 = wednesday, 3 = thursday, 4 = friday, 5 = saturday, 6 = sunday

FEATURE_ORDER = [
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

# In[5]:


def normalize(df):
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


# In[6]:


def load_data():
    # Load data
    df = pd.read_csv(PATH)

    # Normalize df
    df = normalize(df)

    return df


# In[7]:


def load_user_models() -> LogisticRegression:
    # Load user models
    with open(USERMODEL_PATH, "rb") as f:
        user_models = pkl.load(f)

    return user_models


# In[8]:


def load_random_vars() -> np.array:
    # Load random variables - normal with mean 0 and var 1
    with open(RANDOMVARS_PATH, "rb") as f:
        random_vars = pkl.load(f)

    return random_vars


# In[8]:


def get_allocation_function(
    func_type: str,
    B: float,
    randomvars: np.array,
    C: float = 5.0,
    L_min: float = 0.2,
    L_max: float = 0.8,
) -> Callable:
    """
    Gets the allocation function to be used for the run
    """

    def thompson_sampling(mean: float, var: float) -> float:
        """
        Simple thompson sampling allocation function
        """
        prob = 1 - stats.norm.cdf(0, mean, np.sqrt(var))
        return prob

    def logistic_function_infinity(x: float) -> float:
        if x >= 0:
            return L_max
        else:
            return L_min

    def smooth_posterior_sampling_inf(mean: float, var: float) -> float:
        std = np.sqrt(var)
        samples = mean + (randomvars * std)
        prob = np.mean([logistic_function_infinity(i) for i in samples])

        return prob

    def logistic_function(x: float) -> float:
        numerator = L_max - L_min
        denominator_inverse = special.expit(B * x - np.log(C))
        return L_min + numerator * denominator_inverse

    def smooth_posterior_sampling(mean: float, var: float) -> float:
        std = np.sqrt(var)
        samples = mean + (randomvars * std)
        prob = np.mean(logistic_function(samples))

        # prob = stats.norm.expect(func=logistic_function, loc=mean, scale=np.sqrt(var))
        return prob

    if func_type == "thompson":
        return thompson_sampling
    elif func_type == "smooth":
        if np.isinf(B):
            return smooth_posterior_sampling_inf
        else:
            return smooth_posterior_sampling
    else:
        raise NotImplementedError


# In[17]:


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

    classes = user_model.classes_

    flag = 0

    final_weights = []

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
            
            weights_list.append(dosage_weight)
            final_weights.append(weights_list)

    final_weights = np.array(final_weights)
    
    decision = np.dot(data, final_weights.T) + user_model.intercept_
    if len(classes) == 2:
        decision_vector = [[-decision[0], decision[0]]]
    else:
        decision_vector = decision
    probabilities = np.exp(decision_vector) / np.exp(decision_vector).sum()

    return probabilities
# In[18]:


def exponential_decay(dosage: float, multiplier: float, scale_factor: float) -> float:
    """
    Exponential decay function for adjusting habituation multiplier
    """
    return multiplier * np.exp(-scale_factor * dosage)


# In[18]:


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


# In[18]:


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


# In[18]:


def run_simulation(
    num_users: int,
    num_days: int,
    start_day: int,
    seed: int,
    sim_path: str,
    rl_alg: str,
    rl_alg_variant: int,
    alloc_func: Callable,
    act_cost_threshold: float = 2.0,
    posterior_update_cadence: str = "daily",
    hyper_param_update_cadence: str = "daily",
    tx_effect_env: str = "none",
    habituation: str = None,
    habituation_factor: float = 0.0,
    dosage_lookback: int = 6,
    dropout: float = -1,
    dropout_percentage: float = 100.0,
    decay: bool = False,
    debug: bool = False,
):
    """
    Run a simulation - using an algorithm to sample actions
    """

    # Create random number generators
    rng = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    rng_algorithm = np.random.default_rng(seed)

    save_path = os.path.join(sim_path, str(seed))
    os.makedirs(save_path, exist_ok=True)

    # Initialize the RL algorithm

    if rl_alg == "random":
        rl_algorithm = FlatProb(
            num_users=num_users,
            starting_time_of_day=start_day,
            rng=rng_algorithm,
            debug=debug,
            logger_path=save_path,
        )
    elif rl_alg == "BLR":
        rl_algorithm = BayesianLinearRegressionRD(
            variant=rl_alg_variant,
            prior_mean=PRIOR_MEAN[rl_alg_variant],
            prior_cov=PRIOR_VAR[rl_alg_variant],
            noise_var=PRIOR_NOISE_VAR,
            alloc_func=alloc_func,
            num_users=num_users,
            rng=rng_algorithm,
            debug=debug,
            act_cost_threshold=act_cost_threshold,
        )
    elif rl_alg == "rebandit":
        rl_algorithm = BayesianLinearMixedEffectsRegressionRD(
            variant=rl_alg_variant,
            prior_mean=PRIOR_MEAN[rl_alg_variant],
            prior_cov=PRIOR_VAR[rl_alg_variant],
            init_cov_u=INIT_COV[rl_alg_variant],
            init_noise_var=PRIOR_NOISE_VAR,
            alloc_func=alloc_func,
            num_users=num_users,
            rng=rng_algorithm,
            debug=debug,
            logger_path=save_path,
            act_cost_threshold=act_cost_threshold,
        )

    # Load data
    df = load_data()

    # Load user models
    user_models_ref = load_user_models()
    user_models_ref_copy = copy.deepcopy(user_models_ref)
    morning_models = copy.deepcopy(user_models_ref)
    evening_models = copy.deepcopy(user_models_ref)

    # Check if signal needs to be added, and if yes, add it to the corresponding columns
    index = FEATURE_ORDER.index("act_intercept")
    if tx_effect_env == "overall_low":
        user_models_ref = add_signal(
            user_models_ref, index, scale_factor=LOW_MULTIPLIER
        )
    elif tx_effect_env == "overall_high":
        user_models_ref = add_signal(
            user_models_ref, index, scale_factor=HIGH_MULTIPLIER
        )
    elif tx_effect_env == "morning_low":
        morning_models = add_signal(morning_models, index, scale_factor=LOW_MULTIPLIER)
        evening_models = add_signal(evening_models, index, scale_factor=HIGH_MULTIPLIER)
    elif tx_effect_env == "morning_high":
        morning_models = add_signal(morning_models, index, scale_factor=HIGH_MULTIPLIER)
        evening_models = add_signal(evening_models, index, scale_factor=LOW_MULTIPLIER)

    decay_factor = 1.0

    # Create a new dataframe to store the full dataset during the simulation
    full_df = pd.DataFrame()

    # Get the unique users
    list_of_users = df["user_id"].unique()

    # Sample NUSERS from list_of_users with replacement
    sampled_users = rng.choice(list_of_users, num_users, replace=True)

    # Sample some number of users who will have dropout
    num_dropout_users = int(num_users * (dropout_percentage / 100))
    dropout_users = rng2.choice(list(range(num_users)), num_dropout_users, replace=False)

    # Set dosage for each user to be 0
    dosage = np.zeros(num_users)

    # Set dropout counts to be 0 for each user
    dropout_counts = np.zeros(num_users)

    # Generate dosage weights
    discount_factor = float(dosage_lookback - 1) / dosage_lookback
    normalizing_factor = (1 - discount_factor) / (1 - discount_factor ** dosage_lookback)
    dosage_weights = np.array([normalizing_factor * (discount_factor ** i) for i in range(dosage_lookback)])

    # Run simulation for num_days for each user
    for day in range(num_days):
        # Iterate over morning and evenings
        for time_of_day in range(2):
            # Check tx effect setting
            if tx_effect_env in ["morning_low", "morning_high"]:
                if time_of_day == 0:
                    user_models = copy.deepcopy(morning_models)
                else:
                    user_models = copy.deepcopy(evening_models)
            else:
                user_models = copy.deepcopy(user_models_ref)

            if decay:
                decay_factor = 1 - (float(day) / (num_days - 1))
                user_models = add_signal(
                    user_models, index, scale_factor=decay_factor, simple_scale=True
                )


            if habituation is not None:
                user_models = copy.deepcopy(user_models_ref_copy)

                multiplier = 1.0
                if tx_effect_env == "overall_low":
                    multiplier = LOW_MULTIPLIER
                elif tx_effect_env == "overall_high":
                    multiplier = HIGH_MULTIPLIER
                elif tx_effect_env == "morning_low":
                    if time_of_day == 0:
                        multiplier = LOW_MULTIPLIER
                    else:
                        multiplier = HIGH_MULTIPLIER
                elif tx_effect_env == "morning_high":
                    if time_of_day == 0:
                        multiplier = HIGH_MULTIPLIER
                    else:
                        multiplier = LOW_MULTIPLIER
                
                if decay:
                    multiplier = multiplier * decay_factor
                
                if habituation == "exponential":
                    habituation_multipliers = [
                        exponential_decay(dosage[user], multiplier, habituation_factor)
                        for user in range(num_users)
                    ]
                    user_models = add_signal_user_specific(
                        user_models,
                        index,
                        scale_factor=habituation_multipliers,
                    )
                else:
                    pass


            # Iterate over all users
            for user in range(num_users):
                # for user in range(len(list_of_users)):

                # Calculate the decision point index
                decision_point = day * 2 + time_of_day

                # Sample action from the algorithm
                action = rl_algorithm.get_action(user, decision_point)

                # Get the data for the current decision point
                dp_data = pd.DataFrame(
                    df[
                        (df["day"] == day + 1)
                        & (df["time_of_day"] == time_of_day)
                        & (df["user_id"] == sampled_users[user])
                    ]
                )

                # Get the user model
                user_model = user_models[sampled_users[user]]

                if dropout > 0 and user in dropout_users:

                    # Format the data for the user model prediction
                    X = format_data_for_prediction(dp_data, action, day, start_day, dosage[user])

                    probabilities = predict_probabilities(user_model, X, dropout)[0]
                else:
                    X = format_data_for_prediction(dp_data, action, day, start_day)

                    # Predict the reward
                    probabilities = user_model.predict_proba(X)[0]

                # Sample the reward
                reward = rng.choice(user_model.classes_, p=probabilities)

                # Calculate expected reward
                expected_reward = (
                    np.array(probabilities)
                    .ravel()
                    .dot(np.array(user_model.classes_).ravel())
                )


                # # Check if dropout is enabled and at least 6 decision points have passed
                # if dropout != -1 & decision_point >= 5:
                #     # Check the dropout count
                #     if dropout_counts[user] > 0:
                #         # Decrement the dropout count
                #         dropout_counts[user] -= 1
                        
                #         # Set reward to 0
                #         reward = 0
                #         expected_reward = 0

                #     # Check the dosage value
                #     elif dosage[user] >= dropout:
                #         # Set reward to 0
                #         reward = 0
                #         expected_reward = 0
                #         # Set the dropout count
                #         dropout_counts[user] = 5


                # Get the actual survey completion and usage flag from the reward
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
                
                # if debug:
                #     print(decision_point, user, action, dosage[user], reward)

                # Update the algorithm with the new data
                # Make sure to not provide cannabis use when reward < 2
                new_data = {
                    "user": user,
                    "decision_point": decision_point,
                    "day": day,
                    "time_of_day": time_of_day,
                    "app_usage": app_usage_flag,
                    "cannabis_use": dp_data["cannabis_use"].values[0]
                    if reward > 1
                    else None,
                    "survey_completion": survey_completion,
                    "activity_question": activity_question,
                    "action": action,
                    "expected_reward": expected_reward,
                }

                # Update the algorithm with the new data for next decision point state
                rl_algorithm.update_data(user, decision_point, new_data)

                # Create a new row for the full dataset
                new_row = {
                    "user": user,
                    # "original_user": list_of_users[user],
                    "original_user": sampled_users[user],
                    "decision_point": decision_point,
                    "day": day,
                    "time_of_day": time_of_day,
                    "action": action,
                    "reward": reward,
                    "app_usage": app_usage_flag,
                    "cannabis_use": dp_data["cannabis_use"].values[0],
                    "survey_completion": survey_completion,
                    "activity_question": activity_question,
                    "expected_reward": expected_reward,
                }

                # Add the new row to the full dataset
                full_df = pd.concat([full_df, pd.DataFrame(new_row, index=[0])])

                # Update the dosage for the user

                # Get the last 6 actions
                last_6_actions = full_df[
                    (full_df["user"] == user)
                ].sort_values(by="decision_point", ascending=False).head(6)["action"].values

                # do a weighted sum of the last 6 actions
                dosage[user] = np.sum(last_6_actions * dosage_weights[: len(last_6_actions)])

            rl_algorithm.end_decision_point(decision_point)

        if hyper_param_update_cadence == "daily":
            rl_algorithm.update_hyperparameters()
        elif (
            hyper_param_update_cadence == "weekly" and day % 7 == 6
        ):  # end of the 7th day
            rl_algorithm.update_hyperparameters()

        if posterior_update_cadence == "daily":
            rl_algorithm.update_posteriors()
        elif posterior_update_cadence == "weekly" and day % 7 == 6:  # end of 7th day
            rl_algorithm.update_posteriors()

    # Save the algorithm
    rl_algorithm.dump_model_and_data(save_path)

    # Save the full dataset
    full_df.reset_index(drop=True, inplace=True)
    full_df.to_csv(os.path.join(save_path, "simulation_output.csv"), index=False)

    return full_df


# In[19]:


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", "--num_users", type=int, default=120)
    parser.add_argument("-d", "--num_days", type=int, default=30)
    parser.add_argument("-s", "--seed", type=int, default=0)

    parser.add_argument("-st", "--start_day", type=int, default=0)
    parser.add_argument("-dbg", "--debug", type=bool, default=False)
    parser.add_argument("-rl", "--rl_alg", type=str, default="random")

    parser.add_argument("-a", "--alg_variant", type=int, default=0)
    parser.add_argument(
        "-b", "--logistic_b", type=float, default=20
    )  # If negative, then uses infinity
    parser.add_argument("-c", "--logistic_c", type=float, default=5.0)
    parser.add_argument("-lmin", "--l_min", type=float, default=0.2)
    parser.add_argument("-lmax", "--l_max", type=float, default=0.8)
    parser.add_argument("-p", "--posterior_update_cadence", type=str, default="daily")
    parser.add_argument(
        "-hp", "--hyper_param_update_cadence", type=str, default="weekly"
    )
    parser.add_argument("-tx", "--tx_effect_env", type=str, default="none")
    parser.add_argument("-dc", "--decay", type=bool, default=False)
    parser.add_argument("-dl", "--dosage_lookback", type=int, default=6)
    parser.add_argument("-dr", "--dropout", type=float, default=-1)
    parser.add_argument("-dp", "--dropout_percentage", type=float, default=100.0)
    parser.add_argument("-hs", "--habituation", type=int, default=0)
    parser.add_argument("-act", "--act_cost_threshold", type=float, default=0.0)

    args = parser.parse_args()

    # Set the parameters
    num_users = args.num_users
    num_days = args.num_days
    seed = args.seed
    alg_variant = args.alg_variant
    logistic_b = args.logistic_b
    logistic_c = args.logistic_c
    l_min = args.l_min
    l_max = args.l_max
    posterior_update_cadence = args.posterior_update_cadence
    hyper_param_update_cadence = args.hyper_param_update_cadence
    start_day = args.start_day
    dosage_lookback = args.dosage_lookback
    rl_alg = args.rl_alg
    dropout = args.dropout
    act_cost_threshold = args.act_cost_threshold
    dropout_percentage = args.dropout_percentage

    if args.habituation == 0:
        habituation = None
        habituation_factor = 0
    elif args.habituation >= 1 and args.habituation <= 3:
        habituation = "exponential"
        habituation_factor = 3 * args.habituation
    elif args.habituation > 3 and args.habituation <= 6:
        habituation = "linear"
        habituation_factor = 0.5 + (0.15 * (args.habituation - 3))
    elif args.habituation > 6 and args.habituation <= 9:
        habituation = "concave"
        habituation_factor = 0.5 + (0.15 * (args.habituation - 6))
    else:
        raise NotImplementedError

    # Set the folder name
    foldername = (
        str(rl_alg)+ "_"
        + "variant_"
        + str(alg_variant)
        + "_lambda_"
        + str(act_cost_threshold)
        #+ "_hs_"
        #+ str(habituation)
        # + "_hf_"
        # + str(habituation_factor)
        + "_eta_"
        + str(dropout)
        + "_P_"
        + str(dropout_percentage)
        #+ "_dl_"
        #+ str(dosage_lookback)
        #+ "_b_"
        #+ str(logistic_b)
        #+ "_posterior_"
        #+ posterior_update_cadence
        #+ "_hyper_"
        #+ hyper_param_update_cadence
        + "_tx_"
        + args.tx_effect_env
        #+ "_decay_"
        #+ str(args.decay)
        + "_num_users_"
        + str(num_users)
        + "_num_days_"
        + str(num_days)
        #+ "_start_"
        #+ str(start_day)
    )

    # Set the simulation folder
    simfolder = SIMULATION_FOLDER + foldername
    os.makedirs(simfolder, exist_ok=True)

    rv = load_random_vars()

    # Set the allocation function
    alloc_func_type = "smooth"
    if logistic_b < 0:
        logistic_b = np.inf
    else:
        logistic_b = logistic_b / LOGISTIC_SIGMA
    alloc_func = get_allocation_function(
        func_type=alloc_func_type,
        B=logistic_b,
        randomvars=rv,
        C=logistic_c,
        L_min=l_min,
        L_max=l_max,
    )

    # Run the simulation
    sim_data = run_simulation(
        num_users=num_users,
        num_days=num_days,
        start_day=start_day,
        seed=seed,
        sim_path=simfolder,
        rl_alg=rl_alg,
        rl_alg_variant=alg_variant,
        alloc_func=alloc_func,
        posterior_update_cadence=posterior_update_cadence,
        hyper_param_update_cadence=hyper_param_update_cadence,
        tx_effect_env=args.tx_effect_env,
        decay=args.decay,
        debug=args.debug,
        habituation=habituation,
        habituation_factor=habituation_factor,
        dosage_lookback=dosage_lookback,
        dropout=dropout,
        act_cost_threshold=act_cost_threshold,
        dropout_percentage=dropout_percentage,
    )
    return sim_data


# In[ ]:


sim_data = main()

