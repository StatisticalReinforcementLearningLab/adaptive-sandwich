import jax.numpy as jnp
import jax
import numpy as np
import os
import scipy.stats as stats
import scipy.linalg as linalg
import pickle as pkl
import logging
import copy

from functools import partial
from typing import Callable
from RLAlgorithm import RLAlgorithm
from priors import PARAM_SIZE

class FlatProb(RLAlgorithm):
    """Bayesian linear regression RL algorithm"""
    def __init__(
        self,
        num_users: int,
        starting_time_of_day: int = 0,
        rng: np.random.RandomState = None,
        debug: bool = False,
        logger_path: str = None,
    ):
        """Initialize the algorithm"""
        self.current_decision_point = 0
        self.time_of_day = starting_time_of_day
        self.rng = rng
        self.bernoulli = stats.bernoulli
        self.user_data = {}
        self.num_users = num_users
        for i in range(self.num_users):
            self.user_data[i] = {
                "features": [{}],
                "state": [[0, self.time_of_day, 1]],  # default initial starting state
                "design_state": [],
                "action": [],
                "act_prob": [],
                "reward": [],
            }
        self.debug = debug

        # Logging stuff
        logfile = logger_path + "/log.txt"
        self.logger = logging.getLogger("FlatProb")
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode='w')
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        
    def create_state(self, user: int):
        """Create the state vector for the given features and decision point index"""
        avg_reward = np.mean(self.user_data[user]["reward"][-3:])

        # create S1
        if avg_reward >= 2:
            S1 = 1
        else:
            S1 = 0

        # create S2 based on time of day
        S2 = self.time_of_day

        # create S3 based on last cannabis use
        # if user used cannabis in the past decision point, S3 = 1
        if self.user_data[user]["features"][-1]["cannabis_use"] == 1:
            S3 = 1
        # if user did not use cannabis in the past decision point, S3 = 0
        elif self.user_data[user]["features"][-1]["cannabis_use"] == 0:
            S3 = 0
        # if the user did not report cannabis use in the past decision point, S3 = 1
        else:
            S3 = 1
        return [S1, S2, S3]

    def compute_reward(self, user: int):
        """Create the reward given the features for a given time index"""
        reward = 0
        features = self.user_data[user]["features"][-1]
        if features["survey_completion"] == 1:
            if features["activity_question"] == 1:
                reward = 3
            else:
                reward = 2
        elif features["app_usage"] == 1:
            reward = 1
        return reward

    def update_design_row(self, user: int):
        """Update the design row for a given user"""

        # Get the state, action, and action probability
        state = self.user_data[user]["state"][-1]
        action = float(self.user_data[user]["action"][-1])
        act_prob = float(self.user_data[user]["act_prob"][-1])

        # Get the individual state elements
        s1 = state[0]
        s2 = state[1]
        s3 = state[2]

        # Create the baseline and advantage
        baseline = [1, s1, s2, s3, s1 * s2, s1 * s3, s2 * s3, s1 * s2 * s3]
        act_advantage = [(act_prob * i) for i in baseline]
        a_pi_advantage = [((action - act_prob) * i) for i in baseline]

        # Create the design row
        design_row = [
            *baseline[: PARAM_SIZE[0][0]],
            *act_advantage[: PARAM_SIZE[0][1]],
            *a_pi_advantage[: PARAM_SIZE[0][2]],
        ]
        if self.debug:
            print(design_row)

        return design_row
    
    def clip_prob(self, prob, min: float = 0.2, max: float = 0.8):
        """Clip the probability to be between min and max"""
        return np.clip(prob, min, max)
    
    def get_action(self, user: int, decision_idx: int):
        """
        Get the action for a particular user for a particular decision point
        """
        # Get the state
        state = self.user_data[user]["state"][-1]

        # Call the allocation function
        act_prob = 0.5
        
        if self.debug:
            self.logger.info(f"[{decision_idx}] {user} {act_prob}")
            print(user, decision_idx, act_prob)

        # Sample the action
        action = self.bernoulli.rvs(act_prob, random_state=self.rng)

        # Update the user data
        self.user_data[user]["action"].append(action)
        self.user_data[user]["act_prob"].append(act_prob)

        # Update the design state
        self.user_data[user]["design_state"].append(self.update_design_row(user))
        return action
    
    def update_data(self, user: int, decision_idx: int, data: dict):
        """
        Update the data for a particular user for a particular decision point
        """
        self.user_data[user]["features"].append(data)
        
    def end_decision_point(self, decision_idx: int):
        """
        End the decision point. Flip the time of day. This computes the reward
        for each user for the current decision point, increases the current
        decision point by 1, and computes the new state for each user for
        the new decision point
        """
        self.current_decision_point += 1
        self.time_of_day = 1 - self.time_of_day
        for i in range(self.num_users):
            self.user_data[i]["reward"].append(self.compute_reward(i))
            self.user_data[i]["state"].append(self.create_state(i))
    
    def create_design_and_reward_matrix(self):
        """
        Create the design matrix and reward matrix up until the current decision point
        using design rows and reward history for each user
        """

        # Design matrix made by stacking design state of each user
        design_matrix = np.vstack(
            [self.user_data[i]["design_state"] for i in range(self.num_users)]
        )

        # Reward matrix made by stacking reward history of each user
        reward_matrix = np.hstack(
            [self.user_data[i]["reward"] for i in range(self.num_users)]
        )
        return design_matrix, reward_matrix
    
    def update_hyperparameters(self):
        """
        Update the noise variance and random effects variance using data up until
        the current decision point
        """

        # Update noise variance and random effects variance
        pass
    
    def update_posteriors(self):
        """
        Update the posteriors using data up until the current decision point
        """

        pass
    
    def dump_model_and_data(self, path: str):
        """
        Dump the data to a pickle file
        """
        user_data_path = os.path.join(path, "user_data.pkl")
        with open(user_data_path, "wb") as f:
            pkl.dump(self.user_data, f)
            
    def optimize_variance_parameters(self, debug: bool = False):
        """
        Optimize the noise variance and random effects variance using data up until
        the current decision point using empirical Bayes
        """
        pass