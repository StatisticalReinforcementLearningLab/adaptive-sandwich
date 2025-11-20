"""
Implementations of several RL algorithms that may be used in study simulations.
"""

import logging

import jax
import numpy as np
from jax import numpy as jnp
from jax import debug
from functions_to_pass_to_analysis.smooth_thompson_sampling_act_prob_function_no_action_centering import (
    smooth_thompson_sampling_act_prob_function_no_action_centering,
    # smooth_thompson_sampling_act_prob_function_no_action_centering_partial,
)
from functions_to_pass_to_analysis.smooth_thompson_sampling_act_prob_function_no_action_centering_partial import (
    smooth_thompson_sampling_act_prob_function_no_action_centering_partial,
)

from functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)
from functions_to_pass_to_analysis.synthetic_get_action_1_prob_SAC import (
    synthetic_get_action_1_prob_SAC,
)
from helper_functions import clip

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


class FixedRandomization:
    """
    Fixed randomization algorithm; no learning
    """

    def __init__(self, args, state_feats, treat_feats, alg_seed):
        self.args = args
        self.rng = np.random.default_rng(alg_seed)
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.pi_args = {}
        self.rl_update_args = {}

    def update_alg(self, new_data):
        raise NotImplementedError("Fixed randomization never updates")

    def get_action_probs(self, curr_timestep_data):
        raw_probs = jnp.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob # 0.5
        return clip(self.args.lower_clip, self.args.upper_clip, raw_probs)

    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()
    
    # def collect_pi_args(self, all_prev_data, calendar_t):
    #     logger.info(
    #         "Collecting args to pi function at time %d for each user in dictionary format",
    #         calendar_t,
    #     )
    #     assert calendar_t == jnp.max(all_prev_data["calendar_t"].to_numpy())

    #     self.pi_args[calendar_t] = {
    #         user_id: ()
    #         for user_id in self.get_all_users(all_prev_data)
    #     }
    
    # def collect_rl_update_args(self, all_prev_data, calendar_t):
    #     """
    #     NOTE: Must be called AFTER the update it concerns happens, so that the
    #     beta the rest of the data already produced is used.
    #     """
    #     logger.info(
    #         "Collecting args to loss/estimating function at time %d (last time included in update data) for each user in dictionary format",
    #         calendar_t,
    #     )
    #     next_policy_num = int(all_prev_data["policy_num"].max() + 1)
    #     if calendar_t == 20:
    #         print(all_prev_data)
    #         print(1)
    #     self.rl_update_args[next_policy_num] = {}
    #     for user_id in self.get_all_users(all_prev_data):
    #         in_study_user_data = all_prev_data.loc[
    #             (all_prev_data.user_id == user_id) & (all_prev_data.in_study == 1)
    #         ]
    #         self.rl_update_args[next_policy_num][user_id] = (
    #             ()
    #             # We only care about the data overall, however, if there is any
    #             # in-study data for this user so far
    #             if not in_study_user_data.empty
    #             else ()
    #         )

def get_pis_batched_sigmoid(
    beta_est,
    lower_clip,
    steepness,
    upper_clip,
    batched_treat_states_tensor,
):
    return jax.vmap(
        fun=synthetic_get_action_1_prob_pure,
        in_axes=(None, None, None, None, 0),
        out_axes=0,
    )(
        beta_est,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
    )

def get_pis_batched_sigmoid_partial(
    beta_est,
    lower_clip,
    steepness,
    upper_clip,
    batched_treat_states_tensor,
    Z_id,
):
    return jax.vmap(
        fun=synthetic_get_action_1_prob_SAC,
        in_axes=(None, None, None, None, 0, 0), # 0：batch dimension, None: broadcasted or fixed
        out_axes=0, # the first index would be batch dimension
    )(
        beta_est,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
        Z_id, # new
    )

def get_pis_batched_thompson_sampling(
    beta_est,
    batched_treat_states_tensor,
    num_users_entered_before_last_update,
    lower_clip,
    upper_clip,
    steepness,
):
    # equivalent to a loop for each user    
    return jax.vmap(
        fun=smooth_thompson_sampling_act_prob_function_no_action_centering,
        in_axes=(None, 0, None, None, None, None),
        out_axes=0,
    )(
        beta_est,
        batched_treat_states_tensor,
        num_users_entered_before_last_update,
        lower_clip,
        upper_clip,
        steepness,
    )

def get_pis_batched_thompson_sampling_twoarmed(
    beta_est,
    batched_treat_states_tensor,
    num_users_entered_before_last_update,
    lower_clip,
    upper_clip,
    steepness,
    Z_id, # new
):
    # equivalent to a loop for each user    
    return jax.vmap(
        # fun=smooth_thompson_sampling_act_prob_function_no_action_centering,
        fun=smooth_thompson_sampling_act_prob_function_no_action_centering_partial,
        # in_axes=(None, 0, None, None, None, None),
        in_axes=(None, 0, None, None, None, None, 0), # Z_id should be row-wise for computation
        out_axes=0,
    )(
        beta_est,
        batched_treat_states_tensor,
        num_users_entered_before_last_update,
        lower_clip,
        upper_clip,
        steepness,
        Z_id, # new 
    )


class SigmoidLS:
    """
    Sigmoid Least Squares algorithm
    """

    def __init__(
        self,
        state_feats,
        treat_feats,
        alg_seed,
        steepness,
        lower_clip,
        upper_clip,
        action_centering,
    ):
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        # Note that steepness = 0 should remove learning adaptivity, with action probabilities of .5
        self.steepness = steepness
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.rng = np.random.default_rng(self.alg_seed)
        self.beta_dim = len(self.state_feats) + len(self.treat_feats)
        self.pi_args = {}
        self.rl_update_args = {}

        self.treat_feats_action = ["action:" + x for x in self.treat_feats]
        self.treat_bool = jnp.array(
            [
                x in self.treat_feats_action
                for x in self.state_feats + self.treat_feats_action
            ]
        )

        # Set an initial policy
        self.all_policies = [
            {
                "RX": jnp.zeros(self.beta_dim),
                "XX": jnp.zeros(self.beta_dim),
                "beta_est": jnp.zeros(self.beta_dim),
                "inc_data": {},
                "total_obs": 0,
                "seen_user_id": set(),
            }
        ]
        self.action_centering = action_centering
        self.incremental_updates = True

    # TODO: All of these functions arguably should not modify the dataframe...
    # Should be making a new dataframe and modifying that, or expecting the data
    # to be formatted as such (though I don't like the latter). Going with this
    # for now. This modification also raises a warning about setting a slice on
    # a copy, but it seems to work perfectly.

    # TODO: Docstring
    def get_base_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, df):
        treat_states = df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)

    def get_rewards(self, df, reward_col="reward"):
        rewards = df[reward_col].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    def get_actions(self, df, action_col="action"):
        actions = df[action_col].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    def get_action1probs(
        self,
        df,
        actionprob_col="action1prob",
    ):
        action1probs = df[actionprob_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probs)

    def get_action1probstimes(
        self,
        df,
        calendar_t_col="calendar_t",
    ):
        action1probstimes = df[calendar_t_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probstimes)

    # TODO: Docstring
    def get_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        treat_states = df[self.treat_feats].to_numpy()
        return (base_states, treat_states)

    def update_alg(self, new_data):
        """
        Update algorithm with new data

        Inputs:
        - `new_data`: a pandas data frame with new data

        Outputs:
        - None
        """

        # update algorithm with new data
        actions = new_data["action"].to_numpy().reshape(-1, 1)
        action1probs = new_data["action1prob"].to_numpy().reshape(-1, 1)
        rewards = new_data["reward"].to_numpy().reshape(-1, 1)
        base_states, treat_states = self.get_states(new_data)
        if self.action_centering:
            logger.info("Action centering is TURNED ON for RL algorithm updates.")
        design = jnp.concatenate(
            [
                base_states,
                (actions - action1probs * self.action_centering) * treat_states,
            ],
            axis=1,
        )

        # Only include available data
        calendar_t = new_data["calendar_t"].to_numpy().reshape(-1, 1)
        rewards_avail = rewards
        avail_bool = jnp.ones(rewards.shape)
        design_avail = design
        user_id_avail = new_data["user_id"].to_numpy()

        # Get policy estimator
        new_RX = self.all_policies[-1]["RX"] + jnp.sum(design_avail * rewards_avail, 0)
        new_XX = self.all_policies[-1]["XX"] + jnp.einsum(
            "ij,ik->jk", design_avail, design_avail
        )

        # NOTE: this gives NANs and breaks action selection when all
        # users take same action
        inv_XX = jnp.linalg.inv(new_XX)

        beta_est = jnp.matmul(inv_XX, new_RX.reshape(-1)).squeeze()

        seen_user_id = self.all_policies[-1]["seen_user_id"].copy()
        seen_user_id.update(new_data["user_id"].to_numpy())

        # Save Data
        inc_data = {
            "reward": rewards.flatten(),
            "action": actions.flatten(),
            "action1prob": action1probs.flatten(),
            "base_states": base_states,
            "treat_states": treat_states,
            "avail": avail_bool.flatten(),
            "user_id": user_id_avail,
            "calendar_t": calendar_t.flatten(),
            "design": design,
        }
        update_dict = {
            "total_obs": self.all_policies[-1]["total_obs"] + len(new_data),
            "RX": new_RX,
            "XX": new_XX,
            "beta_est": beta_est,
            "inc_data": inc_data,
            "seen_user_id": seen_user_id,
        }

        self.all_policies.append(update_dict)

    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()

    def get_current_beta_estimate(self):
        return self.all_policies[-1]["beta_est"]

    def collect_rl_update_args(self, all_prev_data, calendar_t):
        """
        NOTE: Must be called AFTER the update it concerns happens, so that the
        beta the rest of the data already produced is used.
        """
        logger.info(
            "Collecting args to loss/estimating function at time %d (last time included in update data) for each user in dictionary format",
            calendar_t,
        )
        next_policy_num = int(all_prev_data["policy_num"].max() + 1)
        self.rl_update_args[next_policy_num] = {}
        for user_id in self.get_all_users(all_prev_data):
            in_study_user_data = all_prev_data.loc[
                (all_prev_data.user_id == user_id) & (all_prev_data.in_study == 1)
            ]
            self.rl_update_args[next_policy_num][user_id] = (
                (
                    self.get_current_beta_estimate(),
                    self.get_base_states(in_study_user_data),
                    self.get_treat_states(in_study_user_data),
                    self.get_actions(in_study_user_data),
                    self.get_rewards(in_study_user_data),
                    # NOTE important: we require an entry for all times before the update
                    # regardless of in-study or not. This is necessary because these probabilities
                    # have special meaning and must correspond to particular times if used.
                    self.get_action1probs(in_study_user_data),
                    self.get_action1probstimes(in_study_user_data),
                    self.action_centering,
                )
                # We only care about the data overall, however, if there is any
                # in-study data for this user so far
                if not in_study_user_data.empty
                else ()
            )

    def collect_pi_args(self, all_prev_data, calendar_t):
        logger.info(
            "Collecting args to pi function at time %d for each user in dictionary format",
            calendar_t,
        )
        assert calendar_t == jnp.max(all_prev_data["calendar_t"].to_numpy())

        self.pi_args[calendar_t] = {
            user_id: (
                (
                    self.get_current_beta_estimate(),
                    self.lower_clip,
                    self.steepness,
                    self.upper_clip,
                    self.get_treat_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    )[-1],
                )
                if all_prev_data.loc[
                    (all_prev_data.user_id == user_id)
                    & (all_prev_data.calendar_t == calendar_t)
                ].in_study.item()
                else ()
            )
            for user_id in self.get_all_users(all_prev_data)
        }

    def get_action_probs(self, curr_timestep_data):
        """
        Form action selection probabilities from newly current data (only use when running RL algorithm)

        Inputs:
        - `curr_timestep_data`: Pandas data frame of current data that can be used to form the states
        - `filter_keyval`: None (not needed for this algorithm)

        Outputs:
        - Numpy vector of action selection probabilities
        """

        treat_states = curr_timestep_data[self.treat_feats].to_numpy()

        return get_pis_batched_sigmoid(
            self.get_current_beta_estimate(),
            self.lower_clip,
            self.steepness,
            self.upper_clip,
            treat_states,
        )


class SmoothPosteriorSampling:
    """
    Sigmoid Least Squares algorithm
    """

    def __init__(
        self,
        state_feats,
        treat_feats,
        alg_seed,
        lower_clip,
        upper_clip,
        steepness,
        action_centering,
        prior_mu,
        prior_sigma,
        noise_var,
        twoarmed=False,
    ):
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.steepness = steepness
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.noise_var = noise_var
        self.rng = np.random.default_rng(self.alg_seed)
        self.beta_dim = len(self.state_feats) + len(self.treat_feats)
        self.pi_args = {}
        self.rl_update_args = {}

        self.twoarmed = twoarmed ##### new

        # Set an initial policy
        self.all_policies = [
            {
                "beta_est": self.form_beta_from_posterior(
                    self.prior_mu,
                    self.prior_sigma,
                    0,
                ),
                "inc_data": {},
                "num_users_entered_before_last_update": 0,
            }
        ]
        self.action_centering = action_centering
        self.incremental_updates = False

    # TODO: All of these functions arguably should not modify the dataframe...
    # Should be making a new dataframe and modifying that, or expecting the data
    # to be formatted as such (though I don't like the latter). Going with this
    # for now. This modification also raises a warning about setting a slice on
    # a copy, but it seems to work perfectly.

    # TODO: Docstring
    def get_base_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, df):
        treat_states = df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)

    def get_rewards(self, df, reward_col="reward"):
        rewards = df[reward_col].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    def get_actions(self, df, action_col="action"):
        actions = df[action_col].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    def get_action1probs(
        self,
        df,
        actionprob_col="action1prob",
    ):
        action1probs = df[actionprob_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probs)

    def get_action1probstimes(
        self,
        df,
        calendar_t_col="calendar_t",
    ):
        action1probstimes = df[calendar_t_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probstimes)

    # TODO: Docstring
    def get_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        treat_states = df[self.treat_feats].to_numpy()
        return (base_states, treat_states)

    ####### Posterior computations ########
    def compute_posterior_var(self, design):
        return np.linalg.inv(
            1 / self.noise_var * design.T @ design + np.linalg.inv(self.prior_sigma)
        )

    def compute_posterior_mean(self, design, rewards):

        return self.compute_posterior_var(design) @ (
            1 / self.noise_var * design.T @ rewards
            + np.linalg.inv(self.prior_sigma) @ self.prior_mu
        )

    # update posterior distribution
    def compute_posterior(
        self,
        design,
        rewards,
    ):
        mean = self.compute_posterior_mean(design, rewards)
        var = self.compute_posterior_var(design)

        return mean, var

    @staticmethod # form beta from mean and covariance
    def form_beta_from_posterior(
        posterior_mean: np.ndarray,
        posterior_var: np.ndarray,
        num_users_entered_before_update: int,
    ) -> jnp.ndarray:
        """
        Form the beta vector from the posterior mean and variance.
        This is for after-study analysis, concisely collecting all the information
        in the posterior in a convenint form. Explicitly, we concatenate the posterior
        mean with the upper triangular elements of the inverse posterior variance matrix.

        Parameters:
        posterior_mean (np.ndarray):
            The posterior mean vector.
        posterior_var (np.ndarray):
            The posterior variance matrix.

        Returns:
        jnp.ndarray: The beta vector.
        """
        sigma_inv = np.linalg.inv(posterior_var)
        ut_sigma_inv = sigma_inv[np.triu_indices_from(sigma_inv)]
        return jnp.concatenate(
            [
                posterior_mean,
                ut_sigma_inv.flatten() / max(1, num_users_entered_before_update),
            ]
        )

    def update_alg(self, all_data):
        """
        Update algorithm with new data

        Inputs:
        - `all_data`: a pandas data frame all study data so far

        Outputs:
        - None
        """

        ########## important: only update on the target users: in-study and treatment group Z_id=1 ##########

        in_study_data = all_data[all_data["in_study"] == 1] # only include unit data that have happened so far till t
        
        
        # twoarmed trial: only update data with Z_id=1
        
        if self.twoarmed:
            # only update the algorithm on the treatment group, (similar role of "in_study" column)
            in_study_data = in_study_data[in_study_data['Z_id']==1]
        
        # if self.twoarmed:
        #     actions = in_study_data.loc[in_study_data['Z_id']==1, "action"].to_numpy()
        #     action1probs = in_study_data.loc[in_study_data['Z_id']==1,"action1prob"].to_numpy()
        #     rewards = in_study_data.loc[in_study_data['Z_id']==1,"reward"].to_numpy()
        # else: #
        actions = in_study_data["action"].to_numpy()
        action1probs = in_study_data["action1prob"].to_numpy()
        rewards = in_study_data["reward"].to_numpy()
        
        # only form the data with Z_id=1
        base_states, treat_states = self.get_states(in_study_data)
        if self.action_centering:
            logger.info("Action centering is TURNED ON for RL algorithm updates.")
            design = np.hstack(
                (
                    base_states,
                    np.multiply(treat_states.T, action1probs).T,
                    np.multiply(treat_states.T, (actions - action1probs)).T,
                )
            )
        else:
            design = np.hstack((base_states, np.multiply(treat_states.T, actions).T))

        ################### Step 1: compute posterior ###################
        
       
        posterior_mean, posterior_var = self.compute_posterior(
            design,
            rewards,
        )
        # posterior_mean = self.prior_mu
        # posterior_var = self.prior_sigma

        num_users_before_update = in_study_data["user_id"].nunique()
        
        ################### Step 2: concatenate posterior mean and upper triangular elements of inverse posterior variance ################### 
        beta_est = self.form_beta_from_posterior(
            posterior_mean,
            posterior_var,
            num_users_before_update, # should be the number of users in the treatment group
        )

        # save Data
        inc_data = {
            "reward": rewards.flatten(),
            "action": actions.flatten(),
            "action1prob": action1probs.flatten(),
            "base_states": base_states,
            "treat_states": treat_states,
            "design": design,
        }
        update_dict = {
            "beta_est": beta_est,
            "num_users_entered_before_last_update": num_users_before_update, # ????????? need to be checked
            "inc_data": inc_data,
        }

        self.all_policies.append(update_dict)
        """
        if state = 2
        {'beta_est': Array([-1.1359017 ,  3.5079467 ,  1.0000004 ,  0.33333334,  0.33333367],      dtype=float32), 
        'num_users_entered_before_last_update': 3, 
        'inc_data': {'reward': array([-1.72244138,  2.37204856, -0.54936675]), 
                    'action': array([0., 1., 0.]), 
                    'action1prob': array([0.50411886, 0.50411886, 0.50411886]),
                      'base_states': array([[1],
                    [1],
                    [1]]), 'treat_states': array([[1],
                    [1],
                    [1]]), 'design': array([[1., 0.],
                    [1., 1.],
                    [1., 0.]])}}
        """
        
    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()


    
    def get_current_beta_estimate(self):
        return self.all_policies[-1]["beta_est"]

    def get_num_users_entered_before_last_update(self):
        return self.all_policies[-1]["num_users_entered_before_last_update"]

    def collect_rl_update_args(self, all_prev_data, calendar_t):
        """
        NOTE: Must be called AFTER the update it concerns happens, so that the
        beta the rest of the data already produced is used.
        """
        logger.info(
            "Collecting args to loss/estimating function at time %d (last time included in update data) for each user in dictionary format",
            calendar_t,
        )
        next_policy_num = int(all_prev_data["policy_num"].max() + 1)
        self.rl_update_args[next_policy_num] = {}
        for user_id in self.get_all_users(all_prev_data):
            in_study_user_data = all_prev_data.loc[
                (all_prev_data.user_id == user_id) & (all_prev_data.in_study == 1)
            ]
            if self.twoarmed:
                self.rl_update_args[next_policy_num][user_id] = (
                    (
                        self.get_current_beta_estimate(), # save the model's parameters
                        self.get_num_users_entered_before_last_update(),
                        self.get_treat_states(in_study_user_data),
                        self.get_actions(in_study_user_data),
                        self.get_rewards(in_study_user_data),
                        self.prior_mu,
                        jnp.linalg.inv(self.prior_sigma),
                        self.noise_var,
                        in_study_user_data.loc[in_study_user_data['Z_id']==1,'Z_id'].to_numpy() if self.twoarmed else in_study_user_data['Z_id'].to_numpy(), # new
                    )
                    # We only care about the data overall, however, if there is any
                    # in-study data for this user so far
                    if not in_study_user_data.empty
                    else ()
                )
            else:
                self.rl_update_args[next_policy_num][user_id] = (
                    (
                        self.get_current_beta_estimate(), # save the model's parameters
                        self.get_num_users_entered_before_last_update(),
                        self.get_treat_states(in_study_user_data),
                        self.get_actions(in_study_user_data),
                        self.get_rewards(in_study_user_data),
                        self.prior_mu,
                        jnp.linalg.inv(self.prior_sigma),
                        self.noise_var,
                        # in_study_user_data.loc[in_study_user_data['Z_id']==1,'Z_id'].to_numpy() if self.twoarmed else in_study_user_data['Z_id'].to_numpy(), # new
                    )
                    # We only care about the data overall, however, if there is any
                    # in-study data for this user so far
                    if not in_study_user_data.empty
                    else ()
                )

    # collect the learning policy's information at the current decision time
    def collect_pi_args(self, all_prev_data, calendar_t):
        logger.info(
            "Collecting args to pi function at time %d for each user in dictionary format",
            calendar_t,
        )
        assert calendar_t == jnp.max(all_prev_data["calendar_t"].to_numpy())

        ###### consistent with smooth_thompson_sampling_act_prob_function_no_action_centering_partial ########
        if self.twoarmed:
            self.pi_args[calendar_t] = {
                user_id: (
                    (
                        self.get_current_beta_estimate(),
                        self.get_treat_states(
                            all_prev_data.loc[all_prev_data.user_id == user_id]
                        )[-1],
                        self.get_num_users_entered_before_last_update(), # in self.all_policies, updated in the algorithm update function
                        self.lower_clip,
                        self.upper_clip,
                        self.steepness,
                        all_prev_data.loc[all_prev_data.user_id == user_id, 'Z_id'].values[-1], # new
                    )
                    if all_prev_data.loc[
                        (all_prev_data.user_id == user_id)
                        & (all_prev_data.calendar_t == calendar_t)
                    ].in_study.item()
                    else ()
                )
                for user_id in self.get_all_users(all_prev_data)
            }
        else:
            self.pi_args[calendar_t] = {
                user_id: (
                    (
                        self.get_current_beta_estimate(),
                        self.get_treat_states(
                            all_prev_data.loc[all_prev_data.user_id == user_id]
                        )[-1],
                        self.get_num_users_entered_before_last_update(), # in self.all_policies, updated in the algorithm update function
                        self.lower_clip,
                        self.upper_clip,
                        self.steepness,
                        # all_prev_data.loc[all_prev_data.user_id == user_id, 'Z_id'].values[-1], # new
                    )
                    if all_prev_data.loc[
                        (all_prev_data.user_id == user_id)
                        & (all_prev_data.calendar_t == calendar_t)
                    ].in_study.item()
                    else ()
                )
                for user_id in self.get_all_users(all_prev_data)
            }

    def get_action_probs(self, curr_timestep_data):
        """
        Form action selection probabilities

        Inputs:
        - `curr_timestep_data`: Pandas data frame of current data that can be used to form the states

        Outputs:
        - Numpy vector of action selection probabilities
        """
        
        ####### curr_timestep_data should be all in-study users at the current time point #########
        treat_states = curr_timestep_data[self.treat_feats].to_numpy() # [n, 2] -> (intercept, past_reward)
        Z_id = curr_timestep_data['Z_id'].to_numpy() # [n, 1]

        # evalute the action prob for all users accoeding to Z_id
        if self.twoarmed:
            action_probs = get_pis_batched_thompson_sampling_twoarmed(
                self.get_current_beta_estimate(),
                treat_states,
                self.get_num_users_entered_before_last_update(),
                self.lower_clip,
                self.upper_clip,
                self.steepness,
                Z_id, # new
            )   
        else:
            action_probs = get_pis_batched_thompson_sampling(
                self.get_current_beta_estimate(),
                treat_states,
                self.get_num_users_entered_before_last_update(),
                self.lower_clip,
                self.upper_clip,
                self.steepness,
            )

        return action_probs
        # if self.twoarmed: ###### control group, only assign 0.5
        #     mask = jnp.array(curr_timestep_data['Z_id'].to_numpy()) == 0
        #     return jnp.where(mask, 0.5, action_probs)
        #     # action_probs[curr_timestep_data['Z_id'] == 0]  = 0.5
        # else:
        #     return action_probs
    
        

class SoftActorCritic:
    """
    Soft Actor Critic algorithm with linear value function and sigmoid policy
    """

    def __init__(
        self,
        state_feats,
        treat_feats,
        alg_seed,
        lower_clip,
        upper_clip,
        steepness,
        twoarmed=False,
        history=False,
    ):
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.steepness = steepness
        self.rng = np.random.default_rng(self.alg_seed)
        self.beta_dim_Q = len(self.state_feats) + len(self.treat_feats)
        self.beta_dim_pi = len(self.state_feats)
        self.beta_dim = self.beta_dim_Q + self.beta_dim_pi # 4 + 2 = 6
        self.pi_args = {}
        self.rl_update_args = {}
        self.lambda_entropy = 1.0 # entropy regularization coefficient
        
        self.history=history # False: vanilla SAC; True: historical version of SAC 
        if self.history:
            self.lr_pi = 10.0
            self.ridge_penalty = 1e-3 
        else:
            self.lr_pi = 10.0
            self.ridge_penalty = 1e-3 # 0.0：NAN， 1e-8: still less stable. 1e-3 works better than 1e-5, similar to 1e-1
        # self.lr_Q = 1e-2 # no use
        self.gamma = 0.99
        
        self.next_states_index = ['intercept', 'reward'] 
        

        self.twoarmed = twoarmed ##### new

        # Set an initial policy
        # self.betaQ = jnp.array(self.rng.normal(size=self.beta_dim_Q))# initial critic parameters
        # self.betapi = jnp.array(self.rng.normal(size=self.beta_dim_pi)) # initial actor parameters
        # self.betaQ_tar = self.betaQ.copy() # target critic parameters (previous critic)
        # self.betapi_tar = self.betapi.copy() # target actor parameters (previous policy)
        self.betaQ_tar = jnp.array(self.rng.normal(size=self.beta_dim_Q))
        self.betapi_tar = jnp.array(self.rng.normal(size=self.beta_dim_pi)) # initial actor parameters      
        self.tau = 0.0 # Polyak for target network, tau=0: only based on the last step's critic, i.e., self.betaQ_tar = self.betaQ; tau=1: only based on the target critic, i.e., self.betaQ_tar = self.betaQ_tar (always a fixed target Q)
        self.decay=0.5 # learning rate decay for the actor, which require gradient descent (not necessary, but suggested here)
        self.epoch_actor = 500
        self.all_policies = [
            {
                "beta_est": jnp.concatenate([self.betaQ_tar, self.betapi_tar]),
                "inc_data": {},
                "num_users_entered_before_last_update": 0,
                "betaQ_target": self.betaQ_tar,
                "betapi_target": self.betapi_tar,
                "next_action": jnp.zeros(self.beta_dim_pi),
            }
        ]
        self.incremental_updates = False
        


    # TODO: Docstring
    def get_base_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, df):
        treat_states = df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)
    
    def get_next_states(self, df):
        next_states = df[self.next_states_index].to_numpy()
        return jnp.array(next_states)

    def get_rewards(self, df, reward_col="reward"):
        rewards = df[reward_col].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    def get_actions(self, df, action_col="action"):
        actions = df[action_col].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    def get_action1probs(
        self,
        df,
        actionprob_col="action1prob",
    ):
        action1probs = df[actionprob_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probs)

    def get_action1probstimes(
        self,
        df,
        calendar_t_col="calendar_t",
    ):
        action1probstimes = df[calendar_t_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probstimes)

    # TODO: Docstring
    def get_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        treat_states = df[self.treat_feats].to_numpy() # same as base_states
        next_states = df[self.next_states_index].to_numpy()
        return (base_states, treat_states, next_states)


    def policy(self, treat_states, beta_pi):
        # use the smooth allocation function, which is a scaled sigmoid function. Clipping function is not used as it causes to gradeint vanish.
        logits = jnp.dot(treat_states, beta_pi)
        probs = self.lower_clip + (self.upper_clip - self.lower_clip) * jax.nn.sigmoid(self.steepness * logits) # [self.lower_clip, self.upper_clip]
        return probs

    def entropy(self, p):
        # p = jnp.clip(p, 1e-8, 1 - 1e-8)
        return -(p * jnp.log(p) + (1 - p) * jnp.log(1 - p))

    def Q_value(self, Q_states, beta_Q):
        Q_value = jnp.dot(Q_states, beta_Q)
        return Q_value

    def policy_objective(self, beta_pi, beta_Q, base_states, treat_states, actions):
        def single_policy(base_state, treat_state, action, beta_pi, beta_Q):
            # weighted Q values with actions from the policy
            p = self.policy(treat_state, beta_pi)
            Q_states1 = jnp.hstack([base_state, treat_state * jnp.ones_like(action)])
            Q_states0 = jnp.hstack([base_state, treat_state * jnp.zeros_like(action)])
            Q_value1 = self.Q_value(Q_states1, beta_Q)
            Q_value0 = self.Q_value(Q_states0, beta_Q)
            objectives = p * Q_value1 + (1 - p) * Q_value0 + self.lambda_entropy * self.entropy(p)
            return objectives
        J = jnp.mean(jax.vmap(single_policy, in_axes=(0,0,0,None,None))(base_states, treat_states, actions, beta_pi, beta_Q)) # 0: do the batch loop, None: broadcasted / fixed
        return J

    def update_alg(self, all_data, t):
        """
        Update algorithm with new data
        Inputs:
        - `all_data`: a pandas data frame all study data so far
        Outputs:
        - None
        """
        ########## important: only update on the target users: in-study and treatment group Z_id=1 ##########
        in_study_data1 = all_data[all_data["in_study"] == 1] # only include unit data that have happened so far till t
        
        if self.history:
            in_study_data = in_study_data1 # for historical SAC, use all in-study data till time t
            print('Historical SAC update at time t=', t)
        else:
            # for vanilla SAC, only update on the target users at the current time t (different from Tompson Sampling) ##########
            in_study_data = in_study_data1[in_study_data1["calendar_t"] == t]

        # twoarmed trial: only update data with Z_id=1
        if self.twoarmed:
            # only update the algorithm on the treatment group, (similar role of "in_study" column)
            in_study_data = in_study_data[in_study_data['Z_id']==1]
        
        actions = in_study_data["action"].to_numpy()
        action1probs = in_study_data["action1prob"].to_numpy()
        rewards = in_study_data["reward"].to_numpy()
        
        # only form the data with Z_id=1
        base_states, treat_states, next_states = self.get_states(in_study_data)
        Q_states = jnp.hstack([base_states, actions[:, None] * treat_states])
    
        
        def neg_obj(beta_pi):
            # only update the actor parameters given self.betaQ
            return -self.policy_objective(beta_pi, self.betaQ_tar, base_states, treat_states, actions) # here we use self.betaQ_tar
        
        # Updating rule of Q function has a closed-form solution, no need to use gradient descent
        # def q_loss_func(beta_Q):
        #     def single_q_loss(base_state, treat_state, action, next_state, reward):
        #         # current Q values
        #         current_Q_states = jnp.hstack([base_state, treat_state * action])
        #         current_Q_value = self.Q_value(current_Q_states, beta_Q)
        #         # TD target with expected Q values
        #         p_next = self.policy(next_state, self.betapi)
        #         # next_action = self.rng.binomial(1, p_next)
        #         Q_states_next1 = jnp.hstack([next_state, next_state * 1.0])
        #         Q_states_next0 = jnp.hstack([next_state, next_state * 0.0])
        #         Q_values_next1 = self.Q_value(Q_states_next1, beta_Q)
        #         Q_values_next0 = self.Q_value(Q_states_next0, beta_Q)
        #         Q_values_next = p_next * (Q_values_next1 - self.lambda_entropy * jnp.log(p_next + 1e-8)) + (1 - p_next) * (Q_values_next0 - self.lambda_entropy * jnp.log(1 - p_next + 1e-8))
                
        #         # Q_value_next = self.Q_value(Q_states_next, beta_Q)
        #         # logpi = next_action * jnp.log(p_next + 1e-8) + (1 - next_action) * jnp.log(1 - p_next + 1e-8)
        #         TD_target = reward + self.gamma * Q_values_next
        #         loss = (current_Q_value - jax.lax.stop_gradient(TD_target)) ** 2
        #         return loss
        #     total_loss = jnp.mean(jax.vmap(single_q_loss, in_axes=(0,0,0,0,0))(base_states, treat_states, actions, next_states, rewards))
        #     return total_loss

        
        
        
        ################### [Incremental] Update Critic give the last step's policy
        # grad_beta_Q = jax.grad(q_loss_func)(self.betaQ)

        # close-form solution 
        # print('self.betapi_tar', self.betapi_tar)
        p_next = self.policy(next_states, self.betapi_tar) # very important: use the target policy from the last step to evaluate p_next
        # print('next state', next_states)
        # print('p_next', p_next)
        next_action = self.rng.binomial(1, p_next)
        # print("self.betaQ_tar:", self.betaQ_tar)
        Q_states_next = jnp.hstack([next_states, next_states * next_action[:, None]]) 
        Q_values_next = self.Q_value(Q_states_next, self.betaQ_tar) # target network
        # print('Q_values_next', Q_values_next)
        # print("betapi_tar:", self.betapi_tar)
        # print("next_action:", next_action)
        # print('next_states:', next_states)
        # logp_next = jnp.log(jnp.where(next_action.reshape(-1,1)==1, p_next.reshape(-1,1), jnp.clip(1.0 - p_next.reshape(-1,1), 1e-8, 1.0))) 
        logp_next = jnp.log(jnp.where(next_action==1, p_next, jnp.clip(1.0 - p_next, 1e-8, 1.0)))  # [n/2,] log(\pi(A_{t+1}|S_{t+1}))
        # debug.print('betaQ_tar: {}', self.betaQ_tar)
        # debug.print('rewards: {}', rewards)
        # debug.print('Q_values_next: {}', Q_values_next)
        # debug.print('logp_next: {}', logp_next)
        TD_target = rewards + self.gamma * (Q_values_next - self.lambda_entropy * logp_next) # [n/2, ]
        # print("logp_next:", logp_next)
        # print("reward:", rewards)
        # if t > 3:
        #     in_study_data = all_data[all_data["in_study"] == 1]
        #     print(in_study_data[['user_id', 'past_reward', 'reward']])
        #     print(1)

        current_Q_states = jnp.hstack([base_states, treat_states * actions[:, None] ]) # [n/2, beta_dim_Q=4]
        XTX = current_Q_states.T @ current_Q_states
        XTY = current_Q_states.T @ TD_target.flatten()
    

        if self.history:
            n_realunits = in_study_data["user_id"].nunique() * t # treat all time steps as pseudo units
            assert n_realunits == current_Q_states.shape[0]
        else:
            n_realunits = in_study_data["user_id"].nunique()
       
        ################################ multiply n_treat_units here is very important ############################
        # in statistics, we use the average loss, and thus we also scale the ridge penalty by n_treat_units
        betaQ = jnp.linalg.solve(XTX +  n_realunits * self.ridge_penalty * jnp.eye(XTX.shape[0]), XTY) # closed-form solution instead of gradient descent
        # print("TD target mean:", jnp.mean(TD_target), "Q value mean:", jnp.mean(self.Q_value(current_Q_states, self.betaQ)), 'betaQ_tar/pre', self.betaQ_tar, 'betaQ', self.betaQ)
        # print("States", base_states)
        residuals = TD_target - current_Q_states @ betaQ
        # debug.print('betaQ: {}', self.betaQ)
        # debug.print('TD_target: {}', TD_target)
        # debug.print('Current_Q_values: {}', current_Q_states @ self.betaQ)
        # debug.print('current_Q_states: {}', current_Q_states)
        # debug.print('residuals: {}', residuals)
        if self.history:
            vector_Q0 = -2*current_Q_states*residuals.reshape(-1,1) # [nt, beta_dim_Q] * [nt, 1] -> [nt, beta_dim_Q]
            vector_Q = jnp.mean(vector_Q0, axis=0).reshape(-1,1)  + 2*self.ridge_penalty * betaQ.reshape(-1, 1)
            each_unit_Q = vector_Q0.reshape(-1, t, current_Q_states.shape[1]) # [n, t, beta_dim_Q]
            each_unit_Q = jnp.mean(each_unit_Q, axis=1) + 2*self.ridge_penalty * betaQ.reshape(1, -1) # [n, beta_dim_Q]
        else:
            vector_Q = -2*jnp.dot(current_Q_states.T, residuals.reshape(-1,1)) # [beta_dim_Q, n/2] * [n/2, 1] -> [beta_dim_Q, 1]
            vector_Q = vector_Q / n_realunits + 2*self.ridge_penalty * betaQ.reshape(-1, 1)
            each_unit_Q = -2*residuals.reshape(-1,1) * current_Q_states + 2*self.ridge_penalty * betaQ.reshape(1, -1) # [n, beta_dim_Q]
         # [beta_dim_Q, 1] + [beta_dim_Q, 1]
        # debug.print('t {}', t)
        debug.print("averaged vector_Q {}", vector_Q.reshape(1, -1))
        # debug.print("each unit_q {}", each_unit_Q)
       
        
        
        # this manual gradient is used to check the correctness of jax automatic differentiation (which is very close)
        def gradient_pi(beta_pi, beta_Q):
            def single_grad(treat_state):
                p0 = jax.nn.sigmoid(self.steepness * jnp.dot(treat_state, beta_pi))
                p = self.policy(treat_state, beta_pi) 
                temp = jnp.dot(treat_state, beta_Q[-self.beta_dim_pi:]) - self.lambda_entropy * jnp.log(p/(1-p))
                grad = (1-2*self.lower_clip) * p0 * (1 - p0) * self.steepness * temp * treat_state
                return grad
            # return -jnp.mean(jax.vmap(single_grad, in_axes=(0))(treat_states), axis=0) # [2,1] averaged over n
            return jax.vmap(single_grad, in_axes=(0))(treat_states) 
        
        ################### [Incremental] Update Actor based on the last step's critic self.betaQ_tar
        self.lr_pi_use = self.lr_pi
        threshold = 1e-5
        
        #################### in each time step, we randomly intialize the betapi to perform update to mirror TS, where posterior mean and variance are directly computed based on the data and prior mean and varaince
        betapi = jnp.array(self.rng.normal(size=self.beta_dim_pi))
        for i in range(self.epoch_actor):
            loss_pre = neg_obj(betapi)
            grad_beta_pi = jax.grad(neg_obj)(betapi)
            betapi = betapi - self.lr_pi_use * grad_beta_pi
            loss_after = neg_obj(betapi)
            if jnp.linalg.norm(grad_beta_pi) < threshold:
                break
            if (i+1) % 50 == 0:
                # print("grad_beta_pi", grad_beta_pi)
                print(f"SAC update at step {i}, actor gradient: {grad_beta_pi}, loss before: {loss_pre:.3f}, loss after: {loss_after:.3f}", 'betapi', betapi, 'lr_pi', self.lr_pi_use)
            
            if i % 200 == 0 and i >0:
                self.lr_pi_use = self.lr_pi_use * self.decay # decay the learning rate for the actor
        vector_pi = gradient_pi(betapi, self.betaQ_tar) # [n/2 * t, beta_dim_pi]
        # temp = vector_pi.reshape(-1, t, betapi.shape[0]) # [n/2, t, beta_dim_pi]
        # debug.print("vector_pi for each unit {}", jnp.mean(temp, axis=1)) # for each unit
        debug.print("averaged vector_pi {}", jnp.mean(vector_pi, axis=0))
        debug.print("grad_beta_pi {}", -grad_beta_pi)
        

        beta_est = jnp.concatenate([betaQ, betapi])
        # print(in_study_data1[['user_id', 'past_reward', 'reward']])
        num_users_before_update = in_study_data["user_id"].nunique()
        # save Data
        inc_data = {
            "reward": rewards.flatten(),
            "action": actions.flatten(),
            "action1prob": action1probs.flatten(),
            "base_states": base_states,
            "treat_states": treat_states,
            "design": Q_states,
            "next_action": next_action.flatten(),
        }
        update_dict = {
            "beta_est": beta_est,
            "num_users_entered_before_last_update": num_users_before_update, 
            "inc_data": inc_data,
            "betaQ_target": self.betaQ_tar,
            "betapi_target": self.betapi_tar,
        }

        self.all_policies.append(update_dict)

        if self.history:
            pass
        else:
            # !!!!!!!!!!!!! update the target network after each decison time (after saving the self.policy)
            self.betaQ_tar = self.tau * self.betaQ_tar + (1 - self.tau) * betaQ
            self.betapi_tar = self.tau * self.betapi_tar + (1 - self.tau) * betapi
        """
        if state = 2
        {'beta_est': Array([-1.1359017 ,  3.5079467 ,  1.0000004 ,  0.33333334,  0.33333367],      dtype=float32), 
        'num_users_entered_before_last_update': 3, 
        'inc_data': {'reward': array([-1.72244138,  2.37204856, -0.54936675]), 
                    'action': array([0., 1., 0.]), 
                    'action1prob': array([0.50411886, 0.50411886, 0.50411886]),
                      'base_states': array([[1],
                    [1],
                    [1]]), 'treat_states': array([[1],
                    [1],
                    [1]]), 'design': array([[1., 0.],
                    [1., 1.],
                    [1., 0.]])}}
        """
        
    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()


    
    def get_current_beta_estimate(self):
        return self.all_policies[-1]["beta_est"]

    def get_current_betaQ_target(self):
        return self.all_policies[-1]["betaQ_target"]
    
    def get_current_betapi_target(self):
        return self.all_policies[-1]["betapi_target"]
    
    def get_next_action(self):
        return self.all_policies[-1]['inc_data']["next_action"]
    
    def get_num_users_entered_before_last_update(self):
        return self.all_policies[-1]["num_users_entered_before_last_update"]

    def collect_rl_update_args(self, all_prev_data, calendar_t):
        """
        NOTE: Must be called AFTER the update it concerns happens, so that the
        beta the rest of the data already produced is used.
        """
        logger.info(
            "Collecting args to loss/estimating function at time %d (last time included in update data) for each user in dictionary format",
            calendar_t,
        )
        next_policy_num = int(all_prev_data["policy_num"].max() + 1)
        self.rl_update_args[next_policy_num] = {}
        print('In saving collect_rl_update_args, beta:', self.get_current_beta_estimate())
        for i, user_id in enumerate(self.get_all_users(all_prev_data)): # n
            if self.history:
                in_study_user_data = all_prev_data.loc[
                                (all_prev_data.user_id == user_id) & (all_prev_data.in_study == 1)
                            ]
            else:   # for SAC we only save data at the current time point to form the estimation questions !!!!!!!!!!!
                in_study_user_data = all_prev_data.loc[
                    (all_prev_data.user_id == user_id) & (all_prev_data.in_study == 1) & (all_prev_data.calendar_t == calendar_t)
                ]
            ######### the following is consistent with the function parameters in alg_update_func_filename="functions_to_pass_to_analysis/synthetic_SAC_alg_update_function_partial.py
            num_half = int(all_prev_data["user_id"].nunique()/2)  

            temp_next_action = self.get_next_action() # (n/2)*t
            if self.history:
                temp_next_action = temp_next_action.reshape(-1, calendar_t) # [n/2, t]
            # if calendar_t == 2:
            #     print(1)
            self.rl_update_args[next_policy_num][user_id] = (
                (
                    self.get_current_beta_estimate(), # save the model's parameters
                    self.get_current_betaQ_target(), # save the target critic parameters (last step's critic)
                    self.get_current_betapi_target(), # save the target actor parameters (last step's policy)
                    self.get_num_users_entered_before_last_update(),
                    self.get_treat_states(in_study_user_data),
                    self.get_next_states(in_study_user_data),
                    self.get_actions(in_study_user_data),
                    # self.get_next_action()[i%num_half], # the next action is only for the treatment group, which is the same as the in-study group with Z_id=1. For the control group, the next action is always 0, which is not used in the update of the critic and actor. Here we just simply repeat the next action for the control group as the treatment group
                    temp_next_action[i%num_half], # [i, t],
                    self.get_rewards(in_study_user_data),
                    self.lower_clip,
                    self.upper_clip,
                    self.steepness,
                    self.ridge_penalty,
                    self.gamma,
                    in_study_user_data.loc[in_study_user_data['Z_id']==1,'Z_id'].to_numpy() if self.twoarmed else in_study_user_data['Z_id'].to_numpy(), # new
                )
                # We only care about the data overall, however, if there is any
                # in-study data for this user so far
                if not in_study_user_data.empty
                else ()
            )

    # collect the learning policy's information at the current decision time
    def collect_pi_args(self, all_prev_data, calendar_t):
        logger.info(
            "Collecting args to pi function at time %d for each user in dictionary format",
            calendar_t,
        )
        assert calendar_t == jnp.max(all_prev_data["calendar_t"].to_numpy())

        self.pi_args[calendar_t] = {
            user_id: (
                (
                    self.get_current_beta_estimate(),
                    #  self.get_num_users_entered_before_last_update(), # in self.all_policies, updated in the algorithm update function
                    self.lower_clip,
                    self.steepness,
                    self.upper_clip,
                    self.get_treat_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    )[-1],
                    all_prev_data.loc[all_prev_data.user_id == user_id, 'Z_id'].values[-1], # new
                )
                if all_prev_data.loc[
                    (all_prev_data.user_id == user_id)
                    & (all_prev_data.calendar_t == calendar_t)
                ].in_study.item()
                else ()
            )
            for user_id in self.get_all_users(all_prev_data)
        }

    def get_action_probs(self, curr_timestep_data):
        
        ####### curr_timestep_data should be all in-study users at the current time point #########
        treat_states = curr_timestep_data[self.treat_feats].to_numpy() # [n, 2] -> (intercept, past_reward)
        Z_id = curr_timestep_data['Z_id'].to_numpy() # [n, 1]
        # print('beta_estimate',self.get_current_beta_estimate())
        # print('treat_states',treat_states)
        # evalute the action prob for all users accoeding to Z_id
        action_probs = get_pis_batched_sigmoid_partial(
            self.get_current_beta_estimate(),
            self.lower_clip,
            self.steepness,  # steepness fixed to 1.0
            self.upper_clip,
            treat_states,
            Z_id
        )
        return action_probs