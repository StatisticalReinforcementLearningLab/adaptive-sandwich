"""
Implementations of several RL algorithms that may be used in study simulations.
"""

import logging

import jax
import numpy as np
from jax import numpy as jnp

from functions_to_pass_to_analysis.RL_least_squares_loss_regularized import (
    RL_least_squares_loss_regularized,
)
from functions_to_pass_to_analysis.smooth_thompson_sampling_act_prob_function_no_action_centering import (
    smooth_thompson_sampling_act_prob_function_no_action_centering,
)
from functions_to_pass_to_analysis.synthetic_get_action_1_prob_pure import (
    synthetic_get_action_1_prob_pure,
)
from functions_to_pass_to_analysis.synthetic_get_action_1_prob_generalized_logistic import (
    synthetic_get_action_1_prob_generalized_logistic,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)


def get_pis_batched_sigmoid(
    beta_est,
    lower_clip,
    steepness,
    upper_clip,
    smooth_clip,
    batched_treat_states_tensor,
):
    return jax.vmap(
        fun=(
            synthetic_get_action_1_prob_generalized_logistic
            if smooth_clip
            else synthetic_get_action_1_prob_pure
        ),
        in_axes=(None, None, None, None, 0),
        out_axes=0,
    )(
        beta_est,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
    )


def get_pis_batched_thompson_sampling(
    beta_est,
    batched_treat_states_tensor,
    num_users_entered_before_last_update,
    lower_clip,
    upper_clip,
    steepness,
):
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
        collect_args_to_reconstruct_action_probs,
        lambda_,
        smooth_clip,
    ):
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        # Note that steepness = 0 should remove learning adaptivity, with action probabilities of .5
        self.steepness = steepness
        self.lower_clip = lower_clip
        self.upper_clip = upper_clip
        self.lambda_ = lambda_
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
        self.collect_args_to_reconstruct_action_probs = (
            collect_args_to_reconstruct_action_probs
        )
        self.incremental_updates = True
        self.smooth_clip = smooth_clip

        # These are used for passing to a after-study-analysis bread inverse conditioning
        # monitor if desired
        self.action_prob_func = (
            synthetic_get_action_1_prob_generalized_logistic
            if smooth_clip
            else synthetic_get_action_1_prob_pure
        )
        self.action_prob_func_args_beta_index = 0

        self.alg_update_func = RL_least_squares_loss_regularized
        self.alg_update_func_type = "loss"
        self.alg_update_func_args_beta_index = 0
        self.alg_update_func_args_action_prob_index = 5
        self.alg_update_func_args_action_prob_times_index = 6

    # TODO: All of these functions arguably should not modify the dataframe...
    # Should be making a new dataframe and modifying that, or expecting the data
    # to be formatted as such (though I don't like the latter). Going with this
    # for now. This modification also raises a warning about setting a slice on
    # a copy, but it seems to work perfectly.

    # TODO: Docstring
    def get_base_states(self, in_study_df):
        base_states = in_study_df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, in_study_df):
        treat_states = in_study_df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)

    def get_rewards(self, in_study_df, reward_col="reward"):
        rewards = in_study_df[reward_col].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    def get_actions(self, in_study_df, action_col="action"):
        actions = in_study_df[action_col].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    def get_action1probs(
        self,
        in_study_df,
        actionprob_col="action1prob",
    ):
        action1probs = (
            in_study_df[actionprob_col].to_numpy(dtype="float32").reshape(-1, 1)
        )
        return jnp.array(action1probs)

    def get_action1probstimes(
        self,
        in_study_df,
        calendar_t_col="calendar_t",
    ):
        action1probstimes = (
            in_study_df[calendar_t_col].to_numpy(dtype="float32").reshape(-1, 1)
        )
        return jnp.array(action1probstimes)

    def get_initial_policy_num(self, full_df, policy_num_col="policy_num"):
        policy_nums = full_df[policy_num_col]
        nonnegative_policy_nums = policy_nums[policy_nums >= 0]
        return nonnegative_policy_nums.min() if len(nonnegative_policy_nums) > 0 else 0

    def get_post_update_policy_nums(
        self,
        in_study_df,
        initial_policy_num,
        policy_num_col="policy_num",
    ):
        """
        Get the policy numbers for all decision times for this user after the
        first update. Note that this only includes decision times where the user is in-study,
        so this may include ALL of a user's decision times if the user entered
        after the first update.
        """
        df_post_update = in_study_df[in_study_df[policy_num_col] > initial_policy_num]
        post_update_policy_nums = (
            df_post_update[policy_num_col].to_numpy(dtype="int32").reshape(-1, 1)
        )
        return jnp.array(post_update_policy_nums)

    def get_pre_update_action1probs(
        self,
        in_study_df,
        initial_policy_num,
        actionprob_col="action1prob",
        policy_num_col="policy_num",
    ):
        """
        Note that this is the pre-update action1probs that are IN-STUDY for the given
        user. So there may be none if the user only entered after the first update.
        """

        df_policy_initial = in_study_df[
            in_study_df[policy_num_col] == initial_policy_num
        ]
        action1probs = (
            df_policy_initial[actionprob_col].to_numpy(dtype="float32").reshape(-1, 1)
        )
        return jnp.array(action1probs)

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
        # users take same action, at least with no regularization
        inv_XX = jnp.linalg.inv(new_XX + self.lambda_ * np.eye(design.shape[1]))

        # TODO: Do this multiplication with a solve instead, or use QR decomposition even.
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

    def get_previous_post_update_betas(self):
        # Exclude initial policy and current policy
        # The reshape is so that the empty output is 2D, with shape representing
        # 0 rows of size beta_dim
        return jnp.array(
            [policy["beta_est"] for policy in self.all_policies[1:-1]]
        ).reshape(-1, self.beta_dim)

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
        initial_policy_num = self.get_initial_policy_num(all_prev_data)
        self.rl_update_args[next_policy_num] = {}

        if not self.collect_args_to_reconstruct_action_probs:
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
                        self.lambda_,
                        len(all_prev_data["user_id"].unique()),
                    )
                    # We only care about the data overall, however, if there is any
                    # in-study data for this user so far
                    if not in_study_user_data.empty
                    else ()
                )
        else:
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
                        self.get_action1probs(in_study_user_data),
                        self.get_pre_update_action1probs(
                            in_study_user_data, initial_policy_num
                        ),
                        self.get_previous_post_update_betas(),
                        self.get_post_update_policy_nums(
                            in_study_user_data, initial_policy_num
                        ),
                        self.lower_clip,
                        self.steepness,
                        self.upper_clip,
                        self.action_centering,
                        self.lambda_,
                        len(all_prev_data["user_id"].unique()),
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
            self.smooth_clip,
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

    @staticmethod
    def form_beta_from_posterior(
        posterior_mean: np.ndarray,
        posterior_var: np.ndarray,
        num_users_entered_before_update: int,
    ) -> jnp.ndarray:
        """
        Form the beta vector from the posterior mean and variance.
        This is for after-study analysis, concisely collecting all the information
        in the posterior in a convenient form. Explicitly, we concatenate the posterior
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

        in_study_data = all_data[all_data["in_study"] == 1]
        # update algorithm with new data
        actions = in_study_data["action"].to_numpy()
        action1probs = in_study_data["action1prob"].to_numpy()
        rewards = in_study_data["reward"].to_numpy()
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

        posterior_mean, posterior_var = self.compute_posterior(
            design,
            rewards,
        )

        num_users_before_update = in_study_data["user_id"].nunique()
        beta_est = self.form_beta_from_posterior(
            posterior_mean,
            posterior_var,
            num_users_before_update,
        )

        # Save Data
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
            "num_users_entered_before_last_update": num_users_before_update,
            "inc_data": inc_data,
        }

        self.all_policies.append(update_dict)

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
            self.rl_update_args[next_policy_num][user_id] = (
                (
                    self.get_current_beta_estimate(),
                    self.get_num_users_entered_before_last_update(),
                    self.get_treat_states(in_study_user_data),
                    self.get_actions(in_study_user_data),
                    self.get_rewards(in_study_user_data),
                    self.prior_mu,
                    jnp.linalg.inv(self.prior_sigma),
                    self.noise_var,
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
                    self.get_treat_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    )[-1],
                    self.get_num_users_entered_before_last_update(),
                    self.lower_clip,
                    self.upper_clip,
                    self.steepness,
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
        treat_states = curr_timestep_data[self.treat_feats].to_numpy()

        return get_pis_batched_thompson_sampling(
            self.get_current_beta_estimate(),
            treat_states,
            self.get_num_users_entered_before_last_update(),
            self.lower_clip,
            self.upper_clip,
            self.steepness,
        )
