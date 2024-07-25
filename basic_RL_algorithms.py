"""
Implementations of several RL algorithms that may be used in study simulations.
"""

import logging

import pandas as pd
import scipy.special
import jax
from jax import numpy as jnp
import numpy as np

from functions_to_pass_to_analysis.get_action_1_prob_pure import get_action_1_prob_pure

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

    def update_alg(self, new_data, t):
        raise NotImplementedError("Fixed randomization never updated")

    def get_action_probs(self, curr_timestep_data, filter_keyval):
        raw_probs = jnp.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob
        return clip(self.args.lower_clip, self.args.upper_clip, raw_probs)


def get_pis_batched(
    beta_est,
    lower_clip,
    steepness,
    upper_clip,
    batched_treat_states_tensor,
):
    return jax.vmap(
        fun=get_action_1_prob_pure,
        in_axes=(None, None, None, None, 0),
        out_axes=0,
    )(
        beta_est,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
    )


# TODO: RL Alg abstract base class
# TODO: Switch back to dataclass and put subfunctions in right/consistent place
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
                "beta_est": pd.DataFrame(
                    jnp.zeros(self.beta_dim).reshape(1, -1),
                    columns=self.state_feats + self.treat_feats_action,
                ),
                "inc_data": {},
                "total_obs": 0,
                "seen_user_id": set(),
            }
        ]
        self.upper_left_bread_inverse = None
        self.algorithm_statistics_by_calendar_t = {}
        self.action_centering = action_centering

    # TODO: All of these functions arguably should not modify the dataframe...
    # Should be making a new dataframe and modifying that, or expecting the data
    # to be formatted as such (though I don't like the latter). Going with this
    # for now. This modification also raises a warning about setting a slice on
    # a copy, but it seems to work perfectly.

    # TODO: Docstring
    def get_base_states(self, df, in_study_col="in_study"):
        df.loc[df[in_study_col] == 0, self.state_feats] = 0
        base_states = df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, df, in_study_col="in_study"):
        df.loc[df[in_study_col] == 0, self.treat_feats] = 0
        treat_states = df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)

    def get_rewards(self, df, reward_col="reward", in_study_col="in_study"):
        df.loc[df[in_study_col] == 0, reward_col] = 0
        rewards = df[reward_col].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    def get_actions(self, df, action_col="action", in_study_col="in_study"):
        df.loc[df[in_study_col] == 0, action_col] = 0
        actions = df[action_col].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    def get_action1probs(
        self,
        df,
        actionprob_col="action1prob",
        in_study_col="in_study",
    ):
        df.loc[df[in_study_col] == 0, actionprob_col] = 0
        action1probs = df[actionprob_col].to_numpy(dtype="float64").reshape(-1, 1)
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

        # NOTE: this gives NANs and breaks action selection when the all
        # users take same action
        inv_XX = jnp.linalg.inv(new_XX)

        beta_est = jnp.matmul(inv_XX, new_RX.reshape(-1))
        beta_est_df = pd.DataFrame(
            beta_est.reshape(1, -1), columns=self.state_feats + self.treat_feats_action
        )

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
            "beta_est": beta_est_df,
            "inc_data": inc_data,
            "seen_user_id": seen_user_id,
        }

        self.all_policies.append(update_dict)

    def get_active_users(
        self, study_df, in_study_column="in_study", user_id_column="user_id"
    ):
        return study_df[study_df[in_study_column] == 1][user_id_column].unique()

    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()

    def get_current_beta_estimate(self):
        return self.all_policies[-1]["beta_est"].to_numpy().squeeze()

    def collect_rl_update_args(
        self, all_prev_data, study_df, calendar_t, curr_beta_est
    ):
        logger.info(
            "Collecting args to loss/estimating function at time %d (last time included in update data) for each user in dictionary format",
            calendar_t,
        )
        next_policy_num = int(all_prev_data["policy_num"].max() + 1)
        first_applicable_time = calendar_t + 1
        self.rl_update_args[next_policy_num] = {
            user_id: (
                (
                    curr_beta_est,
                    self.get_base_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    ),
                    self.get_treat_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    ),
                    self.get_actions(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    ),
                    self.get_rewards(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    ),
                    self.get_action1probs(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    ),
                    self.action_centering,
                )
                if study_df.loc[
                    (study_df.user_id == user_id)
                    & (study_df.calendar_t == first_applicable_time)
                ].in_study.item()
                else ()
            )
            for user_id in self.get_all_users(all_prev_data)
        }

    def collect_pi_args(self, all_prev_data, calendar_t, curr_beta_est):
        logger.info(
            "Collecting args to pi function at time %d for each user in dictionary format",
            calendar_t,
        )
        assert calendar_t == jnp.max(all_prev_data["calendar_t"].to_numpy())

        self.pi_args[calendar_t] = {
            user_id: (
                (
                    curr_beta_est,
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

    def get_action_probs_inner(self, beta_est, prob_input_dict):
        """
        Form action selection probabilities from raw inputs (used to form importance weights)

        Inputs:
        - `beta_est`: Algorithm's beta estimators
        - `prob_input_dict`: Dictionary of other information needed to form action selection probabilities
            This dictionary should include:
                - `treat_states` (matrix where each row is a state vector that interacts with treatment effect model)

        Outputs:
        - Numpy vector of action selection probabilities
        """

        treat_est = beta_est[self.treat_bool]
        lin_est = jnp.matmul(prob_input_dict["treat_states"], treat_est)

        raw_probs = scipy.special.expit(self.steepness * lin_est)
        # TODO: hiding things in args like this makes code harder to follow
        # TODO: can use np clip
        probs = clip(self.lower_clip, self.upper_clip, raw_probs)

        return probs.squeeze()

    def get_action_probs(self, curr_timestep_data, filter_keyval=None):
        """
        Form action selection probabilities from newly current data (only use when running RL algorithm)

        Inputs:
        - `curr_timestep_data`: Pandas data frame of current data that can be used to form the states
        - `filter_keyval`: None (not needed for this algorithm)

        Outputs:
        - Numpy vector of action selection probabilities
        """

        beta_est_df = self.all_policies[-1]["beta_est"].copy()
        beta_est = beta_est_df.to_numpy()

        treat_states = curr_timestep_data[self.treat_feats].to_numpy()

        return get_pis_batched(
            beta_est.squeeze(),
            self.lower_clip,
            self.steepness,
            self.upper_clip,
            treat_states,
        )
