"""
Implementations of several RL algorithms that may be used in study simulations.
"""

import logging

import pandas as pd
import scipy.special
import jax
from jax import numpy as jnp
import numpy as np

from helper_functions import (
    conditional_x_or_one_minus_x,
    clip,
)

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

    def get_pi_gradients(self, user_states):
        raise NotImplementedError("Fixed randomization no policy gradients")


# TODO: should be able to stack all users. Update IDK about this for differentiation reasons.
# Update Update: We *could* differentiate for all users at once with jacrev.
# TODO: Docstring
# @jax.jit
def get_loss(
    beta_est,
    base_states,
    treat_states,
    actions,
    rewards,
    action1probs,
    action_centering,
):
    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    actions = jnp.where(
        action_centering, actions.astype(jnp.float32) - action1probs, actions
    )

    return jnp.sum(
        (
            rewards
            - jnp.matmul(base_states, beta_0_est)
            - jnp.matmul(actions * treat_states, beta_1_est)
        )
        ** 2,
    )


# Consider jitting
# For the loss gradients, we can form the sum of all users' values and differentiate that with one
# call. Instead, this alternative structure which generalizes to the pi function case.
# TODO: Docstring
@jax.jit
def get_loss_gradients_batched(
    beta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.grad(get_loss),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        beta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


# TODO: Docstring
@jax.jit
def get_loss_hessians_batched(
    beta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.hessian(get_loss),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        beta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


def get_loss_gradient_derivatives_wrt_pi_batched(
    beta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
    action1probs_batch,
    action_centering,
):
    return jax.vmap(
        fun=jax.jacrev(jax.grad(get_loss), 5),
        in_axes=(None, 2, 2, 2, 2, 2, None),
        out_axes=0,
    )(
        beta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
        action1probs_batch,
        action_centering,
    )


# TODO: Docstring
def get_action_1_prob_pure(beta_est, lower_clip, steepness, upper_clip, treat_states):
    treat_est = beta_est[-len(treat_states) :]
    lin_est = jnp.matmul(treat_states, treat_est)
    raw_prob = jax.scipy.special.expit(steepness * lin_est)

    return jnp.clip(raw_prob, lower_clip, upper_clip)[()]


# TODO: Docstring
def get_pi_gradients_batched(
    beta_est,
    lower_clip,
    steepness,
    upper_clip,
    batched_treat_states_tensor,
):
    return jax.vmap(
        fun=jax.grad(get_action_1_prob_pure),
        in_axes=(None, None, None, None, 0),
        out_axes=0,
    )(
        beta_est,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
    )


# TODO: Docstring
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


# TODO: Docstring
def get_radon_nikodym_weight(
    beta, beta_target, lower_clip, steepness, upper_clip, treat_states, action
):
    common_args = [lower_clip, steepness, upper_clip, treat_states]

    pi_beta = get_action_1_prob_pure(beta, *common_args)[()]
    pi_beta_target = get_action_1_prob_pure(beta_target, *common_args)[()]
    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


# TODO: Docstring
def get_weight_gradients_batched(
    beta_est,
    target_beta,
    lower_clip,
    steepness,
    upper_clip,
    batched_treat_states_tensor,
    batched_actions_tensor,
):
    return jax.vmap(
        fun=jax.grad(get_radon_nikodym_weight),
        in_axes=(None, None, None, None, None, 0, 0),
        out_axes=0,
    )(
        beta_est,
        target_beta,
        lower_clip,
        steepness,
        upper_clip,
        batched_treat_states_tensor,
        batched_actions_tensor,
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

    # TODO: Docstring
    # TODO: JIT whole function? or just gradient and hessian batch functions
    def calculate_loss_derivatives(self, all_prev_data, calendar_t, curr_beta_est):
        logger.info("Calculating loss gradients and hessians with respect to beta.")

        # Because we perform algorithm updates at the *end* of a timestep, the
        # first timestep they apply to is one more than the time of the update.
        # Hence we add 1 here for notational consistency with the paper.

        # TODO: Note that we don't need the loss gradient for the first update
        # time... include anyway?
        first_applicable_time = calendar_t + 1

        # Typically not necessary, but just be safe...
        # TODO: Only if grouping by user id in numpy
        # all_prev_data.sort_values(by=["user_id", "calendar_t"])

        # Do this pandas filtering only once. Also consider doing it in numpy?
        # Or just do everything in matrix form somehow?
        # https://stackoverflow.com/questions/31863083/python-split-numpy-array-based-on-values-in-the-array
        batched_base_states_list = []
        batched_treat_states_list = []
        batched_actions_list = []
        batched_rewards_list = []
        batched_action1probs_list = []

        # This step is the bottleneck, interestingly
        logger.info("Collecting batched input data as lists.")

        # Note that we want to get
        all_user_ids = self.get_all_users(all_prev_data)
        for user_id in all_user_ids:
            filtered_user_data = all_prev_data.loc[all_prev_data.user_id == user_id]
            batched_base_states_list.append(self.get_base_states(filtered_user_data))
            batched_treat_states_list.append(self.get_treat_states(filtered_user_data))
            batched_actions_list.append(self.get_actions(filtered_user_data))
            batched_rewards_list.append(self.get_rewards(filtered_user_data))
            batched_action1probs_list.append(self.get_action1probs(filtered_user_data))

        logger.info("Reforming batched data lists into tensors.")
        batched_base_states_tensor = np.dstack(batched_base_states_list)
        batched_treat_states_tensor = np.dstack(batched_treat_states_list)
        batched_actions_tensor = np.dstack(batched_actions_list)
        batched_rewards_tensor = np.dstack(batched_rewards_list)
        batched_action1probs_tensor = jnp.dstack(batched_action1probs_list)

        logger.info("Forming loss gradients with respect to beta.")
        gradients = get_loss_gradients_batched(
            curr_beta_est,
            batched_base_states_tensor,
            batched_treat_states_tensor,
            batched_actions_tensor,
            batched_rewards_tensor,
            batched_action1probs_tensor,
            self.action_centering,
        )
        logger.info("Forming loss hessians with respect to beta")
        hessians = get_loss_hessians_batched(
            curr_beta_est,
            batched_base_states_tensor,
            batched_treat_states_tensor,
            batched_actions_tensor,
            batched_rewards_tensor,
            batched_action1probs_tensor,
            self.action_centering,
        )
        logger.info(
            "Forming derivatives of loss with respect to beta and then the action probabilites vector at each time"
        )
        loss_gradient_pi_derivatives = get_loss_gradient_derivatives_wrt_pi_batched(
            curr_beta_est,
            batched_base_states_tensor,
            batched_treat_states_tensor,
            batched_actions_tensor,
            batched_rewards_tensor,
            batched_action1probs_tensor,
            self.action_centering,
        )

        self.algorithm_statistics_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {user_id: gradients[i] for i, user_id in enumerate(all_user_ids)}

        self.algorithm_statistics_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = np.mean(hessians, axis=0)

        self.algorithm_statistics_by_calendar_t[first_applicable_time][
            "loss_gradient_pi_derivatives_by_user_id"
        ] = {
            user_id: loss_gradient_pi_derivatives[i].squeeze()
            for i, user_id in enumerate(all_user_ids)
        }

    def calculate_pi_and_weight_gradients(
        self, current_data, calendar_t, curr_beta_est
    ):
        """
        For all users, compute the gradient with respect to beta of both the pi function
        (which takes an action and state and gives the probability of selecting the action)
        and the Radon-Nikodym weight (derived from pi functions as described in the paper)
        for the current time, evaluated at the most recent beta.

        Note that we take the *latest* states and actions from current_data, and assume that
        corresponds to both the most recent beta estimated and the supplied calendar_t.

        The data is saved in `self.algorithm_statistics_by_calendar_t` under the current time.
        See the keys below.

        Inputs:
        - `current_data`: a pandas data frame with study data. Note that only the data
        at calendar_t is needed.
        - `calendar_t`: the current calendar time

        Outputs:
        - None
        """

        logger.info("Calculating pi and weight gradients with respect to beta.")
        assert calendar_t == jnp.max(current_data["calendar_t"].to_numpy())

        batched_treat_states_list = []
        batched_actions_list = []

        all_user_ids = self.get_all_users(current_data)

        logger.info("Collecting batched input data as lists.")
        for user_id in all_user_ids:
            filtered_user_data = current_data.loc[current_data.user_id == user_id]
            batched_treat_states_list.append(
                self.get_treat_states(filtered_user_data)[-1]
            )
            batched_actions_list.append(self.get_actions(filtered_user_data)[-1].item())

        # Note this stacking works with incremental recruitment only because we
        # fill in states for out-of-study times such that all users have the
        # same state matrix size
        logger.info("Reforming batched data lists into tensors.")
        batched_treat_states_tensor = jnp.vstack(batched_treat_states_list)
        batched_actions_tensor = jnp.array(batched_actions_list)

        logger.info("Forming pi gradients with respect to beta.")
        # Note that we care about the probability of action 1 specifically,
        # not the taken action.
        pi_gradients = get_pi_gradients_batched(
            curr_beta_est,
            self.lower_clip,
            self.steepness,
            self.upper_clip,
            batched_treat_states_tensor,
        )

        logger.info("Forming weight gradients with respect to beta.")
        weight_gradients = get_weight_gradients_batched(
            curr_beta_est,
            curr_beta_est,
            self.lower_clip,
            self.steepness,
            self.upper_clip,
            batched_treat_states_tensor,
            batched_actions_tensor,
        )

        logger.info("Collecting pi gradients into algorithm stats dictionary.")
        self.algorithm_statistics_by_calendar_t.setdefault(calendar_t, {})[
            "pi_gradients_by_user_id"
        ] = {user_id: pi_gradients[i] for i, user_id in enumerate(all_user_ids)}

        logger.info("Collecting weight gradients into algorithm stats dictionary.")
        self.algorithm_statistics_by_calendar_t.setdefault(calendar_t, {})[
            "weight_gradients_by_user_id"
        ] = {user_id: weight_gradients[i] for i, user_id in enumerate(all_user_ids)}

    # TODO: Docstring
    # TODO: Use jnp.block to reduce need for indexing
    # TODO: JIT this function?? update times would need to be adjusted (pass them in actually)
    # Make sure jitting makes sense across simulations in terms of common args
    def construct_upper_left_bread_inverse(self):
        logger.info("Constructing upper left bread inverse.")
        # Form the dimensions for our bread matrix portion (pre-inverting). Note that we subtract
        # one from the number of policies  to find the number of updates because there is an
        # initial placeholder policy.

        # TODO: derive from policy_num column or list of betas
        num_updates = len(self.all_policies) - 1
        # TODO: derive from an actual beta
        beta_dim = len(self.state_feats) + len(self.treat_feats)
        overall_dim = beta_dim * num_updates
        output_matrix = jnp.zeros((overall_dim, overall_dim))

        # List of times that were the first applicable time for some update
        # TODO: sort to not rely on insertion order?
        # TODO: use policy_num in df? alg statistics potentially ok too though.
        next_times_after_update = [
            t
            for t, value in self.algorithm_statistics_by_calendar_t.items()
            if "loss_gradients_by_user_id" in value
        ]

        # TODO: use study_df.  Verify we want all users
        user_ids = self.all_policies[-1]["seen_user_id"]
        num_users = len(user_ids)

        # This simply collects the pi derivatives with respect to betas for all
        # decision times for each user. The one complication is that we add some
        # padding of zeros for decision times before the first update to make
        # indexing simpler below.
        # NOTE THAT ROW INDEX i CORRESPONDS TO DECISION TIME i+1!
        pi_derivatives_by_user_id = {
            user_id: jnp.pad(
                jnp.array(
                    [
                        t_dict["pi_gradients_by_user_id"][user_id]
                        for t_dict in self.algorithm_statistics_by_calendar_t.values()
                    ]
                ),
                pad_width=((next_times_after_update[0] - 1, 0), (0, 0)),
            )
            for user_id in user_ids
        }

        # This loop iterates over all times that were the first applicable time
        # for a non-initial policy. Take care to note that update_idx starts at 0.
        # Think of each iteration of this loop as creating a (block) row of the matrix
        for update_idx, update_t in enumerate(next_times_after_update):
            logger.info("Processing update time %s.", update_t)
            t_stats_dict = self.algorithm_statistics_by_calendar_t[update_t]

            # This loop creates the non-diagonal terms for the current update
            # Think of each iteration of this loop as creating one term in the current (block) row
            logger.info("Creating the non-diagonal terms for the current update.")
            for i in range(update_idx):
                lower_t = next_times_after_update[i]
                upper_t = next_times_after_update[i + 1]
                running_entry_holder = jnp.zeros((beta_dim, beta_dim))

                # This loop calculates the per-user quantities that will be
                # averaged for the final matrix entries
                for user_id, loss_gradient in t_stats_dict[
                    "loss_gradients_by_user_id"
                ].items():
                    weight_gradient_sum = jnp.zeros(beta_dim)

                    # This loop iterates over decision times in slices
                    # according to what was used for each update to collect the
                    # right weight gradients
                    for t in range(
                        lower_t,
                        upper_t,
                    ):
                        weight_gradient_sum += self.algorithm_statistics_by_calendar_t[
                            t
                        ]["weight_gradients_by_user_id"][user_id]

                    running_entry_holder += jnp.outer(
                        loss_gradient,
                        weight_gradient_sum,
                    )

                    # TODO: Detailed comment explaining this logic and the data
                    # orientation that makes it work.  Also note the assumption
                    # that the estimating function is additive across times
                    # so that matrix multiplication is the right operation. Also
                    # place this comment on the after study analysis logic or
                    # link to the same explanation in both places.
                    # Maybe link to a document with a picture...

                    mixed_theta_beta_loss_derivative = jnp.matmul(
                        t_stats_dict["loss_gradient_pi_derivatives_by_user_id"][
                            user_id
                        ][
                            :,
                            lower_t - 1 : upper_t - 1,
                        ],
                        pi_derivatives_by_user_id[user_id][
                            lower_t - 1 : upper_t - 1,
                            :,
                        ],
                    )
                    running_entry_holder += mixed_theta_beta_loss_derivative
                # TODO: Use jnp.block instead of indexing
                output_matrix = output_matrix.at[
                    (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                    i * beta_dim : (i + 1) * beta_dim,
                ].set(running_entry_holder / num_users)

            # Add the diagonal hessian entry (which is already an average)
            # TODO: Use jnp.block instead of indexing
            output_matrix = output_matrix.at[
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
            ].set(t_stats_dict["avg_loss_hessian"])

        self.upper_left_bread_inverse = output_matrix

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
