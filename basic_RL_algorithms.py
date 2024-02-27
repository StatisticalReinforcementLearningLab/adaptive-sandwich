"""
Implementations of several RL algorithms that may be used in study simulations.
"""

import logging

import pandas as pd
import scipy.special
import torch
import numpy_indexed as npi
import jax
from jax import numpy as jnp
import numpy as np

import least_squares_helper
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
        raise ValueError("Fixed randomization never updated")

    def get_action_probs(self, curr_timestep_data, filter_keyval):
        raw_probs = jnp.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob
        return clip(self.args, raw_probs)

    def get_pi_gradients(self, user_states):
        raise ValueError("Fixed randomization no policy gradients")

    def get_est_eqn(self, data_sofar):
        raise ValueError(
            "Fixed randomization no need for \
                                     estimating equation of policy"
        )


def torch_clip(args, vals):
    lower_clipped = torch.max(vals, torch.ones(vals.shape) * args.lower_clip)
    clipped = torch.min(lower_clipped, torch.ones(vals.shape) * args.upper_clip)
    return clipped


def sigmoid_LS_torch(args, batch_est_treat, treat_states, allocation_sigma):
    # States
    treat_states_torch = torch.from_numpy(treat_states.to_numpy())

    # Form Probabilities
    lin_est = torch.sum(batch_est_treat * treat_states_torch, axis=1)
    pis = torch_clip(args, torch.sigmoid(args.steepness * lin_est / allocation_sigma))

    # below genralized logistic (different asymptotes)
    # pis = torch_clip( args, args.lower_clip +
    #                 (args.upper_clip-args.lower_clip) * torch.sigmoid( args.steepness*lin_est/allocation_sigma ) )

    return pis


# TODO: should be able to stack all users. Update IDK about this for differentiation reasons.
# Update Update: We can differentiate for all users at once with jacrev.
# TODO: Unite with update_alg logic somehow?
# TODO: Docstring
@jax.jit
def get_loss(beta_est, base_states, treat_states, actions, rewards):
    beta_0_est = beta_est[: base_states.shape[1]].reshape(-1, 1)
    beta_1_est = beta_est[base_states.shape[1] :].reshape(-1, 1)

    return jnp.einsum(
        "ij->",
        (
            rewards
            - jnp.einsum("ij,jk->ik", base_states, beta_0_est)
            - jnp.einsum("ij,jk->ik", actions * treat_states, beta_1_est)
        )
        ** 2,
    )


# Consider jitting
# For the loss gradients, we can form the sum of all users values and differentiate that with one
# call. Instead, this alternative structure which generalizes to the pi function case.
# TODO: Docstring
@jax.jit
def get_loss_gradients_batched(
    beta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
):
    return jax.vmap(
        fun=jax.grad(get_loss),
        in_axes=(None, 2, 2, 2, 2),
        out_axes=0,
    )(
        beta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
    )


# TODO: Docstring
@jax.jit
def get_loss_hessians_batched(
    beta_est,
    batched_base_states_tensor,
    batched_treat_states_tensor,
    actions_batch,
    rewards_batch,
):
    return jax.vmap(
        fun=jax.hessian(get_loss),
        in_axes=(None, 2, 2, 2, 2),
        out_axes=0,
    )(
        beta_est,
        batched_base_states_tensor,
        batched_treat_states_tensor,
        actions_batch,
        rewards_batch,
    )


# TODO: Unite with method that does same thing. Needed pure function here.
# Might need to use jacrev to do so
# TODO: Docstring
# See https://arxiv.org/pdf/2006.06903.pdf
# TODO: Docstring
@jax.jit
def get_action_prob_pure(beta_est, lower_clip, upper_clip, treat_states, action=1):
    treat_est = beta_est[-len(treat_states) :]
    lin_est = jnp.matmul(treat_states, treat_est)

    raw_prob = jax.scipy.special.expit(lin_est)
    prob = conditional_x_or_one_minus_x(
        jnp.clip(raw_prob, lower_clip, upper_clip), action
    )

    return prob[()]


# TODO: Docstring
@jax.jit
def get_pi_gradients_batched(
    beta_est,
    lower_clip,
    upper_clip,
    batched_treat_states_tensor,
    batched_actions_tensor,
):
    return jax.vmap(
        fun=jax.grad(get_action_prob_pure),
        in_axes=(None, None, None, 0, 0),
        out_axes=0,
    )(
        beta_est,
        lower_clip,
        upper_clip,
        batched_treat_states_tensor,
        batched_actions_tensor,
    )


# TODO: Docstring
@jax.jit
def get_radon_nikodym_weight(
    beta, beta_target, lower_clip, upper_clip, treat_states, action
):
    common_args = [lower_clip, upper_clip, treat_states]

    pi_beta = get_action_prob_pure(beta, *common_args)[()]
    pi_beta_target = get_action_prob_pure(beta_target, *common_args)[()]
    return conditional_x_or_one_minus_x(pi_beta, action) / conditional_x_or_one_minus_x(
        pi_beta_target, action
    )


# TODO: Docstring
@jax.jit
def get_weight_gradients_batched(
    beta_est,
    target_beta,
    lower_clip,
    upper_clip,
    batched_treat_states_tensor,
    batched_actions_tensor,
):
    return jax.vmap(
        fun=jax.grad(get_radon_nikodym_weight),
        in_axes=(None, None, None, None, 0, 0),
        out_axes=0,
    )(
        beta_est,
        target_beta,
        lower_clip,
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
        self, args, state_feats, treat_feats, alg_seed, allocation_sigma, steepness
    ):
        self.args = args
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.allocation_sigma = allocation_sigma
        # Note that steepness = 0 removes learning adaptivity, with action probabilities of .5
        self.steepness = steepness

        self.rng = np.random.default_rng(self.alg_seed)
        self.beta_dim = len(self.state_feats) + len(self.treat_feats)
        # Set an initial policy
        self.all_policies = [
            {
                "policy_last_t": 0,
                "RX": jnp.zeros(self.beta_dim),
                "XX": jnp.zeros(self.beta_dim),
                "beta_est": pd.DataFrame(),
                # TODO: verify changing this to inc_data and {} instead of None is ok. Was inconsistent
                # before. also changed beta est to empty df
                # Then standardize this format by making a function that takes all the args
                # This could possibly be more broad than for sigmoid only.
                "inc_data": {},
                "total_obs": 0,
                "seen_user_id": set(),
            }
        ]

        self.treat_feats_action = ["action:" + x for x in self.treat_feats]
        self.treat_bool = jnp.array(
            [
                x in self.treat_feats_action
                for x in self.state_feats + self.treat_feats_action
            ]
        )
        self.upper_left_bread_inverse = None
        self.algorithm_statistics_by_calendar_t = {}

    # TODO: Docstring
    def get_base_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        return jnp.array(base_states)

    def get_treat_states(self, df):
        treat_states = df[self.treat_feats].to_numpy()
        return jnp.array(treat_states)

    # TODO: Allow general reward column names
    def get_rewards(self, df):
        rewards = df["reward"].to_numpy().reshape(-1, 1)
        return jnp.array(rewards)

    # TODO: Allow general action column names
    def get_actions(self, df):
        actions = df["action"].to_numpy().reshape(-1, 1)
        return jnp.array(actions)

    # TODO: Docstring
    def get_states(self, df):
        base_states = df[self.state_feats].to_numpy()
        treat_states = df[self.treat_feats].to_numpy()
        return (base_states, treat_states)

    # TODO: Make sure correct while debugging simulations
    def update_alg(self, new_data, update_last_t):
        """
        Update algorithm with new data

        Inputs:
        - `new_data`: a pandas data frame with new data
        - `update_last_t`: an integer representing the last calendar time
            of data that was used to update the algorithm

        Outputs:
        - None
        """

        # update algorithm with new data
        actions = new_data["action"].to_numpy().reshape(-1, 1)
        action1probs = new_data["action1prob"].to_numpy().reshape(-1, 1)
        rewards = new_data["reward"].to_numpy().reshape(-1, 1)
        base_states, treat_states = self.get_states(new_data)
        design = jnp.concatenate([base_states, actions * treat_states], axis=1)

        # Only include available data
        calendar_t = new_data["calendar_t"].to_numpy().reshape(-1, 1)
        user_t = new_data["user_t"].to_numpy().reshape(-1, 1)
        rewards_avail = rewards
        avail_bool = jnp.ones(rewards.shape)
        design_avail = design
        user_id_avail = new_data["user_id"].to_numpy()

        # Get policy estimator
        new_RX = self.all_policies[-1]["RX"] + jnp.sum(design_avail * rewards_avail, 0)
        new_XX = self.all_policies[-1]["XX"] + jnp.einsum(
            "ij,ik->jk", design_avail, design_avail
        )
        try:
            inv_XX = jnp.linalg.inv(new_XX)
        except Exception:
            import ipdb

            ipdb.set_trace()

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
            "user_t": user_t.flatten(),
            "design": design,
        }
        update_dict = {
            "policy_last_t": update_last_t,
            "total_obs": self.all_policies[-1]["total_obs"] + len(new_data),
            "RX": new_RX,
            "XX": new_XX,
            "beta_est": beta_est_df,
            "inc_data": inc_data,
            "seen_user_id": seen_user_id,
        }

        self.all_policies.append(update_dict)

    # TODO: Docstring
    # TODO: probably cleaner to pass in current beta estimate
    # TODO: JIT whole function? or just gradient and hessian batch functions
    def calculate_loss_derivatives(
        self,
        all_prev_data,
        calendar_t,
    ):
        logger.info("Calculating loss gradients and hessians with respect to beta.")
        # Note that it is possible to reconstruct all the data we need by
        # stitching together the incremental data in self.all_policies.
        # However, this is complicated logic, and it seems cleaner to just pass
        # in all study data so far here

        # Get the beta estimate saved in the last algorithm update
        curr_beta_est = self.all_policies[-1]["beta_est"].to_numpy().squeeze()

        # TODO: verify that we don't need hessian for each user, can
        # differentiate sum of est eqns instead.
        # Yeah, Hessian calculation per user takes lots of time, probably avoid.

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

        logger.info("Collecting batched input data as lists.")
        for user_id in self.all_policies[-1]["seen_user_id"]:
            filtered_user_data = all_prev_data.loc[all_prev_data.user_id == user_id]
            batched_base_states_list.append(self.get_base_states(filtered_user_data))
            batched_treat_states_list.append(self.get_treat_states(filtered_user_data))
            batched_actions_list.append(self.get_actions(filtered_user_data))
            batched_rewards_list.append(self.get_rewards(filtered_user_data))

        # TODO: dstack breaks with incremental recruitment
        logger.info("Reforming batched data lists into tensors.")
        batched_base_states_tensor = jnp.dstack(batched_base_states_list)
        batched_treat_states_tensor = jnp.dstack(batched_treat_states_list)
        batched_actions_tensor = jnp.dstack(batched_actions_list)
        batched_rewards_tensor = jnp.dstack(batched_rewards_list)

        logger.info("Forming loss gradients with respect to beta.")
        gradients = get_loss_gradients_batched(
            curr_beta_est,
            batched_base_states_tensor,
            batched_treat_states_tensor,
            batched_actions_tensor,
            batched_rewards_tensor,
        )
        logger.info("Forming loss hessians with respect to beta")
        hessians = get_loss_hessians_batched(
            curr_beta_est,
            batched_base_states_tensor,
            batched_treat_states_tensor,
            batched_actions_tensor,
            batched_rewards_tensor,
        )
        logger.info("Collecting loss gradients into algorithm stats dictionary.")
        self.algorithm_statistics_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {
            user_id: gradients[i]
            for i, user_id in enumerate(self.all_policies[-1]["seen_user_id"])
        }

        logger.info("Forming average hessian and storing in algorithm stats dictionary")
        self.algorithm_statistics_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = jnp.mean(hessians, axis=0)

    # TODO: kinda weird how most recent beta is implicitly used.  Should perhaps
    # calculate correctly for any t captured in data provided. Also a little
    # weird that only the data at calendar_t is needed but we pass in more data.
    # TODO: Probably cleaner to pass in current beta estimate
    def calculate_pi_and_weight_gradients(self, current_data, calendar_t):
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

        curr_beta_est = self.all_policies[-1]["beta_est"].to_numpy().squeeze()

        batched_treat_states_list = []
        batched_actions_list = []

        logger.info("Collecting batched input data as lists.")
        for user_id in self.all_policies[-1]["seen_user_id"]:
            filtered_user_data = current_data.loc[current_data.user_id == user_id]
            batched_treat_states_list.append(
                self.get_treat_states(filtered_user_data)[-1]
            )
            batched_actions_list.append(self.get_actions(filtered_user_data)[-1].item())

        # TODO: stack may break with incremental recruitment
        logger.info("Reforming batched data lists into tensors.")
        batched_treat_states_tensor = jnp.vstack(batched_treat_states_list)
        batched_actions_tensor = jnp.array(batched_actions_list)

        logger.info("Forming pi gradients with respect to beta.")
        pi_gradients = get_pi_gradients_batched(
            curr_beta_est,
            self.args.lower_clip,
            self.args.upper_clip,
            batched_treat_states_tensor,
            batched_actions_tensor,
        )

        logger.info("Forming weight gradients with respect to beta.")
        weight_gradients = get_weight_gradients_batched(
            curr_beta_est,
            curr_beta_est,
            self.args.lower_clip,
            self.args.upper_clip,
            batched_treat_states_tensor,
            batched_actions_tensor,
        )

        logger.info("Collecting pi gradients into algorithm stats dictionary.")
        self.algorithm_statistics_by_calendar_t.setdefault(calendar_t, {})[
            "pi_gradients_by_user_id"
        ] = {
            user_id: pi_gradients[i]
            for i, user_id in enumerate(self.all_policies[-1]["seen_user_id"])
        }

        logger.info("Collecting weight gradients into algorithm stats dictionary.")
        self.algorithm_statistics_by_calendar_t.setdefault(calendar_t, {})[
            "weight_gradients_by_user_id"
        ] = {
            user_id: weight_gradients[i]
            for i, user_id in enumerate(self.all_policies[-1]["seen_user_id"])
        }

    # TODO: Docstring
    # TODO: Use jnp.block to reduce need for indexing
    # TODO: JIT this function?? update times would need to be adjusted (pass them in actually)
    # Make sure jitting makes sense across simulations in terms of common args
    def construct_upper_left_bread_inverse(self):
        logger.info("Constructing upper left bread inverse.")
        # Form the dimensions for our bread matrix portion (pre-inverting). Note that we subtract
        # one from the number of policies  to find the number of updates because there is an
        # initial placeholder policy.
        num_updates = len(self.all_policies) - 1
        beta_dim = len(self.state_feats) + len(self.treat_feats)
        overall_dim = beta_dim * num_updates
        output_matrix = jnp.zeros((overall_dim, overall_dim))

        # List of times that were the first applicable time for some update
        # TODO: sort to not rely on insertion order?
        update_times = [
            t
            for t, value in self.algorithm_statistics_by_calendar_t.items()
            if "loss_gradients_by_user_id" in value
        ]

        num_users = len(self.all_policies[-1]["seen_user_id"])

        # This loop iterates over all times that were the first applicable time
        # for a non-initial policy, which we call update times. Take care
        # to note that update_idx starts at 0.
        # Think of each iteration of this loop as creating a (block) row of the matrix
        for update_idx, update_t in enumerate(update_times):
            logger.info("Processing update time %s.", update_t)
            t_stats_dict = self.algorithm_statistics_by_calendar_t[update_t]

            # This loop creates the non-diagonal terms for the current update
            # Think of each iteration of this loop as creating one term in the current (block) row
            logger.info("Creating the non-diagonal terms for the current update.")
            for i in range(update_idx):
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
                        update_times[i],
                        update_times[i + 1],
                    ):
                        weight_gradient_sum += self.algorithm_statistics_by_calendar_t[
                            t
                        ]["weight_gradients_by_user_id"][user_id]

                    running_entry_holder += jnp.outer(
                        loss_gradient, weight_gradient_sum
                    )

                    # TODO: if we have action-centering, there will be an additional hessian
                    # type term added here. Update: way to implement is to differentiate wrt
                    # all betas each update and always add extra terms here. If 0, so be it, but
                    # things like action centering will just work.
                output_matrix = output_matrix.at[
                    (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                    i * beta_dim : (i + 1) * beta_dim,
                ].set(running_entry_holder / num_users)

            # Add the diagonal hessian entry (which is already an average)
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

        raw_probs = scipy.special.expit(lin_est)
        # TODO: hiding things in args like this makes code harder to follow
        # TODO: can use np clip
        probs = clip(self.args, raw_probs)

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

        if jnp.sum(jnp.abs(self.all_policies[-1]["XX"])) == 0:
            # check if observed any non-trivial data yet
            raw_probs = (
                jnp.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob
            )
            return clip(self.args, raw_probs)

        beta_est_df = self.all_policies[-1]["beta_est"].copy()
        beta_est = beta_est_df.to_numpy()

        treat_states = curr_timestep_data[self.treat_feats].to_numpy()

        prob_input_dict = {
            "treat_states": treat_states,
        }
        probs = self.get_action_probs_inner(beta_est.squeeze(), prob_input_dict)
        return probs

    def get_weights(self, beta_est, collected_data_dict, return_probs=False):
        """
        Get Radon Nikodym weights for all weights for all decisions made by a given policy update

        Inputs:
        - `beta_est`: Algorithm's beta estimators
        - `collected_data_dict`: Dictionary of other information needed to form weights, specifically, data collected using the policy
            This dictionary should include:
                - `action` (vector where each entry is a binary indicator of what action was taken)
                - `action1prob` (vector where each entry has the probability of treatment)
                - `treat_states` (matrix where each row is a state vector that interacts with treatment effect model)
                - `user_id` (vector of user ids that correspond to which users state/action information is used)
                - `all_user_id` (set of all user ids in entire study)
                - `unique_user_id` (set of unique user ids in collected data)
        - `return_probs`: In addition to weights, also return treatment probabilities

        Outputs:
        - Vector of Radon Nikodym weights
        - If `return_probs` is True, also returns a vector of treatment probabilities
        """

        action = collected_data_dict["action"]
        used_prob1 = collected_data_dict["action1prob"]
        used_probA = action * used_prob1 + (1 - action) * (1 - used_prob1)
        treat_states = collected_data_dict["treat_states"]

        prob1_beta = self.get_action_probs_inner(beta_est, collected_data_dict)
        probA_beta = action * prob1_beta + (1 - action) * (1 - prob1_beta)
        weights_subset = probA_beta / used_probA

        # Group by user id
        pi_user_ids = collected_data_dict["user_id"]
        user_ids_grouped, weights_grouped = npi.group_by(pi_user_ids).prod(
            weights_subset
        )

        add_users = set(collected_data_dict["all_user_id"]) - set(pi_user_ids)

        if len(add_users) > 0:
            all_user_ids_grouped = jnp.concatenate(
                [[x for x in add_users], user_ids_grouped]
            )
            ones = jnp.ones((len(add_users)))
            all_weights_grouped = jnp.concatenate([ones, weights_grouped], axis=0)

            sort_idx = jnp.argsort(all_user_ids_grouped)
            all_user_ids_grouped = all_user_ids_grouped[sort_idx]
            all_weights_grouped = all_weights_grouped[sort_idx]
        else:
            all_weights_grouped = weights_grouped

        if return_probs:
            import ipdb

            ipdb.set_trace()
            return (all_weights_grouped, prob1_beta)

        return all_weights_grouped

    def get_est_eqns(
        self,
        beta_est,
        data_dict,
        info_dict=None,
        return_ave_only=False,
        correction="",
        check=False,
        light=False,
    ):
        """
        Get estimating equations for policy estimators for one update

        Inputs:
        - `beta_est`: Algorithm's beta estimators
        - `data_dict`: Dictionary of other information needed to form estimating equations
            This dictionary should include:
                - `action` (vector where each entry is a binary indicator of what action was taken)
                - `reward` (vector where each entry is a real number reward)
                - `avail` (vector where each entry is a binary indicator of whether the user was available)
                - `base_states` (matrix where each row is a state vector that interacts with baseline reward model)
                - `treat_states` (matrix where each row is a state vector that interacts with treatment effect model)
                - `design` (design matrix where each row is the concatenation of base_states and action * treat_states)
                - `user_id` (vector of user ids that correspond to which users state/action information is used)
                - `all_user_id` (unique set of all user ids in entire study)
        - `info_dict`: Dictionary that contains additional information about the algorithm that may be necessary
            For this algorithm, there is no need for this dictionary. It can be None
        - `return_ave_only`: Returns the estimating equations averaged over the users who have been in the study (we have data for)
        - `correction`: Type of small sample correction (default is none, other options are HC3, CR3VE, CR2VE)
        - `check`: Indicator of whether to check the reconstruction of the action selection probabilities
        - `light`: Indicator of whether to just return a dictionary with estimating equations in it. If it is false, it will return a dictionary with additional information (hessian, present user ids, etc.)

        Outputs:
        - Dictionary with numpy matrix of estimating equations (dictionary has more information if light=False)
        """

        actions = data_dict["action"].reshape(-1, 1)
        base_states = data_dict["base_states"]
        treat_states = data_dict["treat_states"]
        outcome_vec = data_dict["reward"]
        avail_vec = data_dict["avail"]
        design = data_dict["design"]
        user_ids = data_dict["user_id"]
        all_user_id = data_dict["all_user_id"]

        try:
            est_eqn_dict = least_squares_helper.get_est_eqn_LS(
                outcome_vec,
                design,
                user_ids,
                beta_est,
                avail_vec,
                all_user_id,
                correction=correction,
                reconstruct_check=check,
                light=light,
            )
        except Exception:
            import ipdb

            ipdb.set_trace()

        if return_ave_only:
            return jnp.sum(est_eqn_dict["est_eqns"], axis=0) / len(all_user_id)
        return est_eqn_dict

    def prep_algdata(self):
        """
        Preprocess / prepare algorithm data statistics to form adaptive sandwich variance estimate

        Inputs: None

        Outputs: A dictionary with the following keys and values
        - `alg_estimators`: concatenated vector with algorithm statistics (betahats); vector is of dimension num_updates*beta_dim
        - `update2esteqn`: dictionary where the keys are update numbers (starts with 1)
            and the values are dictionaries with the data used in that update, which will be used as the data_dict argument when calling the function study_RLalg.get_est_eqns
        - `policy2collected`: dictionary where keys are policy numbers (policy 1 is prespecified policy, policy 2 is policy used after first update; total number of policies is number of updates plus 1; do not need key for first policy)
            value are dictionaries that will be used as the collected_data_dict argument when calling the function study_RLalg.get_weights
        - `info_dict`: Dictionary with certain algorithm info that doesn't change with updates. It will be used as the `info_dict` argument when calling the function `study_RLalg.get_est_eqns`. This dictionary should include:
            * `beta_dim`: dimension of algorithm statistics
            * `all_user_id`: unique set of all user ids in the study
            * `study_RLalg`: RL algorithm object used to collect data
        """
        all_user_id = self.all_policies[-1]["seen_user_id"]

        all_estimators = []
        policy2collected = {}
        update2esteqn = {}
        # `self.all_policies` includes a ``final policy'' that updates after the study concludes
        # and the initial policy
        for update_num, update_dict in enumerate(self.all_policies):
            policy_last_t = update_dict["policy_last_t"]
            if policy_last_t == 0:
                continue

            # Save Parameters from Policies that were used to select actions
            if update_num != len(self.all_policies):
                beta_est = update_dict["beta_est"].to_numpy().squeeze()
                all_estimators.append(beta_est)

                # Cumulative Data for Forming Estimating Functions
                update2esteqn[update_num] = {}
                update2esteqn[update_num]["all_user_id"] = all_user_id
                if update_num == 1:
                    for key in update_dict["inc_data"].keys():
                        update2esteqn[update_num][key] = update_dict["inc_data"][
                            key
                        ].copy()
                else:
                    for key in update_dict["inc_data"].keys():
                        tmp = update2esteqn[update_num - 1][key].copy()
                        update2esteqn[update_num][key] = jnp.concatenate(
                            [update_dict["inc_data"][key].copy(), tmp], axis=0
                        )

            # Collected Data for Forming Weights
            # policy_num refers to the policy number used to collect the data
            policy_num = update_num
            if (
                policy_num > 1
            ):  # we do not form weights for data prior to the first update
                tmp_collected = update_dict["inc_data"].copy()
                tmp_collected["all_user_id"] = all_user_id
                tmp_collected["user_id"] = tmp_collected["user_id"]
                tmp_collected["unique_user_id"] = set(tmp_collected["user_id"])
                policy2collected[policy_num] = tmp_collected

        all_estimators = jnp.hstack(all_estimators)
        beta_dim = len(beta_est)

        info_dict = {
            "beta_dim": beta_dim,
            "all_user_id": all_user_id,
            "study_RLalg": self,
        }

        return {
            "alg_estimators": all_estimators,
            "update2esteqn": update2esteqn,
            "policy2collected": policy2collected,
            "info_dict": info_dict,
        }

    # OLDER versions of functions below (useful for checking)
    """"""
    """"""
    """"""
    """"""
    """"""
    """"""
    """"""
    """"""

    def get_pi_gradients(self, curr_timestep_data, curr_policy_dict, verbose=False):
        # Batched estimators
        beta_est_df = curr_policy_dict["beta_est"].copy()
        beta_est = beta_est_df.to_numpy()
        beta_est_torch = torch.from_numpy(beta_est)
        batch_beta_est = beta_est_torch.repeat((curr_timestep_data.shape[0], 1))
        batch_beta_est.requires_grad = True

        treat_bool = [
            True if x in self.treat_feats_action else False for x in beta_est_df.columns
        ]
        batch_est_treat = batch_beta_est[:, treat_bool]
        treat_states = curr_timestep_data[self.treat_feats]

        pis = sigmoid_LS_torch(
            self.args, batch_est_treat, treat_states, self.allocation_sigma
        )
        actions = curr_timestep_data["action"].to_numpy()
        actions_torch = torch.from_numpy(actions)
        pis_A = actions_torch * pis + (1 - actions_torch) * (1 - pis)
        pis_behavior = torch.from_numpy(torch.clone(pis_A).detach().numpy())
        weights = pis_A / pis_behavior
        weights.sum().backward()
        weighted_pi_grad = batch_beta_est.grad.numpy()

        # Check that reproduced the action selection probabilities correctly
        assert jnp.all(
            jnp.round(pis.detach().numpy(), 5)
            / jnp.round(curr_timestep_data["action1prob"], 5)
            == 1
        )

        return weighted_pi_grad

    def get_est_eqns_full(self, data_sofar, curr_policy_dict, all_user_ids):
        beta_est = curr_policy_dict["beta_est"].to_numpy()

        actions = data_sofar.action.to_numpy().reshape(-1, 1)
        X_vecs = jnp.concatenate(
            [
                data_sofar[self.state_feats].to_numpy(),
                actions * data_sofar[self.treat_feats].to_numpy(),
            ],
            axis=1,
        )

        outcome_vec = data_sofar.reward.to_numpy()
        design = X_vecs
        user_ids = data_sofar.user_id.to_numpy()

        if self.args.dataset_type == "heartsteps":
            avail_vec = data_sofar.availability.to_numpy()
        else:
            avail_vec = jnp.ones(outcome_vec.shape)

        est_eqn_dict = least_squares_helper.get_est_eqn_LS(
            outcome_vec,
            design,
            user_ids,
            beta_est,
            avail_vec,
            all_user_ids,
            correction="",
        )
        est_eqn_dictHC3 = least_squares_helper.get_est_eqn_LS(
            outcome_vec,
            design,
            user_ids,
            beta_est,
            avail_vec,
            all_user_ids,
            correction="HC3",
        )
        est_eqn_dict["est_eqns_HC3"] = est_eqn_dictHC3["est_eqns"]

        # Checks ##########################

        # total number of observations match
        assert curr_policy_dict["total_obs"] == len(data_sofar)

        # estimating equation sums to zero
        ave_est_eqn = jnp.sum(est_eqn_dict["est_eqns"], axis=0)
        try:
            assert jnp.sum(jnp.abs(ave_est_eqn)) < 1
        except Exception:
            import ipdb

            ipdb.set_trace()

        # hessians are symmetric
        hessian = jnp.around(est_eqn_dict["normalized_hessian"], 10)
        assert jnp.all(hessian == hessian.T)

        return est_eqn_dict
