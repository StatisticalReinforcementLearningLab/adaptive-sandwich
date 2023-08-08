"""
Implementations of several RL algorithms that may be used in study simulations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.special
import torch
import numpy_indexed as npi
import jax
from jax import numpy as jnp

import least_squares_helper
from helper_functions import (
    conditional_x_or_one_minus_x,
    clip,
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
        raw_probs = np.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob
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
        self.steepness = steepness

        self.rng = np.random.default_rng(self.alg_seed)
        total_dim = len(self.state_feats) + len(self.treat_feats)
        # Set an initial policy
        self.all_policies = [
            {
                "policy_last_t": 0,
                "RX": np.zeros(total_dim),
                "XX": np.zeros(total_dim),
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
        self.treat_bool = np.array(
            [
                x in self.treat_feats_action
                for x in self.state_feats + self.treat_feats_action
            ]
        )
        self.upper_left_bread_matrix = None
        self.algorithm_statistics_by_calendar_t = {}

    # TODO: should be able to stack all users. Update IDK about this for differentiation reasons.
    # Update Update: We can differentiate for all users at once with jacrev.
    # TODO: Docstring
    # TODO: Unite with update_alg logic somehow?
    def get_loss(self, beta_est, base_states, treat_states, actions, rewards):
        beta_0 = beta_est[: base_states.shape[1]].reshape(-1, 1)
        beta_1 = beta_est[base_states.shape[1] :].reshape(-1, 1)

        return jnp.sum(
            (
                rewards
                - jnp.matmul(base_states, beta_0)
                - jnp.matmul(actions * treat_states, beta_1)
            )
            ** 2
        )

    def get_loss_gradient(self, *args, **kwargs):
        return jax.grad(self.get_loss, 0)(*args, **kwargs)

    def get_loss_hessian(self, *args, **kwargs):
        return jax.hessian(self.get_loss, 0)(*args, **kwargs)

    # TODO: Unite with method that does same thing. Needed pure function here.
    # Might need to use jacrev to do so
    # TODO: Docstring
    # TODO: Differentiating including the clip is interesting
    # See https://arxiv.org/pdf/2006.06903.pdf
    def get_action_prob_pure(
        self, beta_est, lower_clip, upper_clip, treat_states, action=1
    ):
        treat_est = beta_est[-len(treat_states) :]
        lin_est = jnp.matmul(treat_states, treat_est)

        raw_prob = jax.scipy.special.expit(lin_est)
        prob = conditional_x_or_one_minus_x(
            jnp.clip(raw_prob, lower_clip, upper_clip), action
        )

        return prob[()]

    def get_pi_gradient(self, *args, **kwargs):
        return jax.grad(self.get_action_prob_pure, 0)(*args, **kwargs)

    # TODO: Docstring
    def get_radon_nikodym_weight(
        self, beta, beta_target, lower_clip, upper_clip, treat_states, action
    ):
        common_args = [lower_clip, upper_clip, treat_states]

        pi_beta = self.get_action_prob_pure(beta, *common_args)[()]
        pi_beta_target = self.get_action_prob_pure(beta_target, *common_args)[()]
        return conditional_x_or_one_minus_x(
            pi_beta, action
        ) / conditional_x_or_one_minus_x(pi_beta_target, action)

    def get_weight_gradient(self, *args, **kwargs):
        return jax.grad(self.get_radon_nikodym_weight, 0)(*args, **kwargs)

    # TODO: Docstring
    def get_states(self, tmp_df):
        base_states = tmp_df[self.state_feats].to_numpy()
        treat_states = tmp_df[self.treat_feats].to_numpy()
        return (base_states, treat_states)

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
        design = np.concatenate([base_states, actions * treat_states], axis=1)

        # Only include available data
        calendar_t = new_data["calendar_t"].to_numpy().reshape(-1, 1)
        user_t = new_data["user_t"].to_numpy().reshape(-1, 1)
        rewards_avail = rewards
        avail_bool = np.ones(rewards.shape)
        design_avail = design
        user_id_avail = new_data["user_id"].to_numpy()

        # Get policy estimator
        new_RX = self.all_policies[-1]["RX"] + np.sum(design_avail * rewards_avail, 0)
        new_XX = self.all_policies[-1]["XX"] + np.einsum(
            "ij,ik->jk", design_avail, design_avail
        )
        try:
            inv_XX = np.linalg.inv(new_XX)
        except Exception:
            import ipdb

            ipdb.set_trace()

        beta_est = np.matmul(inv_XX, new_RX.reshape(-1))
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
    def calculate_loss_derivatives(
        self,
        all_prev_data,
        calendar_t,
    ):
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
        # TODO: Will calculate things for T + 1 if the final time is 0 mod
        # decisions between updates.  Doesn't seem quite right
        first_applicable_time = calendar_t + 1
        user_id = 1
        self.get_loss(
            curr_beta_est,
            **self.get_user_states(all_prev_data, user_id),
            actions=self.get_user_actions(all_prev_data, user_id),
            rewards=self.get_user_rewards(all_prev_data, user_id)
        )
        self.algorithm_statistics_by_calendar_t.setdefault(first_applicable_time, {})[
            "loss_gradients_by_user_id"
        ] = {
            user_id: self.get_loss_gradient(
                curr_beta_est,
                **self.get_user_states(all_prev_data, user_id),
                actions=self.get_user_actions(all_prev_data, user_id),
                rewards=self.get_user_rewards(all_prev_data, user_id)
            )
            for user_id in self.all_policies[-1]["seen_user_id"]
        }

        # TODO: Does average work the way I want here? no it does not. I think
        # fixed that but probably get rid of the comprehension
        self.algorithm_statistics_by_calendar_t[first_applicable_time][
            "avg_loss_hessian"
        ] = sum(
            [
                np.array(
                    self.get_loss_hessian(
                        curr_beta_est,
                        **self.get_user_states(all_prev_data, user_id),
                        actions=self.get_user_actions(all_prev_data, user_id),
                        rewards=self.get_user_rewards(all_prev_data, user_id)
                    )
                )
                for user_id in self.all_policies[-1]["seen_user_id"]
            ]
        ) / len(
            self.all_policies[-1]["seen_user_id"]
        )

    def get_user_df(self, study_df, user_id):
        return study_df.loc[study_df.user_id == user_id]

    # TODO: These functions should perhaps be outside this class, or in base
    def get_user_states(self, study_df, user_id):
        """
        Extract just the rewards for the given user in the given study_df as a
        numpy (column) vector.

        Optionally specify a specific calendar time at which to do so.
        """
        user_df = study_df.loc[study_df.user_id == user_id]
        base_states, treat_states = self.get_states(user_df)

        return {"base_states": base_states, "treat_states": treat_states}

    def get_user_actions(self, study_df, user_id):
        """
        Extract just the actions for the given user in the given study_df as a
        numpy (column) vector.
        """
        return (
            study_df.loc[study_df.user_id == user_id]["action"]
            .to_numpy()
            .reshape(-1, 1)
        )

    def get_user_rewards(self, study_df, user_id):
        """
        Extract just the rewards for the given user in the given study_df as a
        numpy (column) vector.
        """
        return (
            study_df.loc[study_df.user_id == user_id]["reward"]
            .to_numpy()
            .reshape(-1, 1)
        )

    # TODO: kinda weird how most recent beta is implicitly used.  Should perhaps
    # calculate correctly for any t captured in data provided. Also a little
    # weird that only the data at calendar_t is needed but we pass in more data.
    def calculate_pi_and_weight_gradients(self, current_data, calendar_t):
        """
        For all users, compute the gradient with respect to beta of both the pi function
        (which takes an action and state and gives the probability of selecting the action)
        and the Radon-Nikodym weight (derived from pi functions as described in the paper)
        for the current time, evaluated at the most recent beta.

        Note that we take the *latest* states and actions from current_data, and assume that
        corresponds to both the most recent beta estimated and the supplied calendar_t.

        The data is saved in `self.algorithm_statistics_by_calendar_t` under the current time.

        Inputs:
        - `current_data`: a pandas data frame with study data. Note that only the data
        at calendar_t is needed.
        - `calendar_t`: the current calendar time

        Outputs:
        - None
        """
        assert calendar_t == np.max(current_data["calendar_t"])

        curr_beta_est = self.all_policies[-1]["beta_est"].to_numpy().squeeze()

        self.algorithm_statistics_by_calendar_t.setdefault(calendar_t, {})[
            "pi_gradients_by_user_id"
        ] = {
            user_id: self.get_pi_gradient(
                curr_beta_est,
                lower_clip=self.args.lower_clip,
                upper_clip=self.args.upper_clip,
                treat_states=self.get_user_states(current_data, user_id)[
                    "treat_states"
                ][-1],
                action=self.get_user_actions(current_data, user_id)[-1].item(),
            )
            for user_id in self.all_policies[-1]["seen_user_id"]
        }

        self.algorithm_statistics_by_calendar_t[calendar_t][
            "weight_gradients_by_user_id"
        ] = {
            user_id: self.get_weight_gradient(
                curr_beta_est,
                beta_target=curr_beta_est,
                lower_clip=self.args.lower_clip,
                upper_clip=self.args.upper_clip,
                treat_states=self.get_user_states(current_data, user_id)[
                    "treat_states"
                ][-1],
                action=self.get_user_actions(current_data, user_id)[-1].item(),
            )
            for user_id in self.all_policies[-1]["seen_user_id"]
        }

    # TODO: Docstring
    def construct_upper_left_bread_matrix(self):
        # Form the dimensions for our bread matrix portion (pre-inverting). Note that we subtract one from the
        # two from the number of policies because there is an initial policy.
        num_updates = len(self.all_policies) - 1
        beta_dim = len(self.state_feats) + len(self.treat_feats)
        total_dim = beta_dim * num_updates
        output_matrix = np.zeros((total_dim, total_dim))

        # List of times that were the first applicable time for some update
        update_times = [
            t
            for t in self.algorithm_statistics_by_calendar_t
            if "loss_gradients_by_user_id" in self.algorithm_statistics_by_calendar_t[t]
        ]

        num_users = len(self.all_policies[-1]["seen_user_id"])

        # This loop iterates over all times that were the first applicable time
        # for a non-initial policy, which we call update times. Take care
        # to note that update_idx starts at 0.
        for update_idx, update_t in enumerate(update_times):
            t_stats_dict = self.algorithm_statistics_by_calendar_t[update_t]

            # This loop creates the non-diagonal terms for the current update
            for i in range(update_idx):
                running_entry_holder = np.zeros((beta_dim, beta_dim))

                # This loop calculates the per-user quantities that will be
                # averaged for the final matrix entries
                for user_id, loss_gradient in t_stats_dict[
                    "loss_gradients_by_user_id"
                ].items():
                    weight_gradient_sum = np.zeros(beta_dim)

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

                    running_entry_holder += np.outer(loss_gradient, weight_gradient_sum)

                    # TODO: if we have action-centering, there will be an additional hessian
                    # type term added here. Update: way to implement is to differentiate wrt
                    # all betas each update and always add extra terms here. If 0, so be it, but
                    # things like action centering will just work.
                output_matrix[
                    (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                    i * beta_dim : (i + 1) * beta_dim,
                ] = (
                    running_entry_holder / num_users
                )

            # Add the diagonal hessian entry (which is already an average)
            output_matrix[
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
                (update_idx) * beta_dim : (update_idx + 1) * beta_dim,
            ] = t_stats_dict["avg_loss_hessian"]

        self.upper_left_bread_matrix = output_matrix

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
        lin_est = np.matmul(prob_input_dict["treat_states"], treat_est)

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

        if np.sum(np.abs(self.all_policies[-1]["XX"])) == 0:
            # check if observed any non-trivial data yet
            raw_probs = (
                np.ones(curr_timestep_data.shape[0]) * self.args.fixed_action_prob
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
            all_user_ids_grouped = np.concatenate(
                [[x for x in add_users], user_ids_grouped]
            )
            ones = np.ones((len(add_users)))
            all_weights_grouped = np.concatenate([ones, weights_grouped], axis=0)

            sort_idx = np.argsort(all_user_ids_grouped)
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
            return np.sum(est_eqn_dict["est_eqns"], axis=0) / len(all_user_id)
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
                        update2esteqn[update_num][key] = np.concatenate(
                            [update_dict["inc_data"][key].copy(), tmp], axis=0
                        )

            # Collected Data for Forming Weights
            policy_num = update_num  # policy_num refers to the policy number used to collect the data
            if (
                policy_num > 1
            ):  # we do not form weights for data prior to the first update
                tmp_collected = update_dict["inc_data"].copy()
                tmp_collected["all_user_id"] = all_user_id
                tmp_collected["user_id"] = tmp_collected["user_id"]
                tmp_collected["unique_user_id"] = set(tmp_collected["user_id"])
                policy2collected[policy_num] = tmp_collected

        all_estimators = np.hstack(all_estimators)
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
        assert np.all(
            np.round(pis.detach().numpy(), 5)
            / np.round(curr_timestep_data["action1prob"], 5)
            == 1
        )

        return weighted_pi_grad

    def get_est_eqns_full(self, data_sofar, curr_policy_dict, all_user_ids):
        beta_est = curr_policy_dict["beta_est"].to_numpy()

        actions = data_sofar.action.to_numpy().reshape(-1, 1)
        X_vecs = np.concatenate(
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
            avail_vec = np.ones(outcome_vec.shape)

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
        ave_est_eqn = np.sum(est_eqn_dict["est_eqns"], axis=0)
        try:
            assert np.sum(np.absolute(ave_est_eqn)) < 1
        except Exception:
            import ipdb

            ipdb.set_trace()

        # hessians are symmetric
        hessian = np.around(est_eqn_dict["normalized_hessian"], 10)
        assert np.all(hessian == hessian.T)

        return est_eqn_dict
