import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm
import torch
import numpy_indexed as npi
import jax.numpy as jnp

from least_squares_helper import get_est_eqn_LS
from helper_functions import clip


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
        raw_probs = np.ones( curr_timestep_data.shape[0] )*self.args.fixed_action_prob
        return clip(self.args, raw_probs)

    def get_pi_gradients(self, user_states):
        raise ValueError("Fixed randomization no policy gradients")

    def get_est_eqn(self, data_sofar):
        raise ValueError("Fixed randomization no need for \
                                     estimating equation of policy")


def torch_clip(args, vals):
    lower_clipped = torch.max( vals,
                torch.ones(vals.shape)*args.lower_clip )
    clipped = torch.min( lower_clipped,
            torch.ones(vals.shape)*args.upper_clip )
    return clipped


def sigmoid_LS_torch(args, batch_est_treat, treat_states, allocation_sigma):
    # States
    treat_states_torch = torch.from_numpy( treat_states.to_numpy() )

    # Form Probabilities
    lin_est = torch.sum(batch_est_treat * treat_states_torch, axis=1)
    pis = torch_clip( args, torch.sigmoid( args.steepness*lin_est/allocation_sigma ) )

    # below genralized logistic (different asymptotes)
    #pis = torch_clip( args, args.lower_clip +
    #                 (args.upper_clip-args.lower_clip) * torch.sigmoid( args.steepness*lin_est/allocation_sigma ) )

    return pis


class SigmoidLS:
    """
    Sigmoid Least Squares algorithm
    """
    def __init__(self, args, state_feats, treat_feats, alg_seed, allocation_sigma, steepness):
        self.args = args
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.rng = np.random.default_rng(alg_seed)
        self.allocation_sigma = allocation_sigma
        self.steepness = steepness

        total_dim = len(state_feats) + len(treat_feats)
        alg_dict = {
            "policy_last_t": 0,
            "RX" : np.zeros(total_dim),
            "XX" : np.zeros(total_dim),
            "beta_est" : None,
            "new_data" : None,
            "total_obs" : 0,
            "seen_user_id": set(),
            # data used to get updated beta_est
        }
        self.all_policies = [alg_dict]

        self.treat_feats_action = ['action:'+x for x in self.treat_feats]
        self.treat_bool = np.array( [True if x in self.treat_feats_action else
                      False for x in self.state_feats+self.treat_feats_action ] )

        # Things added exclusively for Nowell's package
        self.pi_args = {}
        self.rl_update_args = {}

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
        actions = new_data['action'].to_numpy().reshape(-1,1)
        action1probs = new_data['action1prob'].to_numpy().reshape(-1,1)
        rewards = new_data['reward'].to_numpy().reshape(-1,1)
        base_states, treat_states = self.get_states(new_data)
        design = np.concatenate( [ base_states,
                            actions * treat_states ], axis=1 )

        # Only include available data
        calendar_t = new_data['calendar_t'].to_numpy().reshape(-1,1)
        user_t = new_data['user_t'].to_numpy().reshape(-1,1)
        if self.args.dataset_type == 'heartsteps':
            avail_bool = new_data['availability'].astype('bool')
            rewards_avail = rewards[ avail_bool ]
            design_avail = design[ avail_bool ]
            user_id_avail = new_data['user_id'][avail_bool].to_numpy()
            calendar_t = calendar_t[avail_bool]
            user_t = user_t[avail_bool]
        else:
            rewards_avail = rewards
            avail_bool = np.ones(rewards.shape)
            design_avail = design
            user_id_avail = new_data['user_id'].to_numpy()

        # Get policy estimator
        new_RX = self.all_policies[-1]['RX'] + \
                            np.sum(design_avail*rewards_avail, 0)
        new_XX = self.all_policies[-1]['XX'] + \
                            np.einsum( 'ij,ik->jk', design_avail, design_avail )
        try:
            inv_XX = np.linalg.inv( new_XX )
        except:
            import ipdb; ipdb.set_trace()
        beta_est = np.matmul(inv_XX, new_RX.reshape(-1))

        col_names = self.state_feats+['action:'+x for x in self.treat_feats]
        beta_est_df = pd.DataFrame(beta_est.reshape(1,-1),
                        columns=self.state_feats+self.treat_feats_action)

        seen_user_id = self.all_policies[-1]['seen_user_id'].copy()
        seen_user_id.update(new_data['user_id'].to_numpy())

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
            "design" : design,
        }
        update_dict = {
            "policy_last_t": update_last_t,
            "total_obs" : self.all_policies[-1]['total_obs'] + len(new_data),
            "RX": new_RX,
            "XX": new_XX,
            "beta_est": beta_est_df,
            "inc_data": inc_data,
            "seen_user_id": seen_user_id,
            #"new_data" : new_data,  # data used to get updated beta_est
        }

        self.all_policies.append(update_dict)

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
        lin_est = np.matmul(prob_input_dict['treat_states'], treat_est.T)

        raw_probs = scipy.special.expit(self.steepness * lin_est)
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

        if np.sum( np.abs( self.all_policies[-1]["XX"] ) ) == 0:
            # check if observed any non-trivial data yet
            raw_probs = np.ones( curr_timestep_data.shape[0] )*self.args.fixed_action_prob
            return clip(self.args, raw_probs)

        beta_est_df = self.all_policies[-1]['beta_est'].copy()
        beta_est = beta_est_df.to_numpy()

        treat_states = curr_timestep_data[self.treat_feats].to_numpy()

        prob_input_dict = {
                'treat_states': treat_states,
                }
        probs = self.get_action_probs_inner(beta_est.squeeze(),
                                            prob_input_dict)
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
        - Vector of Randon Nikodym weights
        - If `return_probs` is True, also returns a vector of treatment probabilities
        """

        action = collected_data_dict['action']
        used_prob1 = collected_data_dict['action1prob']
        used_probA = action*used_prob1 + (1-action)*(1-used_prob1)
        treat_states = collected_data_dict['treat_states']

        prob1_beta = self.get_action_probs_inner(beta_est, collected_data_dict)
        probA_beta = action*prob1_beta + (1-action)*(1-prob1_beta)
        weights_subset = probA_beta / used_probA

        # Group by user id
        pi_user_ids = collected_data_dict['user_id']
        user_ids_grouped, weights_grouped = npi.group_by(pi_user_ids).prod(weights_subset)

        add_users = set(collected_data_dict['all_user_id']) - set(pi_user_ids)

        if len(add_users) > 0:
            all_user_ids_grouped = np.concatenate( [[x for x in add_users], user_ids_grouped] )
            ones = np.ones((len(add_users)))
            all_weights_grouped = np.concatenate( [ones, weights_grouped], axis=0 )

            sort_idx = np.argsort(all_user_ids_grouped)
            all_user_ids_grouped = all_user_ids_grouped[sort_idx]
            all_weights_grouped = all_weights_grouped[sort_idx]
        else:
            all_weights_grouped = weights_grouped

        if return_probs:
            import ipdb; ipdb.set_trace()
            return (all_weights_grouped, prob1_beta)

        return all_weights_grouped

    def get_est_eqns(self, beta_est, data_dict, info_dict=None,
                     return_ave_only=False, correction="",
                     check=False, light=False):
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

        actions = data_dict['action'].reshape(-1,1)
        base_states = data_dict['base_states']
        treat_states = data_dict['treat_states']
        outcome_vec = data_dict['reward']
        avail_vec = data_dict['avail']
        design = data_dict['design']
        user_ids = data_dict['user_id']
        all_user_id = data_dict['all_user_id']

        est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids,
                                        beta_est, avail_vec, all_user_id,
                                        correction = correction,
                                        reconstruct_check = check,
                                        light=light)

        if return_ave_only:
            return np.sum(est_eqn_dict['est_eqns'], axis=0) / len(all_user_id)
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
        all_user_id = self.all_policies[-1]['seen_user_id']

        all_estimators = []
        policy2collected = {}
        update2esteqn = {}
        # `self.all_policies` includes a ``final policy'' that updates after the study concludes
        # and the initial policy
        for update_num, update_dict in enumerate(self.all_policies):
            policy_last_t = update_dict['policy_last_t']
            if policy_last_t == 0:
                continue

            # Save Parameters from Policies that were used to select actions
            if update_num != len(self.all_policies):
                beta_est = update_dict['beta_est'].to_numpy().squeeze()
                all_estimators.append(beta_est)

                # Cumulative Data for Forming Estimating Functions
                update2esteqn[update_num] = {}
                update2esteqn[update_num]['all_user_id'] = all_user_id
                if update_num == 1:
                    for key in update_dict['inc_data'].keys():
                        update2esteqn[update_num][key] = update_dict['inc_data'][key].copy()
                else:
                    for key in update_dict['inc_data'].keys():
                        tmp = update2esteqn[update_num-1][key].copy()
                        update2esteqn[update_num][key] = \
                                np.concatenate( [ update_dict['inc_data'][key].copy(), tmp ] , axis=0)

            # Collected Data for Forming Weights
            policy_num = update_num  # policy_num refers to the policy number used to collect the data
            if policy_num > 1:  # we do not form weights for data prior to the first update
                tmp_collected = update_dict['inc_data'].copy()
                tmp_collected['all_user_id'] = all_user_id
                tmp_collected['user_id'] = tmp_collected['user_id']
                tmp_collected['unique_user_id'] = set(tmp_collected['user_id'])
                policy2collected[policy_num] = tmp_collected

        all_estimators = np.hstack(all_estimators)
        beta_dim = len(beta_est)

        info_dict = {
                "beta_dim": beta_dim,
                "all_user_id": all_user_id,
                "study_RLalg": self,
                }

        return {
                'alg_estimators': all_estimators,
                'update2esteqn': update2esteqn,
                'policy2collected': policy2collected,
                'info_dict': info_dict,
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
        beta_est_df = curr_policy_dict['beta_est'].copy()
        beta_est = beta_est_df.to_numpy()
        beta_est_torch = torch.from_numpy( beta_est )
        batch_beta_est = beta_est_torch.repeat(
                            (curr_timestep_data.shape[0], 1) )
        batch_beta_est.requires_grad = True

        treat_bool = [True if x in self.treat_feats_action else
                      False for x in beta_est_df.columns ]
        batch_est_treat = batch_beta_est[:,treat_bool]
        treat_states = curr_timestep_data[self.treat_feats]

        pis = sigmoid_LS_torch(self.args, batch_est_treat,
                               treat_states, self.allocation_sigma)
        actions = curr_timestep_data['action'].to_numpy()
        actions_torch = torch.from_numpy( actions )
        pis_A = actions_torch*pis + (1-actions_torch)*(1-pis)
        pis_behavior = torch.from_numpy( torch.clone(pis_A).detach().numpy() )
        weights = pis_A / pis_behavior
        weights.sum().backward()
        weighted_pi_grad = batch_beta_est.grad.numpy()

        # Check that reproduced the action selection probabilities correctly
        assert np.all( np.round( pis.detach().numpy(), 5) /
                          np.round(curr_timestep_data['action1prob'], 5) == 1)

        return weighted_pi_grad

    def get_est_eqns_full(self, data_sofar, curr_policy_dict, all_user_ids):
        beta_est = curr_policy_dict['beta_est'].to_numpy()

        actions = data_sofar.action.to_numpy().reshape(-1,1)
        X_vecs = np.concatenate( [ data_sofar[self.state_feats].to_numpy(),
                    actions * data_sofar[self.treat_feats].to_numpy() ], axis=1 )

        outcome_vec = data_sofar.reward.to_numpy()
        design = X_vecs
        user_ids = data_sofar.user_id.to_numpy()

        if self.args.dataset_type == 'heartsteps':
            avail_vec = data_sofar.availability.to_numpy()
        else:
            avail_vec = np.ones(outcome_vec.shape)

        est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids,
                                      beta_est, avail_vec, all_user_ids,
                                      correction="", debug=True)
        est_eqn_dictHC3 = get_est_eqn_LS(outcome_vec, design, user_ids,
                                      beta_est, avail_vec, all_user_ids,
                                      correction="HC3")
        est_eqn_dict["est_eqns_HC3"] = est_eqn_dictHC3["est_eqns"]

        # Checks ##########################

        # total number of observations match
        assert curr_policy_dict['total_obs'] == len(data_sofar)

        # estimating equation sums to zero
        ave_est_eqn = np.sum(est_eqn_dict["est_eqns"], axis=0)
        try:
            assert np.sum( np.absolute( ave_est_eqn ) ) < 1
        except:
            import ipdb; ipdb.set_trace()

        # hessians are symmetric
        hessian = np.around(est_eqn_dict['normalized_hessian'], 10)
        assert np.all( hessian == hessian.T )

        return est_eqn_dict

    ### The following functions are added to be able to use Nowell's package to
    ### analyze the same data as a run with Kelly's code.
    def collect_pi_args(self, all_prev_data, calendar_t, curr_beta_est):
        assert calendar_t == np.max(all_prev_data["calendar_t"].to_numpy())

        # Don't have to worry about out-of-study users here, no incremental recruitment
        self.pi_args[calendar_t] = {
            user_id: (
                (
                    curr_beta_est,
                    self.args.lower_clip,
                    self.steepness,
                    self.args.upper_clip,
                    self.get_treat_states(
                        all_prev_data.loc[all_prev_data.user_id == user_id]
                    )[-1],
                )
            )
            for user_id in self.get_all_users(all_prev_data)
        }

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
        action1probs = df[calendar_t_col].to_numpy(dtype="float32").reshape(-1, 1)
        return jnp.array(action1probs)

    def collect_rl_update_args(self, all_prev_data, calendar_t, curr_beta_est):
        """
        NOTE: Must be called AFTER the update it concerns happens, so that the
        beta the rest of the data already produced is used.
        """
        next_policy_num = int(all_prev_data["policy_num"].max() + 1)
        self.rl_update_args[next_policy_num] = {}
        action_centering = 0

        # We should always have data for all users, because there is no incremental
        # recruitment here.
        for user_id in self.get_all_users(all_prev_data):
            user_data = all_prev_data.loc[all_prev_data.user_id == user_id]
            self.rl_update_args[next_policy_num][user_id] = (
                curr_beta_est,
                self.get_base_states(user_data),
                self.get_treat_states(user_data),
                self.get_actions(user_data),
                self.get_rewards(user_data),
                self.get_action1probs(user_data),
                self.get_action1probstimes(user_data),
                action_centering,
            )

    def get_all_users(self, study_df, user_id_column="user_id"):
        return study_df[user_id_column].unique()

    def get_current_beta_estimate(self):
        raw_beta = self.all_policies[-1]["beta_est"]
        if raw_beta is not None:
            return self.all_policies[-1]["beta_est"].to_numpy().squeeze()
        return  np.zeros(len(self.state_feats) + len(self.treat_feats))