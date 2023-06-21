import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm
import scipy.stats as stats
import numpy_indexed as npi

from helper_functions import clip, var2suffvec, suffvec2var, get_utri, symmetric_fill_utri 
from least_squares_helper import get_est_eqn_LS
from synthetic_env import make_base_study_df

NUM_POSTERIOR_SAMPLES=2000

def generalized_logistic(args, lin_est):
    inner = args.steepness*lin_est/args.allocation_sigma
    raw = scipy.special.expit( inner )
    pis = args.lower_clip + (args.upper_clip-args.lower_clip) * raw
    return pis
    

class SmoothPosteriorSampling:
    """
    Smooth posterior algorithm
    """
    def __init__(self, args, state_feats, treat_feats, alg_seed, allocation_sigma, 
                 steepness, prior_mean, prior_var, noise_var, action_centering):
        self.args = args
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.rng = np.random.default_rng(alg_seed)
        self.allocation_sigma = allocation_sigma
        self.steepness = steepness

        self.action_centering = action_centering
        if action_centering:
            total_dim = len(state_feats) + len(treat_feats)*2
        else:
            total_dim = len(state_feats) + len(treat_feats)
        self.state_dim = total_dim
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.inv_prior_var = np.linalg.inv(prior_var)
        self.noise_var = noise_var

        var_suffvec = var2suffvec(self, prior_var)
        zero_var_suffvec = np.zeros( var_suffvec.shape )
        initial_beta_est = np.concatenate( [self.prior_mean, zero_var_suffvec] )

        self.prior_dict = {
                "state_dim": self.state_dim,
                "prior_mean": self.prior_mean,
                "prior_var": self.prior_var,
                "noise_var": self.noise_var,
                }
        
        alg_dict = {
            "policy_last_t": 0,
            "post_mean": self.prior_mean,
            "post_var": self.prior_var,
            "new_data" : None,
            "total_obs" : 0,
            "all_user_id": set(),
            "XX": np.zeros(self.prior_var.shape),
            "RX": np.zeros(self.prior_mean.shape),
            "seen_user_id": set(),
            "total_n_users": self.args.n,
            "beta_est": initial_beta_est,
            "n_users": 1,
            "intercept_val": 0,
            # data used to get updated est_est
        }
        self.all_policies = [alg_dict]
        
        self.treat_feats_action = ['action:'+x for x in self.treat_feats]
        self.treat_bool = np.array( [True if x in self.treat_feats_action else 
                      False for x in self.state_feats+self.treat_feats_action ] )

        # used for forming action selection probabilities
        norm_samples_df = make_base_study_df(args)
        norm_samples = self.rng.normal(loc=0, scale=1,
                        size=(args.T*args.n, NUM_POSTERIOR_SAMPLES))
        #tmp_df = pd.DataFrame(norm_samples, 
        #            columns=[x for x in range(NUM_POSTERIOR_SAMPLES)])
        #self.norm_samples_df = pd.concat([norm_samples_df, tmp_df], 
        #                                 axis=1)
        self.norm_samples_df = norm_samples_df
        self.norm_samples = norm_samples

   
    def update_alg(self, new_data, update_last_t):
        """
        Update algorithm with new data
        
        Inputs:
        - `new_data`: a pandas data frame with new data
        - `update_last_t`: an integer representing the first calendar time 
            the newly updated policy will be used

        Outputs:
        - None
        """
        rewards = new_data['reward'].to_numpy().reshape(-1,1)
        actions = new_data['action'].to_numpy().reshape(-1,1)
        action1prob = new_data['action1prob'].to_numpy().reshape(-1,1)
        base_states, treat_states = self.get_states(new_data)

        if self.action_centering:
            design = np.concatenate( [ base_states, action1prob * treat_states,
                        (actions-action1prob) * treat_states ], axis=1 )
        else:
            design = np.concatenate( [ base_states, actions * treat_states ], axis=1 )
        
        # Only include available data
        calendar_t = new_data['calendar_t'].to_numpy().reshape(-1,1)
        user_t = new_data['user_t'].to_numpy().reshape(-1,1)
        user_id = new_data['user_id'].to_numpy()
        if self.args.dataset_type == 'heartsteps':
            avail_bool = new_data['availability'].astype('bool')
            rewards_avail = rewards[ avail_bool ]
            design_avail = design[ avail_bool ]
            user_id_avail = user_id[avail_bool]
            calendar_t = calendar_t[avail_bool]
            user_t = user_t[avail_bool]
        else:
            avail_bool = np.ones(rewards.shape)
            rewards_avail = rewards
            design_avail = design
            user_id_avail = user_id


        # Algorithm Update
        prev_policy_dict = self.all_policies[-1]
        RX = prev_policy_dict['RX'] + np.sum(rewards_avail * design_avail, axis=0)
        XX_add = np.einsum( 'ij,ik->jk', design_avail, design_avail )
        XX = prev_policy_dict['XX'] + XX_add
        inv_post_var = XX + np.linalg.inv( self.prior_var )
        post_var = np.linalg.inv( inv_post_var )
        prior_adj = np.matmul( np.linalg.inv(self.prior_var), self.prior_mean )
        post_mean = np.matmul( post_var, RX + prior_adj )
        
        seen_user_id = prev_policy_dict["seen_user_id"].copy()
        seen_user_id.update( new_data["user_id"].to_numpy() )
        col_names = self.state_feats+['action:'+x for x in self.treat_feats]

        total_n_users = self.args.n
        V_matrix = XX / total_n_users
        V_suffvec = var2suffvec(self, V_matrix)
        beta_est = np.hstack( [ post_mean, V_suffvec ] )

        # Get noise that is saved
        prev_update_t = prev_policy_dict['policy_last_t']
        selected_norm_idx = ( prev_update_t < self.norm_samples_df['calendar_t']) \
                & (self.norm_samples_df['calendar_t'] <= update_last_t)
                # Select decision times after last update
        tmp_norm_samples = self.norm_samples[selected_norm_idx]

        # Save Data
        inc_data = {
                "reward": rewards.flatten(),
                "action": actions.flatten(),
                "action1prob": action1prob.flatten(),
                "og_action1prob": action1prob.flatten(),
                "base_states": base_states,
                "treat_states": treat_states,
                "avail": avail_bool.flatten(),
                "user_id": user_id_avail,
                "calendar_t": calendar_t.flatten(),
                "user_t": user_t.flatten(),
                "norm_samples": tmp_norm_samples, # noise used to collect inc data
                #"design" : design,
            }
        
        alg_dict = {
            "policy_last_t": update_last_t,
            "total_obs" : self.all_policies[-1]['total_obs'] + len(new_data),
            "RX": RX,
            "XX": XX,
            "Vmatrix": V_matrix,
            "intercept_val": V_matrix[0][0],
            "post_mean": post_mean,
            "post_var": post_var,
            "seen_user_id": seen_user_id,
            "n_users": len(seen_user_id),
            "total_n_users": total_n_users,
            "col_names": col_names,
            "beta_est": beta_est,
            "inc_data": inc_data,
            #"new_data" : new_data,  # data used to get updated beta_est
        }

        self.all_policies.append(alg_dict)


    def get_states(self, tmp_df):
        """
        Form algorithm states from pandas dataframe data

        Input:
        - `tmp_df`: pandas dataframe with state info
        
        Output: Tuple with (base_states, treat_states)
        """
        base_states = tmp_df[self.state_feats].to_numpy()
        treat_states = tmp_df[self.treat_feats].to_numpy()
        return (base_states, treat_states)


    def get_action_probs(self, curr_timestep_data, filter_keyval):
        """
        Form action selection probabilities from newly current data (only use when running RL algorithm)

        Inputs:
        - `curr_timestep_data`: Pandas data frame of current data that can be used to form the states
        - `filter_keyval`: Values to filter on to index into saved random noise that is used to estimate action selection probabilities

        Outputs:
        - Numpy vector of action selection probabilities
        """

        post_var = self.all_policies[-1]["post_var"]
        base_states, treat_states = self.get_states(curr_timestep_data)
        
        # Get noise that is saved
        selected_norm_idx = self.norm_samples_df[filter_keyval[0]] == filter_keyval[1]
        tmp_norm_samples = self.norm_samples[selected_norm_idx]
       
        prob_input_dict = {
                'treat_states': treat_states,
                'n_users': self.all_policies[-1]["n_users"],
                'intercept_val': self.all_policies[-1]['intercept_val'],
                'total_n_users': self.args.n,
                #'filter_keyval': filter_keyval,
                'norm_samples': tmp_norm_samples,
                'state_dim': self.state_dim,
                }

        beta_est = self.all_policies[-1]["beta_est"] 
        probs = self.get_action_probs_inner(beta_est, prob_input_dict,
                                            check_post_var=post_var)

        """
        if filter_keyval[1] == 2:
            prob_input_dict['beta_est'] = beta_est
            prob_input_dict['probs'] = probs
            import pickle as pkl
            with open('temp_debug.pkl', 'wb') as f:
                pkl.dump(prob_input_dict, f)
            print('yo')
            import ipdb; ipdb.set_trace()
        """

        return probs

    
    def get_action_probs_inner(self, beta_est, prob_input_dict, check_post_var=None):
        """
        Form action selection probabilities from raw inputs (used to form importance weights)

        Inputs:
        - `beta_est`: Algorithm's beta estimators
        - `prob_input_dict`: Dictionary of other information needed to form action selection probabilities
            This dictionary should include:
                * `treat_states` (matrix where each row is a state vector that interacts with treatment effect model)
                * `state_dim` (dimension of state vector - includes both base_state and treat_states)
                * `intercept_val` (intercept value in posterior variance - non-random quanitity, depends on number of updates)
                * `n_users` (number of users whose data are used to form the action selection probabilities)
                * `norm_samples` (normal noise samples that will be used to estimate action selection probabilities;
                    should be of dimension num_states by NUM_POSTERIOR_SAMPLES)

        Outputs:
        - Numpy vector of action selection probabilities
        """

        # Get posterior parameters
        state_dim = prob_input_dict['state_dim']
        post_mean = beta_est[:state_dim]
        V_suff_vector = beta_est[state_dim:]

        # Construct posterior variance
        intercept_val = prob_input_dict['intercept_val']
        V_matrix = suffvec2var(self, V_suff_vector, intercept_val)
        alg_n_users = prob_input_dict["total_n_users"] 
        post_var = np.linalg.inv( V_matrix * alg_n_users + self.inv_prior_var )

        # Check reconstruction of posterior variance
        if check_post_var is not None:
            try:
                assert np.equal( np.around(post_var,3), 
                                 np.around(check_post_var,3) ).all()
            except:
                import ipdb; ipdb.set_trace()
        

        # Get posterior parameters for treatment effect (action selection)
        treat_dim = len(self.treat_feats)
        post_mean_treat = post_mean[-treat_dim:]
        post_var_treat = post_var[-treat_dim:,-treat_dim:]

        treat_states = prob_input_dict['treat_states']
        post_mean_user = np.einsum('ij,j->i', treat_states, post_mean_treat)
        post_var_user = np.einsum('ij,ij->i', np.matmul(treat_states, 
                                            post_var_treat), treat_states) 
        post_std_user = np.sqrt(post_var_user)


        #print("last")
        #print(collected_data_dict['norm_samples'][:5])
        
        ########### Form action selection probabilities (Sampling)
        try:
            n_tmp = len(post_mean_user)
            post_samples = post_mean_user.reshape(n_tmp, -1) + \
                    prob_input_dict['norm_samples'] * post_std_user.reshape(n_tmp, -1)
            prob_samples = generalized_logistic(self.args, post_samples)
            probs = np.mean(prob_samples, axis=1)
        except:
            import ipdb; ipdb.set_trace()
        
        """
        ########### Form action selection probabilities (Integration)
        all_action1_probs = []
        n_tmp = len(post_mean_user)
        for i in range(n_tmp):
            action1_prob = stats.norm.expect(
                    func=lambda x: generalized_logistic(
                        self.args, x), 
                    loc = post_mean_user[i], 
                    scale = np.sqrt(post_var_user[i]) )
            all_action1_probs.append(action1_prob)
        probs = np.array(all_action1_probs)
        #if PRECISION is not None:
        #    probs = np.around(probs, PRECISION)
        """
        
        return probs
    

    def get_est_eqns(self, beta_est, data_dict, info_dict, return_ave_only=False, 
                    correction="", check=False, light=False):
        """
        Get estimating equations for policy parameters for one update

        Inputs:
        - `beta_est`: Algorithm's beta estimators
        - `data_dict`: Dictionary of other information needed to form estimating equations
            This dictionary should include:
                * `action` (vector where each entry is a binary indicator of what action was taken)
                * `action1prob` (vector where each entry is an action selection probability - could be a plug-in)
                * `reward` (vector where each entry is a real number reward)
                * `avail` (vector where each entry is a binary indicator of whether the user was available)
                * `base_states` (matrix where each row is a state vector that interacts with baseline reward model)
                * `treat_states` (matrix where each row is a state vector that interacts with treatment effect model)
                * `design` (design matrix where each row is the concatenation of base_states and action * treat_states)
                * `user_id` (vector of user ids that correspond to which users state/action information is used)
                * `all_user_id` (unique set of all user ids in entire study)
                * `intercept_val` (intercept value in posterior variance - non-random quanitity, depends on number of updates)
                * `prior_dict` (dictionary with prior and other algorithm information)
        - `info_dict`: Dictionary that contains additional information about the algorithm that may be necessary
            For this algorithm, there is no need for this dictionary. It can be None
        - `return_ave_only`: Returns the estimating equations averaged over the users who have been in the study (we have data for)
        - `correction`: Type of small sample correction (default is none, other options are HC3, CR3VE, CR2VE)
        - `check`: Indicator of whether to check the reconstruction of the action selection probabilities
        - `light`: Indicator of whether to just return a dictionary with estimating equations in it. If it is false, it will return a dictionary with additional information (hessian, present user ids, etc.)

        Outputs:
        - Dictionary with numpy matrix of estimating equations (dictionary has more information if light=False)
        """

        # Posterior Mean and V (using data dictionary) #####################
        outcome_vec = data_dict['reward']
        actions = data_dict['action'].reshape(-1,1)
        base_states = data_dict['base_states']
        treat_states = data_dict['treat_states']
        avail_vec = data_dict['avail'].flatten()
        user_ids = data_dict['user_id']
        all_user_id = data_dict['all_user_id']
        intercept_val = data_dict['intercept_val']
        prior_dict = info_dict['prior_dict']

        if self.action_centering:
            action1prob = data_dict['action1prob'] 
            action1prob = action1prob.reshape(action1prob.shape[0], -1)
            design = np.concatenate( [ base_states, action1prob * treat_states,
                        (actions-action1prob) * treat_states ], axis=1 )
        else:
            design = np.concatenate( [ base_states, actions * treat_states ], axis=1 )

        est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids,
                                      beta_est, avail_vec, all_user_id,
                                      prior_dict = prior_dict,
                                      correction = correction,
                                      reconstruct_check = check,
                                      RL_alg = self,
                                      intercept_val = intercept_val,
                                      light = light)
      
        if return_ave_only:
            return np.sum(est_eqn_dict['est_eqns'], axis=0) / len(all_user_id)
        return est_eqn_dict


    def get_weights(self, beta_est, collected_data_dict, curr_policy_decision_data=None,
                    return_probs=False):
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
        action = collected_data_dict['action']
        used_prob1 = collected_data_dict['action1prob']
        used_probA = action*used_prob1 + (1-action)*(1-used_prob1)
        treat_states = collected_data_dict['treat_states']

        prob_input_dict = {
                'treat_states': treat_states,
                'n_users': collected_data_dict["n_users"], 
                'intercept_val': collected_data_dict['intercept_val'],
                'norm_samples': collected_data_dict['norm_samples'],
                'total_n_users': self.args.n,
                'state_dim': self.state_dim,
                }

        prob1_beta = self.get_action_probs_inner(beta_est, prob_input_dict)
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
            return (all_weights_grouped, prob1_beta)

        return all_weights_grouped


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
        prev_update_dict = None
        for update_num, update_dict in enumerate(self.all_policies):
            policy_last_t = update_dict['policy_last_t']
            if policy_last_t == 0:
                continue

            # Save Parameters from Policies that were used to select actions
            if update_num != len(self.all_policies):
                beta_est = update_dict['beta_est']
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
                update2esteqn[update_num]['intercept_val'] = update_dict['intercept_val']

            # Collected Data for Forming Weights
            policy_num = update_num  # policy_num refers to the policy number used to collect the data
            if policy_num > 1:  # we do not form weights for data prior to the first update
                tmp_collected = update_dict['inc_data'].copy()
                tmp_collected['all_user_id'] = all_user_id
                tmp_collected['user_id'] = tmp_collected['user_id']
                tmp_collected['unique_user_id'] = set(tmp_collected['user_id'])
                
                # Specific to posterior sampling algorithm
                tmp_collected['intercept_val'] = prev_update_dict['intercept_val']
                tmp_collected['n_users'] = prev_update_dict['n_users']
                policy2collected[policy_num] = tmp_collected
            
            prev_update_dict = update_dict

        all_estimators = np.hstack(all_estimators)
        beta_dim = len(beta_est)

        info_dict = {
                "beta_dim": beta_dim,
                "all_user_id": all_user_id,
                "study_RLalg": self,
                "prior_dict": self.prior_dict,
                }

        return {
                'alg_estimators': all_estimators,
                'update2esteqn': update2esteqn,
                'policy2collected': policy2collected,
                'info_dict': info_dict,
                }

