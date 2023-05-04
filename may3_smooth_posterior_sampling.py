import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm
import scipy.stats as stats

from basic_RL_algorithms import clip
from helper_functions import var2suffvec, suffvec2var, get_utri, symmetric_fill_utri 
from least_squares_helper import get_est_eqn_LS
from synthetic_env import make_base_study_df
#get_est_eqn_posterior_sampling

from helper_functions import PRECISION
NUM_POSTERIOR_SAMPLES=10000

def generalized_logistic(args, lin_est):
    pis = args.lower_clip + \
        (args.upper_clip-args.lower_clip) * scipy.special.expit( args.steepness*lin_est/args.allocation_sigma )
    return pis


class SmoothPosteriorSampling:
    """
    Smooth posterior algorithm
    """
    def __init__(self, args, state_feats, treat_feats, alg_seed, allocation_sigma, steepness, prior_mean, prior_var, noise_var, action_centering):
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
        self.prior_mean = prior_mean
        self.prior_var = prior_var
        self.noise_var = noise_var

        var_suffvec = var2suffvec(self, prior_var)
        zero_var_suffvec = np.zeros( var_suffvec.shape )
        initial_beta_est = np.concatenate( [self.prior_mean, zero_var_suffvec] )

        alg_dict = {
            "policy_last_t": 0,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "post_mean": self.prior_mean,
            "post_var": self.prior_var,
            "new_data" : None,
            "total_obs" : 0,
            "all_user_id": set(),
            "XX": np.zeros(self.prior_var.shape),
            "all_user_id": set(),
            "est_params": initial_beta_est,
            "n_users": 1,
            "intercept_val": 0,
            # data used to get updated est_params
        }
        self.all_policies = [alg_dict]
        
        self.treat_feats_action = ['action:'+x for x in self.treat_feats]
        self.treat_bool = np.array( [True if x in self.treat_feats_action else 
                      False for x in self.state_feats+self.treat_feats_action ] )

        # used for forming action selection probabilities
        norm_samples_df = make_base_study_df(args)
        norm_samples = self.rng.normal(loc=0, scale=1,
                        size=(args.T*args.n, NUM_POSTERIOR_SAMPLES))
        tmp_df = pd.DataFrame(norm_samples, 
                    columns=[x for x in range(NUM_POSTERIOR_SAMPLES)])
        self.norm_samples_df = pd.concat([norm_samples_df, tmp_df], 
                                         axis=1)
        
   
    def update_alg(self, new_data, t, all_prev_data=None):

        # update algorithm with new data
        actions = new_data['action'].to_numpy().reshape(-1,1)
        rewards = new_data['reward'].to_numpy().reshape(-1,1)
        action1prob = new_data['action1prob'].to_numpy().reshape(-1,1)
        if self.action_centering:
            X_vecs = np.concatenate( [ new_data[self.state_feats],
                        action1prob * new_data[self.treat_feats],
                        (actions-action1prob) * new_data[self.treat_feats] ], axis=1 )
        else:
            X_vecs = np.concatenate( [ new_data[self.state_feats], 
                                actions * new_data[self.treat_feats] ], axis=1 )

        # Only include available data
        if self.args.dataset_type == 'heartsteps':
            avail_bool = new_data['availability'].astype('bool')
            rewards_avail = rewards[ avail_bool ]
            X_avail = X_vecs[ avail_bool ]
            user_id_avail = new_data['user_id'][avail_bool].to_numpy() 
        else:
            rewards_avail = rewards
            X_avail = X_vecs
            user_id_avail = new_data['user_id'].to_numpy() 
     
        #if self.all_policies[-1]['policy_last_t'] == 0:
        #    design = X_vecs
        #else:
        actions_all = all_prev_data['action'].to_numpy().reshape(-1,1)
        action1prob_all = all_prev_data['action1prob'].to_numpy().reshape(-1,1)
        if self.action_centering:
            design = np.concatenate( [ all_prev_data[self.state_feats],
                        action1prob_all * all_prev_data[self.treat_feats],
                        (actions_all-action1prob_all) * all_prev_data[self.treat_feats] ], axis=1 )
        else:
            design = np.concatenate( [ all_prev_data[self.state_feats], 
                        actions_all * all_prev_data[self.treat_feats] ], axis=1 )
        """
        prev_design = self.all_policies[-1]['design']
        design = np.concatenate([prev_design, X_vecs], axis=0)
        """
        
        import ipdb; ipdb.set_trace()
        # X_vecs == design
         
        alg_dict = {
            "policy_last_t": t-1,
            "new_data" : new_data,  # data used to get updated est_params
            "all_prev_data": all_prev_data,
            "design_new" : X_vecs,
            "design" : design,
            "total_obs" : self.all_policies[-1]['total_obs'] + len(new_data),
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
        }

        # TODO: noise variance

        post_mean = self.all_policies[-1]["post_mean"]
        inv_post_var = np.linalg.inv( self.all_policies[-1]["post_var"] )
        RX = np.matmul(inv_post_var, post_mean)
        new_RX = RX + np.sum(X_avail*rewards_avail, 0)
        add_new_XX = np.einsum( 'ij,ik->jk', X_avail, X_avail )
        new_inv_post_var = inv_post_var + add_new_XX
        new_post_var = np.linalg.inv( new_inv_post_var )
        new_post_mean = np.matmul( new_post_var, new_RX )
        
        col_names = self.state_feats+['action:'+x for x in self.treat_feats]
        all_user_id_set = self.all_policies[-1]["all_user_id"].copy()
        all_user_id_set.update( new_data["user_id"].to_numpy() )
        new_XX = self.all_policies[-1]['XX'] + add_new_XX
        V_matrix = new_XX / len(all_user_id_set)

        # reconstruct posterior variance
        #prior_var_inv = np.linalg.inv( self.all_policies[-1]["prior_var"] )
        #np.linalg.inv( V_matrix * len(all_user_id_set) + prior_var_inv )

        alg_dict["all_user_id"] = all_user_id_set
        alg_dict["n_users"] = len(all_user_id_set)
        alg_dict["post_mean"] = new_post_mean
        alg_dict["post_var"] = new_post_var
        alg_dict["col_names"] = col_names
        alg_dict["V_matrix"] = V_matrix
        alg_dict["XX"] = new_XX
        alg_dict["intercept_val"] = V_matrix[0][0]
        
        V_suffvec = var2suffvec(self, V_matrix)
        #varmatrix = suffvec2var(self, post_V_params)
        #post_V_params = get_utri(post_V)
        
        est_params = np.hstack( [ new_post_mean, V_suffvec ] )
        alg_dict["est_params"] = est_params

        self.all_policies.append(alg_dict)

        #est_params_df = pd.DataFrame(est_params.reshape(1,-1),
        #                         columns=self.state_feats+self.treat_feats_action)
        #alg_dict["est_params"] = est_params_df


    def get_action_probs(self, curr_timestep_data, filter_keyval):
      
        prior_var = self.all_policies[-1]["prior_var"]
        post_var = self.all_policies[-1]["post_var"]
        
        """
        if np.equal( prior_var, post_var ).all():
            # check if observed any non-trivial data yet
            raw_probs = np.ones( curr_timestep_data.shape[0] )*self.args.fixed_action_prob
            return clip(self.args, raw_probs)
        """

        probs = self.get_action_probs_inner(curr_timestep_data, 
                beta_est = self.all_policies[-1]["est_params"], 
                n_users = self.all_policies[-1]["n_users"],
                check_post_var = post_var,
                intercept_val = self.all_policies[-1]['intercept_val'], 
                filter_keyval = filter_keyval)

        return probs


    
    def get_action_probs_inner(self, curr_timestep_data, beta_est, n_users, 
                    intercept_val, filter_keyval, check_post_var=None):
        user_states = curr_timestep_data[self.state_feats]
        treat_states = user_states[self.treat_feats].to_numpy()

        # Get posterior parameters
        if self.action_centering:
            state_dim = len(self.state_feats) + len(self.treat_feats)*2
        else:
            state_dim = len(self.state_feats) + len(self.treat_feats)
        
        post_mean = beta_est[:state_dim]
        V_suff_vector = beta_est[state_dim:]
        
        V_matrix = suffvec2var(self, V_suff_vector, intercept_val)
        #post_V = symmetric_fill_utri(post_V_pieces, post_mean.shape[0])
       
        # construct posterior variance
        prior_var_inv = np.linalg.inv( self.all_policies[-1]["prior_var"] )
        alg_n_users = self.all_policies[-1]["n_users"]
        post_var = np.linalg.inv( V_matrix * alg_n_users + prior_var_inv )

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

        post_mean_user = np.einsum('ij,j->i', treat_states, post_mean_treat)
        post_var_user = np.einsum('ij,ij->i', np.matmul(treat_states, 
                                            post_var_treat), treat_states) 
        post_std_user = np.sqrt(post_var_user)

        ########### Form action selection probabilities (Sampling)
        n_tmp = len(post_mean_user)
        #norm_samples = self.rng.normal(loc=0, scale=1, 
        #                               size=(n_tmp, NUM_POSTERIOR_SAMPLES))
        
        # Get noise that is saved
        selected_norm_idx = self.norm_samples_df[filter_keyval[0]] == filter_keyval[1]
        selected_norm_df = self.norm_samples_df[selected_norm_idx]
        selected_col_names = [x for x in range(NUM_POSTERIOR_SAMPLES)]
        tmp_norm_samples = selected_norm_df[selected_col_names].to_numpy()

        post_samples = tmp_norm_samples * post_std_user.reshape(n_tmp, -1) + \
                post_mean_user.reshape(n_tmp, -1)

        prob_samples = generalized_logistic(self.args, post_samples)
        probs = np.mean(prob_samples, axis=1)
        """
        # Form action selection probabilities (Integration)
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
        """
        if PRECISION is not None:
            probs = np.around(probs, PRECISION)
        return probs


    def get_est_eqns(self, beta_params, data_sofar, all_user_ids, 
                     return_ave_only=False, action1probs=None,
                     correction="", check=False, intercept_val=None):
        """
        Get estimating equations for policy parameters for one update
        """
       
        if self.action_centering:
            state_dim = len(self.state_feats) + len(self.treat_feats)*2
        else:
            state_dim = len(self.state_feats) + len(self.treat_feats)

        prior_dict = {
                "state_dim": state_dim,
                "prior_mean": self.prior_mean,
                "prior_var": self.prior_var,
                "noise_var": self.noise_var,
                }

        # Posterior Mean and V #################################
        actions = data_sofar.action.to_numpy().reshape(-1,1)
        if self.action_centering:
            action1probs = action1probs.reshape(action1probs.shape[0], -1)
            X_vecs = np.concatenate( [ data_sofar[self.state_feats],
                        action1probs * data_sofar[self.treat_feats],
                        (actions-action1probs) * data_sofar[self.treat_feats] ], axis=1 )
        else:
            X_vecs = np.concatenate( [ data_sofar[self.state_feats].to_numpy(),
                        actions * data_sofar[self.treat_feats].to_numpy() ], axis=1 )
        user_ids = data_sofar.user_id.to_numpy()
        outcome_vec = data_sofar.reward.to_numpy()
        design = X_vecs
        if self.args.dataset_type == 'heartsteps':
            avail_vec = data_sofar.availability.to_numpy()
        else:
            avail_vec = np.ones(outcome_vec.shape)

        # assert self.all_policies[1]['design'] == design
        # self.all_policies[2]['all_prev_data'].to_numpy() == data_sofar.to_numpy()[:,:-1]
        # self.all_policies[2]['design'] == design

        tmp = self.all_policies[2]['all_prev_data'].to_numpy()
        design2 = np.concatenate( [ tmp[self.state_feats].to_numpy(),
                        actions * tmp[self.treat_feats].to_numpy() ], axis=1 )

        try:
            est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids,
                                          beta_params, avail_vec, all_user_ids,
                                          prior_dict = prior_dict,
                                          correction = correction,
                                          reconstruct_check = check,
                                          RL_alg = self,
                                          intercept_val = intercept_val)
        except:
            import ipdb; ipdb.set_trace()

        if return_ave_only:
            return np.sum(est_eqn_dict['est_eqns'], axis=0) / len(user_ids)
        return est_eqn_dict



    def get_weights(self, curr_policy_decision_data, beta_params, all_user_ids, filter_keyval, intercept_val):
        """
        Get Radon Nikodym weights for all weights for all decisions made by a given policy update
        """
        action = curr_policy_decision_data['action'].to_numpy()
        
        used_prob1 = curr_policy_decision_data['action1prob'].to_numpy()
        used_probA = action*used_prob1 + (1-action)*(1-used_prob1)

        prob1_beta = self.get_action_probs_inner(curr_policy_decision_data, beta_params, 
                                                 n_users=len(all_user_ids),
                                                 intercept_val=intercept_val,
                                                 filter_keyval=filter_keyval)
        probA_beta = action*prob1_beta + (1-action)*(1-prob1_beta) 

        weights_subset = probA_beta / used_probA

        # cluster by user id 
        pi_user_ids = curr_policy_decision_data['user_id'].to_numpy()
        unique_pi_ids = np.unique(pi_user_ids)
        user_pi_weights = []
        for idx in all_user_ids:
            if idx in pi_user_ids:
                tmp_weight = np.prod( weights_subset[ pi_user_ids == idx ], axis=0 )
            else:
                tmp_weight = 1
            user_pi_weights.append( tmp_weight )
        user_pi_weights = np.array( user_pi_weights ) 

        return user_pi_weights


