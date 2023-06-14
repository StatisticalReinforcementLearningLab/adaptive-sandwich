import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm
import torch

from least_squares_helper import get_est_eqn_LS
#from est_eqn_helper import get_est_eqn_LS_tmp
from helper_functions import PRECISION, clip


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
            "est_params" : None,
            "new_data" : None, 
            "total_obs" : 0,
            # data used to get updated est_params
        }
        self.all_policies = [alg_dict]
        
        self.treat_feats_action = ['action:'+x for x in self.treat_feats]
        self.treat_bool = np.array( [True if x in self.treat_feats_action else 
                      False for x in self.state_feats+self.treat_feats_action ] )
        
    def get_states(self, tmp_df):
        base_states = tmp_df[self.state_feats].to_numpy()
        treat_states = tmp_df[self.treat_feats].to_numpy()
        return (base_states, treat_states)

    def update_alg(self, new_data, t):
        # update algorithm with new data
        actions = new_data['action'].to_numpy().reshape(-1,1)
        action1probs = new_data['action1prob'].to_numpy().reshape(-1,1)
        rewards = new_data['reward'].to_numpy().reshape(-1,1)
        base_states, treat_states = self.get_states(new_data)

        X_vecs = np.concatenate( [ base_states, 
                            actions * treat_states ], axis=1 )

        # Only include available data
        if self.args.dataset_type == 'heartsteps':
            avail_bool = new_data['availability'].astype('bool')
            rewards_avail = rewards[ avail_bool ]
            X_avail = X_vecs[ avail_bool ]
            user_id_avail = new_data['user_id'][avail_bool].to_numpy() 
        else:
            rewards_avail = rewards
            avail_bool = np.ones(rewards.shape)
            X_avail = X_vecs
            user_id_avail = new_data['user_id'].to_numpy() 
        
        alg_dict = {
            "policy_last_t": t-1,
            "new_data" : new_data,  # data used to get updated est_params
            "design" : X_vecs,
            "total_obs" : self.all_policies[-1]['total_obs'] + len(new_data)
        }
        
        new_RX = self.all_policies[-1]['RX'] + np.sum(X_avail*rewards_avail, 0)
        new_XX = self.all_policies[-1]['XX'] + \
                                np.einsum( 'ij,ik->jk', X_avail, X_avail )

        col_names = self.state_feats+['action:'+x for x in self.treat_feats]
            
        # Get policy parameters
        try:
            inv_XX = np.linalg.inv( new_XX )
        except:
            import ipdb; ipdb.set_trace()
        est_params = np.matmul(inv_XX, new_RX.reshape(-1))
        est_params_df = pd.DataFrame(est_params.reshape(1,-1),
                                 columns=self.state_feats+self.treat_feats_action)
            
        alg_dict["RX"] = new_RX
        alg_dict["XX"] = new_XX
        alg_dict["est_params"] = est_params_df

        inc_data_dict = {
                "rewards": rewards.flatten(),
                "actions": actions.flatten(),
                "action1probs": action1probs.flatten(),
                "base_states": base_states,
                "treat_states": treat_states,
                "avail": avail_bool.flatten(),
                "user_ids": user_id_avail,
                }
        alg_dict['inc_data_dict'] = inc_data_dict
        #print(t, "rewards", rewards.shape)
        #print(t, "states", base_states.shape)

        self.all_policies.append(alg_dict)


    def get_action_probs_inner(self, treat_states, est_params, 
                               n_users=None, intercept_val=None, filter_keyval=None):

        treat_params = est_params[self.treat_bool]
        
        lin_est = np.matmul(treat_states, treat_params.T)
        raw_probs = scipy.special.expit(lin_est)
        probs = clip(self.args, raw_probs)
        #if PRECISION is not None:
        #    probs = np.around(probs, PRECISION)
        return probs.squeeze()
        

    def get_action_probs(self, curr_timestep_data, filter_keyval):
        
        if np.sum( np.abs( self.all_policies[-1]["XX"] ) ) == 0:
            # check if observed any non-trivial data yet
            raw_probs = np.ones( curr_timestep_data.shape[0] )*self.args.fixed_action_prob
            return clip(self.args, raw_probs)
       
        est_param_df = self.all_policies[-1]['est_params'].copy() 
        est_params = est_param_df.to_numpy()
        
        #probs0 = self.get_action_probs_inner_old(curr_timestep_data, est_params.squeeze())
        
        treat_states = curr_timestep_data[self.treat_feats].to_numpy()
        #treat_states = user_states[self.treat_feats].to_numpy()
        #treat_params = est_params.squeeze()[self.treat_bool]

        probs = self.get_action_probs_inner(treat_states, est_params.squeeze())
        return probs


    def get_weights(self, data_dict, beta_params, all_user_ids, 
                    collected_data_dict, 
                    curr_policy_decision_data=None, intercept_val=None, 
                    filter_keyval=None):
        """
        Get Radon Nikodym weights for all weights for all decisions made by a given policy update
        """
        
        """
        action = curr_policy_decision_data['action'].to_numpy()
        
        used_prob1 = curr_policy_decision_data['action1prob'].to_numpy()
        used_probA2 = action*used_prob1 + (1-action)*(1-used_prob1)

        prob1_beta = self.get_action_probs_inner_old(curr_policy_decision_data, beta_params, n_users=None)
        probA_beta2 = action*prob1_beta + (1-action)*(1-prob1_beta) 

        weights_subset2 = probA_beta2 / used_probA2
        """
       
        # TODO new stuff
        
        action = collected_data_dict['actions']
        used_prob1 = collected_data_dict['action1probs']
        used_probA = action*used_prob1 + (1-action)*(1-used_prob1)
        treat_states = collected_data_dict['treat_states']
        
        prob1_beta = self.get_action_probs_inner(treat_states, beta_params, 
                                                 n_users=None)
        probA_beta = action*prob1_beta + (1-action)*(1-prob1_beta) 
        weights_subset = probA_beta / used_probA
        
        #import ipdb; ipdb.set_trace()

        # cluster by user id 
        #pi_user_ids = curr_policy_decision_data['user_id'].to_numpy()
        pi_user_ids = collected_data_dict['user_ids']
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

    
    def get_est_eqns(self, beta_params, data_dict, all_user_ids, data_sofar=None, 
                     return_ave_only=False, action1probs=None, correction="", 
                     check=False, intercept_val=None):
        """
        Get estimating equations for policy parameters for one update
        """
        
        actions = data_dict['actions'].reshape(-1,1)
        base_states = data_dict['base_states']
        treat_states = data_dict['treat_states']
        design = np.concatenate( [ base_states, actions * treat_states ], axis=1 )
        outcome_vec = data_dict['rewards']
        user_ids = data_dict['user_ids']
        avail_vec = data_dict['avail']
        
        est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids, 
                                      beta_params, avail_vec, all_user_ids,
                                      correction = correction,
                                      reconstruct_check = check)

        """
        # Use dataframe ##################################
        actions = data_sofar.action.to_numpy().reshape(-1,1)
        base_states, treat_states = self.get_states(data_sofar)
        design = np.concatenate( [ base_states, actions * treat_states ], axis=1 )
        outcome_vec = data_sofar.reward.to_numpy()
        
        user_ids = data_sofar.user_id.to_numpy()
        if self.args.dataset_type == 'heartsteps':
            avail_vec = data_sofar.availability.to_numpy()
        else:
            avail_vec = np.ones(outcome_vec.shape)
            
        est_eqn_dict1 = get_est_eqn_LS(outcome_vec, design, user_ids, 
                                      beta_params, avail_vec, all_user_ids,
                                      correction = correction,
                                      reconstruct_check = check)
        """
        
        # Use dictionary ##################################
        try:
            assert actions.shape[0] == data_dict['actions'].shape[0]
            assert outcome_vec.shape[0] == data_dict['rewards'].shape[0]
            assert base_states.shape[0] == data_dict['base_states'].shape[0]
            assert treat_states.shape[0] == data_dict['treat_states'].shape[0]
            assert user_ids.shape[0] == data_dict['user_ids'].shape[0]
        except:
            import ipdb; ipdb.set_trace()

        #assert ( est_eqn_dict['est_eqns'] == est_eqn_dict1['est_eqns'] ).all()

        if return_ave_only:
            return np.sum(est_eqn_dict['est_eqns'], axis=0) / len(user_ids)
        return est_eqn_dict

    




    def get_action_probs_inner_old(self, curr_timestep_data, beta_est, n_users=None, intercept_val=None, filter_keyval=None):
        user_states = curr_timestep_data[self.state_feats]
        
        treat_states = user_states[self.treat_feats].to_numpy()
        treat_params = beta_est[self.treat_bool]

        lin_est = np.matmul(treat_states, treat_params.T)
        raw_probs = scipy.special.expit(lin_est)
        probs = clip(self.args, raw_probs)
        #if PRECISION is not None:
        #    probs = np.around(probs, PRECISION)
        return probs.squeeze()
        

    def get_est_eqns_df(self, beta_params, data_sofar, all_user_ids, return_ave_only=False, action1probs=None, correction="", check=False, intercept_val=None):
        """
        Get estimating equations for policy parameters for one update
        """
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
            
        #est_eqn_dict = get_est_eqn_LS_tmp(outcome_vec, design, user_ids, 
        est_eqn_dict = get_est_eqn_LS(outcome_vec, design, user_ids, 
                                      beta_params, avail_vec, all_user_ids,
                                      correction = correction,
                                      reconstruct_check = check)

        if return_ave_only:
            return np.sum(est_eqn_dict['est_eqns'], axis=0) / len(user_ids)
        return est_eqn_dict

    
    def get_weights_old(self, curr_policy_decision_data, beta_params, all_user_ids, intercept_val=None, filter_keyval=None):
        """
        Get Radon Nikodym weights for all weights for all decisions made by a given policy update
        """

        action = curr_policy_decision_data['action'].to_numpy()
        
        used_prob1 = curr_policy_decision_data['action1prob'].to_numpy()
        used_probA = action*used_prob1 + (1-action)*(1-used_prob1)

        prob1_beta = self.get_action_probs_inner_old(curr_policy_decision_data, beta_params, n_users=None)
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


    """"""
    """"""
    """"""
    """"""
    """"""
    """"""
    """"""
    """"""
    
    def get_pi_gradients(self, curr_timestep_data, curr_policy_dict, verbose=False):
    
        # Batched parameters
        est_param_df = curr_policy_dict['est_params'].copy()
        est_params = est_param_df.to_numpy()
        est_params_torch = torch.from_numpy( est_params )
        batch_est_params = est_params_torch.repeat( 
                            (curr_timestep_data.shape[0], 1) )
        batch_est_params.requires_grad = True

        treat_bool = [True if x in self.treat_feats_action else 
                      False for x in est_param_df.columns ]
        batch_est_treat = batch_est_params[:,treat_bool]
        treat_states = curr_timestep_data[self.treat_feats]

        pis = sigmoid_LS_torch(self.args, batch_est_treat, treat_states, 
                self.allocation_sigma)
        actions = curr_timestep_data['action'].to_numpy()
        actions_torch = torch.from_numpy( actions ) 
        pis_A = actions_torch*pis + (1-actions_torch)*(1-pis)
        pis_behavior = torch.from_numpy( torch.clone(pis_A).detach().numpy() )
        weights = pis_A / pis_behavior
        weights.sum().backward()
        weighted_pi_grad = batch_est_params.grad.numpy()
        
        # Check that reproduced the action selection probabilities correctly
        assert np.all( np.round( pis.detach().numpy(), 5) / 
                          np.round(curr_timestep_data['action1prob'], 5) == 1)
       
        return weighted_pi_grad
   

    def get_est_eqns_full(self, data_sofar, curr_policy_dict, all_user_ids):
        est_param = curr_policy_dict['est_params'].to_numpy()

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
                                      est_param, avail_vec, all_user_ids,
                                      correction="")
        est_eqn_dictHC3 = get_est_eqn_LS(outcome_vec, design, user_ids, 
                                      est_param, avail_vec, all_user_ids,
                                      correction="HC3")
        est_eqn_dict["est_eqns_HC3"] = est_eqn_dictHC3["est_eqns"]

        # Checks ##########################
        
        # total number of observations match
        assert curr_policy_dict['total_obs'] == len(data_sofar)
      
        # estimating equation sums to zero
        ave_est_eqn = np.sum(est_eqn_dict["est_eqns"], axis=0)
        assert np.sum( np.absolute( ave_est_eqn ) ) < 1
        
        # hessians are symmetric
        hessian = np.around(est_eqn_dict['normalized_hessian'], 10)
        assert np.all( hessian == hessian.T )

        return est_eqn_dict


