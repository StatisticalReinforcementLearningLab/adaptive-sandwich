import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm
import scipy.stats as stats
import torch

from basic_RL_algorithms import torch_clip
from least_squares_helper import get_est_eqn_posterior_sampling

"""
TODO
- Code generalized logistic function***
- Put in prior and fix algorithm update
- Sampling for action selection probabilities
- Figure out mathematically if I need the posterior variance term***

LATER
- Action centering
"""


def generalized_logistic(args, lin_est, allocation_sigma=1):
    pis = args.lower_clip + \
        (args.upper_clip-args.lower_clip) * scipy.special.expit( args.steepness*lin_est/allocation_sigma )
    return pis



def generalized_logistic_torch(args, lin_est, allocation_sigma=1):
    pis = torch_clip( args, args.lower_clip +
                     (args.upper_clip-args.lower_clip) * torch.sigmoid( args.steepness*lin_est/allocation_sigma ) )
    return pis



"""
def smooth_posterior_sampling(args, treat_states, post_mean_treat_torch, 
                              post_var_treat_torch, MC_N, allocation_sigma=1, rng=np.random):
    - `treat_states`: numpy array with dimension (n_users_at_decision_time x treat_dim)
    - `post_mean_treat_torch`: torch tensor with dimension (n_users_at_decision_time x treat_dim)
    - `post_var_treat_torch`: torch tensor with dimension (n_users_at_decision_time x treat_dim x treat_dim)
    
    # Estimate action selection probabilities
    n_tmp, treat_dim = treat_states.shape
    
    normal_samples = rng.normal( size=(MC_N, n_tmp, treat_dim) )
    normal_samples_torch = torch.from_numpy( normal_samples )
    
    var_times_samples = torch.einsum( 'ijk,lik->lij', post_var_treat_torch, normal_samples_torch )
    post_vec_sample_torch = post_mean_treat_torch + var_times_samples
    
    treat_states_torch = torch.from_numpy(treat_states)
    lin_est_torch = torch.sum(treat_states_torch * post_vec_sample_torch, axis=2)

    sampled_pis_torch = generalized_logistic(args, lin_est_torch, allocation_sigma)
    action1probs = torch.mean( sampled_pis_torch, axis=0 )

    std_error = torch.std(sampled_pis_torch, axis=0) / np.sqrt(MC_N)
    #max_std_error = float(torch.max(std_error))
  
    ##############################

    treat_states_torch = torch.from_numpy(treat_states)
    post_state_mean = torch.sum( treat_states_torch * post_mean_treat_torch, axis=1)
    post_state_var = torch.sum ( treat_states_torch * 
                                torch.bmm( post_var_treat_torch, treat_states_torch.unsqueeze(2) ).squeeze(), 1 )

    posterior_prob = stats.norm.expect(func = lambda x : generalized_logistic(args, x, allocation_sigma=1), 
                                       loc = post_state_mean.numpy()[0], 
                                       scale = np.sqrt(post_state_var.numpy()[0]) )
   

    import ipdb; ipdb.set_trace()


    return action1probs, std_error
"""


class PosteriorSampling:
    """
    Posterior sampling algorithm
    """
    def __init__(self, args, state_feats, treat_feats, alg_seed, steepness, allocation_sigma=1):
        self.args = args
        self.state_feats = state_feats
        self.treat_feats = treat_feats
        self.alg_seed = alg_seed
        self.rng = np.random.default_rng(alg_seed)
        self.allocation_sigma = allocation_sigma
        self.steepness = steepness

        total_dim = len(state_feats) + len(treat_feats)
        self.prior_mean = np.zeros(total_dim)
        self.prior_var = np.eye(total_dim)
        
        alg_dict = {
            "policy_last_t": 0,
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
            "post_mean": self.prior_mean,
            "post_var": self.prior_var,
            "new_data" : None, 
            "total_obs" : 0,
            "all_user_id": set(),
            "post_V": np.linalg.inv(self.prior_var),
            "all_user_id": set(),
            # data used to get updated est_params
        }
        self.all_policies = [alg_dict]
        
   
    def update_alg(self, new_data, t):

        # update algorithm with new data
        actions = new_data['action'].to_numpy().reshape(-1,1)
        rewards = new_data['reward'].to_numpy().reshape(-1,1)
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
        
        alg_dict = {
            "policy_last_t": t-1,
            "new_data" : new_data,  # data used to get updated est_params
            "design" : X_vecs,
            "total_obs" : self.all_policies[-1]['total_obs'] + len(new_data),
            "prior_mean": self.prior_mean,
            "prior_var": self.prior_var,
        }
        
        post_mean = self.all_policies[-1]["post_mean"]
        inv_post_var = np.linalg.inv( self.all_policies[-1]["post_var"] )
        RX = np.matmul(inv_post_var, post_mean)
        new_RX = RX + np.sum(X_avail*rewards_avail, 0)
        new_inv_post_var = inv_post_var + np.einsum( 'ij,ik->jk', X_avail, X_avail )
        new_post_var = np.linalg.inv( new_inv_post_var )
        new_post_mean = np.matmul( new_post_var, new_RX )
        
        col_names = self.state_feats+['action:'+x for x in self.treat_feats]
        #self.treat_feats_action = ['action:'+x for x in self.treat_feats]
        #post_mean_df = pd.DataFrame(new_post_mean.reshape(1,-1),
        #                         columns=self.state_feats+self.treat_feats_action)
        
        all_user_id_set = self.all_policies[-1]["all_user_id"].copy()
        all_user_id_set.update( new_data["user_id"].to_numpy() )
        post_V = new_inv_post_var / len(all_user_id_set)
        
        alg_dict["all_user_id"] = all_user_id_set
        alg_dict["post_mean"] = new_post_mean
        alg_dict["post_var"] = new_post_var
        alg_dict["col_names"] = col_names
        alg_dict["post_V"] = post_V

        triu_idx = np.triu_indices(post_V.shape[0])
        post_V_params = post_V[triu_idx]
        est_params = np.hstack( [ new_post_mean, post_V_params ] )
        alg_dict["est_params"] = est_params
       
        self.all_policies.append(alg_dict)


    def get_action_probs(self, curr_timestep_data):
        user_states = curr_timestep_data[self.state_feats]
        treat_states = user_states[self.treat_feats].to_numpy()

        user_states = curr_timestep_data[self.state_feats]
        treat_states = user_states[self.treat_feats].to_numpy()
        n_tmp, treat_dim = treat_states.shape
        
        
        # Batch posterior parameters
        post_mean = self.all_policies[-1]["post_mean"]
        post_V = self.all_policies[-1]["post_V"] 
        post_mean_torch = torch.from_numpy( self.all_policies[-1]["post_mean"] )
        post_V_torch = torch.from_numpy( self.all_policies[-1]["post_V"] )
        
        post_mean_torch_batch = post_mean_torch.repeat( 
                                        (curr_timestep_data.shape[0], 1) )
        post_V_torch_batch = post_V_torch.repeat( 
                                        (curr_timestep_data.shape[0], 1, 1) )
        post_mean_treat_torch_batch = post_mean_torch_batch[:,-treat_dim:]
        post_V_treat_torch_batch = post_V_torch_batch[:,-treat_dim:, -treat_dim:]
        
        
        # function that takes in S, V, Z, Mu and outputs rho
        def smooth_posterior_sampling_eval(args, z_sample, n_users, treat_state_user, post_mean_user, post_V_user):
           
            torch_state = torch.from_numpy(treat_state_user)

            # Form posterior variance
            post_Var_user = torch.linalg.inv( max(n_users,1) * post_V_user )
           
            # Square-root of variance
            A_, B_, C_ = torch.linalg.svd( max(n_users,1) * post_V_user )
                # to reproduce: torch.matmul( torch.matmul(A_, torch.diag(torch.sqrt(B_))), C_.T )
            post_Var_sqrt_user = torch.matmul( torch.matmul(A_, torch.diag( torch.sqrt(B_) )), C_.T )

            SV_norm = torch.linalg.vector_norm( torch.matmul(torch_state, post_Var_sqrt_user), 2 )
            SMu = torch.dot(torch_state, post_mean_user)

            return generalized_logistic_torch(args, SV_norm*z_sample + SMu, allocation_sigma=1)            


        # Form action selection probabilities
        all_action1_probs = []
        n_users = len(self.all_policies[-1]["all_user_id"])
        for i in range(n_tmp):
            #prob = smooth_posterior_sampling_eval(self.args, z_sample=1, n_users=n_users, 
            #                               treat_state_user = treat_states[i], 
            #                               post_mean_user = post_mean_treat_torch_batch[i], 
            #                               post_V_user = post_V_treat_torch_batch[i])

            action1_prob = stats.norm.expect(
                                func = lambda x : smooth_posterior_sampling_eval(
                                           self.args, z_sample=x, n_users=n_users,
                                           treat_state_user = treat_states[i],
                                           post_mean_user = post_mean_treat_torch_batch[i],
                                           post_V_user = post_V_treat_torch_batch[i]), 
                                loc = 0, scale=1 )

            all_action1_probs.append( action1_prob )

        return np.array(all_action1_probs)

    
    def get_pi_gradients(self, curr_timestep_data, curr_policy_dict, verbose=False):

        import ipdb; ipdb.set_trace()

        user_states = curr_timestep_data[self.state_feats]
        post_mean = curr_policy_dict["post_mean"]
        post_V = self.all_policies[-1]["post_V"]
        n_users = len( self.all_policies[-1]["all_user_id"] )

        triu_idx = np.triu_indices(post_V.shape[0])
        post_V_params = post_V[triu_idx]
        
        # Batch policy parameters
        policy_params = np.hstack([post_mean, post_V_params.flatten()])
        policy_params_torch = torch.from_numpy( policy_params )
        batch_policy_params = policy_params_torch.repeat( 
                            (curr_timestep_data.shape[0], 1) )
        batch_policy_params.requires_grad = True
        
        # Treatment effect parameters
        param_dim = len(post_mean)
        treat_dim = len(self.treat_feats)
        
        post_mean_torch = batch_policy_params[:,:param_dim]
        post_mean_treat_torch = post_mean_torch[:,treat_dim:]
        
        # Posterior mean parameters
        post_V_param_torch = batch_policy_params[:,param_dim:]
        triu_row_lens = [0] + [x for x in range(param_dim, 0, -1)]
        triu_row_cumlens = np.cumsum(triu_row_lens)
        begin_end_idx = [ (triu_row_cumlens[x], triu_row_cumlens[x+1]) for x in range(len(triu_row_cumlens)-1) ]
        row_pieces = [ post_V_param_torch[:,x:y] for x,y in begin_end_idx]
        
        new_row_pieces = []
        for j in range(len(row_pieces)):
            tmp_row = row_pieces[j]
            num_add = param_dim - tmp_row.shape[1]
            prefix = []
            for k in range(num_add):
                tmp = row_pieces[k][:,-(param_dim-j)]
                prefix.append( tmp.reshape(-1,1) )
            new_row = torch.cat(prefix + [ tmp_row ], axis=1)
            new_row_pieces.append( new_row )
        
        post_V_torch = torch.stack(new_row_pieces, axis=2)
        post_var_torch = torch.linalg.inv(post_V_torch) / n_users
        post_var_treat_torch = post_var_torch[:,treat_dim:,treat_dim:]
        
        """
        test = [[1,2,3,4], [5,6,7], [8,9], [10]]
        new_row_pieces = []
        row_dim = 4
        for j in range(len(test)):
            print(j, "hi")
            tmp_row = test[j]
            num_add = row_dim - len(tmp_row)
            prefix = []
            for k in range(num_add):
                tmp = test[k]
                prefix.append(tmp[-(row_dim-j)])
            new_row_pieces.append( prefix + tmp_row )
        print(new_row_pieces)
        """

        """
        triu_idx_torch = torch.triu_indices(post_V.shape[0], post_V.shape[1])
        
        post_V_torch = torch.zeros((post_mean_torch.shape[0], post_mean_torch.shape[1], post_mean_torch.shape[1]))
        post_V_torch[:,triu_idx_torch] = batch_policy_params[:,param_dim:]

        aaa = np.zeros(post_V.shape)
        aaa[triu_idx] = post_V_params
        aaa + aaa.T - np.diag( np.diag(aaa) )
        
        post_V_torch = batch_policy_params[:,param_dim:].reshape(-1, param_dim, param_dim)
        post_var_torch = torch.linalg.inv(post_V_torch) / n_users
        post_var_treat_torch = post_var_torch[:,treat_dim:,treat_dim:]
        """
        
        # Treatment states
        treat_states = curr_timestep_data[self.treat_feats].to_numpy()
        
        # Forming action 1 selection probability
        action1probs, max_std_error = smooth_posterior_sampling(self.args, treat_states, post_mean_treat_torch,
                                                                post_var_treat_torch, MC_N=self.MC_N, 
                                                                allocation_sigma=self.allocation_sigma, rng=self.rng)
        
        # Weighted policy gradients
        actions = curr_timestep_data['action'].to_numpy()
        actions_torch = torch.from_numpy( actions ) 
        pis_A = actions_torch*action1probs + (1-actions_torch)*(1-action1probs)
        pis_behavior = torch.from_numpy( torch.clone(pis_A).detach().numpy() )
        weights = pis_A / pis_behavior
        weights.sum().backward()
        weighted_pi_grad = batch_policy_params.grad.numpy()
        
        # Check that reproduced the action selection probabilities correctly
        prob_error = action1probs.detach().numpy() - curr_timestep_data['action1prob'].to_numpy()
        if np.max( np.absolute(prob_error) ) > 0.01:
            print("prob error issue", np.max( np.absolute(prob_error) ))
            import ipdb; ipdb.set_trace()
        
        #prob_ratio = np.round( action1probs.detach().numpy(), 2) / np.round(curr_timestep_data['action1prob'], 2)
        #if not np.all( np.round(prob_ratio, 1) == 1 ):
        #    print("prob ratio issue", np.max(prob_ratio), np.min(prob_ratio))
        #    import ipdb; ipdb.set_trace()

        #assert np.all( np.round( action1probs.detach().numpy(), 1) / 
        #                  np.round(curr_timestep_data['action1prob'], 1) == 1)

        #assert np.all( np.round( action1probs.detach().numpy(), 2) / 
        #              np.round(curr_timestep_data['action1prob'], 2) == 1)
        
        return weighted_pi_grad
    

    def get_est_eqns(self, data_sofar, curr_policy_dict, all_user_ids):
        actions = data_sofar.action.to_numpy().reshape(-1,1)
        X_vecs = np.concatenate( [ data_sofar[self.state_feats].to_numpy(), 
                    actions * data_sofar[self.treat_feats].to_numpy() ], axis=1 )
        
        outcome_vec = data_sofar.reward.to_numpy()
        design = X_vecs
        user_ids = data_sofar.user_id.to_numpy()
        
        post_mean = curr_policy_dict['post_mean']
        post_V = curr_policy_dict['post_V']
        
        if self.args.dataset_type == 'heartsteps':
            avail_vec = data_sofar.availability.to_numpy()
        else:
            avail_vec = np.ones(outcome_vec.shape)
        
        est_eqn_dict = get_est_eqn_posterior_sampling(outcome_vec, design, user_ids, 
                                            post_mean=post_mean, post_V=post_V, avail_vec=avail_vec, 
                                            prior_mean=curr_policy_dict['prior_mean'], 
                                            prior_var=curr_policy_dict['prior_var']) #HC3

        
        # Checks ##########################
        
        # total number of observations match
        assert curr_policy_dict['total_obs'] == len(data_sofar)
        
        # estimating equation sums to zero
        ave_est_eqn = np.sum(est_eqn_dict["est_eqns"], axis=0)
        assert np.sum( np.absolute( ave_est_eqn ) ) < 1
        
        # hessians are symmetric
        inv_hessian = np.around(est_eqn_dict['normalized_inv_hessian'], 5)
        try:
            assert np.all( inv_hessian == inv_hessian.T )
        except:
            print( inv_hessian )
            assert np.all( inv_hessian == inv_hessian.T )

        return est_eqn_dict


