import argparse
import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import scipy.stats as stats
import pickle as pkl
import csv
import os
import time
import matplotlib.pyplot as plt
import scipy
from scipy import optimize
import scipy.special

from least_squares_helper import get_est_eqn_LS, fit_WLS

def c_vec2string(c_vec):
    return np.array2string( c_vec )[1:-1].replace(" ", ",")


###############################################################
# Initialize Hyperparameters ##################################
###############################################################

parser = argparse.ArgumentParser(description='Analyze data')
parser.add_argument('--dataset_type', type=str, default='synthetic',
                    choices=['heartsteps', 'synthetic', 'oralytics'])
parser.add_argument('--verbose', type=int, default=0,
                    help='Prints helpful info')
parser.add_argument('--heartsteps_mode', default='medium',
                    choices=['evalSim', 'realistic', 'medium', 'easy'],
                    help='Sets default parameter values accordingly')
parser.add_argument('--inference_mode', default='value',
                    choices=['value', 'model'],
                    help='Sets default parameter values accordingly')
parser.add_argument('--synthetic_mode', type=str, default='delayed_effects',
                    help='File name of synthetic env params')
parser.add_argument('--RL_alg', default='sigmoid_LS',
                    choices=["fixed_randomization", "sigmoid_LS", "posterior_sampling"],
                    help='RL algorithm used to select actions')
parser.add_argument('--err_corr', default='time_corr',
                    choices=["time_corr", "independent"],
                    help='Noise error correlation structure')
parser.add_argument('--N', type=int, default=10,
                    help='Number of Monte Carlo repetitions')
parser.add_argument('--n', type=int, default=90, help='Total number of users')
parser.add_argument('--eta', type=float, default=0, help='Regularization of alg bread matrix')
parser.add_argument('--upper_clip', type=float, default=0.9,
                    help='Upper action selection probability constraint')
parser.add_argument('--lower_clip', type=float, default=0.1,
                    help='Lower action selection probability constraint')
parser.add_argument('--fixed_action_prob', type=float, default=0.5,
                    help='Used if not using learning alg to select actions')
parser.add_argument('--min_users', type=int, default=25,
                    help='Min number of users needed to update alg')
parser.add_argument('--decisions_between_updates', type=int, default=1,
                    help='Number of decision times beween algorithm updates')
parser.add_argument('--save_dir', type=str, default=".",
                    help='Directory to save all results in')
parser.add_argument('--debug', type=int, default=0,
                    help='Debug mode')
parser.add_argument('--steepness', type=float, default=10,
                    help='Allocation steepness')
parser.add_argument('--alg_state_feats', type=str, default="intercept",
                    help='Comma separated list of algorithm state features')
tmp_args = parser.parse_known_args()[0]

print(tmp_args)

if tmp_args.dataset_type == 'heartsteps':
    raise ValueError("Not implemented")
elif tmp_args.dataset_type == 'synthetic':
    arg_dict = { "T" : 2, 'recruit_n': tmp_args.n, 'recruit_t': 1, 
               "allocation_sigma": 1}
    dval_alg = 2*len( tmp_args.alg_state_feats.split(",") )
   
    """
    # Inference Objective
    feature_names = ['intercept', 'past_reward', 'action', 'action_past_reward']
    outcome_name = 'reward'
    c_vec_list = [ np.array([0,0,1,0]), np.array([0,0,0,1]) ]
    dval = 4
    
    cvec2name = {
        c_vec2string(np.array([1,1,1,1])) : "Entire Vector",
        c_vec2string(np.array([0,0,1,1])) : "Margin Vector",
        c_vec2string(np.array([0,0,0,1])) : "Moderated Treatment Effect", 
        c_vec2string(np.array([0,0,1,0])) : "Marginal Treatment Effect", 
    }
    """
    if tmp_args.inference_mode == "value":
        # Inference Objective
        feature_names = ['intercept']
        outcome_name = 'reward'
        c_vec_list = [ np.array([1]) ]
        dval = 1
        
        cvec2name = {
            c_vec2string(np.array([1])) : "Entire Vector",
        }
    elif tmp_args.inference_mode == "model_simple":
        # Inference Objective
        feature_names = ['intercept', 'action']
        outcome_name = 'reward'
        c_vec_list = [ np.array([1, 1]), np.array([0,1]) ]
        dval = 2
        
        cvec2name = {
            c_vec2string(np.array([1,1])) : "Entire Vector",
            c_vec2string(np.array([0,1])) : "Treatment Effect",
        }
    elif tmp_args.inference_mode == "model":
        # Inference Objective
        feature_names = ['intercept', 'past_reward', 'action']
        outcome_name = 'reward'
        c_vec_list = [ np.array([1,1,1]), np.array([0,0,1]) ]
        dval = 3
        
        cvec2name = {
            c_vec2string(np.array([1,1,1])) : "Entire Vector",
            c_vec2string(np.array([0,0,1])) : "Treatment Effect",
        }
    else:
        raise ValueError("invalid inference mode")
    
    stat2name = {
            'hotelling': "Hotelling's t-squared Statistic",
    }
elif tmp_args.dataset_type == 'oralytics':
    arg_dict = { "T" : 50, 'recruit_n': tmp_args.n, 'recruit_t': 1, 
                "allocation_sigma": 5.7 }
    #allocation_sigma: 163 (truncated brush times); 5.7 (square-root of truncatred brush times)
    
    # Inference Objective
    feature_names = ['intercept', 'time_of_day', 'prior_day_brush', 'action']
    #feature_names = ['intercept', 'time_of_day', 'weekend', 'prior_day_brush', 'action']
    outcome_name = 'reward'
    """
    c_vec_list = [ np.array([1,1,1,1,1]), np.array([0,0,0,0,1]), np.array([0,0,0,1,0]) ]
    dval = 5
    
    cvec2name = {
        c_vec2string(np.array([1,1,1,1,1])) : "Entire Vector",
        c_vec2string(np.array([0,0,0,1,0])) : "prior_day_brush",
        c_vec2string(np.array([0,0,0,0,1])) : "Margin Vector",
    }
    """
    stat2name = {
            'hotelling': "Hotelling's t-squared Statistic",
    }
    
    c_vec_list = [ np.array([1,1,1,1]), np.array([0,0,0,1]), np.array([0,0,1,0]) ]
    dval = 4
    
    cvec2name = {
        c_vec2string(np.array([1,1,1,1])) : "Entire Vector",
        c_vec2string(np.array([0,0,1,0])) : "prior_day_brush",
        c_vec2string(np.array([0,0,0,1])) : "Margin Vector",
    }

parser.add_argument('--T', type=int, default=arg_dict['T'],
                    help='Total number of decision times per user')
parser.add_argument('--recruit_n', type=int, default=arg_dict['recruit_n'],
                    help='Number of users recruited on each recruitment times')
parser.add_argument('--recruit_t', type=int, default=arg_dict['recruit_t'],
                    help='Number of updates between recruitment times')
parser.add_argument('--allocation_sigma', type=float, default=arg_dict['allocation_sigma'],
                    help='Sigma used in allocation of algorithm')

args = parser.parse_args()
print(vars(args))



###############################################################
# Load Data ###################################################
###############################################################

def load_data(folder_path):
    study_df = pd.read_csv( os.path.join(folder_path, "data.csv") )
    if args.dataset_type == 'synthetic':
        study_df['action_past_reward'] = study_df['past_reward']*study_df['action']
    
    if args.RL_alg == 'fixed_randomization':
        return study_df
   
    with open( os.path.join(folder_path, "out_dict.pkl"), 'rb' ) as file:
        alg_out_dict = pkl.load( file )

    with open('{}/study_RLalg.pkl'.format(folder_path), 'rb') as f:
        study_RLalg = pkl.load(f)

    return study_df, alg_out_dict, study_RLalg



###############################################################
# Form Estimator ##############################################
###############################################################


def form_LS_estimator(study_df):
    if args.dataset_type == 'heartsteps':
        avail_vec = study_df['availability'].to_numpy()
    else:
        avail_vec = np.ones(study_df['user_id'].shape)
    design = study_df[feature_names].to_numpy() * avail_vec.reshape(-1,1)
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec
    
    LS_estimator = fit_WLS(design, outcome_vec)
    return LS_estimator


def form_LS_est_eqn(est_param, study_df, all_user_ids, correction="HC3"):
    if args.dataset_type == 'heartsteps':
        avail_vec = study_df['availability'].to_numpy()
    else:
        avail_vec = np.ones(study_df['user_id'].shape)
    design = study_df[feature_names].to_numpy() * avail_vec.reshape(-1,1)
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec
 
    present_user_ids = study_df['user_id'].to_numpy()
    LS_dict = get_est_eqn_LS(outcome_vec, design, present_user_ids, 
                             est_param, avail_vec, all_user_ids, 
                             correction=correction)

    return LS_dict["est_eqns"]


###############################################################
# Sandwich Variance Estimator #################################
###############################################################


def get_sandwich_var(est_eqns, normalized_hessian, LS_estimator):
    n_unique = est_eqns.shape[0]
    
    meat = np.einsum( 'ij,ik->jk', est_eqns, est_eqns )
    meat = meat / n_unique
    
    # degrees of freedom adjustment
    meat = meat * (n_unique-1) / (n_unique - len(LS_estimator)) 
    est_val_dict['meat'] = meat

    inv_hessian = np.linalg.inv( normalized_hessian )
    sandwich_var = np.matmul( np.matmul(inv_hessian, meat), inv_hessian )
    sandwich_var = sandwich_var / n_unique
    
    return sandwich_var


###############################################################
# Adaptive Sandwich Variance Estimator ########################
###############################################################

def get_stacked_estimating_function(all_beta_params, beta_dim, theta_dim,
                                    study_RLalg, study_df, return_full=False):
    all_user_ids = np.unique( study_df['user_id'].to_numpy() )

    thetahat = all_beta_params[-theta_dim:]

    num_updates = int( len(all_beta_params[:-theta_dim]) / beta_dim )
    est_list = np.split(all_beta_params[:-theta_dim], num_updates)
    est_list.append( thetahat )

    all_weights = [ np.ones(len(all_user_ids)) ]
    all_est_eqns = []
    all_est_eqns_full = []
   
    for policy_num, curr_policy_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue

        beta_est = est_list[policy_num-1]

        
        # Form estimating equations ################
        data_sofar = study_df[ study_df['policy_last_t'] < policy_last_t ]
            # data_sofar = all data used to estimate the policy at time policy_last_t
        assert curr_policy_dict['total_obs'] == len(data_sofar)
            # total number of observations match

        est_eqns_dict = study_RLalg.get_est_eqns(beta_params=beta_est,
                                                 data_sofar=data_sofar,
                                                 all_user_ids=all_user_ids)
            # we require all RL algorithms to have a function `get_est_eqns` that
                # forms the estimating equation for policy parameters

        # Check estimating equation sums to zero
        ave_est_eqn = np.sum(est_eqns_dict["est_eqns"], axis=0) # TODO incremental recruitment
        assert np.sum( np.absolute( ave_est_eqn ) ) < 1
       
        # Multiply by weights
        prev_weights_prod = np.prod(all_weights, axis=0)
        prev_weights_prod = np.expand_dims(prev_weights_prod, 1)
        weighted_est_eqn = prev_weights_prod * est_eqns_dict["est_eqns"]
        all_est_eqns.append( np.mean(weighted_est_eqn, axis= 0)  )
        all_est_eqns_full.append( weighted_est_eqn )


        # Form weights ################
        curr_policy_decision_data = study_df[ study_df['policy_last_t'] == policy_last_t ]
            # curr_policy_decision_data = all data collected using the policy policy_last_t

        user_pi_weights = study_RLalg.get_weights(curr_policy_decision_data, 
                                                  beta_est, all_user_ids=all_user_ids)

        # Check that reproduced the action selection probabilities correctly
        try:
            assert np.all(np.around(user_pi_weights,5) == 1)
        except:
            import ipdb; ipdb.set_trace()

        all_weights.append( user_pi_weights )
   

    # Estimating Equation for theta ######################
    theta_est_eqn = form_LS_est_eqn(thetahat, study_df, all_user_ids,
                                    correction="")
        
    # Multiply by weights
    prev_weights_prod = np.prod(all_weights, axis=0)
    prev_weights_prod = np.expand_dims(prev_weights_prod, 1)
    weighted_est_eqn = prev_weights_prod * theta_est_eqn
    all_est_eqns.append( np.mean(weighted_est_eqn, axis= 0)  )
    
    all_est_eqns_full.append( weighted_est_eqn )

    if return_full:
        return {
                "ave_est_eqn": np.concatenate( all_est_eqns ),
                "all_est_eqn": all_est_eqns_full,
                }

    return np.concatenate( all_est_eqns )



def get_adaptive_sandwich_new(all_est_eqn_dict, study_RLalg, study_df, 
                              alg_correction=""):

    all_est_params = []
    for policy_num, curr_policy_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue
        
        est_params = curr_policy_dict['est_params'].to_numpy().squeeze()
        all_est_params.append(est_params)

    # This is new
    thetahat = all_est_eqn_dict['estimator']
    theta_dim = len(thetahat)
    all_est_params.append( thetahat )
    
    beta_dim = len(est_params)
    all_est_params = np.hstack(all_est_params)
    
    
    # Form Stacked Meat Estimator ##########################
    stacked_est_dict = get_stacked_estimating_function(all_est_params, 
                                                  beta_dim, theta_dim, 
                                                  study_RLalg, study_df,
                                                  return_full=True)
    cat_est_eqn = np.hstack( stacked_est_dict['all_est_eqn'] )
    stacked_raw_meat = np.einsum('ij,ik->jk', cat_est_eqn, cat_est_eqn)
    n_unique = cat_est_eqn.shape[0]

    stacked_meat = stacked_raw_meat / n_unique
    stacked_meat = stacked_meat * (n_unique-1) / (n_unique - theta_dim)


    # Form Stacked Bread Estimator ##########################
    eps = np.sqrt(np.finfo(float).eps)
    stacked_hessian = optimize.approx_fprime(all_est_params, 
                                        get_stacked_estimating_function,
                                        eps, beta_dim, theta_dim,
                                        study_RLalg, study_df)
  
    # Form Adaptive Sandwich Variance Estimator ###############
    stacked_hessian_inv = np.linalg.inv(stacked_hessian)

    adaptive_sandwich_stacked = np.matmul( \
            np.matmul( stacked_hessian_inv, stacked_meat ), \
            stacked_hessian_inv.T )
    adaptive_sandwich_stacked = adaptive_sandwich_stacked / n_unique

    adaptive_sandwich_theta = adaptive_sandwich_stacked[-theta_dim:,-theta_dim:]


    return_dict = {
            "stacked_meat": stacked_meat,
            "stacked_hessian": stacked_hessian,
            "stacked_sandwich": adaptive_sandwich_stacked,
            "adaptive_sandwich": adaptive_sandwich_theta,
            }
    return return_dict


def hi():
    print("123")
    import ipdb; ipdb.set_trace()
    
    
    all_user_ids = np.unique( study_df['user_id'].to_numpy() )
    for policy_num, curr_policy_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue
        data_sofar = study_df[ study_df['policy_last_t'] < policy_last_t ]
        #import ipdb; ipdb.set_trace()
    
        # TESTING FD FOR GETTING Hessian
        eps = np.sqrt(np.finfo(float).eps)
        normalized_hessian = optimize.approx_fprime(est_params, study_RLalg.get_est_eqns,
                                                    eps, data_sofar, all_user_ids, True)

        est_eqns_dict_oracle = study_RLalg.get_est_eqns_full(data_sofar=data_sofar,
                                                 curr_policy_dict=curr_policy_dict,
                                                 all_user_ids=all_user_ids)

        all_user_ids = np.unique( study_df['user_id'].to_numpy() )
        import ipdb; ipdb.set_trace()

    # Update t Estimating Function (takes beta params as first arg)

    # TODO make function that makes whole stacked estimating function to make big bread
    # TODO make big stacked meat
    # TODO add in theta estimating function


    for policy_num, curr_policy_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue

        est_params = curr_policy_dict['est_params'].to_numpy().squeeze()

        # Form estimating equations ################
        data_sofar = study_df[ study_df['policy_last_t'] < policy_last_t ]
            # data_sofar = all data used to estimate the policy at time policy_last_t
        assert curr_policy_dict['total_obs'] == len(data_sofar)
            # total number of observations match

        est_eqns_dict = study_RLalg.get_est_eqns(beta_params=est_params,
                                                 data_sofar=data_sofar,
                                                 all_user_ids=all_user_ids)
            # we require all RL algorithms to have a function `get_est_eqns` that
                # forms the estimating equation for policy parameters

        # Check estimating equation sums to zero
        ave_est_eqn = np.sum(est_eqns_dict["est_eqns"], axis=0)
        assert np.sum( np.absolute( ave_est_eqn ) ) < 1



        # Form weights ################
        curr_policy_decision_data = study_df[ study_df['policy_last_t'] == policy_last_t ]
            # curr_policy_decision_data = all data collected using the policy policy_last_t

        user_pi_weights = study_RLalg.get_weights(curr_policy_decision_data, est_params,
                                all_user_ids=all_user_ids)

        # Check that reproduced the action selection probabilities correctly
        try:
            assert np.all(np.around(user_pi_weights,5) == 1)
        except:
            import ipdb; ipdb.set_trace()

        # TESTING FD FOR GETTING Hessian
        eps = np.sqrt(np.finfo(float).eps)
        normalized_hessian = optimize.approx_fprime(est_params, study_RLalg.get_est_eqns,
                                                    eps, data_sofar, all_user_ids, True)

        est_eqns_dict_oracle = study_RLalg.get_est_eqns_full(data_sofar=data_sofar,
                                                 curr_policy_dict=curr_policy_dict,
                                                 all_user_ids=all_user_ids)


        import ipdb; ipdb.set_trace()


###############################################################
# Adaptive Sandwich Variance Estimator OLD ########################
###############################################################


def data_check():
    # Estimating equations sum to zero
    # Hessian is symmetric
    # Pigradient?
    pass


def get_adaptive_sandwich(all_est_eqn_dict, Teval, alg_correction=""):
    """
    - est_val_dict : Dictionary with estimating equations for inference Z-estimator
    - alg_out_dict : Dictionary with estimating equations for algorithm parameter Z-estimators
    """
    if alg_correction != "":
        alg_eqn_name = "est_eqns_{}".format(alg_correction)
    else:
        alg_eqn_name = "est_eqns"

    # Stack estimating equations #############################
    all_est_eqns = []
    for u in range(1, Teval):
        all_est_eqns.append( all_est_eqn_dict[u][alg_eqn_name] )
    all_est_eqns.append( all_est_eqn_dict[Teval]['est_eqns_HC3'] )
    stacked_est_eqn = np.hstack(all_est_eqns) 
        # num_users x dim_all_policy_inf_params
  
    #import ipdb; ipdb.set_trace() incremental recruitment

    # Form Meat Matrix ##################################
    n_unique, theta_dim = all_est_eqn_dict[Teval]['est_eqns_HC3'].shape
    beta1dim = all_est_eqn_dict[1][alg_eqn_name].shape[1]
    
    stacked_raw_meat = np.einsum( 'ij,ik->jk', stacked_est_eqn, stacked_est_eqn )
    stacked_meat = stacked_raw_meat / n_unique
    stacked_meat = stacked_meat * (n_unique-1) / (n_unique - theta_dim)
    all_param_dim = stacked_est_eqn.shape[1]
    #all_est_eqn_dictstacked_meat = stacked_meat * (n_unique-1) / (n_unique - all_param_dim)
    # TODO: improve small sample correction

    """
    # Code Check: Reproduce Standard Sandwich Meat
    print( stacked_meat[-theta_dim:,-theta_dim:] )
    print( est_val_dict['meat'] )
    import ipdb; ipdb.set_trace()
    """

    # Stack Policy Gradients #################################
    all_pi_grads = []
    for u in range(1, Teval):
        all_pi_grads.append( all_est_eqn_dict[u]['pi_grads'] )

    stacked_pi_grads = np.hstack(all_pi_grads) 
        # num_users x dim_all_policy_params

    # Form "Bread" Matrix #####################################

    # Form Block Diagonal ##############
    all_hessian_matrices = []
    all_hessian_mask = []
    for u in range(1, Teval+1):
        normalized_hessian = all_est_eqn_dict[u]['normalized_hessian']
        all_hessian_matrices.append( normalized_hessian )
        all_hessian_mask.append( np.ones(normalized_hessian.shape) ) 

    #np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
    bread_block_diag = scipy.linalg.block_diag(*all_hessian_matrices)
    bread_block_diag_mask = scipy.linalg.block_diag(*all_hessian_mask)

    upper_triangle_ones = np.tri(*bread_block_diag_mask.shape).T
    bread_block_upper_mask = bread_block_diag_mask + upper_triangle_ones
    bread_block_upper_mask -= 1*(bread_block_upper_mask == 2)

    ########################################
    # Form V Matrices ######################
    # V_{1,1}  0        0 
    # V_{2,1}  V_{2,2}  0
    # V_{3,1}  V_{3,2}  V_{3,3}
    ###############################
    # For the example above, `Vmatrix_normalized_raw` is the matrix
    # (part of the ``bread'' that involves policy gradients)
    # V_{2,1}  V_{2,2}
    # V_{3,1}  V_{3,2}
    ################################
    stacked_est_eqn_noBeta1 = stacked_est_eqn[:,beta1dim:]
    Vmatrix_raw = np.einsum( 'ij,ik->jk', stacked_est_eqn_noBeta1, stacked_pi_grads )
    Vmatrix_normalized_raw = Vmatrix_raw / n_unique
    ################################
    # Now put Vmatrix_normalized_raw in the bottom left corner to get
    # 0        0        0
    # V_{2,1}  V_{2,2}  0
    # V_{3,1}  V_{3,2}  0
    ################################
    all_Vmatrix = np.zeros( bread_block_diag.shape )
    all_Vmatrix[beta1dim:,:-theta_dim] = Vmatrix_normalized_raw
   
    final_bread = all_Vmatrix*(1-bread_block_upper_mask) + bread_block_diag
    # Regularization
    final_bread[:-theta_dim,:-theta_dim] = final_bread[:-theta_dim,:-theta_dim] + \
            np.eye(final_bread[:-theta_dim,:-theta_dim].shape[0])*args.eta
    
    final_bread_inv = np.linalg.inv(final_bread)    
   
    #np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    #import ipdb; ipdb.set_trace()

    # Form "Adaptive Sandwich" Matrix #####################################
    adaptive_sandwich_full = np.matmul( np.matmul( final_bread_inv, stacked_meat ), \
            final_bread_inv.T )
    adaptive_sandwich_full = adaptive_sandwich_full / n_unique

    adaptive_sandwich_theta = adaptive_sandwich_full[-theta_dim:,-theta_dim:]

    #import ipdb; ipdb.set_trace()
    #print( 'bread_inv_max', np.max(final_bread_inv) )
    
    """
    # Code Check: Reproduce Standard Sandwich
    noV_bread = bread_block_diag_mask * final_bread
    noV_bread_inv = np.linalg.inv(noV_bread)
    sandwich_full = np.matmul( np.matmul( noV_bread_inv, \
            bread_block_diag_mask*stacked_meat ), noV_bread_inv.T )
    sandwich_full = sandwich_full / len(all_user_ids)
    sandwich_theta = sandwich_full[-theta_dim:,-theta_dim:]
    print(sandwich_theta)
   
    sandwich_bread = noV_bread_inv[-theta_dim:,-theta_dim:]
    import ipdb; ipdb.set_trace()
    """
    
    # Examining Eigenvalues #####################################
    alg_bread = final_bread[:-theta_dim,:-theta_dim]
    inf_bread = final_bread[-theta_dim:,-theta_dim:]

    inf_eig, inf_eigvec = np.linalg.eig(inf_bread)
    min_inf_eig = np.min(inf_eig)

    alg_eig, alg_eigvec = np.linalg.eig(alg_bread)
    min_alg_eig = np.min(alg_eig)

    eig_dict = {
            "min_inf_eig": min_inf_eig,
            "min_alg_eig": min_alg_eig,
            }

    return adaptive_sandwich_theta, adaptive_sandwich_full, eig_dict, final_bread, stacked_meat


###############################################################
# Analyze Experiments #########################################
###############################################################

def init_nan(size):
    array = np.empty(size)
    array[:] = np.NaN
    return array


tic = time.perf_counter()

if args.dataset_type == "heartsteps":
    mode = args.heartsteps_mode
elif args.dataset_type == "synthetic":
    mode = args.synthetic_mode
elif args.dataset_type == "oralytics":
    mode = None
else:
    raise ValueError("Invalid dataset type")
    

if args.dataset_type == 'oralytics':
    exp_str = '{}_alg={}_T={}_n={}_recruitN={}_decisionsBtwnUpdates={}_steep={}'.format(
            args.dataset_type, args.RL_alg, args.T, args.n, 
            args.recruit_n, args.decisions_between_updates, args.steepness)
else:
    exp_str = '{}_mode={}_alg={}_T={}_n={}_steepness={}_algfeats={}_errcorr={}'.format(
            args.dataset_type, mode, args.RL_alg, args.T, args.n, args.steepness, args.alg_state_feats, args.err_corr)
     
# Load Data
all_folder_path = os.path.join(args.save_dir, "simulated_data/{}".format(exp_str))

# Parameter info
inf_feat_names = [x for x in range(len(feature_names))]

# Policy parameters
all_beta_dict = {}

folder_path = os.path.join(all_folder_path, "exp={}".format(1))
if args.RL_alg == 'fixed_randomization':
    study_df = load_data(folder_path)
else:
    study_df, alg_out_dict, study_RLalg = load_data(folder_path) 


all_estimators = init_nan((args.N,dval))
all_sandwich = init_nan((args.N,dval,dval))
all_adaptive_sandwich = init_nan((args.N,dval,dval))
all_adaptive_sandwich_HC3 = init_nan((args.N,dval,dval))

for i in range(1,args.N+1):
    if i % 100 == 0:
        print('dataset num {}'.format(i))
    
    # Load Data #########################################
    folder_path = os.path.join(all_folder_path, "exp={}".format(i))
    if args.RL_alg == 'fixed_randomization':
        study_df = load_data(folder_path)
    else:
        study_df, alg_out_dict, study_RLalg = load_data(folder_path)

    # Form Estimator #########################################
    LS_estimator = form_LS_estimator(study_df)
    all_estimators[i-1] = LS_estimator
    
    # Form Sandwich Var #######################################
    user_ids = study_df['user_id'].to_numpy()
    unique_user_ids = np.unique(user_ids)
    if args.dataset_type == 'heartsteps':
        avail_vec = study_df['availability'].to_numpy()
    else:
        avail_vec = np.ones(study_df['user_id'].shape)
    design = study_df[feature_names].to_numpy()
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec
    
    est_val_dict = get_est_eqn_LS(outcome_vec, design, user_ids, LS_estimator, avail_vec, 
                   all_user_ids=unique_user_ids, correction="", weights=None)
    est_val_dict_HC3 = get_est_eqn_LS(outcome_vec, design, user_ids, LS_estimator, avail_vec, 
                   all_user_ids=unique_user_ids, correction="HC3", weights=None)
    est_val_dict['est_eqns_HC3'] = est_val_dict_HC3['est_eqns']
    sandwich_var = get_sandwich_var(est_eqns = est_val_dict['est_eqns'], 
                                    normalized_hessian = est_val_dict['normalized_hessian'],
                                    LS_estimator = LS_estimator)
    all_sandwich[i-1] = sandwich_var

    if args.RL_alg == 'fixed_randomization':
        if i == 1:
            policy_param_dict = None
        continue

    # Form Adaptive Sandwich Var ######################################

    #if args.recruit_n != args.n:
    #    raise ValueError("adaptive sandwich variance not implemented for \
    #            incremental recruitment yet")

    num_updates = max( alg_out_dict.keys() )
    
    #### OLD STUFF
    alg_out_dict[num_updates+1] = est_val_dict
    adaptive_sandwich, adaptive_sandwich_full, eig_dict, bread_stacked, stacked_meat = \
            get_adaptive_sandwich(alg_out_dict, Teval=num_updates+1)
    all_adaptive_sandwich[i-1] = adaptive_sandwich
    
    #### NEW STUFF
    # TODO CHANGE ARGUMENT TO EST VAL FUNCTIONS
    tmp = get_adaptive_sandwich_new(est_val_dict, study_RLalg, study_df)

    # np.around(adaptive_sandwich, 3) == np.around(tmp['adaptive_sandwich'], 3)
    # np.around(stacked_meat, 3) == np.around(tmp['stacked_meat'], 3)
    # np.around(tmp['stacked_hessian'],3) == np.around(bread_stacked, 3) 
    import ipdb; ipdb.set_trace()




    adaptive_sandwich_HC3, adaptive_sandwich_full_HC3, eig_dict_HC3, bread_stacked_HC3 = \
            get_adaptive_sandwich(alg_out_dict, Teval=num_updates+1, alg_correction="HC3")
    all_adaptive_sandwich_HC3[i-1] = adaptive_sandwich_HC3

    if args.debug:
    #if True:
        print("sandwich")
        print(sandwich_var)
        
        print("adaptive_sandwich")
        print(adaptive_sandwich)
        
        #import ipdb; ipdb.set_trace()
    



# Compute Standard and Adaptive Sandwich Variance Estimators ###################
sandwich_var = np.mean(all_sandwich, 0)
    
# Functions for Evaluating Variance Estimators ###################

def process_var_hotelling(results_dict, n, dval, fignum=0, name=""):
    assert "Empirical" in results_dict.keys()

    # Vector
    vec_array = np.array(results_dict["Empirical"]['raw_values']).squeeze()
    errors_vec = vec_array - np.mean(vec_array, 0)

    hotelling_dict = {}
    maxX = 0
    for fignum, key in enumerate(results_dict.keys()):
        if key == "Empirical":
            matrix_array = np.expand_dims(results_dict[key]['cov_matrix_normalized'], 0)
            if matrix_array.shape[-1] == 1:
                matrix_array = matrix_array.reshape(1,1,1)
        elif key == "beta_param":
            continue
        else:
            matrix_array = np.array( results_dict[key]['raw_values'] )

        # Hotelling's T statistic for entire vector
        if matrix_array.shape[-1] > 1:
            Sigma_inv = np.linalg.inv(matrix_array)
            hotelling_stat = np.einsum('ij,ij->i', errors_vec,
                            np.einsum('ijk,ik->ij', Sigma_inv, errors_vec) )
        else:
            Sigma_inv = 1/matrix_array.squeeze()
            hotelling_stat = np.square(errors_vec) * Sigma_inv

        cutoff_raw = scipy.stats.f.ppf(0.95, dfn=dval, dfd=n-dval)
        cutoff = cutoff_raw * dval * (n-1) / (n-dval)

        accepts = hotelling_stat <= cutoff
        coverage = round( np.mean(accepts), 4 )
        coverage_sd = round( np.std(accepts) / np.sqrt(len(accepts)), 5 )

        results_dict[key]["hotelling"] = {
            "coverage": ( coverage, coverage_sd ),
        }

        empirical_cutoff = np.percentile(hotelling_stat, 95)
        maxX = max(max(hotelling_stat), maxX)

        hotelling_dict[key] = {
                "cutoff": cutoff,
                "empirical_cutoff": empirical_cutoff,
                "hotelling_stat": hotelling_stat,
                }

    for tmpfignum, key in enumerate(results_dict.keys()):
        if key == "beta_param":
            continue
        # Plotting statistic
        plt.figure(tmpfignum+fignum*100)
        plt.hist(hotelling_dict[key]["hotelling_stat"])
        plt.axvline(x = hotelling_dict[key]["cutoff"], 
                color = 'b', label = 'cutoff')
        plt.axvline(x = hotelling_dict[key]["empirical_cutoff"], 
                color = 'm', label = '95 Percentile')
        plt.legend()
        plt.xlim(0, maxX)
        plt.ylim(0, args.N)
        plt.title("{}".format( key))
        plt.savefig( "{}/hotelling_{}_{}.png".format(all_folder_path, key, name) )

        
def process_var(c_vec, results_dict, n, N, fignum=0):
    assert "Empirical" in results_dict.keys()

    # Vector
    vec_array = results_dict["Empirical"]['raw_values']
    est_eqns = np.einsum('ij,j->i', vec_array, c_vec)
    errors = est_eqns - np.mean(est_eqns)

    #Info on standard error for empirical varaiance:
        #https://en.wikipedia.org/w/index.php?title=Variance&oldid=735567901
            #Distribution_of_the_sample_variance
    est_var = round( np.var(est_eqns*np.sqrt(args.n)), 4 )
    N_tmp = len(est_eqns)
    fourth_m = stats.moment(est_eqns*np.sqrt(args.n), moment=4)
    second_m = np.var(est_eqns*np.sqrt(args.n))
    pre_std_error = ( fourth_m - (N_tmp-3)/(N_tmp-1)*np.square(second_m) ) / N_tmp
    std_error = round( np.sqrt( pre_std_error ), 5 )
    
    est_var_normalized = round( np.var(est_eqns), 4 )
    fourth_m_normalized = stats.moment(est_eqns, moment=4)
    second_m_normalized = np.var(est_eqns)
    pre_std_error_normalized = ( fourth_m - (N_tmp-3)/(N_tmp-1)*np.square(second_m) ) / N_tmp
    std_error_normalized = round( np.sqrt( pre_std_error_normalized ), 5 )
    
    results_dict["Empirical"][c_vec2string( c_vec )] = {
        "var_est": ( est_var, std_error ),
        "var_est_normalized": ( est_var_normalized, std_error_normalized ),
        "errors": errors,
    }

    var_dict = {}
    maxX = 0
    for key in results_dict.keys():
        if key == "Empirical":
            if len(c_vec) > 1:
                est_vars = np.dot( np.matmul(
                                results_dict["Empirical"]["cov_matrix"],
                                c_vec ), c_vec )
                est_vars_normalized = np.dot( np.matmul(
                                results_dict["Empirical"]["cov_matrix_normalized"],
                                c_vec ), c_vec )
            else:
                est_vars = results_dict["Empirical"]["cov_matrix"]
                est_vars_normalized = results_dict["Empirical"]["cov_matrix_normalized"]
            est_vars = np.ones(N)*est_vars

        else:
            # Matrix
            matrix_array = results_dict[key]['raw_values']
            est_vars = np.einsum('ij,j', np.einsum('ijk,k->ij',
                                                   matrix_array, c_vec), c_vec ) * args.n
            est_vars_normalized = np.einsum('ij,j', np.einsum('ijk,k->ij',
                                                   matrix_array, c_vec), c_vec )

        # Variance Estimate with Standard Errors
        ave_var = round( np.mean(est_vars), 4 )
        std_error = round( np.std(est_vars) / np.sqrt(len(est_vars)), 5 )
        median_var = round( np.median(est_vars), 4 )

        # 95% Confidence Interval Coverage
        #cutoff = stats.norm.ppf(1-0.05/2)
        cutoff = stats.t.ppf(1-0.05/2, n-1)
        var_stat = np.absolute( errors / np.sqrt( est_vars_normalized ) )
        accepts = var_stat <= cutoff

        coverage = round( np.mean(accepts), 4 )
        coverage_sd = round( np.std(accepts) / np.sqrt(len(accepts)), 5 )

        if key == "Empirical":
            results_dict[key][c_vec2string( c_vec )]["coverage"] = \
                ( coverage, coverage_sd )
        else:
            results_dict[key][c_vec2string( c_vec )] = {
                "var_est": ( ave_var, std_error ),
                "coverage": ( coverage, coverage_sd ),
                "var_median": median_var,
            }
        
        empirical_cutoff = np.percentile(var_stat, 95)
        maxX = max(max(var_stat), maxX)

        var_dict[key] = {
                "cutoff": cutoff,
                "empirical_cutoff": empirical_cutoff,
                "var_stat": var_stat,
                }
    

# Compute Empirical Variance ###################

results_dict = {}

empirical_var = np.cov(all_estimators.T, ddof=0) * args.n
empirical_var_normalized = np.cov(all_estimators.T, ddof=0)

results_dict['Empirical'] = {
    "raw_values" : all_estimators,
    "cov_matrix" : empirical_var,
    "cov_matrix_normalized" : empirical_var_normalized,
}

# Compute Standard and Adaptive Sandwich Variance Estimators ###################
sandwich_var = np.mean(all_sandwich, 0)

results_dict['Sandwich'] = {
    "raw_values" : all_sandwich,
    "cov_matrix" : sandwich_var * args.n,
    "cov_matrix_normalized" : sandwich_var,
}

if args.RL_alg != 'fixed_randomization':
    adaptive_sandwich_var = np.mean(all_adaptive_sandwich, 0)
    adaptive_sandwich_var_HC3 = np.mean(all_adaptive_sandwich_HC3, 0)

    results_dict['Adaptive_Sandwich'] = {
        "raw_values" : all_adaptive_sandwich,
        "cov_matrix" : adaptive_sandwich_var*args.n,
        "cov_matrix_normalized" : adaptive_sandwich_var,
    }
    results_dict['Adaptive_Sandwich_algHC3'] = {
        "raw_values" : all_adaptive_sandwich_HC3,
        "cov_matrix" : adaptive_sandwich_var_HC3*args.n,
        "cov_matrix_normalized" : adaptive_sandwich_var_HC3,
    }


# Evaluate Confidence Regions ###################

for fignum, c_vec in enumerate(c_vec_list):
    process_var(c_vec, results_dict, args.n, args.N, fignum=fignum)

process_var_hotelling(results_dict, args.n, dval)


###############################################################
# Write Out Results ###########################################
###############################################################

def writeout(outf, results_dict, c_vec_list, printlines=False):
    with open(outf, 'w') as f:
        f.write("\n================================================================================")
        f.write("\n================================================================================\n")
        f.write("T={}".format(args.T))
        
        # Average Covariance Matrices
        f.write("\n========================================\n")
        for key in results_dict.keys():
            f.write( key+"\n")
            f.write( np.array2string( results_dict[key]['cov_matrix'] ) + "\n" )

        # Estimated Variance
        f.write("========================================")
        for c_vec in c_vec_list:
            cvec_str = c_vec2string( c_vec )
            f.write("\nEstimated Variance for {}\n".format(c_vec))
            for key in results_dict.keys():
                varstr = str( results_dict[key][cvec_str]["var_est"] )
                f.write(str(key)+"\t"+varstr+"\n")
                if key != "Empirical":
                    varstr_median = str( results_dict[key][cvec_str]["var_median"] )
                    f.write(str(key)+"\t"+varstr_median+"\n")

        # Confidence Interval Coverage
        f.write("========================================")
        for c_vec in c_vec_list:
            cvec_str = c_vec2string( c_vec )
            f.write("\nEmpirical Coverage for {} (95% CI)\n".format(c_vec))
            for key in results_dict.keys():
                coveragestr = str( results_dict[key][cvec_str]["coverage"] )
                f.write(str(key)+"\t"+coveragestr+"\n")

        f.write("========================================")

        # Hotellings-t statistics
        f.write("\nHotelling's t-statistic (95% CI)\n")
        for key in results_dict.keys():
            #if key == "Empirical":
            #    continue
            coveragestr = str( results_dict[key]["hotelling"]["coverage"] )
            f.write(str(key)+"\t"+coveragestr+"\n")

        f.write("========================================")
        
    if printlines:
        with open(outf, 'r') as f:
            lines = f.readlines()
        for L in lines:
            print(L)


            
def write_latex(latex_path, results_dict, c_vec_list):
    if args.dataset_type == "heartsteps":
        model = args.heartsteps_model
    elif args.dataset_type == "synthetic":
        mode = args.synthetic_mode
    elif args.dataset_type == "oralytics":
        mode = "none"
    else:
        raise ValueError("dataset_type")
        
    with open(latex_path, 'w') as f:
        for c_vec in c_vec_list:
            cvec_str = c_vec2string( c_vec )
            f.write(cvec2name[cvec_str]+"\n")
            all_result_str = []
            result_str = "T={}, n={}, dataset={}, mode={}".format(
                        args.T, args.n, args.dataset_type, mode)
            name_str = ""
            for key in results_dict.keys():
                if key == "Empirical":
                    continue
                name_str = name_str + " & " + key
                coveragestr = str( results_dict[key][cvec_str]["coverage"] )
                result_str = result_str + " & " + coveragestr
            name_str = name_str + " \\\\ \hline\n"
            result_str = result_str + " \\\\ \hline\n"
            all_result_str.append(result_str)
            
            f.write(name_str)
            for result_str in all_result_str:
                f.write(result_str)
            f.write("\n")

        for stat in ['hotelling']:
            f.write(stat2name[stat]+"\n")
            all_result_str = []

            result_str = "T={}, n={}, dataset={}, mode={}".format(
                        args.T, args.n, args.dataset_type, mode)
            name_str = ""
            for key in results_dict.keys():
                if key == "Empirical":
                    continue
                name_str = name_str + " & " + key
                coveragestr = str( results_dict[key][stat]["coverage"] )
                result_str = result_str + " & " + coveragestr
            name_str = name_str + " \\\\ \hline\n"
            result_str = result_str + " \\\\ \hline\n"
            all_result_str.append(result_str)
            
            f.write(name_str)
            for result_str in all_result_str:
                f.write(result_str)
            f.write("\n")
            
  

# Writing Out Results ##########################################

summary_path = os.path.join(all_folder_path, "summary.txt")
writeout(summary_path, results_dict, c_vec_list, printlines=False)
print(summary_path)

with open(summary_path, 'r') as f:
    lines = f.readlines()
for L in lines:
    print(L)

latex_path = os.path.join(all_folder_path, "latex.txt")
write_latex(latex_path, results_dict, c_vec_list)

results_path = os.path.join(all_folder_path, "results.pkl")
print("\nWriting results to {}".format(results_path))
with open(results_path, 'wb') as f:
    pkl.dump(results_dict, f)

