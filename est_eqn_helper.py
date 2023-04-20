import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm


def fit_WLS(design, outcome, weights=None):
    if weights is None:
        numerator = np.sum( design*np.expand_dims(outcome,1), 0 )
        matrix = np.einsum("ij,ik->jk", design, design)
    else:
        numerator = np.sum( np.expand_dims(weights,1)*design*np.expand_dims(outcome,1),
                           0 )
        matrix = np.einsum("ij,ik->jk", design, np.expand_dims(weights,1) * design)

    inv_matrix = np.linalg.inv(matrix)
    return np.matmul( inv_matrix, numerator )



def get_est_eqn_LS_tmp(outcome_vec, design, present_user_ids, est_param, avail_vec, 
                   all_user_ids):
    
    # Get residuals
    lin_prod = design * est_param
    residuals = outcome_vec - np.sum( lin_prod, axis=1 )
    residuals = residuals*avail_vec

    # Get inverse hessian (bread)
    design_avail = design * avail_vec.reshape(-1,1)

    unique_user_ids = np.unique( present_user_ids )
    n_unique = len(unique_user_ids)

    raw_est_eqns = residuals.reshape(-1,1) * design_avail
    
    # Group by / sum over user_id
    est_eqns = []
    user_bool = []
    for idx in all_user_ids:
        if idx in present_user_ids:
            user_est_eqn = np.sum( raw_est_eqns[ present_user_ids == idx ], axis=0 )
            user_bool.append(1)
        else:
            user_est_eqn = np.zeros( raw_est_eqns.shape[1] )
            user_bool.append(0)
        est_eqns.append(user_est_eqn)

    return {
            "est_eqns" : np.vstack(est_eqns),
            "present_user_ids": unique_user_ids,
            "all_user_ids": all_user_ids,
            "user_bool": np.array(user_bool),
            }
            


def get_est_eqn_posterior_sampling(outcome_vec, design, user_ids, post_mean, post_V, avail_vec, 
                                   prior_mean, prior_var, weights=None):
    
    # Preparing Estimating Equation for Posterior Mean ===
    
    # Get residuals
    lin_prod = design * post_mean
    residuals = outcome_vec - np.sum( lin_prod, axis=1 )
    residuals = residuals*avail_vec

    # Get inverse hessian (bread)
    design_avail = design * avail_vec.reshape(-1,1)
    if weights is None:
        W_design = design_avail
    else:
        W_design = design_avail * np.expand_dims(weights, 1)
        
    # Get estimating equations for posterior mean
    residuals_adjust = residuals
    raw_est_eqns_mean = residuals_adjust.reshape(-1,1) * W_design
    
    # Preparing Estimating Equation for Posterior Variance ===
    outer_prod = np.einsum( 'ij,ik->ijk', design_avail, W_design )
    raw_est_eqns_var = outer_prod.reshape( outer_prod.shape[0], -1 )
    
    # Concatenating ====
    raw_est_eqns = np.hstack([raw_est_eqns_mean, raw_est_eqns_var])

    # Group by / sum over user_id ===
    unique_user_ids = np.unique( user_ids )
    n_unique = len(unique_user_ids)
    est_eqns_list = []
    for idx in unique_user_ids:
        user_est_eqn = np.sum( raw_est_eqns[ user_ids == idx ], axis=0 )
        est_eqns_list.append(user_est_eqn)
    
    # Subtract terms ===
    mean_dim = len(post_mean)
    est_eqns = np.vstack(est_eqns_list)
    
    L2_term_mean = np.matmul( np.linalg.inv(prior_var), post_mean - prior_mean ) / n_unique
    est_eqns[:,:mean_dim] = est_eqns[:,:mean_dim] - L2_term_mean
    
    L2_term_var_square = np.linalg.inv(prior_var) / n_unique - post_V
    est_eqns[:,mean_dim:] = est_eqns[:,mean_dim:] + L2_term_var_square.flatten()

    V_esteqn = est_eqns[:,mean_dim:]
    V_esteqn_square = V_esteqn.reshape(n_unique, mean_dim, mean_dim)
    triu_idx = np.triu_indices(mean_dim)
    #np.arange(0, n_unique), triu_idx[0], triu_idx[1]
    V_esteqn_select = np.vstack([V_esteqn_square[i][triu_idx] for i in range(n_unique)])
    
    final_est_eqn = np.hstack([ est_eqns[:,:mean_dim], V_esteqn_select ])
    
    # Hessian term ===
    hessian_mean = np.einsum( 'ij,ik->jk', design_avail, W_design ) + np.linalg.inv( prior_var )
    hessian = np.eye( final_est_eqn.shape[1] )*n_unique
    hessian[:mean_dim,:mean_dim] = hessian_mean
    
    inv_hessian = np.linalg.inv( hessian )
    normalized_inv_hessian = inv_hessian*n_unique

    return {
            "est_eqns" : final_est_eqn,
            "inv_hessian" : inv_hessian,
            "normalized_inv_hessian" : normalized_inv_hessian,
            "user_ids": unique_user_ids,
            }
