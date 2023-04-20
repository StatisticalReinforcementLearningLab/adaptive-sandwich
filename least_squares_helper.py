import numpy as np
import pandas as pd
import scipy.special
from scipy.linalg import sqrtm


def get_utri(matrix):
    triu_idx = np.triu_indices(matrix.shape[0])
    return matrix[triu_idx]


def symmetric_fill_utri(values, mdim):
    matrix = np.zeros((mdim, mdim))
    triu_idx = np.triu_indices(mdim)
    matrix[triu_idx] = values
    return matrix + matrix.T - np.diag( np.diag(matrix) )


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



def get_est_eqn_LS(outcome_vec, design, present_user_ids, est_param, avail_vec, 
                   all_user_ids, prior_dict=None, correction="HC3", weights=None):
    """
    Small Sample Correction Variants: "HC3", "CR3VE", "CR2VE"
    Our original paper used "HC3" adjustment
    """
    if prior_dict is not None:
        est_param_raw = est_param
        state_dim = prior_dict["state_dim"]
        est_param = est_param_raw[:state_dim]
        V_param_raw = est_param_raw[state_dim:]

    # Get residuals
    lin_prod = design * est_param
    residuals = outcome_vec - np.sum( lin_prod, axis=1 )
    residuals = residuals*avail_vec

    # Get inverse hessian (bread)
    design_avail = design * avail_vec.reshape(-1,1)
    if weights is None:
        W_design = design_avail
    else:
        W_design = design_avail * np.expand_dims(weights, 1)

    unique_user_ids = np.unique( present_user_ids )
    n_unique = len(unique_user_ids)
    hessian = - np.einsum( 'ij,ik->jk', design_avail, W_design )
    normalized_hessian = hessian / n_unique
    #normalized_inv_hessian = inv_hessian*n_unique

    # Get estimating equations
    if correction == "HC3":
        inv_hessian = np.linalg.inv( hessian )

        # Adjustment from "Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model"
        hvals = np.sum(np.einsum('ik,jk->ij', design, inv_hessian)*W_design, 1)
        residuals_adjust = residuals / (1-hvals)
        
        # Degrees of freedom adjustment: pg 24 of https://cran.r-project.org/web/packages/sandwich/sandwich.pdf
        #residuals_adjust *= np.sqrt( (n_unique-1) / (n_unique - len(est_param)) )
        #residuals_adjust *= np.sqrt( n_unique / (n_unique - len(est_param)) )
        
    elif correction in ["CR3VE", "CR2VE"]:
        # HC3 bias adjustment based on the following resource:
        # http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf (pg25)
        inv_hessian = np.linalg.inv( hessian )
        residuals_adjust = np.zeros(residuals.shape)

        for uid in unique_user_ids:
            # Mask for available times for the user
            bool_mask = np.logical_and(present_user_ids == uid, avail_vec)
            bool_sum = np.sum(bool_mask)

            if bool_sum == 0:
                continue
            design_tmp = design_avail[bool_mask]
            W_design_tmp = W_design[bool_mask]

            Hmatrix = np.matmul( np.matmul( design_tmp, inv_hessian ), W_design_tmp.T )
            sqrtInvHdiff = np.linalg.inv( np.eye(bool_sum)-Hmatrix )
            residuals_adjust_tmp = np.matmul( sqrtInvHdiff, residuals[bool_mask] )

            # Gets indices of mask and puts adjusted residual values there
            np.put(residuals_adjust, np.nonzero(bool_mask)[0], residuals_adjust_tmp)
        
        if correction == "CR3VE":
            # Cluster correction
            # http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf (pg25)
            nclusters = len( unique_user_ids )
            residuals_adjust = residuals * np.sqrt( (nclusters-1) / nclusters)
    else:
        residuals_adjust = residuals
    
    raw_est_eqns = residuals_adjust.reshape(-1,1) * W_design
    
    # Adjustments if there is a prior
    if prior_dict is not None:
        n_tmp = len(present_user_ids)
        inv_var = np.linalg.inv(prior_dict["prior_var"])
        
        # Posterior mean estimating equation
        post_mean_adj = -np.matmul(inv_var, est_param-prior_dict["prior_mean"]) / n_tmp

        # Posterior V estimating equation
        V_param_matrix = symmetric_fill_utri(V_param_raw, state_dim)
        post_V_adj_matrix = inv_var / n_tmp - V_param_matrix
        post_V_adj = get_utri(post_V_adj_matrix)

        # Combined
        post_adj = np.concatenate([post_mean_adj, post_V_adj])

        # Add raw estimating equations to have variance component
        XXouter = np.einsum( 'ij,ik->ijk', design_avail, design_avail)
        XXouter_flat = XXouter.reshape((XXouter.shape[0], -1))
        
        triu_idx = np.triu_indices(state_dim)
        matrix_idx = np.arange(0,state_dim**2).reshape((state_dim, state_dim))
        triu_idx_cols = matrix_idx[triu_idx]

        post_V_raw_est_eqn = XXouter_flat[:,triu_idx_cols]
        raw_est_eqns = np.concatenate( [raw_est_eqns, post_V_raw_est_eqn], axis=1 )
        raw_est_eqns = raw_est_eqns / prior_dict["noise_var"]

    
    # Group by / sum over user_id
    est_eqns = []
    for idx in all_user_ids:
        if idx in present_user_ids:
            user_est_eqn = np.sum( raw_est_eqns[ present_user_ids == idx ], axis=0 )
            if prior_dict is not None:
                user_est_eqn = user_est_eqn + post_adj
        else:
            user_est_eqn = np.zeros( raw_est_eqns.shape[1] )
        est_eqns.append(user_est_eqn)


    return_dict = {
            "est_eqns" : np.vstack(est_eqns),
            "hessian" : hessian,
            "normalized_hessian" : normalized_hessian,
            #"inv_hessian" : inv_hessian,
            #"normalized_inv_hessian" : normalized_inv_hessian,
            "present_user_ids": unique_user_ids,
            "all_user_ids": all_user_ids,
            "estimator": est_param, 
            }
    return return_dict


"""
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
"""
