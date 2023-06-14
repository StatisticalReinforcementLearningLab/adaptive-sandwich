import numpy as np
import scipy

def output_variance_pieces(study_df, study_RLalg, args, verbose=False):

    all_policy_t = [tmp_dict['policy_last_t'] for tmp_dict in study_RLalg.all_policies]
    all_user_ids = np.unique( study_df['user_id'].to_numpy() )
    out_dict = {}

    # loop over policy updates
    for policy_num, curr_policy_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue

        out_dict[policy_num] = {}

        # Policy Parameters (solution to estimating equation) ################
        if args.RL_alg == "sigmoid_LS":
            est_params = curr_policy_dict['beta_est']
        elif args.RL_alg == "posterior_sampling":

            post_V = curr_policy_dict['post_V']
            triu_idx = np.triu_indices(post_V.shape[0])
            post_V_params = post_V[triu_idx]

            est_params = np.hstack( [ curr_policy_dict['post_mean'], post_V_params ] )

        out_dict[policy_num]['beta_est'] = est_params

        # Estimating Equation and Inverse Hessian (Bread) ####################
        data_sofar = study_df[ study_df['policy_last_t'] < policy_last_t ]
            # data_sofar = all data used to estimate the policy at time policy_last_t

        est_eqns_dict = study_RLalg.get_est_eqns_full(data_sofar=data_sofar,
                                                 curr_policy_dict=curr_policy_dict,
                                                 all_user_ids=all_user_ids)
            # we require all RL algorithms to have a function `get_est_eqns` that
                # forms the estimating equation for policy parameters
        
        out_dict[policy_num]['est_eqns'] = est_eqns_dict['est_eqns']
        out_dict[policy_num]['est_eqns_HC3'] = est_eqns_dict['est_eqns_HC3']
        out_dict[policy_num]['est_eqns_user_ids'] = est_eqns_dict['all_user_ids']
        out_dict[policy_num]['normalized_hessian'] = est_eqns_dict['normalized_hessian']

        # Pi Gradient ########################################################
        curr_timestep_data = study_df[ study_df['policy_last_t'] == policy_last_t ]
            # curr_timestep_data = all data collected using the policy at time policy_last_t

        weighted_pi_grad = study_RLalg.get_pi_gradients(curr_timestep_data,
                                                        curr_policy_dict, verbose)
            # we require all RL algorithms to have a function `get_pi_gradients` that
                # computes the weighted \pi gradients

        pi_user_ids = curr_timestep_data['user_id'].to_numpy()
        unique_pi_ids = np.unique(pi_user_ids)
        user_pi_grads = []
        for idx in all_user_ids:
            if idx in pi_user_ids:
                tmp_grad = np.sum( weighted_pi_grad[ pi_user_ids == idx ], axis=0 )
            else:
                tmp_grad = np.zeros( weighted_pi_grad.shape[1] )
            user_pi_grads.append( tmp_grad )

        user_pi_grads = np.vstack(user_pi_grads)

        out_dict[policy_num]['pi_grads'] = user_pi_grads
        out_dict[policy_num]['pi_grads_user_ids'] = unique_pi_ids

    return out_dict




###############################################################
# Adaptive Sandwich Variance Estimator OLD ########################
###############################################################


def get_adaptive_sandwich(all_est_eqn_dict, args, Teval, alg_correction=""):
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
    #all_est_eqns.append( all_est_eqn_dict[Teval]['est_eqns'] )
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

