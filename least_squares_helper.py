import numpy as np
import numpy_indexed as npi

from helper_functions import var2suffvec, suffvec2var


def fit_WLS(design, outcome, prior_dict=None, weights=None):
    # TODO check dimensions of inputs
    if weights is None:
        numerator = np.sum(design * np.expand_dims(outcome, 1), 0)
        matrix = np.einsum("ij,ik->jk", design, design)
    else:
        numerator = np.sum(
            np.expand_dims(weights, 1) * design * np.expand_dims(outcome, 1), 0
        )
        matrix = np.einsum("ij,ik->jk", design, np.expand_dims(weights, 1) * design)

    if prior_dict is None:
        inv_matrix = np.linalg.inv(matrix)
        est_LS = np.matmul(inv_matrix, numerator)
    else:
        prior_var = prior_dict["prior_var"]
        prior_mean = prior_dict["prior_mean"]
        prior_var_inv = np.linalg.inv(prior_var)

        inv_matrix = np.linalg.inv(matrix + prior_var_inv)
        num_adjust = np.matmul(prior_var_inv, prior_mean)
        est_LS = np.matmul(inv_matrix, numerator + num_adjust)

    return est_LS


def get_est_eqn_LS(
    outcome_vec,
    design,
    present_user_ids,
    est_param,
    avail_vec,
    all_user_ids,
    prior_dict=None,
    correction="HC3",
    weights=None,
    reconstruct_check=False,
    RL_alg=None,
    intercept_val=None,
    light=False,
):
    """
    Small Sample Correction Variants: "HC3", "CR3VE", "CR2VE"
    Our original paper used "HC3" adjustment
    """
    if prior_dict is not None:
        # TODO: Is this a bug?
        est_param_raw = est_param
        state_dim = prior_dict["state_dim"]
        est_param = est_param_raw[:state_dim]
        V_param_raw = est_param_raw[state_dim:]

    # Get inverse hessian (bread)
    design_avail = design * avail_vec.reshape(-1, 1)
    if weights is None:
        W_design = design_avail
    else:
        W_design = design_avail * np.expand_dims(weights, 1)

    n_unique = len(all_user_ids)
    if reconstruct_check:
        # Check that we can reconstruct the est_param
        check_est_params = fit_WLS(design_avail, outcome_vec, prior_dict)
        try:
            assert np.all(np.isclose(check_est_params, est_param))
        except Exception:
            import ipdb

            ipdb.set_trace()

        if prior_dict is not None:
            XXouter = np.einsum("ij,ik->ijk", design_avail, design_avail)
            XXouter_flat = XXouter.reshape((XXouter.shape[0], -1))

            V_matrix = suffvec2var(RL_alg, V_param_raw, intercept_val=intercept_val)
            suffvec_idx = var2suffvec(RL_alg, V_matrix, return_idx=True)
            V_param_reconstruct = (
                np.sum(XXouter_flat[:, suffvec_idx], axis=0) / n_unique
            )

            try:
                assert np.all(np.isclose(V_param_reconstruct, V_param_raw))
            except Exception:
                import ipdb

                ipdb.set_trace()

    # Get residuals
    lin_prod = design * est_param
    residuals = outcome_vec - np.sum(lin_prod, axis=1)
    residuals = residuals * avail_vec

    unique_user_ids = set(present_user_ids)
    n_unique = len(unique_user_ids)
    n_users = len(all_user_ids)
    # TODO: Appears setting light and a correction would break things because
    # hessian variable won't exist
    if not light:
        hessian = -np.einsum("ij,ik->jk", design_avail, W_design)
        normalized_hessian = hessian / n_unique  # n_users

    # Get estimating equations
    if correction == "HC3":
        inv_hessian = np.linalg.inv(hessian)

        # Adjustment from "Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model"
        hvals = np.sum(np.einsum("ik,jk->ij", design, inv_hessian) * W_design, 1)
        residuals_adjust = residuals / (1 - hvals)

        # TODO: remove?
        # Degrees of freedom adjustment: pg 24 of https://cran.r-project.org/web/packages/sandwich/sandwich.pdf
        # residuals_adjust *= np.sqrt( (n_users-1) / (n_users - len(est_param)) )
        # residuals_adjust *= np.sqrt( n_users / (n_users - len(est_param)) )

    elif correction in {"CR3VE", "CR2VE"}:
        # TODO: comments seem mixed up
        # HC3 bias adjustment based on the following resource:
        # http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf (pg25)
        inv_hessian = np.linalg.inv(hessian)
        residuals_adjust = np.zeros(residuals.shape)

        for uid in unique_user_ids:
            # Mask for available times for the user
            bool_mask = np.logical_and(present_user_ids == uid, avail_vec)
            bool_sum = np.sum(bool_mask)

            if bool_sum == 0:
                continue
            design_tmp = design_avail[bool_mask]
            W_design_tmp = W_design[bool_mask]

            Hmatrix = np.matmul(np.matmul(design_tmp, inv_hessian), W_design_tmp.T)
            sqrtInvHdiff = np.linalg.inv(np.eye(bool_sum) - Hmatrix)
            residuals_adjust_tmp = np.matmul(sqrtInvHdiff, residuals[bool_mask])

            # Gets indices of mask and puts adjusted residual values there
            np.put(residuals_adjust, np.nonzero(bool_mask)[0], residuals_adjust_tmp)

        if correction == "CR3VE":
            # Cluster correction
            # http://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015_February.pdf (pg25)
            nclusters = len(unique_user_ids)
            residuals_adjust = residuals * np.sqrt((nclusters - 1) / nclusters)
    else:
        residuals_adjust = residuals

    raw_est_eqns = residuals_adjust.reshape(-1, 1) * W_design

    if prior_dict is not None:
        ############ Adjustments if there is a prior
        # Posterior mean adjustment
        inv_prior_var = np.linalg.inv(prior_dict["prior_var"])
        post_mean_adj = (
            np.matmul(inv_prior_var, prior_dict["prior_mean"] - est_param) / n_users
        )

        # V matrix adjustment
        V_adj = -V_param_raw

        # Combined
        post_adj = np.concatenate([post_mean_adj, V_adj])

        ########### Form variance estimating equation
        V_matrix = suffvec2var(RL_alg, V_param_raw, intercept_val=intercept_val)
        XXouter = np.einsum("ij,ik->ijk", design_avail, design_avail)
        XXouter_flat = XXouter.reshape((XXouter.shape[0], -1))

        suffvec_idx = var2suffvec(RL_alg, V_matrix, return_idx=True)
        V_param_est_eqn = XXouter_flat[:, suffvec_idx]

        ########### Concatenate estimating equations
        post_mean_est_eqn = raw_est_eqns / prior_dict["noise_var"]
        raw_est_eqns = np.concatenate([post_mean_est_eqn, V_param_est_eqn], axis=1)

    # Group by / sum over user_id
    user_ids_grouped, est_eqn_grouped = npi.group_by(present_user_ids).sum(raw_est_eqns)
    if prior_dict is not None:
        est_eqn_grouped = est_eqn_grouped + post_adj

    add_users = set(all_user_ids) - set(present_user_ids)
    if len(add_users) > 0:
        all_user_ids_grouped = np.concatenate(
            [[x for x in add_users], user_ids_grouped]
        )
        zeros = np.zeros((len(add_users), est_eqn_grouped.shape[1]))
        all_est_eqn_grouped = np.concatenate([zeros, est_eqn_grouped], axis=0)

        sort_idx = np.argsort(all_user_ids_grouped)
        user_ids_grouped = all_user_ids_grouped[sort_idx]
        est_eqn_grouped = all_est_eqn_grouped[sort_idx]

    if reconstruct_check:
        try:
            if correction == "":
                est_eqn_error = np.absolute(np.array(est_eqn_grouped).mean(axis=0))
                assert np.all(np.isclose(est_eqn_error, 0))
            else:
                assert np.all(
                    np.absolute(np.array(est_eqn_grouped).mean(axis=0)) < 0.01
                )
        except Exception:
            import ipdb

            ipdb.set_trace()

    if light:
        return_dict = {
            "est_eqns": est_eqn_grouped,
        }
    else:
        return_dict = {
            "est_eqns": est_eqn_grouped,
            "hessian": hessian,
            "normalized_hessian": normalized_hessian,
            "present_user_ids": present_user_ids,
            "all_user_ids": user_ids_grouped,
            "estimator": est_param,
        }
    return return_dict
