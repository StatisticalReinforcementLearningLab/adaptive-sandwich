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
import numpy_indexed as npi

from least_squares_helper import get_est_eqn_LS, fit_WLS
from debug_helper import get_adaptive_sandwich


def c_vec2string(c_vec):
    return np.array2string(c_vec)[1:-1].replace(" ", ",")


###############################################################
# Initialize Hyperparameters ##################################
###############################################################

parser = argparse.ArgumentParser(description="Analyze data")
parser.add_argument(
    "--dataset_type",
    type=str,
    default="synthetic",
    choices=["heartsteps", "synthetic", "oralytics"],
)
parser.add_argument("--verbose", type=int, default=0, help="Prints helpful info")
parser.add_argument(
    "--heartsteps_mode",
    default="medium",
    choices=["evalSim", "realistic", "medium", "easy"],
    help="Sets default parameter values accordingly",
)
parser.add_argument(
    "--inference_mode",
    default="value",
    choices=["value", "model"],
    help="Sets default parameter values accordingly",
)
parser.add_argument(
    "--synthetic_mode",
    type=str,
    default="delayed_effects",
    help="File name of synthetic env params",
)
parser.add_argument(
    "--RL_alg",
    default="sigmoid_LS",
    choices=["fixed_randomization", "sigmoid_LS", "posterior_sampling"],
    help="RL algorithm used to select actions",
)
parser.add_argument(
    "--err_corr",
    default="time_corr",
    choices=["time_corr", "independent"],
    help="Noise error correlation structure",
)
parser.add_argument(
    "--N", type=int, default=10, help="Number of Monte Carlo repetitions"
)
parser.add_argument("--n", type=int, default=90, help="Total number of users")
parser.add_argument(
    "--eta", type=float, default=0, help="Regularization of alg bread matrix"
)
parser.add_argument(
    "--upper_clip",
    type=float,
    default=0.9,
    help="Upper action selection probability constraint",
)
parser.add_argument(
    "--lower_clip",
    type=float,
    default=0.1,
    help="Lower action selection probability constraint",
)
parser.add_argument(
    "--fixed_action_prob",
    type=float,
    default=0.5,
    help="Used if not using learning alg to select actions",
)
parser.add_argument(
    "--min_users", type=int, default=25, help="Min number of users needed to update alg"
)
parser.add_argument(
    "--decisions_between_updates",
    type=int,
    default=1,
    help="Number of decision times beween algorithm updates",
)
parser.add_argument(
    "--save_dir", type=str, default=".", help="Directory to save all results in"
)
parser.add_argument("--debug", type=int, default=0, help="Debug mode")
parser.add_argument(
    "--redo_analyses", type=int, default=0, help="Redo any saved analyses"
)
parser.add_argument("--steepness", type=float, default=10, help="Allocation steepness")
parser.add_argument(
    "--action_centering",
    type=int,
    default=0,
    help="Whether posterior sampling algorithm uses action centering",
)
parser.add_argument(
    "--alg_state_feats",
    type=str,
    default="intercept",
    help="Comma separated list of algorithm state features",
)
tmp_args = parser.parse_known_args()[0]

print(tmp_args)

if tmp_args.dataset_type == "heartsteps":
    raise ValueError("Not implemented")
elif tmp_args.dataset_type == "synthetic":
    arg_dict = {"T": 2, "recruit_n": tmp_args.n, "recruit_t": 1, "allocation_sigma": 1}
    dval_alg = 2 * len(tmp_args.alg_state_feats.split(","))

    if tmp_args.inference_mode == "value":
        # Inference Objective
        feature_names = ["intercept"]
        outcome_name = "reward"
        c_vec_list = [np.array([1])]
        dval = 1

        cvec2name = {
            c_vec2string(np.array([1])): "Entire Vector",
        }
    elif tmp_args.inference_mode == "model_simple":
        # Inference Objective
        feature_names = ["intercept", "action"]
        outcome_name = "reward"
        c_vec_list = [np.array([1, 1]), np.array([0, 1])]
        dval = 2

        cvec2name = {
            c_vec2string(np.array([1, 1])): "Entire Vector",
            c_vec2string(np.array([0, 1])): "Treatment Effect",
        }
    elif tmp_args.inference_mode == "model":
        # Inference Objective
        feature_names = ["intercept", "past_reward", "action"]
        outcome_name = "reward"
        c_vec_list = [np.array([1, 1, 1]), np.array([0, 0, 1])]
        dval = 3

        cvec2name = {
            c_vec2string(np.array([1, 1, 1])): "Entire Vector",
            c_vec2string(np.array([0, 0, 1])): "Treatment Effect",
        }
    else:
        raise ValueError("invalid inference mode")

    stat2name = {
        "hotelling": "Hotelling's t-squared Statistic",
    }
elif tmp_args.dataset_type == "oralytics":
    arg_dict = {
        "T": 50,
        "recruit_n": tmp_args.n,
        "recruit_t": 1,
        "allocation_sigma": 5.7,
    }
    # allocation_sigma: 163 (truncated brush times); 5.7 (square-root of truncatred brush times)

    # Inference Objective
    feature_names = ["intercept", "time_of_day", "prior_day_brush", "action"]
    # feature_names = ['intercept', 'time_of_day', 'weekend', 'prior_day_brush', 'action']
    outcome_name = "reward"
    stat2name = {
        "hotelling": "Hotelling's t-squared Statistic",
    }

    c_vec_list = [
        np.array([1, 1, 1, 1]),
        np.array([0, 0, 0, 1]),
        np.array([0, 0, 1, 0]),
    ]
    dval = 4

    cvec2name = {
        c_vec2string(np.array([1, 1, 1, 1])): "Entire Vector",
        c_vec2string(np.array([0, 0, 1, 0])): "prior_day_brush",
        c_vec2string(np.array([0, 0, 0, 1])): "Margin Vector",
    }

parser.add_argument(
    "--T",
    type=int,
    default=arg_dict["T"],
    help="Total number of decision times per user",
)
parser.add_argument(
    "--recruit_n",
    type=int,
    default=arg_dict["recruit_n"],
    help="Number of users recruited on each recruitment times",
)
parser.add_argument(
    "--recruit_t",
    type=int,
    default=arg_dict["recruit_t"],
    help="Number of updates between recruitment times",
)
parser.add_argument(
    "--allocation_sigma",
    type=float,
    default=arg_dict["allocation_sigma"],
    help="Sigma used in allocation of algorithm",
)

args = parser.parse_args()
print(vars(args))


###############################################################
# Load Data ###################################################
###############################################################


def load_data(folder_path):
    """
    Loads data

    Input
    - Experiment folder path

    Output
    - study data pandas dataframe
    - RL algorithm object
    - RL algorithm saved output (this is only to ensure compatibility with old code)
    """
    # study_df = pd.read_csv( os.path.join(folder_path, "data.csv") )
    with open("{}/study_df.pkl".format(folder_path), "rb") as f:
        study_df = pkl.load(f)

    if args.dataset_type == "synthetic":
        study_df["action_past_reward"] = study_df["past_reward"] * study_df["action"]

    if args.RL_alg == "fixed_randomization":
        return study_df

    with open("{}/study_RLalg.pkl".format(folder_path), "rb") as f:
        study_RLalg = pkl.load(f)

    # OLD STUFF
    if args.RL_alg == "sigmoid_LS":
        with open(os.path.join(folder_path, "out_dict.pkl"), "rb") as file:
            alg_out_dict = pkl.load(file)
    else:
        alg_out_dict = None

    return study_df, study_RLalg, alg_out_dict


###############################################################
# Form Estimator ##############################################
###############################################################


def form_LS_estimator(study_df):
    """
    Forms least squares estimator for inference

    Input
    - Study data pandas dataframe

    Output
    - Least squares estimator
    """

    if args.dataset_type == "heartsteps":
        avail_vec = study_df["availability"].to_numpy()
    else:
        avail_vec = np.ones(study_df["user_id"].shape)
    design = study_df[feature_names].to_numpy() * avail_vec.reshape(-1, 1)
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec

    LS_estimator = fit_WLS(design, outcome_vec)
    return LS_estimator


def form_LS_est_eqn(est_param, study_df, all_user_id, correction="HC3", check=False):
    """
    Forms estimating equations for the least squares estimator used for inference

    Input
    - `est_param`: estimator of the inferential target (thetahat)
    - `study_df`: study data pandas dataframe
    - `all_user_id`: Unique set of all user ids in the study
    - `correction`: Small sample correction (default is HC3; other options are none [use an empty string ''], CR3VE, CR2VE)

    Output
    - Least squares estimator estimating equations (matrix of dimension num_users by dim_theta)
    """

    if args.dataset_type == "heartsteps":
        avail_vec = study_df["availability"].to_numpy()
    else:
        avail_vec = np.ones(study_df["user_id"].shape)
    design = study_df[feature_names].to_numpy() * avail_vec.reshape(-1, 1)
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec

    present_user_ids = study_df["user_id"].to_numpy()
    LS_dict = get_est_eqn_LS(
        outcome_vec,
        design,
        present_user_ids,
        est_param,
        avail_vec,
        all_user_id,
        correction=correction,
        reconstruct_check=check,
    )

    return LS_dict["est_eqns"]


###############################################################
# Sandwich Variance Estimator #################################
###############################################################


def get_sandwich_var(est_eqns, normalized_hessian, LS_estimator):
    """
    Forms standard sandwich variance estimator for inference (thetahat)

    Input:
    - `est_eqns`: Estimating equation matrix (matrix of dimension num_users by dim_theta)
    - `normalized_hessian`: (Hessian matrix of size dim_theta by dim_theta that is normalized by num_users)
    - `LS_estimator`: Least squares estimator (vector)

    Output:
    - Sandwich variance estimator matrix (size dim_theta by dim_theta)
    """
    n_unique = est_eqns.shape[0]

    meat = np.einsum("ij,ik->jk", est_eqns, est_eqns)
    meat = meat / n_unique

    # degrees of freedom adjustment
    meat = meat * (n_unique - 1) / (n_unique - len(LS_estimator))
    est_val_dict["meat"] = meat

    inv_hessian = np.linalg.inv(normalized_hessian)
    sandwich_var = np.matmul(np.matmul(inv_hessian, meat), inv_hessian)
    sandwich_var = sandwich_var / n_unique

    return sandwich_var


###############################################################
# Adaptive Sandwich Variance Estimator ########################
###############################################################


# @profile
def get_stacked_estimating_function(
    all_estimators,
    update2esteqn,
    policy2collected,
    info_dict,
    return_full=False,
    alg_correction="",
    theta_correction="",
    check=False,
):
    """
    Form stacked estimating function for both algorithm statistics (betahats) and estimator of interest (thetahat).
    This is used to form the adaptive sandwich variance.

    Inputs:
    - `all_estimators`: concatenated vector of algorithm statistics (betahats) and estimator of interest (thetahat) of dimension dim_beta*num_updates + dim_theta
    - `update2esteqn`: dictionary where the keys are update numbers (starts with 1)
            and the values are dictionaries with the data used in that update, which will be used as the data_dict argument when calling the function study_RLalg.get_est_eqns
    - `policy2collected`: dictionary where keys are policy numbers (policy 1 is prespecified policy, policy 2 is policy used after first update; total number of policies is number of updates plus 1; do not need key for first policy)
        value are dictionaries that will be used as the collected_data_dict argument when calling the function study_RLalg.get_weights

    - `info_dict`: Dictionary with certain algorithm info that doesn't change with updates. It will be used as the `info_dict` argument when calling the function `study_RLalg.get_est_eqns`. This dictionary should include:
        - `theta_dim`: dimension of inferential target
        - `beta_dim`: dimension of algorithm statistics
        - `all_user_id`: unique set of all user ids in the study
        - `study_df`: study data pandas dataframe
        - `study_RLalg`: RL algorithm object used to collect data
        - `prior_dict`: dictionary of algorithm information including prior (only for posterior sampling algorithm)
    - `return_full`: Indicator of whether return dictionary with more information; if False just return estimating equation
            (dimension num_theta by square matrix of dimension dim_beta*num_updates + dim_theta)
    - `alg_correction`: Small sample correction for algorithm statistic estimating equation (default is none; other options are HC3, CR3VE, CR2VE)
    - `theta_correction`: Small sample correction for theta estimating equation (default is none; other options are HC3, CR3VE, CR2VE)
    - `check`: If true, then check that estimating function sums to near zero and check reconstruction of action selection probabilities

    Outputs:
    - if return_full is true, return a dictionary with
        - `ave_est_eqn`: numpy vector of estimating equation averaged across uses of dimension dim_beta*num_updates + dim_theta
        - `all_est_eqn`: numpy array of estimating equation of dimension (num_users by dim_beta*num_updates + dim_theta) - grouped by user, summed over time
    - if return_full is false return a numpy array `ave_est_eqn`
    """

    theta_dim = info_dict["theta_dim"]
    beta_dim = info_dict["beta_dim"]
    all_user_id = info_dict["all_user_id"]

    thetahat = all_estimators[-theta_dim:]
    num_updates = int(len(all_estimators[:-theta_dim]) / beta_dim)
    est_list = np.split(all_estimators[:-theta_dim], num_updates)
    est_list.append(thetahat)

    all_weights = [np.ones(len(all_user_id))]
    all_est_eqns = []
    all_est_eqns_full = []

    if args.action_centering:
        action1prob = update2esteqn[1]["action1prob"]

    for update_num, update_dict in enumerate(study_RLalg.all_policies):
        policy_last_t = update_dict["policy_last_t"]
        if update_num in [0, len(study_RLalg.all_policies)]:
            continue

        prev_beta_est = est_list[update_num - 2]
        beta_est = est_list[update_num - 1]

        # Form estimating equations ################
        if args.action_centering:
            if update_num > 1:
                if check:  # check reconstruction of action selection probabilities
                    og_action1prob = update2esteqn[update_num]["og_action1prob"]
                    assert np.all(action1prob == og_action1prob[: len(action1prob)])
                update2esteqn[update_num]["action1prob"][
                    : len(action1prob)
                ] = action1prob

        # we require all RL algorithms to have a function `get_est_eqns` that
        # forms the estimating equation for policy parameters
        est_eqns_dict = study_RLalg.get_est_eqns(
            beta_est=beta_est,
            data_dict=update2esteqn[update_num],
            info_dict=info_dict,
            correction=alg_correction,
            check=check,
            light=True,
        )

        # Check estimating equation sums to zero
        if check:
            try:
                tmp_ave_est_eqn = np.sum(est_eqns_dict["est_eqns"], axis=0) / len(
                    all_user_id
                )
                # assert np.all( np.absolute( tmp_ave_est_eqn ) < 0.001 )
                assert np.all(np.absolute(tmp_ave_est_eqn) < 0.001)
            except:
                print("Estimating equation sum to zero check failed")
                import ipdb

                ipdb.set_trace()

        # Multiply by weights
        prev_weights_prod = np.expand_dims(np.prod(all_weights, axis=0), 1)
        weighted_est_eqn = prev_weights_prod * est_eqns_dict["est_eqns"]
        all_est_eqns.append(np.sum(weighted_est_eqn, axis=0) / len(all_user_id))
        all_est_eqns_full.append(weighted_est_eqn)

        # Form weights ################
        if update_num != len(study_RLalg.all_policies) - 1:
            policy_num = update_num + 1
            # we consider policy formed after update, and data collected with this policy
            if args.action_centering:
                user_pi_weights, action1prob = study_RLalg.get_weights(
                    beta_est,
                    collected_data_dict=policy2collected[policy_num],
                    return_probs=True,
                )
            else:
                user_pi_weights = study_RLalg.get_weights(
                    beta_est, collected_data_dict=policy2collected[policy_num]
                )
            all_weights.append(user_pi_weights)

            # Check that reproduced the action selection probabilities correctly
            if check:
                try:
                    assert np.all(np.around(user_pi_weights, 5) == 1)
                except:
                    print("Reproducing action selection probabilities check failed")
                    import ipdb

                    ipdb.set_trace()

    # Estimating Equation for theta ######################
    theta_est_eqn = form_LS_est_eqn(
        thetahat,
        info_dict["study_df"],
        all_user_id,
        correction=theta_correction,
        check=check,
    )

    # Multiply by weights
    prev_weights_prod = np.prod(all_weights, axis=0)
    prev_weights_prod = np.expand_dims(prev_weights_prod, 1)
    weighted_est_eqn = prev_weights_prod * theta_est_eqn
    all_est_eqns.append(np.mean(weighted_est_eqn, axis=0))
    all_est_eqns_full.append(weighted_est_eqn)

    if return_full:
        return {
            "ave_est_eqn": np.concatenate(all_est_eqns),
            "all_est_eqn": all_est_eqns_full,
        }
    return np.concatenate(all_est_eqns)


# @profile
def get_adaptive_sandwich_new(
    all_est_eqn_dict, study_RLalg, study_df, alg_correction="", theta_correction=""
):
    """
    Form adaptive sandwich variance estimator

    Inputs:
    - `all_est_eqn_dict`: dictionary with estimating equation information
        - `estimator`
        - ``
    - `study_RLalg`: RL algorithm object
    - `study_df`: Study data pandas dataframe
    - `alg_correction`: Small sample correction for algorithm statistic estimating equation (default is none; other options are HC3, CR3VE, CR2VE)
    - `theta_correction`: Small sample correction for theta estimating equation (default is none; other options are HC3, CR3VE, CR2VE)

    Outputs:
    - dictionary with the following
        "stacked_meat": full stacked meat matrix (square matrix of dimension dim_beta*num_updates + dim_theta)
        "stacked_hessian": stacked hessian matrix (square matrix of dimension dim_beta*num_updates + dim_theta)
        "stacked_sandwich": full stacked adaptive sandwich variance estimator (square matrix of dimension dim_beta*num_updates + dim_theta)
        "adaptive_sandwich": adaptive sandwich variance estimator (dim_theta by dim_theta)
    """

    alg_stat_dict = study_RLalg.prep_algdata()
    alg_estimators = alg_stat_dict["alg_estimators"]
    update2esteqn = alg_stat_dict["update2esteqn"]
    policy2collected = alg_stat_dict["policy2collected"]
    info_dict = alg_stat_dict["info_dict"]

    # Form estimators
    thetahat = form_LS_estimator(study_df)
    all_estimators = np.concatenate([alg_estimators, thetahat])
    theta_dim = len(thetahat)

    # Add more info to info_dict
    info_dict["theta_dim"] = theta_dim
    info_dict["study_df"] = study_df

    # Form Stacked Meat Estimator ##########################
    stacked_est_dict = get_stacked_estimating_function(
        all_estimators,
        update2esteqn,
        policy2collected,
        info_dict,
        return_full=True,
        alg_correction=alg_correction,
        theta_correction=theta_correction,
        check=True,
    )

    cat_est_eqn = np.hstack(stacked_est_dict["all_est_eqn"])
    stacked_raw_meat = np.einsum("ij,ik->jk", cat_est_eqn, cat_est_eqn)
    n_unique = cat_est_eqn.shape[0]

    stacked_meat = stacked_raw_meat / n_unique
    stacked_meat = stacked_meat * (n_unique - 1) / (n_unique - theta_dim)

    if args.debug:
        print("Adaptive: stacked meat done")

    # Form Stacked Bread Estimator ##########################
    eps = np.sqrt(np.finfo(float).eps)
    tic = time.perf_counter()
    stacked_hessian = optimize.approx_fprime(
        all_estimators,
        get_stacked_estimating_function,
        eps,
        update2esteqn,
        policy2collected,
        info_dict,
    )
    # np.allclose(adaptive_sandwich_dict_HC3['stacked_hessian'], bread_stacked_old, atol=0.004)
    toc = time.perf_counter()
    print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")

    if args.debug:
        print("Adaptive: stacked bread done")

    # Form Adaptive Sandwich Variance Estimator ###############
    stacked_hessian_inv = np.linalg.inv(stacked_hessian)

    adaptive_sandwich_stacked = np.matmul(
        np.matmul(stacked_hessian_inv, stacked_meat), stacked_hessian_inv.T
    )
    adaptive_sandwich_stacked = adaptive_sandwich_stacked / n_unique

    adaptive_sandwich_theta = adaptive_sandwich_stacked[-theta_dim:, -theta_dim:]

    if args.debug:
        print("Adaptive: function done")

    return_dict = {
        "stacked_meat": stacked_meat,
        "stacked_hessian": stacked_hessian,
        "stacked_sandwich": adaptive_sandwich_stacked,
        "adaptive_sandwich": adaptive_sandwich_theta,
    }
    return return_dict


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


if args.dataset_type == "oralytics":
    exp_str = "{}_alg={}_T={}_n={}_recruitN={}_decisionsBtwnUpdates={}_steep={}_actionC={}".format(
        args.dataset_type,
        args.RL_alg,
        args.T,
        args.n,
        args.recruit_n,
        args.decisions_between_updates,
        args.steepness,
        args.action_centering,
    )
else:
    exp_str = "{}_mode={}_alg={}_T={}_n={}_recruitN={}_decisionsBtwnUpdates={}_steepness={}_algfeats={}_errcorr={}_actionC={}".format(
        args.dataset_type,
        mode,
        args.RL_alg,
        args.T,
        args.n,
        args.recruit_n,
        args.decisions_between_updates,
        args.steepness,
        args.alg_state_feats,
        args.err_corr,
        args.action_centering,
    )

# Load Data
all_folder_path = os.path.join(args.save_dir, "simulated_data/{}".format(exp_str))

# Parameter info
inf_feat_names = [x for x in range(len(feature_names))]

# Policy parameters
folder_path = os.path.join(all_folder_path, "exp={}".format(1))
if args.RL_alg == "fixed_randomization":
    study_df = load_data(folder_path)
else:
    study_df, study_RLalg, alg_out_dict = load_data(folder_path)


all_estimators = init_nan((args.N, dval))
all_sandwich = init_nan((args.N, dval, dval))
all_adaptive_sandwich = init_nan((args.N, dval, dval))
all_adaptive_sandwich_HC3 = init_nan((args.N, dval, dval))
all_adaptive_sandwich_old = init_nan((args.N, dval, dval))

for i in range(1, args.N + 1):
    if i % 100 == 0:
        print("dataset num {}".format(i))

    # Load Data #########################################
    folder_path = os.path.join(all_folder_path, "exp={}".format(i))
    analyses_out_path = "{}/analysis.pkl".format(folder_path)

    if args.redo_analyses:
        pass
    else:
        # do not redo existing analyses
        analysis_already = os.path.isfile(analyses_out_path)
        if analysis_already:
            # Found that this already is a saved analysis file
            with open(analyses_out_path, "rb") as f:
                exp_analysis_dict = pkl.load(f)

            all_estimators[i - 1] = exp_analysis_dict["LS_estimator"]
            all_sandwich[i - 1] = exp_analysis_dict["sandwich_var"]
            all_adaptive_sandwich[i - 1] = exp_analysis_dict["adaptive_sandwich"]
            all_adaptive_sandwich_HC3[i - 1] = exp_analysis_dict[
                "adaptive_sandwich_HC3"
            ]
            if args.RL_alg == "sigmoid_LS":
                all_adaptive_sandwich_old[i - 1] = exp_analysis_dict[
                    "adaptive_sandwich_old"
                ]
            continue

    if args.RL_alg == "fixed_randomization":
        study_df = load_data(folder_path)
    else:
        study_df, study_RLalg, alg_out_dict = load_data(folder_path)

    # Form Estimator #########################################
    LS_estimator = form_LS_estimator(study_df)
    all_estimators[i - 1] = LS_estimator

    if args.debug:
        print("Empirical Done")

    # Form Sandwich Var #######################################
    user_ids = study_df["user_id"].to_numpy()
    unique_user_ids = np.unique(user_ids)
    if args.dataset_type == "heartsteps":
        avail_vec = study_df["availability"].to_numpy()
    else:
        avail_vec = np.ones(study_df["user_id"].shape)
    design = study_df[feature_names].to_numpy()
    outcome_vec = study_df[outcome_name].to_numpy() * avail_vec

    est_val_dict = get_est_eqn_LS(
        outcome_vec,
        design,
        user_ids,
        LS_estimator,
        avail_vec,
        all_user_ids=unique_user_ids,
        correction="",
        weights=None,
    )
    est_val_dict_HC3 = get_est_eqn_LS(
        outcome_vec,
        design,
        user_ids,
        LS_estimator,
        avail_vec,
        all_user_ids=unique_user_ids,
        correction="HC3",
        weights=None,
    )
    est_val_dict["est_eqns_HC3"] = est_val_dict_HC3["est_eqns"]
    sandwich_var = get_sandwich_var(
        est_eqns=est_val_dict["est_eqns"],
        normalized_hessian=est_val_dict["normalized_hessian"],
        LS_estimator=LS_estimator,
    )
    all_sandwich[i - 1] = sandwich_var

    if args.RL_alg == "fixed_randomization":
        continue

    if args.debug:
        print("Sandwich Done")

    # Form Adaptive Sandwich Var ######################################
    num_updates = len(study_RLalg.all_policies) - 1
    adaptive_sandwich_dict = get_adaptive_sandwich_new(
        est_val_dict, study_RLalg, study_df
    )

    # if args.recruit_n != args.n:
    #    raise ValueError("adaptive sandwich variance not implemented for \
    #            incremental recruitment yet")

    adaptive_sandwich_dict_HC3 = get_adaptive_sandwich_new(
        est_val_dict,
        study_RLalg,
        study_df,
        alg_correction="HC3",
        theta_correction="HC3",
    )

    adaptive_sandwich = adaptive_sandwich_dict["adaptive_sandwich"]
    all_adaptive_sandwich[i - 1] = adaptive_sandwich
    all_adaptive_sandwich_HC3[i - 1] = adaptive_sandwich_dict_HC3["adaptive_sandwich"]

    #### OLD STUFF
    if args.RL_alg == "sigmoid_LS":
        num_updates = max(alg_out_dict.keys())
        alg_out_dict[num_updates + 1] = est_val_dict

        (
            adaptive_sandwich_old,
            adaptive_sandwich_full_old,
            eig_dict,
            bread_stacked_old,
            stacked_meat_old,
        ) = get_adaptive_sandwich(alg_out_dict, args, Teval=num_updates + 1)
        all_adaptive_sandwich_old[i - 1] = adaptive_sandwich_old

    ### Some checks on adaptive sandwich varaince

    # 1) Check ``bread'' from sandwich and adaptive sandwich variances
    theta_dim = len(LS_estimator)
    theta_bread = est_val_dict["normalized_hessian"]
    theta_bread_adaptive = adaptive_sandwich_dict["stacked_hessian"][
        -theta_dim:, -theta_dim:
    ]
    assert np.all(np.isclose(theta_bread, theta_bread_adaptive))

    # 2) Check ``meat'' from sandwich and adaptive sandwich variances
    theta_meat = est_val_dict["meat"]
    theta_meat_adaptive = adaptive_sandwich_dict["stacked_meat"][
        -theta_dim:, -theta_dim:
    ]
    assert np.all(np.isclose(theta_meat, theta_meat_adaptive))

    #### Check eigenvalues
    try:
        theta_dim = LS_estimator.shape[0]
        hessian_betas = adaptive_sandwich_dict["stacked_hessian"][
            :-theta_dim, :-theta_dim
        ]
        beta_dim = int(hessian_betas.shape[0] / (args.T - 1))
        for t in range(args.T - 1):
            start_idx = beta_dim * t
            end_idx = beta_dim * (t + 1)
            tmp_beta_hessian = adaptive_sandwich_dict["stacked_hessian"][
                start_idx:end_idx, start_idx:end_idx
            ]
            eigvals = scipy.linalg.eigvals(tmp_beta_hessian)
            try:
                assert np.all(np.iscomplex(eigvals) == False)
            except:
                import ipdb

                ipdb.set_trace()

        eigvals2 = scipy.linalg.eigvals(
            adaptive_sandwich_dict["stacked_hessian"][-theta_dim:, -theta_dim:]
        )

        # eigvals = scipy.linalg.eigvals( adaptive_sandwich_dict['stacked_hessian'] )
        # hessian_betas = adaptive_sandwich_dict['stacked_hessian'][:-3,:-3]
        # eigvals = scipy.linalg.eigvals( hessian_betas + np.eye(hessian_betas.shape[0]) )
        assert np.all(np.iscomplex(eigvals2) == False)
        # assert np.min(np.absolute(eigvals)) > 0.001
    except:
        print("Checking eigenvalues")
        import ipdb

        ipdb.set_trace()

    if args.debug:
        # if True:
        print("sandwich")
        print(sandwich_var)

        print("adaptive_sandwich")
        print(adaptive_sandwich)

        print("adaptive_sandwich HC3")
        print(adaptive_sandwich_dict_HC3["adaptive_sandwich"])

        # alg_out_dict[1]['est_eqns_HC3']
        # est_val_dict['est_eqns_HC3']

        # scipy.linalg.eigvals( adaptive_sandwich_dict['stacked_hessian'][:4,:4] )

        if args.RL_alg == "sigmoid_LS":
            print("adaptive_sandwich_old")
            print(adaptive_sandwich_old)

        import ipdb

        ipdb.set_trace()

    # Save experimental analysis data
    exp_analysis_dict = {
        "LS_estimator": LS_estimator,
        "sandwich_var": sandwich_var,
        "adaptive_sandwich": adaptive_sandwich,
        "adaptive_sandwich_HC3": adaptive_sandwich_dict_HC3["adaptive_sandwich"],
        "adaptive_sandwich_dict": adaptive_sandwich_dict,
    }
    if args.RL_alg == "sigmoid_LS":
        exp_analysis_dict["adaptive_sandwich_old"] = adaptive_sandwich_old
    with open(analyses_out_path, "wb") as f:
        pkl.dump(exp_analysis_dict, f)


# Compute Standard and Adaptive Sandwich Variance Estimators ###################
sandwich_var = np.mean(all_sandwich, 0)

# Functions for Evaluating Variance Estimators ###################


def process_var_hotelling(results_dict, n, dval, fignum=0, name=""):
    assert "Empirical" in results_dict.keys()

    # Vector
    vec_array = np.array(results_dict["Empirical"]["raw_values"]).squeeze()
    errors_vec = vec_array - np.mean(vec_array, 0)

    hotelling_dict = {}
    maxX = 0
    for fignum, key in enumerate(results_dict.keys()):
        if key == "Empirical":
            matrix_array = np.expand_dims(results_dict[key]["cov_matrix_normalized"], 0)
            if matrix_array.shape[-1] == 1:
                matrix_array = matrix_array.reshape(1, 1, 1)
        elif key == "beta_est":
            continue
        else:
            matrix_array = np.array(results_dict[key]["raw_values"])

        # Hotelling's T statistic for entire vector
        if matrix_array.shape[-1] > 1:
            Sigma_inv = np.linalg.inv(matrix_array)
            hotelling_stat = np.einsum(
                "ij,ij->i", errors_vec, np.einsum("ijk,ik->ij", Sigma_inv, errors_vec)
            )
        else:
            Sigma_inv = 1 / matrix_array.squeeze()
            hotelling_stat = np.square(errors_vec) * Sigma_inv

        cutoff_raw = scipy.stats.f.ppf(0.95, dfn=dval, dfd=n - dval)
        cutoff = cutoff_raw * dval * (n - 1) / (n - dval)

        accepts = hotelling_stat <= cutoff
        coverage = round(np.mean(accepts), 4)
        coverage_sd = round(np.std(accepts) / np.sqrt(len(accepts)), 5)

        results_dict[key]["hotelling"] = {
            "coverage": (coverage, coverage_sd),
        }

        empirical_cutoff = np.percentile(hotelling_stat, 95)
        maxX = max(max(hotelling_stat), maxX)

        hotelling_dict[key] = {
            "cutoff": cutoff,
            "empirical_cutoff": empirical_cutoff,
            "hotelling_stat": hotelling_stat,
        }


def process_var(c_vec, results_dict, n, N, fignum=0):
    assert "Empirical" in results_dict.keys()

    # Vector
    vec_array = results_dict["Empirical"]["raw_values"]
    est_eqns = np.einsum("ij,j->i", vec_array, c_vec)
    errors = est_eqns - np.mean(est_eqns)

    # Info on standard error for empirical varaiance:
    # https://en.wikipedia.org/w/index.php?title=Variance&oldid=735567901
    # Distribution_of_the_sample_variance
    est_var = round(np.var(est_eqns * np.sqrt(args.n)), 4)
    N_tmp = len(est_eqns)
    fourth_m = stats.moment(est_eqns * np.sqrt(args.n), moment=4)
    second_m = np.var(est_eqns * np.sqrt(args.n))
    pre_std_error = (fourth_m - (N_tmp - 3) / (N_tmp - 1) * np.square(second_m)) / N_tmp
    std_error = round(np.sqrt(pre_std_error), 5)

    est_var_normalized = round(np.var(est_eqns), 4)
    fourth_m_normalized = stats.moment(est_eqns, moment=4)
    second_m_normalized = np.var(est_eqns)
    pre_std_error_normalized = (
        fourth_m - (N_tmp - 3) / (N_tmp - 1) * np.square(second_m)
    ) / N_tmp
    std_error_normalized = round(np.sqrt(pre_std_error_normalized), 5)

    results_dict["Empirical"][c_vec2string(c_vec)] = {
        "var_est": (est_var, std_error),
        "var_est_normalized": (est_var_normalized, std_error_normalized),
        "errors": errors,
    }

    var_dict = {}
    maxX = 0
    for key in results_dict.keys():
        if key == "Empirical":
            if len(c_vec) > 1:
                est_vars = np.dot(
                    np.matmul(results_dict["Empirical"]["cov_matrix"], c_vec), c_vec
                )
                est_vars_normalized = np.dot(
                    np.matmul(
                        results_dict["Empirical"]["cov_matrix_normalized"], c_vec
                    ),
                    c_vec,
                )
            else:
                est_vars = results_dict["Empirical"]["cov_matrix"]
                est_vars_normalized = results_dict["Empirical"]["cov_matrix_normalized"]
            est_vars = np.ones(N) * est_vars

        else:
            # Matrix
            matrix_array = results_dict[key]["raw_values"]
            est_vars = (
                np.einsum("ij,j", np.einsum("ijk,k->ij", matrix_array, c_vec), c_vec)
                * args.n
            )
            est_vars_normalized = np.einsum(
                "ij,j", np.einsum("ijk,k->ij", matrix_array, c_vec), c_vec
            )

        # Variance Estimate with Standard Errors
        ave_var = round(np.mean(est_vars), 4)
        std_error = round(np.std(est_vars) / np.sqrt(len(est_vars)), 5)
        median_var = round(np.median(est_vars), 4)

        # 95% Confidence Interval Coverage
        # cutoff = stats.norm.ppf(1-0.05/2)
        cutoff = stats.t.ppf(1 - 0.05 / 2, n - 1)
        var_stat = np.absolute(errors / np.sqrt(est_vars_normalized))
        accepts = var_stat <= cutoff

        coverage = round(np.mean(accepts), 4)
        coverage_sd = round(np.std(accepts) / np.sqrt(len(accepts)), 5)

        if key == "Empirical":
            results_dict[key][c_vec2string(c_vec)]["coverage"] = (coverage, coverage_sd)
        else:
            results_dict[key][c_vec2string(c_vec)] = {
                "var_est": (ave_var, std_error),
                "coverage": (coverage, coverage_sd),
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

results_dict["Empirical"] = {
    "raw_values": all_estimators,
    "cov_matrix": empirical_var,
    "cov_matrix_normalized": empirical_var_normalized,
}

# Compute Standard and Adaptive Sandwich Variance Estimators ###################
sandwich_var = np.mean(all_sandwich, 0)

results_dict["Sandwich"] = {
    "raw_values": all_sandwich,
    "cov_matrix": sandwich_var * args.n,
    "cov_matrix_normalized": sandwich_var,
}

if args.RL_alg != "fixed_randomization":
    adaptive_sandwich_var = np.mean(all_adaptive_sandwich, 0)
    adaptive_sandwich_var_HC3 = np.mean(all_adaptive_sandwich_HC3, 0)

    results_dict["Adaptive_Sandwich"] = {
        "raw_values": all_adaptive_sandwich,
        "cov_matrix": adaptive_sandwich_var * args.n,
        "cov_matrix_normalized": adaptive_sandwich_var,
    }
    results_dict["Adaptive_Sandwich_algHC3"] = {
        "raw_values": all_adaptive_sandwich_HC3,
        "cov_matrix": adaptive_sandwich_var_HC3 * args.n,
        "cov_matrix_normalized": adaptive_sandwich_var_HC3,
    }


# Evaluate Confidence Regions ###################

for fignum, c_vec in enumerate(c_vec_list):
    process_var(c_vec, results_dict, args.n, args.N, fignum=fignum)

process_var_hotelling(results_dict, args.n, dval)


###############################################################
# Write Out Results ###########################################
###############################################################


def writeout(outf, results_dict, c_vec_list, printlines=False):
    with open(outf, "w") as f:
        f.write(
            "\n================================================================================"
        )
        f.write(
            "\n================================================================================\n"
        )
        f.write("T={}".format(args.T))

        # Average Covariance Matrices
        f.write("\n========================================\n")
        for key in results_dict.keys():
            f.write(key + "\n")
            f.write(np.array2string(results_dict[key]["cov_matrix"]) + "\n")

        # Estimated Variance
        f.write("========================================")
        for c_vec in c_vec_list:
            cvec_str = c_vec2string(c_vec)
            f.write("\nEstimated Variance for {}\n".format(c_vec))
            for key in results_dict.keys():
                varstr = str(results_dict[key][cvec_str]["var_est"])
                f.write(str(key) + "\t" + varstr + "\n")
                if key != "Empirical":
                    varstr_median = str(results_dict[key][cvec_str]["var_median"])
                    f.write(str(key) + "\t" + varstr_median + "\n")

        # Confidence Interval Coverage
        f.write("========================================")
        for c_vec in c_vec_list:
            cvec_str = c_vec2string(c_vec)
            f.write("\nEmpirical Coverage for {} (95% CI)\n".format(c_vec))
            for key in results_dict.keys():
                coveragestr = str(results_dict[key][cvec_str]["coverage"])
                f.write(str(key) + "\t" + coveragestr + "\n")

        f.write("========================================")

        # Hotellings-t statistics
        f.write("\nHotelling's t-statistic (95% CI)\n")
        for key in results_dict.keys():
            # if key == "Empirical":
            #    continue
            coveragestr = str(results_dict[key]["hotelling"]["coverage"])
            f.write(str(key) + "\t" + coveragestr + "\n")

        f.write("========================================")

    if printlines:
        with open(outf, "r") as f:
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

    with open(latex_path, "w") as f:
        for c_vec in c_vec_list:
            cvec_str = c_vec2string(c_vec)
            f.write(cvec2name[cvec_str] + "\n")
            all_result_str = []
            result_str = "T={}, n={}, dataset={}, mode={}".format(
                args.T, args.n, args.dataset_type, mode
            )
            name_str = ""
            for key in results_dict.keys():
                if key == "Empirical":
                    continue
                name_str = name_str + " & " + key
                coveragestr = str(results_dict[key][cvec_str]["coverage"])
                result_str = result_str + " & " + coveragestr
            name_str = name_str + " \\\\ \hline\n"
            result_str = result_str + " \\\\ \hline\n"
            all_result_str.append(result_str)

            f.write(name_str)
            for result_str in all_result_str:
                f.write(result_str)
            f.write("\n")

        for stat in ["hotelling"]:
            f.write(stat2name[stat] + "\n")
            all_result_str = []

            result_str = "T={}, n={}, dataset={}, mode={}".format(
                args.T, args.n, args.dataset_type, mode
            )
            name_str = ""
            for key in results_dict.keys():
                if key == "Empirical":
                    continue
                name_str = name_str + " & " + key
                coveragestr = str(results_dict[key][stat]["coverage"])
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

with open(summary_path, "r") as f:
    lines = f.readlines()
for L in lines:
    print(L)

latex_path = os.path.join(all_folder_path, "latex.txt")
write_latex(latex_path, results_dict, c_vec_list)

results_path = os.path.join(all_folder_path, "results.pkl")
print("\nWriting results to {}".format(results_path))
with open(results_path, "wb") as f:
    pkl.dump(results_dict, f)
