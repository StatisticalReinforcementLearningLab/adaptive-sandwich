import argparse
import pickle as pkl
import os

import jax
from jax import numpy as jnp
import numpy as np

from least_squares_helper import get_est_eqn_LS, fit_WLS
from debug_helper import get_adaptive_sandwich

from helper_functions import get_user_actions, get_user_rewards, get_user_action1probs


def c_vec2string(c_vec):
    return np.array2string(c_vec)[1:-1].replace(" ", ",")


def load_data(args, folder_path):
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
    with open(f"{folder_path}/study_df.pkl", "rb") as f:
        study_df = pkl.load(f)

    if args.dataset_type == "synthetic":
        study_df["action_past_reward"] = study_df["past_reward"] * study_df["action"]

    if args.RL_alg == "fixed_randomization":
        return study_df

    with open(f"{folder_path}/study_RLalg.pkl", "rb") as f:
        study_RLalg = pkl.load(f)

    # OLD STUFF
    if args.RL_alg == "sigmoid_LS":
        with open(os.path.join(folder_path, "out_dict.pkl"), "rb") as file:
            alg_out_dict = pkl.load(file)
    else:
        alg_out_dict = None

    return study_df, study_RLalg, alg_out_dict


# TODO: think about the mixed derivative. If we want to actually differentiate to get it, we
# need to actually plug in betas to get action1probs here, so that we can differentiate with
# respect to beta. But there are major questions about how to specify this interface to users.
# If they refer to an action probability, it needs to be translated into the actual function
# to get action probabilities from the RL side.  This points to an option to toggle action centering,
# rather than just allowing it to be directly specified in the model.
# Also if we require actual pi functions, that can only really happen if initial after study side is in python
def get_loss(theta_est, base_states, treat_states, actions, rewards, action1probs=None):
    theta_0 = theta_est[: base_states.shape[1]].reshape(-1, 1)
    theta_1 = theta_est[base_states.shape[1] :].reshape(-1, 1)

    # Perform action centering if given action probabilities
    if action1probs is not None:
        # TODO: deal with types more cleanly
        actions = actions.astype(jnp.float64)
        actions -= action1probs

    return jnp.sum(
        (
            rewards
            - jnp.matmul(base_states, theta_0)
            - jnp.matmul(actions * treat_states, theta_1)
        )
        ** 2
    )


get_loss_gradient = jax.grad(get_loss)
get_loss_hessian = jax.hessian(get_loss)


def analyze_dataset(dataset_num, args, folder_template):
    # TODO: Not sure why this would be 100
    if dataset_num % 100 == 0:
        print(f"dataset num {dataset_num}")

    # Load Data #########################################
    folder_path = os.path.join(folder_template, f"exp={dataset_num}")
    if args.RL_alg == "fixed_randomization":
        study_df = load_data(args, folder_path)
    else:
        study_df, study_RLalg, alg_out_dict = load_data(args, folder_path)

    # List of times that were the first applicable time for some update
    # TODO: sort to not rely on insertion order?
    update_times = [
        t
        for t, value in study_RLalg.algorithm_statistics_by_calendar_t.items()
        if "loss_gradients_by_user_id" in value
    ]

    theta_est = estimate_theta(study_df)

    # TODO: what about when theta and beta have different dimensions?
    # TODO: state features should not be the same as RL alg for full generality
    bread_matrix = form_bread_matrix(
        study_RLalg.upper_left_bread_inverse,
        study_df,
        study_RLalg.algorithm_statistics_by_calendar_t,
        update_times,
        study_RLalg.state_feats,
        study_RLalg.treat_feats,
        study_RLalg.beta_dim,
        theta_est,
    )

    meat_matrix = form_meat_matrix(
        study_df,
        theta_est,
        study_RLalg.state_feats,
        study_RLalg.treat_feats,
        update_times,
        study_RLalg.beta_dim,
        study_RLalg.algorithm_statistics_by_calendar_t,
    )

    variance = np.linalg.multi_dot([bread_matrix, meat_matrix, bread_matrix.T])
    print(variance)


# TODO: Implement. Using loss function would be nice (seems possible with some
# scipy least squares functions)
def estimate_theta(study_df):
    return np.array([1.0, 2, 3, 4])


def form_meat_matrix(
    study_df,
    theta_est,
    state_feats,
    treat_feats,
    update_times,
    beta_dim,
    algo_stats_dict,
):
    user_ids = study_df.user_id.unique()
    running_meat_vector = np.zeros((beta_dim * (len(update_times) + 1), 1)).astype(
        "float64"
    )
    for user_id in user_ids:
        running_meat_vector += np.concatenate(
            [
                algo_stats_dict[t]["loss_gradients_by_user_id"][user_id]
                for t in update_times
            ]
            + [
                get_loss_gradient(
                    theta_est,
                    *get_user_states(study_df, state_feats, treat_feats, user_id),
                    actions=get_user_actions(study_df, user_id),
                    rewards=get_user_rewards(study_df, user_id),
                    action1probs=get_user_action1probs(study_df, user_id),
                )
            ],
        ).reshape(-1, 1)

    average_meat_vector = running_meat_vector / len(user_ids)
    return np.outer(average_meat_vector, average_meat_vector)


def get_user_states(study_df, state_feats, treat_feats, user_id):
    """
    Extract just the rewards for the given user in the given study_df as a
    numpy (column) vector.

    Optionally specify a specific calendar time at which to do so.
    """
    user_df = study_df.loc[study_df.user_id == user_id]
    base_states = user_df[state_feats].to_numpy()
    treat_states = user_df[treat_feats].to_numpy()
    return (base_states, treat_states)


def form_bread_matrix(
    upper_left_bread_inverse,
    study_df,
    algo_stats_dict,
    update_times,
    state_feats,
    treat_feats,
    beta_dim,
    theta_est,
):
    existing_rows = upper_left_bread_inverse.shape[0]
    user_ids = study_df.user_id.unique()

    max_t = study_df.calendar_t.max()
    # This is useful for sweeping through the decision times between updates
    # but also those after the final update
    update_times_and_upper_limit = (
        update_times if update_times[-1] == max_t + 1 else update_times + [max_t + 1]
    )

    theta_dim = len(theta_est)

    # pylint: disable=consider-using-enumerate
    # Think of each iteration of this loop as creating one term in the final (block) row
    bottom_left_row_blocks = []
    for i in range(len(update_times)):
        running_entry_holder = np.zeros((theta_dim, theta_dim))

        # This loop calculates the per-user quantities that will be
        # averaged for the final matrix entries
        for user_id in user_ids:
            theta_loss_gradient = get_loss_gradient(
                theta_est,
                *get_user_states(study_df, state_feats, treat_feats, user_id),
                actions=get_user_actions(study_df, user_id),
                rewards=get_user_rewards(study_df, user_id),
                action1probs=get_user_action1probs(study_df, user_id),
            )
            weight_gradient_sum = np.zeros(beta_dim)

            # This loop iterates over decision times in slices between updates
            # to collect the right weight gradients
            for t in range(
                update_times_and_upper_limit[i],
                update_times_and_upper_limit[i + 1],
            ):
                weight_gradient_sum += algo_stats_dict[t][
                    "weight_gradients_by_user_id"
                ][user_id]

            running_entry_holder += np.outer(theta_loss_gradient, weight_gradient_sum)

            # TODO: Figure out how to add mixed derivative term here.
            # Hard because need actual pi functions to use jax, otherwise need to construct from pi gradients
            # But these have implications for the statistician's interface
        bottom_left_row_blocks.append(running_entry_holder / len(user_ids))

    bottom_right_hessian = sum(
        (
            np.array(
                get_loss_hessian(
                    theta_est,
                    *get_user_states(study_df, state_feats, treat_feats, user_id),
                    actions=get_user_actions(study_df, user_id),
                    rewards=get_user_rewards(study_df, user_id),
                )
            )
            for user_id in user_ids
        )
    ) / len(user_ids)

    return np.block(
        [
            [
                upper_left_bread_inverse,
                np.zeros((existing_rows, theta_dim)),
            ],
            [
                np.block(bottom_left_row_blocks),
                bottom_right_hessian,
            ],
        ]
    )


def main():
    # TODO: Nearly all args should be removed.
    # TODO: Should take its input data location as an argument
    ###############################################################
    # Initialize Hyperparameters ##################################
    ###############################################################

    parser = argparse.ArgumentParser(description="Analyze data")
    # TODO: Perhaps omit next three and specify model in a cleaner way
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
    # TODO: Fixed rand not crticial for anything, just a useful test
    parser.add_argument(
        "--RL_alg",
        default="sigmoid_LS",
        choices=["fixed_randomization", "sigmoid_LS", "posterior_sampling"],
        help="RL algorithm used to select actions",
    )
    # TODO: Probably remove, simulation part
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
        "--min_users",
        type=int,
        default=25,
        help="Min number of users needed to update alg",
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
    parser.add_argument(
        "--steepness", type=float, default=10, help="Allocation steepness"
    )
    # TODO: Possibly just encode in description of model
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

    ############################################################################
    # Construct inference targets according to dataset type and inference mode
    ############################################################################
    arg_dict = feature_names = outcome_name = c_vec_list = dval = cvec2name = None
    stat2name = {
        "hotelling": "Hotelling's t-squared Statistic",
    }

    if tmp_args.dataset_type == "synthetic":
        arg_dict = {
            "T": 2,
            "recruit_n": tmp_args.n,
            "recruit_t": 1,
            "allocation_sigma": 1,
        }

        if tmp_args.inference_mode == "value":
            # Inference Objective
            feature_names = ["intercept"]
            outcome_name = "reward"
            c_vec_list = [np.array([1])]
            dval = 1

            cvec2name = {
                c_vec2string(np.array([1])): "Entire Vector",
            }
        # TODO: Can I remove this? It's not a supported option currently
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
    else:
        raise ValueError("Dataset type not supported")

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

    analyze_experiments_and_output_results(
        args, feature_names, outcome_name, c_vec_list, dval
    )


def analyze_experiments_and_output_results(
    args, feature_names, outcome_name, c_vec_list, dval
):
    mode = None
    if args.dataset_type == "heartsteps":
        mode = args.heartsteps_mode
    elif args.dataset_type == "synthetic":
        mode = args.synthetic_mode

    # TODO: Need to write generic code for this and do it on both RL side and here
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
    folder_template = os.path.join(args.save_dir, f"simulated_data/{exp_str}")
    # TODO: Probably real use case won't really have multiple datasets
    for i in range(1, args.N + 1):
        analyze_dataset(i, args, folder_template)


if __name__ == "__main__":
    main()
