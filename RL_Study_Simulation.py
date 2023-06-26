import argparse
import pickle as pkl
import time
import json
import os

import numpy as np

from synthetic_env import load_synthetic_env, SyntheticEnv
from oralytics_env import load_oralytics_env, OralyticsEnv
from debug_helper import output_variance_pieces
from basic_RL_algorithms import FixedRandomization, SigmoidLS
from smooth_posterior_sampling import SmoothPosteriorSampling
from constants import RLStudyArgs

###############################################################
# Simulation Functions ########################################
###############################################################


def run_study_simulation(args, study_env, study_RLalg, user_env_data):
    """
    Goal: Simulates a study with n users
    Input: Environment object (e.g., instance of HeartstepStudyEnv)
           RL algorithm object (e.g., instance of SigmoidLS)
    Output: Dataframe with all collected study data (study_df)
            Original RL algorithm object (populated with additional data)
    """

    # study_df is a data frame with a record of all data collected in study
    study_df = study_env.make_empty_study_df(args, user_env_data)

    # Loop over all decision times ###############################################
    for t in range(1, study_env.calendar_T + 1):
        if args.RL_alg != RLStudyArgs.FIXED_RANDOMIZATION:
            # Update study_df with info on latest policy used to select actions
            study_df.loc[
                study_df["calendar_t"] == t, "policy_last_t"
            ] = study_RLalg.all_policies[-1]["policy_last_t"]
            study_df.loc[study_df["calendar_t"] == t, "policy_num"] = len(
                study_RLalg.all_policies
            )

        else:
            study_df.loc[study_df["calendar_t"] == t, "policy_last_t"] = 0
            study_df.loc[study_df["calendar_t"] == t, "policy_num"] = 1

        curr_timestep_data = study_df[study_df["calendar_t"] == t]

        # Sample Actions #####################################################
        action_probs = study_RLalg.get_action_probs(
            curr_timestep_data, filter_keyval=("calendar_t", t)
        )

        if args.dataset_type == RLStudyArgs.HEARTSTEPS:
            action_probs *= curr_timestep_data["availability"]
        actions = study_RLalg.rng.binomial(1, action_probs)

        # Sample Rewards #####################################################
        if args.dataset_type == RLStudyArgs.ORALYTICS:
            rewards, brush_times = study_env.sample_rewards(
                curr_timestep_data, actions, t
            )
        else:
            rewards = study_env.sample_rewards(curr_timestep_data, actions, t)

        # Record all collected data #######################################
        if args.dataset_type == RLStudyArgs.ORALYTICS:
            fill_columns = ["reward", "brush_time", "action", "action1prob"]
            fill_vals = np.vstack([rewards, brush_times, actions, action_probs]).T
            study_df.loc[study_df["calendar_t"] == t, fill_columns] = fill_vals
        else:
            fill_columns = ["reward", "action", "action1prob"]
            fill_vals = np.vstack([rewards, actions, action_probs]).T
            study_df.loc[study_df["calendar_t"] == t, fill_columns] = fill_vals

        if t < study_env.calendar_T:
            # Record data to prepare for state at next decision time
            current_users = study_df[study_df["calendar_t"] == t]["user_id"]
            study_df = study_env.update_study_df(study_df, t)

        # Check if need to update algorithm #######################################
        if (
            t % args.decisions_between_updates == 0
            and args.RL_alg != RLStudyArgs.FIXED_RANDOMIZATION
        ):
            # check enough avail data and users; if so, update algorithm
            most_recent_policy_t = study_RLalg.all_policies[-1]["policy_last_t"]
            new_obs_bool = np.logical_and(
                study_df["calendar_t"] <= t,
                study_df["calendar_t"] > most_recent_policy_t,
            )
            new_update_data = study_df[new_obs_bool]
            all_prev_data = study_df[study_df["calendar_t"] <= t]

            if args.dataset_type == RLStudyArgs.HEARTSTEPS:
                num_avail = np.sum(new_update_data["availability"])
            else:
                num_avail = 1
            prev_num_users = len(study_df[study_df["calendar_t"] == t])

            if num_avail > 0 and prev_num_users >= args.min_users:
                # Update Algorithm ##############################################
                study_RLalg.update_alg(new_update_data, update_last_t=t)

    if args.RL_alg == RLStudyArgs.POSTERIOR_SAMPLING:
        fill_columns = ["policy_last_t", "policy_num"]
        for col in fill_columns:
            study_RLalg.norm_samples_df[col] = study_df[col].to_numpy().copy()

    return study_df, study_RLalg


def load_data_and_simulate_studies(args, gen_feats, alg_state_feats, alg_treat_feats):
    ###############################################################
    # Load Data and Models ########################################
    ###############################################################

    if args.dataset_type == RLStudyArgs.HEARTSTEPS:
        raise NotImplementedError()

    elif args.dataset_type == RLStudyArgs.SYNTHETIC:
        user_env_data = None
        paramf_path = f"./synthetic_env_params/{args.synthetic_mode}.txt"
        env_params = load_synthetic_env(paramf_path)
        if len(env_params.shape) == 2:
            assert env_params.shape[0] >= args.T

    elif args.dataset_type == RLStudyArgs.ORALYTICS:
        paramf_path = "./oralytics_env_params/non_stat_zero_infl_pois_model_params.csv"
        param_names, bern_params, poisson_params = load_oralytics_env(paramf_path)
        # TODO: should these be used?
        treat_feats = [
            RLStudyArgs.INTERCEPT,
            RLStudyArgs.TIME_OF_DAY,
            RLStudyArgs.WEEKEND,
            RLStudyArgs.DAY_IN_STUDY_NORM,
            RLStudyArgs.PRIOR_DAY_BRUSH,
        ]

        user_env_data = {"bern_params": bern_params, "poisson_params": poisson_params}

    else:
        raise ValueError("Invalid Dataset Type")

    ###############################################################
    # Simulate Studies ############################################
    ###############################################################

    tic = time.perf_counter()

    if args.dataset_type == RLStudyArgs.HEARTSTEPS:
        mode = args.heartsteps_mode
    elif args.dataset_type == RLStudyArgs.SYNTHETIC:
        mode = args.synthetic_mode
    elif args.dataset_type == RLStudyArgs.ORALYTICS:
        mode = None
    else:
        raise ValueError("Invalid dataset type")

    print("Running simulations...")
    if args.dataset_type == RLStudyArgs.ORALYTICS:
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

    simulation_data_path = os.path.join(args.save_dir, "simulated_data")
    if not os.path.isdir(simulation_data_path):
        os.mkdir(simulation_data_path)
    all_folder_path = os.path.join(simulation_data_path, exp_str)
    if not os.path.isdir(all_folder_path):
        os.mkdir(all_folder_path)

    with open(os.path.join(all_folder_path, "args.json"), "w") as f:
        json.dump(vars(args), f)

    policy_grad_norm = []
    for i in range(1, args.N + 1):
        env_seed = i * 5000
        alg_seed = (args.N + i) * 5000

        if i == 10 or i % 25 == 0:
            toc = time.perf_counter()
            print(f"{i} ran in {toc - tic:0.4f} seconds")

        # Initialize study environment ############################################
        if args.dataset_type == RLStudyArgs.HEARTSTEPS:
            raise NotImplementedError()
        elif args.dataset_type == RLStudyArgs.SYNTHETIC:
            study_env = SyntheticEnv(
                args,
                env_params,
                env_seed=env_seed,
                gen_feats=gen_feats,
                err_corr=args.err_corr,
            )
        elif args.dataset_type == RLStudyArgs.ORALYTICS:
            study_env = OralyticsEnv(
                args, param_names, bern_params, poisson_params, env_seed=env_seed
            )
        else:
            raise ValueError("Invalid Dataset Type")

        # Initialize RL algorithm ###################################################
        if args.RL_alg == RLStudyArgs.FIXED_RANDOMIZATION:
            study_RLalg = FixedRandomization(
                args, alg_state_feats, alg_treat_feats, alg_seed=alg_seed
            )
        elif args.RL_alg == RLStudyArgs.SIGMOID_LS:
            study_RLalg = SigmoidLS(
                args,
                alg_state_feats,
                alg_treat_feats,
                allocation_sigma=args.allocation_sigma,
                alg_seed=alg_seed,
                steepness=args.steepness,
            )
        elif args.RL_alg == RLStudyArgs.POSTERIOR_SAMPLING:
            if args.prior == RLStudyArgs.NAIVE:
                if args.action_centering:
                    total_dim = len(alg_state_feats) + len(alg_treat_feats) * 2
                    prior_mean = np.ones(total_dim) * 0.1
                    prior_var = np.eye(total_dim) * 2
                else:
                    total_dim = len(alg_state_feats) + len(alg_treat_feats)
                    prior_mean = np.ones(total_dim) * 0.1
                    prior_var = np.eye(total_dim) * 0.5

            else:
                raise ValueError(f"Invalid prior type: {args.prior}")
            study_RLalg = SmoothPosteriorSampling(
                args,
                alg_state_feats,
                alg_treat_feats,
                alg_seed=alg_seed,
                allocation_sigma=args.allocation_sigma,
                steepness=args.steepness,
                prior_mean=prior_mean,
                prior_var=prior_var,
                noise_var=args.noise_var,
                action_centering=args.action_centering,
            )
        else:
            raise ValueError("Invalid RL Algorithm Type")

        # Run Study Simulation #######################################################
        study_df, study_RLalg = run_study_simulation(
            args, study_env, study_RLalg, user_env_data
        )

        # Print summary statistics
        if i == 25 and args.RL_alg != RLStudyArgs.FIXED_RANDOMIZATION:
            print(f"\nTotal Update Times: {len(study_RLalg.all_policies) - 1}")
            study_df.action = study_df.action.astype("int")
            study_df.policy_last_t = study_df.policy_last_t.astype("int")
            if args.dataset_type == RLStudyArgs.HEARTSTEPS:
                study_df.stepcount = (np.exp(study_df.reward) - 0.5).astype("int")

        # Make histogram of rewards (available)
        if i == 0:
            print(study_df.head())
            # TODO: For heartsteps evalsim mode we may want to make step count
            # histograms here

        # Save Data #################################################################
        if args.verbose:
            print("Saving data...")

        folder_path = os.path.join(all_folder_path, f"exp={i}")
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        if args.RL_alg != RLStudyArgs.FIXED_RANDOMIZATION:
            study_df = study_df.astype(
                {"policy_num": "int32", "policy_last_t": "int32", "action": "int32"}
            )

        study_df.to_csv(f"{folder_path}/data.csv", index=False)
        with open(f"{folder_path}/study_df.pkl", "wb") as f:
            pkl.dump(study_df, f)

        with open(f"{folder_path}/study_RLalg.pkl", "wb") as f:
            pkl.dump(study_RLalg, f)

        # TODO eventually remove Save Variance Components #############################################
        if args.RL_alg == RLStudyArgs.SIGMOID_LS:
            out_dict = output_variance_pieces(study_df, study_RLalg, args)
            with open(f"{folder_path}/out_dict.pkl", "wb") as file:
                pkl.dump(out_dict, file)

            policy_grad_norm.append(
                np.max(np.absolute([y["pi_grads"] for x, y in out_dict.items()]))
            )

    toc = time.perf_counter()
    print(f"Final ran in {toc - tic:0.4f} seconds")


def main():
    ###############################################################
    # Initialize Simulation Hyperparameters #######################
    ###############################################################

    parser = argparse.ArgumentParser(description="Generate simulation data")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=RLStudyArgs.SYNTHETIC,
        choices=[RLStudyArgs.HEARTSTEPS, RLStudyArgs.SYNTHETIC, RLStudyArgs.ORALYTICS],
    )
    parser.add_argument("--verbose", type=int, default=0, help="Prints helpful info")
    parser.add_argument(
        "--heartsteps_mode",
        default=RLStudyArgs.MEDIUM,
        choices=[
            RLStudyArgs.EVALSIM,
            RLStudyArgs.REALISTIC,
            RLStudyArgs.MEDIUM,
            RLStudyArgs.EASY,
        ],
        help="Sets default parameter values accordingly",
    )
    parser.add_argument(
        "--synthetic_mode",
        type=str,
        default=RLStudyArgs.DELAYED_EFFECTS,
        help="File name of synthetic env params",
    )
    parser.add_argument(
        "--RL_alg",
        default=RLStudyArgs.SIGMOID_LS,
        choices=[
            RLStudyArgs.FIXED_RANDOMIZATION,
            RLStudyArgs.SIGMOID_LS,
            RLStudyArgs.POSTERIOR_SAMPLING,
        ],
        help="RL algorithm used to select actions",
    )
    parser.add_argument(
        "--N", type=int, default=10, help="Number of Monte Carlo repetitions"
    )
    parser.add_argument("--n", type=int, default=90, help="Total number of users")
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
        "--err_corr",
        default=RLStudyArgs.TIME_CORR,
        choices=[RLStudyArgs.TIME_CORR, RLStudyArgs.INDEPENDENT],
        help="Noise error correlation structure",
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
    parser.add_argument(
        "--steepness", type=float, default=10, help="Allocation steepness"
    )
    parser.add_argument(
        "--alg_state_feats",
        type=str,
        default=RLStudyArgs.INTERCEPT,
        help="Comma separated list of algorithm state features",
    )
    parser.add_argument(
        "--action_centering",
        type=int,
        default=0,
        help="Whether posterior sampling algorithm uses action centering",
    )
    parser.add_argument(
        "--prior",
        type=str,
        default=RLStudyArgs.NAIVE,
        choices=[RLStudyArgs.NAIVE, RLStudyArgs.ORALYTICS],
        help="Prior for posterior sampling algorithm",
    )
    tmp_args = parser.parse_known_args()[0]

    if tmp_args.dataset_type == RLStudyArgs.HEARTSTEPS:
        raise NotImplementedError()
    elif tmp_args.dataset_type == RLStudyArgs.SYNTHETIC:
        arg_dict = {
            RLStudyArgs.T: 2,
            RLStudyArgs.RECRUIT_N: tmp_args.n,
            RLStudyArgs.RECRUIT_T: 1,
            RLStudyArgs.ALLOCATION_SIGMA: 1,
            RLStudyArgs.NOISE_VAR: 1,
        }

        # Algorithm state features
        alg_state_feats = tmp_args.alg_state_feats.split(",")
        alg_treat_feats = alg_state_feats

        # Generation features
        past_action_len = 1
        past_action_cols = [RLStudyArgs.INTERCEPT] + [
            f"past_action_{i}" for i in range(1, past_action_len + 1)
        ]
        past_reward_action_cols = ["past_reward"] + [
            f"past_action_{i}_reward" for i in range(1, past_action_len + 1)
        ]
        gen_feats = past_action_cols + past_reward_action_cols + ["dosage"]

    elif tmp_args.dataset_type == RLStudyArgs.ORALYTICS:
        arg_dict = {
            RLStudyArgs.T: 50,
            RLStudyArgs.RECRUIT_N: tmp_args.n,
            RLStudyArgs.RECRUIT_T: 1,
            RLStudyArgs.ALLOCATION_SIGMA: 1,
            RLStudyArgs.NOISE_VAR: 1,
        }

        # allocation_sigma: 163 (truncated brush times); 5.7 (square-root of truncated brush times)

        alg_state_feats = [
            RLStudyArgs.INTERCEPT,
            RLStudyArgs.TIME_OF_DAY,
            RLStudyArgs.PRIOR_DAY_BRUSH,
        ]
        alg_treat_feats = alg_state_feats

    else:
        raise ValueError("Invalid Dataset Type")

    parser.add_argument(
        "--T",
        type=int,
        default=arg_dict[RLStudyArgs.T],
        help="Total number of decision times per user",
    )
    parser.add_argument(
        "--recruit_n",
        type=int,
        default=arg_dict[RLStudyArgs.RECRUIT_N],
        help="Number of users recruited on each recruitment times",
    )
    parser.add_argument(
        "--recruit_t",
        type=int,
        default=arg_dict[RLStudyArgs.RECRUIT_T],
        help="Number of updates between recruitment times (minmum 1)",
    )
    parser.add_argument(
        "--allocation_sigma",
        type=float,
        default=arg_dict[RLStudyArgs.ALLOCATION_SIGMA],
        help="Sigma used in allocation of algorithm",
    )
    parser.add_argument(
        "--noise_var",
        type=float,
        default=arg_dict[RLStudyArgs.ALLOCATION_SIGMA],
        help="Posterior sampling noise variance",
    )

    args = parser.parse_args()
    print(vars(args))

    assert args.T >= args.decisions_between_updates

    load_data_and_simulate_studies(args, gen_feats, alg_state_feats, alg_treat_feats)


if __name__ == "__main__":
    main()
