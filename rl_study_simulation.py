import argparse
import time
import json
import os
import cProfile
from pstats import Stats
import logging

import numpy as np
import pandas as pd
import cloudpickle as pickle

from synthetic_env import load_synthetic_env_params, SyntheticEnv
from basic_RL_algorithms import FixedRandomization, SigmoidLS, SmoothPosteriorSampling
from constants import RLStudyArgs

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-2s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)

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

    logger.info("Beginning single simulation, in Python code.")
    # study_df is a data frame with a record of all data collected in study
    study_df = study_env.make_empty_study_df(args, user_env_data)

    max_calendar_t = study_df["calendar_t"].max()

    # Loop over all decision times ###############################################
    logger.info("Maximum decision time: %s.", max_calendar_t)
    for t in range(1, max_calendar_t + 1):
        logger.info("Processing decision time %s.", t)

        curr_time_bool = (study_df["calendar_t"] == t) & (study_df["in_study"] == 1)

        # Update study_df with info on latest policy used to select actions
        study_df.loc[curr_time_bool, "policy_num"] = (
            1
            if args.RL_alg == RLStudyArgs.FIXED_RANDOMIZATION
            else len(study_RLalg.all_policies)
        )

        curr_timestep_data = study_df[curr_time_bool]

        # Sample Actions #####################################################
        logger.info("Sampling actions for time %s.", t)
        action_probs = study_RLalg.get_action_probs(curr_timestep_data)

        actions = study_RLalg.rng.binomial(1, action_probs)

        # Sample Rewards #####################################################
        logger.info("Sampling rewards for time %s.", t)
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
            study_df.loc[
                (study_df["calendar_t"] == t) & (study_df["in_study"] == 1),
                fill_columns,
            ] = fill_vals
        else:
            fill_columns = ["reward", "action", "action1prob"]
            fill_vals = np.vstack([rewards, actions, action_probs]).T
            study_df.loc[
                (study_df["calendar_t"] == t) & (study_df["in_study"] == 1),
                fill_columns,
            ] = fill_vals

        if t < study_env.calendar_T:
            logger.info("Updating study df for time %s.", t)
            # Record data to prepare for state at next decision time
            study_df = study_env.update_study_df(study_df, t)

        # Note that we DO NOT filter to in_study == 1 here.  The way we calculate the gradients
        # we need in batches benefits from same-size state inputs for each user, so we actually
        # want to pass states for when users are not in the study but zero them out.
        all_prev_data_bool = study_df["calendar_t"] <= t
        all_prev_data = study_df[all_prev_data_bool]

        # Record quantities that will be needed to form the "bread" matrix.
        # These are actually only needed if t > args.decisions_between_updates
        # but always collect anyway for clarity since that
        # structure is the simplest to ask for and so we may as well test the
        # full pipeline doing that.
        logger.info("Collecting pi args for time %s.", t)
        study_RLalg.collect_pi_args(all_prev_data, t)

        # Check if need to update algorithm #######################################
        # TODO: recruit_t not respected here.  Either remove it or use here.
        if (
            t < study_env.calendar_T
            and t % args.decisions_between_updates == args.update_cadence_offset
            and args.RL_alg != RLStudyArgs.FIXED_RANDOMIZATION
            and t >= args.min_update_time
        ):
            last_policy_num = len(study_RLalg.all_policies) - 1

            # Min users used to be enforced here. It is ignored now.

            # Update Algorithm ##############################################
            logger.info("Updating algorithm parameters for time %s.", t)
            if study_RLalg.incremental_updates:
                # If incremental updates, we only need new data to update the algorithm.
                new_obs_bool = (
                    all_prev_data_bool
                    & (study_df["policy_num"] > last_policy_num)
                    & (study_df["in_study"] == 1)
                )
                update_data = study_df[new_obs_bool]
            else:
                # Otherwise use all data so far
                update_data = all_prev_data

            study_RLalg.update_alg(update_data)

            # NOTE: Very important that this is called AFTER the above update.
            # It is a little confusing that the beta here is the beta that the
            # rest of the data already produced, whereas for the pis the beta
            # is used to produce the probability at that decision time.
            study_RLalg.collect_rl_update_args(all_prev_data, t)
    return study_df, study_RLalg


def load_data_and_simulate_studies(args, gen_feats, alg_state_feats, alg_treat_feats):
    ###############################################################
    # Load Data and Models ########################################
    ###############################################################

    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    if args.dataset_type == RLStudyArgs.HEARTSTEPS:
        raise NotImplementedError()

    logging.info("Loading environment...")
    if args.dataset_type == RLStudyArgs.SYNTHETIC:
        mode = args.synthetic_mode
        user_env_data = None
        paramf_path = f"./synthetic_env_params/{mode}.txt"
        env_params = load_synthetic_env_params(paramf_path)
        if len(env_params.shape) == 2:
            assert env_params.shape[0] >= args.T

        if args.RL_alg == RLStudyArgs.SIGMOID_LS:
            exp_str = (
                f"{args.dataset_type}_mode={mode}_alg={args.RL_alg}_T={args.T}_n={args.n}_"
                f"recruitN={args.recruit_n}_decisionsBtwnUpdates={args.decisions_between_updates}_"
                f"steepness={args.steepness}_algfeats={args.alg_state_feats}_errcorr={args.err_corr}_"
                f"actionC={args.action_centering}"
            )
        elif args.RL_alg == RLStudyArgs.SMOOTH_POSTERIOR_SAMPLING:
            exp_str = (
                f"{args.dataset_type}_mode={mode}_alg={args.RL_alg}_T={args.T}_n={args.n}_"
                f"recruitN={args.recruit_n}_decisionsBtwnUpdates={args.decisions_between_updates}_"
                f"algfeats={args.alg_state_feats}_errcorr={args.err_corr}_"
                f"actionC={args.action_centering}"
            )
        else:
            raise ValueError("Invalid RL Algorithm Type For Synthetic Dataset")

    elif args.dataset_type == RLStudyArgs.ORALYTICS:
        raise NotImplementedError()
        # If we want this, there is an implementation in the replicable  bandits
        # repo

    else:
        raise ValueError("Invalid Dataset Type")

    ###############################################################
    # Simulate Studies ############################################
    ###############################################################
    simulation_data_path = os.path.join(args.save_dir, "simulated_data")
    if not os.path.isdir(simulation_data_path):
        os.mkdir(simulation_data_path)
    all_folder_path = os.path.join(simulation_data_path, exp_str)
    if not os.path.isdir(all_folder_path):
        os.mkdir(all_folder_path)

    logger.info("Dumping arguments to json file...")
    with open(os.path.join(all_folder_path, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f)

    tic = time.perf_counter()
    toc1 = tic
    toc2 = None

    logger.info("Running simulations...")
    # *5000 to avoid neighboring seeds just in case...
    # Important that env and alg seed are different.
    # Note how parallel_task_index is used to get different seeds
    # when multiple simulations are being run in parallel.  In that case
    # we should have N = 1 in each simulation, so that there is only
    # one iteration of this loop with i = 1, and the seed will be determined
    # by the parallel task index. On the other hand, by default the parallel
    # task index is 1, so for a typical non-parallel run with N > 1 the seeds
    # will be determined by the iterator i.
    for i in range(1, args.N + 1):
        # Dynamic seeds are useful for quick testing, but for reproducibility
        # fixed seeds should be used. Dynamic seeds are also never really useful
        # in large simulations anyway.
        time_bump = 0 if not args.dynamic_seeds else int(time.time())
        env_seed = (
            time_bump + args.parallel_task_index * i * 5000 + 1
            if (args.env_seed_override is None or args.env_seed_override < 0)
            else args.env_seed_override
        )
        alg_seed = (
            time_bump + args.parallel_task_index * (args.N + i) * 5000
            if (args.alg_seed_override is None or args.alg_seed_override < 0)
            else args.alg_seed_override
        )
        logger.info("Seeds: env=%d, alg=%d", env_seed, alg_seed)

        toc2 = time.perf_counter()
        if i > 1:
            logger.info(
                "Simulation %d of %d ran in %.4f seconds", i - 1, args.N, toc2 - toc1
            )
        toc1 = toc2

        # Initialize study environment ############################################
        if args.dataset_type == RLStudyArgs.SYNTHETIC:
            study_env = SyntheticEnv(
                args,
                env_params,
                env_seed=env_seed,
                gen_feats=gen_feats,
                err_corr=args.err_corr,
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
                state_feats=alg_state_feats,
                treat_feats=alg_treat_feats,
                alg_seed=alg_seed,
                steepness=args.steepness,
                lower_clip=args.lower_clip,
                upper_clip=args.upper_clip,
                action_centering=args.action_centering,
            )
        elif args.RL_alg == RLStudyArgs.SMOOTH_POSTERIOR_SAMPLING:
            num_regression_params = len(alg_state_feats + alg_treat_feats)
            if args.prior_mean == RLStudyArgs.NAIVE:
                prior_mean = np.zeros(num_regression_params)
            else:
                prior_mean = np.array(args.prior_mean.split(","), dtype=np.float32)

            if args.prior_var_upper_triangle == RLStudyArgs.NAIVE:
                prior_var = 1000000 * np.eye(num_regression_params)
            else:
                # Note this is row-major, moving left to right across rows in sequence
                upper_triangle_terms = np.array(
                    args.prior_var_upper_triangle.split(","), dtype=np.float32
                )
                upper_triangle_indices = np.triu_indices(num_regression_params)

                prior_var = np.zeros(
                    (num_regression_params, num_regression_params), dtype=np.float32
                )
                prior_var[upper_triangle_indices] = upper_triangle_terms
                prior_var = prior_var + prior_var.T - np.diag(np.diag(prior_var))

            study_RLalg = SmoothPosteriorSampling(
                state_feats=alg_state_feats,
                treat_feats=alg_treat_feats,
                alg_seed=alg_seed,
                lower_clip=args.lower_clip,
                upper_clip=args.upper_clip,
                steepness=args.steepness,
                action_centering=args.action_centering,
                prior_mu=prior_mean,
                prior_sigma=prior_var,
                noise_var=args.noise_var,
            )
        else:
            raise ValueError("Invalid RL Algorithm Type")

        # Run Study Simulation #######################################################
        study_df, study_RLalg = run_study_simulation(
            args, study_env, study_RLalg, user_env_data
        )

        # Save Data #################################################################
        logging.info("Saving data...")

        folder_path = os.path.join(all_folder_path, f"exp={i}")
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)

        study_df.to_csv(f"{folder_path}/data.csv", index=False)

        with open(f"{folder_path}/study_df.pkl", "wb") as f:
            pickle.dump(study_df, f)

        with open(f"{folder_path}/study_RLalg.pkl", "wb") as f:
            pickle.dump(study_RLalg, f)

        with open(f"{folder_path}/pi_args.pkl", "wb") as f:
            pickle.dump(study_RLalg.pi_args, f)

        with open(f"{folder_path}/rl_update_args.pkl", "wb") as f:
            pickle.dump(study_RLalg.rl_update_args, f)

        beta_dim = study_RLalg.get_current_beta_estimate().size
        beta_df = pd.DataFrame(
            data=np.array(
                [
                    np.concatenate(
                        # Note the plus 1 in policy num. This is just how we do
                        # things when setting up study df.
                        [np.array([i + 1]), policy["beta_est"]]
                    )
                    for i, policy in enumerate(study_RLalg.all_policies)
                ]
            ),
            columns=["policy_num", *[f"beta_{j}" for j in range(beta_dim)]],
        )
        beta_df = beta_df.astype({"policy_num": "Int64"})

        with open(f"{folder_path}/beta_df.pkl", "wb") as f:
            pickle.dump(beta_df, f)

    logger.info(
        "Simulation %d of %d ran in %.4f seconds",
        args.N,
        args.N,
        time.perf_counter() - toc2,
    )
    print(f"All simulations ran in {time.perf_counter() - tic:0.4f} seconds")

    if args.profile:
        pr.disable()
        stats = Stats(pr)
        stats.sort_stats("cumtime").print_stats(50)


def main():
    ###############################################################
    # Initialize Simulation Hyperparameters #######################
    ###############################################################
    logging.info("Parsing arguments.")
    parser = argparse.ArgumentParser(description="Generate simulation data")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=RLStudyArgs.SYNTHETIC,
        choices=[RLStudyArgs.SYNTHETIC],
    )
    parser.add_argument("--verbose", type=int, default=0, help="Prints helpful info")
    parser.add_argument(
        "--synthetic_mode",
        type=str,
        default=RLStudyArgs.DELAYED_1_ACTION_DOSAGE,
        choices=[
            RLStudyArgs.DELAYED_1_ACTION_DOSAGE,
            RLStudyArgs.DELAYED_2_ACTION_DOSAGE,
            RLStudyArgs.DELAYED_5_ACTION_DOSAGE,
            RLStudyArgs.DELAYED_1_DOSAGE_PAPER,
            RLStudyArgs.DELAYED_2_DOSAGE_PAPER,
            RLStudyArgs.DELAYED_5_DOSAGE_PAPER,
        ],
        help="File name of synthetic env params. The paper versions do not multiply dosage by the latest action to derive the reward mean.",
    )
    parser.add_argument(
        "--RL_alg",
        default=RLStudyArgs.SIGMOID_LS,
        choices=[
            RLStudyArgs.FIXED_RANDOMIZATION,
            RLStudyArgs.SIGMOID_LS,
            RLStudyArgs.SMOOTH_POSTERIOR_SAMPLING,
        ],
        help="RL algorithm used to select actions",
    )
    parser.add_argument(
        "--N", type=int, default=10, help="Number of Monte Carlo repetitions"
    )
    parser.add_argument(
        "--parallel_task_index",
        type=int,
        default=1,
        help="Identifier of task in parallel setting, needed to set different seeds",
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
        "--update_cadence_offset",
        type=int,
        default=1,
        help="If nonzero, updates will occur whenever calendar_t (mod decisions_between_updates) == update_cadence_offset",
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
        help="Comma-separated list of algorithm state features",
    )
    parser.add_argument(
        "--action_centering",
        type=int,
        default=0,
        help="Whether RL algorithm uses action centering (if applicable)",
    )
    parser.add_argument(
        "--min_update_time",
        type=int,
        default=0,
        help="The algorithm will not update before this decision time",
    )
    parser.add_argument(
        "--prior_mean",
        type=str,
        default=RLStudyArgs.NAIVE,
        help="Prior mean for posterior sampling algorithm. This is a comma-separated list of values. If 'NAIVE', then a zero vector is used.",
    )
    parser.add_argument(
        "--prior_var_upper_triangle",
        type=str,
        default=RLStudyArgs.NAIVE,
        help="Upper triangle of posterior variance for sampling algorithm. This is a comma-separated list of values, row-major, moving left to right across rows in sequence. If NAIVE, then a diagonal matrix with large values is used.",
    )
    parser.add_argument(
        "--noise_var",
        type=float,
        default=1.0,
        help="Noise variance for the Bayesian Linear Regression used with the posterior sampling algorithm. Should be scaled for environment.",
    )
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        help="If supplied, the important computations will be profiled with summary output shown",
    )
    parser.add_argument(
        "--dynamic_seeds",
        type=int,
        default=0,
        help="Whether RL simulation uses time-based vs fixed seeds",
    )
    parser.add_argument(
        "--env_seed_override",
        type=int,
        help="An optional fixed seed for the environment",
    )
    parser.add_argument(
        "--alg_seed_override",
        type=int,
        help="An optional fixed seed for the algorithm",
    )
    tmp_args = parser.parse_known_args()[0]

    if tmp_args.dataset_type == RLStudyArgs.HEARTSTEPS:
        raise NotImplementedError()
    elif tmp_args.dataset_type == RLStudyArgs.SYNTHETIC:
        default_arg_dict = {
            RLStudyArgs.T: 2,
            RLStudyArgs.RECRUIT_N: tmp_args.n,
            RLStudyArgs.RECRUIT_T: 1,
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
    else:
        raise ValueError("Invalid Dataset Type")

    parser.add_argument(
        "--T",
        type=int,
        default=default_arg_dict[RLStudyArgs.T],
        help="Total number of decision times per user",
    )
    parser.add_argument(
        "--recruit_n",
        type=int,
        default=default_arg_dict[RLStudyArgs.RECRUIT_N],
        help="Number of users recruited on each recruitment times",
    )
    parser.add_argument(
        "--recruit_t",
        type=int,
        default=default_arg_dict[RLStudyArgs.RECRUIT_T],
        help="Number of updates between recruitment times (minimum 1)",
    )

    args = parser.parse_args()
    print("Args provided to RL_Study_Simulation.py:")
    print(args)

    assert args.T >= args.decisions_between_updates

    load_data_and_simulate_studies(args, gen_feats, alg_state_feats, alg_treat_feats)


if __name__ == "__main__":
    main()
