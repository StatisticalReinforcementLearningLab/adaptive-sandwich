import numpy as np
import argparse
import config
import os
from simulator import Simulator
from MixedEffectsBanditPolicy import MixedEffectsBanditPolicy
from random_policy import RandomPolicy
from smooth_allocation import load_random_vars, get_allocation_function

RESULTS_DIR = "miwaves_sample_data/results/"

if __name__ == "__main__":
    # Take inputs from command line

    parser = argparse.ArgumentParser(
        description="Run the simulation with the Mixed Effects Bandit Policy."
    )
    parser.add_argument(
        "--num_users", "-m", type=int, default=100, help="Number of users."
    )
    parser.add_argument(
        "--num_time_steps",
        "-T",
        type=int,
        default=10,
        help="Number of decision points.",
    )
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--delta_seed",
        "-ds",
        type=int,
        default=0,
        help="Random seed for the delta generation.",
    )
    parser.add_argument(
        "--beta_mean", "-bm", type=float, default=1, help="Prior Mean beta"
    )
    parser.add_argument(
        "--beta_var",
        "-bv",
        type=float,
        default=1,
        help="Prior Sigma beta",
    )
    parser.add_argument(
        "--gamma_var",
        "-gv",
        type=float,
        default=0.1,
        help="Prior Sigma gamma",
    )
    parser.add_argument(
        "--sigma_e2", "-se", type=float, default=0.1, help="Noise variance"
    )
    parser.add_argument(
        "--policy_type",
        "-p",
        type=str,
        default="random",
        help="Policy type: mixed_effects or random",
    )
    parser.add_argument(
        "--save_dir",
        "-d",
        type=str,
        help="Location to save the results. Needed for runs on cluster.",
    )
    args = parser.parse_args()

    # Run the simulation with the random policy and a fixed seed
    seed = args.seed
    delta_seed = args.delta_seed

    # Generate delta
    rng = np.random.default_rng(delta_seed)
    # Delta_t = rng.uniform(-1, 1, args.num_time_steps)
    Delta_t = 1 * np.ones(args.num_time_steps)
    Xi_beta = rng.uniform(-1, 1, args.num_time_steps)
    Xi_gamma = rng.uniform(-1, 1, args.num_time_steps)
    xi_beta = rng.uniform(-1, 1, args.num_time_steps)
    xi_gamma = rng.uniform(-1, 1, args.num_time_steps)

    # Get the rho i.e. allocation function
    RANDOMVARS_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../data/randomvars.pkl")
    )
    rho_func = get_allocation_function(
        func_type=config.ALLOCATION_FUNC,
        B=config.B,
        randomvars=load_random_vars(RANDOMVARS_PATH),
        C=config.C,
        L_min=config.LMIN,
        L_max=config.LMAX,
    )

    print("Delta_t: ", Delta_t)
    print("Mean delta: ", np.mean(Delta_t))

    sim = Simulator(
        num_users=args.num_users,
        num_time_steps=args.num_time_steps,
        seed=seed,
        Delta_t=Delta_t,
    )

    args.beta_mean = [args.beta_mean]
    args.beta_var = [[args.beta_var]]
    args.gamma_var = [[args.gamma_var]]

    exp_name = (
        "num_users"
        + str(args.num_users)
        + "_num_time_steps"
        + str(args.num_time_steps)
        + "_seed"
        + str(seed)
        + "_delta_seed"
        + str(delta_seed)
        + "_beta_mean"
        + str(args.beta_mean)
        + "_beta_var"
        + str(args.beta_var)
        + "_gamma_var"
        + str(args.gamma_var)
        + "_sigma_e2"
        + str(args.sigma_e2)
        + "_policy_type"
        + str(args.policy_type)
    )

    fullpath = (args.save_dir or RESULTS_DIR + exp_name) + "/"

    # Create the results directory if it does not exist
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    # Create the experiment directory if it does not exist
    if not os.path.exists(fullpath):
        os.makedirs(fullpath)

    if args.policy_type == "random":
        policy = RandomPolicy(seed=seed)

    elif args.policy_type == "mixed_effects":

        # Reformat the input arguments
        args.beta_mean = np.array(args.beta_mean).reshape(-1, 1)
        args.beta_var = np.array(args.beta_var).reshape(
            args.beta_mean.shape[0], args.beta_mean.shape[0]
        )
        args.gamma_var = np.array(args.gamma_var).reshape(
            args.beta_mean.shape[0], args.beta_mean.shape[0]
        )

        print(args)

        policy = MixedEffectsBanditPolicy(
            m=args.num_users,
            T=args.num_time_steps,
            mu_beta=args.beta_mean,
            Sigma_beta=args.beta_var,
            Sigma_gamma=args.gamma_var,
            sigma_e2=args.sigma_e2,
            rho_func=rho_func,
            Xi_beta=Xi_beta,
            Xi_gamma=Xi_gamma,
            xi_beta=xi_beta,
            xi_gamma=xi_gamma,
            output_path=fullpath,
            seed=seed,
        )

    else:
        raise ValueError("Invalid policy type.")

    df_results = sim.run_simulation(policy)

    # Display the first few rows of the results
    print(df_results)
