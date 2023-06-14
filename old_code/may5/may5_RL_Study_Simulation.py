import argparse
import numpy as np
import pandas as pd
import math
import csv
import pickle as pkl
import time
import json
import os

from synthetic_env import load_synthetic_env, SyntheticEnv
from oralytics_env import load_oralytics_env, OralyticsEnv
from debug_helper import output_variance_pieces

from basic_RL_algorithms import FixedRandomization, SigmoidLS
from smooth_posterior_sampling import SmoothPosteriorSampling
#from posterior_sampling_RL_algorithm import PosteriorSampling


###############################################################
# Initialize Simulation Hyperparameters #######################
###############################################################

parser = argparse.ArgumentParser(description='Generate simulation data')
parser.add_argument('--dataset_type', type=str, default='synthetic',
                    choices=['heartsteps', 'synthetic', 'oralytics'])
parser.add_argument('--verbose', type=int, default=0,
                    help='Prints helpful info')
parser.add_argument('--heartsteps_mode', default='medium',
                    choices=['evalSim', 'realistic', 'medium', 'easy'],
                    help='Sets default parameter values accordingly')
parser.add_argument('--synthetic_mode', type=str, default='delayed_effects',
                    help='File name of synthetic env params')
parser.add_argument('--RL_alg', default='sigmoid_LS',
                    choices=["fixed_randomization", "sigmoid_LS", "posterior_sampling"],
                    help='RL algorithm used to select actions')
parser.add_argument('--N', type=int, default=10,
                    help='Number of Monte Carlo repetitions')
parser.add_argument('--n', type=int, default=90, help='Total number of users')
parser.add_argument('--upper_clip', type=float, default=0.9,
                    help='Upper action selection probability constraint')
parser.add_argument('--lower_clip', type=float, default=0.1,
                    help='Lower action selection probability constraint')
parser.add_argument('--fixed_action_prob', type=float, default=0.5,
                    help='Used if not using learning alg to select actions')
parser.add_argument('--min_users', type=int, default=25,
                    help='Min number of users needed to update alg')
parser.add_argument('--err_corr', default='time_corr',
                    choices=["time_corr", "independent"],
                    help='Noise error correlation structure')
parser.add_argument('--decisions_between_updates', type=int, default=1,
                    help='Number of decision times beween algorithm updates')
parser.add_argument('--save_dir', type=str, default=".",
                    help='Directory to save all results in')
parser.add_argument('--steepness', type=float, default=10,
                    help='Allocation steepness')
parser.add_argument('--alg_state_feats', type=str, default="intercept",
                    help='Comma separated list of algorithm state features')
parser.add_argument('--action_centering', type=int, default=0,
                    help='Whether posterior sampling algorithm uses action centering')
parser.add_argument('--prior', type=str, default="naive",
                    choices=["naive", "oralytics"],
                    help='Prior for posterior sampling algorithm')
tmp_args = parser.parse_known_args()[0]

if tmp_args.dataset_type == 'heartsteps':
    raise ValueError("Not implemented")
elif tmp_args.dataset_type == 'synthetic':
    arg_dict = { "T" : 2, 'recruit_n': tmp_args.n, 'recruit_t': 1, 
                "allocation_sigma": 1, "noise_var": 1}
   
    # Algorithm state features
    alg_state_feats = tmp_args.alg_state_feats.split(",")
    alg_treat_feats = tmp_args.alg_state_feats.split(",")

    """
    past_action_len = 1
    past_action_cols = ['intercept'] + \
            ['past_action_{}'.format(i) for i in range(1,past_action_len+1)]
    past_reward_action_cols = ['past_reward'] + \
            ['past_action_{}_reward'.format(i) for i in range(1,past_action_len+1)]
    gen_feats = past_action_cols + past_reward_action_cols
    """
   
    # Generation features
    past_action_len = 1
    past_action_cols = ['intercept'] + \
            ['past_action_{}'.format(i) for i in range(1,past_action_len+1)]
    past_reward_action_cols = ['past_reward'] + \
            ['past_action_{}_reward'.format(i) for i in range(1,past_action_len+1)]
    gen_feats = past_action_cols + past_reward_action_cols + ['dosage']
   
    """
    alg_state_feats = ['intercept']
    alg_treat_feats = ['intercept']

    past_action_len = 8
    past_action_cols = ['past_action_{}'.format(i) for i in range(1,past_action_len+1)]
    past_action_cols.reverse()
    gen_feats = past_action_cols

    gen_feats = gen_feats
    gen_feats_reward = ['reward_'+x for x in gen_feats]
    """

    
elif tmp_args.dataset_type == 'oralytics':
    arg_dict = { "T" : 50, 'recruit_n': tmp_args.n, 'recruit_t': 1, 
                "allocation_sigma": 1, "noise_var": 1 }

    #allocation_sigma: 163 (truncated brush times); 5.7 (square-root of truncatred brush times)

    #alg_state_feats = ['intercept', 'time_of_day', 'prev_brush', 'prev_message', 'weekend']
    #alg_treat_feats = ['intercept', 'time_of_day', 'prev_brush', 'prev_message']
    
    alg_state_feats = ['intercept', 'time_of_day', 'prior_day_brush']
    alg_treat_feats = ['intercept', 'time_of_day', 'prior_day_brush']
    
    #alg_state_feats = ['intercept']
    #alg_treat_feats = ['intercept']
    
else:
    raise ValueError()

parser.add_argument('--T', type=int, default=arg_dict['T'],
                    help='Total number of decision times per user')
parser.add_argument('--recruit_n', type=int, default=arg_dict['recruit_n'],
                    help='Number of users recruited on each recruitment times')
parser.add_argument('--recruit_t', type=int, default=arg_dict['recruit_t'],
                    help='Number of updates between recruitment times (minmum 1)')
parser.add_argument('--allocation_sigma', type=float, default=arg_dict['allocation_sigma'],
                    help='Sigma used in allocation of algorithm')
parser.add_argument('--noise_var', type=float, default=arg_dict['allocation_sigma'],
                    help='Posterior sampling noise variance')

args = parser.parse_args()
print(vars(args))

assert args.T >= args.decisions_between_updates


###############################################################
# Load Data and Models ########################################
###############################################################

if args.dataset_type == 'heartsteps':
    #user_env_data, env_params = load_heartsteps_env()
    raise ValueError("Not implemented")

elif args.dataset_type == 'synthetic':
    user_env_data = None
    paramf_path= "./synthetic_env_params/{}.txt".format(args.synthetic_mode)
    env_params = load_synthetic_env(paramf_path)
    if len(env_params.shape) == 2:
        assert env_params.shape[0] >= args.T

elif args.dataset_type == 'oralytics':
    paramf_path= "./oralytics_env_params/non_stat_zero_infl_pois_model_params.csv"
    param_names, bern_params, poisson_params = load_oralytics_env(paramf_path)
    treat_feats = ['intercept', 'time_of_day', 'weekend', 'day_in_study_norm', 'prior_day_brush']
    
    user_env_data = {
        'bern_params' : bern_params,
        'poisson_params' : poisson_params
    }
    
else:
    raise ValueError("Invalid Dataset Type")

    
###############################################################
# Simulation Functions ########################################
###############################################################

def run_study_simulation(study_env, study_RLalg):
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
    for t in range(1, study_env.calendar_T+1):

        # Check if need to update algorithm #######################################
        if t > 1 and (t-1) % args.decisions_between_updates == 0 and args.RL_alg != 'fixed_randomization':
            # check enough avail data and users; if so, update algorithm
            most_recent_policy_t = study_RLalg.all_policies[-1]["policy_last_t"]
            new_obs_bool = np.logical_and( study_df['calendar_t'] < t,
                                    study_df['calendar_t'] > most_recent_policy_t)
            new_update_data = study_df[ new_obs_bool ]
            
            all_prev_data = study_df[ study_df['calendar_t'] < t ]
            #print(all_prev_data.shape, new_update_data.shape, t)
            
            if args.dataset_type == 'heartsteps':
                num_avail = np.sum(new_update_data['availability'])
            else:
                num_avail = 1
            prev_num_users = len(study_df[ study_df['calendar_t'] == t-1 ])

            if num_avail > 0 and prev_num_users >= args.min_users:
                # Update Algorithm ##############################################
                study_RLalg.update_alg(new_update_data, t)
                #print("update", t)
        
        if args.RL_alg != 'fixed_randomization':
            # Update study_df with info on latest policy used to select actions
            study_df.loc[ study_df['calendar_t'] == t, 'policy_last_t'] = \
                                                study_RLalg.all_policies[-1]["policy_last_t"]
            study_df.loc[ study_df['calendar_t'] == t, 'policy_num'] = \
                                                len(study_RLalg.all_policies)
        else:
            study_df.loc[ study_df['calendar_t'] == t, 'policy_last_t'] = 0
            study_df.loc[ study_df['calendar_t'] == t, 'policy_num'] = 0

        curr_timestep_data = study_df[ study_df['calendar_t'] == t ]
  
        
        # Sample Actions #####################################################
        action_probs = study_RLalg.get_action_probs(curr_timestep_data, 
                                        filter_keyval = ('calendar_t', t) )
    
        """
        curr_timestep_data.to_csv( 'test_curr_timestep_data.csv', 
            index=False) # compression={'method': None, 
                         #             'float_precision': 'high'} )
        with open('test_study_RLalg.pkl', 'wb') as f:
            pkl.dump(study_RLalg, f)

        curr_timestep_data2 = pd.read_csv( 'test_curr_timestep_data.csv')
                                          #float_precision="high" )
        with open('test_study_RLalg.pkl', 'rb') as f:
            study_RLalg2 = pkl.load(f)

        action_probs2 = study_RLalg2.get_action_probs_inner(
                curr_timestep_data = curr_timestep_data2,
                beta_est = study_RLalg2.all_policies[-1]['est_params'], 
                n_users = args.n,
                intercept_val = study_RLalg2.all_policies[-1]['intercept_val'],
                filter_keyval = ('calendar_t', t) )

        #action_probs2 = study_RLalg.get_action_probs_inner(
        #        curr_timestep_data = curr_timestep_data,
        #        beta_est = study_RLalg.all_policies[-1]['est_params'], 
        #        n_users = args.n,
        #        intercept_val = study_RLalg.all_policies[-1]['intercept_val'],
        #        filter_keyval = ('calendar_t', t) )
                #filter_keyval = ('policy_last_t', policy_last_t))
        
        #action_probs2 = study_RLalg.get_action_probs(curr_timestep_data, 
        #                                filter_keyval = ('calendar_t', t) )

        try:
            assert (action_probs/action_probs2 == 1).all()
        except:
            print("yo")
            action_probs3 = study_RLalg2.get_action_probs_inner(
                    curr_timestep_data = curr_timestep_data,
                    beta_est = study_RLalg2.all_policies[-1]['est_params'], 
                    n_users = args.n,
                    intercept_val = study_RLalg2.all_policies[-1]['intercept_val'],
                    filter_keyval = ('calendar_t', t) )
            print( (action_probs3/action_probs == 1).all() ) 
            import ipdb; ipdb.set_trace()
        """

        if args.dataset_type == 'heartsteps':
            action_probs *= curr_timestep_data['availability']
        actions = study_RLalg.rng.binomial(1, action_probs)
        
        # Sample Rewards #####################################################
        if args.dataset_type == 'oralytics':
            rewards, brush_times = study_env.sample_rewards(curr_timestep_data, actions, t)
        else:
            rewards = study_env.sample_rewards(curr_timestep_data, actions, t)

        # Record all collected data #######################################
        if args.dataset_type == 'oralytics':
            fill_columns = ['reward', 'brush_time', 'action', 'action1prob']
            fill_vals = np.vstack( [rewards, brush_times, actions, action_probs] ).T
            study_df.loc[ study_df['calendar_t'] == t, fill_columns] = fill_vals
        else:
            fill_columns = ['reward', 'action', 'action1prob']
            fill_vals = np.vstack( [rewards, actions, action_probs] ).T
            study_df.loc[ study_df['calendar_t'] == t, fill_columns] = fill_vals
        
        """
        # Check weights are 1
        if t > 1:
            new_curr_timestep_data = study_df[ study_df['calendar_t'] == t ]
            all_user_ids = np.unique( study_df['user_id'].to_numpy() )
            weights = study_RLalg.get_weights(new_curr_timestep_data,
                    beta_params = study_RLalg.all_policies[-1]['est_params'],
                    all_user_ids = all_user_ids,
                    intercept_val = study_RLalg.all_policies[-1]['intercept_val'])
        """

        if t < study_env.calendar_T:
            # Record data to prepare for state at next decision time
            current_users = study_df[ study_df['calendar_t'] == t ]['user_id']
            study_df = study_env.update_study_df(study_df, t)
           
            #print(study_df[ study_df['calendar_t'] == t+1 ])
            #print(t)
       

    if args.RL_alg == 'posterior_sampling':
        # fill in some columns
        study_RLalg.norm_samples_df['policy_last_t'] = study_df['policy_last_t']
        study_RLalg.norm_samples_df['policy_num'] = study_df['policy_num']

    """
    # TESTING ##############################
    study_df.to_csv( 'test_data.csv', index=False, compression={'method': None,
                                      'float_precision': 'high'} )
    with open('test_study_RLalg.pkl', 'wb') as f:
        pkl.dump(study_RLalg, f)

    study_df2 = pd.read_csv( 'test_data.csv', float_precision="high")
    with open('test_study_RLalg.pkl', 'rb') as f:
        study_RLalg2 = pkl.load(f)

    all_user_ids = np.unique( study_df2['user_id'].to_numpy() )
    for policy_num, curr_policy_dict in enumerate(study_RLalg2.all_policies):
        policy_last_t = curr_policy_dict['policy_last_t']
        if policy_last_t == 0:
            continue
        
        if args.RL_alg == "sigmoid_LS":
            est_params = curr_policy_dict['est_params'].to_numpy().squeeze()
        elif args.RL_alg == "posterior_sampling":
            est_params = curr_policy_dict['est_params']
        else:
            raise ValueError("RL algorithm")

        #beta_est = est_list[policy_num-1]
        beta_est = est_params

        curr_policy_decision_data = study_df2[ study_df2['policy_last_t'] == policy_last_t ]
            # curr_policy_decision_data = all data collected using the policy policy_last_t

        user_pi_weights = study_RLalg2.get_weights(curr_policy_decision_data,
                                    beta_est, all_user_ids=all_user_ids,
                                    intercept_val = curr_policy_dict['intercept_val'],
                                    filter_keyval = ('policy_last_t', policy_last_t) )

        # Check that reproduced the action selection probabilities correctly
        try:
            assert np.all(user_pi_weights == 1)
            #assert np.all(np.around(user_pi_weights,10) == 1)
            #assert np.mean(np.absolute(user_pi_weights - 1)) < 0.01
        except:
            print("uh oh")
            import ipdb; ipdb.set_trace()

    import ipdb; ipdb.set_trace()
    """
    return study_df, study_RLalg



###############################################################
# Simulate Studies ############################################
###############################################################

tic = time.perf_counter()

if args.dataset_type == "heartsteps":
    mode = args.heartsteps_mode
elif args.dataset_type == "synthetic":
    mode = args.synthetic_mode
    #assert args.recruit_n == args.n
elif args.dataset_type == "oralytics":
    mode = None
else:
    raise ValueError("Invalid dataset type")

print("Running simulations...")
if args.dataset_type == 'oralytics':
    exp_str = '{}_alg={}_T={}_n={}_recruitN={}_decisionsBtwnUpdates={}_steep={}_actionC={}'.format(
            args.dataset_type, args.RL_alg, args.T, args.n, 
            args.recruit_n, args.decisions_between_updates, args.steepness, args.action_centering)
else:
    exp_str = '{}_mode={}_alg={}_T={}_n={}_recruitN={}_decisionsBtwnUpdates={}_steepness={}_algfeats={}_errcorr={}_actionC={}'.format(
            args.dataset_type, mode, args.RL_alg, args.T, args.n, args.recruit_n, args.decisions_between_updates,
            args.steepness, args.alg_state_feats, args.err_corr, args.action_centering)

simulation_data_path = os.path.join(args.save_dir, "simulated_data")
if not os.path.isdir(simulation_data_path):
    os.mkdir(simulation_data_path)
all_folder_path = os.path.join(simulation_data_path, exp_str)
if not os.path.isdir(all_folder_path):
    os.mkdir(all_folder_path)
    
with open(os.path.join(all_folder_path, "args.json"), "w") as f:
    json.dump(vars(args), f)
    
policy_grad_norm = []
for i in range(1,args.N+1):
    
    env_seed = i*5000
    alg_seed = (args.N+i)*5000

    if i == 10 or i % 25 == 0:
        toc = time.perf_counter()
        print(f"{i} ran in {toc - tic:0.4f} seconds")

        
    # Initialize study environment ############################################
    if args.dataset_type == "heartsteps":
        #study_env = HeartstepStudyEnv(args, env_params, gen_feats)
        raise ValueError("Not implemented")
    elif args.dataset_type == "synthetic":
        study_env = SyntheticEnv(args, env_params, env_seed=env_seed, 
                gen_feats=gen_feats, err_corr=args.err_corr)
    elif args.dataset_type == "oralytics":
        study_env = OralyticsEnv(args, param_names, bern_params, 
                                 poisson_params, env_seed=env_seed)
    else:
        raise ValueError("Invalid Dataset Type")
        
        
    # Initialize RL algorithm ###################################################
    if args.RL_alg == "fixed_randomization":
        study_RLalg = FixedRandomization(args, alg_state_feats, alg_treat_feats, alg_seed=alg_seed)
    elif args.RL_alg == "sigmoid_LS":
        study_RLalg = SigmoidLS(args, alg_state_feats, alg_treat_feats, 
                                allocation_sigma=args.allocation_sigma, alg_seed=alg_seed,
                                steepness=args.steepness)
    elif args.RL_alg == "posterior_sampling":
        if args.prior == "naive":
            if args.action_centering:
                total_dim = len(alg_state_feats) + len(alg_treat_feats)*2
                prior_mean = np.ones(total_dim)*0.1
                prior_var = np.eye(total_dim)*2
            else:
                total_dim = len(alg_state_feats) + len(alg_treat_feats)
                prior_mean = np.ones(total_dim)*0.1
                prior_var = np.eye(total_dim)*2
        else:
            raise ValueError("Invalid prior type: {}".format(args.prior))
        study_RLalg = SmoothPosteriorSampling(args, alg_state_feats, 
                                              alg_treat_feats, 
                                              alg_seed=alg_seed, 
                                              allocation_sigma=args.allocation_sigma,
                                              steepness=args.steepness,
                                              prior_mean=prior_mean,
                                              prior_var=prior_var,
                                              noise_var=args.noise_var,
                                              action_centering=args.action_centering)
    else:
        raise ValueError("Invalid RL Algorithm Type")
        
        
    # Run Study Simulation #######################################################
    study_df, study_RLalg = run_study_simulation(study_env, study_RLalg)

    # Print summary statistics
    if i == 25 and args.RL_alg != 'fixed_randomization':
        print("\nTotal Update Times: {}".format( len(study_RLalg.all_policies)-1 ) )
        study_df.action = study_df.action.astype('int')
        study_df.policy_last_t = study_df.policy_last_t.astype('int')
        if args.dataset_type == 'heartsteps':
            study_df.stepcount = (np.exp(study_df.reward)-0.5).astype('int')

    # Make histogram of rewards (available)
    if i == 0:
        print(study_df.head())
        if args.dataset_type == 'heartsteps' and args.mode == 'evalSim':
            # Make histograms of step counts for real data vs simulated data
            study_env.make_stepcount_hist(args.env_path, user_env_data, study_df)

    # Save Data #################################################################
    if args.verbose:
        print("Saving data...")
   
    folder_path = os.path.join(all_folder_path, "exp={}".format(i))
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    if args.RL_alg != 'fixed_randomization':
        study_df = study_df.astype({'policy_num': 'int32', 
                                    "policy_last_t": 'int32', "action": 'int32'})
    
    study_df.to_csv( '{}/data.csv'.format(folder_path), index=False )
    with open('{}/study_df.pkl'.format(folder_path), 'wb') as f:
        pkl.dump(study_df, f)

    with open('{}/study_RLalg.pkl'.format(folder_path), 'wb') as f:
        pkl.dump(study_RLalg, f)




    # TODO eventually removeSave Variance Components ##################################################
    if args.RL_alg == 'sigmoid_LS':
        out_dict = output_variance_pieces(study_df, study_RLalg, args)
        with open('{}/out_dict.pkl'.format(folder_path), 'wb') as file:
            pkl.dump( out_dict, file )

        policy_grad_norm.append( np.max( np.absolute([ y['pi_grads'] for x, y in out_dict.items() ]) ) )


toc = time.perf_counter()
print(f"Final ran in {toc - tic:0.4f} seconds")

