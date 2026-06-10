import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
import sys
from sklearn.linear_model import LinearRegression

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--alg', type=str, default='TS', help="algorithm name", choices=['TS', 'SAC'])
parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
parser.add_argument('--variance_rewards', type=int, default=0, help="evaluate the reward variance to intialize b")
parser.add_argument('--evaluate', type=int, default=0, help="0: fixed true values after evaluation, 1: evaluate the true thetahat on a large n, 2: evaluate on the current n")
parser.add_argument('--habituation', type=int, default=1, help="1: high, 6 low")
parser.add_argument('--treatment', type=int, default=2, help="0:none, 1: low, 2: high")
parser.add_argument('--steepness', type=float, default=5.0, help="steepness b")
args = parser.parse_args()
n=args.n
T=args.T

Variance_rewards = args.variance_rewards

N_seed = args.N_seed
algo_name = 'smooth_posterior_sampling' if args.alg == 'TS' else 'sac'

n_true = 10000 # the true theta should be evaluated on a large n

b=1
# true variance

path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
# path = ''

if Variance_rewards:
    rewards_list_initialization = []
    residuals_list_initialization = []
    for i in tqdm(range(N_seed)):
        subpath = path + f"Miwaves_TS_n42_T60/{i}/simulated_data/miwaves_alg=fixed_randomization_T=60_n=42_decisionBtwnUpdates=1_averagerewards/exp=1"
        data = pd.read_csv(subpath+'/data.csv')
        rewards_list_initialization.append(data['reward'].mean())
        data['S1A'] = data['S1'] * data['action']
        data['S2A'] = data['S2'] * data['action']
        data['S3A'] = data['S3'] * data['action']
        df_X = data[['intercept','S1','S2','S3','action','S1A','S2A','S3A']]
        df_Y = data['reward']
        linear_model_single = LinearRegression(fit_intercept=False)
        linear_model_single.fit(df_X, df_Y)
        y_pred = linear_model_single.predict(df_X)
        residuals = df_Y - y_pred
        residuals_list_initialization.append(np.std(residuals))
    reward_mean = np.mean(rewards_list_initialization)
    # reward_std = np.std(np.array(rewards_list_initialization))
    # print(rewards_list_initialization)
    mean_std_residuals = np.mean(residuals_list_initialization)
    print('mean rewards: ', reward_mean)
    print('std of reward residuals: ', mean_std_residuals)
    print(f'b: 20/{mean_std_residuals:.3f}= {20/mean_std_residuals:.3f}, 10/{mean_std_residuals:.3f}= {10/mean_std_residuals:.3f}')
    sys.exit()

    
if args.evaluate > 0: # n = 10000
    n_true = args.n if args.evaluate == 2 else n_true # 1: use n=10000 as the true reward, 2: use the reward evaluated on n as the true reward
    rewards_list_true = []
    for i in tqdm(range(N_seed)):
        if args.alg == 'TS': # old path
            subpath = path + f"Miwaves_TS_n{n_true}_T{T}/{i}/decisionBtwnUpdates=1_habituation={args.habituation}_treatment={args.treatment}_steepness={args.steepness}_averagerewards/exp=1"
        else: # SAC
            subpath = path + f"Miwaves_sac_n{n_true}_T{T}/{i}/decisionBtwnUpdates=1_habituation={args.habituation}_treatment={args.treatment}_steepness={args.steepness}_averagerewards/exp=1"
        data = pd.read_csv(subpath+'/data.csv')
        rewards_list_true.append(data['reward'].mean())

    true_reward_mean = np.mean(rewards_list_true)
    true_reward_median = np.median(rewards_list_true)
    true_var = np.var(np.array(rewards_list_true)) 
else: # 0
    if args.alg == 'TS': # TOBE UPDATED
        if args.habituation == 1 and args.treatment == 2: # high habituation, high treatment effect
            if args.steepness == 5.0:
                true_reward_mean = 1.944902 # 2.106088
                true_reward_median = 1.944872 # 2.106066
                true_var = 0.000023 # 0.000014
            elif args.steepness == 3.0:
                true_reward_mean = 1.964477
                true_reward_median = 1.964489
                true_var = 0.000021
        elif args.habituation == 1 and args.treatment == 1: # high habituation, low treatment effect
            if args.steepness == 5.0:
                true_reward_mean = 1.926816
                true_reward_median = 1.926825
                true_var = 0.000024
            elif args.steepness == 3.0: # our focus in the paper
                true_reward_mean = 1.941743 # 1.941743
                true_reward_median = 1.941744 # 1.941744
                true_var = 0.000022 # 0.000022
        elif args.habituation == 6 and args.treatment == 1: # low habituation, low treatment effect
            if args.steepness == 5.0:
                true_reward_mean = 2.105260
                true_reward_median = 2.105208
                true_var = 0.000014
            elif args.steepness == 3.0: 
                true_reward_mean = 2.105013
                true_reward_median = 2.104938
                true_var = 0.000014
        else:
            raise NotImplementedError
    elif args.alg == 'SAC':
        true_reward_mean = None
        true_reward_median = None
        true_var = None
    else:
        raise NotImplementedError


######################################## Step 2: access all replications, compute all the classical and adaptive variance estimates. 
theta_hat_list = []
classical_var_list = []
adaptive_var_list = []

count = 0
rewards_list = []
for i in tqdm(range(N_seed)):
    if args.alg == 'TS': # old path
        subpath = path + f"Miwaves_TS_n{n}_T{T}/{i}/decisionBtwnUpdates=1_habituation={args.habituation}_treatment={args.treatment}_steepness={args.steepness}_averagerewards/exp=1"
    else:
        subpath = path + f"Miwaves_sac_n{n}_T{T}/{i}/decisionBtwnUpdates=1_habituation={args.habituation}_treatment={args.treatment}_steepness={args.steepness}_averagerewards/exp=1"
    try:
        RL_data = pd.read_csv(subpath+'/data.csv')
        rewards_list.append(RL_data['reward'].mean())
        data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
        theta_hat_list.append(data['theta_est'].item())
        classical_var_list.append(data['classical_sandwich_var_estimate'].item())
        adaptive_var_list.append(data['adaptive_sandwich_var_estimate'].item())
        count += 1
    except:
        continue

reward_mean = np.mean(rewards_list)
reward_median = np.median(rewards_list)
reward_variance = np.var(np.array(rewards_list))


print(f'Evaluation on the average reward inference when n={n}')
print(f'true mean of thetahat when n={n_true}: {true_reward_mean:.6f}')
print(f'true median of thetahat when n={n_true}: {true_reward_median:.6f}')
print(f'true variance of thetahat when n={n_true}: {true_var:.6f}')
print(f'number of successful experiments: {count}/{N_seed}')
print(f'mean of thetahat: {reward_mean:.6f}')
print(f'median of thetahat: {reward_median:.6f}')
print(f'variance of thetahat: {reward_variance:.6f}')
print(f'mean of classical variance estimate: {np.mean(classical_var_list, axis=0):.6f}')
print(f'median of classical variance estimate: {np.median(classical_var_list, axis=0):.6f}')
print(f'mean of adaptive variance estimate: {np.mean(adaptive_var_list, axis=0):.6f}' )
print(f'median of adaptive variance estimate: {np.median(adaptive_var_list, axis=0):.6f}' )



######################################## Step 3: compute the coverage rates 

def Coverage(theta_hat, variance, true_mean):
    # confidence interval: theta_hat +- 1.96 * sqrt(variance) / n
    count = 0
    CI_width = 1.96 * np.sqrt(variance) # / np.sqrt(n), the variance here is var/n
    low_bound = theta_hat - CI_width
    up_bound = theta_hat + CI_width
    if low_bound <= true_mean and up_bound >= true_mean:
        count += 1
    return count

l_classical_covered, l_adaptive_covered = [],[]
for i in range(len(theta_hat_list)):
    num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_reward_mean)
    num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_reward_mean)
    l_classical_covered.append(num_classical_covered)
    l_adaptive_covered.append(num_adaptive_covered)

l_classical_covered = np.array(l_classical_covered)
l_adaptive_covered = np.array(l_adaptive_covered)

print(f'coverage rate of classical variance estiamte: {l_classical_covered.mean():.6f} / std errors: {l_classical_covered.std() / np.sqrt(count):.6f}')
print(f'coverage rate of adaptive variance estiamte: {l_adaptive_covered.mean():.6f} / std errors: {l_adaptive_covered.std() / np.sqrt(count):.6f}')
print(args)



############################################################# results: TS  #############################################################

################################ high habituation, low treatment effect=1, b=3.0
"""
######### n=20
true mean of thetahat when n=10000: 1.941743
true median of thetahat when n=10000: 1.941744
true variance of thetahat when n=10000: 0.000022
number of successful experiments: 1000/1000
mean of thetahat: 1.941898
median of thetahat: 1.944167
variance of thetahat: 0.010290
mean of classical variance estimate: 0.009187
median of classical variance estimate: 0.008675
mean of adaptive variance estimate: 0.016171
median of adaptive variance estimate: 0.012991
coverage rate of classical variance estiamte: 0.923000 / std errors: 0.008430
coverage rate of adaptive variance estiamte: 0.951000 / std errors: 0.006826


######### n=30
true mean of thetahat when n=10000: 1.941743
true median of thetahat when n=10000: 1.941744
true variance of thetahat when n=10000: 0.000022
number of successful experiments: 1000/1000
mean of thetahat: 1.939479
median of thetahat: 1.938611
variance of thetahat: 0.006710
mean of classical variance estimate: 0.006315
median of classical variance estimate: 0.006149
mean of adaptive variance estimate: 0.009701
median of adaptive variance estimate: 0.008566
coverage rate of classical variance estiamte: 0.925000 / std errors: 0.008329
coverage rate of adaptive variance estiamte: 0.951000 / std errors: 0.006826

######### n=50
true mean of thetahat when n=10000: 1.941743
true median of thetahat when n=10000: 1.941744
true variance of thetahat when n=10000: 0.000022
number of successful experiments: 1000/1000
mean of thetahat: 1.941490
median of thetahat: 1.942667
variance of thetahat: 0.003789
mean of classical variance estimate: 0.003848
median of classical variance estimate: 0.003764
mean of adaptive variance estimate: 0.005493
median of adaptive variance estimate: 0.005121
coverage rate of classical variance estiamte: 0.934000 / std errors: 0.007851
coverage rate of adaptive variance estiamte: 0.965000 / std errors: 0.005812

######### n=75
true mean of thetahat when n=10000: 1.941743
true median of thetahat when n=10000: 1.941744
true variance of thetahat when n=10000: 0.000022
number of successful experiments: 1000/1000
mean of thetahat: 1.942439
median of thetahat: 1.941889
variance of thetahat: 0.002668
mean of classical variance estimate: 0.002553
median of classical variance estimate: 0.002535
mean of adaptive variance estimate: 0.003426
median of adaptive variance estimate: 0.003202
coverage rate of classical variance estiamte: 0.940000 / std errors: 0.007510
coverage rate of adaptive variance estiamte: 0.962000 / std errors: 0.006046

######### n=100
true mean of thetahat when n=10000: 1.941743
true median of thetahat when n=10000: 1.941744
true variance of thetahat when n=10000: 0.000022
number of successful experiments: 1000/1000
mean of thetahat: 1.942310
median of thetahat: 1.943167
variance of thetahat: 0.001929
mean of classical variance estimate: 0.001925
median of classical variance estimate: 0.001907
mean of adaptive variance estimate: 0.002482
median of adaptive variance estimate: 0.002370
coverage rate of classical variance estiamte: 0.953000 / std errors: 0.006693
coverage rate of adaptive variance estiamte: 0.971000 / std errors: 0.005307

######### n=300


"""

################################ high habituation, high treatment effect, b=3.0
"""
######### n=20
true mean of thetahat when n=10000: 1.964477
true median of thetahat when n=10000: 1.964489
true variance of thetahat when n=10000: 0.000021
number of successful experiments: 1000/1000
mean of thetahat: 1.967692
median of thetahat: 1.972500
variance of thetahat: 0.009845
mean of classical variance estimate: 0.008820
median of classical variance estimate: 0.008319
mean of adaptive variance estimate: 0.016693
median of adaptive variance estimate: 0.012537
coverage rate of classical variance estiamte: 0.918000 / std errors: 0.008676
coverage rate of adaptive variance estiamte: 0.946000 / std errors: 0.007147


######### n=30
true mean of thetahat when n=10000: 1.964477
true median of thetahat when n=10000: 1.964489
true variance of thetahat when n=10000: 0.000021
number of successful experiments: 1000/1000
mean of thetahat: 1.964551
median of thetahat: 1.963889
variance of thetahat: 0.006371
mean of classical variance estimate: 0.006080
median of classical variance estimate: 0.005931
mean of adaptive variance estimate: 0.010139
median of adaptive variance estimate: 0.008629
coverage rate of classical variance estiamte: 0.924000 / std errors: 0.008380
coverage rate of adaptive variance estiamte: 0.958000 / std errors: 0.006343

######### n=50
true mean of thetahat when n=10000: 1.964477
true median of thetahat when n=10000: 1.964489
true variance of thetahat when n=10000: 0.000021
number of successful experiments: 1000/1000
mean of thetahat: 1.965792
median of thetahat: 1.966333
variance of thetahat: 0.003579
mean of classical variance estimate: 0.003710
median of classical variance estimate: 0.003646
mean of adaptive variance estimate: 0.005562
median of adaptive variance estimate: 0.004991
coverage rate of classical variance estiamte: 0.937000 / std errors: 0.007683
coverage rate of adaptive variance estiamte: 0.964000 / std errors: 0.005891

######### n=75
true mean of thetahat when n=10000: 1.964477
true median of thetahat when n=10000: 1.964489
true variance of thetahat when n=10000: 0.000021
number of successful experiments: 1000/1000
mean of thetahat: 1.966214
median of thetahat: 1.964222
variance of thetahat: 0.002552
mean of classical variance estimate: 0.002463
median of classical variance estimate: 0.002431
mean of adaptive variance estimate: 0.003397
median of adaptive variance estimate: 0.003166
coverage rate of classical variance estiamte: 0.935000 / std errors: 0.007796
coverage rate of adaptive variance estiamte: 0.960000 / std errors: 0.006197

######### n=100
true mean of thetahat when n=10000: 1.964477
true median of thetahat when n=10000: 1.964489
true variance of thetahat when n=10000: 0.000021
number of successful experiments: 1000/1000
mean of thetahat: 1.965606
median of thetahat: 1.966000
variance of thetahat: 0.001884
mean of classical variance estimate: 0.001859
median of classical variance estimate: 0.001841
mean of adaptive variance estimate: 0.002472
median of adaptive variance estimate: 0.002347
coverage rate of classical variance estiamte: 0.947000 / std errors: 0.007085
coverage rate of adaptive variance estiamte: 0.967000 / std errors: 0.005649

######### n=300



"""

############################################################# results: SAC  #############################################################


"""

######### n=30

######### n=50

######### n=100



######### n=300


######### n=500



"""
