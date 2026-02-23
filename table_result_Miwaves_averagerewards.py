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
parser.add_argument('--evaluate', type=int, default=0, help="evaluate the true thetahat on a large n")
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
            subpath = path + f"Miwaves_TS_n{n_true}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n_true}_decisionBtwnUpdates=1_averagerewards/exp=1"
        else: # SAC
            subpath = path + f"Miwaves_sac_n{n_true}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n_true}_decisionBtwnUpdates=1_averagerewards/exp=1"
        data = pd.read_csv(subpath+'/data.csv')
        rewards_list_true.append(data['reward'].mean())

    true_reward_mean = np.mean(rewards_list_true)
    true_reward_median = np.median(rewards_list_true)
    true_var = np.var(np.array(rewards_list_true)) 
else: # 0
    if args.alg == 'TS': # TOBE UPDATED
        true_reward_mean = 2.106088
        true_reward_median = 2.106066
        true_var = 0.000014
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
        subpath = path + f"Miwaves_TS_n{n}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n}_decisionBtwnUpdates=1_averagerewards/exp=1"
    else:
        subpath = path + f"Miwaves_sac_n{n}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n}_decisionBtwnUpdates=1_averagerewards/exp=1"
    # try:
    RL_data = pd.read_csv(subpath+'/data.csv')
    rewards_list.append(RL_data['reward'].mean())
    data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
    theta_hat_list.append(data['theta_est'].item())
    classical_var_list.append(data['classical_sandwich_var_estimate'].item())
    adaptive_var_list.append(data['adaptive_sandwich_var_estimate'].item())
    count += 1
    # except:
    #     continue

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

##### b=10.0
"""
######### n=30
true mean of thetahat when n=30: 2.104946
true median of thetahat when n=30: 2.106111
true variance of thetahat when n=30: 0.004239
mean of thetahat: 2.104946
median of thetahat: 2.106111
variance of thetahat: 0.004239
mean of classical variance estimate: 0.004243
median of classical variance estimate: 0.004096
mean of adaptive variance estimate: 0.087116
median of adaptive variance estimate: 0.028394
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.930000 / std errors: 0.008068
coverage rate of adaptive variance estiamte: 0.991000 / std errors: 0.002986



######### n=50
Evaluation on the average reward inference when n=50
true mean of thetahat when n=50: 2.106880
true median of thetahat when n=50: 2.107500
true variance of thetahat when n=50: 0.002395
mean of thetahat: 2.106880
median of thetahat: 2.107500
variance of thetahat: 0.002395
mean of classical variance estimate: 0.002586
median of classical variance estimate: 0.002510
mean of adaptive variance estimate: 0.031296
median of adaptive variance estimate: 0.014415
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.942000 / std errors: 0.007392
coverage rate of adaptive variance estiamte: 0.996000 / std errors: 0.001996

######### n=100
true mean of thetahat when n=100: 2.107318
true median of thetahat when n=100: 2.107500
true variance of thetahat when n=100: 0.001240
mean of thetahat: 2.107318
median of thetahat: 2.107500
variance of thetahat: 0.001240
mean of classical variance estimate: 0.001291
median of classical variance estimate: 0.001278
mean of adaptive variance estimate: 0.008600
median of adaptive variance estimate: 0.005571
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.946000 / std errors: 0.007147
coverage rate of adaptive variance estiamte: 0.998000 / std errors: 0.001413


######### n=300



######### n=500




######### n=1000 



"""
##### b=1.0

"""
######### n=30
true mean of thetahat when n=30: 2.102693
true median of thetahat when n=30: 2.103889
true variance of thetahat when n=30: 0.004311
mean of thetahat: 2.102693
median of thetahat: 2.103889
variance of thetahat: 0.004311
mean of classical variance estimate: 0.004278
median of classical variance estimate: 0.004126
mean of adaptive variance estimate: 0.004419
median of adaptive variance estimate: 0.004198
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.927000 / std errors: 0.008226
coverage rate of adaptive variance estiamte: 0.930000 / std errors: 0.008068

true mean of thetahat when n=10000: 2.106088
true median of thetahat when n=10000: 2.106066
true variance of thetahat when n=10000: 0.000014
number of successful experiments: 1000/1000
mean of thetahat: 2.102693
median of thetahat: 2.103889
variance of thetahat: 0.004311
mean of classical variance estimate: 0.004278
median of classical variance estimate: 0.004126
mean of adaptive variance estimate: 0.004419
median of adaptive variance estimate: 0.004198
coverage rate of classical variance estiamte: 0.928000 / std errors: 0.008174
coverage rate of adaptive variance estiamte: 0.934000 / std errors: 0.007851

######### n=50
true mean of thetahat when n=50: 2.105157
true median of thetahat when n=50: 2.105333
true variance of thetahat when n=50: 0.002403
mean of thetahat: 2.105157
median of thetahat: 2.105333
variance of thetahat: 0.002403
mean of classical variance estimate: 0.002602
median of classical variance estimate: 0.002516
mean of adaptive variance estimate: 0.002669
median of adaptive variance estimate: 0.002588
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.945000 / std errors: 0.007209
coverage rate of adaptive variance estiamte: 0.948000 / std errors: 0.007021

Evaluation on the average reward inference when n=50
true mean of thetahat when n=10000: 2.106088
true median of thetahat when n=10000: 2.106066
true variance of thetahat when n=10000: 0.000014
number of successful experiments: 1000/1000
mean of thetahat: 2.105157
median of thetahat: 2.105333
variance of thetahat: 0.002403
mean of classical variance estimate: 0.002602
median of classical variance estimate: 0.002516
mean of adaptive variance estimate: 0.002669
median of adaptive variance estimate: 0.002588
coverage rate of classical variance estiamte: 0.948000 / std errors: 0.007021
coverage rate of adaptive variance estiamte: 0.949000 / std errors: 0.006957

######### n=100
true mean of thetahat when n=100: 2.105751
true median of thetahat when n=100: 2.106250
true variance of thetahat when n=100: 0.001245
mean of thetahat: 2.105751
median of thetahat: 2.106250
variance of thetahat: 0.001245
mean of classical variance estimate: 0.001300
median of classical variance estimate: 0.001284
mean of adaptive variance estimate: 0.001322
median of adaptive variance estimate: 0.001303
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.953000 / std errors: 0.006693
coverage rate of adaptive variance estiamte: 0.955000 / std errors: 0.006556


Evaluation on the average reward inference when n=100
true mean of thetahat when n=10000: 2.106088
true median of thetahat when n=10000: 2.106066
true variance of thetahat when n=10000: 0.000014
number of successful experiments: 1000/1000
mean of thetahat: 2.105751
median of thetahat: 2.106250
variance of thetahat: 0.001245
mean of classical variance estimate: 0.001300
median of classical variance estimate: 0.001284
mean of adaptive variance estimate: 0.001322
median of adaptive variance estimate: 0.001303
coverage rate of classical variance estiamte: 0.953000 / std errors: 0.006693
coverage rate of adaptive variance estiamte: 0.955000 / std errors: 0.006556

######### n=300
true mean of thetahat when n=300: 2.104612
true median of thetahat when n=300: 2.104806
true variance of thetahat when n=300: 0.000413
mean of thetahat: 2.104612
median of thetahat: 2.104806
variance of thetahat: 0.000413
mean of classical variance estimate: 0.000440
median of classical variance estimate: 0.000438
mean of adaptive variance estimate: 0.000443
median of adaptive variance estimate: 0.000442
number of successful experiments: 999/1000
coverage rate of classical variance estiamte: 0.951952 / std errors: 0.006766
coverage rate of adaptive variance estiamte: 0.951952 / std errors: 0.006766

######### n=500


"""


############################################################# results: SAC  #############################################################


"""

######### n=30

######### n=50

######### n=100



######### n=300


######### n=500



"""
