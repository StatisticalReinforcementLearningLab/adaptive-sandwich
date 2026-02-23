import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sys

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")
parser.add_argument('--alg', type=str, default='TS', help="algorithm name", choices=['TS', 'SAC'])
parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
parser.add_argument('--evaluate', type=int, default=0, help="evaluate the true thetahat on a large n")
args = parser.parse_args()
n=args.n
T=args.T

# Variance_rewards = True

N_seed = args.N_seed

alpha1 = 0.01
alpha2 = 0.1
algo_name = 'smooth_posterior_sampling' if args.alg == 'TS' else 'sac'

n_true = 10000 # the true theta should be evaluated on a large n


# true variance

path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
# path = ''

# if Variance_rewards:
#     rewards_list_initialization = []
#     for i in tqdm(range(N_seed)):
#         subpath = path + f"Miwaves_TS_n42_T60/{i}/simulated_data/miwaves_alg=fixed_randomization_T=60_n=42_decisionBtwnUpdates=1_averagerewards/exp=1"
#         data = pd.read_csv(subpath+'/data.csv')
#         rewards_list_initialization.append(data['reward'].mean())
#     reward_mean = np.mean(rewards_list_initialization)
#     reward_std = np.std(np.array(rewards_list_initialization))
#     print('mean rewards: ', reward_mean)
#     print('std of rewards: ', reward_std)
#     print(f'b: 20/{reward_std:.3d}= {20/reward_std:.3f}, 10/{reward_std:.3d}= {10/reward_std:.3f}')
#     sys.exit()

    
if args.evaluate == 1: # n = 10000
    theta1_list_true = []
    for i in tqdm(range(N_seed)):
        if args.alg == 'TS': # old path
            subpath = path + f"Miwaves_TS_n{n_true}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n_true}_decisionBtwnUpdates=1_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
        else: # SAC
            subpath = path + f"Miwaves_sac_n{n_true}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n_true}_decisionBtwnUpdates=1_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
        df = pd.read_csv(subpath+'/data.csv')
        df_X = df[['intercept','pretreat_feature1','pretreat_feature2','Z_id']]
        df_Y = df['reward']
        linear_model_single = LinearRegression(fit_intercept=False)
        linear_model_single.fit(df_X, df_Y)
        theta1_list_true.append(linear_model_single.coef_[-1])

    true_theta1_mean = np.mean(np.array(theta1_list_true))
    true_theta1_median = np.median(np.array(theta1_list_true))
    true_var_theta1 = np.var(np.array(theta1_list_true)) 
else:
    if args.alg == 'TS': # TOBE UPDATED
        true_theta1_mean = None
        true_theta1_median = None
        true_var_theta1 = None
    elif args.alg == 'SAC':
        true_theta1_mean = None
        true_theta1_median = None
        true_var_theta1 = None
    else:
        raise NotImplementedError


######################################## Step 2: access all replications, compute all the classical and adaptive variance estimates. 
theta_hat_list = []
classical_var_list = []
adaptive_var_list = []

count = 0
theta1_list = []
for i in tqdm(range(N_seed)):
    if args.alg == 'TS': # old path
        subpath = path + f"Miwaves_TS_n{n}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
    else:
        subpath = path + f"Miwaves_sac_n{n}_T{T}/{i}/simulated_data/miwaves_alg={algo_name}_T={T}_n={n}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
    try:
        RL_data = pd.read_csv(subpath+'/data.csv')
        df_X = RL_data[['intercept','pretreat_feature1','pretreat_feature2','Z_id']]
        df_Y = RL_data['reward']
        linear_model_single = LinearRegression(fit_intercept=False)
        linear_model_single.fit(df_X, df_Y)
        theta1_list.append(linear_model_single.coef_[-1])
        data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
        theta_hat_list.append(data['theta_est'][1].item())
        classical_var_list.append(data['classical_sandwich_var_estimate'][1,1].item())
        adaptive_var_list.append(data['adaptive_sandwich_var_estimate'][1,1].item())
        count += 1
    except:
        continue

theta1_mean = np.mean(np.array(theta1_list))
theta1_median = np.median(np.array(theta1_list))
theta1_var = np.var(np.array(theta1_list))

print(f'Evaluation on the treatment effect inference when n={n}')
print(f'true mean of thetahat when n={n_true}: {true_theta1_mean:.6f}')
print(f'true median of thetahat when n={n_true}: {true_theta1_median:.6f}')
print(f'true variance of thetahat when n={n_true}: {true_var_theta1:.6f}')
print(f'mean of thetahat: {theta1_mean:.6f}')
print(f'median of thetahat: {theta1_median:.6f}')
print(f'variance of thetahat: {theta1_var:.6f}')
print(f'mean of classical variance estimate: {np.mean(classical_var_list, axis=0):.6f}')
print(f'median of classical variance estimate: {np.median(classical_var_list, axis=0):.6f}')
print(f'mean of adaptive variance estimate: {np.mean(adaptive_var_list, axis=0):.6f}' )
print(f'median of adaptive variance estimate: {np.median(adaptive_var_list, axis=0):.6f}' )
print(f'number of successful experiments: {count}/{N_seed}')


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
    num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_theta1_mean)
    num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_theta1_mean)
    l_classical_covered.append(num_classical_covered)
    l_adaptive_covered.append(num_adaptive_covered)

l_classical_covered = np.array(l_classical_covered)
l_adaptive_covered = np.array(l_adaptive_covered)

print(f'coverage rate of classical variance estiamte: {l_classical_covered.mean():.6f} / std errors: {l_classical_covered.std() / np.sqrt(count):.6f}')
print(f'coverage rate of adaptive variance estiamte: {l_adaptive_covered.mean():.6f} / std errors: {l_adaptive_covered.std() / np.sqrt(count):.6f}')
print(args)



############################################################# results: TS  #############################################################
"""
######### n=30


######### n=50


######### n=100


######### n=300



######### n=500




######### n=1000 



"""


############################################################# results: SAC  #############################################################


"""

######### n=30

######### n=50

######### n=100



######### n=300


######### n=500



"""
