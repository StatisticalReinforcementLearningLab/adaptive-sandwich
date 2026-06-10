import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import sys, os

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")

parser.add_argument('--alg', type=str, default='TS', help="algorithm name", choices=['TS', 'SAC'])
parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--ridge_penalty', type=float, default=20.0, help="ridge regression parameter")
parser.add_argument('--s', type=float, default=1.0, help="steepness for the sigmoid function in SAC")
parser.add_argument('--epoch', type=int, default=500, help="epoch of actor update in SAC")
parser.add_argument('--lr', type=float, default=10.0, help="lr in SAC")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
parser.add_argument('--evaluate', type=int, default=0, help="evaluate the true thetahat on a large n")
args = parser.parse_args()
n=args.n
T=args.T
N_seed = args.N_seed

alpha1 = 0.01
alpha2 = 0.1
algo_name = 'smooth_posterior_sampling' if args.alg == 'TS' else 'sac'
n_true = 10000 
# n_true = args.n

######################################## Step 1: access all replications, compute each average rewards, assess the true theta1
path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'

if args.evaluate > 0:
    n_true = args.n if args.evaluate == 2 else n_true 
    theta1_list_true = []
    for i in tqdm(range(N_seed)):
        if args.alg == 'TS': # old path
            subpath = path + f"syn_TS_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
        else: # SAC
            subpath = path + f"syn_sac_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_ridge{args.ridge_penalty}_s{args.s}_epoch{args.epoch}_lr{args.lr}_nostopping_expectation_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1" # use the largest n to compute true variance
            
        df = pd.read_csv(subpath+'/data.csv')
        df_X = df[['intercept','pretreat_feature1','pretreat_feature2','Z_id']]
        df_Y = df['reward']
        linear_model_single = LinearRegression(fit_intercept=False)
        linear_model_single.fit(df_X, df_Y)
        theta1_list_true.append(linear_model_single.coef_[-1])


    ## Remark: the true mean is averaged over all replications with mutiple theta1 estimate instead of one-time theta1 estimation on the combined large dataset n_true * replications
    true_theta1_mean = np.mean(np.array(theta1_list_true))
    true_theta1_median = np.median(np.array(theta1_list_true))
    true_var_theta1 = np.var(np.array(theta1_list_true)) 
else:
    if args.alg == 'TS': # TOBE UPDATED
        true_theta1_mean = 0.147552
        true_theta1_median = 0.147443
        true_var_theta1 = 0.000067
    elif args.alg == 'SAC':
        true_theta1_mean = 0.234817
        true_theta1_median = 0.234835
        true_var_theta1 = 0.000084
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
        subpath = path + f"syn_TS_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
    else:
        subpath = path + f"syn_sac_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_ridge{args.ridge_penalty}_s{args.s}_epoch{args.epoch}_lr{args.lr}_nostopping_expectation_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1" # use the largest n to compute true variance
            
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
##### n=30
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.133062
median of thetahat: 0.130657
variance of thetahat: 0.019427
mean of classical variance estimate: 0.014094
median of classical variance estimate: 0.013642
mean of adaptive variance estimate: 0.063839
median of adaptive variance estimate: 0.029325
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.900000 / std errors: 0.009487
coverage rate of adaptive variance estiamte: 0.967000 / std errors: 0.005649

##### n=50
Evaluation on the treatment effect inference when n=50
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.135509
median of thetahat: 0.136156
variance of thetahat: 0.012279
mean of classical variance estimate: 0.008714
median of classical variance estimate: 0.008561
mean of adaptive variance estimate: 0.031031
median of adaptive variance estimate: 0.017905
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.893000 / std errors: 0.009775
coverage rate of adaptive variance estiamte: 0.968000 / std errors: 0.005566

##### n=100
Evaluation on the treatment effect inference when n=100
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.142783
median of thetahat: 0.144153
variance of thetahat: 0.006265
mean of classical variance estimate: 0.004442
median of classical variance estimate: 0.004431
mean of adaptive variance estimate: 0.010775
median of adaptive variance estimate: 0.007721
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.896000 / std errors: 0.009653
coverage rate of adaptive variance estiamte: 0.964000 / std errors: 0.005891

##### n=300
Evaluation on the treatment effect inference when n=300
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.145818
median of thetahat: 0.145149
variance of thetahat: 0.002136
mean of classical variance estimate: 0.001505
median of classical variance estimate: 0.001499
mean of adaptive variance estimate: 0.002854
median of adaptive variance estimate: 0.002429
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.901000 / std errors: 0.009445
coverage rate of adaptive variance estiamte: 0.952000 / std errors: 0.006760


##### n=500
Evaluation on the treatment effect inference when n=500
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.144502
median of thetahat: 0.144186
variance of thetahat: 0.001340
mean of classical variance estimate: 0.000903
median of classical variance estimate: 0.000902
mean of adaptive variance estimate: 0.001601
median of adaptive variance estimate: 0.001374
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.904000 / std errors: 0.009316
coverage rate of adaptive variance estiamte: 0.955000 / std errors: 0.006556

#### n=1000
Evaluation on the treatment effect inference when n=1000
true mean of thetahat when n=10000: 0.147552
true median of thetahat when n=10000: 0.147443
true variance of thetahat when n=10000: 0.000067
mean of thetahat: 0.146655
median of thetahat: 0.146045
variance of thetahat: 0.000673
mean of classical variance estimate: 0.000454
median of classical variance estimate: 0.000455
mean of adaptive variance estimate: 0.000752
median of adaptive variance estimate: 0.000700
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.899000 / std errors: 0.009529
coverage rate of adaptive variance estiamte: 0.950000 / std errors: 0.006892

"""


############################################################# results: SAC  #############################################################
"""

################# n=30
true mean of thetahat when n=30: 0.160271
true median of thetahat when n=30: 0.159228
true variance of thetahat when n=30: 0.020450
mean of thetahat: 0.160271
median of thetahat: 0.159228
variance of thetahat: 0.020450
mean of classical variance estimate: 0.015361
median of classical variance estimate: 0.014786
mean of adaptive variance estimate: 0.031460
median of adaptive variance estimate: 0.024268
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.906000 / std errors: 0.009228
coverage rate of adaptive variance estiamte: 0.960000 / std errors: 0.006197

################# n=50
true mean of thetahat when n=50: 0.220626
true median of thetahat when n=50: 0.222870
true variance of thetahat when n=50: 0.011533
mean of thetahat: 0.220626
median of thetahat: 0.222870
variance of thetahat: 0.011533
mean of classical variance estimate: 0.009339
median of classical variance estimate: 0.009140
mean of adaptive variance estimate: 0.051173
median of adaptive variance estimate: 0.015291
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.908000 / std errors: 0.009140
coverage rate of adaptive variance estiamte: 0.962000 / std errors: 0.006046

################# n=100
true mean of thetahat when n=100: 0.274155
true median of thetahat when n=100: 0.278305
true variance of thetahat when n=100: 0.005309
mean of thetahat: 0.274155
median of thetahat: 0.278305
variance of thetahat: 0.005309
mean of classical variance estimate: 0.004568
median of classical variance estimate: 0.004533
mean of adaptive variance estimate: 7.384125
median of adaptive variance estimate: 0.007644
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.930000 / std errors: 0.008068
coverage rate of adaptive variance estiamte: 0.976000 / std errors: 0.004840

################# n=300
true mean of thetahat when n=300: 0.285240
true median of thetahat when n=300: 0.285118
true variance of thetahat when n=300: 0.001726
mean of thetahat: 0.285240
median of thetahat: 0.285118
variance of thetahat: 0.001726
mean of classical variance estimate: 0.001506
median of classical variance estimate: 0.001502
mean of adaptive variance estimate: 0.003528
median of adaptive variance estimate: 0.001871
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.932000 / std errors: 0.007961
coverage rate of adaptive variance estiamte: 0.951000 / std errors: 0.006826

################# n=500
true mean of thetahat when n=500: 0.278355
true median of thetahat when n=500: 0.279289
true variance of thetahat when n=500: 0.001060
mean of thetahat: 0.278355
median of thetahat: 0.279289
variance of thetahat: 0.001060
mean of classical variance estimate: 0.000902
median of classical variance estimate: 0.000901
mean of adaptive variance estimate: 0.001309
median of adaptive variance estimate: 0.001106
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.932000 / std errors: 0.007961
coverage rate of adaptive variance estiamte: 0.957000 / std errors: 0.006415


################# n=1000
true mean of thetahat when n=1000: 0.269441
true median of thetahat when n=1000: 0.269040
true variance of thetahat when n=1000: 0.000551
mean of thetahat: 0.269441
median of thetahat: 0.269040
variance of thetahat: 0.000551
mean of classical variance estimate: 0.000451
median of classical variance estimate: 0.000452
mean of adaptive variance estimate: 0.000708
median of adaptive variance estimate: 0.000541
number of successful experiments: 998/1000
coverage rate of classical variance estiamte: 0.931864 / std errors: 0.007976
coverage rate of adaptive variance estiamte: 0.954910 / std errors: 0.006568


"""



