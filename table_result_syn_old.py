import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")

parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--N_seed', type=int, default=3, help="N_seed")
args = parser.parse_args()
n=args.n
T=args.T

N_seed = args.N_seed

# true variance

# path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
path = ''

rewards_list = []
for i in tqdm(range(N_seed)):
    subpath = path + f"n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=smooth_posterior_sampling_T={T}_n={n}_recruitN={n}_decisionsBtwnUpdates=1_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1"
    data = pd.read_csv(subpath+'/data.csv')
    # data = pd.read_csv('n100_T50/data.csv') # local laptop
    print(data)
    rewards_list.append(data['reward'].mean())

true_var = np.var(np.array(rewards_list)) # assume n * N_seed units, evaluate the population variance here as the true one
print(f'true variance of rewards: {true_var:.6f}')


### estiamted mean and variance
theta_hat_list = []
classical_var_list = []
adaptive_var_list = []

for i in tqdm(range(N_seed)):
    subpath = path + f"n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=smooth_posterior_sampling_T={T}_n={n}_recruitN={n}_decisionsBtwnUpdates=1_algfeats=intercept,past_reward_errcorr=time_corr_actionC=0/exp=1"
    data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
    theta_hat_list.append(data['theta_est'].item())
    classical_var_list.append(data['classical_sandwich_var_estimate'].item())
    adaptive_var_list.append(data['adaptive_sandwich_var_estimate'].item())

true_reward_mean = np.mean(rewards_list)
print(f'mean of thetahat: {true_reward_mean:.6f}')
print(f'mean of classical variance estimate: {np.mean(classical_var_list, axis=0):.6f}')
print(f'median of classical variance estimate: {np.median(classical_var_list, axis=0):.6f}')
print(f'mean of adaptive variance estimate: {np.mean(adaptive_var_list, axis=0):.6f}' )
print(f'median of adaptive variance estimate: {np.median(adaptive_var_list, axis=0):.6f}' )


### converage rate

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
for i in range(N_seed):
    num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_reward_mean)
    num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_reward_mean)
    l_classical_covered.append(num_classical_covered)
    l_adaptive_covered.append(num_adaptive_covered)

l_classical_covered = np.array(l_classical_covered)
l_adaptive_covered = np.array(l_adaptive_covered)

print(f'coverage rate of classical variance estiamte: {l_classical_covered.mean():.6f} / std errors: {l_classical_covered.std() / np.sqrt(N_seed):.6f}')
print(f'coverage rate of adaptive variance estiamte: {l_adaptive_covered.mean():.6f} / std errors: {l_adaptive_covered.std() / np.sqrt(N_seed):.6f}')
print(args)


###### results
"""
######### n=30
true variance of rewards: 0.005726
mean of thetahat: 0.296367
mean of classical variance estiamte: 0.003712
median of classical variance estiamte: 0.003655
mean of adaptive variance estiamte: 0.020846
median of adaptive variance estiamte: 0.010398
coverage rate of classical variance estiamte: 0.865000 / std errors: 0.010806
coverage rate of adaptive variance estiamte: 0.971000 / std errors: 0.005307

-----sample correction [TODO]


######### n=50
true variance of rewards: 0.003659
mean of thetahat: 0.301267
mean of classical variance estiamte: 0.002249
median of classical variance estiamte: 0.002225
mean of adaptive variance estiamte: 0.008838
median of adaptive variance estiamte: 0.005826
coverage rate of classical variance estiamte: 0.861000 / std errors: 0.010940
coverage rate of adaptive variance estiamte: 0.975000 / std errors: 0.004937

######### n=100
true variance of rewards: 0.002060
mean of thetahat: 0.301623
mean of classical variance estiamte: 0.001138
median of classical variance estiamte: 0.001128
mean of adaptive variance estiamte: 0.003522
median of adaptive variance estiamte: 0.002513
coverage rate of classical variance estiamte: 0.847000 / std errors: 0.011384
coverage rate of adaptive variance estiamte: 0.951000 / std errors: 0.006826

######### n=300
true variance of rewards: 0.000792
mean of thetahat: 0.302433
mean of classical variance estiamte: 0.000383
median of classical variance estiamte: 0.000383
mean of adaptive variance estiamte: 0.000953
median of adaptive variance estiamte: 0.000829
coverage rate of classical variance estiamte: 0.819000 / std errors: 0.012175
coverage rate of adaptive variance estiamte: 0.947000 / std errors: 0.007085


######### n=500
true variance of rewards: 0.000448
mean of thetahat: 0.304311
mean of classical variance estiamte: 0.000230
median of classical variance estiamte: 0.000230
mean of adaptive variance estiamte: 0.000517
median of adaptive variance estiamte: 0.000473
coverage rate of classical variance estiamte: 0.848000 / std errors: 0.011353
coverage rate of adaptive variance estiamte: 0.946000 / std errors: 0.007147



######### n=1000 [convergence rate is O(1/n), which is correct here.]
true variance of rewards: 0.000224
mean of thetahat: 0.304409
mean of classical variance estiamte: 0.000115
median of classical variance estiamte: 0.000115
mean of adaptive variance estiamte: 0.000254
median of adaptive variance estiamte: 0.000236
coverage rate of classical variance estiamte: 0.843000 / std errors: 0.011504
coverage rate of adaptive variance estiamte: 0.956000 / std errors: 0.006486

"""



