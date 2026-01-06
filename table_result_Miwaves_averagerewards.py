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

path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
# path = ''

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


############################################################# results: TS  #############################################################
"""
######### n=30
Mean parameter estimate:
[2.1062086]

Empirical variance of parameter estimates:
[[0.00420813]]

Mean adaptive sandwich variance estimate:
[[0.857533]]

Mean classical sandwich variance estimate:
[[0.00424518]]

Median adaptive sandwich variance estimate:
[[0.09142613]]

Median classical sandwich variance estimate:
[[0.00408459]]

Adaptive sandwich 95.0% standard normal CI coverage:
1.0

Classical sandwich 95.0% standard normal CI coverage:
0.931

######### n=50
Mean parameter estimate:
[2.1086166]

Empirical variance of parameter estimates:
[[0.0024041]]

Mean adaptive sandwich variance estimate:
[[0.3931545]]

Mean classical sandwich variance estimate:
[[0.00258562]]

Median adaptive sandwich variance estimate:
[[0.05722548]]

Median classical sandwich variance estimate:
[[0.00252763]]

Adaptive sandwich 95.0% standard normal CI coverage:
1.0

Classical sandwich 95.0% standard normal CI coverage:
0.944

######### n=100
Mean parameter estimate:
[2.109141]

Empirical variance of parameter estimates:
[[0.00124483]]

Mean adaptive sandwich variance estimate:
[[0.06147134]]

Mean classical sandwich variance estimate:
[[0.00128868]]

Median adaptive sandwich variance estimate:
[[0.0224028]]

Median classical sandwich variance estimate:
[[0.00127273]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.999

Classical sandwich 95.0% standard normal CI coverage:
0.949

######### n=300

Mean parameter estimate:
[2.1080263]

Empirical variance of parameter estimates:
[[0.00040944]]

Mean adaptive sandwich variance estimate:
[[0.00509669]]

Mean classical sandwich variance estimate:
[[0.00043647]]

Median adaptive sandwich variance estimate:
[[0.00279501]]

Median classical sandwich variance estimate:
[[0.00043344]]

Adaptive sandwich 95.0% standard normal CI coverage:
1.0

Classical sandwich 95.0% standard normal CI coverage:
0.958

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
