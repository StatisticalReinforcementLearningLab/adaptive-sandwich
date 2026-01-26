import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")

parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--reg', type=float, default=0.1, help="ridge regression parameter")
parser.add_argument('--reg_true', type=float, default=0.1, help="ridge regression parameter for evaluation")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
args = parser.parse_args()
n=args.n
T=args.T
reg = args.reg
reg_true = args.reg_true
N_seed = args.N_seed

# true variance

path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
# path = ''
"synthetic_mode=delayed_1_action_dosage_alg=sac_T=50_n=30_averagerewards"
rewards_list = []
for i in tqdm(range(N_seed)):
    # subpath = path + f"syn_sac_n{n}_T{T}_reg{reg}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sac_T={T}_n={n}_averagerewards/exp=1"
    subpath = path + f"syn_sac_n500_T{T}_reg{reg_true}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sac_T={T}_n=500_averagerewards/exp=1" # use the largest n to compute true variance
    data = pd.read_csv(subpath+'/data.csv')
    # data = pd.read_csv('n100_T50/data.csv') # local laptop
    # print(data)
    rewards_list.append(data['reward'].mean())

true_var = np.var(np.array(rewards_list)) # assume n * N_seed units, evaluate the population variance here as the true one
print(f'true variance of rewards: {true_var:.6f}')


### estiamted mean and variance
theta_hat_list = []
classical_var_list = []
adaptive_var_list = []

count = 0
for i in tqdm(range(N_seed)):
    subpath = path + f"syn_sac_n{n}_T{T}_reg{reg}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sac_T={T}_n={n}_averagerewards/exp=1"
    try:
        data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
        theta_hat_list.append(data['theta_est'].item())
        classical_var_list.append(data['classical_sandwich_var_estimate'].item())
        adaptive_var_list.append(data['adaptive_sandwich_var_estimate'].item())
        count += 1
    except:
        continue
    

true_reward_mean = np.mean(rewards_list)
true_reward_median = np.median(rewards_list)
print(f'mean of thetahat: {true_reward_mean:.6f}')
print(f'median of thetahat: {true_reward_median:.6f}')
print(f'mean of classical variance estimate: {np.mean(classical_var_list, axis=0):.6f}')
print(f'median of classical variance estimate: {np.median(classical_var_list, axis=0):.6f}')
print(f'mean of adaptive variance estimate: {np.mean(adaptive_var_list, axis=0):.6f}' )
print(f'median of adaptive variance estimate: {np.median(adaptive_var_list, axis=0):.6f}' )
print(f'number of successful experiments: {count}/{N_seed}')

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
for i in range(len(theta_hat_list)):
    num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_reward_mean)
    # num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_reward_median)
    num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_reward_mean)
    # num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_reward_median)
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
true variance of rewards: 0.005726
mean of thetahat: 0.296367
mean of classical variance estiamte: 0.003712
median of classical variance estiamte: 0.003655
mean of adaptive variance estiamte: 0.020846
median of adaptive variance estiamte: 0.010398
coverage rate of classical variance estiamte: 0.865000 / std errors: 0.010806
coverage rate of adaptive variance estiamte: 0.971000 / std errors: 0.005307

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


############################################################# results: SAC  #############################################################


"""

################# n=30
# ridge_regression=1.0
Mean parameter estimate:
[0.2736279]

Empirical variance of parameter estimates:
[[0.00544728]]

Mean adaptive sandwich variance estimate:
[[0.00601462]]

Mean classical sandwich variance estimate:
[[0.00421583]]

Median adaptive sandwich variance estimate:
[[0.00546962]]

Median classical sandwich variance estimate:
[[0.00415732]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.95

Classical sandwich 95.0% standard normal CI coverage:
0.902

# ridge_regression=1.0*3/n
Mean parameter estimate:
[0.4331892]

Empirical variance of parameter estimates:
[[0.00439431]]

Mean adaptive sandwich variance estimate:
[[1.2191781]]

Mean classical sandwich variance estimate:
[[0.00380698]]

Median adaptive sandwich variance estimate:
[[0.01266304]]

Median classical sandwich variance estimate:
[[0.00378271]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9909638554216867

Classical sandwich 95.0% standard normal CI coverage:
0.9347389558232931

################# n=50
# ridge_regression=1.0
Mean parameter estimate:
[0.27498797]

Empirical variance of parameter estimates:
[[0.00332289]]

Mean adaptive sandwich variance estimate:
[[0.00336951]]

Mean classical sandwich variance estimate:
[[0.00255753]]

Median adaptive sandwich variance estimate:
[[0.00313932]]

Median classical sandwich variance estimate:
[[0.00252592]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.95

Classical sandwich 95.0% standard normal CI coverage:
0.917

# ridge_regression=1.0*30/n
Mean parameter estimate:
[0.33922848]

Empirical variance of parameter estimates:
[[0.00344663]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00018765]]

Mean adaptive sandwich variance estimate:
[[0.2739766]]

Mean classical sandwich variance estimate:
[[0.00261799]]

Median adaptive sandwich variance estimate:
[[0.0034383]]

Median classical sandwich variance estimate:
[[0.00260686]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.949

Classical sandwich 95.0% standard normal CI coverage:
0.915

# ridge_regression=1.0*3/n
Mean parameter estimate:
[0.43459508]

Empirical variance of parameter estimates:
[[0.00263142]]

Mean adaptive sandwich variance estimate:
[[0.2804591]]

Mean classical sandwich variance estimate:
[[0.00226091]]

Median adaptive sandwich variance estimate:
[[0.00657748]]

Median classical sandwich variance estimate:
[[0.00224159]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.991

Classical sandwich 95.0% standard normal CI coverage:
0.932


################# n=100
# ridge_regression=1.0
Mean parameter estimate:
[0.27688453]

Empirical variance of parameter estimates:
[[0.00169417]]

Mean adaptive sandwich variance estimate:
[[0.00155251]]

Mean classical sandwich variance estimate:
[[0.00129056]]

Median adaptive sandwich variance estimate:
[[0.0015275]]

Median classical sandwich variance estimate:
[[0.00128569]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.929

Classical sandwich 95.0% standard normal CI coverage:
0.91

# ridge_regression=1.0*30/n
Mean parameter estimate:
[0.41509205]

Empirical variance of parameter estimates:
[[0.00164293]]

Mean adaptive sandwich variance estimate:
[[0.01969936]]

Mean classical sandwich variance estimate:
[[0.00124683]]

Median adaptive sandwich variance estimate:
[[0.00166864]]

Median classical sandwich variance estimate:
[[0.00124059]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.954

Classical sandwich 95.0% standard normal CI coverage:
0.911

# ridge_regression=1.0*3/n

Mean parameter estimate:
[0.42765185]

Empirical variance of parameter estimates:
[[0.00139269]]

Mean adaptive sandwich variance estimate:
[[0.51008356]]

Mean classical sandwich variance estimate:
[[0.001129]]

Median adaptive sandwich variance estimate:
[[0.00286803]]

Median classical sandwich variance estimate:
[[0.00112382]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.994

Classical sandwich 95.0% standard normal CI coverage:
0.914


################# n=300
Mean parameter estimate:
[0.2765282]

Empirical variance of parameter estimates:
[[0.00064934]]

Mean adaptive sandwich variance estimate:
[[0.00049266]]

Mean classical sandwich variance estimate:
[[0.00043498]]

Median adaptive sandwich variance estimate:
[[0.00048934]]

Median classical sandwich variance estimate:
[[0.0004342]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.905

Classical sandwich 95.0% standard normal CI coverage:
0.888

## [ridge_regression=1.0*30/n]
Mean parameter estimate:
[0.4450719]

Empirical variance of parameter estimates:
[[0.00056656]]

Mean adaptive sandwich variance estimate:
[[0.03817153]]

Mean classical sandwich variance estimate:
[[0.0003868]]

Median adaptive sandwich variance estimate:
[[0.00045294]]

Median classical sandwich variance estimate:
[[0.00038536]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.928

Classical sandwich 95.0% standard normal CI coverage:
0.903

## ridge_regression=1.0*3/n]
Mean parameter estimate:
[0.40956098]

Empirical variance of parameter estimates:
[[0.00053729]]

Mean adaptive sandwich variance estimate:
[[0.00099715]]

Mean classical sandwich variance estimate:
[[0.00037584]]

Median adaptive sandwich variance estimate:
[[0.00073118]]

Median classical sandwich variance estimate:
[[0.00037504]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.98

Classical sandwich 95.0% standard normal CI coverage:
0.906


################# n=500
## ridge_regression=1.0
Mean parameter estimate:
[0.2778481]

Empirical variance of parameter estimates:
[[0.00039302]]

Mean adaptive sandwich variance estimate:
[[0.00029076]]

Mean classical sandwich variance estimate:
[[0.00026066]]

Median adaptive sandwich variance estimate:
[[0.00029091]]

Median classical sandwich variance estimate:
[[0.0002607]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.905

Classical sandwich 95.0% standard normal CI coverage:
0.893

## ridge_regression=1.0*30/n
Mean parameter estimate:
[0.44310087]

Empirical variance of parameter estimates:
[[0.00035467]]

Mean adaptive sandwich variance estimate:
[[0.00027257]]

Mean classical sandwich variance estimate:
[[0.00022975]]

Median adaptive sandwich variance estimate:
[[0.00026344]]

Median classical sandwich variance estimate:
[[0.00022993]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.906

Classical sandwich 95.0% standard normal CI coverage:
0.886

## ridge_regression=1.0*3/n
Mean parameter estimate:
[0.40301606]

Empirical variance of parameter estimates:
[[0.00033206]]

Mean adaptive sandwich variance estimate:
[[0.00394498]]

Mean classical sandwich variance estimate:
[[0.00022471]]

Median adaptive sandwich variance estimate:
[[0.00038897]]

Median classical sandwich variance estimate:
[[0.00022437]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9599198396793587

Classical sandwich 95.0% standard normal CI coverage:
0.8967935871743486


"""
