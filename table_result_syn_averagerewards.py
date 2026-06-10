import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm


parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")

parser.add_argument('--alg', type=str, default='TS', help="algorithm name", choices=['TS', 'SAC'])
parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--ridge_penalty', type=float, default=20.0, help="ridge regression parameter")
parser.add_argument('--constant_ridge', type=int, default=0, help="1: constant ridge penalty, 0: decayed ridge penalty (classical)")
parser.add_argument('--s', type=float, default=1.0, help="steepness for the sigmoid function in SAC")
parser.add_argument('--epoch', type=int, default=500, help="epoch of actor update in SAC")
parser.add_argument('--lr', type=float, default=10.0, help="lr in SAC")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
parser.add_argument('--evaluate', type=int, default=0, help="0: fixed true values after evaluation, 1: evaluate the true thetahat on a large n, 2: evaluate on the current n")
args = parser.parse_args()
n=args.n
T=args.T
N_seed = args.N_seed

algo_name = 'smooth_posterior_sampling' if args.alg == 'TS' else 'sac'

n_true = 10000 # the true theta should be evaluated on a large n

######################################## Step 1: access all replications, compute each average rewards, assess the true average rewards and its variance
path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
constant_ridge = '_constant1' if args.constant_ridge == 1 else ''
# path = ''
# "synthetic_mode=delayed_1_action_dosage_alg=sac_T=50_n=30_averagerewards"
if args.evaluate > 0:
    n_true = args.n if args.evaluate == 2 else n_true 
    rewards_list_true = []
    for i in tqdm(range(N_seed)):
        if args.alg == 'TS': # old path
            subpath = path + f"syn_TS_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_averagerewards/exp=1"
        else: # SAC
            subpath = path + f"syn_sac_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_ridge{args.ridge_penalty}_s{args.s}_epoch{args.epoch}_lr{args.lr}_nostopping_expectation{constant_ridge}_averagerewards/exp=1" # use the largest n to compute true variance
            # subpath = path + f"syn_sac_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_averagerewards/exp=1" # use the largest n to compute true variance
        data = pd.read_csv(subpath+'/data.csv')
        # data = pd.read_csv('n100_T50/data.csv') # local laptop
        # print(data)
        rewards_list_true.append(data['reward'].mean())

    true_reward_mean = np.mean(rewards_list_true)
    true_reward_median = np.median(rewards_list_true)
    true_var = np.var(np.array(rewards_list_true)) # assume n * N_seed units, evaluate the population variance here as the true one
else:
    if args.alg == 'TS': # TOBE UPDATED
        true_reward_mean = 0.305089
        true_reward_median = 0.305021
        true_var = 0.000023
    elif args.alg == 'SAC':
        if args.ridge_penalty == 20.0:
            true_reward_mean = 0.391321
            true_reward_median = 0.391504
            true_var = 0.000053
        elif args.ridge_penalty == 10.0:
            if args.constant_ridge == 1:
                true_reward_mean = 0.166955
                true_reward_median = 0.166928
                true_var = 0.000017
            else:
                true_reward_mean = 0.385006 # without stopping: 0.385006 # with stopping: 0.385007 
                true_reward_median = 0.385421 # without stopping: 0.385421 # with stopping: 0.385421 
                true_var = 0.000053 # without stopping: 0.000053 # with stopping: 0.000053 
        else:
            raise NotImplementedError
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
        subpath = path + f"syn_TS_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n}_averagerewards/exp=1"
    else:
        subpath = path + f"syn_sac_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n}_ridge{args.ridge_penalty}_s{args.s}_epoch{args.epoch}_lr{args.lr}_nostopping_expectation{constant_ridge}_averagerewards/exp=1" # use the largest n to compute true variance
        # subpath = path + f"syn_sac_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_ridge{args.ridge_penalty}_s{args.s}_epoch{args.epoch}_lr{args.lr}_averagerewards/exp=1" # use the largest n to compute true variance
        # subpath = path + f"syn_sac_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sac_T={T}_n={n}_ridge{args.ridge_penalty}_averagerewards/exp=1"
        # subpath = path + f"syn_sac_n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=sac_T={T}_n={n}_averagerewards/exp=1"
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
print(f'mean of thetahat: {reward_mean:.6f}')
print(f'median of thetahat: {reward_median:.6f}')
print(f'variance of thetahat: {reward_variance:.6f}')
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
"""
######### n=30 
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.005611
mean of thetahat: 0.296057
median of thetahat: 0.295965
mean of classical variance estimate: 0.003704
median of classical variance estimate: 0.003661
mean of adaptive variance estimate: 0.020428
median of adaptive variance estimate: 0.010364
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.878000 / std errors: 0.010350
coverage rate of adaptive variance estiamte: 0.975000 / std errors: 0.004937

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=30
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.296057
median of thetahat: 0.295965
variance of thetahat: 0.005611
mean of classical variance estimate: 0.003704
median of classical variance estimate: 0.003661
mean of adaptive variance estimate: 0.020428
median of adaptive variance estimate: 0.010364
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.887000 / std errors: 0.010012
coverage rate of adaptive variance estiamte: 0.975000 / std errors: 0.004937

######### n=50 [updated]
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.003759
mean of thetahat: 0.300352
median of thetahat: 0.300081
mean of classical variance estimate: 0.002257
median of classical variance estimate: 0.002228
mean of adaptive variance estimate: 0.008704
median of adaptive variance estimate: 0.005753
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.852000 / std errors: 0.011229
coverage rate of adaptive variance estiamte: 0.964000 / std errors: 0.005891

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=50
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.300352
median of thetahat: 0.300081
variance of thetahat: 0.003759
mean of classical variance estimate: 0.002257
median of classical variance estimate: 0.002228
mean of adaptive variance estimate: 0.008704
median of adaptive variance estimate: 0.005753
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.852000 / std errors: 0.011229
coverage rate of adaptive variance estiamte: 0.961000 / std errors: 0.006122

######### n=100 [updated]
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.002040
mean of thetahat: 0.301689
median of thetahat: 0.300029
mean of classical variance estimate: 0.001138
median of classical variance estimate: 0.001132
mean of adaptive variance estimate: 0.003560
median of adaptive variance estimate: 0.002657
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.840000 / std errors: 0.011593
coverage rate of adaptive variance estiamte: 0.959000 / std errors: 0.006270

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=100
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.301689
median of thetahat: 0.300029
variance of thetahat: 0.002040
mean of classical variance estimate: 0.001138
median of classical variance estimate: 0.001132
mean of adaptive variance estimate: 0.003560
median of adaptive variance estimate: 0.002657
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.841000 / std errors: 0.011564
coverage rate of adaptive variance estiamte: 0.962000 / std errors: 0.006046

######### n=300 [updated]
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.000750
mean of thetahat: 0.303317
median of thetahat: 0.303054
mean of classical variance estimate: 0.000384
median of classical variance estimate: 0.000383
mean of adaptive variance estimate: 0.000941
median of adaptive variance estimate: 0.000810
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.840000 / std errors: 0.011593
coverage rate of adaptive variance estiamte: 0.948000 / std errors: 0.007021

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=300
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.303317
median of thetahat: 0.303054
variance of thetahat: 0.000750
mean of classical variance estimate: 0.000384
median of classical variance estimate: 0.000383
mean of adaptive variance estimate: 0.000941
median of adaptive variance estimate: 0.000810
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.841000 / std errors: 0.011564
coverage rate of adaptive variance estiamte: 0.940000 / std errors: 0.007510


######### n=500 [updated]
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.000453
mean of thetahat: 0.303996
median of thetahat: 0.304308
mean of classical variance estimate: 0.000230
median of classical variance estimate: 0.000230
mean of adaptive variance estimate: 0.000521
median of adaptive variance estimate: 0.000468
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.844000 / std errors: 0.011474
coverage rate of adaptive variance estiamte: 0.943000 / std errors: 0.007332

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=500
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.303996
median of thetahat: 0.304308
variance of thetahat: 0.000453
mean of classical variance estimate: 0.000230
median of classical variance estimate: 0.000230
mean of adaptive variance estimate: 0.000521
median of adaptive variance estimate: 0.000468
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.840000 / std errors: 0.011593
coverage rate of adaptive variance estiamte: 0.946000 / std errors: 0.007147

######### n=1000 
# only use the current n to evaluate the true thetahat
true variance of rewards: 0.000224
mean of thetahat: 0.304409
mean of classical variance estiamte: 0.000115
median of classical variance estiamte: 0.000115
mean of adaptive variance estiamte: 0.000254
median of adaptive variance estiamte: 0.000236
coverage rate of classical variance estiamte: 0.843000 / std errors: 0.011504
coverage rate of adaptive variance estiamte: 0.956000 / std errors: 0.006486

# use large n = 10000 to evaluate the true thetahat, which is more accurate.
Evaluation on the average reward inference when n=1000
true mean of thetahat when n=10000: 0.305089
true median of thetahat when n=10000: 0.305021
true variance of thetahat when n=10000: 0.000023
mean of thetahat: 0.304492
median of thetahat: 0.303800
variance of thetahat: 0.000225
mean of classical variance estimate: 0.000115
median of classical variance estimate: 0.000115
mean of adaptive variance estimate: 0.000254
median of adaptive variance estimate: 0.000239
number of successful experiments: 998/1000
coverage rate of classical variance estiamte: 0.861723 / std errors: 0.010927
coverage rate of adaptive variance estiamte: 0.957916 / std errors: 0.006356


"""


############################################################# results: SAC  #############################################################


"""

################# n=30
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=30: 0.397378
true median of thetahat when n=30: 0.399895
true variance of thetahat when n=30: 0.005268
mean of thetahat: 0.397378
median of thetahat: 0.399895
variance of thetahat: 0.005268
mean of classical variance estimate: 0.004167
median of classical variance estimate: 0.004072
mean of adaptive variance estimate: 0.223250
median of adaptive variance estimate: 0.009073
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.913000 / std errors: 0.008912
coverage rate of adaptive variance estiamte: 0.978000 / std errors: 0.004639

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.397378
median of thetahat: 0.399895
variance of thetahat: 0.005268
mean of classical variance estimate: 0.004167
median of classical variance estimate: 0.004072
mean of adaptive variance estimate: 0.223250
median of adaptive variance estimate: 0.009073
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.905000 / std errors: 0.009272
coverage rate of adaptive variance estiamte: 0.970000 / std errors: 0.005394

################# n=50
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=50: 0.431741
true median of thetahat when n=50: 0.430571
true variance of thetahat when n=50: 0.002875
mean of thetahat: 0.431741
median of thetahat: 0.430571
variance of thetahat: 0.002875
mean of classical variance estimate: 0.002382
median of classical variance estimate: 0.002351
mean of adaptive variance estimate: 0.136148
median of adaptive variance estimate: 0.005002
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.930000 / std errors: 0.008068
coverage rate of adaptive variance estiamte: 0.982000 / std errors: 0.004204

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.431741
median of thetahat: 0.430571
variance of thetahat: 0.002875
mean of classical variance estimate: 0.002382
median of classical variance estimate: 0.002351
mean of adaptive variance estimate: 0.136148
median of adaptive variance estimate: 0.005002
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.804000 / std errors: 0.012553
coverage rate of adaptive variance estiamte: 0.941000 / std errors: 0.007451

################# n=100
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=100: 0.443914
true median of thetahat when n=100: 0.443191
true variance of thetahat when n=100: 0.001358
mean of thetahat: 0.443914
median of thetahat: 0.443191
variance of thetahat: 0.001358
mean of classical variance estimate: 0.001150
median of classical variance estimate: 0.001142
mean of adaptive variance estimate: 0.022267
median of adaptive variance estimate: 0.001758
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.925000 / std errors: 0.008329
coverage rate of adaptive variance estiamte: 0.973000 / std errors: 0.005126

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.443914
median of thetahat: 0.443191
variance of thetahat: 0.001358
mean of classical variance estimate: 0.001150
median of classical variance estimate: 0.001142
mean of adaptive variance estimate: 0.022267
median of adaptive variance estimate: 0.001758
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.589000 / std errors: 0.015559
coverage rate of adaptive variance estiamte: 0.733000 / std errors: 0.013990

################# n=300
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=300: 0.433757
true median of thetahat when n=300: 0.433857
true variance of thetahat when n=300: 0.000535
mean of thetahat: 0.433757
median of thetahat: 0.433857
variance of thetahat: 0.000535
mean of classical variance estimate: 0.000381
median of classical variance estimate: 0.000378
mean of adaptive variance estimate: 0.000610
median of adaptive variance estimate: 0.000541
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.912000 / std errors: 0.008959
coverage rate of adaptive variance estiamte: 0.952000 / std errors: 0.006760

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.433757
median of thetahat: 0.433857
variance of thetahat: 0.000535
mean of classical variance estimate: 0.000381
median of classical variance estimate: 0.000378
mean of adaptive variance estimate: 0.000610
median of adaptive variance estimate: 0.000541
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.330000 / std errors: 0.014869
coverage rate of adaptive variance estiamte: 0.457000 / std errors: 0.015753

################# n=500
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=500: 0.426610
true median of thetahat when n=500: 0.426133
true variance of thetahat when n=500: 0.000337
mean of thetahat: 0.426610
median of thetahat: 0.426133
variance of thetahat: 0.000337
mean of classical variance estimate: 0.000227
median of classical variance estimate: 0.000227
mean of adaptive variance estimate: 0.000385
median of adaptive variance estimate: 0.000318
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.886000 / std errors: 0.010050
coverage rate of adaptive variance estiamte: 0.943000 / std errors: 0.007332

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.426610
median of thetahat: 0.426133
variance of thetahat: 0.000337
mean of classical variance estimate: 0.000227
median of classical variance estimate: 0.000227
mean of adaptive variance estimate: 0.000385
median of adaptive variance estimate: 0.000318
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.252000 / std errors: 0.013729
coverage rate of adaptive variance estiamte: 0.379000 / std errors: 0.015341

################# n=1000
### decayed ridge penalty (classical)
# each n
true mean of thetahat when n=1000: 0.414542
true median of thetahat when n=1000: 0.414513
true variance of thetahat when n=1000: 0.000188
mean of thetahat: 0.414542
median of thetahat: 0.414513
variance of thetahat: 0.000188
mean of classical variance estimate: 0.000113
median of classical variance estimate: 0.000113
mean of adaptive variance estimate: 0.000859
median of adaptive variance estimate: 0.000155
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.860000 / std errors: 0.010973
coverage rate of adaptive variance estiamte: 0.920000 / std errors: 0.008579

# large n = 10000
true mean of thetahat when n=10000: 0.385006
true median of thetahat when n=10000: 0.385421
true variance of thetahat when n=10000: 0.000053
mean of thetahat: 0.414542
median of thetahat: 0.414513
variance of thetahat: 0.000188
mean of classical variance estimate: 0.000113
median of classical variance estimate: 0.000113
mean of adaptive variance estimate: 0.000859
median of adaptive variance estimate: 0.000155
number of successful experiments: 1000/1000
coverage rate of classical variance estiamte: 0.259000 / std errors: 0.013853
coverage rate of adaptive variance estiamte: 0.354000 / std errors: 0.015122
"""
