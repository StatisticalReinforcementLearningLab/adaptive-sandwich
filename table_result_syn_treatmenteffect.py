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
parser.add_argument('--reg', type=float, default=0.1, help="ridge regression parameter")
parser.add_argument('--reg_true', type=float, default=0.1, help="ridge regression parameter for evaluation")
parser.add_argument('--N_seed', type=int, default=1000, help="N_seed")
parser.add_argument('--evaluate', type=int, default=0, help="evaluate the true thetahat on a large n")
args = parser.parse_args()
n=args.n
T=args.T
reg = args.reg
reg_true = args.reg_true
N_seed = args.N_seed

alpha1 = 0.01
alpha2 = 0.1
algo_name = 'smooth_posterior_sampling' if args.alg == 'TS' else 'sac'
n_true = 10000 
# n_true = args.n

######################################## Step 1: access all replications, compute each average rewards, assess the true theta1
path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'

if args.evaluate == 1:
    theta1_list_true = []
    for i in tqdm(range(N_seed)):
        if args.alg == 'TS': # old path
            subpath = path + f"syn_TS_n{n_true}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
        else: # SAC
            subpath = path + f"syn_sac_n{n_true}_T{T}_reg{reg_true}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n_true}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1" # use the largest n to compute true variance
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
        true_theta1_mean = 0.1
        true_theta1_median = 0.1
        true_var_theta1 = 0.0
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
        subpath = path + f"syn_sac_n{n}_T{T}_reg{reg}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg={algo_name}_T={T}_n={n}_treatmenteffect_alpha1{alpha1}_alpha2{alpha2}/exp=1"
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
Mean parameter estimate:
[0.15643144 0.11524222 0.0069426  0.10920726]

Empirical variance of parameter estimates:
[[ 0.007826   -0.00774365  0.00013311  0.00030865]
 [-0.00774365  0.01956957  0.00013125 -0.00066964]
 [ 0.00013311  0.00013125  0.00659428 -0.00332849]
 [ 0.00030865 -0.00066964 -0.00332849  0.00641364]]

Mean adaptive sandwich variance estimate:
[[ 7.1914787e-03 -7.0899217e-03  7.3448151e-05 -2.4757679e-05]
 [-7.0899222e-03  2.0704076e-02  1.5522722e-04 -1.8698040e-04]
 [ 7.3448107e-05  1.5522729e-04  7.1038483e-03 -3.5856392e-03]
 [-2.4757637e-05 -1.8698045e-04 -3.5856396e-03  7.0588291e-03]]

Mean classical sandwich variance estimate:
[[ 7.0069460e-03 -6.9747390e-03  5.1918472e-05 -2.1687383e-05]
 [-6.9747390e-03  1.5059580e-02  6.0809921e-06 -7.3927447e-05]
 [ 5.1918516e-05  6.0810080e-06  5.2379258e-03 -2.6130001e-03]
 [-2.1687414e-05 -7.3927498e-05 -2.6129999e-03  5.1870048e-03]]

Median adaptive sandwich variance estimate:
[[ 6.77447580e-03 -6.74793124e-03  1.43074518e-04  1.35341616e-05]
 [-6.74793078e-03  1.84118487e-02  1.06577107e-04 -7.82176940e-05]
 [ 1.43073499e-04  1.06576947e-04  5.97175770e-03 -2.78622797e-03]
 [ 1.35344935e-05 -7.82180214e-05 -2.78622843e-03  5.63397072e-03]]

Median classical sandwich variance estimate:
[[ 6.6320570e-03 -6.6012908e-03  1.2978146e-04 -9.1832062e-06]
 [-6.6012908e-03  1.4490068e-02 -4.8717980e-06 -7.7435638e-05]
 [ 1.2978137e-04 -4.8708953e-06  4.6767415e-03 -2.2741691e-03]
 [-9.1828633e-06 -7.7435703e-05 -2.2741696e-03  4.5328466e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9236947791164659

Classical sandwich 95.0% standard normal CI coverage:
0.9006024096385542

## [epoch=1000, threshold=1e-7 to contribute more to the zero estimation function]
Mean parameter estimate:
[0.15639761 0.11519118 0.00747066 0.10862108]

Empirical variance of parameter estimates:
[[ 0.00780855 -0.00772102  0.00013218  0.00031828]
 [-0.00772102  0.01961319  0.00016491 -0.00066293]
 [ 0.00013218  0.00016491  0.00668535 -0.00339302]
 [ 0.00031828 -0.00066293 -0.00339302  0.00647503]]

Mean adaptive sandwich variance estimate:
[[ 0.00728956 -0.00706018  0.00047254 -0.00018883]
 [-0.00706018  0.02121208  0.00052804 -0.00049947]
 [ 0.00047254  0.00052804  0.00862727 -0.00422598]
 [-0.00018883 -0.00049947 -0.00422598  0.00736415]]

Mean classical sandwich variance estimate:
[[ 7.0068818e-03 -6.9732531e-03  4.9439666e-05 -2.5056015e-05]
 [-6.9732536e-03  1.5057282e-02  6.5463969e-06 -6.9414011e-05]
 [ 4.9439699e-05  6.5463664e-06  5.2406811e-03 -2.6167470e-03]
 [-2.5056059e-05 -6.9414047e-05 -2.6167470e-03  5.1888875e-03]]

Median adaptive sandwich variance estimate:
[[ 6.76653115e-03 -6.71494938e-03  1.52717650e-04  2.77078470e-07]
 [-6.71494845e-03  1.83745697e-02  1.19263139e-04 -6.42267551e-05]
 [ 1.52718378e-04  1.19262964e-04  5.94255701e-03 -2.79027782e-03]
 [ 2.77286063e-07 -6.42267041e-05 -2.79027806e-03  5.64690167e-03]]

Median classical sandwich variance estimate:
[[ 6.6257566e-03 -6.5908260e-03  1.2978146e-04 -7.8981848e-06]
 [-6.5908264e-03  1.4490068e-02 -4.8717980e-06 -7.1030110e-05]
 [ 1.2978137e-04 -4.8708953e-06  4.6862187e-03 -2.2832765e-03]
 [-7.8976627e-06 -7.1030256e-05 -2.2832770e-03  4.5259488e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.923

Classical sandwich 95.0% standard normal CI coverage:
0.9


################# n=50
Mean parameter estimate:
[0.15738548 0.11578687 0.01323248 0.1002    ]

Empirical variance of parameter estimates:
[[ 4.40261022e-03 -4.55480197e-03 -1.08680904e-04  6.31479091e-06]
 [-4.55480197e-03  1.16783312e-02  7.50287419e-05 -3.13517815e-05]
 [-1.08680904e-04  7.50287419e-05  3.64889499e-03 -1.80038992e-03]
 [ 6.31479091e-06 -3.13517815e-05 -1.80038992e-03  3.41279073e-03]]

Mean adaptive sandwich variance estimate:
[[ 4.3583834e-03 -4.3454510e-03 -1.3590394e-05  1.9071198e-05]
 [-4.3454515e-03  1.1645216e-02 -7.6065353e-06  4.0998755e-05]
 [-1.3590398e-05 -7.6065580e-06  3.8661766e-03 -1.9359377e-03]
 [ 1.9071225e-05  4.0998741e-05 -1.9359376e-03  3.8813609e-03]]

Mean classical sandwich variance estimate:
[[ 4.3171444e-03 -4.3098466e-03 -1.2874476e-05  1.4113235e-05]
 [-4.3098466e-03  9.2925020e-03 -3.3918657e-06  1.0742562e-05]
 [-1.2874467e-05 -3.3918725e-06  3.1868485e-03 -1.6019146e-03]
 [ 1.4113232e-05  1.0742572e-05 -1.6019145e-03  3.1855504e-03]]

Median adaptive sandwich variance estimate:
[[ 4.2524366e-03 -4.2372788e-03 -3.2339947e-06 -3.9209772e-05]
 [-4.2372784e-03  1.0973946e-02  3.6975598e-06 -9.6247768e-06]
 [-3.2341800e-06  3.6974702e-06  3.4683037e-03 -1.6438222e-03]
 [-3.9209841e-05 -9.6248850e-06 -1.6438222e-03  3.4871325e-03]]

Median classical sandwich variance estimate:
[[ 4.22372203e-03 -4.19766735e-03 -1.15621415e-05 -3.98038392e-05]
 [-4.19766735e-03  9.10305511e-03  1.70076983e-05 -4.27044870e-05]
 [-1.15618823e-05  1.70077947e-05  2.95121875e-03 -1.40100776e-03]
 [-3.98040720e-05 -4.27043669e-05 -1.40100776e-03  2.95141689e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9359359359359359

Classical sandwich 95.0% standard normal CI coverage:
0.9109109109109109

################# n=100
Mean parameter estimate:
[0.15719199 0.11846436 0.01021192 0.10494081]

Empirical variance of parameter estimates:
[[ 2.32827957e-03 -2.36267844e-03  5.61628327e-05 -1.21611040e-04]
 [-2.36267844e-03  5.72757608e-03  6.30093179e-05  2.19619091e-04]
 [ 5.61628327e-05  6.30093179e-05  1.68074556e-03 -8.76816132e-04]
 [-1.21611040e-04  2.19619091e-04 -8.76816132e-04  1.72750202e-03]]

Mean adaptive sandwich variance estimate:
[[ 2.19552428e-03 -2.20092712e-03  1.24912349e-05 -4.36754544e-06]
 [-2.20092712e-03  5.57145569e-03  1.83849243e-05 -1.82760996e-05]
 [ 1.24912185e-05  1.83849334e-05  1.78693258e-03 -8.91034259e-04]
 [-4.36754135e-06 -1.82761141e-05 -8.91034259e-04  1.77507766e-03]]

Mean classical sandwich variance estimate:
[[ 2.19050236e-03 -2.19402090e-03  1.20978275e-05 -3.69909890e-06]
 [-2.19402066e-03  4.76095220e-03  2.02511387e-06 -1.47474602e-05]
 [ 1.20978175e-05  2.02511114e-06  1.61237631e-03 -8.04396288e-04]
 [-3.69910026e-06 -1.47474702e-05 -8.04396230e-04  1.59614556e-03]]

Median adaptive sandwich variance estimate:
[[ 2.1521593e-03 -2.1597217e-03  1.8659910e-05 -5.1833940e-06]
 [-2.1597217e-03  5.4140184e-03  4.0369825e-05 -2.3847209e-05]
 [ 1.8659905e-05  4.0369938e-05  1.6998332e-03 -8.3887530e-04]
 [-5.1834231e-06 -2.3847228e-05 -8.3887542e-04  1.6581630e-03]]

Median classical sandwich variance estimate:
[[ 2.1457020e-03 -2.1530127e-03  2.1567681e-05 -4.9833843e-06]
 [-2.1530124e-03  4.7422699e-03  2.6744689e-05 -2.2551945e-05]
 [ 2.1567630e-05  2.6744619e-05  1.5382734e-03 -7.6417578e-04]
 [-4.9833397e-06 -2.2551851e-05 -7.6417578e-04  1.5016809e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.946

Classical sandwich 95.0% standard normal CI coverage:
0.925

################# n=300
Mean parameter estimate:
[0.15642679 0.12007388 0.01024075 0.10387681]

Empirical variance of parameter estimates:
[[ 7.90471398e-04 -7.41214642e-04 -1.11068677e-06  1.51976504e-05]
 [-7.41214642e-04  1.88846439e-03 -3.55111246e-05  1.82746514e-06]
 [-1.11068677e-06 -3.55111246e-05  5.67259842e-04 -2.81671771e-04]
 [ 1.51976504e-05  1.82746514e-06 -2.81671771e-04  5.83432045e-04]]


Mean adaptive sandwich variance estimate:
[[ 7.65481207e-04 -6.17136597e-04 -1.19507597e-04  4.71572421e-05]
 [-6.17136655e-04  2.41336669e-03 -6.31802424e-04  2.53047328e-04]
 [-1.19507655e-04 -6.31802541e-04  1.16121792e-03 -5.18586137e-04]
 [ 4.71574676e-05  2.53047474e-04 -5.18586545e-04  6.50887610e-04]]

Mean classical sandwich variance estimate:
[[ 7.4188859e-04 -7.4154971e-04 -5.5221454e-07  2.6003005e-07]
 [-7.4154971e-04  1.6080234e-03  3.4973905e-07 -9.8121370e-07]
 [-5.5221381e-07  3.4973945e-07  5.3669396e-04 -2.7049772e-04]
 [ 2.6003093e-07 -9.8121245e-07 -2.7049772e-04  5.3657463e-04]]

Median adaptive sandwich variance estimate:
[[ 7.3753227e-04 -7.3881418e-04 -2.6530538e-06  4.4619463e-07]
 [-7.3881412e-04  1.7330154e-03  5.7568122e-06  2.5796677e-08]
 [-2.6530477e-06  5.7568295e-06  5.4898742e-04 -2.7448119e-04]
 [ 4.4620799e-07  2.5813417e-08 -2.7448114e-04  5.5032934e-04]]

Median classical sandwich variance estimate:
[[ 7.3630706e-04 -7.3879235e-04 -2.9432479e-06 -3.9381342e-07]
 [-7.3879235e-04  1.6011922e-03  4.1876010e-06 -4.1630119e-06]
 [-2.9432770e-06  4.1875801e-06  5.2917370e-04 -2.6443682e-04]
 [-3.9381752e-07 -4.1630024e-06 -2.6443682e-04  5.2902324e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.933

Classical sandwich 95.0% standard normal CI coverage:
0.925

################# n=500
Mean parameter estimate:
[0.15815185 0.11881006 0.01022531 0.10433598]

Empirical variance of parameter estimates:
[[ 4.42398306e-04 -4.28774820e-04  9.13392784e-06  2.54636101e-06]
 [-4.28774820e-04  1.19587200e-03  2.85676220e-06 -1.12344983e-06]
 [ 9.13392784e-06  2.85676220e-06  3.34479786e-04 -1.75266740e-04]
 [ 2.54636101e-06 -1.12344983e-06 -1.75266740e-04  3.31841936e-04]]

Mean adaptive sandwich variance estimate:
[[ 4.4529641e-04 -4.4528782e-04 -9.8525686e-07 -1.6502399e-07]
 [-4.4528782e-04  1.0406240e-03  3.1541601e-06  1.3654180e-06]
 [-9.8525675e-07  3.1541585e-06  3.3030388e-04 -1.6609095e-04]
 [-1.6502446e-07  1.3654191e-06 -1.6609092e-04  3.3057685e-04]]

Mean classical sandwich variance estimate:
[[ 4.4524224e-04 -4.4521948e-04 -1.0002227e-06 -1.4678695e-07]
 [-4.4521957e-04  9.6730463e-04  2.5707548e-06 -2.6846880e-07]
 [-1.0002225e-06  2.5707566e-06  3.2301687e-04 -1.6246787e-04]
 [-1.4678686e-07 -2.6846925e-07 -1.6246788e-04  3.2316975e-04]]

Median adaptive sandwich variance estimate:
[[ 4.4395970e-04 -4.4398534e-04 -2.2698873e-06  2.0940377e-07]
 [-4.4398534e-04  1.0403958e-03  4.5239995e-06  2.3863108e-06]
 [-2.2698855e-06  4.5239854e-06  3.2878859e-04 -1.6434060e-04]
 [ 2.0939913e-07  2.3863308e-06 -1.6434060e-04  3.2921508e-04]]

Median classical sandwich variance estimate:
[[ 4.4391240e-04 -4.4397637e-04 -2.1958131e-06  2.1585922e-07]
 [-4.4397634e-04  9.6620910e-04  4.4209110e-06 -6.4933965e-07]
 [-2.1958144e-06  4.4209237e-06  3.2232518e-04 -1.6143723e-04]
 [ 2.1586914e-07 -6.4934540e-07 -1.6143723e-04  3.2124147e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.936

Classical sandwich 95.0% standard normal CI coverage:
0.926

"""



