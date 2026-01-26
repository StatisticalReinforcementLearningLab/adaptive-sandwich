import numpy as np
import pandas as pd
from argparse import ArgumentParser
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

parser = ArgumentParser(description="Parameters for the code - ARTD on gym envs")

parser.add_argument('--T', type=int, default=50, help="number of decision times")
parser.add_argument('--n', type=int, default=30, help="sample size")
parser.add_argument('--N_seed', type=int, default=10, help="N_seed")
parser.add_argument('--filename', type=str, default='', help="filename", choices=['','full', 'simplfied'])
args = parser.parse_args()
n=args.n
T=args.T
N_seed = args.N_seed
FILENAME = args.filename # full


########### true mean and variance
# path = '/n/netscratch/murphy_lab/Lab/kesun/2Longitudinal/adaptive-sandwich/'
path = ''

dfs = []
theta1_list = []
for i in tqdm(range(N_seed)):
    subpath = path + f"n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=smooth_posterior_sampling_T={T}_n={n}_partial{FILENAME}/exp=1"
    df = pd.read_csv(subpath+'/data.csv')
    # df2 = pd.read_csv(subpath+'/data_random.csv')
    # df = pd.concat([df1, df2], axis=0, ignore_index=True)
    # data = pd.read_csv('n100_T50/data.csv') # local laptop
    if FILENAME in ['', 'simplfied']:
        df_X = df[['intercept','Z_id']]
    elif FILENAME == 'full':
        df_X = df[['intercept','pretreat_feature1','pretreat_feature2','Z_id']]
    else:
        raise ValueError('Invalid filename')
    df_Y = df['reward']
    linear_model_single = LinearRegression(fit_intercept=False)
    linear_model_single.fit(df_X, df_Y)
    theta1_list.append(linear_model_single.coef_[-1])
    # print(df)
    # theta1_list.append(df.loc[df['Z_id']==1, 'reward'].mean() - df.loc[df['Z_id']==0, 'reward'].mean())
    if FILENAME in ['', 'simplfied']:
        dfs.append(df[['intercept','Z_id','reward']])
    elif FILENAME == 'full':
        dfs.append(df[['intercept','pretreat_feature1','pretreat_feature2','Z_id','reward']])
    else:
        raise ValueError('Invalid filename')

true_var_theta1 = np.var(np.array(theta1_list)) 
print(f'true variance of \hat(theta1): {true_var_theta1:.6f}')

df_all = pd.concat(dfs, axis=0, ignore_index=True)

intercept = df_all["intercept"].values.reshape(n*N_seed, T)[:,0:1]
ave_reward = df_all["reward"].values.reshape(n*N_seed, T).mean(axis=1, keepdims=True) # [n, 1]
if FILENAME in ['', 'simplfied']:
    pass
elif FILENAME == 'full':
    pretreat_features1 = df_all["pretreat_feature1"].values.reshape(n*N_seed, T)[:,0:1]
    pretreat_features2 = df_all["pretreat_feature2"].values.reshape(n*N_seed, T)[:,0:1]
else:
    raise ValueError('Invalid filename')
Z_id = df_all["Z_id"].values.reshape(n*N_seed, T)[:,0:1]


# C_design = np.hstack((intercept, pretreat_features1, pretreat_features2)) # [n, 3]

if FILENAME == 'simplfied':
    C_design_full = np.hstack((intercept, Z_id))
elif FILENAME == 'full':
    C_design_full = np.hstack((intercept, pretreat_features1, pretreat_features2, Z_id)) 
else:
    raise ValueError('Invalid filename')
# df = pd.DataFrame(np.hstack((intercept, pretreat_features1, pretreat_features2, Z_id, ave_reward)), columns=['intercept','pretreat_feature1','pretreat_feature2','Z_id', 'reward'])
# data['reward'] = ave_reward
# data.to_csv(f'data_n{n}_T{T}.csv', index=False)

##### inv
# CtC = C_design.T @ C_design
# CtC_inv = np.linalg.inv(CtC)
# P = C_design @ (CtC_inv @ C_design.T)
# e_y = ave_reward - P @ ave_reward
# e_z = Z_id - P @ Z_id
# theta_fwl = float((e_z.T @ e_y) / (e_z.T @ e_z))
# print(f'true mean of theta (FWL): {theta_fwl:.6f}', np.linalg.pinv(e_z) @ e_y)


# resi = ave_reward - C_design @ (np.linalg.pinv(C_design) @ ave_reward)
# resi_ = Z_id - C_design @ (np.linalg.pinv(C_design) @ Z_id) 
# true_theta_mean = np.linalg.pinv(resi_) @ resi
# true_theta_mean = true_theta_mean.flatten().item()
# print(f'true mean of theta (two-phase): {true_theta_mean:.6f}')

# [n, 4]

# true_theta_mean_pinv = np.linalg.pinv(C_design_full) @ ave_reward
# true_theta_mean_pinv = true_theta_mean_pinv.flatten()[-1]
# print(f'true mean of theta (pinv): {true_theta_mean_pinv:.6f}')


###### run a full linear regression to evaluate the true mean of theta
linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(C_design_full, ave_reward)
true_theta_mean = linear_model.coef_[0][-1]
print(f'true mean of theta (linear regression): {true_theta_mean:.6f}')

### estiamted mean and variance
theta_hat_list = []
classical_var_list = []
adaptive_var_list = []

for i in tqdm(range(N_seed)):
    subpath = path + f"n{n}_T{T}/{i}/simulated_data/synthetic_mode=delayed_1_action_dosage_alg=smooth_posterior_sampling_T={T}_n={n}_partial_{FILENAME}/exp=1"
    data = np.load(subpath+'/analysis.pkl', allow_pickle=True)
    theta_hat_list.append(data['theta_est'].item())
    classical_var_list.append(data['classical_sandwich_var_estimate'].item())
    adaptive_var_list.append(data['adaptive_sandwich_var_estimate'].item())


true_var = np.var(np.array(theta_hat_list)) 
print(f'true variance of theta: {true_var:.6f}')



print(f'mean of thetahat: {true_theta_mean:.6f}')
print(f'mean of classical variance estimate: {np.mean(classical_var_list, axis=0):.6f}')
print(f'median of classical variance estimate: {np.median(classical_var_list, axis=0):.6f}')
print(f'mean of adaptive variance estimate: {np.mean(adaptive_var_list, axis=0):.6f}' )
print(f'median of adaptive variance estimate: {np.median(adaptive_var_list, axis=0):.6f}' )


### converage rate

def Coverage(theta_hat, variance, true_mean):
    # confidence interval: theta_hat +- 1.96 * sqrt(variance) / n
    count = 0
    CI_width = 1.96 * np.sqrt(variance) #  the variance here is var/n
    low_bound = theta_hat - CI_width
    up_bound = theta_hat + CI_width
    if low_bound <= true_mean and up_bound >= true_mean:
        count += 1
    return count

l_classical_covered, l_adaptive_covered = [],[]
for i in range(N_seed):
    num_classical_covered = Coverage(theta_hat_list[i], classical_var_list[i], true_theta_mean)
    num_adaptive_covered = Coverage(theta_hat_list[i], adaptive_var_list[i], true_theta_mean)
    l_classical_covered.append(num_classical_covered)
    l_adaptive_covered.append(num_adaptive_covered)

l_classical_covered = np.array(l_classical_covered)
l_adaptive_covered = np.array(l_adaptive_covered)

print(f'coverage rate of classical variance estiamte: {l_classical_covered.mean():.6f} / std errors: {l_classical_covered.std() / np.sqrt(N_seed):.6f}')
print(f'coverage rate of adaptive variance estiamte: {l_adaptive_covered.mean():.6f} / std errors: {l_adaptive_covered.std() / np.sqrt(N_seed):.6f}')
print(args)




############################################################# results: TS  #############################################################
###### [partial_01envfeatures_inference]: 10/15, add C in both environments and inference

"""
##### n=30

Mean parameter estimate:
[0.15712914 0.13003476 0.00740601 0.09734621]

Empirical variance of parameter estimates:
[[ 7.81717176e-03 -7.72711755e-03 -2.44378817e-04  1.00062663e-04]
 [-7.72711755e-03  1.89196844e-02  2.81402890e-04 -8.88551995e-05]
 [-2.44378817e-04  2.81402890e-04  6.62372072e-03 -3.13739734e-03]
 [ 1.00062663e-04 -8.88551995e-05 -3.13739734e-03  5.95697597e-03]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00043975 0.00105069 0.00043975 0.00043975]
 [0.00105069 0.00105069 0.00105069 0.00105069]
 [0.00043975 0.00105069 0.00039953 0.00039953]
 [0.00043975 0.00105069 0.00039953 0.00035416]]

Mean adaptive sandwich variance estimate:
[[ 0.00865306 -0.01048793  0.00041532  0.00037535]
 [-0.01048794  0.0797765  -0.00071353 -0.00385377]
 [ 0.00041532 -0.00071353  0.01987583 -0.00982322]
 [ 0.00037535 -0.00385376 -0.00982321  0.02315233]]

Mean classical sandwich variance estimate:
[[ 6.9464585e-03 -6.9361757e-03 -4.6558318e-05  4.9199756e-05]
 [-6.9361757e-03  1.4072265e-02  1.1170364e-05 -3.2480169e-05]
 [-4.6558300e-05  1.1170434e-05  5.0303987e-03 -2.4562308e-03]
 [ 4.9199756e-05 -3.2480188e-05 -2.4562308e-03  4.9341456e-03]]

Median adaptive sandwich variance estimate:
[[ 7.3636505e-03 -6.9549726e-03 -1.0955885e-04  3.5452860e-05]
 [-6.9549726e-03  2.7703129e-02  2.7006710e-05  1.7222794e-04]
 [-1.0955788e-04  2.7005250e-05  9.7275116e-03 -4.1211359e-03]
 [ 3.5452154e-05  1.7222541e-04 -4.1211341e-03  9.1651790e-03]]

Median classical sandwich variance estimate:
[[ 6.4961212e-03 -6.4212522e-03 -3.5848348e-05  2.0196503e-05]
 [-6.4212517e-03  1.3528196e-02 -1.4157299e-04  2.5535133e-05]
 [-3.5848378e-05 -1.4157302e-04  4.3175309e-03 -2.0132530e-03]
 [ 2.0196529e-05  2.5534971e-05 -2.0132530e-03  4.3405904e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.962

Classical sandwich 95.0% standard normal CI coverage:
0.894


##### n=50

Mean parameter estimate:
[0.15698916 0.13621828 0.01102616 0.09599094]

Empirical variance of parameter estimates:
[[ 4.57253651e-03 -4.70730911e-03 -1.29730359e-04  7.98780986e-05]
 [-4.70730911e-03  1.19798024e-02  4.11900510e-04 -4.46809555e-04]
 [-1.29730359e-04  4.11900510e-04  3.38637239e-03 -1.62954940e-03]
 [ 7.98780986e-05 -4.46809555e-04 -1.62954940e-03  3.30192389e-03]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00024918 0.00066365 0.00024918 0.00024918]
 [0.00066365 0.00066365 0.00066365 0.00066365]
 [0.00024918 0.00066365 0.00018667 0.00018667]
 [0.00024918 0.00066365 0.00018667 0.00017807]]

Mean adaptive sandwich variance estimate:
[[ 4.6564899e-03 -4.6714772e-03 -1.6842286e-04  1.7569211e-05]
 [-4.6714777e-03  3.4825400e-02 -2.2390839e-03  2.6044720e-03]
 [-1.6842298e-04 -2.2390853e-03  1.0950172e-02 -6.0818335e-03]
 [ 1.7569317e-05  2.6044731e-03 -6.0818321e-03  1.1033015e-02]]

Mean classical sandwich variance estimate:
[[ 4.2752568e-03 -4.2681708e-03 -2.2346827e-05  8.3418927e-06]
 [-4.2681708e-03  8.6927759e-03  3.8967308e-05 -1.7987704e-05]
 [-2.2346836e-05  3.8967337e-05  3.0279967e-03 -1.5267927e-03]
 [ 8.3418990e-06 -1.7987697e-05 -1.5267925e-03  3.0123962e-03]]

Median adaptive sandwich variance estimate:
[[ 4.4479803e-03 -4.3429593e-03 -6.4642727e-06  4.2973967e-05]
 [-4.3429602e-03  1.7226841e-02  1.2368819e-04  3.2242722e-05]
 [-6.4643291e-06  1.2368902e-04  5.5347383e-03 -2.4617952e-03]
 [ 4.2973967e-05  3.2242890e-05 -2.4617927e-03  5.4184939e-03]]

Median classical sandwich variance estimate:
[[ 4.2035915e-03 -4.1922317e-03 -1.8851350e-05 -3.2680618e-05]
 [-4.1922312e-03  8.5263029e-03  2.7200749e-05 -1.3387571e-05]
 [-1.8851337e-05  2.7200629e-05  2.7720199e-03 -1.3466107e-03]
 [-3.2680648e-05 -1.3387643e-05 -1.3466107e-03  2.7551465e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.971


Classical sandwich 95.0% standard normal CI coverage:
0.884

##### n=100

Mean parameter estimate:
[0.1573859  0.1422507  0.00966151 0.09984484]

Empirical variance of parameter estimates:
[[ 2.32108692e-03 -2.35805136e-03  1.18102709e-04 -7.14531564e-05]
 [-2.35805136e-03  6.34026079e-03 -7.20783833e-05  4.47527964e-05]
 [ 1.18102709e-04 -7.20783833e-05  1.50090128e-03 -7.87721485e-04]
 [-7.14531564e-05  4.47527964e-05 -7.87721485e-04  1.54742673e-03]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[1.27131950e-04 3.35433452e-04 1.27131950e-04 1.27131950e-04]
 [3.35433452e-04 3.35433452e-04 3.35433452e-04 3.35433452e-04]
 [1.27131950e-04 3.35433452e-04 7.94693310e-05 8.38295671e-05]
 [1.27131950e-04 3.35433452e-04 8.38295671e-05 8.38295671e-05]]

Mean adaptive sandwich variance estimate:
[[ 2.24333908e-03 -2.27259332e-03  1.58457660e-05 -2.50452413e-05]
 [-2.27259309e-03  1.13852350e-02 -4.67495811e-05  1.03898303e-04]
 [ 1.58457951e-05 -4.67498357e-05  3.08924704e-03 -1.51191908e-03]
 [-2.50452722e-05  1.03898354e-04 -1.51191908e-03  3.17589752e-03]]

Mean classical sandwich variance estimate:
[[ 2.1925361e-03 -2.1934719e-03  6.2793879e-06  5.2922019e-06]
 [-2.1934719e-03  4.4393102e-03 -5.3471481e-06 -3.2102034e-05]
 [ 6.2793888e-06 -5.3471458e-06  1.5050212e-03 -7.6594960e-04]
 [ 5.2921991e-06 -3.2102023e-05 -7.6594960e-04  1.5239984e-03]]

Median adaptive sandwich variance estimate:
[[ 2.2202113e-03 -2.1954495e-03  1.0131691e-05  1.3367276e-05]
 [-2.1954500e-03  7.8176325e-03 -5.2951116e-05  1.4816784e-05]
 [ 1.0131355e-05 -5.2951371e-05  2.3801445e-03 -1.1473638e-03]
 [ 1.3367233e-05  1.4816932e-05 -1.1473637e-03  2.4514995e-03]]

Median classical sandwich variance estimate:
[[ 2.1637501e-03 -2.1636744e-03  8.7577200e-06  6.8228164e-06]
 [-2.1636749e-03  4.3996195e-03 -3.7277910e-06 -2.7452852e-05]
 [ 8.7576136e-06 -3.7277894e-06  1.4385545e-03 -7.1888365e-04]
 [ 6.8228151e-06 -2.7452834e-05 -7.1888370e-04  1.4650859e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9677093844601413

Classical sandwich 95.0% standard normal CI coverage:
0.8980827447023209

##### n=300

Mean parameter estimate:
[0.15650621 0.145417   0.00954527 0.1002373 ]

Empirical variance of parameter estimates:
[[ 7.88873718e-04 -7.59425418e-04  1.84270187e-07 -7.24346007e-06]
 [-7.59425418e-04  2.11213185e-03 -1.10925304e-05  2.61650786e-05]
 [ 1.84270187e-07 -1.10925304e-05  5.49305780e-04 -2.62541425e-04]
 [-7.24346007e-06  2.61650786e-05 -2.62541425e-04  4.91172565e-04]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[4.51664970e-05 1.18881866e-04 4.51664970e-05 4.51664970e-05]
 [1.18881866e-04 1.18881866e-04 1.18881866e-04 1.18881866e-04]
 [4.51664970e-05 1.18881866e-04 3.25668022e-05 3.25668022e-05]
 [4.51664970e-05 1.18881866e-04 3.25668022e-05 2.61360483e-05]]

Mean adaptive sandwich variance estimate:
[[ 7.4372062e-04 -7.4247341e-04  1.7516372e-06 -6.9347340e-07]
 [-7.4247341e-04  2.9452110e-03 -1.9922837e-07  3.2951801e-05]
 [ 1.7516351e-06 -1.9922376e-07  7.0850004e-04 -3.5436486e-04]
 [-6.9347283e-07  3.2951797e-05 -3.5436492e-04  6.9978746e-04]]

Mean classical sandwich variance estimate:
[[ 7.4169179e-04 -7.4175542e-04  1.0712337e-06 -3.1494267e-07]
 [-7.4175536e-04  1.5064322e-03 -1.6116103e-06  7.3907913e-08]
 [ 1.0712336e-06 -1.6116093e-06  5.0487596e-04 -2.5089338e-04]
 [-3.1494270e-07  7.3907543e-08 -2.5089338e-04  4.9957848e-04]]

Median adaptive sandwich variance estimate:
[[ 7.4012898e-04 -7.3638716e-04  2.3267580e-06 -1.0211265e-06]
 [-7.3638727e-04  2.4244864e-03  7.0293123e-07  1.7293667e-05]
 [ 2.3267278e-06  7.0298171e-07  6.5305718e-04 -3.2159919e-04]
 [-1.0211265e-06  1.7293629e-05 -3.2159919e-04  6.4077077e-04]]

Median classical sandwich variance estimate:
[[ 7.3863019e-04 -7.3684019e-04  1.4099897e-06 -2.8343472e-06]
 [-7.3684024e-04  1.5008608e-03 -3.0792148e-06  1.1416740e-06]
 [ 1.4099838e-06 -3.0792251e-06  4.9722136e-04 -2.4715584e-04]
 [-2.8343459e-06  1.1416693e-06 -2.4715581e-04  4.9219071e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.95995995995996


Classical sandwich 95.0% standard normal CI coverage:
0.9049049049049049


##### n=500

Mean parameter estimate:
[0.15823668 0.1453746  0.00965101 0.10107199]

Empirical variance of parameter estimates:
[[ 4.47942490e-04 -4.40673255e-04 -6.00117370e-06  8.29560298e-06]
 [-4.40673255e-04  1.33133568e-03 -1.31771069e-05  3.65176062e-05]
 [-6.00117370e-06 -1.31771069e-05  3.23474654e-04 -1.62240461e-04]
 [ 8.29560298e-06  3.65176062e-05 -1.62240461e-04  3.08645430e-04]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[2.54164859e-05 7.53247895e-05 2.54164859e-05 2.54164859e-05]
 [7.53247895e-05 7.53247895e-05 7.53247895e-05 7.53247895e-05]
 [2.54164859e-05 7.53247895e-05 1.73359454e-05 1.73359454e-05]
 [2.54164859e-05 7.53247895e-05 1.73359454e-05 1.60969466e-05]]

Mean adaptive sandwich variance estimate:
[[ 4.4491034e-04 -4.4499859e-04 -1.8311063e-06 -3.3950471e-07]
 [-4.4499859e-04  1.5630804e-03 -5.1511842e-07  1.8574598e-06]
 [-1.8311064e-06 -5.1512291e-07  3.7833583e-04 -1.8926835e-04]
 [-3.3950491e-07  1.8574628e-06 -1.8926836e-04  3.8209249e-04]]

Mean classical sandwich variance estimate:
[[ 4.4441322e-04 -4.4440111e-04 -1.3910376e-06 -1.0330369e-07]
 [-4.4440114e-04  9.0309611e-04  8.4382737e-07 -5.0985494e-07]
 [-1.3910374e-06  8.4382782e-07  3.0042062e-04 -1.4948063e-04]
 [-1.0330387e-07 -5.0985483e-07 -1.4948063e-04  2.9982466e-04]]

Median adaptive sandwich variance estimate:
[[ 4.4379802e-04 -4.4259097e-04 -2.4355550e-06 -8.9604583e-08]
 [-4.4259103e-04  1.3764334e-03 -3.5114913e-06  4.1853775e-07]
 [-2.4355575e-06 -3.5115004e-06  3.6175846e-04 -1.7870305e-04]
 [-8.9610694e-08  4.1856197e-07 -1.7870305e-04  3.6289013e-04]]

Median classical sandwich variance estimate:
[[ 4.4330128e-04 -4.4318548e-04 -1.8496835e-06  2.0915924e-07]
 [-4.4318553e-04  9.0210827e-04  1.9063201e-06 -1.7267314e-06]
 [-1.8496821e-06  1.9063241e-06  2.9697808e-04 -1.4757458e-04]
 [ 2.0916090e-07 -1.7267316e-06 -1.4757458e-04  2.9620060e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.953


Classical sandwich 95.0% standard normal CI coverage:
0.891

#### n=1000


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



