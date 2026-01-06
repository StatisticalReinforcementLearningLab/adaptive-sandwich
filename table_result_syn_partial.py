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



###### results[partial_0]: 10/04
"""
######### n=30
Mean parameter estimate:
[0.22249842]

Empirical variance of parameter estimates:
[[0.00454615]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00024887]]

Mean adaptive sandwich variance estimate:
[[0.01548968]]

Mean classical sandwich variance estimate:
[[0.00383807]]

Median adaptive sandwich variance estimate:
[[0.0080901]]

Median classical sandwich variance estimate:
[[0.00377557]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.978

Classical sandwich 95.0% standard normal CI coverage:
0.925

######### n=50
Mean parameter estimate:
[0.2247158]

Empirical variance of parameter estimates:
[[0.00288899]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00015577]]

Mean adaptive sandwich variance estimate:
[[0.00938615]]

Mean classical sandwich variance estimate:
[[0.00233266]]

Median adaptive sandwich variance estimate:
[[0.0048192]]

Median classical sandwich variance estimate:
[[0.00228628]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.987

Classical sandwich 95.0% standard normal CI coverage:
0.918

######### n=100
Mean parameter estimate:
[0.22879328]

Empirical variance of parameter estimates:
[[0.00148607]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[8.03358925e-05]]

Mean adaptive sandwich variance estimate:
[[0.00294364]]

Mean classical sandwich variance estimate:
[[0.00117752]]

Median adaptive sandwich variance estimate:
[[0.0021057]]

Median classical sandwich variance estimate:
[[0.00117365]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.971

Classical sandwich 95.0% standard normal CI coverage:
0.913

######### n=300

Mean parameter estimate:
[0.22901689]

Empirical variance of parameter estimates:
[[0.00054021]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[2.86467286e-05]]

Mean adaptive sandwich variance estimate:
[[0.00075349]]

Mean classical sandwich variance estimate:
[[0.000396]]

Median adaptive sandwich variance estimate:
[[0.00062922]]

Median classical sandwich variance estimate:
[[0.0003948]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.960960960960961

Classical sandwich 95.0% standard normal CI coverage:
0.9029029029029029

######### n=500

Mean parameter estimate:
[0.2308117]

Empirical variance of parameter estimates:
[[0.00033679]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[1.92833415e-05]]

Mean adaptive sandwich variance estimate:
[[0.00041026]]

Mean classical sandwich variance estimate:
[[0.00023715]]

Median adaptive sandwich variance estimate:
[[0.00036184]]

Median classical sandwich variance estimate:
[[0.00023693]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.96

Classical sandwich 95.0% standard normal CI coverage:
0.9

######### n=1000


"""

###### results[partial_01]: 10/04
"""
######### n=30
Mean parameter estimate:
[0.1562236  0.13254973]

Empirical variance of parameter estimates:
[[ 0.00707705 -0.0070573 ]
 [-0.0070573   0.01810563]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00040562 0.00100349]
 [0.00100349 0.00100349]]

Mean adaptive sandwich variance estimate:
[[ 0.00695902 -0.00695902]
 [-0.00695902  0.05738722]]

Mean classical sandwich variance estimate:
[[ 0.00695903 -0.00695903]
 [-0.00695903  0.01416372]]

Median adaptive sandwich variance estimate:
[[ 0.00663255 -0.00663255]
 [-0.00663255  0.02995899]]

Median classical sandwich variance estimate:
[[ 0.00663256 -0.00663256]
 [-0.00663256  0.01400174]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.974

Classical sandwich 95.0% standard normal CI coverage:
0.901

######### n=50
Mean parameter estimate:
[0.15666448 0.13610247]

Empirical variance of parameter estimates:
[[ 0.00436625 -0.00439886]
 [-0.00439886  0.01168642]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00023567 0.00063981]
 [0.00063981 0.00063981]]

Mean adaptive sandwich variance estimate:
[[ 0.00428456 -0.00428456]
 [-0.00428456  0.03511086]]

Mean classical sandwich variance estimate:
[[ 0.00428455 -0.00428455]
 [-0.00428455  0.00872665]]

Median adaptive sandwich variance estimate:
[[ 0.00419367 -0.00419367]
 [-0.00419367  0.0173477 ]]

Median classical sandwich variance estimate:
[[ 0.00419366 -0.00419366]
 [-0.00419366  0.00863468]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.968

Classical sandwich 95.0% standard normal CI coverage:
0.902

######### n=100
Mean parameter estimate:
[0.15746553 0.14265543]

Empirical variance of parameter estimates:
[[ 0.00225233 -0.00231909]
 [-0.00231909  0.00621132]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00012265 0.00032685]
 [0.00032685 0.00032685]]

Mean adaptive sandwich variance estimate:
[[ 0.00218823 -0.00218823]
 [-0.00218823  0.01115876]]

Mean classical sandwich variance estimate:
[[ 0.00218823 -0.00218823]
 [-0.00218823  0.00444451]]

Median adaptive sandwich variance estimate:
[[ 0.00214917 -0.00214917]
 [-0.00214917  0.0079287 ]]

Median classical sandwich variance estimate:
[[ 0.00214917 -0.00214917]
 [-0.00214917  0.00442925]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.969

Classical sandwich 95.0% standard normal CI coverage:
0.893

######### n=300

Mean parameter estimate:
[0.15639773 0.14521033]

Empirical variance of parameter estimates:
[[ 0.00078483 -0.00076579]
 [-0.00076579  0.00208446]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[4.49509466e-05 1.18251895e-04]
 [1.18251895e-04 1.18251895e-04]]

Mean adaptive sandwich variance estimate:
[[ 0.00074122 -0.00074122]
 [-0.00074122  0.00288367]]

Mean classical sandwich variance estimate:
[[ 0.00074122 -0.00074122]
 [-0.00074122  0.00150656]]

Median adaptive sandwich variance estimate:
[[ 0.00073796 -0.00073796]
 [-0.00073796  0.00239358]]

Median classical sandwich variance estimate:
[[ 0.00073796 -0.00073796]
 [-0.00073796  0.00150396]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.954954954954955


Classical sandwich 95.0% standard normal CI coverage:
0.9119119119119119


######### n=500

Mean parameter estimate:
[0.15823898 0.14514528]

Empirical variance of parameter estimates:
[[ 0.00044202 -0.00043452]
 [-0.00043452  0.00131715]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[2.49861895e-05 7.45336375e-05]
 [7.45336375e-05 7.45336375e-05]]

Mean adaptive sandwich variance estimate:
[[ 0.00044449 -0.00044449]
 [-0.00044449  0.0015757 ]]

Mean classical sandwich variance estimate:
[[ 0.00044448 -0.00044448]
 [-0.00044448  0.00090383]]

Median adaptive sandwich variance estimate:
[[ 0.00044327 -0.00044327]
 [-0.00044327  0.00138789]]

Median classical sandwich variance estimate:
[[ 0.00044326 -0.00044326]
 [-0.00044326  0.000904  ]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.949

Classical sandwich 95.0% standard normal CI coverage:
0.897

######### n=1000


"""


###### results[partial_01feature]: 10/04
"""
######### n=30
Mean parameter estimate:
[0.15655148 0.13298166 0.00197849 0.00028855]

Empirical variance of parameter estimates:
[[ 0.00747215 -0.00742038 -0.00020626  0.00030146]
 [-0.00742038  0.01907834  0.00054832 -0.0003346 ]
 [-0.00020626  0.00054832  0.00569991 -0.00282771]
 [ 0.00030146 -0.0003346  -0.00282771  0.00583221]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00042016 0.00102416 0.00042016 0.00042016]
 [0.00102416 0.00102416 0.00102416 0.00102416]
 [0.00042016 0.00102416 0.00031317 0.00032602]
 [0.00042016 0.00102416 0.00032602 0.00032602]]

Mean adaptive sandwich variance estimate:
[[ 8.4800897e-03 -8.3248308e-03 -6.6106244e-05 -1.8207106e-04]
 [-8.3248308e-03  5.7028174e-02 -1.7050948e-04  1.8426389e-03]
 [-6.6106833e-05 -1.7051064e-04  1.9741423e-02 -1.0175218e-02]
 [-1.8207148e-04  1.8426382e-03 -1.0175220e-02  2.3503635e-02]]

Mean classical sandwich variance estimate:
[[ 7.00015482e-03 -6.99360110e-03 -1.02587670e-04  1.67649923e-05]
 [-6.99360063e-03  1.42292352e-02  8.50888318e-05 -4.16769835e-05]
 [-1.02587655e-04  8.50887882e-05  5.06989053e-03 -2.56807706e-03]
 [ 1.67650014e-05 -4.16769690e-05 -2.56807706e-03  5.00360550e-03]]

Median adaptive sandwich variance estimate:
[[ 7.43028894e-03 -7.04649743e-03 -2.44440394e-04 -1.12411335e-05]
 [-7.04649696e-03  2.96461470e-02 -1.67974678e-04  2.85160622e-05]
 [-2.44440278e-04 -1.67974707e-04  9.52514261e-03 -4.33684792e-03]
 [-1.12410562e-05  2.85182141e-05 -4.33685444e-03  9.34806094e-03]]

Median classical sandwich variance estimate:
[[ 6.6437842e-03 -6.6414494e-03 -1.0880448e-04 -3.6039157e-05]
 [-6.6414503e-03  1.3761424e-02 -1.0927919e-04 -4.1983018e-05]
 [-1.0880445e-04 -1.0927929e-04  4.4009341e-03 -2.1361001e-03]
 [-3.6038840e-05 -4.1982872e-05 -2.1361001e-03  4.3483069e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.975

Classical sandwich 95.0% standard normal CI coverage:
0.895

######### n=50
Mean parameter estimate:
[ 0.15707113  0.135732   -0.00059519  0.0008005 ]

Empirical variance of parameter estimates:
[[ 4.55313861e-03 -4.64466095e-03  2.01837062e-04  1.20553900e-05]
 [-4.64466095e-03  1.22122003e-02 -3.38637459e-04  2.00426025e-04]
 [ 2.01837062e-04 -3.38637459e-04  3.31179476e-03 -1.58379420e-03]
 [ 1.20553900e-05  2.00426025e-04 -1.58379420e-03  3.37637801e-03]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00024542 0.00066569 0.00024542 0.00024542]
 [0.00066569 0.00066569 0.00066569 0.00066569]
 [0.00024542 0.00066569 0.00017291 0.00018742]
 [0.00024542 0.00066569 0.00018742 0.00018742]]

Mean adaptive sandwich variance estimate:
[[ 4.7252839e-03 -4.6830815e-03 -5.9741964e-05  1.9709318e-05]
 [-4.6830815e-03  3.4368314e-02  2.7896222e-04  1.5357130e-03]
 [-5.9741964e-05  2.7896246e-04  9.9702049e-03 -4.8113116e-03]
 [ 1.9709303e-05  1.5357133e-03 -4.8113111e-03  1.0095515e-02]]

Mean classical sandwich variance estimate:
[[ 4.2834533e-03 -4.2779376e-03 -2.8819875e-06 -1.0872937e-05]
 [-4.2779376e-03  8.7298015e-03 -1.5798030e-05  7.4690390e-05]
 [-2.8819916e-06 -1.5798001e-05  2.9603641e-03 -1.5013537e-03]
 [-1.0872953e-05  7.4690368e-05 -1.5013537e-03  2.9909934e-03]]

Median adaptive sandwich variance estimate:
[[ 4.3720780e-03 -4.1920641e-03  1.8541534e-05  3.7883554e-05]
 [-4.1920636e-03  1.7515380e-02 -2.3139926e-04  9.7846103e-05]
 [ 1.8541599e-05 -2.3139850e-04  5.2724369e-03 -2.3994250e-03]
 [ 3.7883856e-05  9.7844997e-05 -2.3994255e-03  5.4014167e-03]]

Median classical sandwich variance estimate:
[[ 4.1242684e-03 -4.0818518e-03  9.7337224e-06 -4.8979177e-06]
 [-4.0818518e-03  8.5779727e-03 -7.3272429e-05  5.1330982e-05]
 [ 9.7337215e-06 -7.3272473e-05  2.7250133e-03 -1.3588828e-03]
 [-4.8978904e-06  5.1331008e-05 -1.3588828e-03  2.8069080e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.959

Classical sandwich 95.0% standard normal CI coverage:
0.899

######### n=100
Mean parameter estimate:
[ 1.5772654e-01  1.4231712e-01  1.8248806e-03 -2.5532714e-05]

Empirical variance of parameter estimates:
[[ 2.31353577e-03 -2.39431996e-03 -8.41154635e-05  3.74958543e-05]
 [-2.39431996e-03  6.34529344e-03  2.23641295e-04 -7.00642283e-05]
 [-8.41154635e-05  2.23641295e-04  1.56710340e-03 -8.08591759e-04]
 [ 3.74958543e-05 -7.00642283e-05 -8.08591759e-04  1.58819536e-03]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[1.25273005e-04 3.35228691e-04 1.25273005e-04 1.25273005e-04]
 [3.35228691e-04 3.35228691e-04 3.35228691e-04 3.35228691e-04]
 [1.25273005e-04 3.35228691e-04 8.43876190e-05 8.68081012e-05]
 [1.25273005e-04 3.35228691e-04 8.68081012e-05 8.68081012e-05]]

Mean adaptive sandwich variance estimate:
[[ 2.2292514e-03 -2.2365311e-03  2.9995106e-05 -1.4496494e-05]
 [-2.2365309e-03  1.1082906e-02 -2.0566114e-04 -7.2253701e-05]
 [ 2.9995110e-05 -2.0566124e-04  3.1606110e-03 -1.5776196e-03]
 [-1.4496494e-05 -7.2253752e-05 -1.5776196e-03  2.9988931e-03]]

Mean classical sandwich variance estimate:
[[ 2.1861864e-03 -2.1850308e-03  2.6103784e-05 -1.2784669e-05]
 [-2.1850308e-03  4.4451393e-03 -1.8617871e-05  8.8007901e-06]
 [ 2.6103782e-05 -1.8617868e-05  1.5044885e-03 -7.6875545e-04]
 [-1.2784674e-05  8.8007882e-06 -7.6875550e-04  1.5117069e-03]]

Median adaptive sandwich variance estimate:
[[ 2.1868614e-03 -2.1482389e-03  1.6273403e-05  2.5132883e-06]
 [-2.1482389e-03  7.9046162e-03 -4.6238762e-05  3.4844623e-05]
 [ 1.6273199e-05 -4.6238980e-05  2.3376918e-03 -1.1264591e-03]
 [ 2.5133054e-06  3.4844619e-05 -1.1264591e-03  2.3345710e-03]]

Median classical sandwich variance estimate:
[[ 2.1529617e-03 -2.1510625e-03  1.7831775e-05 -1.6448491e-06]
 [-2.1510627e-03  4.4199321e-03 -9.7010570e-07 -4.4955568e-06]
 [ 1.7831788e-05 -9.7010354e-07  1.4536588e-03 -7.2269497e-04]
 [-1.6448334e-06 -4.4955641e-06 -7.2269491e-04  1.4646517e-03]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.965

Classical sandwich 95.0% standard normal CI coverage:
0.891

######### n=300

Mean parameter estimate:
[1.5645871e-01 1.4506038e-01 1.0835000e-03 4.5880388e-05]

Empirical variance of parameter estimates:
[[ 7.85603987e-04 -7.66018671e-04  2.91856231e-05 -1.35513548e-05]
 [-7.66018671e-04  2.09347310e-03  2.97471125e-05  1.32591792e-06]
 [ 2.91856231e-05  2.97471125e-05  5.31635357e-04 -2.53518865e-04]
 [-1.35513548e-05  1.32591792e-06 -2.53518865e-04  5.15238465e-04]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[4.51495708e-05 1.19123702e-04 4.51495708e-05 4.51495708e-05]
 [1.19123702e-04 1.19123702e-04 1.19123702e-04 1.19123702e-04]
 [4.51495708e-05 1.19123702e-04 2.90756752e-05 2.90756752e-05]
 [4.51495708e-05 1.19123702e-04 2.90756752e-05 2.89598789e-05]]

Mean adaptive sandwich variance estimate:
[[ 7.4275729e-04 -7.4631668e-04 -7.0177151e-07 -1.4183503e-06]
 [-7.4631663e-04  2.8802142e-03  2.1226711e-07 -1.9576205e-06]
 [-7.0177038e-07  2.1227726e-07  7.1577507e-04 -3.5642929e-04]
 [-1.4183487e-06 -1.9576128e-06 -3.5642926e-04  7.0015801e-04]]

Mean classical sandwich variance estimate:
[[ 7.4079516e-04 -7.4092596e-04 -1.0111116e-06 -1.5558542e-06]
 [-7.4092596e-04  1.5063266e-03 -4.7336584e-07  3.6384963e-06]
 [-1.0111108e-06 -4.7336584e-07  5.0972356e-04 -2.5648600e-04]
 [-1.5558546e-06  3.6384972e-06 -2.5648598e-04  5.0558074e-04]]

Median adaptive sandwich variance estimate:
[[ 7.3976920e-04 -7.4215699e-04 -9.4078945e-07 -1.1225745e-06]
 [-7.4215693e-04  2.3949284e-03  4.9114351e-06  7.1622394e-06]
 [-9.4079644e-07  4.9115347e-06  6.5740111e-04 -3.2078239e-04]
 [-1.1225704e-06  7.1622803e-06 -3.2078242e-04  6.5151165e-04]]

Median classical sandwich variance estimate:
[[ 7.3850737e-04 -7.3752017e-04 -1.8209953e-08 -1.8045688e-06]
 [-7.3752023e-04  1.5016802e-03 -1.2468895e-07  7.0191578e-07]
 [-1.8214116e-08 -1.2468793e-07  5.0592312e-04 -2.5268906e-04]
 [-1.8045662e-06  7.0192345e-07 -2.5268906e-04  4.9951620e-04]]

 Adaptive sandwich 95.0% standard normal CI coverage:
0.955955955955956


Classical sandwich 95.0% standard normal CI coverage:
0.9059059059059059

######### n=500

Mean parameter estimate:
[0.1582681  0.14509869 0.00109277 0.00046132]

Empirical variance of parameter estimates:
[[ 4.43998905e-04 -4.37470639e-04  4.84569804e-06 -9.04666293e-06]
 [-4.37470639e-04  1.31968885e-03 -6.16818089e-06  3.56881060e-06]
 [ 4.84569804e-06 -6.16818089e-06  3.06415127e-04 -1.54881664e-04]
 [-9.04666293e-06  3.56881060e-06 -1.54881664e-04  3.02490091e-04]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[2.50868913e-05 7.47703179e-05 2.50868913e-05 2.50868913e-05]
 [7.47703179e-05 7.47703179e-05 7.47703179e-05 7.47703179e-05]
 [2.50868913e-05 7.47703179e-05 1.70793231e-05 1.70793231e-05]
 [2.50868913e-05 7.47703179e-05 1.70793231e-05 1.70392358e-05]]

Mean adaptive sandwich variance estimate:
[[ 4.4493782e-04 -4.4440990e-04 -7.2851060e-07 -6.3747018e-07]
 [-4.4440990e-04  1.5718888e-03  3.7382110e-06  5.0088197e-06]
 [-7.2851043e-07  3.7382113e-06  3.7929296e-04 -1.8965898e-04]
 [-6.3747103e-07  5.0088220e-06 -1.8965897e-04  3.8465706e-04]]

Mean classical sandwich variance estimate:
[[ 4.4444588e-04 -4.4447716e-04 -5.7394203e-07 -5.2572415e-07]
 [-4.4447713e-04  9.0374990e-04  1.8800531e-06  9.1856329e-08]
 [-5.7394243e-07  1.8800531e-06  3.0213202e-04 -1.4992309e-04]
 [-5.2572415e-07  9.1856620e-08 -1.4992309e-04  3.0206857e-04]]

Median adaptive sandwich variance estimate:
[[ 4.4368164e-04 -4.4263309e-04 -7.7131835e-07 -5.3164433e-07]
 [-4.4263326e-04  1.3856665e-03  2.2686086e-06  1.6153865e-06]
 [-7.7132847e-07  2.2686609e-06  3.6089076e-04 -1.7715893e-04]
 [-5.3164365e-07  1.6154190e-06 -1.7715895e-04  3.6458546e-04]]

Median classical sandwich variance estimate:
[[ 4.4318370e-04 -4.4280678e-04 -4.8100981e-07 -2.7298353e-07]
 [-4.4280681e-04  9.0506527e-04  1.5710943e-06  9.2268425e-07]
 [-4.8101299e-07  1.5710932e-06  2.9913164e-04 -1.4740747e-04]
 [-2.7298370e-07  9.2267970e-07 -1.4740747e-04  3.0013645e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9469469469469469

Classical sandwich 95.0% standard normal CI coverage:
0.8998998998998999


######### n=1000





"""

###### results[partial_01envfeatures]: 10/11


"""
######### n=30
Mean parameter estimate:
[0.15704693 0.1331404 ]

Empirical variance of parameter estimates:
[[ 0.00793453 -0.00825922]
 [-0.00825922  0.02014379]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00044916 0.0011478 ]
 [0.0011478  0.0011478 ]]

Mean adaptive sandwich variance estimate:
[[ 0.00769216 -0.00769216]
 [-0.00769216  0.06422266]]

Mean classical sandwich variance estimate:
[[ 0.00769217 -0.00769217]
 [-0.00769217  0.0156229 ]]

Median adaptive sandwich variance estimate:
[[ 0.00725626 -0.00725626]
 [-0.00725626  0.03069103]]

Median classical sandwich variance estimate:
[[ 0.00725627 -0.00725627]
 [-0.00725627  0.01528695]]

 Adaptive sandwich 95.0% standard normal CI coverage:
0.967

Classical sandwich 95.0% standard normal CI coverage:
0.898

######### n=50

Mean parameter estimate:
[0.15708375 0.13718876]

Empirical variance of parameter estimates:
[[ 0.00469875 -0.00477092]
 [-0.00477092  0.01223744]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00025337 0.00069295]
 [0.00069295 0.00069295]]

Mean adaptive sandwich variance estimate:
[[ 0.00474403 -0.00474403]
 [-0.00474403  0.03517093]]

Mean classical sandwich variance estimate:
[[ 0.00474402 -0.00474402]
 [-0.00474402  0.00963685]]

Median adaptive sandwich variance estimate:
[[ 0.00459426 -0.00459426]
 [-0.00459426  0.01947442]]

Median classical sandwich variance estimate:
[[ 0.00459425 -0.00459425]
 [-0.00459425  0.0094664 ]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.975

Classical sandwich 95.0% standard normal CI coverage:
0.907

######### n=100
Mean parameter estimate:
[0.15786481 0.1415722 ]

Empirical variance of parameter estimates:
[[ 0.00249042 -0.00251676]
 [-0.00251676  0.0066971 ]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[0.00014031 0.00035868]
 [0.00035868 0.00035868]]

Mean adaptive sandwich variance estimate:
[[ 0.00240422 -0.00240422]
 [-0.00240422  0.01220649]]

Mean classical sandwich variance estimate:
[[ 0.00240422 -0.00240422]
 [-0.00240422  0.00487279]]

Median adaptive sandwich variance estimate:
[[ 0.00235969 -0.00235969]
 [-0.00235969  0.00861138]]

Median classical sandwich variance estimate:
[[ 0.00235969 -0.00235969]
 [-0.00235969  0.0048518 ]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.968

Classical sandwich 95.0% standard normal CI coverage:
0.901

######### n=300

Mean parameter estimate:
[0.15627219 0.14581767]

Empirical variance of parameter estimates:
[[ 0.0008585  -0.00086364]
 [-0.00086364  0.00229677]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[4.72137123e-05 1.26695412e-04]
 [1.26695412e-04 1.26695412e-04]]

Mean adaptive sandwich variance estimate:
[[ 0.00081594 -0.00081594]
 [-0.00081594  0.00310408]]

Mean classical sandwich variance estimate:
[[ 0.00081594 -0.00081594]
 [-0.00081594  0.0016562 ]]

Median adaptive sandwich variance estimate:
[[ 0.00081077 -0.00081077]
 [-0.00081077  0.00261903]]

Median classical sandwich variance estimate:
[[ 0.00081077 -0.00081077]
 [-0.00081077  0.00165354]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.9579579579579579

Classical sandwich 95.0% standard normal CI coverage:
0.908908908908909

######### n=500

Mean parameter estimate:
[0.1581754  0.14493303]

Empirical variance of parameter estimates:
[[ 0.00048568 -0.00046796]
 [-0.00046796  0.0013951 ]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[2.74986149e-05 7.97500811e-05]
 [7.97500811e-05 7.97500811e-05]]

Mean adaptive sandwich variance estimate:
[[ 0.00048935 -0.00048935]
 [-0.00048935  0.0016698 ]]

Mean classical sandwich variance estimate:
[[ 0.00048934 -0.00048934]
 [-0.00048934  0.00099326]]

Median adaptive sandwich variance estimate:
[[ 0.00048662 -0.00048662]
 [-0.00048662  0.00147158]]

Median classical sandwich variance estimate:
[[ 0.00048661 -0.00048661]
 [-0.00048661  0.00099426]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.958958958958959

Classical sandwich 95.0% standard normal CI coverage:
0.913913913913914


######### n=1000

Mean parameter estimate:
[0.15734106 0.14653736 0.0100045  0.10032322]

Empirical variance of parameter estimates:
[[ 2.19872597e-04 -2.16141243e-04  3.80806131e-06 -5.91923851e-08]
 [-2.16141243e-04  6.62755788e-04 -5.04453060e-06 -1.23234254e-05]
 [ 3.80806131e-06 -5.04453060e-06  1.52435578e-04 -7.72040285e-05]
 [-5.91923851e-08 -1.23234254e-05 -7.72040285e-05  1.56096710e-04]]

Empirical variance standard errors (off-diagonals approximated by taking max of corresponding two diagonal terms):
[[1.19093526e-05 3.55759733e-05 1.19093526e-05 1.19093526e-05]
 [3.55759733e-05 3.55759733e-05 3.55759733e-05 3.55759733e-05]
 [1.19093526e-05 3.55759733e-05 8.41725399e-06 8.75994547e-06]
 [1.19093526e-05 3.55759733e-05 8.75994547e-06 8.75994547e-06]]

Mean adaptive sandwich variance estimate:
[[ 2.2412825e-04 -2.2410696e-04 -7.7286913e-08  2.2919798e-07]
 [-2.2410693e-04  7.4028329e-04  1.4198474e-06  1.4361902e-06]
 [-7.7286913e-08  1.4198506e-06  1.7144799e-04 -8.6144209e-05]
 [ 2.2919836e-07  1.4361895e-06 -8.6144217e-05  1.7357070e-04]]

Mean classical sandwich variance estimate:
[[ 2.2406006e-04 -2.2404663e-04 -2.1844965e-08  1.9325417e-07]
 [-2.2404664e-04  4.5374327e-04  3.5099885e-07 -8.0200113e-07]
 [-2.1845063e-08  3.5099876e-07  1.5074950e-04 -7.5460266e-05]
 [ 1.9325410e-07 -8.0200084e-07 -7.5460266e-05  1.5157953e-04]]

Median adaptive sandwich variance estimate:
[[ 2.2382756e-04 -2.2373101e-04 -3.3276766e-07 -2.0313229e-07]
 [-2.2373107e-04  6.8957836e-04 -2.0803692e-07  1.1737366e-06]
 [-3.3276916e-07 -2.0802162e-07  1.6706096e-04 -8.3571278e-05]
 [-2.0313189e-07  1.1737145e-06 -8.3571285e-05  1.6961309e-04]]

Median classical sandwich variance estimate:
[[ 2.2371669e-04 -2.2365397e-04 -3.0833874e-07 -2.3437163e-07]
 [-2.2365397e-04  4.5405561e-04  1.7398997e-07 -6.4651624e-07]
 [-3.0833834e-07  1.7398921e-07  1.5004167e-04 -7.5186428e-05]
 [-2.3437151e-07 -6.4651670e-07 -7.5186443e-05  1.5093249e-04]]

Adaptive sandwich 95.0% standard normal CI coverage:
0.952


Classical sandwich 95.0% standard normal CI coverage:
0.897


"""


###### results[partial_01envfeatures_inference]: 10/15, add C in both environments and inference

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

