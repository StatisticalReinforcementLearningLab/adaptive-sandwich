import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

variances_methods = {}
# theta_estimate, true variance, median of standard variance estimate, median adaptive variance estimate

################# synthetic env + mean reward + Thompson sampling
# variances_methods['30'] = [0.296367, 0.005726, 0.003655, 0.010398]
# variances_methods['50'] = [0.301267, 0.003659, 0.002225, 0.005826]
# variances_methods['100'] = [0.301623, 0.002060,0.001128, 0.002513]
# variances_methods['300'] = [0.302433, 0.000792, 0.000383, 0.000829]
# variances_methods['500'] = [0.304311, 0.000448, 0.000230, 0.000473]
# variances_methods['1000'] = [0.304409, 0.000224, 0.000115, 0.000236]

################# synthetic env + partially linear regression + Thompson sampling
### 0
# variances_methods['30'] = [0.22249842, 0.00454615, 0.00377557, 0.0080901]
# variances_methods['50'] = [0.2247158, 0.00288899, 0.00228628, 0.0048192]
# variances_methods['100']= [0.22879328, 0.00148607, 0.00117365, 0.0021057]
# variances_methods['300'] = [0.22901689, 0.00054021, 0.0003948, 0.00062922]
# variances_methods['500'] = [0.2308117, 0.00033679, 0.00023693, 0.00036184]

### 01
# variances_methods['30'] = [0.13254973, 0.01810563, 0.01400174, 0.02995899]
# variances_methods['50'] = [0.13610247, 0.01168642, 0.00863468, 0.0173477]
# variances_methods['100']= [0.14265543, 0.00621132, 0.00442925, 0.0079287]
# variances_methods['300'] = [0.14521033, 0.00208446, 0.00150396, 0.00239358]
# variances_methods['500'] = [0.14514528, 0.00131715, 0.000904, 0.00138789]

### 01feature
# variances_methods['30'] = [0.13298166, 0.01907834, 1.3761424e-02, 2.96461470e-02]
# variances_methods['50'] = [0.135732, 1.22122003e-02, 8.5779727e-03, 1.7515380e-02]
# variances_methods['100']= [1.4231712e-01, 6.34529344e-03, 4.4199321e-03, 7.9046162e-03]
# variances_methods['300'] = [1.4506038e-01, 2.09347310e-03, 1.5016802e-03, 2.3949284e-03]
# variances_methods['500'] = [0.14509869, 1.31968885e-03, 9.0506527e-04, 1.3856665e-03]

### 01env
# variances_methods['30'] = [0.1331404, 0.02014379, 0.0156229, 0.03069103]
# variances_methods['50'] = [ 0.13718876  ,0.01223744,0.0094664,0.01947442]
# variances_methods['100']= [0.1415722, 0.0066971, 0.0048518 ,0.00861138]
# variances_methods['300'] = [0.14581767, 0.00229677, 0.00165354, 0.00261903]
# variances_methods['500'] = [0.14493303, 0.0013951, 0.00099426, 0.00147158]

### 01env+inference
# variances_methods['30'] = [0.13003476, 1.89196844e-02, 1.3528196e-02, 2.7703129e-02]
# variances_methods['50'] = [0.13621828, 1.19798024e-02, 8.5263029e-03, 1.7226841e-02]
# variances_methods['100']= [0.1422507, 6.34026079e-03, 4.3996195e-03, 7.8176325e-03 ]
# variances_methods['300'] = [0.145417, 2.11213185e-03, 1.5008608e-03, 2.4244864e-03]
# variances_methods['500'] = [0.1453746, 1.33133568e-03, 9.0210827e-04, 1.3764334e-03]
# variances_methods['1000'] = [0.14653736, 6.62755788e-04, 4.5405561e-04, 6.8957836e-04]

### SAC: average rewards
# median
# variances_methods['30'] = [0.24456318, 0.006194,0.00392348, 0.00407279]
# variances_methods['50'] = [0.23989122, 0.00299804, 0.00232505, 0.00237417]
# variances_methods['100']= [0.24534565, 0.00131415, 0.00117638, 0.00118786]
# variances_methods['300'] = [0.25961846, 0.00046748, 0.00040852, 0.00041054]
# variances_methods['500'] = [0.26483545, 0.00026997, 0.00024645, 0.00024706]
# mean
# variances_methods['30'] = [0.24456318, 0.006194,0.00399096,0.8197159]
# variances_methods['50'] = [0.23989122, 0.00299804,0.00235252, 0.0061214 ]
# variances_methods['100']= [0.24534565, 0.00131415, 0.00118791, 0.00198201]
# variances_methods['300'] = [ 0.25961846, 0.00046748, 0.00040899, 0.00047262]
# variances_methods['500'] = [0.26483545, 0.00026997, 0.00024676, 0.00025795]

### SAC: treatment effect
# median
# variances_methods['30'] = [0.17996839, 2.54966810e-2, 1.3580132e-2, 1.4134381e-2]
# variances_methods['50'] = [0.16762662, 1.25121634e-2, 8.4480355e-3, 8.6873807e-3]
# variances_methods['100']= [ 0.1762619, 5.41489872e-3, 4.35435958e-3, 4.3835528e-3]
# variances_methods['300'] = [0.20686363, 1.71520708e-03, 1.4831771e-03, 1.4899302e-03]
# variances_methods['500'] = [ 0.21479316, 1.08079133e-03, 8.9439156e-04, 8.9574198e-04]
# mean
variances_methods['30'] = [0.17996839, 2.54966810e-2, 1.4023899e-2, 1.2976654]
variances_methods['50'] = [0.16762662, 1.25121634e-2, 8.62125214e-3, 107.55174]
variances_methods['100']= [ 0.1762619, 5.41489872e-3, 4.3909219e-3, 6.5868977e-3]
variances_methods['300'] = [0.20686363, 1.71520708e-03, 1.4874872e-03, 1.5772188e-03]
variances_methods['500'] = [ 0.21479316, 1.08079133e-03, 8.9394127e-04, 9.0586691e-04]




COLORS = ['black', 'blue', 'red', 'green', 'orange']
METHODS = [r'Empirical Variance of $\hat{\theta}_1^{(n)}$ (Oracle)', r'Standard Variance Estimate of Var($\hat{\theta}_1^{(n)}$)', r'Pooling RL Adjusted Estimate of Var($\hat{\theta}_1^{(n)}$)']



plt.figure(figsize=(12,8))
ax = plt.gca()
n_list = [int(n) for n in variances_methods.keys()]
theta_hat_mean = [variances_methods[n][0] for n in variances_methods.keys()]
theta_hat_var_true = [variances_methods[n][1] for n in variances_methods.keys()]
theta_hat_var_std = [variances_methods[n][2] for n in variances_methods.keys()]
theta_hat_var_adaptive = [variances_methods[n][3] for n in variances_methods.keys()]
theta_var = [theta_hat_var_true, theta_hat_var_std, theta_hat_var_adaptive]

plt.plot(n_list, theta_hat_mean, color='black', label=r"Estimated $\hat{\theta}_1^{(n)}$", linewidth=5)
lines = []
for i, method_name in enumerate(METHODS):
    variances = theta_var[i]
    stds = np.sqrt(variances)
    # lower = theta_hat_mean - 1.96 * stds / np.sqrt(n_list[i])
    lower = theta_hat_mean - 1.96 * stds
    # upper = theta_hat_mean + 1.96 * stds / np.sqrt(n_list[i])
    upper = theta_hat_mean + 1.96 * stds 
    print(f'Width of the CI for method {method_name}, n={n_list}:', np.round(upper - lower, 4))
    plt.plot(n_list, lower, color=COLORS[i], label=METHODS[i], linestyle='--', linewidth=5)
    plt.plot(n_list, upper, color=COLORS[i], linestyle='--', linewidth=5)
    plt.fill_between(n_list, lower, upper, color=COLORS[i], alpha=0.2)
    
plt.xscale("log")
plt.xticks(n_list, [str(n) for n in n_list], fontsize=40)
plt.yticks(fontsize=40)
plt.xlabel("Sample size n (log-scale)", fontsize=40)
# plt.title(r"$\hat{\theta}$ with 95% Confidence Interval (CI)", fontsize=20)
# plt.title("95% Confidence Intervals for Mean Rewards after Using Pooling RL", fontsize=30)
plt.title("95% Confidence Intervals for Treatment Effect in Two-group Trials (TS)", fontsize=35)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=26)
# plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)



plt.tight_layout()
plt.show()