import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

# use Times for mathtext too
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

variances_methods = {}
# theta_estimate, true variance, median of standard variance estimate, median adaptive variance estimate

################################################## synthetic env ##################################################
################# TS: average rewards [introduction]
# variances_methods['30'] = [0.296367, 0.005726, 0.003655, 0.010398]
# variances_methods['50'] = [0.301267, 0.003659, 0.002225, 0.005826]
# variances_methods['100'] = [0.301623, 0.002060,0.001128, 0.002513]
# variances_methods['300'] = [0.302433, 0.000792, 0.000383, 0.000829]
# variances_methods['500'] = [0.304311, 0.000448, 0.000230, 0.000473]
# variances_methods['1000'] = [0.304409, 0.000224, 0.000115, 0.000236]

################# TS: treatment effect
# variances_methods['30'] = [0.13003476, 1.89196844e-02, 1.3528196e-02, 2.7703129e-02]
# variances_methods['50'] = [0.13621828, 1.19798024e-02, 8.5263029e-03, 1.7226841e-02]
# variances_methods['100']= [0.1422507, 6.34026079e-03, 4.3996195e-03, 7.8176325e-03 ]
# variances_methods['300'] = [0.145417, 2.11213185e-03, 1.5008608e-03, 2.4244864e-03]
# variances_methods['500'] = [0.1453746, 1.33133568e-03, 9.0210827e-04, 1.3764334e-03]
# variances_methods['1000'] = [0.14653736, 6.62755788e-04, 4.5405561e-04, 6.8957836e-04]

################# SAC: average rewards
variances_methods['30'] = []
variances_methods['50'] = []
variances_methods['100']= []
variances_methods['300'] = []
variances_methods['500'] = []

################# SAC: treatment effect
# variances_methods['30'] = []
# variances_methods['50'] = []
# variances_methods['100']= []
# variances_methods['300'] = []
# variances_methods['500'] = []


################################################## Miwaves env ##################################################
################# TS: average rewards

################# TS: treatment effect

################# SAC: average rewards

################# SAC: treatment effect


COLORS = ['black', 'blue', 'red', 'green', 'orange']
METHODS = [r'Empirical Variance of $\hat{\theta}_{\text{pooling-RL}}^{(n)}$ (Oracle)', r'Standard Variance Estimate of Var($\hat{\theta}_{\text{pooling-RL}}^{(n)}$)', r'Pooling RL Adjusted Estimate of Var($\hat{\theta}_{\text{pooling-RL}}^{(n)}$)']



plt.figure(figsize=(12,8))
ax = plt.gca()
n_list = [int(n) for n in variances_methods.keys()]
theta_hat_mean = [variances_methods[n][0] for n in variances_methods.keys()]
theta_hat_var_true = [variances_methods[n][1] for n in variances_methods.keys()]
theta_hat_var_std = [variances_methods[n][2] for n in variances_methods.keys()]
theta_hat_var_adaptive = [variances_methods[n][3] for n in variances_methods.keys()]
theta_var = [theta_hat_var_true, theta_hat_var_std, theta_hat_var_adaptive]

plt.plot(n_list, theta_hat_mean, color='black', label=r"Estimated $\hat{\theta}_{\text{pooling-RL}}^{(n)}$", linewidth=5)
lines = []
for i, method_name in enumerate(METHODS):
    variances = theta_var[i]
    stds = np.sqrt(variances)
    lower = theta_hat_mean - 1.96 * stds
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
plt.title("95% Confidence Intervals for Mean Rewards after Using Pooling RL", fontsize=40)
# plt.title("95% Confidence Intervals for Treatment Effect in Two-group Trials (TS)", fontsize=35)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=26)
# plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)



plt.tight_layout()
plt.show()