import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams['font.family'] = 'sans-serif'
# matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.use('TkAgg')
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']

# use Times for mathtext too
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'
matplotlib.rcParams['mathtext.bf'] = 'Times New Roman:bold'

variances_methods = {}
# theta_estimate, empirical variance, median of standard variance estimate, median adaptive variance estimate

################################################## synthetic env ##################################################
################# TS: average rewards [introduction]
# variances_methods['30'] = [0.296057, 0.005611, 0.003661, 0.010364]
# variances_methods['50'] = [0.300352, 0.003759, 0.002228, 0.005753]
# variances_methods['100'] = [0.301689, 0.002040, 0.001132, 0.002657]
# variances_methods['300'] = [0.303317, 0.000750, 0.000383, 0.000810]
# variances_methods['500'] = [0.303996, 0.000453, 0.000230, 0.000468]
# variances_methods['1000'] = [0.304492, 0.000225, 0.000115, 0.000239]

################# TS: treatment effect
variances_methods['30'] = [0.133062, 0.019427,  0.013642, 0.029325]
variances_methods['50'] = [0.135509, 0.012279, 0.008561, 0.017905]
variances_methods['100']= [0.142783, 0.006265, 0.004431, 0.007721]
variances_methods['300'] = [0.145818, 0.002136, 0.001499, 0.002429]
variances_methods['500'] = [0.144502, 0.001340, 0.000902, 0.001374]
variances_methods['1000'] = [0.146655, 0.000673, 0.000455, 0.000700]

################# SAC: average rewards
# variances_methods['30'] = []
# variances_methods['50'] = []
# variances_methods['100']= []
# variances_methods['300'] = []
# variances_methods['500'] = []

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
METHODS = [r'Empirical Variance of $\hat{\theta}_{\text{multitask-RL}}^{(n)}$ (Oracle)', r'Standard Variance Estimate of Var($\hat{\theta}_{\text{multitask-RL}}^{(n)}$)', r'Multitask RL Adjusted Estimate of Var($\hat{\theta}_{\text{multitask-RL}}^{(n)}$)']



plt.figure(figsize=(12,8))
ax = plt.gca()
n_list = [int(n) for n in variances_methods.keys()]
theta_hat_mean = [variances_methods[n][0] for n in variances_methods.keys()]
theta_hat_var_true = [variances_methods[n][1] for n in variances_methods.keys()]
theta_hat_var_std = [variances_methods[n][2] for n in variances_methods.keys()]
theta_hat_var_adaptive = [variances_methods[n][3] for n in variances_methods.keys()]
theta_var = [theta_hat_var_true, theta_hat_var_std, theta_hat_var_adaptive]

plt.plot(n_list, theta_hat_mean, color='black', label=r"Estimated $\hat{\theta}_{\text{multitask-RL}}^{(n)}$", linewidth=5)
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
# plt.title("95% Confidence Intervals for Mean Rewards after Using Multitask RL", fontsize=40)
plt.title("95% Confidence Intervals for Treatment Effect in Two-group Trials (TS)", fontsize=35)
# plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), ncol=2, fontsize=26)
# plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=30)



plt.tight_layout()
plt.show()