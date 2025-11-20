import numpy as np
import pandas as pd




N_seed = 100
p = 5
n_list = [25, 50, 100, 200, 400, 800, 1600]
for n in n_list:
    print(f'######### n={n}')
    beta_list = []
    for i in range(N_seed):
        np.random.seed(i)
        X = np.random.randn(n, p)
        Y = np.random.randn(n, 1)
        beta = np.linalg.pinv(X) @ Y
        beta_list.append(np.array(beta.flatten()).mean(axis=0))
    beta_list = np.array(beta_list)
    print(f'n: {n}, Var(\hat(theta)): {np.var(beta_list)}' )

