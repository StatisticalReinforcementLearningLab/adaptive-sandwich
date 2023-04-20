import numpy as np
import scipy
from scipy import optimize
import scipy.special
import time

#import torch

def func(x, c0, c1):
    "Coordinate vector `x` should be an array of size two."
    return c0 * x[0]**2 + c1*x[1]**2

x = np.ones(2)
c0, c1 = (1, 200)
eps = np.sqrt(np.finfo(float).eps)

optimize.approx_fprime(x, func, [eps, np.sqrt(200) * eps], c0, c1)

# Finite differences GRAD ###############################
est_params = np.array([0.0,1,2,3,4,5,6,7,7,8,0.1,-0.3,1,2,3,4,5,6,7,7,8,0.1,-0.3])

def form_pis(est_params, hi):
    pis = scipy.special.expit(est_params)
    print('hi', hi)
    return pis #np.sum(pis)

tic = time.perf_counter()
fd_grads = optimize.approx_fprime(est_params, form_pis, eps, 1)
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
print(fd_grads)

# TORCH AUTOGRAD ###############################
import torch

tic = time.perf_counter()
est_params_torch = torch.from_numpy( est_params )
est_params_torch.requires_grad = True
pis_torch = torch.sigmoid( est_params_torch )
pis_torch.sum().backward()
torch_grad= est_params_torch.grad.numpy()
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
print("torch_grad", torch_grad)

import ipdb; ipdb.set_trace()
