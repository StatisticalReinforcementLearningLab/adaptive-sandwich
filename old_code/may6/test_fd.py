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

"""
import numdifftools as nd
Hfunc = nd.Hessian(lambda x: func(x, c0, c1))
print( Hfunc(x) )
"""


# Finite differences GRAD ###############################
est_params = np.array([0.0,1,2,3,4,5,6,7,7,8,0.1,-0.3,1,2,3,4,5,6,7,7,8,0.1,-0.3])

def form_pis(est_params, hi):
    pis = scipy.special.expit(est_params)
    #print('hi', hi)
    return pis #np.sum(pis)

tic = time.perf_counter()
fd_grads = optimize.approx_fprime(est_params, form_pis, eps, 1)
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
#print(fd_grads)


"""
# AUTOGRAD PACKAGE ###############################
import autograd.numpy as np2   # Thinly-wrapped version of Numpy
from autograd import grad

est_params2 = np2.array([0.0,1,2,3,4,5,6,7,7,8,0.1,-0.3,1,2,3,4,5,6,7,7,8,0.1,-0.3])
#pis2 = form_pis(est_params2, 2)
#pisum = np.sum(pis2)
gradfunc = grad( lambda x: form_pis(x, 2) )
print( gradfunc( est_params2 ) )

import ipdb; ipdb.set_trace()


tic = time.perf_counter()
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
"""


# NUMDIFF TOOLS ###############################
import numdifftools as nd
from numdifftools import nd_statsmodels

#Hfunc = nd.Hessian(lambda x: form_pis(x, 2))
#Hfunc(est_params)

tic = time.perf_counter()
fd_grads_nd = nd_statsmodels.approx_fprime( est_params, lambda x: form_pis(x, 2) )
toc = time.perf_counter()
print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")


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
