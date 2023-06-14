import numpy as np
import torch
import scipy 
import scipy.linalg
import findiff
from findiff import FinDiff, coefficients, Coefficient

import ipdb; ipdb.set_trace()

x = np.linspace(0, 10, 100)
dx = x[1] - x[0]
f = np.sin(x)
g = np.cos(x)
d2_dx2 = FinDiff(0, dx, 2)


"""
Matrix Square root
"""
# https://math.stackexchange.com/questions/540361/derivative-or-differential-of-symmetric-square-root-of-a-matrix?noredirect=1&lq=1

# Getting square root of matrix gradient ###########################

matrix = np.array([[3,4.0],[4,7]])
#matrix = np.array([[1.0,0],[0,1.0]])
matrix_torch = torch.from_numpy( matrix )
matrix_torch.requires_grad = True

#L, Q = torch.linalg.eigh(matrix_torch)
#matrix_sqrt = torch.matmul( torch.matmul(Q, torch.diag(torch.sqrt(L))), torch.inverse(Q) )

A, B, C = torch.linalg.svd(matrix_torch)
#print( torch.matmul( torch.matmul(A, torch.diag(B)), C.T ) )
#matrix_sqrt = torch.matmul( torch.matmul(A, torch.diag(B)), C.T )
matrix_sqrt = torch.matmul( torch.matmul(A, torch.diag(torch.sqrt(B))), C.T )
matrix_sqrt.sum().backward()

print( matrix_torch.grad.numpy() )

# Getting oracle square root of matrix gradient ###########################

matrix_sqrt_np = scipy.linalg.sqrtm(matrix)

w, x, y, z = [np.linspace(0, 10, 100)]*4
dw, dx, dy, dz = w[1] - w[0], x[1] - x[0], y[1] - y[0], z[1] - z[0]
W, X, Y, Z = np.meshgrid(w, x, y, z, indexing='ij')



import ipdb; ipdb.set_trace()

grad = Gradient(h=[dx, dy, dz])
grad_f = grad(f)

import ipdb; ipdb.set_trace()

"""
Matrix Inverse
"""
# https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf (page 9)

# Getting matrix inverse gradient ##########################

#matrix = np.array([[3,4,5.0],[4,7,2],[5,2,3]])
matrix = np.array([[3,4.0],[4,7]])
#matrix = np.array([[1.0,0],[0,1.0]])
matrix_torch = torch.from_numpy( matrix )
matrix_torch.requires_grad = True

matrix_inv = torch.linalg.inv( matrix_torch )
matrix_inv.sum().backward()
grad = matrix_torch.grad.numpy()
print(grad)

"""
array([[-0.36,  0.12],
       [ 0.12, -0.04]])
"""

# Getting oracle matrix inverse gradient ##########################

np_inv = np.linalg.inv(matrix)
tmp = np.array([[1,1],[1,1]])
print( - np.matmul( np_inv, np.matmul( tmp, np_inv ) ) )

