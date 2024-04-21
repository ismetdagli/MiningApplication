

import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor


# REMOVE THE COMMENTS FOR CPU EXECUTION
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
torch.backends.cudnn.enabled = False
torch.cuda.is_available = lambda : False
torch.device('cpu')

use_cuda = False #torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor       

# N, D = 10000 if use_cuda else 1000, 2  # Number of samples, dimension
# M = 1000 if use_cuda else 100
N=1000000
D=2
dtype=torch.FloatTensor
M=1000
print(N,D)
x = torch.rand(N, D).type(dtype)  # Random samples on the unit square
x = x.to('cpu')


# Random-ish class labels:
def fth(x):
    return 3 * x * (x - 0.5) * (x - 1) + x


cl = x[:, 1] + 0.1 * torch.randn(N).type(dtype) < fth(x[:, 0])


tmp = torch.linspace(0, 1, M).type(dtype)
g2, g1 = torch.meshgrid(tmp, tmp)
g = torch.cat((g1.contiguous().view(-1, 1), g2.contiguous().view(-1, 1)), dim=1)

for i, K in enumerate((3,10)):   # for i, K in enumerate((1, 3, 10, 20, 50)):
    start = time.time()  # Benchmark:

    G_i = LazyTensor(g[:, None, :])  # (M**2, 1, 2)
    X_j = LazyTensor(x[None, :, :])  # (1, N, 2)
    D_ij = ((G_i - X_j) ** 2).sum(-1)  # (M**2, N) symbolic matrix of squared distances
    indKNN = D_ij.argKmin(K, dim=1)  # Grid <-> Samples, (M**2, K) integer tensor

    clg = cl[indKNN].float().mean(1) > 0.5  # Classify the Grid points
    end = time.time()
    print("time: ",end-start)