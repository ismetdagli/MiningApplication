
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from pykeops.torch import LazyTensor

#CPU
torch.device('cpu')
data = torch.randn(100000, 1000).to('cpu')
test = torch.randn(1, 1000).to('cpu')

#GPU
# torch.device('cuda')
# data = torch.randn(100000, 1000).to('cuda')
# test = torch.randn(1, 1000).to('cuda')

print("started")
start = time.time()  # Benchmark:
for i in range(1000):
    dist = torch.norm(data - test, dim=1, p=None)
    knn = dist.topk(10, largest=False) # make it 3,10,20 etc.
end = time.time()
print("time: ",end-start)

print('kNN dist: {}, index: {}'.format(knn.values, knn.indices))