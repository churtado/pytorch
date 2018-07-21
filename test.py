from __future__ import print_function
import torch

# these are tensors, they are matrices that use gpu

# construct a matrix
x = torch.empty(5,3)

# randomly initialized matrix
x = torch.rand(5,3)

# zeros
x = torch.zeros(5,3, dtype=torch.long)

# from data
x = torch.tensor([5.5, 3])

# create tensor from existing
x = x.new_ones(5, 3, dtype=torch.double)

x = torch.randn_like(x, dtype=torch.float)

print(x)