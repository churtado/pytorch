#https://medium.com/datathings/neural-networks-and-backpropagation-explained-in-a-simple-way-f540a3611f5e
from __future__ import print_function
import torch
import numpy as np

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
#print(x)

x = torch.randn_like(x, dtype=torch.float)
#print(x)

# size is a tuple
#print(x.size())

y = torch.rand(5,3)
#print(x+y)

# addition in place, in-place mutators use underscore
y.add_(x)
#print(y)

# resizing:
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8) # the size -1 is inferred from other dimensions

#print(x.size(), y.size(), z.size())
#print(x)
#print(z)

x = torch.randn(1)
#print(x)
#print(x.item())

# from tensor to numpy
a = torch.ones(5)
#print(a)

b = a.numpy()
#print(b)

a.add_(1)
#print(a)
#print(b) # see how the derived numpy array changes value too!!

#numpy to torch tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)