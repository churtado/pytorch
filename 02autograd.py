from __future__ import print_function
import torch

"""
autograd provides automatic differentiation for operations on tensors
"""

# a tensor can have all operations on it tracked:
x = torch.ones(2,2, requires_grad=True)
print("A tensor with tracking enabled")
print(x)

# let's do an operation on a tensor: y is the result of an operation
y = x + 2
print("")
print("An operation on a tensor:")
print(y)
print("Y's function:")
print(y.grad_fn) # if created by a user, it has no function

z = y * y * 3
out = z.mean()
print("")
print("A couple of tensors:")
print(z, out)

# backprop
out.backward()
print(x.grad)