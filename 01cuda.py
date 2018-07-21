from __future__ import print_function
import torch

# cuda tensors: tensors can be moved to a device with the .to method

x = torch.randn(1)

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to(device)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))


