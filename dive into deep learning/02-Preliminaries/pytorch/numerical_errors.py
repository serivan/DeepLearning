import numpy as np
import torch

# Rounding and truncation errors
a = np.array([0., 1e-8]).astype('float32')
a

# Convert to a PyTorch tensor
a_torch = torch.tensor(a)
a_torch + 1

# exp(x) overflows for large x
x = 88
torch.exp(torch.tensor(x, dtype=torch.float32))

x = -100  # Check what happens at different values of x
torch.exp(torch.tensor(x, dtype=torch.float32))

# log(0) = -inf
torch.log(torch.tensor(0.))

# log(sum(exp))
# Naive implementation
a = np.array([0, 88]).astype('float32')
a_torch = torch.tensor(a)
torch.log(torch.sum(torch.exp(a_torch)))

a = np.array([-100]).astype('float32')
a_torch = torch.tensor(a)
torch.exp(a_torch)

torch.log(torch.sum(torch.exp(a_torch)))

# Stable version
a = np.array([0, 100]).astype('float32')
a_torch = torch.tensor(a)
mx = torch.max(a_torch)
safe_array = a_torch - mx
log_sum_exp = mx + torch.log(torch.sum(torch.exp(safe_array)))
log_sum_exp

# Built-in version in PyTorch (logsumexp)
torch.logsumexp(a_torch, dim=0)

