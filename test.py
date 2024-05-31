import torch
import numpy as np

a = torch.tensor(np.zeros((2,1)), requires_grad=True)
b = torch.tensor(np.ones((1,2)), requires_grad=True)
c = a + b
c = c.sum()
c.backward()
print(a.grad)
print(b.grad)