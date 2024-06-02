import torch
import numpy as np

# a = torch.tensor(np.zeros((2,1)), requires_grad=True)
# b = torch.tensor(np.ones((1,2)), requires_grad=True)
# c = a + b
# c.retain_grad()
# d = c.sum()
# d.retain_grad()
# d.backward(retain_graph=True)
# print(a.grad)
# print(b.grad)
# print(c.grad)
# print(d.grad)

a = torch.tensor(np.ones((5,1)), requires_grad=True)
b = torch.tensor(np.ones((1,5)), requires_grad=True)
c = a * b
c.retain_grad()
d = c.broadcast_to((5,5,5,5,5))
d.retain_grad()
e = d.sum()
e.retain_grad()
e.backward()
print(e)
print(a.grad)
print(b.grad)
print(c.grad)
print(d.grad)
print(e.grad)