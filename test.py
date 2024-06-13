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


a = torch.tensor(np.array([[2.0, 1.0, 0.1], [0.5, 2.0, 0.3]]), requires_grad=True)
b = torch.tensor(np.array([[1., 0., 0.], [0., 1., 0.]]), requires_grad=True)

exps = np.exp(a - (a.max(axis=-1, keepdims=True)))
softmax = exps / np.sum(exps, axis=-1, keepdims=True)
log_softmax = np.log(softmax + 1e-12)
loss = -np.sum(b.data * log_softmax, axis=-1)
# loss.backward()
