from optimizer import Optimizer
from tensor import Tensor

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr
    def step(self):
        for _, param in enumerate(self.params):
            grad = Tensor.from_numpy(param.grad.data, dtype=param.grad.dtype)
            param.data = param.data - grad * self.lr
    def zero_grad(self):
        for param in self.params:
            param.grad = None
