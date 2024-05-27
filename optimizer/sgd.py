from optimizer.optimizer import Optimizer
from tensor.tensor import Tensor

class SGD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params)
        self.lr = lr
    def step(self):
        for _, param in enumerate(self.params):
            grad = Tensor(param.grad, dtype='float32').data
            param.data= param.data - grad * self.lr
