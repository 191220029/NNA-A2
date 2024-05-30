import sys
sys.path.append("..")
sys.path.append(".")

from tensor import Tensor, MatMul
from module import Module
import numpy

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = init_He(in_features, out_features, dtype) #请自行实现初始化算法
        if bias:
            self.bias = Tensor(numpy.zeros(self.out_features))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        X_out = X @ self.weight
        if self.bias:
            t = self.bias.broadcast_to(X_out.shape)
            return X_out + t
        return X_out
    
def init_He(in_features, out_features, dtype):
    stddev = numpy.sqrt(2 / in_features)
    return Tensor(numpy.random.randn(in_features, out_features).astype(dtype) * stddev, requires_grad=True)
