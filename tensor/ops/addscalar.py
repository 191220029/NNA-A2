from op import Op
from numpy import ndarray
from tensor.tensor import Tensor

# 加常数
class AddScalar(Op):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: ndarray):
        return a + self.scalar
    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,) #重载gradient函数的输出必须是Tuple