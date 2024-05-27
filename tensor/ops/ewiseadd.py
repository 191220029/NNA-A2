from op import Op
from numpy import ndarray
from tensor.tensor import Tensor

# 对应元素相加 
class EWiseAdd(Op):
    def compute(self, a: ndarray, b: ndarray):
        return a + b
    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad