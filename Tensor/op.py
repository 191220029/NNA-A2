from abc import abstractmethod
from typing import Tuple
from numpy import ndarray

class Op():
    @abstractmethod
    def compute(self,
    *args: Tuple["ndarray"]) -> ndarray:
    # 前向计算. 参数args是由NDArray组成的序列Tuple，输出计算的结果NDArray
        assert(False)
    @abstractmethod
    def gradient(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
    # 后向求导. 计算每个输入变量对应的局部伴随值(partial adjoint)
    # 参数out_grad是输出变量对应的伴随值，node是计算操作所在的计算图节点
    # 为方便编程，输出总是一个序列Tuple
        assert(False)