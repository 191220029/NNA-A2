from typing import List
from typing import Dict
from value import Value
from ops.op import Op
from ops.addscalar import AddScalar
from ops.ewiseadd import EWiseAdd
import numpy

class Tensor (Value):
    grad: "Tensor" 
    def __init__(self, array, *, dtype=None, requires_grad=True, **kwargs):
        self.cached_data = numpy.array(array, dtype=dtype, **kwargs)
        self.requires_grad = requires_grad
        self.grad = None
        self.op = None
        self.inputs = []

    @staticmethod
    def from_numpy(numpy_array, dtype):
        return Tensor(numpy_array, dtype=dtype)
        
    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        # if not LAZY_MODE: #LAZY_MODE用于实现先建图后计算 （静态计算图，作业中无需实现）
        #   tensor.realize_cached_data()
        return tensor

    @ property
    def data (self): #对cached_data进行封装
        assert(False)

    @ data.setter
    def data (self, value):
        assert(False)

    @ property
    def shape (self):
        return self.cached_data.shape

    @ property
    def dtype (self):
        return self.cached_data.dtype

    def backward (self, out_grad=None):
        # 最后一个节点时，out_grad为1
        if out_grad:
            out_grad = out_grad
        else:
            out_grad = Tensor(numpy.ones(self.shape))
            compute_gradient_of_variables(self, out_grad)


    def detach (self):
        # 创建一个新的张量，但不接入计算图
        return Tensor.make_const(self.realize_cached_data())
    def __add__ (self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __sub__ (self, other):
        assert(False)

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        if isinstance(data, Tensor):
            tensor_data = data
        else:
            tensor_data = data.realize_cached_data()
        tensor._init(None, [], # 将前置节点置空
                    cached_data = tensor_data, requires_grad = requires_grad)
        return tensor
    
def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor]))) # 请自行实现拓扑排序函数
    for node in reverse_topo_order:
        node.grad = sum(node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        for i, grad in enumerate(node.op.gradient(node.grad, node)): # 计算node.inputs的partial adjoint
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict

    
# class EWiseMul(TensorOp): # 对应元素乘
# class MulScalar(TensorOp): # 乘常数
# class PowerScalar(TensorOp): # 常数幂
# class EWiseDiv(TensorOp): # 对应元素除
# class DivScalar(TensorOp): # 除以常数
# class Transpose(TensorOp): # 矩阵转置
# class Reshape(TensorOp): # 变形
# class BroadcastTo(TensorOp): # 广播
# class Summation(TensorOp): # 按维度求和
# class MatMul(TensorOp): # 矩阵相乘
# class Negate(TensorOp): # 求相反数
# class Log(TensorOp): # 求对数
# class Exp(TensorOp): # 求指