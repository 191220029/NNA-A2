import numpy
from typing import Dict, List
from op import Op
from value import Value
from numpy import ndarray

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
        return self.cached_data

    @ data.setter
    def data (self, value):
        self.cached_data = numpy.array(value)

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
        if isinstance(other, Tensor):
            return EWiseAdd()(self, -other)
        else:
            return AddScalar(-other)(self)

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

def find_topo_sort(output_tensors):
    visited = set()
    topo_order = []

    def dfs(tensor):
        if tensor in visited:
            return
        visited.add(tensor)
        for inp in tensor.inputs:
            dfs(inp)
        topo_order.append(tensor)

    for tensor in output_tensors:
        dfs(tensor)
    return topo_order


class TensorOp(Op):
    # 继承计算操作类，实现张量特有的计算
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)

class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: ndarray):
        return a + self.scalar
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad,) #重载gradient函数的输出必须是Tuple
    
# 对应元素相加 
class EWiseAdd(TensorOp):
    def compute(self, a: ndarray, b: ndarray):
        return a + b
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return out_grad, out_grad

# 对应元素乘
class EWiseMul(TensorOp):
    def compute(self, a: numpy.ndarray, b: numpy.ndarray):
        return a * b
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return out_grad * node.inputs[1], out_grad * node.inputs[0]
    
# 乘常数
class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: numpy.ndarray):
        return a * self.scalar
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad * self.scalar,)

# 常数幂
class PowerScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: numpy.ndarray):
        return numpy.power(a, self.scalar)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad * self.scalar * numpy.power(node.inputs[0], self.scalar - 1),)

# 对应元素除
class EWiseDiv(TensorOp):
    def compute(self, a: numpy.ndarray, b: numpy.ndarray):
        return a / b
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        a, b = node.inputs
        return out_grad / b, -out_grad * a / (b * b)

# 除以常数
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar
    def compute(self, a: numpy.ndarray):
        return a / self.scalar
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad / self.scalar,)

# 矩阵转置
class Transpose(TensorOp):
    def compute(self, a: numpy.ndarray):
        return numpy.transpose(a)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (numpy.transpose(out_grad),)

# 变形
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, a: numpy.ndarray):
        return numpy.reshape(a, self.shape)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (numpy.reshape(out_grad, node.inputs[0].shape),)

# 广播
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape
    def compute(self, a: numpy.ndarray):
        return numpy.broadcast_to(a, self.shape)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        input_shape = node.inputs[0].shape
        axes = tuple(range(out_grad.ndim - len(input_shape)))
        axes += tuple(i for i, s in enumerate(input_shape) if s == 1)
        return (numpy.sum(out_grad, axis=axes, keepdims=True),)

# 按维度求和
class Summation(TensorOp):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, a: numpy.ndarray):
        return numpy.sum(a, axis=self.axis)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        input_shape = node.inputs[0].shape
        if self.axis is None:
            return (numpy.broadcast_to(out_grad, input_shape),)
        else:
            shape = list(out_grad.shape)
            for ax in sorted(self.axis if isinstance(self.axis, tuple) else (self.axis,)):
                shape.insert(ax, 1)
            return (numpy.broadcast_to(numpy.reshape(out_grad, shape), input_shape),)

# 矩阵相乘
class MatMul(TensorOp):
    def compute(self, a: numpy.ndarray, b: numpy.ndarray):
        return numpy.matmul(a, b)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        a, b = node.inputs
        grad_a = numpy.matmul(out_grad, numpy.transpose(b))
        grad_b = numpy.matmul(numpy.transpose(a), out_grad)
        return grad_a, grad_b

# 求相反数
class Negate(TensorOp):
    def compute(self, a: numpy.ndarray):
        return -a
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (-out_grad,)

# 求对数
class Log(TensorOp):
    def compute(self, a: numpy.ndarray):
        return numpy.log(a)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad / node.inputs[0],)

# 求指
class Exp(TensorOp):
    def compute(self, a: numpy.ndarray):
        return numpy.exp(a)
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad * numpy.exp(node.inputs[0]),)
