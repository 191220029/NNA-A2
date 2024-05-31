import copy
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

    def __del__(self):
        self.op = None
        self.inputs = []
        self.cached_data = None
        if self.grad:
            self.grad.__del__()
    def clear(self):
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
        tensor.realize_cached_data()
        return tensor

    @ property
    def data(self) -> numpy.ndarray: #对cached_data进行封装
        return self.cached_data
    
    @ property
    def size(self):
        return self.cached_data.size

    @ data.setter
    def data(self, value):
        self.cached_data = numpy.array(value)

    @ property
    def shape(self):
        return self.cached_data.shape

    @ property
    def dtype(self):
        return self.cached_data.dtype

    def backward(self, out_grad=None):
        self.debug()
        # 最后一个节点时，out_grad为1
        if out_grad:
            out_grad = out_grad
        else:
            out_grad = Tensor(numpy.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)


    def detach(self):
        # 创建一个新的张量，但不接入计算图
        return Tensor.make_const(self.realize_cached_data())
    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)
    def __radd__(self, other):
        return self.__add__(other)
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, -other)
        else:
            return AddScalar(-other)(self)
    def __rsub__(self, other):
        return -self.__sub__(other)    
    def __neg__(self):
        return Negate()(self)
    def __matmul__(self, other):
        if isinstance(other, Tensor):
            return MatMul()(self, other)
        else:
            raise TypeError(f"Unsupported operand type(s) for @: 'Tensor' and '{type(other).__name__}'")
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __pow__(self, scalar):
        return PowerScalar(scalar)(self)
    def sum(self, axis=None, *args, **kwargs):
        return Summation(axis)(self)
    def reshape(self, shape, *args, **kwargs):
        return Reshape(shape)(self)
    
    @staticmethod
    def sum_tensors(tensor_list: List["Tensor"]):
        result = Tensor.from_numpy(tensor_list[0].data, tensor_list[0].dtype)
        for tensor in tensor_list[1:]:
            result = result + tensor
        return result
        
    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

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
    
    def clear_cache(self):
        self.cached_data = None
        for inp in self.inputs:
            inp.clear_cache()

    def debug(self):
        print(f"Tensor{{{self}, inputs: {self.inputs}, grad: {self.grad}}}")
        for i in self.inputs:
            i.debug()
    
def compute_gradient_of_variables(output_tensor, out_grad):
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {} # dict结构，用于存储partial adjoint
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor]))) # 请自行实现拓扑排序函数
    for node in reverse_topo_order:
        node.grad = Tensor.sum_tensors(node_to_output_grads_list[node]) # 求node的partial adjoint之和，存入属性grad
        if node.is_leaf():
            continue
        for i, grad in enumerate(node.op.gradient(node.grad, node)): # 计算node.inputs的partial adjoint
            j = node.inputs[i]
            if j not in node_to_output_grads_list:
                node_to_output_grads_list[j] = []
            node_to_output_grads_list[j].append(grad) # 将计算出的partial adjoint存入dict

    # 清理计算图
    for node in reverse_topo_order:
        node.clear()
    

def find_topo_sort(output_tensors) -> List[Tensor]:
    visited = set()
    topo_order = []

    def dfs(t):
        if t in visited:
            return
        visited.add(t)
        for inp in t.inputs:
            dfs(inp)
        topo_order.append(t)

    for t in output_tensors:
        dfs(t)
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
        return (out_grad, out_grad)

# 对应元素乘
class EWiseMul(TensorOp):
    def compute(self, a: numpy.ndarray, b: numpy.ndarray):
        return a * b
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        return (out_grad * node.inputs[1], out_grad * node.inputs[0])
    
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
        return (out_grad / b, -out_grad * a / (b * b))

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
        broadcasted_data = numpy.broadcast_to(a.data, self.shape)
        return broadcasted_data
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        input_shape = node.inputs[0].shape
        grad = Tensor.from_numpy(out_grad.data, out_grad.dtype)

        # 将广播后的梯度还原到原始形状
        for axis, dim in enumerate(input_shape):
            if dim == 1:
                grad = grad.sum(axis=axis, keepdims=True)
        
        return (grad,)

# 按维度求和
class Summation(TensorOp):
    def __init__(self, axis=None):
        self.axis = axis
    def compute(self, a: numpy.ndarray):
        return numpy.sum(a, axis=self.axis, keepdims=True)
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
        return a @ b
    def gradient(self, out_grad: 'Tensor', node: 'Tensor'):
        a, b = node.inputs
        grad_a = out_grad.data @ numpy.transpose(b.data)
        grad_b = numpy.transpose(a.data) @ out_grad.data
        return (grad_a, grad_b)

# 求相反数
class Negate(TensorOp):
    def compute(self, a: numpy.ndarray):
        return Tensor(-a)
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

