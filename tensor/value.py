from numpy import ndarray
from typing import List, Optional
from tensor.ops.op import Op

class Value:
    op: Optional[Op] # 节点对应的计算操作， Op是自定义的计算操作类
    inputs: List["Value"]
    cached_data: ndarray
    requires_grad: bool

    def realize_cached_data(self): # 进行计算得到节点对应的变量，存储在cached_data里
        if self.is_leaf() or self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(*[x.realize_cached_data() for x in self.inputs])
        return self.cached_data
        
    def is_leaf(self):
        return self.op is None

    def __del__(self):
        assert(False)

    def _init(self, op: Optional[Op], inputs: List["Value"],
        *
        , num_outputs: int = 1,
        cached_data: ndarray = None, requires_grad: Optional[bool] = None):
        assert(False)   

    @classmethod
    def make_const(cls, data,
        *
        , requires_grad=False): # 建立一个用data生成的独立节点
        assert(False)   

    def make_from_op(cls, op: Op, inputs: List["Value"]): # 根据op生成节点
        assert(False)   
