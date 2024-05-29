from typing import List
from abc import abstractmethod
from tensor import Tensor


class Parameter(Tensor): # 声明一个类专门表示网络参数
    def _unpack_params(value: object) -> List[Tensor]:
        if isinstance(value, Parameter): return [value]
        elif isinstance(value, Module): return value.parameters()
        elif isinstance(value, dict):
            params = []
            for _, v in value.items():
                params += Parameter._unpack_params(v)
            return params
        elif isinstance(value, (list, tuple)):
            params = []
            for v in value:
                params += Parameter._unpack_params(v)
            return params
        else: return []


class Module():
    def __init__(self):
        self.training = True

    def parameters(self) -> List["Tensor"]:
        return Parameter._unpack_params(self.__dict__)
    
    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    @abstractmethod
    def forward(self):
        pass

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for _, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []

# class Flatten(Module) # 平铺层
# class ReLU(Module) # ReLU激活函数
# class Sigmoid(Module) # Sigmoid激活函数
# class Softmax(Module) # Softmax层
# class CrossEntrophyLoss(Module) # 交叉熵损失
# class BinaryCrossEntrophyLoss(Module) # 二元交叉熵损失
# class MSELoss(Module) # 均方损失
# class BatchNorm1d(Module) # 一维批归一化 （选做）
# class LayerNorm1d(Module) # 一维层归一化 （选做）
# class DropOut(Module) # Dropout层 （选做）
# class Residual(Module) # 残差连