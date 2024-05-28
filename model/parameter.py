from tensor.tensor import Tensor
from module.module import Module
from typing import List

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