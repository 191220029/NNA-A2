from abc import abstractmethod

class Optimizer():
    def __init__(self, params):
        self.params = params
    @abstractmethod
    def step(self):
        pass
    def reset_grad(self):
        for p in self.params:
            p.grad = None

# class SGD(Optimizer) # 基本随机梯度下降
# class SGD_WeightDecay(Optimizer) # L2 (L1) 正则化
# class Momentum(Optimizer) # 动量法 （Nestrov），以上三项可合并
# class Adam(Optimizer) # Adam
# class StepDecay(Scheduler) # 阶梯型衰减
# class LinearWarmUp(Scheduler) # 线性热启动
# class CosineDecayWithWarmRestarts(Scheduler) # 带热重启的Cosine衰减
