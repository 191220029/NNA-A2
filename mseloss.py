class MSELoss:
    def __call__(self, predicted: "Tensor", target: "Tensor") -> "Tensor":
        t = ((predicted - target) ** 2).sum()
        return t