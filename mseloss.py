class MSELoss:
    def __call__(self, predicted: "Tensor", target: "Tensor") -> "Tensor":
        return ((predicted - target)**2).sum()