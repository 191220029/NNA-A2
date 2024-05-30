class MSELoss:
    def __call__(self, predicted: "Tensor", target: "Tensor") -> "Tensor":
        t = predicted - target
        t = t ** 2
        t = t.sum()
        return t