from tensor.tensor import Tensor
import numpy as np



if __name__ == '__main__':
    t = Tensor.from_numpy(np.array([1, 2, 3]))

    print(t.data)
