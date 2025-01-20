import numpy as np


class Metric:
    def distance(self, x:np.ndarray, y:np.ndarray) -> float:
        pass

class Euclidean(Metric):
    def __init__(self, dim:int):
        self.dim = dim

    def __call__(self, x:np.ndarray, y:np.ndarray) -> float:
        """
            Computes the Euclidean distance between points. 
            X,Y can be of shape (n, dim) or (dim,).
        """

        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        return np.linalg.norm(x - y, axis=1)
