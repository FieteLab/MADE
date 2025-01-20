import numpy as np


class Metric:
    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the distance between points x and y.
        x, y can be of shape (n, dim) or (dim,)
        """
        pass

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between all points in X.
        X: array of shape (n_points, dim)
        Returns: array of shape (n_points, n_points) with pairwise distances
        """
        pass


class Euclidean(Metric):
    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
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

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all points.
        Uses a vectorized approach that avoids explicit loops.

        X: array of shape (n_points, dim)
        Returns: array of shape (n_points, n_points)
        """
        # Compute squared norms for each point
        square_norms = np.sum(X**2, axis=1)

        # Use broadcasting to compute pairwise distances:
        # dist^2(x,y) = ||x||^2 + ||y||^2 - 2<x,y>
        distances = np.sqrt(
            square_norms[:, None] + square_norms[None, :] - 2 * X @ X.T
        )

        return distances
