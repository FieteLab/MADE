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


# ---------------------------------------------------------------------------- #
#                               Euclidean                                    #
# ---------------------------------------------------------------------------- #
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
        dot_product = X @ X.T

        squared_distances = (
            square_norms[:, None] + square_norms[None, :] - 2 * dot_product
        )

        # Add small epsilon to prevent numerical issues with sqrt of very small numbers
        # and clip negative values that might occur due to numerical precision
        epsilon = 1e-10
        distances = np.sqrt(np.maximum(squared_distances, epsilon))
        return distances


# ---------------------------------------------------------------------------- #
#                               PeriodicEuclidean                              #
# ---------------------------------------------------------------------------- #
class PeriodicEuclidean(Metric):
    def __init__(self, dim: int, periodic: list[bool]):
        """
        Initialize PeriodicEuclidean metric.

        Args:
            dim: number of dimensions
            periodic: list of booleans indicating which dimensions are periodic
        """
        if len(periodic) != dim:
            raise ValueError(
                f"periodic must have length {dim}, got {len(periodic)}"
            )
        self.dim = dim
        self.periodic = np.array(periodic)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the distance between points with periodic boundary conditions.
        For periodic dimensions, computes angular distance with period 2π.
        For non-periodic dimensions, uses regular Euclidean distance.

        x, y can be of shape (n, dim) or (dim,)
        """
        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        # Compute differences
        diff = x - y

        # For periodic dimensions, take the minimum distance around the circle
        periodic_mask = self.periodic[None, :]  # Add batch dimension
        periodic_diff = np.where(periodic_mask, diff, 0)

        # Wrap periodic differences to [-π, π]
        wrapped_diff = np.mod(periodic_diff + np.pi, 2 * np.pi) - np.pi

        # Combine periodic and non-periodic differences
        final_diff = np.where(periodic_mask, wrapped_diff, diff)

        return np.linalg.norm(final_diff, axis=1)

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between all points, respecting periodic boundaries.

        X: array of shape (n_points, dim)
        Returns: array of shape (n_points, n_points)
        """
        n_points = X.shape[0]
        distances = np.zeros((n_points, n_points))

        # Compute differences for all pairs
        for i in range(n_points):
            distances[i, :] = self(X[i : i + 1], X)

        return distances


# ---------------------------------------------------------------------------- #
#                               MobiusEuclidean                                  #
# ---------------------------------------------------------------------------- #
class MobiusEuclidean(Metric):
    def __init__(self, T: float = 2.0, threshold: float = np.pi):
        """
        Initialize MobiusEuclidean metric for a Möbius strip.

        The metric assumes points are parametrized by:
            - t ∈ [-T, T] (height)
            - θ ∈ [0, 2π] (angle)

        Args:
            T: height of the manifold in the non-periodic direction
            threshold: angular threshold to determine if points are on the "same side"
        """
        self.T = T
        self.threshold = threshold
        # Create periodic metric for the angular dimension
        self.periodic = PeriodicEuclidean(dim=2, periodic=[False, True])

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute distance between points on a Möbius strip.

        For points with angular distance > threshold, one point's height
        is flipped before computing the distance to account for the
        strip's twist.

        Args:
            x, y: points of shape (n, 2) or (2,) where each point is (t, θ)
        """
        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)
        delta_theta_1 = np.abs(x[:, 1] - y[:, 1])
        to_flip = np.where(delta_theta_1 > np.pi)[0]
        x_transformed = x.copy()
        x_transformed[to_flip, 0] = -x_transformed[to_flip, 0]

        return self.periodic(x_transformed, y)

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between all points on the Möbius strip.

        Args:
            X: array of shape (n_points, 2) where each point is (t, θ)
        Returns:
            array of shape (n_points, n_points) with pairwise distances
        """
        n_points = X.shape[0]
        distances = np.zeros((n_points, n_points))

        for i in range(n_points):
            distances[i, :] = self(X, X[i : i + 1])

        return distances
