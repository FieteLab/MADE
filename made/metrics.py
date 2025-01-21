"""Module for defining distance metrics on various manifolds.

This module provides different distance metrics used to compute distances between points
on various manifolds. Each metric is tailored to the specific topology of its manifold,
handling properties like periodicity (Ring, Torus) or special geometries (Möbius band, Sphere).
"""

import numpy as np


class Metric:
    """Base class for all distance metrics.

    This abstract class defines the interface that all metrics must implement.
    Each metric must provide methods to compute distances between individual points
    and pairwise distances between sets of points.
    """

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance between points x and y.

        Args:
            x (np.ndarray): First point(s) with shape (n, dim) or (dim,)
            y (np.ndarray): Second point(s) with shape (n, dim) or (dim,)

        Returns:
            float: Distance between x and y
        """
        raise NotImplementedError("Distance metric not implemented")

    def pairwise_distances(
        self, X: np.ndarray, weights_offset=lambda x: x
    ) -> np.ndarray:
        """Compute pairwise distances between all points in X.

        Args:
            X (np.ndarray): Points array of shape (n_points, dim)
            weights_offset (callable): Function to transform coordinates before computing distances,
                used in QANs to create asymmetric weights

        Returns:
            np.ndarray: Matrix of shape (n_points, n_points) with pairwise distances[i,j] = dist(x_i, f(x_j))
        """
        raise NotImplementedError("Pairwise distances not implemented")


# ---------------------------------------------------------------------------- #
#                               Euclidean                                        #
# ---------------------------------------------------------------------------- #
class Euclidean(Metric):
    """Standard Euclidean distance metric.

    Computes straight-line distances between points in Euclidean space.
    Used for manifolds like Line and Plane.

    Attributes:
        dim (int): Dimensionality of the space
    """

    def __init__(self, dim: int):
        """Initialize Euclidean metric.

        Args:
            dim (int): Number of dimensions
        """
        self.dim = dim

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute Euclidean distance between points.

        Args:
            x (np.ndarray): First point(s) with shape (n, dim) or (dim,)
            y (np.ndarray): Second point(s) with shape (n, dim) or (dim,)

        Returns:
            float: Euclidean distance(s) between x and y
        """
        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        return np.linalg.norm(x - y, axis=1)

    def pairwise_distances(
        self, X: np.ndarray, weights_offset=lambda x: x
    ) -> np.ndarray:
        """Compute pairwise Euclidean distances between all points.
        Uses a vectorized approach that avoids explicit loops.

        Args:
            X (np.ndarray): Points array of shape (n_points, dim)
            weights_offset (callable): Function to transform coordinates before computing distances

        Returns:
            np.ndarray: Matrix of shape (n_points, n_points) with pairwise distances[i,j] = dist(x_i, f(x_j))
            where f is the weights_offset function
        """
        # Apply weights offset to second set of points
        X_transformed = weights_offset(X.copy())

        # Compute squared norms for each point
        square_norms_orig = np.sum(X**2, axis=1)
        square_norms_trans = np.sum(X_transformed**2, axis=1)

        # Use broadcasting to compute pairwise distances:
        # dist^2(x,y) = ||x||^2 + ||y||^2 - 2<x,y>
        dot_product = X @ X_transformed.T

        squared_distances = (
            square_norms_orig[:, None]
            + square_norms_trans[None, :]
            - 2 * dot_product
        )

        # Add small epsilon to prevent numerical issues with sqrt of very small numbers
        # and clip negative values that might occur due to numerical precision
        epsilon = 1e-10
        distances = np.sqrt(np.maximum(squared_distances, epsilon))
        return distances


# ---------------------------------------------------------------------------- #
#                               PeriodicEuclidean                               #
# ---------------------------------------------------------------------------- #
class PeriodicEuclidean(Metric):
    """Euclidean distance metric with periodic boundary conditions.

    Used for manifolds with periodic dimensions like Ring, Cylinder, and Torus.
    For periodic dimensions, computes the shortest distance around the circle.

    Attributes:
        dim (int): Number of dimensions
        periodic (list[bool]): Which dimensions are periodic
    """

    def __init__(self, dim: int, periodic: list[bool]):
        """Initialize PeriodicEuclidean metric.

        Args:
            dim (int): Number of dimensions
            periodic (list[bool]): List indicating which dimensions are periodic

        Raises:
            ValueError: If periodic list length doesn't match dimension
        """
        if len(periodic) != dim:
            raise ValueError(
                f"periodic must have length {dim}, got {len(periodic)}"
            )
        self.dim = dim
        self.periodic = np.array(periodic)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance between points with periodic boundary conditions.

        For periodic dimensions, computes angular distance with period 2π.
        For non-periodic dimensions, uses regular Euclidean distance.

        Args:
            x (np.ndarray): First point(s) with shape (n, dim) or (dim,)
            y (np.ndarray): Second point(s) with shape (n, dim) or (dim,)

        Returns:
            float: Distance(s) between x and y respecting periodic boundaries
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

    def pairwise_distances(
        self, X: np.ndarray, weights_offset=lambda x: x
    ) -> np.ndarray:
        """Compute pairwise distances between all points, respecting periodic boundaries.

        Args:
            X (np.ndarray): Points array of shape (n_points, dim)
            weights_offset (callable): Function to transform coordinates before computing distances

        Returns:
            np.ndarray: Matrix of shape (n_points, n_points) with pairwise distances[i,j] = dist(x_i, f(x_j))
        """
        # Apply weights offset to second set of points
        X_transformed = weights_offset(X.copy())
        n_points = X.shape[0]
        distances = np.zeros((n_points, n_points))

        # Compute differences for all pairs
        for i in range(n_points):
            distances[i, :] = self(X[i : i + 1], X_transformed)

        return distances


# ---------------------------------------------------------------------------- #
#                               MobiusEuclidean                                  #
# ---------------------------------------------------------------------------- #
class MobiusEuclidean(Metric):
    """Distance metric for points on a Möbius strip.

    Handles the twist in the Möbius strip by flipping the height coordinate
    when points are on opposite sides of the strip (large angular separation).

    Attributes:
        T (float): Height of the manifold in the non-periodic direction
        threshold (float): Angular threshold to determine if points are on opposite sides
    """

    def __init__(self, T: float = 2.0, threshold: float = np.pi):
        """Initialize MobiusEuclidean metric.

        Args:
            T (float): Height of the manifold in the non-periodic direction
            threshold (float): Angular threshold to determine if points are on opposite sides
        """
        self.T = T
        self.threshold = threshold

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute distance between points on a Möbius strip.

        For points with angular distance > threshold, one point's height
        is flipped before computing the distance to account for the
        strip's twist.

        Args:
            x (np.ndarray): First point(s) with shape (n, 2) or (2,), each point is (height, angle)
            y (np.ndarray): Second point(s) with shape (n, 2) or (2,), each point is (height, angle)

        Returns:
            float: Distance(s) between x and y on the Möbius strip

        Raises:
            AssertionError: If shape mismatch occurs during computation
        """
        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        # Compute angular distance
        delta_theta = np.abs(x[:, 1] - y[:, 1])

        # For points with large angular distance, flip height of y
        y_transformed = y.copy()
        to_flip = delta_theta > self.threshold
        assert (
            to_flip.shape == y_transformed[:, 0].shape
        ), f"to_flip shape: {to_flip.shape}, y_transformed[:, 0] shape: {y_transformed[:, 0].shape}"
        y_transformed[to_flip, 0] = -y_transformed[to_flip, 0]

        # Compute Euclidean distance with periodic boundary on θ
        diff = x - y_transformed
        # Wrap angular differences to [-π, π]
        diff[:, 1] = np.mod(diff[:, 1] + np.pi, 2 * np.pi) - np.pi

        return np.linalg.norm(diff, axis=1)

    def pairwise_distances(
        self, X: np.ndarray, weights_offset=lambda x: x
    ) -> np.ndarray:
        """Compute pairwise distances between all points on the Möbius strip.

        Args:
            X (np.ndarray): Points array of shape (n_points, 2), each point is (height, angle)
            weights_offset (callable): Function to transform coordinates before computing distances

        Returns:
            np.ndarray: Matrix of shape (n_points, n_points) with pairwise distances[i,j] = dist(x_i, f(x_j))
        """
        # Apply weights offset to second set of points
        X_transformed = weights_offset(X.copy())
        n_points = X.shape[0]
        distances = np.zeros((n_points, n_points))

        for i in range(n_points):
            distances[i, :] = self(X[i : i + 1], X_transformed)

        return distances


# ---------------------------------------------------------------------------- #
#                               SphericalDistance                               #
# ---------------------------------------------------------------------------- #
class SphericalDistance(Metric):
    """Great circle distance metric for points on a sphere.

    Computes the shortest path distance between points along the surface
    of a sphere using the great circle distance formula.

    Attributes:
        radius (float): Radius of the sphere
        dim (int): Always 3 for points in 3D Cartesian coordinates
    """

    def __init__(self, radius: float = 1.0):
        """Initialize SphericalDistance metric.

        Args:
            radius (float): Radius of the sphere (default=1.0 for unit sphere)
        """
        self.radius = radius
        self.dim = 3  # x,y,z coordinates

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute the great circle distance between points on a sphere.

        Uses the dot product formula: d = R * arccos(<x,y>/(|x||y|))

        Args:
            x (np.ndarray): First point(s) with shape (n, 3) or (3,) in Cartesian coordinates
            y (np.ndarray): Second point(s) with shape (n, 3) or (3,) in Cartesian coordinates

        Returns:
            float: Great circle distance(s) between x and y
        """
        # ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        # Normalize vectors to ensure they're on unit sphere
        x_norm = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = y / np.linalg.norm(y, axis=1, keepdims=True)

        # Compute dot product
        dot_product = np.sum(x_norm * y_norm, axis=1)

        # Clip to avoid numerical issues with arccos
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Distance = R * arccos(dot_product)
        return self.radius * np.arccos(dot_product)

    def pairwise_distances(
        self, X: np.ndarray, weights_offset=lambda x: x
    ) -> np.ndarray:
        """Compute pairwise great circle distances between all points on the sphere.

        Args:
            X (np.ndarray): Points array of shape (n_points, 3) in Cartesian coordinates
            weights_offset (callable): Function to transform coordinates before computing distances

        Returns:
            np.ndarray: Matrix of shape (n_points, n_points) with pairwise distances[i,j] = dist(x_i, f(x_j))
        """
        # Apply weights offset to second set of points
        X_transformed = weights_offset(X)

        # Normalize all vectors to unit sphere
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
        X_transformed_norm = X_transformed / np.linalg.norm(
            X_transformed, axis=1, keepdims=True
        )

        # Compute all pairwise dot products
        dot_products = X_norm @ X_transformed_norm.T

        # Clip to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Convert to distances
        distances = self.radius * np.arccos(dot_products)

        return distances
