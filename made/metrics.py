import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors


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


# ---------------------------------------------------------------------------- #
#                               SphericalDistance                               #
# ---------------------------------------------------------------------------- #
class SphericalDistance(Metric):
    def __init__(self, radius: float = 1.0):
        """
        Initialize SphericalDistance metric for points on a sphere using great circle distance.

        Args:
            radius: radius of the sphere (default=1.0 for unit sphere)
        """
        self.radius = radius
        self.dim = 3  # x,y,z coordinates

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the great circle distance between points on a sphere.
        Uses the dot product formula: d = R * arccos(<x,y>/(|x||y|))

        Args:
            x, y: points of shape (n, 3) or (3,) representing points in 3D Cartesian coordinates
        Returns:
            distances with same units as radius
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

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise great circle distances between all points on the sphere.

        Args:
            X: array of shape (n_points, 3) representing points in 3D Cartesian coordinates
        Returns:
            array of shape (n_points, n_points) with pairwise distances
        """
        # Normalize all vectors to unit sphere
        X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

        # Compute all pairwise dot products
        dot_products = X_norm @ X_norm.T

        # Clip to avoid numerical issues
        dot_products = np.clip(dot_products, -1.0, 1.0)

        # Convert to distances
        distances = self.radius * np.arccos(dot_products)

        return distances


# ---------------------------------------------------------------------------- #
#                              KLEIN BOTTLE METRIC                             #
# ---------------------------------------------------------------------------- #
class KleinBottleMetric(Metric):
    def __init__(
        self,
        dim: int,
        parameter_space,
        n_points: int = 30,
        n_neighbors: int = 8,
    ):
        """
        Initialize KleinBottleMetric using a graph-based approach.

        Args:
            dim: dimension of the parameter space (should be 2)
            parameter_space: parameter space object defining the ranges
            n_points: number of points to sample in each dimension for graph construction
            n_neighbors: number of neighbors to connect in the graph
        """
        self.dim = dim
        self.n_points = n_points
        self.n_neighbors = n_neighbors

        # Sample points from parameter space
        self.points = parameter_space.sample(n_points)

        # Pre-compute embedded points in R4
        self.embedded_points = self.klein_bottle_embedding(self.points)

        # Create graph and compute distances
        self.graph = self._create_klein_bottle_graph()
        self.distance_matrix = nx.floyd_warshall_numpy(self.graph)

        # Pre-compute KDTree for fast nearest neighbor search in R4
        self.kdtree = NearestNeighbors(
            n_neighbors=8, algorithm="ball_tree", metric="euclidean"
        )
        self.kdtree.fit(self.embedded_points)

    def klein_bottle_embedding(self, pts: np.ndarray) -> np.ndarray:
        """Embed a point from the parameter space into R4."""
        # Parametric equations for the 4D Klein bottle
        x1 = (2 + np.cos(pts[:, 1])) * np.cos(pts[:, 0])
        x2 = (2 + np.cos(pts[:, 1])) * np.sin(pts[:, 0])
        x3 = np.sin(pts[:, 1])
        x4 = np.sin(pts[:, 1]) * np.cos(pts[:, 1] / 2)
        out = np.array([x1, x2, x3, x4]).T
        return out

    def _create_klein_bottle_graph(self) -> nx.Graph:
        """Create a graph where vertices are points and edges connect nearest neighbors in R4."""
        print("Creating Klein bottle graph...")
        # Create empty weighted graph
        G = nx.Graph()

        # Add all points as vertices
        for i in range(len(self.points)):
            G.add_node(i)

        # Use kNN to find nearest neighbors in R4 space
        kNN = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric="euclidean"
        )
        kNN.fit(self.embedded_points)

        # Get distances and neighbors
        distances, neighbors = kNN.kneighbors(self.embedded_points)

        # Add edges with weights based on R4 Euclidean distances
        for i in range(len(self.points)):
            for j, dist in zip(neighbors[i], distances[i]):
                if i != j:  # Avoid self-loops
                    G.add_edge(i, j, weight=dist)

        # Log graph statistics
        print(
            f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
        )
        return G

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the distance between points on the Klein bottle using shortest path
        through the pre-computed graph.

        Args:
            x, y: points of shape (n, 2) or (2,) where each point is (t, θ)
        Returns:
            distances between points
        """
        # Ensure shapes consistency
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        # Embed points in R4
        x_embedded = self.klein_bottle_embedding(x)
        y_embedded = self.klein_bottle_embedding(y)

        # Find k nearest neighbors and distances for all points at once
        x_dists, x_neighbors = self.kdtree.kneighbors(x_embedded)
        y_dists, y_neighbors = self.kdtree.kneighbors(y_embedded)

        # Get all pairwise shortest path distances between neighbors using the pre-computed distance matrix
        graph_distances = self.distance_matrix[
            x_neighbors[:, :, None], y_neighbors[0, None, :]
        ]

        # Compute weighted distances based on how close points are to their nearest neighbors
        weights_x = np.exp(
            -x_dists
        )  # Use exponential weighting for better numerical stability
        weights_y = np.exp(-y_dists)
        weights_x = weights_x / weights_x.sum(axis=1, keepdims=True)
        weights_y = weights_y / weights_y.sum(axis=1, keepdims=True)

        # Compute final distances as weighted average of shortest path distances
        distances = np.sum(
            weights_x[:, :, None] * weights_y[0, None, :] * graph_distances,
            axis=(1, 2),
        )

        return distances

    def pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between all points on the Klein bottle.

        Args:
            X: array of shape (n_points, 2) where each point is (u, v)
        Returns:
            array of shape (n_points, n_points) with pairwise distances
        """
        n_points = X.shape[0]
        distances = np.zeros((n_points, n_points))

        for i in range(n_points):
            distances[i, :] = self(X[i : i + 1], X)

        return distances
