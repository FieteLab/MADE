"""Module for defining various manifold types and their parameter spaces.

This module provides classes for different types of manifolds (Line, Ring, Plane, etc.)
and their associated parameter spaces. Each manifold has a specific dimensionality,
parameter space, and metric for computing distances.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from made.metrics import (
    Metric,
    Euclidean,
    PeriodicEuclidean,
    MobiusEuclidean,
    SphericalDistance,
)


# ----------------------------------- range ---------------------------------- #
@dataclass
class Range:
    """Represents a range of values with optional periodicity.

    Attributes:
        start (float): Start value of the range
        end (float): End value of the range
        periodic (bool): Whether the range wraps around (default: False)
    """

    start: float
    end: float
    periodic: bool = False

    def sample(self, n: int, pad: float = 0.0) -> np.ndarray:
        """Sample n points uniformly from the range.
        If the range is periodic, we skip the end point to avoid duplicates.

        Args:
            n (int): Number of points to sample
            pad (float): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of n sampled points
        """
        return np.linspace(
            self.start + pad, self.end - pad, n, endpoint=not self.periodic
        )

    def rel2coord(self, rel: float) -> float:
        """Convert a relative position [0,1] to coordinates in [start, end].

        Args:
            rel (float): Relative position in [0,1]

        Returns:
            float: Absolute coordinate in the range
        """
        return self.start + rel * (self.end - self.start)


# --------------------------------- parameter space --------------------------------- #
@dataclass
class ParameterSpace:
    """Defines a parameter space composed of multiple ranges, one for each dimension.
    Parameter spaces effectively represent the "neural lattice" of a CAN at the limit
    of infinite neurons. The sampled points are the lattice points for a CAN with
    finite neurons.

    This class handles sampling points from multi-dimensional parameter spaces,
    supporting both uniform grid sampling and fixed-spacing sampling.
    Spacing by density ensures that a similar density of points is sampled
    across each dimensions, regardless of the range size.

    Attributes:
        ranges (list[Range]): List of Range objects defining each dimension
        dim (int): Number of dimensions in the parameter space
    """

    ranges: list[Range]

    def __post_init__(self):
        self.dim = len(self.ranges)

    def sample(self, n: int, pads: list[float] = None) -> np.ndarray:
        """
        Returns points sampled from the parameter space.
        For 1D: returns n points as an (n, 1) array
        For 2D: returns n^2 points as an (n^2, 2) array in grid format
        For higher dimensions we return dn points as an (dn, d) array

        Args:
            n (int): Number of points to sample
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim

        if self.dim == 1:
            assert (
                len(pads) == 1
            ), "Incorrect number of pasd for manifold dimension"
            return np.array([r.sample(n, pads[0]) for r in self.ranges]).T
        elif self.dim == 2:
            assert (
                len(pads) == 2
            ), "Incorrect number of pasd for manifold dimension"
            # Create meshgrid
            x = self.ranges[0].sample(n, pads[0])
            y = self.ranges[1].sample(n, pads[1])
            X, Y = np.meshgrid(x, y)
            # Return as (n^2, 2) array
            return np.column_stack((X.ravel(), Y.ravel()))
        else:
            return np.array([r.sample(n) for r in self.ranges]).T

    def sample_with_spacing(
        self, spacing: float, pads: list[float] = None
    ) -> np.ndarray:
        """
        Returns points sampled from the parameter space with a fixed spacing.
        For 1D: returns array of shape (n, 1) where n depends on the range size and spacing
        For 2D: returns array of shape (n*m, 2) where n,m depend on the range sizes and spacing
        For higher dimensions we return dn points as an (dn, d) array

        Args:
            spacing (float): Fixed spacing between points
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim

        if self.dim == 1:
            assert (
                len(pads) == 1
            ), "Incorrect number of pads for manifold dimension"
            range_size = self.ranges[0].end - self.ranges[0].start
            n = int(np.ceil(range_size / spacing))
            return np.array([self.ranges[0].sample(n, pads[0])]).T
        elif self.dim == 2:
            assert (
                len(pads) == 2
            ), "Incorrect number of pads for manifold dimension"
            # Calculate number of points needed in each dimension
            range_sizes = [r.end - r.start for r in self.ranges]
            ns = [int(np.ceil(size / spacing)) for size in range_sizes]

            # Create meshgrid
            x = self.ranges[0].sample(ns[0], pads[0])
            y = self.ranges[1].sample(ns[1], pads[1])
            X, Y = np.meshgrid(x, y)
            # Return as (n*m, 2) array
            return np.column_stack((X.ravel(), Y.ravel()))
        else:
            raise NotImplementedError("Only 1D and 2D manifolds are supported")

    def visualize(self, ax: plt.Axes):
        """Visualize the parameter space on a matplotlib axes.

        Args:
            ax (plt.Axes): Matplotlib axes for plotting
        """
        # if 1D, plot a line
        if self.dim == 1:
            ax.plot([self.ranges[0].start, self.ranges[0].end], [0, 0], "k-")

        # if 2D, plot a rectangle
        elif self.dim == 2:
            w, h = (
                self.ranges[0].end - self.ranges[0].start,
                self.ranges[1].end - self.ranges[1].start,
            )
            rect = plt.Rectangle(
                (self.ranges[0].start, self.ranges[1].start),
                w,
                h,
                fill=True,
                color="k",
                alpha=0.1,
            )
            ax.add_patch(rect)


class SphereParameterSpace(ParameterSpace):
    """Special parameter space for sampling points on a unit sphere.

    Uses the Fibonacci sphere method to generate approximately evenly
    distributed points on a unit sphere surface.
    """

    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Returns approximately evenly distributed points on a unit sphere.

        Args:
            n (int): Number of points to sample
            **kwargs: Additional arguments (unused)

        Returns:
            np.ndarray: Array of shape (n, 3) containing 3D coordinates on unit sphere
        """
        points = np.zeros((n, 3))
        phi = np.pi * (3 - np.sqrt(5))  # golden angle in radians

        for i in range(n):
            y = 1 - (i / float(n)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points[i] = [x, y, z]

        return points

    def sample_with_spacing(
        self, spacing: float, pads: list[float] = None
    ) -> np.ndarray:
        """For sphere, we ignore spacing and just return evenly distributed points"""
        return self.sample(1250)


# ---------------------------------------------------------------------------- #
#                                   MANIFOLDS                                    #
# ---------------------------------------------------------------------------- #
class AbstractManifold:
    """Base class for all manifolds.

    Provides common functionality for visualization and point containment checking.
    All specific manifold types should inherit from this class.
    """

    def visualize(self, ax: plt.Axes):
        """Visualize the manifold on a matplotlib axes.

        Args:
            ax (plt.Axes): Matplotlib axes for plotting
        """
        self.parameter_space.visualize(ax)

    def contains(self, point: np.ndarray) -> bool:
        """Check if a point lies within the manifold's parameter space.

        Args:
            point (np.ndarray): Point coordinates to check

        Returns:
            bool: True if point is contained in the manifold, False otherwise

        Raises:
            AssertionError: If point dimensionality doesn't match manifold
        """
        assert len(point) == self.dim, "Incorrect number of dimensions"
        for i, r in enumerate(self.parameter_space.ranges):
            if not r.start <= point[i] <= r.end:
                return False
        return True


# ----------------------------------- line ---------------------------------- #
@dataclass
class Line(AbstractManifold):
    """1D manifold representing a line segment.

    A simple 1D manifold with Euclidean metric.
    """

    dim: int = 1
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 10, periodic=False)]
    )
    metric: Metric = Euclidean(dim)


# ----------------------------------- ring ---------------------------------- #
@dataclass
class Ring(AbstractManifold):
    """1D manifold representing a circular ring.

    A periodic 1D manifold with periodic Euclidean metric.
    """

    dim: int = 1
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = PeriodicEuclidean(dim, periodic=[True])


# ----------------------------------- plane ---------------------------------- #
@dataclass
class Plane(AbstractManifold):
    """2D manifold representing a rectangular region of a plane.

    A simple 2D manifold with Euclidean metric.
    """

    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 2.5, periodic=False), Range(0, 2.5, periodic=False)]
    )
    metric: Metric = Euclidean(dim)


# --------------------------------- cylinder --------------------------------- #
@dataclass
class Cylinder(AbstractManifold):
    """2D manifold representing a cylinder surface.

    One dimension is periodic (angular) and one is non-periodic (height).
    """

    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 3, periodic=False), Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = PeriodicEuclidean(dim, periodic=[False, True])


# ----------------------------------- torus ---------------------------------- #
@dataclass
class Torus(AbstractManifold):
    """2D manifold representing a torus surface.

    Both dimensions are periodic, representing the two angular coordinates.
    """

    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [
            Range(0, 2 * np.pi, periodic=True),
            Range(0, 2 * np.pi, periodic=True),
        ]
    )
    metric: Metric = PeriodicEuclidean(dim, periodic=[True, True])


# --------------------------------- mobius band --------------------------------- #
@dataclass
class MobiusBand(AbstractManifold):
    """2D manifold representing a Möbius band.

    One dimension is non-periodic (width) and one is periodic (length) with a twist.
    Uses a special Möbius metric to handle the topological twist.
    """

    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(-2, 2, periodic=False), Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = MobiusEuclidean(T=2.0)


# ---------------------------------- sphere ---------------------------------- #
@dataclass
class Sphere(AbstractManifold):
    """2D Sphere manifold represented as a unit sphere embedded in 3D space.

    Although topologically 2D, points are represented in 3D coordinates.
    """

    dim: int = 3
    parameter_space: ParameterSpace = SphereParameterSpace(
        [
            Range(-1, 1, periodic=False),
            Range(-1, 1, periodic=False),
            Range(-1, 1, periodic=False),
        ]
    )
    metric: Metric = SphericalDistance(dim)


# Dictionary of padding values for different manifold types
PADS = dict(
    Plane=[0.2, 0.2],
    Torus=[0, 0],
    Cylinder=[0.2, 0.0],
    Line=[
        0.2,
    ],
    Ring=[
        0,
    ],
    MobiusBand=[0.2, 0.0],
    Sphere=[0, 0, 0],
)
