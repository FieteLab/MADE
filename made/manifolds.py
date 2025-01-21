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
    start: float
    end: float
    periodic: bool = False

    def sample(self, n: int, pad: float = 0.0) -> np.ndarray:
        return np.linspace(
            self.start + pad, self.end - pad, n, endpoint=not self.periodic
        )

    def rel2coord(self, rel: float) -> float:
        return self.start + rel * (self.end - self.start)


# --------------------------------- parameter space --------------------------------- #
@dataclass
class ParameterSpace:
    ranges: list[Range]

    def __post_init__(self):
        self.dim = len(self.ranges)

    def sample(self, n: int, pads: list[float] = None) -> np.ndarray:
        """
        Returns points sampled from the parameter space.
        For 1D: returns n points as an (n, 1) array
        For 2D: returns n^2 points as an (n^2, 2) array in grid format
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
    def sample(self, n: int, **kwargs) -> np.ndarray:
        """Returns n approximately evenly distributed points on a unit sphere using fibonacci sphere method"""
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
        """For sphere, we ignore spacing and just return 1000 evenly distributed points"""
        return self.sample(1000)


# ---------------------------------------------------------------------------- #
#                                   MANIFOLDS                                  #
# ---------------------------------------------------------------------------- #
class AbstractManifold:
    def visualize(self, ax: plt.Axes):
        self.parameter_space.visualize(ax)

    def contains(self, point: np.ndarray) -> bool:
        assert len(point) == self.dim, "Incorrect number of dimensions"
        for i, r in enumerate(self.parameter_space.ranges):
            if not r.start <= point[i] <= r.end:
                return False
        return True


# ----------------------------------- line ---------------------------------- #
@dataclass
class Line(AbstractManifold):
    dim: int = 1
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 10, periodic=False)]
    )
    metric: Metric = Euclidean(dim)


# ----------------------------------- ring ---------------------------------- #
@dataclass
class Ring(AbstractManifold):
    dim: int = 1
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = PeriodicEuclidean(dim, periodic=[True])


# ----------------------------------- plane ---------------------------------- #
@dataclass
class Plane(AbstractManifold):
    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 2.5, periodic=False), Range(0, 2.5, periodic=False)]
    )
    metric: Metric = Euclidean(dim)


# --------------------------------- cylinder --------------------------------- #
@dataclass
class Cylinder(AbstractManifold):
    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 3, periodic=False), Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = PeriodicEuclidean(dim, periodic=[False, True])


# ----------------------------------- torus ---------------------------------- #
@dataclass
class Torus(AbstractManifold):
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
    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(-2, 2, periodic=False), Range(0, 2 * np.pi, periodic=True)]
    )
    metric: Metric = MobiusEuclidean(T=2.0)


# ---------------------------------- sphere ---------------------------------- #
@dataclass
class Sphere(AbstractManifold):
    """
    Although the sphere is a 2D manifold, we consider the unit sphere embeded in 3D space
    here, and thus have 3D coordinates.
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
