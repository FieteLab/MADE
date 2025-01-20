from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from made.metrics import Metric, Euclidean, PeriodicEuclidean


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


# ---------------------------------------------------------------------------- #
#                                   MANIFOLDS                                  #
# ---------------------------------------------------------------------------- #
class AbstractManifold:
    def visualize(self, ax: plt.Axes):
        self.parameter_space.visualize(ax)

    def contains(self, point: np.ndarray) -> bool:
        for i, r in enumerate(self.parameter_space.ranges):
            if not r.start <= point[i] <= r.end:
                return False
        return True


# ----------------------------------- plane ---------------------------------- #
@dataclass
class Plane(AbstractManifold):
    dim: int = 2
    parameter_space: ParameterSpace = ParameterSpace(
        [Range(0, 1), Range(0, 1)]
    )
    metric: Metric = Euclidean(dim)


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
