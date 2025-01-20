from dataclasses import dataclass
import numpy as np
from loguru import logger
from typing import Literal

from .manifolds import AbstractManifold, Plane, Torus, Cylinder


def soft_relu(x):
    return np.log(1 + np.exp(x))


# ---------------------------------------------------------------------------- #
#                                    kernel                                    #
# ---------------------------------------------------------------------------- #


@dataclass
class Kernel:
    alpha: float
    sigma: float

    def __call__(self, x: float) -> float:
        # Prevent overflow in exp by clipping large values
        exp_term = -(x**2) / (2 * self.sigma**2)
        return self.alpha * np.exp(exp_term) - self.alpha


# ---------------------------------------------------------------------------- #
#                                      CAN                                     #
# ---------------------------------------------------------------------------- #


def quality_check(X: np.ndarray, name: str):
    if np.any(np.isnan(X)):
        logger.error(f"NaN values detected in {name}")
    if np.any(np.isinf(X)):
        logger.error(f"Inf values detected in {name}")


@dataclass
class CAN:
    manifold: AbstractManifold
    spacing: float  # spacing between neurons
    alpha: float
    sigma: float
    tau: float = 1.7

    def __post_init__(self):
        self.kernel = Kernel(self.alpha, self.sigma)

        # Calculate number of neurons needed in each dimension based on spacing
        ranges = self.manifold.parameter_space.ranges
        N_per_dim = []
        for r in ranges:
            range_size = r.end - r.start
            n = int(np.ceil(range_size / self.spacing))
            N_per_dim.append(n)

        # sample neurons in a uniform grid with the calculated spacing
        self.neurons_coordinates = (
            self.manifold.parameter_space.sample_with_spacing(self.spacing)
        )

        # get connectivity matrix for all neurons
        total_neurons = self.neurons_coordinates.shape[0]
        distances = self.manifold.metric.pairwise_distances(
            self.neurons_coordinates
        )
        quality_check(distances, "distances")

        # apply kernel
        self.connectivity_matrix = self.kernel(distances)
        quality_check(self.connectivity_matrix, "connectivity_matrix")

        # initialize arrays to store the state and change in state of each neuron
        self.S = np.zeros((total_neurons, 1))

    def __repr__(self):
        return f"CAN(spacing={self.spacing}, N neurons={self.connectivity_matrix.shape[0]})"

    def idx2coord(self, idx: int, dim: int) -> np.ndarray:
        n = self.nx(dim)
        rel = idx / n
        return self.manifold.parameter_space.ranges[dim].rel2coord(rel)

    @classmethod
    def default(
        cls, topology: Literal["Plane", "Torus", "Cylinder"] = "Plane"
    ):
        if topology.lower() == "plane":
            manifold = Plane()
            return cls(manifold, spacing=0.075, alpha=3, sigma=1)

        elif topology.lower() == "torus":
            manifold = Torus()
            return cls(manifold, spacing=0.2, alpha=2.5, sigma=5)

        elif topology.lower() == "cylinder":
            manifold = Cylinder()
            return cls(manifold, spacing=0.1, alpha=2, sigma=1)

        else:
            raise ValueError(f"Invalid topology: {topology}")

    def nx(self, dim: int) -> int:
        ranges = self.manifold.parameter_space.ranges
        return int(
            np.ceil((ranges[dim].end - ranges[dim].start) / self.spacing)
        )

    def reset(
        self,
        mode: Literal["random", "uniform", "point"] = "random",
        point: np.ndarray = None,
    ):
        N = self.connectivity_matrix.shape[0]
        if mode == "random":
            self.S = np.random.rand(N, 1)
        elif mode == "uniform":
            self.S = np.ones((N, 1)) * 0.5
        elif mode == "point":
            if point is None:
                raise ValueError(
                    "For point mode, both point and radius must be provided"
                )

            # Ensure point is 2D array for consistency
            if len(point.shape) == 1:
                point = point.reshape(1, -1)

            # Calculate distances from the point to all neurons
            distances = self.manifold.metric(point, self.neurons_coordinates)
            radius = np.max(distances) * 0.1

            # Set states based on distances
            self.S = np.zeros((N, 1))
            self.S[distances <= radius] = 1.0
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __call__(self):
        self.S = self.step_stateless(self.S)

    def step_stateless(self, S):
        """Stateless version of the step function that takes state as input and returns new state"""
        S_dot = self.connectivity_matrix @ S + 1
        new_S = S + (soft_relu(S_dot) - S) / self.tau

        if np.any(np.isnan(new_S)):
            logger.error("NaN values detected in new state")

        return new_S

    def run(self, n_steps: int):
        for _ in range(n_steps):
            self()
        return self.S

    def run_stateless(self, S, n_steps: int):
        """Stateless version of run that takes initial state and returns final state"""
        current_S = S.copy()
        for _ in range(n_steps):
            current_S = self.step_stateless(current_S)
        return current_S
