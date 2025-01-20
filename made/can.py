from dataclasses import dataclass
import numpy as np
from loguru import logger
from typing import Literal

from .manifolds import AbstractManifold


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
        return self.alpha * (np.exp(exp_term) - 1)


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
    N: int  # N here represents the number of neurons per dimension
    alpha: float
    sigma: float
    tau: float = 1.5

    def __post_init__(self):
        self.kernel = Kernel(self.alpha, self.sigma)

        # sample N^dim neurons in a uniform grid
        self.neurons_coordinates = self.manifold.parameter_space.sample(self.N)

        # get an N^2 x N^2 connectivity matrix
        total_neurons = self.neurons_coordinates.shape[
            0
        ]  # This is N^2 for 2D manifolds
        distances = self.manifold.metric.pairwise_distances(
            self.neurons_coordinates
        )
        quality_check(distances, "distances")

        # apply kernel
        self.connectivity_matrix = self.kernel(distances)
        quality_check(self.connectivity_matrix, "connectivity_matrix")

        # initialize arrays to store the state and change in state of each neuron
        self.S = np.zeros((total_neurons, 1))
        self.S_dot = np.zeros((total_neurons, 1))

    def reset(
        self,
        mode: Literal["random", "uniform", "point"] = "random",
        point: np.ndarray = None,
        radius: float = None,
    ):
        N = self.connectivity_matrix.shape[0]
        if mode == "random":
            self.S = np.random.rand(N, 1)
        elif mode == "uniform":
            self.S = np.ones((N, 1)) * 0.5
        elif mode == "point":
            if point is None or radius is None:
                raise ValueError(
                    "For point mode, both point and radius must be provided"
                )

            # Ensure point is 2D array for consistency
            if len(point.shape) == 1:
                point = point.reshape(1, -1)

            # Calculate distances from the point to all neurons
            distances = self.manifold.metric(point, self.neurons_coordinates)

            # Set states based on distances
            self.S = np.zeros((N, 1))
            self.S[distances <= radius] = 1.0
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __call__(self):
        self.S_dot = self.connectivity_matrix @ self.S + 1
        self.S += (soft_relu(self.S_dot) - self.S) / self.tau

        if np.any(np.isnan(self.S)) or np.any(np.isnan(self.S_dot)):
            logger.error("NaN values detected in S or S_dot")
