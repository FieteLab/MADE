from dataclasses import dataclass
import numpy as np
from loguru import logger


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
        return (
            self.alpha * np.exp(-(x**2 / 2 * self.sigma**2)) - self.alpha
        )


# ---------------------------------------------------------------------------- #
#                                      CAN                                     #
# ---------------------------------------------------------------------------- #
@dataclass
class CAN:
    manifold: AbstractManifold
    N: int  # N here represents the number of neurons per dimension
    alpha: float
    sigma: float

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

        # apply kernel
        self.connectivity_matrix = self.kernel(distances)
        logger.debug(
            f"Created CAN with {total_neurons} neurons. {self.connectivity_matrix.shape=}"
        )

        # initialize arrays to store the state and change in state of each neuron
        self.S = np.zeros((total_neurons, 1))
        self.S_dot = np.zeros((total_neurons, 1))
