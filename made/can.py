"""Implementation of Continuous Attractor Networks (CANs) on various manifolds.

This module provides the core implementation of Continuous Attractor Networks,
which are neural networks that can maintain a continuous family of stable states.
The networks can be constructed on various manifolds (Line, Ring, Plane, etc.)
with different topological properties.
"""

from dataclasses import dataclass
import numpy as np
from loguru import logger
from typing import Literal, Callable

from .manifolds import (
    AbstractManifold,
    Plane,
    Torus,
    Cylinder,
    Line,
    Ring,
    MobiusBand,
    Sphere,
)


def relu(x):
    """Rectified Linear Unit activation function.

    Args:
        x: Input array or scalar

    Returns:
        Array or scalar with negative values set to 0
    """
    return np.maximum(0, x)


# ---------------------------------------------------------------------------- #
#                                    kernel                                    #
# ---------------------------------------------------------------------------- #


@dataclass
class Kernel:
    """Gaussian-like kernel function for neural connectivity.

    The kernel determines the connection strength between neurons based on their distance.
    It is defined as: K(x) = α * (exp(-x²/2σ²) - 1)

    Attributes:
        alpha (float): Scaling factor for the kernel
        sigma (float): Width parameter of the Gaussian
    """

    alpha: float
    sigma: float

    def __call__(self, d: float) -> float:
        """Compute kernel value for given distance.

        Args:
            d (float): Distance between neurons

        Returns:
            float: Connection strength between neurons at distance d
        """
        # Prevent overflow in exp by clipping large values
        exp_term = -(d**2) / (2 * self.sigma**2)
        return self.alpha * np.exp(exp_term) - self.alpha


# ---------------------------------------------------------------------------- #
#                                      CAN                                     #
# ---------------------------------------------------------------------------- #


def quality_check(X: np.ndarray, name: str):
    """Check array for NaN or Inf values and log errors if found.

    Args:
        X (np.ndarray): Array to check
        name (str): Name of the array for error reporting
    """
    if np.any(np.isnan(X)):
        logger.error(f"NaN values detected in {name}")
    if np.any(np.isinf(X)):
        logger.error(f"Inf values detected in {name}")


def default_weights_offset(x):
    """Placeholder functions for weights offset.

    Weights offsets are used to construct QANs, normal CANs do not need them.
    We use this placeholder such that the QAN class can create CANs with different offsets.
    """
    return x


@dataclass
class CAN:
    """Continuous Attractor Network implementation.

    A CAN is a neural network that can maintain a continuous family of stable states.
    The network is constructed on a manifold with a specific topology, and neurons
    are arranged in a regular grid on this manifold.

    Attributes:
        manifold (AbstractManifold): The manifold on which the CAN is constructed
        spacing (float): Spacing between neurons in the grid, toghether with the manifold
            parameter space ranges determines the number of neurons.
        alpha (float): Scaling factor for the connectivity kernel
        sigma (float): Width parameter for the connectivity kernel
        tau (float): Time constant for neural dynamics
        weights_offset (Callable): Function to modify distances before computing weights
    """

    manifold: AbstractManifold
    spacing: float  # spacing between neurons
    alpha: float
    sigma: float
    tau: float = 5
    weights_offset: Callable = default_weights_offset

    def __post_init__(self):
        """Initialize the CAN by setting up neurons and computing connectivity."""
        self.kernel = Kernel(self.alpha, self.sigma)

        # sample neurons in a uniform grid with the spacing
        self.neurons_coordinates = (
            self.manifold.parameter_space.sample_with_spacing(self.spacing)
        )

        # get connectivity matrix for all neurons
        total_neurons = self.neurons_coordinates.shape[0]
        distances = self.manifold.metric.pairwise_distances(
            self.neurons_coordinates, weights_offset=self.weights_offset
        )
        quality_check(distances, "distances")

        # apply kernel
        self.connectivity_matrix = self.kernel(distances)
        quality_check(self.connectivity_matrix, "connectivity_matrix")

        # initialize arrays to store the state and change in state of each neuron
        self.S = np.zeros((total_neurons, 1))

    def __repr__(self):
        return f"CAN(spacing={self.spacing}, N neurons={self.connectivity_matrix.shape[0]})"

    @classmethod
    def default(
        cls,
        topology: Literal[
            "Line",
            "Ring",
            "Plane",
            "Torus",
            "Cylinder",
            "MobiusBand",
            "Sphere",
        ] = "Plane",
    ):
        """Create a CAN with default parameters for a given topology.

        Args:
            topology (str): Name of the manifold topology to use

        Returns:
            CAN: A new CAN instance with default parameters for the specified topology

        Raises:
            ValueError: If the topology name is invalid
        """
        if topology.lower() == "line":
            manifold = Line()
            return cls(manifold, spacing=0.075, alpha=3, sigma=1)

        elif topology.lower() == "ring":
            manifold = Ring()
            return cls(manifold, spacing=0.075, alpha=3, sigma=1)

        elif topology.lower() == "plane":
            manifold = Plane()
            return cls(manifold, spacing=0.075, alpha=3, sigma=1)

        elif topology.lower() == "torus":
            manifold = Torus()
            return cls(manifold, spacing=0.2, alpha=2.5, sigma=5)

        elif topology.lower() == "cylinder":
            manifold = Cylinder()
            return cls(manifold, spacing=0.2, alpha=2, sigma=1)

        elif topology.lower() == "mobiusband":
            manifold = MobiusBand()
            return cls(manifold, spacing=0.2, alpha=2, sigma=2)

        elif topology.lower() == "sphere":
            manifold = Sphere()
            return cls(manifold, spacing=0.075, alpha=2, sigma=3)

        else:
            raise ValueError(f"Invalid topology: {topology}")

    def nx(self, dim: int) -> int:
        """Get number of neurons along a specific dimension.

        Args:
            dim (int): Index of the dimension

        Returns:
            int: Number of neurons along the specified dimension
        """
        ranges = self.manifold.parameter_space.ranges
        return int(
            np.ceil((ranges[dim].end - ranges[dim].start) / self.spacing)
        )

    def idx2coord(self, idx: int, dim: int) -> np.ndarray:
        """Convert neuron index to coordinate along a specific dimension.

        Args:
            idx (int): Index of the neuron
            dim (int): Index of the dimension

        Returns:
            float: Coordinate of the neuron along the specified dimension
        """
        rel = idx / self.nx(dim)
        return self.manifold.parameter_space.ranges[dim].rel2coord(rel)

    def reset(
        self,
        mode: Literal["random", "uniform", "point"] = "random",
        point: np.ndarray = None,
        radius: float = 0.1,
    ):
        """Reset the network state according to specified mode.

        The state can be reset in three different ways:
        - random: each neuron is assigned a random state between 0 and 1
        - uniform: each neuron is assigned a state of 0.5
        - point: each neuron is assigned a state of 1 if it is within a certain radius from a given point on the manifold,
            and 0 otherwise

        Args:
            mode (str): How to initialize the state ("random", "uniform", or "point")
            point (np.ndarray, optional): Center point for "point" mode
            radius (float, optional): Radius around point for "point" mode as a fraction of the maximum distance

        Raises:
            ValueError: If mode is "point" but point is not provided, or if mode is invalid
        """
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
            radius = np.max(distances) * radius

            # Set states based on distances
            self.S = np.zeros((N, 1))
            self.S[distances <= radius] = 1.0
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def __call__(self):
        """Update network state by one timestep."""
        self.S = self.step_stateless(self.S)

    def step_stateless(self, S, u=0):
        """Stateless version of the step function that takes state as input and returns new state.

        Args:
            S (np.ndarray): Current state of the network
            u (float): External input (default: 0)

        Returns:
            np.ndarray: New state of the network

        Raises:
            ValueError: If NaN values are detected in the new state
        """
        S_dot = self.connectivity_matrix @ S + u + 1
        new_S = S + (relu(S_dot) - S) / self.tau

        if np.any(np.isnan(new_S)):
            raise ValueError(f"NaN values detected in new state: {new_S}")

        return new_S

    def run(self, n_steps: int):
        """Run the network for a specified number of steps.

        Args:
            n_steps (int): Number of timesteps to simulate

        Returns:
            np.ndarray: Final state of the network
        """
        for _ in range(n_steps):
            self()
        return self.S

    def run_stateless(self, S, n_steps: int):
        """Stateless version of run that takes initial state and returns final state.

        Args:
            S (np.ndarray): Initial state
            n_steps (int): Number of timesteps to simulate

        Returns:
            np.ndarray: Final state after n_steps
        """
        current_S = S.copy()
        for _ in range(n_steps):
            current_S = self.step_stateless(current_S)
        return current_S
