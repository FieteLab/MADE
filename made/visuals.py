import matplotlib.pyplot as plt
import numpy as np

from .manifolds import AbstractManifold
from .can import CAN


def plot_lattice(
    mfld: AbstractManifold,
    show_distances: bool = False,
    distance_point: np.ndarray = None,
):
    f, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set(xlabel="$\\theta_1$", ylabel="$\\theta_2$")

    mfld.visualize(ax)

    # If show_distances, sample from param space and plot contours of distance from distance_point
    if show_distances and distance_point is not None:
        param_space = mfld.parameter_space
        n = 50  # number of points per dimension
        points = param_space.sample(n)  # n^2 x dim array for 2D
        distances = mfld.metric(points, distance_point)

        # Reshape distances back to grid for contour plot
        X = points[:, 0].reshape(n, n)
        Y = points[:, 1].reshape(n, n)
        Z = distances.reshape(n, n)

        # Create contour plot
        contour = ax.contourf(X, Y, Z, levels=25)
        plt.colorbar(contour, ax=ax, label="Distance")

    return f, ax


def can_connectivity_matrix(can: CAN):
    """imshow the connectivity matrix"""
    plt.imshow(can.connectivity_matrix, cmap="gray")
    plt.colorbar()


def can_connectivity(can: CAN):
    """
    Select 4 random neurons and plot their connectivity
    to the rest of the lattice using contour plots.
    """
    f, axes = plt.subplots(2, 2, figsize=(10, 10))

    total_neurons = can.neurons_coordinates.shape[0]
    neurons_idx = np.random.choice(total_neurons, 4, replace=False)

    # Reshape coordinates back to grid for plotting
    X = can.neurons_coordinates[:, 0].reshape(can.N, can.N)
    Y = can.neurons_coordinates[:, 1].reshape(can.N, can.N)

    for i, ax in enumerate(axes.flatten()):
        ax.set_aspect("equal")
        ax.set(xlabel="$\\theta_1$", ylabel="$\\theta_2$")
        ax.set_title(f"Neuron {neurons_idx[i]}")

        # Get connectivity for this neuron and reshape to grid
        neuron_connectivity = can.connectivity_matrix[neurons_idx[i]].reshape(
            can.N, can.N
        )

        # Create contour plot
        contour = ax.contourf(X, Y, neuron_connectivity, levels=25)
        plt.colorbar(contour, ax=ax)

        # Plot the selected neuron location
        neuron_coords = can.neurons_coordinates[neurons_idx[i]]
        ax.scatter(
            neuron_coords[0],
            neuron_coords[1],
            color="red",
            s=100,
            marker="*",
            label="Selected neuron",
        )
        ax.legend()

    plt.tight_layout()
    return f, axes
