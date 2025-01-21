import matplotlib.pyplot as plt
import numpy as np

from .manifolds import AbstractManifold, Sphere
from .can import CAN


def clean_axes(
    ax: plt.Axes,
    aspect: str = "equal",
    title: str = "",
    ylabel: str = "$\\theta_2$",
):
    ax.set_aspect(aspect)
    ax.set(xlabel="$\\theta_1$", ylabel=ylabel)
    # remove splines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # space out left/bottom splines
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_title(title)


def plot_lattice(
    mfld: AbstractManifold,
    show_distances: bool = False,
    distance_point: np.ndarray = None,
    cmap="Greens_r",
):
    if mfld.dim == 1:
        f, ax = plt.subplots()
        mfld.visualize(ax)

        if show_distances and distance_point is not None:
            # Plot the reference point
            ax.scatter(
                distance_point[0],
                0,
                color="red",
                s=100,
                marker="*",
                label="Reference point",
            )

            # Sample points along the manifold
            points = mfld.parameter_space.sample(100)
            # Ensure points are 2D array for metric calculation
            points = points.reshape(-1, 1)
            distance_point = distance_point.reshape(1, -1)

            # Calculate distances from the reference point to all sampled points
            distances = mfld.metric(distance_point, points)

            # Plot distances as a line above the manifold
            ax.plot(points[:, 0], distances.ravel(), "k-", label="Distance")
            ax.legend()

        clean_axes(ax, ylabel="Distance")

    elif not isinstance(mfld, Sphere):
        f, ax = plt.subplots()
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
            contour = ax.contourf(X, Y, Z, cmap=cmap, levels=25)
            plt.colorbar(contour, ax=ax, label="Distance")

            ax.scatter(
                distance_point[0],
                distance_point[1],
                color="red",
                s=100,
                marker="*",
                label="Selected point",
            )

        clean_axes(ax)

    else:
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")

        pts = mfld.parameter_space.sample(1000)

        if show_distances and distance_point is not None:
            distances = mfld.metric(pts, distance_point)
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=distances,
                cmap="inferno",
                s=15,
            )
        else:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=15)

        clean_axes(ax, title="Neuron state")

    return f, ax


def can_connectivity(can: CAN, cmap="bwr", vmin=-1, vmax=0):
    """
    Select 4 random neurons and plot their connectivity
    to the rest of the lattice using contour plots for 2D,
    line plots for 1D, or 3D scatter for sphere.
    """
    total_neurons = can.neurons_coordinates.shape[0]
    neurons_idx = np.random.choice(total_neurons, 4, replace=False)

    if isinstance(can.manifold, Sphere):
        f = plt.figure(figsize=(15, 10))
        for i in range(4):
            ax = f.add_subplot(2, 2, i + 1, projection="3d")

            # Get connectivity for this neuron
            neuron_connectivity = can.connectivity_matrix[neurons_idx[i]]

            # Create 3D scatter plot with connectivity as color
            scatter = ax.scatter(
                can.neurons_coordinates[:, 0],
                can.neurons_coordinates[:, 1],
                can.neurons_coordinates[:, 2],
                c=neuron_connectivity,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                s=15,
            )

            # Plot the selected neuron location
            neuron_coords = can.neurons_coordinates[neurons_idx[i]]
            ax.scatter(
                neuron_coords[0],
                neuron_coords[1],
                neuron_coords[2],
                color="black",
                s=100,
                marker="*",
                label="Selected neuron",
            )

            plt.colorbar(scatter, ax=ax)
            ax.legend()
            ax.set_title(f"Neuron {neurons_idx[i]}")

        axes = f.axes

    elif can.manifold.dim == 1:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, ax in enumerate(axes.flatten()):
            # Get connectivity for this neuron
            neuron_connectivity = can.connectivity_matrix[neurons_idx[i]]

            # Plot connectivity as a line
            ax.plot(
                can.neurons_coordinates[:, 0],
                neuron_connectivity,
                "b-",
                label="Connectivity",
            )

            # Plot the selected neuron location
            neuron_coord = can.neurons_coordinates[neurons_idx[i]]
            ax.scatter(
                neuron_coord[0],
                0,
                color="red",
                s=100,
                marker="*",
                label="Selected neuron",
            )

            ax.legend()
            clean_axes(
                ax, title=f"Neuron {neurons_idx[i]}", ylabel="Connectivity"
            )
    else:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        # Calculate grid dimensions based on spacing
        nx = can.nx(0)
        ny = can.nx(1)

        # Reshape coordinates into 2D grids
        X = can.neurons_coordinates[:, 0].reshape(ny, nx)
        Y = can.neurons_coordinates[:, 1].reshape(ny, nx)

        for i, ax in enumerate(axes.flatten()):
            # Get connectivity for this neuron and reshape to grid
            neuron_connectivity = can.connectivity_matrix[
                neurons_idx[i]
            ].reshape(ny, nx)

            # Create contour plot
            contour = ax.contourf(
                X,
                Y,
                neuron_connectivity,
                levels=50,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(contour, ax=ax)

            # Plot the selected neuron location
            neuron_coords = can.neurons_coordinates[neurons_idx[i]]
            ax.scatter(
                neuron_coords[0],
                neuron_coords[1],
                color="black",
                s=100,
                marker="*",
                label="Selected neuron",
            )

            ax.legend()
            clean_axes(ax, title=f"Neuron {neurons_idx[i]}")

    plt.tight_layout()
    return f, axes


def plot_can_state(can: CAN):
    """
    Visualize the current state of the CAN using a scatter plot.
    For 1D manifolds, plots along a line. For 2D manifolds, plots
    in the plane with color indicating state value. For sphere,
    plots in 3D with color indicating state value.
    """
    if isinstance(can.manifold, Sphere):
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")

        # Create 3D scatter plot with state as color
        scatter = ax.scatter(
            can.neurons_coordinates[:, 0],
            can.neurons_coordinates[:, 1],
            can.neurons_coordinates[:, 2],
            c=can.S.ravel(),
            cmap="inferno",
            s=15,
        )
        plt.colorbar(scatter, ax=ax, label="Neuron state")
        ax.set_title("Neuron state")

    else:
        f, ax = plt.subplots()
        can.manifold.visualize(ax)

        if can.manifold.dim == 1:
            # For 1D, plot state values as heights above the line
            ax.plot(
                can.neurons_coordinates[:, 0],
                can.S.ravel(),
                "b-",
                label="Neuron states",
            )
            ax.scatter(
                can.neurons_coordinates[:, 0],
                can.S.ravel(),
                c=can.S.ravel(),
                cmap="inferno",
                s=15,
            )
        else:
            # For 2D, use scatter plot with color indicating state
            scatter = ax.scatter(
                can.neurons_coordinates[:, 0],
                can.neurons_coordinates[:, 1],
                c=can.S.ravel(),
                cmap="inferno",
                s=15,
            )
            plt.colorbar(scatter, ax=ax, label="Neuron state")

        clean_axes(
            ax,
            title="Neuron state",
            ylabel="Activation" if can.manifold.dim == 1 else "$\\theta_2$",
        )

    return f, ax
