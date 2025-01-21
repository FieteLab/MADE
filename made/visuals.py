"""Visualization utilities for manifolds, CANs, and QANs.

This module provides functions to visualize:
1. Manifold geometries and distances
2. CAN connectivity and states
3. QAN trajectories and states
"""

import matplotlib.pyplot as plt
import numpy as np

from .manifolds import AbstractManifold, Sphere
from .can import CAN
from .qan import QAN

# ---------------------------------------------------------------------------- #
#                                     UTILS                                    #
# ---------------------------------------------------------------------------- #


def clean_axes(
    ax: plt.Axes,
    aspect: str = "equal",
    title: str = "",
    ylabel: str = "$\\theta_2$",
):
    """Apply consistent styling to matplotlib axes.

    Args:
        ax: The matplotlib axes to style
        aspect: Aspect ratio for the plot ('equal' or 'auto')
        title: Plot title
        ylabel: Label for y-axis (defaults to θ₂)
    """
    ax.set_aspect(aspect)
    ax.set(xlabel="$\\theta_1$", ylabel=ylabel)
    # remove splines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # space out left/bottom splines
    ax.spines["left"].set_position(("outward", 10))
    ax.spines["bottom"].set_position(("outward", 10))

    ax.set_title(title)


def _visualize_conn_sphere(ax, can, neuron_idx, cmap="bwr", vmin=-1, vmax=0):
    """Helper function to visualize connectivity for a sphere manifold.

    Args:
        ax: The matplotlib 3D axes to plot on
        can: The CAN instance
        neuron_idx: Index of the neuron whose connectivity to visualize
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        The matplotlib axes with the plot
    """
    # Get connectivity for this neuron
    neuron_connectivity = can.connectivity_matrix[neuron_idx]

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
    neuron_coords = can.neurons_coordinates[neuron_idx]
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
    return ax


def _visualize_conn_1d(ax, can, neuron_idx):
    """Helper function to visualize connectivity for a 1D manifold.

    Args:
        ax: The matplotlib axes to plot on
        can: The CAN instance
        neuron_idx: Index of the neuron whose connectivity to visualize

    Returns:
        The matplotlib axes with the plot
    """
    # Get connectivity for this neuron
    neuron_connectivity = can.connectivity_matrix[neuron_idx]

    # Plot connectivity as a line
    ax.plot(
        can.neurons_coordinates[:, 0],
        neuron_connectivity,
        "b-",
        label="Connectivity",
    )

    # Plot the selected neuron location
    neuron_coord = can.neurons_coordinates[neuron_idx]
    ax.scatter(
        neuron_coord[0],
        0,
        color="red",
        s=100,
        marker="*",
        label="Selected neuron",
    )

    ax.legend()
    clean_axes(ax, ylabel="Connectivity")
    return ax


def _visualize_conn_2d(ax, can, neuron_idx, cmap="bwr", vmin=-1, vmax=0):
    """Helper function to visualize connectivity for a 2D manifold.

    Args:
        ax: The matplotlib axes to plot on
        can: The CAN instance
        neuron_idx: Index of the neuron whose connectivity to visualize
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        The matplotlib axes with the plot
    """
    # Calculate grid dimensions based on spacing
    nx = can.nx(0)
    ny = can.nx(1)

    # Reshape coordinates into 2D grids
    X = can.neurons_coordinates[:, 0].reshape(ny, nx)
    Y = can.neurons_coordinates[:, 1].reshape(ny, nx)

    # Get connectivity for this neuron and reshape to grid
    neuron_connectivity = can.connectivity_matrix[neuron_idx].reshape(ny, nx)

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
    neuron_coords = can.neurons_coordinates[neuron_idx]
    ax.scatter(
        neuron_coords[0],
        neuron_coords[1],
        color="black",
        s=100,
        marker="*",
        label="Selected neuron",
    )

    ax.legend()
    clean_axes(ax)
    return ax


def _visualize_conn_sphere_2d(
    ax, can, neuron_idx, cmap="bwr", vmin=-1, vmax=0
):
    """Helper function to visualize connectivity for a sphere manifold using Mollweide projection.

    Args:
        ax: The matplotlib axes to plot on
        can: The CAN instance
        neuron_idx: Index of the neuron whose connectivity to visualize
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        The matplotlib axes with the plot
    """
    # Get connectivity for this neuron
    neuron_connectivity = can.connectivity_matrix[neuron_idx]

    # Convert cartesian coordinates to spherical coordinates (theta, phi)
    x = can.neurons_coordinates[:, 0]
    y = can.neurons_coordinates[:, 1]
    z = can.neurons_coordinates[:, 2]

    # Calculate spherical coordinates
    theta = np.arccos(z)  # polar angle [0, pi]
    phi = np.arctan2(y, x)  # azimuthal angle [-pi, pi]

    # Convert to Mollweide projection coordinates
    # First, rescale phi to [-pi, pi] and theta to [-pi/2, pi/2]
    lat = np.pi / 2 - theta  # latitude [-pi/2, pi/2]
    lon = phi  # longitude [-pi, pi]

    # Mollweide projection equations
    # We need to solve the equation: 2θ + sin(2θ) = π sin(lat) for θ
    # Use a simple iterative solution
    theta = lat
    for i in range(4):  # usually converges in a few iterations
        theta = theta - (
            2 * theta + np.sin(2 * theta) - np.pi * np.sin(lat)
        ) / (2 + 2 * np.cos(2 * theta))

    x_proj = 2 * np.sqrt(2) / np.pi * lon * np.cos(theta)
    y_proj = np.sqrt(2) * np.sin(theta)

    # Create scatter plot with connectivity as color
    scatter = ax.scatter(
        x_proj,
        y_proj,
        c=neuron_connectivity,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        s=15,
    )

    # Plot the selected neuron location
    neuron_coords = can.neurons_coordinates[neuron_idx]
    # Project the selected neuron's coordinates
    theta_n = np.arccos(neuron_coords[2])
    phi_n = np.arctan2(neuron_coords[1], neuron_coords[0])
    lat_n = np.pi / 2 - theta_n
    lon_n = phi_n

    # Solve for projection coordinates
    theta_n = lat_n
    for i in range(4):
        theta_n = theta_n - (
            2 * theta_n + np.sin(2 * theta_n) - np.pi * np.sin(lat_n)
        ) / (2 + 2 * np.cos(2 * theta_n))

    x_proj_n = 2 * np.sqrt(2) / np.pi * lon_n * np.cos(theta_n)
    y_proj_n = np.sqrt(2) * np.sin(theta_n)

    ax.scatter(
        x_proj_n,
        y_proj_n,
        color="black",
        s=100,
        marker="*",
        label="Selected neuron",
    )

    plt.colorbar(scatter, ax=ax)
    ax.legend()
    clean_axes(ax)
    return ax


# ---------------------------------------------------------------------------- #
#                                      CAN                                     #
# ---------------------------------------------------------------------------- #


def visualize_manifold(
    mfld: AbstractManifold,
    show_distances: bool = False,
    distance_point: np.ndarray = None,
    cmap="Greens",
):
    """Visualize a manifold and optionally show distances from a reference point.

    Args:
        mfld: The manifold to visualize
        show_distances: Whether to show distances from a reference point
        distance_point: Reference point for distance calculations
        cmap: Colormap for distance visualization

    Returns:
        fig, ax: The matplotlib figure and axes with the plot
    """
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


def visualize_can_connectivity(can: CAN, cmap="bwr", vmin=-1, vmax=0):
    """Visualize the connectivity of randomly selected neurons in a CAN.

    Selects 4 random neurons and plots their connectivity to the rest of the network.
    The visualization method depends on the manifold type:
    - 1D manifolds: Line plots
    - 2D manifolds: Contour plots
    - Sphere: Mollweide projection

    Args:
        can: The CAN instance to visualize
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        fig, axes: The matplotlib figure and axes with the plots
    """
    total_neurons = can.neurons_coordinates.shape[0]
    neurons_idx = np.random.choice(total_neurons, 4, replace=False)

    if isinstance(can.manifold, Sphere):
        # Create a figure with both 3D and 2D projections
        f = plt.figure(figsize=(20, 10))
        for i, neuron_idx in enumerate(neurons_idx):
            # 2D Mollweide projection
            ax2 = f.add_subplot(2, 2, i + 5)
            _visualize_conn_sphere_2d(ax2, can, neuron_idx, cmap, vmin, vmax)
            ax2.set_title(f"Neuron {neuron_idx} (Mollweide)")

    elif can.manifold.dim == 1:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, neuron_idx in zip(axes.flatten(), neurons_idx):
            _visualize_conn_1d(ax, can, neuron_idx)
            ax.set_title(f"Neuron {neuron_idx}")

    else:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        for ax, neuron_idx in zip(axes.flatten(), neurons_idx):
            _visualize_conn_2d(ax, can, neuron_idx, cmap, vmin, vmax)
            ax.set_title(f"Neuron {neuron_idx}")

    plt.tight_layout()
    return f, axes


def visualize_can_state(can: CAN):
    """Visualize the current state of neurons in a CAN.

    The visualization method depends on the manifold type:
    - 1D manifolds: Line plot with heights showing activation
    - 2D manifolds: Scatter plot with colors showing activation
    - Sphere: 3D scatter plot with colors showing activation

    Args:
        can: The CAN instance to visualize

    Returns:
        fig, ax: The matplotlib figure and axes with the plot
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


# ---------------------------------------------------------------------------- #
#                                      QAN                                     #
# ---------------------------------------------------------------------------- #


def visualize_qan_connectivity(qan: QAN, cmap="bwr", vmin=-1, vmax=0):
    """Visualize the connectivity in each CAN of a QAN.

    Selects a random neuron and shows its connectivity in each component CAN.
    The visualization method depends on the manifold type, similar to
    visualize_can_connectivity.

    Args:
        qan: The QAN instance to visualize
        cmap: Colormap for connectivity values
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling

    Returns:
        fig, axes: The matplotlib figure and axes with the plots
    """
    # Select random neuron index
    total_neurons = qan.cans[0].neurons_coordinates.shape[0]
    neuron_idx = np.random.choice(total_neurons)

    # Create figure based on manifold type
    if isinstance(qan.cans[0].manifold, Sphere):
        f = plt.figure(figsize=(15, 10))
        for i, can in enumerate(qan.cans):
            ax = f.add_subplot(3, 2, i + 1)
            _visualize_conn_sphere_2d(ax, can, neuron_idx, cmap, vmin, vmax)
            ax.set_title(f"CAN {i+1}")

    elif qan.cans[0].manifold.dim == 1:
        f, axes = plt.subplots(2, 1, figsize=(10, 10))
        for i, (ax, can) in enumerate(zip(axes.flatten(), qan.cans)):
            _visualize_conn_1d(ax, can, neuron_idx)
            ax.set_title(f"CAN {i+1}")

    else:
        f, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, (ax, can) in enumerate(zip(axes.flatten(), qan.cans)):
            _visualize_conn_2d(ax, can, neuron_idx, cmap, vmin, vmax)
            ax.set_title(f"CAN {i+1}")

    plt.tight_layout()
    return f, f.axes


def remove_jump(x):
    """Remove discontinuities in a trajectory by inserting NaN values.

    Used to prevent drawing lines across periodic boundaries or
    discontinuities when plotting trajectories.

    Args:
        x: Array of trajectory points

    Returns:
        Array with NaN values inserted at jump points
    """
    delta_x = np.diff(x, prepend=0, axis=0)
    jumps = np.where(delta_x > 0.5)[0]
    x[jumps] = np.nan
    return x


def visualize_trajectory(
    mfld: AbstractManifold,
    traj1: np.ndarray,
    traj2: np.ndarray = None,
    title: str = "Trajectory",
):
    """Visualize one or two trajectories on a manifold.

    Can be used to compare a target trajectory with a simulated one.
    The visualization method depends on the manifold type:
    - 1D manifolds: Plot coordinate vs time
    - 2D manifolds: Plot in the plane
    - Sphere: Plot on the surface in 3D

    Args:
        mfld: The manifold the trajectory lives on
        traj1: First trajectory as (n_steps, dim) array
        traj2: Optional second trajectory as (n_steps, dim) array
        title: Plot title

    Returns:
        fig, ax: The matplotlib figure and axes with the plot
    """
    if isinstance(mfld, Sphere):
        f = plt.figure()
        ax = f.add_subplot(111, projection="3d")

        # Plot the sphere surface points
        pts = mfld.parameter_space.sample(1000)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.1, s=1)

        # Plot trajectories
        ax.plot(
            traj1[:, 0],
            traj1[:, 1],
            traj1[:, 2],
            "b-",
            label="Ground Truth",
            alpha=0.7,
        )
        ax.scatter(traj1[0, 0], traj1[0, 1], traj1[0, 2], c="b", s=25)

        if traj2 is not None:
            ax.plot(
                traj2[:, 0],
                traj2[:, 1],
                traj2[:, 2],
                "r--",
                label="Model",
                alpha=0.7,
            )
            ax.scatter(traj2[0, 0], traj2[0, 1], traj2[0, 2], c="r", s=25)
            ax.legend()

        ax.set_title(title)
        return f, ax

    f, ax = plt.subplots()

    # Plot the manifold first
    mfld.visualize(ax)

    # Plot trajectories based on manifold dimension
    if mfld.dim == 1:
        # For 1D, x is coordinate and y is time
        times = np.arange(len(traj1)) / 250
        ax.plot(traj1[:, 0], times, "b-", label="Real", alpha=0.7)
        if traj2 is not None:
            ax.plot(
                traj2[:, 0],
                np.arange(len(traj2)) / 250,
                "r--",
                label="Simulated",
                alpha=0.7,
            )
        clean_axes(ax, ylabel="Time (250x)", title=title)

    else:
        # For 2D, plot x,y coordinates
        traj1 = remove_jump(traj1)

        ax.plot(
            traj1[:, 0], traj1[:, 1], "b-", label="Trajectory 1", alpha=0.7
        )
        ax.scatter(traj1[2, 0], traj1[2, 1], c="b", s=25)
        if traj2 is not None:
            ax.plot(
                traj2[:, 0],
                traj2[:, 1],
                "r--",
                label="Trajectory 2",
                alpha=0.7,
            )
            ax.scatter(traj2[2, 0], traj2[2, 1], c="r", s=25)
        clean_axes(ax, title=title)

    if traj2 is not None:
        ax.legend()

    return f, ax
