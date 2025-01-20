import matplotlib.pyplot as plt
from .manifolds import AbstractManifold
import numpy as np

def plot_lattice(mfld:AbstractManifold, show_distances:bool=False, distance_point:np.ndarray=None):
    f, ax = plt.subplots()
    ax.set_aspect('equal')
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
        plt.colorbar(contour, ax=ax, label='Distance')

    return f, ax
