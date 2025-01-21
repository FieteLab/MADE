import numpy as np
from dataclasses import dataclass

from made.manifolds import AbstractManifold
from made import manifolds
from made.can import CAN


# ---------------------------------------------------------------------------- #
#                                      QAN                                     #
# ---------------------------------------------------------------------------- #


class QAN:
    def __post_init__(self):
        self.offset_magnitude = self.offset_magnitude

        # create copies of the CAN with offset coordinates
        mfld = self.manifold
        self.cans = []
        for d in range(self.manifold.dim):
            for direction in [1, -1]:
                self.cans.append(
                    CAN(
                        mfld,
                        self.spacing,
                        self.alpha,
                        self.sigma,
                        weights_offset=lambda x: self.coordinates_offset(
                            x, d, direction, self.offset_magnitude
                        ),
                    )
                )

    def simulate(self, trajectory: np.ndarray) -> np.ndarray:
        dT = 1 / 10

        # 1. reset each CAN to the start of the trajectory
        theta_0 = trajectory[0, :].copy()
        states = []
        for can in self.cans:
            can.reset(mode="point", point=theta_0, radius=0.05)
            states.append(can.S)

        # 2. run simulation
        decoded_trajectory = []
        for t, theta in enumerate(trajectory):
            if t == 0:
                theta_dot = np.zeros(theta.shape)
            else:
                # compute variable velocity
                theta_dot = (
                    self.compute_theta_dot(
                        theta.copy(), trajectory[t - 1, :].copy()
                    )
                    * dT
                )

            # get total state
            S_tot = np.mean(states, axis=0)

            # update each CAN using the total state
            for i, can in enumerate(self.cans):
                can_input = self.compute_can_input(i, theta_dot, theta)
                states[i] = can.step_stateless(S_tot, can_input)

            # decode the state into a trajectory
            decoded_trajectory.append(self.decode_state(S_tot))

        out = np.array(decoded_trajectory)
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        assert (
            out.shape == trajectory.shape
        ), f"out.shape: {out.shape}, trajectory.shape: {trajectory.shape}"

        return out

    def decode_state(self, S: np.ndarray) -> np.ndarray:
        """Decode the network state into manifold coordinates by finding the peak activation location.

        Args:
            S: The network state vector

        Returns:
            The coordinates on the manifold corresponding to the peak activation
        """
        max_idx = np.argmax(S)
        return self.cans[0].neurons_coordinates[max_idx]


# ---------------------------------------------------------------------------- #
#                              SPECIFIC TOPOLOGIES                             #
# ---------------------------------------------------------------------------- #


# ----------------------------------- Line ----------------------------------- #
@dataclass
class LineQAN(QAN):
    manifold: AbstractManifold = manifolds.Line()
    spacing: float = 0.075
    alpha: float = 3
    sigma: float = 1
    offset_magnitude: float = 0.2
    beta: float = 1e2

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        # make a trajectory that moves up and down the line multiple times
        trajectory = np.zeros((n_steps, self.manifold.dim))

        vmin = self.manifold.parameter_space.ranges[0].start
        vmax = self.manifold.parameter_space.ranges[0].end

        # Split into 4 segments: vmin->vmax->vmin->vmax
        n_per_segment = n_steps // 4

        trajectory[:n_per_segment, 0] = np.linspace(vmin, vmax, n_per_segment)
        trajectory[n_per_segment : 2 * n_per_segment, 0] = np.linspace(
            vmax, vmin, n_per_segment
        )
        trajectory[2 * n_per_segment : 3 * n_per_segment, 0] = np.linspace(
            vmin, vmax, n_per_segment
        )
        trajectory[3 * n_per_segment :, 0] = np.linspace(
            vmax, vmin, n_steps - 3 * n_per_segment
        )

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        return theta - theta_prev

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        if i == 1:
            theta_dot = -theta_dot
        return self.beta * theta_dot


# ----------------------------------- Ring ----------------------------------- #
@dataclass
class RingQAN(QAN):
    manifold: AbstractManifold = manifolds.Ring()
    spacing: float = 0.075
    alpha: float = 3
    sigma: float = 1
    offset_magnitude: float = 0.2
    beta: float = 1e2

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        # make a trajectory that goes 0->2pi->0->2pi
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Split into 4 segments
        n_per_segment = n_steps // 4

        # First segment: 0 -> 2pi
        trajectory[:n_per_segment, 0] = np.linspace(
            0, 2 * np.pi, n_per_segment
        )

        # Second segment: 2pi -> 0
        trajectory[n_per_segment : 2 * n_per_segment, 0] = np.linspace(
            2 * np.pi, 0, n_per_segment
        )

        # Third segment: 0 -> 2pi
        trajectory[2 * n_per_segment : 3 * n_per_segment, 0] = np.linspace(
            0, 2 * np.pi, n_per_segment
        )

        # Fourth segment: 2pi -> 0
        trajectory[3 * n_per_segment :, 0] = np.linspace(
            2 * np.pi, 0, n_steps - 3 * n_per_segment
        )

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        delta_theta = theta - theta_prev

        if delta_theta > np.pi:
            if theta < theta_prev:
                theta += 2 * np.pi
            else:
                theta -= 2 * np.pi
            delta_theta = theta - theta_prev
        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        if i == 1:
            theta_dot = -theta_dot
        return self.beta * theta_dot


# ----------------------------------- Plane ---------------------------------- #
@dataclass
class PlaneQAN(QAN):
    manifold: AbstractManifold = manifolds.Plane()
    spacing: float = 0.065
    alpha: float = 3
    sigma: float = 1
    offset_magnitude: float = 0.2
    beta: float = 1e2

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Creates a space-filling trajectory over the plane using a modified Lissajous curve."""
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Get parameter space bounds with padding
        padding = 0.4
        x_min = self.manifold.parameter_space.ranges[0].start + padding
        x_max = self.manifold.parameter_space.ranges[0].end - padding
        y_min = self.manifold.parameter_space.ranges[1].start + padding
        y_max = self.manifold.parameter_space.ranges[1].end - padding

        # Create time parameter
        t = np.linspace(np.pi / 2, 2 * np.pi + np.pi / 2 - 0.1, n_steps)

        # Generate modified Lissajous curve
        # Using different frequencies and phase shifts creates a space-filling pattern
        x_scale = (x_max - x_min) / 2
        y_scale = (y_max - y_min) / 2
        x_offset = (x_max + x_min) / 2
        y_offset = (y_max + y_min) / 2

        trajectory[:, 0] = x_scale * np.sin(2 * t + np.pi) + x_offset
        trajectory[:, 1] = y_scale * np.sin(3 * t + np.pi / 2) + y_offset

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute velocity as the difference between current and previous position."""
        return theta - theta_prev

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1
        return sign * self.beta * theta_dot[dim]


# ----------------------------------- Torus ---------------------------------- #
@dataclass
class TorusQAN(QAN):
    manifold: AbstractManifold = manifolds.Torus()
    spacing: float = 0.2
    alpha: float = 2.5
    sigma: float = 2
    offset_magnitude: float = 0.25
    beta: float = 1.25e2  # control gain for velocity input

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude

        # wrap to [0, 2pi]
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Creates a trajectory that wraps around the torus multiple times.
        The trajectory follows a line with slope 2/3, creating an interesting pattern.
        """
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Create time parameter that goes around multiple times
        t = np.linspace(0.25, 6 * np.pi - 0.25, n_steps)  # 3 full rotations

        # First coordinate goes around major circle
        trajectory[:, 0] = np.mod(t, 2 * np.pi)
        # Second coordinate goes around minor circle at different rate
        trajectory[:, 1] = np.mod(2 / 3 * t, 2 * np.pi)

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute angular velocities accounting for periodic boundary conditions."""
        delta_theta = theta - theta_prev

        # Handle periodic boundary crossings for both angles
        for dim in range(2):
            if delta_theta[dim] > np.pi:
                delta_theta[dim] -= 2 * np.pi
            elif delta_theta[dim] < -np.pi:
                delta_theta[dim] += 2 * np.pi

        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on angular velocities and current position.

        Args:
            i: CAN index (0-3, two CANs per dimension)
            theta_dot: Angular velocities [dθ₁/dt, dθ₂/dt]
            theta: Current position on the manifold [θ₁, θ₂]
        """
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1

        return sign * self.beta * theta_dot[dim]


# ----------------------------------- Cylinder ---------------------------------- #
@dataclass
class CylinderQAN(QAN):
    manifold: AbstractManifold = manifolds.Cylinder()
    spacing: float = 0.2
    alpha: float = 2
    sigma: float = 1
    offset_magnitude: float = 0.2
    beta: float = 1.6e2  # control gain for velocity input

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude

        if dim == 1:
            # wrap to [0, 2pi]
            theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Creates a spiral trajectory that wraps around the cylinder multiple times."""
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Get parameter space bounds for height (z) with padding
        padding = 0.2
        z_min = self.manifold.parameter_space.ranges[0].start + padding
        z_max = self.manifold.parameter_space.ranges[0].end - padding

        # Create time parameter
        t = np.linspace(
            np.pi / 2, 4 * np.pi - 0.25, n_steps
        )  # 2 full rotations

        # Height varies linearly up and down
        z = np.concatenate(
            [
                np.linspace(z_min, z_max, n_steps // 2),  # Up
                np.linspace(z_max, z_min, n_steps // 2),  # Down
            ]
        )

        # First coordinate is height
        trajectory[:, 0] = z
        # Second coordinate is angle that wraps around
        trajectory[:, 1] = np.mod(t, 2 * np.pi)

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute velocities accounting for periodic boundary in angular dimension."""
        delta_theta = theta - theta_prev

        # Handle periodic boundary crossing for angular dimension
        if delta_theta[1] > np.pi:
            delta_theta[1] -= 2 * np.pi
        elif delta_theta[1] < -np.pi:
            delta_theta[1] += 2 * np.pi

        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities and current position.

        Args:
            i: CAN index (0-3, two CANs per dimension)
            theta_dot: Velocities [dz/dt, dθ/dt]
            theta: Current position on the manifold [z, θ]
        """
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1
        return sign * self.beta * theta_dot[dim]


# -------------------------------- Mobius band ------------------------------- #
@dataclass
class MobiusBandQAN(QAN):
    manifold: AbstractManifold = manifolds.MobiusBand()
    spacing: float = 0.2
    alpha: float = 2
    sigma: float = 2
    offset_magnitude: float = 0.2
    beta: float = 5e2  # control gain for velocity input

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta = theta.copy()  # Make a copy to avoid modifying the original
        if dim == 1:  # Angular dimension
            theta[:, dim] += direction * offset_magnitude
            theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)  # wrap to [0, 2π]
        else:  # Height dimension
            theta[:, dim] += direction * offset_magnitude
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Creates a trajectory that demonstrates the Mobius band's characteristic flip."""
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Get parameter space bounds for height with padding
        padding = 0.25
        h_min = self.manifold.parameter_space.ranges[0].start + padding
        h_max = self.manifold.parameter_space.ranges[0].end - padding

        # Create time parameter for one full rotation
        t = np.linspace(0, 2 * np.pi, n_steps)

        # Height flips from positive to negative as we go around
        # Using cosine to smoothly transition the height
        trajectory[:, 0] = (h_max - h_min) / 2 * np.cos(t / 2) + (
            h_max + h_min
        ) / 2

        # Angle simply wraps around once
        trajectory[:, 1] = t

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute velocities accounting for periodic boundary in angular dimension."""
        delta_theta = theta - theta_prev

        # Handle periodic boundary crossing for angular dimension
        if delta_theta[1] > np.pi:
            delta_theta[1] -= 2 * np.pi
        elif delta_theta[1] < -np.pi:
            delta_theta[1] += 2 * np.pi

        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities and current position.

        Args:
            i: CAN index (0-3, two CANs per dimension)
            theta_dot: Velocities [dh/dt, dθ/dt]
            theta: Current position on the manifold [h, θ]
        """
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1

        # For height dimension, flip direction based on angle
        if dim == 0:
            # Flip the height direction when crossing the twist
            if theta[1] > np.pi:
                sign = -sign

        return sign * self.beta * theta_dot[dim]


# ---------------------------------- Sphere ---------------------------------- #
@dataclass
class SphereQAN(QAN):
    manifold: AbstractManifold = manifolds.Sphere()
    spacing: float = 0.075
    alpha: float = 2
    sigma: float = 3
    offset_magnitude: float = 0.2
    beta: float = (
        2e2  # reduced from 2e4 to be more in line with other manifolds
    )

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        offset = direction * offset_magnitude
        x, y, z = 0, 1, 2  # map indices to coords
        if dim == 0:
            # Rotation around X axis: [0, -z, y]
            theta[:, y] += -offset * theta[:, z]  # -z
            theta[:, z] += offset * theta[:, y]  # y
        elif dim == 1:
            # Rotation around Y axis: [z, 0, -x]
            theta[:, x] += offset * theta[:, z]  # z
            theta[:, z] += -offset * theta[:, x]  # -x
        else:
            # Rotation around Z axis: [-y, x, 0]
            theta[:, x] += -offset * theta[:, y]  # -y
            theta[:, y] += offset * theta[:, x]  # x

        # Normalize to keep points on the sphere
        norms = np.sqrt(np.sum(theta**2, axis=1))
        theta /= norms[:, None]
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Creates a trajectory that traces interesting paths on the sphere."""
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Create time parameter
        t = np.linspace(0, 4 * np.pi, n_steps) + np.pi

        # Create a spiral-like trajectory on the sphere
        phi = t  # azimuthal angle
        theta = np.pi / 4 * np.sin(t / 2) + np.pi / 2  # polar angle

        # Convert from spherical to Cartesian coordinates
        trajectory[:, 0] = np.sin(theta) * np.cos(phi)  # x
        trajectory[:, 1] = np.sin(theta) * np.sin(phi)  # y
        trajectory[:, 2] = np.cos(theta)  # z

        # Normalize to ensure points are exactly on sphere
        norms = np.sqrt(np.sum(trajectory**2, axis=1))
        trajectory /= norms[:, None]

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute velocities in the tangent space of the sphere."""
        # Compute the raw difference
        delta = theta - theta_prev
        return delta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities and current position.

        Args:
            i: CAN index (0-5, two CANs per dimension)
            theta_dot: Velocities in R³ tangent to the sphere
            theta: Current position on the sphere [x, y, z]
        """
        # Determine which dimension this CAN handles (0=X, 1=Y, 2=Z)
        dim = i // 2

        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1

        # Get projection vector
        x, y, z = 0, 1, 2
        if dim == 0:  # X-axis rotation: [0, -z, y]
            psi = np.array([0, -theta[z], theta[y]])
        elif dim == 1:  # Y-axis rotation: [z, 0, -x]
            psi = np.array([theta[z], 0, -theta[x]])
        else:  # Z-axis rotation: [-y, x, 0]
            psi = np.array([-theta[y], theta[x], 0])

        # Project velocity onto the projection vector
        proj = np.dot(theta_dot, psi)
        return sign * self.beta * proj
