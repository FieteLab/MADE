import numpy as np
from dataclasses import dataclass

from made.manifolds import AbstractManifold
from made import manifolds
from made.can import CAN


# ---------------------------------------------------------------------------- #
#                                      QAN                                     #
# ---------------------------------------------------------------------------- #


class QAN:
    """A Quasi-Attractor Network (QAN) implementation that uses multiple Continuous Attractor Networks (CANs)
    to track movement on manifolds.

    The QAN uses pairs of offset CANs for each dimension of the manifold. Each pair consists of
    two CANs with coordinates offset in opposite directions along a particular dimension.
    This allows the network to detect and represent movement along that dimension.
    """

    def __post_init__(self):
        """Initialize the QAN by creating pairs of offset CANs for each dimension."""
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
        """Simulate the QAN's response to a given trajectory.

        Args:
            trajectory: An array of shape (n_steps, manifold_dim) containing the target trajectory
                      points on the manifold.

        Returns:
            An array of shape (n_steps, manifold_dim) containing the decoded trajectory from
            the network's activity.
        """
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
    """QAN implementation for a 1D line manifold.

    The network uses two offset CANs to track movement along the line.
    """

    manifold: AbstractManifold = manifolds.Line()
    spacing: float = 0.075  # Spacing between neurons
    alpha: float = 3  # Sharpness of neural tuning curves
    sigma: float = 1  # Width of neural tuning curves
    offset_magnitude: float = 0.2  # Magnitude of CAN coordinate offsets
    beta: float = 1e2  # Gain for velocity inputs

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        """Apply offset to coordinates on the line.

        Args:
            theta: Points on line [x]
            dim: Dimension to offset (always 0 for line)
            direction: Direction of offset (+1/-1)
            offset_magnitude: Size of offset

        Returns:
            Offset coordinates
        """
        theta[:, dim] += direction * offset_magnitude
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a trajectory that moves back and forth along the line.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 1) containing points on the line
        """
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
        """Compute velocity as difference between current and previous position.

        Args:
            theta: Current position [x]
            theta_prev: Previous position [x]

        Returns:
            Velocity [dx/dt]
        """
        return theta - theta_prev

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocity.

        Args:
            i: CAN index (0-1)
            theta_dot: Velocity [dx/dt]
            theta: Current position [x]

        Returns:
            Input to the CAN
        """
        if i == 1:
            theta_dot = -theta_dot
        return self.beta * theta_dot


# ----------------------------------- Ring ----------------------------------- #
@dataclass
class RingQAN(QAN):
    """QAN implementation for a ring manifold (S¹).

    The network uses two offset CANs to track movement around the ring.
    Handles periodic boundary conditions at 0 and 2π.
    """

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
        """Apply offset to coordinates on the ring, wrapping around at 2π.

        Args:
            theta: Points on ring [θ]
            dim: Dimension to offset (always 0 for ring)
            direction: Direction of offset (+1/-1)
            offset_magnitude: Size of offset

        Returns:
            Offset coordinates wrapped to [0, 2π]
        """
        theta[:, dim] += direction * offset_magnitude
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a trajectory that goes around the ring multiple times.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 1) containing angles on the ring [0, 2π]
        """
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
        """Compute angular velocity accounting for periodic boundary conditions.

        Args:
            theta: Current angle [θ]
            theta_prev: Previous angle [θ]

        Returns:
            Angular velocity [dθ/dt]
        """
        delta_theta = theta - theta_prev

        # Handle periodic boundary crossings for both angles
        if delta_theta > np.pi:
            delta_theta -= 2 * np.pi
        elif delta_theta < -np.pi:
            delta_theta += 2 * np.pi

        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on angular velocity.

        Args:
            i: CAN index (0-1)
            theta_dot: Angular velocity [dθ/dt]
            theta: Current angle [θ]

        Returns:
            Input to the CAN
        """
        if i == 1:
            theta_dot = -theta_dot
        return self.beta * theta_dot


# ----------------------------------- Plane ---------------------------------- #
@dataclass
class PlaneQAN(QAN):
    """QAN implementation for a 2D plane manifold (R²).

    The network uses two pairs of offset CANs (4 total) to track movement in both
    x and y directions independently.
    """

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
        """Apply offset to coordinates on the plane.

        Args:
            theta: Points on plane [x, y]
            dim: Dimension to offset (0=x, 1=y)
            direction: Direction of offset (+1/-1)
            offset_magnitude: Size of offset

        Returns:
            Offset coordinates
        """
        theta[:, dim] += direction * offset_magnitude
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a space-filling trajectory over the plane using a modified Lissajous curve.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 2) containing points on the plane
        """
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
        """Compute velocity as the difference between current and previous position.

        Args:
            theta: Current position [x, y]
            theta_prev: Previous position [x, y]

        Returns:
            Velocity [dx/dt, dy/dt]
        """
        return theta - theta_prev

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities.

        Args:
            i: CAN index (0-3)
            theta_dot: Velocities [dx/dt, dy/dt]
            theta: Current position [x, y]

        Returns:
            Input to the CAN
        """
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
    """QAN implementation for a cylinder manifold (R × S¹).

    The network uses two pairs of offset CANs (4 total) to track movement along the height
    and around the circumference. Handles periodic boundary conditions in the angular dimension.
    """

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
        """Apply offset to coordinates on the cylinder.

        Args:
            theta: Points on cylinder [z, θ]
            dim: Dimension to offset (0=height, 1=angle)
            direction: Direction of offset (+1/-1)
            offset_magnitude: Size of offset

        Returns:
            Offset coordinates with angle wrapped to [0, 2π]
        """
        theta[:, dim] += direction * offset_magnitude

        if dim == 1:
            # wrap to [0, 2pi]
            theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a spiral trajectory that wraps around the cylinder multiple times.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 2) containing points on the cylinder [z, θ]
        """
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
        """Compute velocities accounting for periodic boundary in angular dimension.

        Args:
            theta: Current position [z, θ]
            theta_prev: Previous position [z, θ]

        Returns:
            Velocities [dz/dt, dθ/dt]
        """
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
        """Compute input for each CAN based on velocities.

        Args:
            i: CAN index (0-3)
            theta_dot: Velocities [dz/dt, dθ/dt]
            theta: Current position [z, θ]

        Returns:
            Input to the CAN
        """
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1
        return sign * self.beta * theta_dot[dim]


# -------------------------------- Mobius band ------------------------------- #
@dataclass
class MobiusBandQAN(QAN):
    """QAN implementation for a Möbius band manifold.

    The network uses two pairs of offset CANs (4 total) to track movement along the height
    and around the band. Handles both periodic boundary conditions and the characteristic
    height flip when completing a full rotation.
    """

    manifold: AbstractManifold = manifolds.MobiusBand()
    spacing: float = 0.15
    alpha: float = 2
    sigma: float = 2
    offset_magnitude: float = 0.25
    beta: float = 1.25e2  # control gain for velocity input

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        """Apply offset to coordinates on Mobius band.

        Args:
            theta: Points on band [height, angle]
            dim: Dimension to offset (0=height, 1=angle)
            direction: Direction of offset (+1/-1)
            offset_magnitude: Size of offset
        """
        theta = theta.copy()  # Make a copy to avoid modifying the original
        if dim == 0:  # Height dimension
            # Simple offset for height
            theta[:, dim] += direction * offset_magnitude
        else:  # Angular dimension
            # For angular dimension, when we cross boundary we flip height
            theta[:, dim] += direction * offset_magnitude
            mask_over = theta[:, dim] >= 2 * np.pi
            mask_under = theta[:, dim] < 0

            # Handle wrapping and height flipping
            if np.any(mask_over):
                theta[mask_over, dim] -= 2 * np.pi
                theta[mask_over, 0] = -theta[mask_over, 0]  # flip height

            if np.any(mask_under):
                theta[mask_under, dim] += 2 * np.pi
                theta[mask_under, 0] = -theta[mask_under, 0]  # flip height

        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a trajectory that demonstrates the Mobius band's characteristic flip.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 2) containing points on the Mobius band [height, angle]
        """
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Get parameter space bounds for height with padding
        padding = 0.2
        h_min = self.manifold.parameter_space.ranges[0].start + padding
        h_max = self.manifold.parameter_space.ranges[0].end - padding

        # Create time parameter that goes around multiple times
        t = np.linspace(np.pi / 2, 4 * np.pi, n_steps)  # 2 full rotations

        # Height varies sinusoidally
        trajectory[:, 0] = (h_max - h_min) * np.sin(t / 2) * 0.9 + h_min
        # Angle increases linearly
        trajectory[:, 1] = t % (2 * np.pi)

        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Compute velocities accounting for periodic boundary and height flip.

        Args:
            theta: Current position [height, angle]
            theta_prev: Previous position [height, angle]

        Returns:
            Velocities [dh/dt, dθ/dt]
        """
        delta_theta = theta - theta_prev

        # Handle periodic boundary crossing for angular dimension
        if delta_theta[1] > np.pi:
            delta_theta[1] -= 2 * np.pi
            delta_theta[0] = -delta_theta[0]  # flip height velocity
        elif delta_theta[1] < -np.pi:
            delta_theta[1] += 2 * np.pi
            delta_theta[0] = -delta_theta[0]  # flip height velocity

        return delta_theta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities.

        Args:
            i: CAN index (0-3)
            theta_dot: Velocities [dh/dt, dθ/dt]
            theta: Current position [height, angle]

        Returns:
            Input to the CAN
        """
        # Determine which dimension this CAN handles
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1
        return sign * self.beta * theta_dot[dim]


# ---------------------------------- Sphere ---------------------------------- #
@dataclass
class SphereQAN(QAN):
    """QAN implementation for a sphere manifold (S²).

    The network uses three pairs of offset CANs (6 total) to track rotational movement
    around the X, Y, and Z axes using Killing vector fields. Movement is represented
    in the tangent space of the sphere.
    """

    manifold: AbstractManifold = manifolds.Sphere()
    spacing: float = 0.075
    alpha: float = 2.5
    sigma: float = 2.5
    offset_magnitude: float = 0.05
    beta: float = 9e2

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        """Apply rotational offset to coordinates on sphere using Killing vector fields.

        Args:
            theta: Points on sphere [x, y, z]
            dim: Dimension to rotate around (0=X, 1=Y, 2=Z)
            direction: Direction of rotation (+1/-1)
            offset_magnitude: Size of rotation
        """
        theta = theta.copy()

        # Initialize rotation vector with same shape as input
        v = np.zeros_like(theta)

        # Get rotation vector based on axis
        if dim == 0:  # Rotation around X: [0, -z, y]
            v[:, 0] = 0
            v[:, 1] = -theta[:, 2]
            v[:, 2] = theta[:, 1]
        elif dim == 1:  # Rotation around Y: [z, 0, -x]
            v[:, 0] = theta[:, 2]
            v[:, 1] = 0
            v[:, 2] = -theta[:, 0]
        else:  # Rotation around Z: [-y, x, 0]
            v[:, 0] = -theta[:, 1]
            v[:, 1] = theta[:, 0]
            v[:, 2] = 0

        # Apply rotation
        theta += direction * offset_magnitude * v

        # Normalize to keep points on sphere
        norms = np.sqrt(np.sum(theta**2, axis=1))
        theta /= norms[:, None]

        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Create a spiral trajectory from south pole to north pole on the sphere.

        Args:
            n_steps: Number of timesteps in the trajectory

        Returns:
            Array of shape (n_steps, 3) containing points on the unit sphere [x, y, z]
        """
        trajectory = np.zeros((n_steps, self.manifold.dim))

        # Create parameters for spiral
        # theta goes from pi (south pole) to 0 (north pole)
        theta = np.linspace(np.pi, 0, n_steps)
        # phi wraps around multiple times as we go up
        phi = np.linspace(0, 4 * np.pi, n_steps)  # 3 full rotations

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
        """Compute velocity vector in R³ tangent to the sphere.

        Args:
            theta: Current position [x, y, z]
            theta_prev: Previous position [x, y, z]

        Returns:
            Velocity vector tangent to the sphere at theta
        """
        # Get raw difference
        delta = theta - theta_prev

        # Project onto tangent space at theta
        # The tangent space projection is: v - <v,n>n where n is the normal (which is theta for unit sphere)
        normal_component = np.dot(delta, theta) * theta
        tangent_delta = delta - normal_component

        return tangent_delta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Compute input for each CAN based on velocities using Killing fields.

        Args:
            i: CAN index (0-5)
            theta_dot: Velocity vector tangent to sphere at theta
            theta: Current position [x, y, z]

        Returns:
            Input to the CAN based on projection onto appropriate Killing field
        """
        # Determine which dimension this CAN handles (0=X, 1=Y, 2=Z)
        dim = i // 2
        # If even index, use positive direction, if odd use negative
        sign = 1 if i % 2 == 0 else -1

        # Get Killing field vector based on rotation axis
        if dim == 0:  # X-axis rotation: [0, -z, y]
            v = np.array([0, -theta[2], theta[1]])
        elif dim == 1:  # Y-axis rotation: [z, 0, -x]
            v = np.array([theta[2], 0, -theta[0]])
        else:  # Z-axis rotation: [-y, x, 0]
            v = np.array([-theta[1], theta[0], 0])

        # Normalize the Killing field vector
        v = v / np.linalg.norm(v) if np.linalg.norm(v) > 0 else v

        # Project velocity onto the normalized Killing field
        proj = np.dot(theta_dot, v)

        return sign * self.beta * proj
