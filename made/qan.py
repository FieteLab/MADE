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
            can.reset(mode="point", point=theta_0)
            states.append(can.S)

        # 2. run simulation
        S = self.cans[0].S.copy()

        decoded_trajectory = []
        for t, theta in enumerate(trajectory):
            if t == 0:
                continue
            # compute variable velocity
            theta_dot = (
                self.compute_theta_dot(
                    theta.copy(), trajectory[t - 1, :].copy()
                )
                * dT
            )

            # update each CAN
            for i, can in enumerate(self.cans):
                can_input = self.compute_can_input(i, theta_dot)
                states[i] = can.step_stateless(S, can_input)

            # get the current state as the sum of all CAN states
            S = np.sum(states, axis=0)

            # decode the state into a trajectory
            decoded_trajectory.append(self.decode_state(S))
        out = np.array(decoded_trajectory)
        if len(out.shape) == 1:
            out = out.reshape(-1, 1)
        return out

    def decode_state(self, S: np.ndarray) -> np.ndarray:
        max_idx = np.argmax(S)
        return self.cans[0].idx2coord(max_idx, 0)


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
    offset_magnitude: float = 0.5
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

    def compute_can_input(self, i: int, theta_dot: np.ndarray) -> np.ndarray:
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
    beta: float = 4e2

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

    def compute_can_input(self, i: int, theta_dot: np.ndarray) -> np.ndarray:
        if i == 1:
            theta_dot = -theta_dot
        return self.beta * theta_dot


# ----------------------------------- Plane ---------------------------------- #
@dataclass
class PlaneQAN(QAN):
    manifold: AbstractManifold = manifolds.Plane()
    spacing: float = 0.075
    alpha: float = 3
    sigma: float = 1
    offset_magnitude: float = 0.1

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        return theta


# ----------------------------------- Torus ---------------------------------- #
@dataclass
class TorusQAN(QAN):
    manifold: AbstractManifold = manifolds.Torus()
    spacing: float = 0.2
    alpha: float = 2.5
    sigma: float = 2
    offset_magnitude: float = 0.1

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude

        # wrap to [0, 2pi]
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta


# ----------------------------------- Cylinder ---------------------------------- #
@dataclass
class CylinderQAN(QAN):
    manifold: AbstractManifold = manifolds.Cylinder()
    spacing: float = 0.1
    alpha: float = 2
    sigma: float = 1
    offset_magnitude: float = 0.1

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude

        if dim == 1:
            # wrap to [0, 2pi]
            theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta


# -------------------------------- Mobius band ------------------------------- #
@dataclass
class MobiusBandQAN(QAN):
    manifold: AbstractManifold = manifolds.MobiusBand()
    spacing: float = 0.1
    alpha: float = 2
    sigma: float = 2
    offset_magnitude: float = 0.1

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


# ---------------------------------- Sphere ---------------------------------- #
@dataclass
class SphereQAN(QAN):
    manifold: AbstractManifold = manifolds.Sphere()
    spacing: float = 0.075
    alpha: float = 2
    sigma: float = 3
    offset_magnitude: float = 0.1

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
            theta[:, x] += offset * theta[:, z]  # -x
            theta[:, z] += -offset * theta[:, x]  # z
        else:
            # Rotation around Z axis: [-y, x, 0]
            theta[:, x] += -offset * theta[:, y]  # x
            theta[:, y] += offset * theta[:, x]  # -y

        # Normalize to keep points on the sphere
        norms = np.sqrt(np.sum(theta**2, axis=1))
        theta /= norms[:, None]
        return theta
