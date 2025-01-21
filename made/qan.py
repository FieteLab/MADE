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

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        return theta


# ----------------------------------- Ring ----------------------------------- #
@dataclass
class RingQAN(QAN):
    manifold: AbstractManifold = manifolds.Ring()
    spacing: float = 0.075
    alpha: float = 3
    sigma: float = 1
    offset_magnitude: float = 0.1

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta


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
            theta[:, x] += -offset * theta[:, z]  # -x
            theta[:, z] += offset * theta[:, x]  # z
        else:
            # Rotation around Z axis: [-y, x, 0]
            theta[:, y] += -offset * theta[:, x]  # -y
            theta[:, x] += offset * theta[:, y]  # x

        # Normalize to keep points on the sphere
        norms = np.sqrt(np.sum(theta**2, axis=1))
        theta /= norms[:, None]
        return theta
