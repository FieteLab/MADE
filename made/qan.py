import numpy as np

from made.manifolds import Plane
from made.can import CAN


class PlaneQAN:
    def __init__(
        self,
        spacing: float = 0.075,
        alpha: float = 3,
        sigma: float = 1,
        offset_magnitude: float = 0.1,
    ):
        self.offset_magnitude = offset_magnitude

        # crate four copies of the plane with offset coordinates
        mfld = Plane()
        om = offset_magnitude
        self.cans = [
            CAN(
                mfld,
                spacing,
                alpha,
                sigma,
                weights_offset=lambda x: self.coordinates_offset(x, 0, 1, om),
            ),
            CAN(
                mfld,
                spacing,
                alpha,
                sigma,
                weights_offset=lambda x: self.coordinates_offset(x, 0, -1, om),
            ),
            CAN(
                mfld,
                spacing,
                alpha,
                sigma,
                weights_offset=lambda x: self.coordinates_offset(x, 1, 1, om),
            ),
            CAN(
                mfld,
                spacing,
                alpha,
                sigma,
                weights_offset=lambda x: self.coordinates_offset(x, 1, -1, om),
            ),
        ]

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        theta[:, dim] += direction * offset_magnitude
        return theta
