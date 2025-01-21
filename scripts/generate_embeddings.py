import numpy as np
from loguru import logger
from sklearn.manifold import Isomap

from made.can import CAN
from made.utils import simulate_many_with_initial_states
from made.manifolds import PADS


N_SAMPLES = dict(
    Plane=100,
    Cylinder=100,
    Torus=100,
    Sphere=1000,
    MobiusBand=100,
)

TOPOLOGIES = ["Plane", "Cylinder", "Torus", "Sphere", "MobiusBand"]


def main():
    for topology in TOPOLOGIES:
        logger.info(f"Generating embeddings for {topology}")
        can = CAN.default(topology=topology)
        samples = can.manifold.parameter_space.sample(
            N_SAMPLES[topology], pads=PADS[topology]
        )
        if can.manifold.dim > 1:
            samples = samples[::3, :]

        final_states = simulate_many_with_initial_states(can, samples, 25)
        logger.info(f"Simulated {final_states.shape[0]} samples")

        isomap = Isomap(n_components=3, n_neighbors=25)
        final_states_isomap = isomap.fit_transform(final_states)

        # save as numpy in ./imgs
        logger.info(f"Saving embeddings to ./imgs/{topology}_isomap.npy")
        np.save(f"./imgs/{topology}_isomap.npy", final_states_isomap)


if __name__ == "__main__":
    main()
