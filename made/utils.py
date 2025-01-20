import numpy as np
from multiprocessing import Pool
from functools import partial

from .can import CAN


def _simulate_single(initial_state, can, n_steps, radius=0.5):
    # Initialize state based on the point and radius
    N = can.connectivity_matrix.shape[0]
    S = np.zeros((N, 1))

    # Calculate distances from the point to all neurons
    if len(initial_state.shape) == 1:
        initial_state = initial_state.reshape(1, -1)
    distances = can.manifold.metric(initial_state, can.neurons_coordinates)

    # Set initial state
    S[distances <= radius] = 1.0

    # Run simulation with this state
    return can.run_stateless(S, n_steps)


def simulate_many_with_initial_states(
    can: CAN, initial_states: np.ndarray, n_steps: int
):
    # Create a partial function with fixed parameters
    sim_func = partial(_simulate_single, can=can, n_steps=n_steps)

    # Run simulations in parallel
    with Pool() as pool:
        final_states = pool.map(sim_func, initial_states)

    return np.array(final_states).reshape(initial_states.shape[0], -1)
