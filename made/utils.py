import numpy as np
from multiprocessing import Pool
from functools import partial
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from .can import CAN


def _simulate_single(initial_state, can, n_steps):
    # Initialize state based on the point and radius
    N = can.connectivity_matrix.shape[0]
    S = np.zeros((N, 1))

    # Calculate distances from the point to all neurons
    if len(initial_state.shape) == 1:
        initial_state = initial_state.reshape(1, -1)
    distances = can.manifold.metric(can.neurons_coordinates, initial_state)

    # Set initial state
    radius = np.max(distances) * 0.2
    S[distances <= radius] = 1.0

    # Run simulation with this state
    return can.run_stateless(S, n_steps)


def simulate_many_with_initial_states(
    can: CAN, initial_states: np.ndarray, n_steps: int
):
    # Create a partial function with fixed parameters
    sim_func = partial(_simulate_single, can=can, n_steps=n_steps)

    # Run simulations in parallel with progress bar
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task(
            "[cyan]Running simulations...", total=len(initial_states)
        )
        with Pool() as pool:
            final_states = []
            for result in pool.imap(sim_func, initial_states):
                final_states.append(result)
                progress.advance(task)

    return np.array(final_states).reshape(initial_states.shape[0], -1)
