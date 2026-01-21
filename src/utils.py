from dataclasses import dataclass
from enum import Enum

import jax
import jax.numpy as jnp

from systems.integrator import rk4_step, simulate
from systems.lorenz import lorenz


@dataclass
class SystemConfig:
    dim: int
    default_dt: float
    default_init: tuple  # or a callable


SYSTEMS = {
    lorenz: SystemConfig(dim=3, default_dt=0.01, default_init=(0.0, 1.0, 1.05)),
}


def prepare_data(
    system=lorenz,
    num_steps: int = 1000,
    num_trajectories: int = 100,
    window_size: int = 10,
    shuffle=True,
    output_rollouts=False,
    rngs=jax.random.PRNGKey(0),
):
    """
    B: batch_size or num_trajectories
    T: num_steps
    D: state dimension
    """
    B, T, D = num_trajectories, num_steps, SYSTEMS[system].dim
    dt = SYSTEMS[system].default_dt
    default_init = jnp.array(SYSTEMS[system].default_init)

    # Init and rollout
    # shape: [B, T, D]
    init_states = default_init + jax.random.normal(rngs, (num_trajectories, D))
    rollouts = jax.vmap(simulate(rk4_step, system, num_steps=T, dt=dt))(init_states)

    # Add initial state
    # shape: [B, T + 1, D]
    rollouts = jnp.concatenate([init_states[:, None, :], rollouts], axis=1)

    # Create windows
    # basic conv formula: width - kernel_size + 1
    # shape: [B, T - window_size + 1, window_size, D]
    def _hankelize(x, num_steps, window_size):
        """
        Build base, [0, window_size - 1]
        Build offset, then broadcast base and sum to get indices
        """
        base = jnp.arange(window_size)
        offset = jnp.arange(num_steps - window_size + 1)[:, None]
        indices = base + offset
        return x[indices]

    batch_hankel = jax.vmap(lambda x: _hankelize(x, T, window_size))(rollouts)

    # shape: [B * (T - window_size + 1), window_size, D]
    batch_hankel = batch_hankel.reshape(-1, window_size, D)

    if shuffle:
        perm = jax.random.permutation(rngs, batch_hankel.shape[0])
        batch_hankel = batch_hankel[perm]

    if output_rollouts:
        return rollouts, batch_hankel
    else:
        return batch_hankel
