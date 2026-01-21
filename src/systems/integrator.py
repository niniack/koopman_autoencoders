from typing import Callable

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def euler_step(func: Callable, state: ArrayLike, t: float, dt: float, *args):
    return state + dt * func(state, t, *args)


def rk4_step(func: Callable, state: ArrayLike, t: float, dt: float, *args):
    k1 = func(state, t, *args)
    k2 = func(state + 0.5 * dt * k1, t + 0.5 * dt, *args)
    k3 = func(state + 0.5 * dt * k2, t + 0.5 * dt, *args)
    k4 = func(state + dt * k3, t + dt, *args)

    return state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(
    integrator: Callable,
    func: Callable,
    num_steps: int,
    dt: float = 1e-1,
    *args,
):
    time_steps = jnp.arange(0, num_steps * dt, dt)

    def step(state, t):
        next_state = integrator(func, state, t, dt, *args)
        return next_state, next_state

    def simulator(initial_state):
        _, traj = jax.lax.scan(step, initial_state, time_steps)
        return traj

    return simulator
