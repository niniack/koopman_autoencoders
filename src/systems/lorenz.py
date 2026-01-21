"""Lorenz system implementation."""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def lorenz(
    state: ArrayLike, t: float, sigma: float = 10.0, rho: float = 28.0, beta: float = 8 / 3
) -> Array:
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return jnp.array([dxdt, dydt, dzdt])
