# test_koopman_operators.py
import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from models.kae.operators import (  # adjust import path
    ContinuousBilinearKoopmanOperator,
    ContinuousExponentialKoopmanOperator,
    ContinuousIntegratedKoopmanOperator,
    DiscreteDenseKoopmanOperator,
)


@pytest.mark.parametrize("koopman_dim", [16, 32])
@pytest.mark.parametrize("steps", [5, 10, 50])
def test_operators(koopman_dim, steps):
    key = jax.random.key(1)
    batch_size = 16
    dt = 1e-3

    key_A, key_z0 = jax.random.split(key)
    A = jax.random.normal(key_A, (koopman_dim, koopman_dim)) / koopman_dim
    z0 = jax.random.normal(key_z0, (batch_size, koopman_dim))

    disc_op = DiscreteDenseKoopmanOperator(koopman_dim=koopman_dim, dt=None, rngs=nnx.Rngs(0))
    exp_op = ContinuousExponentialKoopmanOperator(koopman_dim=koopman_dim, dt=dt, rngs=nnx.Rngs(1))
    bil_op = ContinuousBilinearKoopmanOperator(koopman_dim=koopman_dim, dt=dt, rngs=nnx.Rngs(2))
    int_op = ContinuousIntegratedKoopmanOperator(koopman_dim=koopman_dim, dt=dt, rngs=nnx.Rngs(3))

    # Same generator A
    disc_op.dynamics.kernel = jax.scipy.linalg.expm(A * dt)
    bil_op.dynamics.kernel = A
    exp_op.dynamics.kernel = A
    int_op.dynamics.kernel = A

    # Compare multi-step trajectories; for small dt the two
    # discretizations should be very close
    # shape: [B, T, F]
    z_exp = exp_op(z0, steps)
    z_bil = bil_op(z0, steps)
    z_int = int_op(z0, steps)
    z_disc = disc_op(z0, steps)

    atol, rtol = 1e-3, 1e-3
    assert jnp.allclose(z_disc, z_exp, atol=atol, rtol=rtol)
    assert jnp.allclose(z_bil, z_exp, atol=atol, rtol=rtol)
    assert jnp.allclose(z_int, z_exp, atol=atol, rtol=rtol)
