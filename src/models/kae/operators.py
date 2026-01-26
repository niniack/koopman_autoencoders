import jax
import jax.numpy as jnp
from flax import nnx
from jax.experimental.ode import odeint


class ContinuousExponentialKoopmanOperator(nnx.Module):
    """
    Continuous-time operator stepped via matrix exponential.

    z_1 = expm(K * dt) @ z_0
    """

    def __init__(self, koopman_dim: int, dt: float, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.dynamics = nnx.Linear(koopman_dim, koopman_dim, use_bias=False, rngs=rngs)
        self.dt = dt

    def __call__(self, z0, T):
        K_discrete = jax.scipy.linalg.expm(self.dynamics.kernel * self.dt)

        def step(z, _):
            z_next = z @ K_discrete
            return z_next, z_next

        # shape: [T, B, F]
        _, preds = jax.lax.scan(f=step, init=z0, xs=None, length=T)

        # shape: [B, T, F]
        return jnp.transpose(preds, (1, 0, 2))


class ContinuousBilinearKoopmanOperator(nnx.Module):
    """
    Continuous-time operator stepped via bilinear / Tustin discretization.

    z_1 = (I - 0.5*dt*K)^(-1) @ (I + 0.5*dt*K) @ z_0
    """

    def __init__(self, koopman_dim: int, dt: float, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.dynamics = nnx.Linear(koopman_dim, koopman_dim, use_bias=False, rngs=rngs)
        self.log_dt = nnx.Param(jnp.log(dt))

    @property
    def dt(self):
        return jnp.exp(self.log_dt)

    def __call__(self, z0, T):
        dt = self.dt
        eye = jnp.eye(self.dynamics.kernel.shape[0], dtype=self.dynamics.kernel.dtype)
        kernel = self.dynamics.kernel

        # Bilinear / Tustin discretization
        K_discrete = jnp.linalg.solve(
            eye - 0.5 * dt * kernel,
            eye + 0.5 * dt * kernel,
        )

        def step(z, _):
            z_next = z @ K_discrete
            return z_next, z_next

        # shape: [T, B, F]
        _, preds = jax.lax.scan(f=step, init=z0, xs=None, length=T)

        # shape: [B, T, F]
        return jnp.transpose(preds, (1, 0, 2))


class ContinuousIntegratedKoopmanOperator(nnx.Module):
    """
    Continuous-time operator stepped via ODE integration.

    z_1 = odeint(K(z), t)
    """

    def __init__(self, koopman_dim: int, dt: float, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.dynamics = nnx.Linear(koopman_dim, koopman_dim, use_bias=False, rngs=rngs)
        self.dt = dt

    # NOTE: Linear dynamics have a closed form solution!
    # This is implemented via `odeint`, following Fathi [2023],
    # but scan with matrix exponential should be more efficient.
    def __call__(self, x, T):
        def dynamics_fn(z, t):
            return self.dynamics(z)

        time_points = self.dt * jnp.arange(1, T + 1)

        # shape: [T, B, F]
        preds = odeint(func=dynamics_fn, y0=x, t=time_points)

        # Exclude initial state
        # shape: [B, T, F]
        return jnp.transpose(preds, (1, 0, 2))


class DiscreteDenseKoopmanOperator(nnx.Module):
    """
    Discrete-time operator with dense dynamics.

    z_1 = K @ z_0
    """

    def __init__(self, koopman_dim: int, dt=None, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.dynamics = nnx.Linear(koopman_dim, koopman_dim, use_bias=False, rngs=rngs)

    def __call__(self, z0, T):
        def step(z, _):
            z_next = self.dynamics(z)
            return z_next, z_next

        # shape: [T, B, F]
        _, preds = jax.lax.scan(f=step, init=z0, xs=None, length=T)

        # shape: [B, T, F]
        return jnp.transpose(preds, (1, 0, 2))
