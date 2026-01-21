import jax
import jax.numpy as jnp
from flax import nnx


class VanillaKoopmanOperator(nnx.Module):
    def __init__(self, koopman_dim: int, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.dynamics = nnx.Linear(koopman_dim, koopman_dim, use_bias=False, rngs=rngs)

    def __call__(self, x, T):
        def step(z, _):
            z_next = self.dynamics(z)
            return z_next, z_next

        _, preds = jax.lax.scan(f=step, init=x, xs=None, length=T)
        return jnp.transpose(preds, (1, 0, 2))
