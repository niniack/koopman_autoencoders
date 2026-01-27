import jax
import jax.numpy as jnp
from flax import nnx


class NICE(nnx.Module):
    """
    JAX version of your PyTorch NICE bijector.

    d: total dimension
    k: number of conditioning dims
    hidden: hidden size of MLP
    even_odd: if True, we alternate which block of k dims is used (first k vs last k)
    """

    def __init__(self, d: int, k: int, hidden: int, even_odd: bool = False, rngs: nnx.Rngs = None):
        self.d = d
        self.k = k
        self.even_odd = even_odd

        # same MLP: k -> hidden -> (d - k)
        self.net = nnx.Sequential(
            nnx.Linear(k, hidden, rngs=rngs),
            nnx.leaky_relu,
            nnx.Linear(hidden, d - k, rngs=rngs),
        )

    def _split(self, x: jnp.ndarray, flip: bool):
        """
        Two masking patterns, mirroring the PyTorch logic:

        flip = False:
          x1 = first k dims, x2 = remaining d-k dims
        flip = True:
          x1 = last k dims,  x2 = first d-k dims
        """
        if not flip:
            x1 = x[..., : self.k]  # (.., k)
            x2 = x[..., self.k :]  # (.., d - k)
        else:
            x1 = x[..., -self.k :]  # (.., k)
            x2 = x[..., : self.d - self.k]  # (.., d - k)
        return x1, x2

    def _merge(self, x1: jnp.ndarray, x2: jnp.ndarray, flip: bool):
        if not flip:
            # [x1 | x2] = [first k | last d-k]
            return jnp.concatenate([x1, x2], axis=-1)
        else:
            # [x2 | x1] = [first d-k | last k]
            return jnp.concatenate([x2, x1], axis=-1)

    def __call__(self, x: jnp.ndarray, flip: bool = False):
        """
        Forward bijector, like PyTorch __call__:
          x2' = x2 + s(x1)
        """
        x1, x2 = self._split(x, flip)
        t = self.net(x1)
        z2 = x2 + t
        z = self._merge(x1, z2, flip)

        # Volume-preserving: log|det J| = 0
        batch_shape = x.shape[:-1]
        logdet = jnp.zeros(batch_shape, dtype=x.dtype)
        return z, logdet

    def forward(self, x: jnp.ndarray, flip: bool = False):
        return self(x, flip=flip)

    def inverse(self, z: jnp.ndarray, flip: bool = False):
        """
        Inverse:
          x2 = z2 - s(z1)
        """
        z1, z2 = self._split(z, flip)
        t = self.net(z1)
        x2 = z2 - t
        x = self._merge(z1, x2, flip)
        return x


class StackedNICE(nnx.Module):
    """
    Stacked NICE flow, faithful to your PyTorch NICEFlow:

        for bijector, f in zip(self.bijectors, self.flips):
            x, log_pz = bijector(x, flip=f)
        return x, zeros_like(x[:,0])
    """

    def __init__(
        self,
        d: int,
        k: int,
        hidden: int,
        n_layers: int,
        even_odd: bool = False,
        rngs: nnx.Rngs = None,
    ):
        self.d = d
        self.k = k
        self.hidden = hidden
        self.n_layers = n_layers
        self.even_odd = even_odd

        # build bijectors
        self.bijectors = nnx.List()
        for i in range(n_layers):
            layer_rngs = rngs if rngs is None else rngs.fork()
            self.bijectors.append(NICE(d=d, k=k, hidden=hidden, even_odd=even_odd, rngs=layer_rngs))

        # flip pattern, like your self.flips
        if even_odd:
            # alternate False / True / False / True ...
            self.flips = [(i % 2 == 1) for i in range(n_layers)]
        else:
            # no flipping behavior
            self.flips = [False] * n_layers

    def forward(self, x: jnp.ndarray):
        z = x
        # logdet is always zero for each step
        for bijector, f in zip(self.bijectors, self.flips):
            z, _ = bijector.forward(z, flip=f)

        batch_shape = x.shape[:-1]
        logdet = jnp.zeros(batch_shape, dtype=x.dtype)
        return z, logdet

    def inverse(self, z: jnp.ndarray):
        x = z
        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):
            x = bijector.inverse(x, flip=f)
        return x

    def __call__(self, x: jnp.ndarray):
        return self.forward(x)
