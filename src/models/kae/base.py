import jax
import jax.numpy as jnp
from flax import nnx


class KoopmanAutoencoder(nnx.Module):
    encoder: nnx.Module
    decoder: nnx.Module
    koopman_operator: nnx.Module

    def rollout_latent(self, z0: jnp.ndarray, T: int, reencode_every: int | None = None):
        if reencode_every is None or reencode_every >= T:
            return self.koopman_operator(z0, T=T)

        k = reencode_every
        z_chunks = []
        z = z0
        remaining = T

        while remaining > 0:
            steps = min(k, remaining)
            z_chunk = self.koopman_operator(z, T=steps)
            z_chunks.append(z_chunk)

            if remaining > k:
                x_last = self.decoder(z_chunk[:, -1, :])
                z = self.encoder(x_last)

            remaining -= steps

        return jnp.concatenate(z_chunks, axis=1)
