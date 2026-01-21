"""
Just a vanilla autoencoder
"""

import jax
import jax.numpy as jnp
from flax import nnx

from models.kae.common import VanillaKoopmanOperator


class VanillaAutoencoder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        koopman_dim: int,
        dt: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **kwargs,
    ):
        self.encoder = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, koopman_dim, rngs=rngs),
        )
        self.decoder = nnx.Sequential(
            nnx.Linear(koopman_dim, hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.tanh,
            nnx.Linear(hidden_dim, input_dim, rngs=rngs),
        )

        self.koopman_operator = VanillaKoopmanOperator(koopman_dim, rngs=rngs)

        # TODO: how to do initialization?

    def __call__(self, x):
        raise ValueError("Not implemented. Use forward_and_loss_function instead.")

    def forward_and_loss_function(self, window):
        """
        B: batch size
        T: time steps
        D: state dimension
        F: koopman dimension

        Window should be shape [B, T, D]
        """
        # Encode whole window
        # shape: [B, T, F]
        z_window = jax.vmap(jax.vmap(self.encoder))(window)
        z0 = z_window[:, 0, :]

        # Decode initial step
        # shape: [B, D]
        x0_recon = self.decoder(z_window[:, 0, :])

        # Roll forward and backward in latent space
        # shape: [B, T, F]
        z_fwd_pred = self.koopman_operator(z0, T=window.shape[1] - 1)

        # Decode forward and backward predictions
        # shape: [B, T, D]
        x_fwd_pred = jax.vmap(jax.vmap(self.decoder))(z_fwd_pred)

        # Reconstruction loss
        loss_recon = jnp.mean((x0_recon - window[:, 0, :]) ** 2)

        # Linearity loss
        loss_linear = jnp.mean((z_fwd_pred - z_window[:, 1:, :]) ** 2)

        # Forward loss
        loss_fwd = jnp.mean((x_fwd_pred - window[:, 1:, :]) ** 2)

        total = loss_recon + loss_linear + loss_fwd
        return total, {
            "recon": loss_recon,
            "linear": loss_linear,
            "forward_pred": loss_fwd,
        }
