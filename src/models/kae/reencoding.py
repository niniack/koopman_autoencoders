"""
Fathi, Mahan, et al. "Course correcting Koopman representations."
International Conference on Learning Representations. PMLR, 2024.
https://openreview.net/forum?id=A18gWgc5mi

Notes:
1. We normalize the final decoder layer with a UnitNormWrapper.
The original paper is unclear about how decoder columns are normalized
"""

import jax
import jax.numpy as jnp
from flax import nnx

from models.kae.base import KoopmanAutoencoder
from models.kae.operators import ContinuousBilinearKoopmanOperator
from models.kae.utils import UnitNormWrapper


class ReencodingAutoencoder(KoopmanAutoencoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        koopman_dim: int,
        dt: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **kwargs,
    ):
        self.dt = dt

        self.encoder = nnx.Sequential(
            nnx.Linear(input_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, koopman_dim, rngs=rngs),
        )
        self.decoder = nnx.Sequential(
            nnx.Linear(koopman_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.relu,
            UnitNormWrapper(nnx.Linear(hidden_dim, input_dim, rngs=rngs)),
        )

        self.koopman_operator = ContinuousBilinearKoopmanOperator(koopman_dim, dt=dt, rngs=rngs)

    def __call__(self, x):
        raise ValueError("Not implemented. Use forward_and_loss_function instead.")

    def forward_and_loss_function(self, window):
        """
        Window must be shaped [B, T, D].

        B: batch size
        T: time steps
        D: state dimension
        F: koopman dimension
        """
        assert len(window.shape) == 3, "Input must be 3D [B, T, D]"

        # Encode whole window.
        # Double vmap over batch and time
        # shape: [B, T, F]
        z_window = jax.vmap(jax.vmap(self.encoder))(window)

        # shape: [B, F]
        z0 = z_window[:, 0, :]

        # Decode initial step
        # shape: [B, D]
        x0_recon = self.decoder(z_window[:, 0, :])

        # Roll forward in latent space
        # shape: [B, T, F]
        z_fwd_pred = self.rollout_latent(z0, T=window.shape[1] - 1, reencode_every=30)

        # Decode forward predictions.
        # Double vmap over batch and time
        # shape: [B, T, D]
        x_fwd_pred = jax.vmap(jax.vmap(self.decoder))(z_fwd_pred)

        # Reconstruction loss (only on initial step)
        loss_recon = jnp.mean((x0_recon - window[:, 0, :]) ** 2)

        # Linearity loss
        loss_linear = jnp.mean((z_fwd_pred - z_window[:, 1:, :]) ** 2)

        # Forward loss
        loss_fwd = jnp.mean((x_fwd_pred - window[:, 1:, :]) ** 2)

        # Sparsity loss
        loss_sparse = jnp.mean(jnp.abs(z_window))

        total = loss_recon + loss_linear + loss_fwd + 1e-3 * loss_sparse
        return total, {
            "recon": loss_recon,
            "linear": loss_linear,
            "forward_pred": loss_fwd,
            "sparsity": loss_sparse,
        }
