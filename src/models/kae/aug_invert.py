"""
Frion, Anthony, et al. "Augmented Invertible Koopman Autoencoder
for long-term time series forecasting."
Transactions on Machine Learning Research, 2025.
https://github.com/edflorian/aikae

Notes:
1. No reconstruction loss needed
2. Koopman operator is discrete-time dense matrix
3. Orthogonality loss on K for long-term stability
"""

import jax
import jax.numpy as jnp
from flax import nnx

from models.kae.base import KoopmanAutoencoder
from models.kae.operators import DiscreteDenseKoopmanOperator
from models.nice import StackedNICE


class AIKAEEncoder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        augment_dim: int,
        n_flow_layers: int,
        rngs: nnx.Rngs,
    ):
        self.input_dim = input_dim

        k = input_dim // 2  # matches NICE(d, k, ...)
        self.invertible_encoder = StackedNICE(
            d=input_dim,
            k=k,
            hidden=hidden_dim,
            n_layers=n_flow_layers,
            even_odd=True,  # set according to your PyTorch use
            rngs=rngs,
        )

        self.augmentation_encoder = nnx.Sequential(
            nnx.Linear(input_dim, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 128, rngs=rngs),
            nnx.relu,
            nnx.Linear(128, augment_dim, rngs=rngs),
        )

    def __call__(self, x):
        z_inv, _ = self.invertible_encoder.forward(x)
        z_aug = self.augmentation_encoder(x)
        return jnp.concatenate([z_inv, z_aug], axis=-1)


class AIKAEDecoder(nnx.Module):
    def __init__(self, invertible_encoder: StackedNICE, input_dim: int):
        self.invertible_encoder = invertible_encoder
        self.input_dim = input_dim

    def __call__(self, z):
        z_inv = z[..., : self.input_dim]
        return self.invertible_encoder.inverse(z_inv)


class AugmentedInvertibleAutoencoder(KoopmanAutoencoder):
    """
    Augmented Invertible Koopman Autoencoder.
    Latent dim = input_dim + augment_dim
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        augment_dim: int = 16,
        n_flow_layers: int = 3,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **kwargs,
    ):
        self.input_dim = input_dim
        self.augment_dim = augment_dim
        self.koopman_dim = input_dim + augment_dim

        self.encoder = AIKAEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            augment_dim=augment_dim,
            n_flow_layers=n_flow_layers,
            rngs=rngs,
        )

        self.decoder = AIKAEDecoder(
            invertible_encoder=self.encoder.invertible_encoder,
            input_dim=input_dim,
        )

        self.koopman_operator = DiscreteDenseKoopmanOperator(
            self.koopman_dim,
            rngs=rngs,
        )

        # Initialize K as orthogonal
        orth_init = nnx.initializers.orthogonal()
        kernel = self.koopman_operator.dynamics.kernel
        kernel_shape = kernel.value.shape
        kernel.value = orth_init(rngs.params(), kernel_shape)

    def orthogonality_loss(self):
        """
        Encourages K to be orthogonal: ||K^T K - I||_F^2
        """
        K = self.koopman_operator.dynamics.kernel.value
        eye = jnp.eye(K.shape[0], dtype=K.dtype)
        return jnp.sum((K.T @ K - eye) ** 2)

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

        B, T, D = window.shape  # B, D unused but kept for clarity

        # Encode whole window
        # shape: [B, T, F]
        z_window = jax.vmap(jax.vmap(self.encoder))(window)

        # Initial latent state
        # shape: [B, F]
        z0 = z_window[:, 0, :]

        # Roll forward in latent space
        # shape: [B, T-1, F]
        z_fwd_pred = self.rollout_latent(z0, T=T - 1)

        # Decode forward predictions
        # shape: [B, T-1, D]
        x_fwd_pred = jax.vmap(jax.vmap(self.decoder))(z_fwd_pred)

        # Linearity loss
        loss_linear = jnp.mean((z_fwd_pred - z_window[:, 1:, :]) ** 2)

        # Prediction loss
        loss_pred = jnp.mean((x_fwd_pred - window[:, 1:, :]) ** 2)

        # Orthogonality loss on K
        loss_orth = self.orthogonality_loss()

        total = loss_linear + loss_pred + 1e-3 * loss_orth

        return total, {
            "linear": loss_linear,
            "forward_pred": loss_pred,
            "orthogonality": loss_orth,
        }
