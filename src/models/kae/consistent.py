"""
Azencot, Omri, et al. "Forecasting sequential data using consistent Koopman autoencoders."
International Conference on Machine Learning. PMLR, 2020.

https://github.com/erichson/koopmanAE

A Koopman autoencoder with two linear operators (forward and backward). The encoder and decoder
are feedforward neural networks with tanh activations. The model is trained with a reconstruction loss,
a forward loss, a backward loss, and a consistency loss which encourages the two linear
operators to be inverses of each other.

Notes:
1. We do not use tanh on the output of the decoder, as in the original implementation.
2. We do not scale reconstruction loss with `steps`, as in the original implementation.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Just a linear layer
from models.kae.common import VanillaKoopmanOperator


class ConsistentAutoencoder(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        koopman_dim: int,
        dt: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
        lambda_bwd: float = 0.1,
        lambda_consistency: float = 0.01,
        init_scale: float = 1,
        **kwargs,
    ):
        self.lambda_bwd = lambda_bwd
        self.lambda_consistency = lambda_consistency

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
        self.bwd_koopman_operator = VanillaKoopmanOperator(koopman_dim, rngs=rngs)

        # Initialize forward operator as orthogonal
        orth_init = nnx.initializers.orthogonal(scale=init_scale)
        self.koopman_operator.dynamics.kernel.value = orth_init(rngs[0], (koopman_dim, koopman_dim))

        # Initialize backward operator as pseudo-inverse of forward
        self.bwd_koopman_operator.dynamics.kernel.value = jnp.linalg.pinv(
            self.koopman_operator.dynamics.kernel.value
        )

    def __call__(self, x):
        raise ValueError("Not implemented. Use forward_and_loss_function instead.")

    def consistency_loss(self):
        # We transpose to match Pytorch convention
        F = self.koopman_operator.dynamics.kernel.T
        B = self.bwd_koopman_operator.dynamics.kernel.T

        assert B.shape == F.shape, "Forward and backward operators should match."
        koopman_dim = B.shape[1]

        total_loss = 0.0
        for k in range(1, koopman_dim + 1):
            term1 = F[:k, :] @ B[:, :k] - jnp.eye(k)
            term2 = B[:k, :] @ F[:, :k] - jnp.eye(k)
            total_loss += (jnp.sum(term1**2) + jnp.sum(term2**2)) / (2 * k)

        return total_loss

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
        z0 = z_window[:, 0, :]
        zT = z_window[:, -1, :]

        # Decode initial step
        # shape: [B, D]
        x0_recon = self.decoder(z_window[:, 0, :])

        # Roll forward and backward in latent space
        # shape: [B, T, F]
        z_fwd_pred = self.koopman_operator(z0, T=window.shape[1] - 1)
        z_bwd_pred = self.bwd_koopman_operator(zT, T=window.shape[1] - 1)

        # Decode forward and backward predictions.
        # Double vmap over batch and time
        # shape: [B, T, D]
        x_fwd_pred = jax.vmap(jax.vmap(self.decoder))(z_fwd_pred)
        x_bwd_pred = jax.vmap(jax.vmap(self.decoder))(z_bwd_pred)

        # Reconstruction loss (only on initial step)
        loss_recon = jnp.mean((x0_recon - window[:, 0, :]) ** 2)

        # Forward loss
        loss_fwd = jnp.mean((x_fwd_pred - window[:, 1:, :]) ** 2)

        # Backward loss
        flipped_window = jnp.flip(window, axis=1)
        loss_bwd = jnp.mean((x_bwd_pred - flipped_window[:, 1:, :]) ** 2)

        # Consistency loss
        loss_consistency = self.consistency_loss()

        total = (
            loss_recon
            + loss_fwd
            + self.lambda_bwd * loss_bwd
            + self.lambda_consistency * loss_consistency
        )
        return total, {
            "recon": loss_recon,
            "forward_pred": loss_fwd,
            "backward_pred": loss_bwd,
            "consistency": loss_consistency,
        }
