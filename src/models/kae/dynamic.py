"""
Based on:
https://www.nature.com/articles/s41467-018-07210-0
https://github.com/BethanyL/DeepKoopman/tree/master?tab=readme-ov-file
"""

import jax
import jax.numpy as jnp
from flax import nnx


class DynamicKoopmanOperator(nnx.Module):
    def __init__(self, koopman_dim: int, dt: float, rngs: nnx.Rngs = nnx.Rngs(0), **kwargs):
        self.num_eigenvalues = int(koopman_dim / 2)
        self.dt = dt

        self.parameterization = nnx.Sequential(
            nnx.Linear(koopman_dim, self.num_eigenvalues * 2, rngs=rngs),
            nnx.tanh,
            nnx.Linear(self.num_eigenvalues * 2, self.num_eigenvalues * 2, rngs=rngs),
        )

    def __call__(self, x, T):
        # Get eigenvalues from network
        batch_size, num_features = x.shape
        eigs_param = self.parameterization(x)
        eigs_param = eigs_param.reshape((batch_size, self.num_eigenvalues, 2))

        # Separate real and imaginary parts
        mu = eigs_param[..., 0]
        omega = eigs_param[..., 1]

        # Compute discrete-time eigenvalues
        exp_mu = jnp.exp(mu * self.dt)
        cos_omega = jnp.cos(omega * self.dt)
        sin_omega = jnp.sin(omega * self.dt)

        # Build 2x2 blocks: (batch, num_eig, 2, 2)
        blocks = jnp.stack(
            [
                jnp.stack([exp_mu * cos_omega, -exp_mu * sin_omega], axis=-1),
                jnp.stack([exp_mu * sin_omega, exp_mu * cos_omega], axis=-1),
            ],
            axis=-2,
        )

        # Apply blocks directly without building full matrix
        def apply_block_diag(z, blocks):
            # z: (koopman_dim,) -> reshape to (num_eig, 2)
            z_pairs = z.reshape(-1, 2)
            # blocks: (num_eig, 2, 2)
            z_next = jnp.einsum("nij,nj->ni", blocks, z_pairs)
            return z_next.flatten()

        def step(z, _):
            z_next = jax.vmap(apply_block_diag)(z, blocks)
            return z_next, z_next

        _, preds = jax.lax.scan(f=step, init=x, xs=None, length=T)
        return jnp.transpose(preds, (1, 0, 2))


class DynamicAutoencoder(nnx.Module):
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

        self.koopman_operator = DynamicKoopmanOperator(koopman_dim, dt=dt, rngs=rngs)

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

        # shape: [B, D]
        initial_step = window[:, 0, :]

        # shape: [B, T, D]
        targets = window[:, 1:, :]

        # Encode and decode initial step
        # shape: [B, F]
        z0 = self.encoder(initial_step)
        # shape: [B, D]
        x0_recon = self.decoder(z0)

        # Roll forward in latent space
        # shape: [B, T, F]
        z_pred = self.koopman_operator(z0, T=targets.shape[1])

        # Decode predicted observables with vmap
        # shape: [B, T, D]
        x_pred = jax.vmap(jax.vmap(self.decoder))(z_pred)

        # Encode targets into observable space
        # shape: [B, T, F]
        z_true = jax.vmap(jax.vmap(self.encoder))(targets)

        # Reconstruction loss
        loss_recon = jnp.mean((x0_recon - initial_step) ** 2)

        # Linearity loss
        loss_linear = jnp.mean((z_pred - z_true) ** 2)

        # Prediction loss
        loss_pred = jnp.mean((x_pred - targets) ** 2)

        total = loss_recon + loss_linear + loss_pred
        return total, {
            "recon": loss_recon,
            "linear": loss_linear,
            "forward_pred": loss_pred,
        }
