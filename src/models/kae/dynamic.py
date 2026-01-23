"""
Based on:
https://www.nature.com/articles/s41467-018-07210-0
https://github.com/BethanyL/DeepKoopman/tree/master?tab=readme-ov-file
"""

import jax
import jax.numpy as jnp
from flax import nnx


class DynamicKoopmanOperator(nnx.Module):
    def __init__(
        self,
        num_real: int,
        num_pair_complex: int,
        dt: float,
        rngs: nnx.Rngs = nnx.Rngs(0),
        **kwargs,
    ):
        self.dt = dt
        self.num_real = num_real
        self.num_pair_complex = num_pair_complex

        # See discussion here: https://github.com/google/flax/discussions/4048
        # https://github.com/google/flax/discussions/4804

        def make_omega_ensemble(num_nets, out_dim):
            @nnx.split_rngs(splits=num_nets)
            @nnx.vmap
            def make(rngs):
                return nnx.Sequential(
                    nnx.Linear(1, 10, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(10, 10, rngs=rngs),
                    nnx.relu,
                    nnx.Linear(10, out_dim, rngs=rngs),
                )

            return make

        if num_pair_complex > 0:
            self.param_net_complex = make_omega_ensemble(num_nets=num_pair_complex, out_dim=2)(rngs)

        if num_real > 0:
            self.param_net_real = make_omega_ensemble(num_nets=num_real, out_dim=1)(rngs)

    def __call__(self, z0, T):
        """
        This forward function takes advantage of `nnx.scan` and avoids a `for` loop.
        """

        # `z` is carried, don't scan over ensembles, and scan over time steps
        @nnx.scan(in_axes=(nnx.Carry, None, None, 0), out_axes=(nnx.Carry, 0))
        def step(z, param_net_complex, param_net_real, _):
            z_next = self._apply_one_step(z, param_net_complex, param_net_real)
            return z_next, z_next

        # shape: [T, B, F]
        _, trajectory = step(z0, self.param_net_complex, self.param_net_real, jnp.arange(T))
        # shape: [B, T, F]
        return jnp.transpose(trajectory, (1, 0, 2))

    def _apply_one_step(self, z, param_net_complex, param_net_real):
        """
        B: batch size
        """

        # Vmap over ensembles
        @nnx.vmap(in_axes=(0, 1), out_axes=1)
        def forward_param_network(model, x):
            return model(x)

        B = z.shape[0]
        split_idx = self.num_pair_complex * 2
        parts = []

        if self.num_pair_complex > 0:
            # shape: [B, num_pair_complex, 2]
            z_complex = z[:, :split_idx].reshape(B, self.num_pair_complex, 2)

            # shape: [B, num_pair_complex, 1]
            radii_squared = jnp.sum(z_complex**2, axis=-1, keepdims=True)

            # shape: [B, num_pair_complex, 2] where last dim is (omega, mu)
            params = forward_param_network(param_net_complex, radii_squared)
            omega, mu = params[..., 0], params[..., 1]

            # Advance
            scale = jnp.exp(mu * self.dt)
            cos_t, sin_t = jnp.cos(omega * self.dt), jnp.sin(omega * self.dt)

            y0, y1 = z_complex[..., 0], z_complex[..., 1]
            y0_next = scale * (cos_t * y0 - sin_t * y1)
            y1_next = scale * (sin_t * y0 + cos_t * y1)

            parts.append(jnp.stack([y0_next, y1_next], axis=-1).reshape(B, -1))

        if self.num_real > 0:
            z_real = z[:, split_idx:, None]
            mu = forward_param_network(param_net_real, z_real)[..., 0]
            parts.append(z[:, split_idx:] * jnp.exp(mu * self.dt))

        return jnp.concatenate(parts, axis=-1)


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
            nnx.Linear(hidden_dim, input_dim, rngs=rngs),
        )

        num_complex_pairs = koopman_dim // 2
        num_real = koopman_dim % 2
        self.koopman_operator = DynamicKoopmanOperator(
            num_real, num_complex_pairs, dt=dt, rngs=rngs
        )

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
