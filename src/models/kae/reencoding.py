"""
Based on:
https://openreview.net/forum?id=A18gWgc5mi
"""

from flax import nnx


class ReencodingAutoencoder(nnx.Module):
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

        self.koopman_operator = None

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
