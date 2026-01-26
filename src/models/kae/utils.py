import jax.numpy as jnp
from flax import nnx


class UnitNormWrapper(nnx.Module):
    """
    Wrapper that normalizes the weights of a given layer to have unit norm
    """

    def __init__(self, layer, weight_name: str = "kernel", axis: int = 0):
        self.layer = layer
        self.weight_name = weight_name
        self.axis = axis

    def __call__(self, x):
        W = getattr(self.layer, self.weight_name).value
        W_normed = W / jnp.linalg.norm(W, axis=self.axis, keepdims=True)
        # Recompute forward pass with normalized weight
        return x @ W_normed + self.layer.bias.value
