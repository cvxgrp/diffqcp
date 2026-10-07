"""Projection onto the nonnegative orthant (self-dual)."""
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from .abstract_projector import AbstractConeProjector


class NonnegativeConeProjector(AbstractConeProjector):

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        proj_x = jnp.maximum(x, 0)
        dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
        return proj_x, dproj_x
