"""Functionality for projecting onto the zero cone.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from .abstract_projector import AbstractConeProjector

class ZeroConeProjector(AbstractConeProjector):
    """
    The zero cone {0} has dual cone R, projection operator P(z) = 0
    and DP(z)[dz] = 0.
    """

    onto_dual: bool = False

    def proj(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        return x if self.onto_dual else jnp.zeros_like(x)

    def dproj(self, x: Float[Array, " n"]) -> AbstractLinearOperator:
        return ZeroConeProjectorJacobian(x, self.onto_dual)

    def proj_dproj(self, x):
        return self.proj(x), self.dproj(x)
    

class ZeroConeProjectorJacobian(AbstractLinearOperator):
    """
    The Jacobian of the projection onto the zero cone is 0.
    The Jacobian of the projection onto the dual of the zero cone, R, is I
    (the identity matrix).
    """

    x: Float[Array, "*batch n"]
    onto_dual: bool = eqx.field(static=False)

    def __init__(self, x: Float[Array, "*batch n"], onto_dual: bool):
        self.x = x
        self.onto_dual = onto_dual

    def mv(self, dx: Float[Array, "*batch n"]):
        return jnp.zeros_like(dx) if not self.onto_dual else dx
    
    def as_matrix(self):
        xdim = jnp.ndim(self.x)
        if xdim == 1:
            pass
        elif xdim == 2:
            pass
        else:
            raise ValueError
    
    def transpose(self) -> AbstractLinearOperator:
        return self
    
    def in_structure(self):
        raise NotImplementedError
    
    def out_structure(self):
        raise NotImplementedError