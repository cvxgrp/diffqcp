"""Functionality for projecting onto the second-order cone.

Keep in mind we want to enforce contiguousness as much as we can.

"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
from jaxtyping import (
    Array,
    Float,
    PyTree # pyright: ignore
)
from lineax import AbstractLinearOperator

from .abstract_projector import AbstractConeProjector
from .helpers import batch_cone_dims

if jax.config.jax_enable_x64:
    EPS = 1e-12
else:
    EPS = 1e-6

class SecondOrderConeProjector(AbstractConeProjector):
    """PyTree-valued second-order cone projector.
    
    The second-order cone {(t, u) in R x R^n | norm(u) <= t} is self-dual
    and has projection operator P given by the following three cases:

        1. norm(u) <= -t: P((t, u)) = 0
        2. norm(u) <= t: P((t, u)) = (t, u)
        3. norm(u) >= abs(t): P((t, u)) = 0.5 * (1 + t / norm(u)) (norm(u), u)

    The projection is differentiable so long as norm(u) != t. The corresponding
    Jacobian-vector product is given by the following three cases:

        1. norm(u) < -t: DP((t, u))[(dt, du)] = 0.
        2. norm(u) < t: DP((t, u))[(dt, du)] = (dt, du)
        3. norm(u) > abs(t):
            DP((t, u))[(dt, du)] = (1 / 2*norm(u)) * (norm(u)*dt + u @ du, u * dt + (t + norm(u))*du - (t / norm(u)**2) * (u @ du)* u)

    norm is the Euclidean norm.

    Note on computational efficiency: If you pass a 1D array in order.
    Force good structure by placing constraints next to each other in problem
    formulation.

    NOTE(quill): no point in separating out a projector that doesn't require static dims
    since methods get re-traced whenever shapes of arrays change.
    """

    dims: list[int] = eqx.field(static=True)
    batched_dims: list[tuple[int, int]] = eqx.field(static=True)
    
    def __init__(self, dims: list[int] | PyTree[jax.ShapeDtypeStruct]):
        # check if dims is a list of `int`s or a `PyTree` of `ShapeDtypeStruct`s.
        # if a PyTree then determine a method for extracting batched info
        self.dims = dims
        self.summed_dims = sum(self.sims)
        self.batched_dims = batch_cone_dims(self.dims)
    
    def proj(self, x: Float[Array, " summed_dims"] | PyTree[Float[Array, " dims_i"]]):
        if isinstance(x, Array):
            # split
            pass
        else:
            # perhaps check dimensionality of the PyTree?
            pass
        
        raise NotImplementedError
    
    def dproj(self, x: Float[Array, " summed_dims"]):
        # If `x` is a PyTree then the return value should be
        # a PyTree of Jacobians?
        # jax helpers should make this pretty easy to handle.
        raise NotImplementedError
    
    def proj_dproj(self, x: Float[Array, " summed_dims"]):
        raise NotImplementedError


class SecondOrderConeProjectorJacobian(AbstractLinearOperator):

    t: Float[Array, "*batch 1"]
    z : Float[Array, "*batch n-1"]

    def __init__(self):
        pass

    def mv(self, dx: Float[Array, "*batch "]):
        # how does this work with a PyTree <=> how do you batch a PyTree?
        # => pretty sure leading dimension of the PyTree becomes batched?
        pass


def _soc_proj_dproj(x: Float[Array, " n+1"], return_) -> Float[Array, " n+1"]:
    t, z = x[0], x[1:]
    norm_z = jnp.maximum(jla.norm(z), EPS) # safe norm
    unit_z = z / norm_z

    def identity_case():
        pass