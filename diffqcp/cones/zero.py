"""Projection onto the zero cone {0} (and its dual, R^n)."""
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from diffqcp.linops import _dense_from_mv

from .abstract_projector import AbstractConeProjector


class _ZeroConeProjectorJacobian(lx.AbstractLinearOperator):
    # NOTE(quill): this Jacobian operator already works on arbitrarily-dimensioned arrays.
    #   i.e., it already can operate on batches of points to project.
    x: Float[Array, "*B n"]
    onto_dual: bool = eqx.field(static=True) # NOTE(quill): known at compile time

    def __init__(self, x: Float[Array, "*B n"], onto_dual: bool):
        # self.shape_dtype = jax.eval_shape(lambda: x) # NOTE(quill): this didn't work with `vmap`
        self.x = x # NOTE(quill): this is hacky; see if you can store shape w/o storing array.
        self.onto_dual = onto_dual

    def mv(self, dx: Float[Array, "*B n"]):
        if not self.onto_dual:
            return jnp.zeros_like(dx)
        else:
            return dx

    def as_matrix(self):
        return _dense_from_mv(self)

    def transpose(self) -> lx.AbstractLinearOperator:
        # NOTE(quill): while the projector is not self-dual, the Jacobian of the
        #   projection in either case is symmetric.
        return self

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)

    def out_structure(self):
        return self.in_structure()

@lx.is_symmetric.register(_ZeroConeProjectorJacobian)
def _(op):
    return True


class ZeroConeProjector(AbstractConeProjector):

    onto_dual: bool

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        if self.onto_dual:
            return (x, _ZeroConeProjectorJacobian(x=x, onto_dual=True))
        else:
            return (jnp.zeros_like(x), _ZeroConeProjectorJacobian(x=x, onto_dual=False))
