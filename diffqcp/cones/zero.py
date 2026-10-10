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
    """Jacobian of the projection onto the zero cone (or onto its dual, R^n).

    The projection onto {0} is constant, so its Jacobian is the zero operator;
    the projection onto R^n is the identity map, so its Jacobian is the identity.
    `mv` is elementwise and therefore also works on inputs with leading batch axes.

    **Arguments:**
    - `x`: the point at which the projection was evaluated. Only its shape and
        dtype are used (for `in_structure`).
    - `onto_dual`: if `True`, this is the Jacobian of the projection onto R^n
        (the identity); otherwise of the projection onto {0} (zero).
    """

    x: Float[Array, "*B n"]
    onto_dual: bool = eqx.field(static=True)

    def __init__(self, x: Float[Array, "*B n"], onto_dual: bool):
        # NOTE(quill): storing `x` only to recover its shape is hacky; storing
        #   `jax.eval_shape(lambda: x)` instead did not work with `vmap`.
        self.x = x
        self.onto_dual = onto_dual

    def mv(self, dx: Float[Array, "*B n"]) -> Float[Array, "*B n"]:
        if not self.onto_dual:
            return jnp.zeros_like(dx)
        else:
            return dx

    def as_matrix(self) -> Float[Array, "n n"]:
        return _dense_from_mv(self)

    def transpose(self) -> lx.AbstractLinearOperator:
        # NOTE(quill): while the zero cone is not self-dual, the Jacobian of the
        #   projection in either case is symmetric.
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.eval_shape(lambda: self.x)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()

@lx.is_symmetric.register(_ZeroConeProjectorJacobian)
def _(op: _ZeroConeProjectorJacobian) -> bool:
    return True


class ZeroConeProjector(AbstractConeProjector):
    """Projector onto the zero cone {0} = {x in R^n : x = 0}, or onto its dual R^n.

    The input is the `n` entries of the `z` block of the cone vector.

    **Arguments:**
    - `onto_dual`: if `False`, project onto {0} (the projection is `0` and its
        Jacobian is zero). If `True`, project onto the dual cone {0}^* = R^n (the
        projection is the identity map and so is its Jacobian).
    """

    onto_dual: bool

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        """Project `x` onto {0} (or onto R^n if `onto_dual`).

        **Arguments:**
        - `x`: the point to project. Since everything is elementwise, a leading
            batch axis (e.g. from `vmap`) is also handled.

        **Returns:**

        A 2-tuple of the projection (`zeros_like(x)`, or `x` itself if
        `onto_dual`) and its Jacobian as a lineax operator (zero, or the identity
        if `onto_dual`).
        """
        if self.onto_dual:
            return (x, _ZeroConeProjectorJacobian(x=x, onto_dual=True))
        else:
            return (jnp.zeros_like(x), _ZeroConeProjectorJacobian(x=x, onto_dual=False))
