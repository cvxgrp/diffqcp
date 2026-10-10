"""Projection onto the nonnegative orthant (self-dual)."""
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from .abstract_projector import AbstractConeProjector


class NonnegativeConeProjector(AbstractConeProjector):
    """Projector onto the nonnegative orthant R^n_+ = {x : x_i >= 0 for all i}.

    The orthant is self-dual, so there is no `onto_dual` flag. The input is the
    `n` entries of the `l` block of the cone vector. Takes no constructor
    arguments.
    """

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        """Project `x` onto the nonnegative orthant.

        **Arguments:**
        - `x`: the point to project. Operations are elementwise, so a leading
            batch axis (e.g. from `vmap`) is also handled.

        **Returns:**

        A 2-tuple of the projection `max(x, 0)` and its Jacobian, a
        `lineax.DiagonalLinearOperator` with diagonal entries 1 where `x > 0`, 0
        where `x < 0` and 1/2 where `x = 0` (a fixed element of the generalized
        Jacobian at the kink).
        """
        proj_x = jnp.maximum(x, 0)
        dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
        return proj_x, dproj_x
