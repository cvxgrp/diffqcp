"""The abstract base class shared by every cone projector."""
from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator


class AbstractConeProjector(eqx.Module):
    """Abstract base class for a projector onto a (product of) convex cone(s) K.

    A projector maps a point `x` to its Euclidean projection `Pi_K(x)` together
    with the Jacobian `DPi_K(x)` of the projection, represented as a lineax
    operator. Concrete subclasses define `proj_dproj`; calling the projector is
    the same as calling `proj_dproj`.

    Subclasses that take an `onto_dual` flag project onto the dual cone K^*
    instead of K when it is `True`.
    """

    @abstractmethod
    def proj_dproj(self, x: Float[Array, " _n"]) -> tuple[Float[Array, " _n"], AbstractLinearOperator]:
        """Project `x` onto the cone and form the Jacobian of the projection at `x`.

        **Arguments:**
        - `x`: the point to project, a 1D array whose length is the cone's
            dimension.

        **Returns:**

        A 2-tuple `(proj_x, dproj_x)`: the projection of `x` onto the cone, and
        the Jacobian of the projection evaluated at `x`, as a lineax operator
        mapping perturbations of `x` to perturbations of `proj_x`.
        """
        pass

    def __call__(self, x: Float[Array, " _n"]) -> tuple[Float[Array, " _n"], AbstractLinearOperator]:
        """Alias for `proj_dproj`."""
        return self.proj_dproj(x)
