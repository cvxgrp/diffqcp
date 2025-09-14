from abc import abstractmethod

import equinox as eqx
from lineax import AbstractLinearOperator
from jaxtyping import Float, Array

class AbstractConeProjector(eqx.Module):

    # TODO(quill): re-consider the need to define `is_dual` or `dim`/`dims` here.
    #   How does this play with the abstract/final pattern.

    @abstractmethod
    def proj_dproj(self, x: Float[Array, " _n"]) -> tuple[Float[Array, " _n"], AbstractLinearOperator]:
        pass

    def __call__(self, x: Float[Array, " _n"]):
        return self.proj_dproj(x)