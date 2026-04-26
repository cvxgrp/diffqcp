from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator


class AbstractConeProjector(eqx.Module):

    @abstractmethod
    def proj_dproj(self, x: Float[Array, " _n"]) -> tuple[Float[Array, " _n"], AbstractLinearOperator]:
        pass

    def __call__(self, x: Float[Array, " _n"]):
        return self.proj_dproj(x)
