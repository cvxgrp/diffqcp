from __future__ import annotations

import equinox as eqx
from lineax import AbstractLinearOperator
from jaxtyping import Array, Float

from diffqcp._problem_data import ObjMatrix, ConstrMatrix

class _DuQAdjoint(AbstractLinearOperator):
    P: ObjMatrix
    A: ConstrMatrix
    AT: ConstrMatrix
    
    def __init__(
        self,
        P: ObjMatrix,
        A: ConstrMatrix,

    ):
        pass

    def mv(self, vector):
        pass

    def transpose(self) -> _DuQ:
        pass


class _DuQ(AbstractLinearOperator):
    """
    NOTE(quill): we know at compile time if this is batched or not.
    """
    P: ObjMatrix
    A: ConstrMatrix
    AT: ConstrMatrix
    q: Float[Array, "*batch n"]
    b: Float[Array, "*batch m"]
    x: Float[Array, "*batch n"]
    y: Float[Array, "*batch m"]
    tau: Float[Array, "*batch 1"]
    n: int = eqx.field(static=True)
    is_batched: bool = eqx.field(static=True)

    def __init__(
        self,
        P: ObjMatrix,
        A: ConstrMatrix,
        q: Float[Array, "*batch n"],
        b: Float[Array, "*batch m"],
        x: Float[Array, "*batch n"],
        y: Float[Array, "*batch m"],
        tau: Float[Array, "*batch 1"] # TODO(quill): ensure you handle shape correctly here
    ):
        pass

    def mv(self, du: Float[Array, "*batch n+m+1"]):
        # dx, dy, dtau = du[]
        pass
    
    def transpose(self) -> _DuQAdjoint:
        pass
