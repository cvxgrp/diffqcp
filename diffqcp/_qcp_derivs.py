from lineax import AbstractLinearOperator
from jaxtyping import Array, Float


def _Du_Q(
    n: int,
    x: Float[Array, " n"],
    tau: Float[Array, ""],
    P: AbstractLinearOperator,
    A: Float[Array, "m n"],
    AT: Float[Array, "n m"],
    q: Float[Array, " n"],
    b: Float[Array, " m"],
) -> AbstractLinearOperator:
    # TODO(quill): determine how to handle `n`
    #   -- before I grabbed P.shape[0]. have to be careful with batch dimension
    #   -- with current implementation, need to think about JAX API boundary.
    Px = P @ x
    xT_P_x = x @ Px

    # TODO(quill): think about using PyTrees here.
    # def mv(du: Float[Array, "n+m+1"]) -> Float[Array, "n+m+1"]:
