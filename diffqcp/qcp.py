import equinox as eqx
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR

from diffqcp.cones.canonical import AbstractConeProjector
# TODO(quill): provide helpers to convert problem data?

class QCP(eqx.Module):
    """Quadratic Cone Program.

    Represents a (solved) convex cone program given
    by the primal-dual problems

        (P) minimize    (1/2)x^T P x + q^T x
            subject to  Ax + s = b
                        s in K

        (D) minimize    -(1/2)x^T P x - b^T y
            subject to  A^T y + q = 0
                        y in K^*,

    where P, A, q, b are mutable problem data, K and K^* are
    immutable problem data, and (x, y, s) are the optimization
    variables.
    """
    P: Float[Array | BCOO | BCSR, "*batch n n"]
    A: Float[Array | BCOO | BCSR, "*batch m n"]
    q: Float[Array, "*batch n"]
    b: Float[Array, "*batch m"]
    x: Float[Array, "*batch n"]
    y: Float[Array, "*batch m"]
    s: Float[Array, "*batch m"]

    # cones: dict[str, int | list[int] | list[float]] # NOTE(quill): don't need since just store projector?
    # TODO(quill): need to pass in dtype or device?
    # cone_projector: AbstractConeProjector
    cone_projector: AbstractConeProjector
    n: int
    m : int
    N: int
    is_batched: bool

    def __init__(
        self,
        P: Float[Array | BCOO | BCSR, "*batch n n"],
        A: Float[Array | BCOO | BCSR, "*batch m n"],
        q: Float[Array, "*batch n"],
        b: Float[Array, "*batch m"],
        x: Float[Array, "*batch n"],
        y: Float[Array, "*batch m"],
        s: Float[Array, "*batch m"],
        cones: dict[str, int | list[int] | list[float]]
    ) -> None:
        # TODO(quill): will need to check type of `P` and `A`
        #   and create transpose op if they are of type `BCSR`.
        
        # === dimensionality checks ===

        P_shape = P.shape
        P_num_dims = len(P_shape)
        if P_num_dims == 2:
            self.is_batched = False
            self.n = P_shape[0]
            # TODO(quill): add check(s) on `P`?
            #   e.g., full (i.e., not upper triangular), symmetric
            #   Also, this `__init__` will be run each time...so
            #   consider what's critical. Can always create a helper
            #   function to run on the data before creating a `QCP`
            #   the first time.
        elif P_num_dims == 3:
            self.is_batched = True
            self.n = P_shape[1]
        else:
            raise ValueError("The quadratic objective matrix `P` must be"
                             + " a 2D or 3D array. The provided `P`"
                             + f" is a {P_num_dims}D array.")
        
        A_shape = A.shape
        A_num_dims = len(A_shape)
        if A_num_dims != P_num_dims:
            raise ValueError("The constraint matrix `A` must have the"
                             + " same dimensionality as the quadratic objective"
                             + f" matrix `P`, however `P` is a {P_num_dims}D"
                             + f" array while `A` is a {A_num_dims}D array.")
        self.A = A
        self.m = A_shape
        self.N = self.n + self.m + 1
        
        
        q_shape = q.shape
        q_num_dims = len(q_shape)
        if q_num_dims != P_num_dims - 1:
            raise ValueError("Since the quadratic objective matrix `P`"
                             + f" is a {P_num_dims}D array, `q` must"
                             + f" be a {P_num_dims-1}D array, but it is"
                             + f" actually a {q_num_dims}D array.")
        # TODO(quill): finish checking that `q` size is `n` then check `b`
        

        # === dtype checks ===

        # === create cone module ===
    
    def _form_atoms(
        self
    ):
        pass

    def _form_atoms_batched(
        self
    ):
        # NOTE(quill): return matrix operator?
        pass
    
    def jvp(
        self,
        dP: Float[Array | BCOO | BCSR, "*batch n n"],
        dA: Float[Array | BCOO | BCSR, "*batch m n"],
        dq: Float[Array, "*batch n"],
        db: Float[Array, "*batch m"]
    ):
        # TODO(quill): return a `PyTree` or a `tuple`?
        # TODO(quill): should be a pure function / can be jited, vmaped, autodiffed, etc.
        pass

    def vjp(
        self,
        dx: Float[Array, "*batch n"],
        dy: Float[Array, "*batch m"],
        ds: Float[Array, "*batch m"]
    ):
        # TODO(quill): return a `PyTree` or a tuple?
        # TODO(quill): should be a pure function / can be jited, vmaped, autodiffed, etc.
        pass

QCP.__init__.__doc__ = """**Parameters**
- 
"""

def _check_qcp_arguments_dimensionality():
    pass