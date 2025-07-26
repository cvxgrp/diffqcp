from jax import vmap, eval_shape
import jax.numpy as jnp
import equinox as eqx
from lineax import AbstractLinearSolver, AbstractLinearOperator, IdentityLinearOperator
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR

from diffqcp._problem_data import QCPStructureCPU
from diffqcp.cones.canonical import ProductConeProjector
from diffqcp._linops import _BlockLinearOperator
from diffqcp._qcp_derivs import _DuQ
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

    Attributes
    ----------
    """
    P: Float[Array | BCOO | BCSR, "*batch n n"]
    A: Float[Array | BCOO | BCSR, "*batch m n"]
    q: Float[Array, "*batch n"]
    b: Float[Array, "*batch m"]
    x: Float[Array, "*batch n"]
    y: Float[Array, "*batch m"]
    s: Float[Array, "*batch m"]

    cone_projector: ProductConeProjector
    n: int = eqx.field(static=True)
    m : int = eqx.field(static=True)
    N: int = eqx.field(static=True)
    is_batched: bool = eqx.field(static=True)

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
        P_num_dims = jnp.ndim(P)
        if P_num_dims == 2:
            self.is_batched = False
            self.n = P_shape[0]
            # TODO(quill): add check(s) on `P`?
            #   e.g., full (i.e., not upper triangular), symmetric
            #   Also, this `__init__` will be run each time in a learning loop
            #   ...so consider what's critical. Can always create a helper
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
        A_num_dims = jnp.ndim(A)

        if A_num_dims != P_num_dims:
            raise ValueError("The constraint matrix `A` must have the"
                             + " same dimensionality as the quadratic objective"
                             + f" matrix `P`, however `P` is a {P_num_dims}D"
                             + f" array while `A` is a {A_num_dims}D array.")
        self.A = A
        self.m = A_shape[1] if self.is_batched else A_shape[0]
        self.N = self.n + self.m + 1
        
        q_shape = q.shape
        q_num_dims = jnp.ndim(q_shape)
        if q_num_dims != P_num_dims - 1:
            raise ValueError("Since the quadratic objective matrix `P`"
                             + f" is a {P_num_dims}D array, `q` must"
                             + f" be a {P_num_dims-1}D array, but it is"
                             + f" actually a {q_num_dims}D array.")
        # TODO(quill): finish checking that `q` size is `n` then check `b`
        

        # === dtype checks ===

        _cone_projector = ProductConeProjector(cones)
        # TODO(quill): the following makes the projector a `Callable`
        self.cone_projector = vmap(_cone_projector) if self.is_batched else _cone_projector

    
    def _compute_atoms(self):
        pass
    
    def jvp(
        self,
        dP: Float[Array | BCOO | BCSR, "n n"],
        dA: Float[Array | BCOO | BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ):
        proj_kstar_v, dproj_kstar_v = self.cone_projector(self.y - self.s) # so this needs to know if batched
        pi_z = jnp.concatenate([self.x, proj_kstar_v, jnp.array([1.0], dtype=self.x.dtype)]) # what if x is batched
        dpi_z = _BlockLinearOperator([IdentityLinearOperator(eval_shape(lambda: self.x)),
                                      dproj_kstar_v,
                                      IdentityLinearOperator(eval_shape(lambda: jnp.array([1.0])))])
        Px = self.P @ self.x
        xTPx = self.x @ Px
        AT = self.A.T
        # NOTE(quill): seems hard to avoid the `DzQ` bit of the variable name.
        # NOTE(quill): Note that we're skipping the step of extracting the first n components of
        #   `pi_z` and just using `P @ pi_z[:n] = P @ x`. 
        DzQ_pi_z = _DuQ(P=self.P, Px=Px, xTPx=xTPx, A=self.A, AT=AT, q=self.q,
                        b=self.b, x=self.x, tau=jnp.array(1.0, dtype=self.x.dtype),
                        n=self.problem_structure.n, m=self.problem_structure.m)
        
        # NOTE(quill): we use that z_N (as defined in paper) is always 1.0, thus don't
        #   include that division.
        F = DzQ_pi_z @ dpi_z - dpi_z + IdentityLinearOperator(eval_shape(lambda: pi_z))

        # TODO(quill): we will not convert the perturbation data; add checks
        

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ):
        # TODO(quill): return a `PyTree` or a tuple?
        # TODO(quill): should be a pure function / can be jited, vmaped, autodiffed, etc.
        pass

QCP.__init__.__doc__ = """**Parameters**
- 
"""

def _check_qcp_arguments_dimensionality():
    pass