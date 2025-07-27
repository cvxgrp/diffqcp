from abc import abstractmethod
import jax
from jax import eval_shape
import jax.numpy as jnp
import equinox as eqx
from lineax import AbstractLinearOperator, IdentityLinearOperator, linear_solve, LSMR
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR

from diffqcp._problem_data import QCPStructureCPU, QCPStructureGPU, QCPStructure
from diffqcp.cones.canonical import ProductConeProjector
from diffqcp._linops import _BlockLinearOperator
from diffqcp._qcp_derivs import _DuQ
# TODO(quill): provide helpers to convert problem data?

class AbstractQCP(eqx.Module):
    """Quadratic Cone Program.

    Represents a (solved) quadratic convex cone program given
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

    P: eqx.AbstractVar[BCSR | AbstractLinearOperator]
    A: eqx.AbstractVar[BCSR | BCOO]
    q: eqx.AbstractVar[Array]
    b: eqx.AbstractVar[Array]
    x: eqx.AbstractVar[Array]
    y: eqx.AbstractVar[Array]
    s: eqx.AbstractVar[Array]
    problem_structure: eqx.AbstractVar[QCPStructure]

    def _form_atoms(self) -> tuple[AbstractLinearOperator, AbstractLinearOperator]:
        proj_kstar_v, dproj_kstar_v = self.problem_structure.cone_projector(self.y - self.s)
        pi_z = jnp.concatenate([self.x, proj_kstar_v, jnp.array([1.0], dtype=self.x.dtype)])
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

        return (F, dproj_kstar_v)
    
    @abstractmethod
    def jvp(
        self,
        dP: Float[BCOO | BCSR, "n n"],
        dA: Float[BCOO | BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        raise NotImplementedError
    
    @abstractmethod
    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ) -> tuple[
        Float[BCSR, "n n"], Float[BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        raise NotImplementedError


class HostQCP(AbstractQCP):
    """QCP whose subroutines are optimized to run on host.
    """
    P: Float[ObjMatrix, "n n"]
    A: Float[BCOO, "m n"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    y: Float[Array, " m"]
    s: Float[Array, " m"]

    problem_structure: QCPStructureCPU = eqx.field(static=True)

    def __init__(
        self,
        P: Float[BCOO, "n n"],
        A: Float[BCOO, "m n"],
        q: Float[Array, " n"],
        b: Float[Array, " m"],
        x: Float[Array, " n"],
        y: Float[Array, " m"],
        s: Float[Array, " m"],
        problem_structure: QCPStructureCPU
    ):
        self.A, self.q, self.b = A, q, b
        self.x, self.y, self.s = x, y, s
        self.problem_structure = problem_structure

        # Now create some sort of operator for host

    
    def jvp(
        self,
        dP: Float[BCOO, "n n"],
        dA: Float[BCOO, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ):
        F, dproj_kstar_v = self._form_atoms()

        # so if we are on the cpu, then dP is just the upper triangular bit of the
        #   matrix P, so we'll need to wrap that
        # -> make dP a new linop
        

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ):
        F, dproj_kstar_v = self._form_atoms()


class DeviceQCP(AbstractQCP):
    """QCP whose subroutines are optimized to run on device.
    """
    # NOTE(quill): when we allow for batched problem data, will need
    #   to wrap `P` in an `AbstractLinearOperator` to dictate how the `mv`
    #   operation should behave.
    P: Float[BCSR, "n n"]
    A: Float[BCOO, "m n"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    y: Float[Array, " m"]
    s: Float[Array, " m"]

    problem_structure: QCPStructureCPU = eqx.field(static=True)

    # NOTE(quill): don't need a custom `__init__`` (until we wrap `P`).
    
    def jvp(
        self,
        dP: Float[BCOO, "n n"],
        dA: Float[BCOO, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        F, dproj_kstar_v = self._form_atoms()
        # TODO(quill): start solver from previous spot?
        #   => (so would need `dz`)
        dAT = ...
        d_data_N = dD

        # so if we are on the cpu, then dP is just the upper triangular bit of the
        #   matrix P, so we'll need to wrap that
        # -> make dP a new linop
        

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ) -> tuple[
        Float[BCSR, "n n"], Float[BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        
        F, dproj_kstar_v = self._form_atoms()
        dz = jnp.concatenate([dx,
                              dproj_kstar_v @ (dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])]
                              )
        
        def zero_case():
            return jnp.zeros_like(dz)
        
        def non_zero_case():
            # TODO(quill): start solver from previous spot?
            #   => (so would need previous `d_data_N`)
            return linear_solve(F.T, -dz, solver=LSMR)

        d_data_N = jax.lax.cond(jnp.allclose(dz, 0),
                                zero_case,
                                non_zero_case)
        
        # TODO(quill): save/output residual?

        # so create a `dData_Q_adjoint_device`

# QCP.__init__.__doc__ = """**Parameters**
# - 
# """

    # def __init__(
    #     self,
    #     P: Float[Array | BCOO | BCSR, "*batch n n"],
    #     A: Float[Array | BCOO | BCSR, "*batch m n"],
    #     q: Float[Array, "*batch n"],
    #     b: Float[Array, "*batch m"],
    #     x: Float[Array, "*batch n"],
    #     y: Float[Array, "*batch m"],
    #     s: Float[Array, "*batch m"],
    #     problem_structure: QCPStructure
    # ) -> None:
        # TODO(quill): will need to check type of `P` and `A`
        #   and create transpose op if they are of type `BCSR`.
        
        # === dimensionality checks ===

        # P_shape = P.shape
        # P_num_dims = jnp.ndim(P)
        # if P_num_dims == 2:
        #     self.is_batched = False
        #     self.n = P_shape[0]
        #     # TODO(quill): add check(s) on `P`?
        #     #   e.g., full (i.e., not upper triangular), symmetric
        #     #   Also, this `__init__` will be run each time in a learning loop
        #     #   ...so consider what's critical. Can always create a helper
        #     #   function to run on the data before creating a `QCP`
        #     #   the first time.
        # elif P_num_dims == 3:
        #     self.is_batched = True
        #     self.n = P_shape[1]
        # else:
        #     raise ValueError("The quadratic objective matrix `P` must be"
        #                      + " a 2D or 3D array. The provided `P`"
        #                      + f" is a {P_num_dims}D array.")
        
        # A_shape = A.shape
        # A_num_dims = jnp.ndim(A)

        # if A_num_dims != P_num_dims:
        #     raise ValueError("The constraint matrix `A` must have the"
        #                      + " same dimensionality as the quadratic objective"
        #                      + f" matrix `P`, however `P` is a {P_num_dims}D"
        #                      + f" array while `A` is a {A_num_dims}D array.")
        # self.A = A
        # self.m = A_shape[1] if self.is_batched else A_shape[0]
        # self.N = self.n + self.m + 1
        
        # q_shape = q.shape
        # q_num_dims = jnp.ndim(q_shape)
        # if q_num_dims != P_num_dims - 1:
        #     raise ValueError("Since the quadratic objective matrix `P`"
        #                      + f" is a {P_num_dims}D array, `q` must"
        #                      + f" be a {P_num_dims-1}D array, but it is"
        #                      + f" actually a {q_num_dims}D array.")
        # # TODO(quill): finish checking that `q` size is `n` then check `b`

def _check_qcp_arguments_dimensionality():
    pass

def _convert_perturbation(dP: Float[Array | BCOO | BCSR, "n n"]) -> ObjMatrix:
    pass