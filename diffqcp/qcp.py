from abc import abstractmethod
from typing import Callable
import jax
from jax import eval_shape
import jax.numpy as jnp
import equinox as eqx
from lineax import AbstractLinearOperator, IdentityLinearOperator, linear_solve, LSMR
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR

from diffqcp._problem_data import (QCPStructureCPU, QCPStructureGPU,
                                   QCPStructure, ObjMatrixCPU, ObjMatrixGPU, ObjMatrix)
from diffqcp._linops import _BlockLinearOperator
from diffqcp._qcp_derivs import (_DuQ, _d_data_Q, _d_data_Q_adjoint_cpu, _d_data_Q_adjoint_gpu)
# TODO(quill): make a note that the "CPU" and "GPU" qualifiers are somewhat misleading.

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

    P: eqx.AbstractVar[ObjMatrix]
    A: eqx.AbstractVar[BCSR | BCOO]
    q: eqx.AbstractVar[Array]
    b: eqx.AbstractVar[Array]
    x: eqx.AbstractVar[Array]
    y: eqx.AbstractVar[Array]
    s: eqx.AbstractVar[Array]
    problem_structure: eqx.AbstractVar[QCPStructure]

    def _form_atoms(self) -> tuple[Float[Array, " n+m+1"], AbstractLinearOperator, AbstractLinearOperator]:
        proj_kstar_v, dproj_kstar_v = self.problem_structure.cone_projector(self.y - self.s)
        pi_z = jnp.concatenate([self.x, proj_kstar_v, jnp.array([1.0], dtype=self.x.dtype)])
        dpi_z = _BlockLinearOperator([IdentityLinearOperator(eval_shape(lambda: self.x)),
                                      dproj_kstar_v,
                                      IdentityLinearOperator(eval_shape(lambda: jnp.array([1.0])))])
        Px = self.P.mv(self.x)
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

        return (pi_z, F, dproj_kstar_v)
    
    def _jvp_common(
        self,
        dP: ObjMatrix,
        dA: Float[BCOO | BCSR, "m n"],
        dAT: Float[BCOO | BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        pi_z, F, dproj_kstar_v = self._form_atoms()
        n, m = self.problem_structure.n, self.problem_structure.m
        pi_z_n, pi_z_m, pi_z_N = pi_z[:n], pi_z[n:n+m], pi_z[-1]
        d_data_N = _d_data_Q(x=pi_z_n, y=pi_z_m, tau=pi_z_N, dP=dP,
                             dA=dA, dAT=dAT, dq=dq, db=db)
        
        def zero_case():
            return jnp.zeros_like(d_data_N)
        
        def nonzero_case():
            # TODO(quill): start solver from previous spot?
            #   => (so would need `dz`)
            soln = linear_solve(F, -d_data_N, solver=LSMR(rtol=1e-6, atol=1e-6))
            return soln.value

        # patdb.debug()
        dz = jax.lax.cond(jnp.allclose(d_data_N, 0),
                          zero_case,
                          nonzero_case)
        
        dz_n, dz_m, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dx = dz_n - self.x * dz_N
        dproj_k_star_v_dz_m = dproj_kstar_v.mv(dz_m)
        dy = dproj_k_star_v_dz_m - self.y * dz_N
        ds = dproj_k_star_v_dz_m - dz_m - self.s * dz_N
        return dx, dy, ds
    
    @abstractmethod
    def jvp(
        self,
        dP: Float[BCOO | BCSR, "n n"],
        dA: Float[BCOO | BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        """Apply the derivative of the QCP's solution map to an input perturbation.
        """
        raise NotImplementedError
    
    def _vjp_common(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        produce_output: Callable
    ) -> tuple[
        Float[BCOO | BCSR, "n n"], Float[BCOO | BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        n, m = self.problem_structure.n, self.problem_structure.m
        pi_z, F, dproj_kstar_v = self._form_atoms()
        dz = jnp.concatenate([dx,
                              dproj_kstar_v @ (dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])]
                              )
        
        def zero_case():
            return jnp.zeros_like(dz)
        
        def nonzero_case():
            # TODO(quill): start solver from previous spot?
            #   => (so would need previous `d_data_N`)
            soln = linear_solve(F.T, -dz, solver=LSMR(rtol=1e-6, atol=1e-6))
            return soln.value

        d_data_N = jax.lax.cond(jnp.allclose(dz, 0),
                                zero_case,
                                nonzero_case)
        
        # TODO(quill): save/output residual?

        pi_z_n = pi_z[:n]
        pi_z_m = pi_z[n:n+m]
        pi_z_N = pi_z[-1]
        d_data_N_n = d_data_N[:n]
        d_data_N_m = d_data_N[n:n+m]
        d_data_N_N = d_data_N[-1]
        
        return produce_output(x=pi_z_n, y=pi_z_m, tau=pi_z_N,
                              w1=d_data_N_n, w2=d_data_N_m, w3=d_data_N_N,
                              P_rows=self.problem_structure.P_nonzero_rows,
                              P_cols=self.problem_structure.P_nonzero_cols,
                              A_rows=self.problem_structure.A_nonzero_rows,
                              A_cols=self.problem_structure.A_nonzero_cols,
                              n=n, m=m)
        
    @abstractmethod
    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ) -> tuple[
        Float[BCOO | BCSR, "n n"], Float[BCOO | BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the QCP's solution map to a solution perturbation.
        """
        raise NotImplementedError


class HostQCP(AbstractQCP):
    """QCP whose subroutines are optimized to run on host (CPU).
    """
    P: ObjMatrixCPU
    A: Float[BCOO, "m n"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    y: Float[Array, " m"]
    s: Float[Array, " m"]

    problem_structure: QCPStructureCPU

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
        """**Arguments:**
        - `P`: BCOO, shape (n, n). The quadratic objective matrix. Must be symmetric and provided in sparse BCOO format.
            Only the upper triangular part is required and used for efficiency.
        - `A`: BCOO, shape (m, n). The constraint matrix in sparse BCOO format.
        - `q`: ndarray, shape (n,). The linear objective vector.
        - `b`: ndarray, shape (m,). The constraint vector.
        - `x`: ndarray, shape (n,). The primal solution vector.
        - `y`: ndarray, shape (m,). The dual solution vector.
        - `s`: ndarray, shape (m,). The primal slack variable.
        - `problem_structure`: QCPStructureCPU. Structure object containing metadata about the problem, including sparsity patterns (such as the nonzero row and column indices for P and A), and cone information.

        **Notes:**
        - The sparsity structure of `P` and `A` must match that described in `problem_structure`.
        - `P` should only contain the upper triangular part of the matrix.
        - All arrays should be on the host (CPU) and compatible with JAX operations.
        """
        self.A, self.q, self.b = A, q, b
        self.x, self.y, self.s = x, y, s
        self.problem_structure = problem_structure
        self.P = self.problem_structure.form_obj(P)
    
    def jvp(
        self,
        dP: Float[BCOO, "n n"],
        dA: Float[BCOO, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        """Apply the derivative of the QCP's solution map to an input perturbation.

        Specifically, an implementation of the method given in section 3.1 of the paper.
        
        **Arguments:**
        - `dP`: should have the same sparsity structure as `P`. *Note* that
            this means it should only contain the upper triangular part of `dP`.
        - `dA`: should have the same sparsity structure as `A`.
        - `dq`
        - `db`
    
        **Returns:**
        
        A 3-tuple containing the perturbations to the solution: `(dx, dy, ds)`.
        """
        # NOTE(quill): this implementation is identitcal to `DeviceQCP`'s implementation
        #   minus the `dAT = dA.T`.
        #   Can this be consolidated / does it indicate incorrect design decision/execution?
        #   => NOTE(quill): I've attempted to address this annoyance with `_jvp_common`.
        dAT = dA.T
        dP = self.problem_structure.form_obj(dP)
        # need to wrap dP.
        return self._jvp_common(dP=dP, dA=dA, dAT=dAT, dq=dq, db=db)

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ) -> tuple[
        Float[BCSR, "n n"], Float[BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the QCP's solution map to a solution perturbation.
        
        Specifically, an implementation of the method given in section 3.2 of the paper.
        
        **Arguments:**
        - `dx`: A perturbation to the primal solution.
        - `dy`: A perturbation to the dual solution.
        - `ds`: A perturbation to the primal slack solution.

        **Returns**

        A four-tuple containing the perturbations to the objective matrix, constraint matrix,
        linear cost function vector, and constraint vector. Note that these perturbation matrices
        will have the same sparsity patterns as their corresponding problem matrices. (So, importantly,
        the first matrix will only contain the upper triangular part of the true perturbation to the
        objective matrix perturbation.)
        """
        # NOTE(quill): This is a similar note to the one I left in this class's `jvp`. That is, this
        #   implementation is identical to `DeviceQCP`'s `vjp` minus the function call at the very bottom.
        #   Can this be consolidated / does it indicate incorrect design decision/execution?
        return self._vjp_common(dx=dx, dy=dy, ds=ds,
                                produce_output=_d_data_Q_adjoint_cpu)


class DeviceQCP(AbstractQCP):
    """QCP whose subroutines are optimized to run on device (GPU).
    """
    # NOTE(quill): when we allow for batched problem data, will need
    #   to wrap `P` in an `AbstractLinearOperator` to dictate how the `mv`
    #   operation should behave.
    P: ObjMatrixGPU
    A: Float[BCSR, "m n"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    y: Float[Array, " m"]
    s: Float[Array, " m"]

    problem_structure: QCPStructureGPU

    def __init__(
        self,
        P: Float[BCSR, "n n"],
        A: Float[BCSR, "m n"],
        q: Float[Array, " n"],
        b: Float[Array, " m"],
        x: Float[Array, " n"],
        y: Float[Array, " m"],
        s: Float[Array, " m"],
        problem_structure: QCPStructureCPU
    ):
        """**Arguments:**
        - `P`: BCSR, shape (n, n). The quadratic objective matrix in sparse BCSR format.
            Must be symmetric. For device execution, the full matrix (not just upper triangular) is required.
        - `A`: BCSR, shape (m, n). The constraint matrix in sparse BCSR format.
        - `q`: ndarray, shape (n,). The linear objective vector.
        - `b`: ndarray, shape (m,). The constraint vector.
        - `x`: ndarray, shape (n,). The primal solution vector.
        - `y`: ndarray, shape (m,). The dual solution vector.
        - `s`: ndarray, shape (m,). The primal slack variable.
        - `problem_structure`: QCPStructureGPU. Structure object containing metadata about the problem, including sparsity patterns
            (such as the nonzero row and column indices for P and A), and cone information.

        **Notes:**
        - The sparsity structure of `P` and `A` must match that described in `problem_structure`.
        - `P` should contain the full symmetric matrix (not just upper triangular).
        - All arrays should be on the device (GPU) and compatible with JAX operations.
        """
        self.P = ObjMatrixGPU(P)
        self.A, self.q, self.b = P, A, q, b
        self.x, self.y, self.s = x, y, s
        self.problem_structure = problem_structure
    
    def jvp(
        self,
        dP: Float[BCSR, "n n"],
        dA: Float[BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]:
        """Apply the derivative of the QCP's solution map to an input perturbation.
        
        Specifically, an implementation of the method given in section 3.1 of the paper.
        
        **Arguments:**
        - `dP` should have the same sparsity structure as `P`. *Note* that
            this means it should only contain the entirety of `dP`.
            (i.e., not just the upper triangular part.)
        - `dA` should have the same sparsity structure as `A`.
        - `dq`
        - `db`
    
        **Returns:**
        
        A 3-tuple containing the perturbations to the solution: `(dx, dy, ds)`.
        """
        dP = ObjMatrixGPU(dP)
        dAT = self.problem_structure.form_A_transpose(dA)
        return self._jvp_common(dP=dP, dA=dA, dAT=dAT, dq=dq, db=db)

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ) -> tuple[
        Float[BCSR, "n n"], Float[BCSR, "m n"],
        Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the QCP's solution map to a solution perturbation.
        
        Specifically, an implementation of the method given in section 3.2 of the paper.
        
        **Arguments:**
        - `dx`: A perturbation to the primal solution.
        - `dy`: A perturbation to the dual solution.
        - `ds`: A perturbation to the primal slack solution.

        **Returns**

        A four-tuple containing the perturbations to the objective matrix, constraint matrix,
        linear cost function vector, and constraint vector. Note that these perturbation matrices
        will have the same sparsity patterns as their corresponding problem matrices.
        """
        return self._vjp_common(dx=dx, dy=dy, ds=ds,
                                produce_output=_d_data_Q_adjoint_gpu)
    
        
    
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