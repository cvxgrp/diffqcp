from abc import abstractmethod
from typing import Callable
import functools as ft
import jax
from jax import eval_shape
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from lineax import AbstractLinearOperator, IdentityLinearOperator, linear_solve, LSMR
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR
import cupy as cp
from cupy import from_dlpack as cp_from_dlpack
from cupyx.scipy.sparse import csr_matrix
import nvmath
from nvmath.sparse.advanced import DirectSolverOptions

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
    AT: eqx.AbstractVar[BCSR | BCOO]
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
        AT = self.AT
        # NOTE(quill): seems hard to avoid the `DzQ` bit of the variable name.
        # NOTE(quill): Note that we're skipping the step of extracting the first n components of
        #   `pi_z` and just using `P @ pi_z[:n] = P @ x`. 
        DzQ_pi_z = _DuQ(P=self.P, Px=Px, xTPx=xTPx, A=self.A, AT=AT, q=self.q,
                        b=self.b, x=self.x, tau=jnp.array(1.0, dtype=self.x.dtype),
                        n=self.problem_structure.n, m=self.problem_structure.m)
        # NOTE(quill): we use that z_N (as defined in paper) is always 1.0, thus don't
        #   include that division.
        F = (DzQ_pi_z @ dpi_z) - dpi_z + IdentityLinearOperator(eval_shape(lambda: pi_z))

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
            soln = linear_solve(F, -d_data_N, solver=LSMR(rtol=1e-8, atol=1e-8))
            return soln.value

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
                              dproj_kstar_v.mv(dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])]
                              )
        
        def zero_case():
            return jnp.zeros_like(dz)
        
        def nonzero_case():
            # TODO(quill): start solver from previous spot?
            #   => (so would need previous `d_data_N`)
            soln = linear_solve(F.T, -dz, solver=LSMR(rtol=1e-8, atol=1e-8))
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
                              w1=d_data_N_n, w2=d_data_N_m, w3=d_data_N_N)
        
    
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
    AT: Float[BCOO, "n m"]
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
        self.AT = A.T
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
        
        partial_d_data_Q_adjoint_cpu = ft.partial(_d_data_Q_adjoint_cpu,
                                                  P_rows=self.problem_structure.P_nonzero_rows,
                                                  P_cols=self.problem_structure.P_nonzero_cols,
                                                  A_rows=self.problem_structure.A_nonzero_rows,
                                                  A_cols=self.problem_structure.A_nonzero_cols,
                                                  n=self.problem_structure.n,
                                                  m=self.problem_structure.m)
        
        return self._vjp_common(dx=dx, dy=dy, ds=ds,
                                produce_output=partial_d_data_Q_adjoint_cpu)


class DeviceQCP(AbstractQCP):
    """QCP whose subroutines are optimized to run on device (GPU).
    """
    # NOTE(quill): when we allow for batched problem data, will need
    #   to wrap `P` in an `AbstractLinearOperator` to dictate how the `mv`
    #   operation should behave.
    P: ObjMatrixGPU
    A: Float[BCSR, "m n"]
    AT: Float[BCSR, "n m"]
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
        self.problem_structure = problem_structure
        self.P = ObjMatrixGPU(P)
        self.A, self.q, self.b = A, q, b
        self.AT = self.problem_structure.form_A_transpose(self.A)
        self.x, self.y, self.s = x, y, s
    
    @eqx.filter_jit
    def _jvp_direct_solve_get_F(self, F) -> Float[Array, "N N"]:
        """For prototyping purposes; obviously not efficient.
        """

        def _get_dense_mat(mat: lx.AbstractLinearOperator):
            mv = lambda vec: mat.mv(vec)
            mm = jax.vmap(mv, in_axes=1, out_axes=1)
            return mm(jnp.eye(self.problem_structure.N))
        
        return _get_dense_mat(F)
    
    @eqx.filter_jit
    def _jvp_first_block(
        self,
        dP: ObjMatrix,
        dA: Float[BCSR, "m n"],
        dAT: Float[BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ) -> tuple[Float[Array, " N"], AbstractLinearOperator, AbstractLinearOperator]:
        n = self.problem_structure.n
        m = self.problem_structure.m
        pi_z, F, dproj_k_star_v = self._form_atoms()
        pi_z_n, pi_z_m, pi_z_N = pi_z[:n], pi_z[n:n+m], pi_z[-1]
        d_data_N = _d_data_Q(x=pi_z_n, y=pi_z_m, tau=pi_z_N, dP=dP,
                             dA=dA, dAT=dAT, dq=dq, db=db)
        
        return -d_data_N, F, dproj_k_star_v
    
    def _jvp_actual_solve(self, F, d_data_N_minus):
        # NOTE(quill): separating this out for timing purposes.
        return nvmath.sparse.advanced.direct_solver(F, d_data_N_minus)
    
    def _jvp_nvmath_direct_solve(self, F, d_data_N_minus):
        # F_cupy_csr = csr_matrix(cp_from_dlpack(F))
        # d_data_N_minus_cupy = cp_from_dlpack(d_data_N_minus)
        # dz_cupy = self._jvp_actual_solve(F_cupy_csr, d_data_N_minus_cupy)
        # dz = jax.dlpack.from_dlpack(dz_cupy)
        # return dz
        # soln = linear_solve(lx.MatrixLinearOperator(F), d_data_N_minus, solver=LSMR(rtol=1e-8, atol=1e-8))
        # return soln.value
        soln = linear_solve(lx.MatrixLinearOperator(F), d_data_N_minus)
        return soln.value
    
    @eqx.filter_jit
    def _jvp_direct_solve_get_output(self, dz, dproj_kstar_v):
        n = self.problem_structure.n
        m = self.problem_structure.m

        dz_n, dz_m, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dx = dz_n - self.x * dz_N
        dproj_k_star_v_dz_m = dproj_kstar_v.mv(dz_m)
        dy = dproj_k_star_v_dz_m - self.y * dz_N
        ds = dproj_k_star_v_dz_m - dz_m - self.s * dz_N
        return dx, dy, ds
    
    def _jvp_direct_solve(
        self,
        dP: ObjMatrix,
        dA: Float[BCSR, "m n"],
        dAT: Float[BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"]
    ):
        d_data_N_minus, F, dproj_k_star_v = self._jvp_first_block(dP, dA, dAT, dq, db)

        # `_jvp_direct_solve` cannot be jitted, so can use regular
        # Python control flow
        # TODO(quill): use a norm tolerance instead?
        if jnp.allclose(d_data_N_minus, 0):
            return jnp.zeros_like(d_data_N_minus)
        else:
            F = self._jvp_direct_solve_get_F(F)
            dz = self._jvp_nvmath_direct_solve(F, d_data_N_minus)

        return self._jvp_direct_solve_get_output(dz, dproj_k_star_v)

    def jvp(
        self,
        dP: Float[BCSR, "n n"],
        dA: Float[BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
        use_direct_solve: bool = False
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
        dAT = eqx.filter_jit(self.problem_structure.form_A_transpose)(dA)
        if use_direct_solve:
            return self._jvp_direct_solve(dP=dP, dA=dA, dAT=dAT, dq=dq, db=db)
        else:
            return self._jvp_common(dP=dP, dA=dA, dAT=dAT, dq=dq, db=db)

    @eqx.filter_jit
    def _vjp_direct_solve_block1(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ):
        pi_z, F, dproj_kstar_v = self._form_atoms()
        dz = jnp.concatenate([dx,
                              dproj_kstar_v.mv(dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])]
                            )
        return -dz, F, pi_z
    
    @eqx.filter_jit
    def _vjp_direct_solve_get_FT(self, F):
        """For prototyping purposes; obviously not efficient.
        """

        def _get_dense_mat(mat: lx.AbstractLinearOperator):
            mv = lambda vec: mat.mv(vec)
            mm = jax.vmap(mv, in_axes=1, out_axes=1)
            return mm(jnp.eye(self.problem_structure.N))
        
        return _get_dense_mat(F.T)
    
    def _vjp_actual_solve(self, FT, dz_minus):
        # NOTE(quill): separating this out for timing purposes.
        return nvmath.sparse.advanced.direct_solver(FT, dz_minus)
    
    def _vjp_nvmath_direct_solve(self, FT, dz_minus):
        # FT is a full JAX array
        # FT_cupy_csr = csr_matrix(cp_from_dlpack(FT))
        # dz_minus_cupy = cp_from_dlpack(dz_minus)
        # d_data_N_cupy = self._vjp_actual_solve(FT_cupy_csr, dz_minus_cupy)
        # d_data_N = jax.dlpack.from_dlpack(d_data_N_cupy)
        # return d_data_N
        soln = linear_solve(lx.MatrixLinearOperator(FT), dz_minus)
        return soln.value
        # soln = linear_solve(lx.MatrixLinearOperator(FT), dz_minus, solver=LSMR(rtol=1e-8, atol=1e-8))
        # return soln.value
    
    @eqx.filter_jit()
    def _vjp_direct_solve_get_output(
        self,
        pi_z,
        d_data_N
    ):
        n = self.problem_structure.n
        m = self.problem_structure.m
        
        pi_z_n = pi_z[:n]
        pi_z_m = pi_z[n:n+m]
        pi_z_N = pi_z[-1]
        d_data_N_n = d_data_N[:n]
        d_data_N_m = d_data_N[n:n+m]
        d_data_N_N = d_data_N[-1]

        return _d_data_Q_adjoint_gpu(
            x=pi_z_n,
            y=pi_z_m,
            tau=pi_z_N,
            w1=d_data_N_n,
            w2=d_data_N_m,
            w3=d_data_N_N,
            P_rows=self.problem_structure.P_nonzero_rows,
            P_cols=self.problem_structure.P_nonzero_cols,
            P_csr_indices=self.problem_structure.P_csr_indices,
            P_csr_indtpr=self.problem_structure.P_csr_indptr,
            A_rows=self.problem_structure.A_nonzero_rows,
            A_cols=self.problem_structure.A_nonzero_cols,
            A_csr_indices=self.problem_structure.A_csr_indices,
            A_csr_indtpr=self.problem_structure.A_csr_indptr,
            n=n,
            m=m
        )

    def _vjp_direct_solve(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"]
    ):
        dz_minus, F, pi_z = self._vjp_direct_solve_block1(dx, dy, ds)

        # now check if 0 or not. This will not be jitted, so we can 
        # just use typical Python control flow
        # if jnp.allclose(dz_minus, 0):
        #     return jnp.zeros_like(dz_minus)
        # else:
        #     # obtain FT
        #     FT = self._vjp_direct_solve_get_FT(F)
        #     d_data_N = self._vjp_nvmath_direct_solve(FT, dz_minus)

        def zero_case():
            return jnp.zeros_like(dz_minus)
        
        def nonzero_case():
            FT = self._vjp_direct_solve_get_FT(F)
            d_data_N = self._vjp_nvmath_direct_solve(FT, dz_minus)
            return d_data_N

        d_data_N = jax.lax.cond(jnp.allclose(dz_minus, 0),
                                zero_case,
                                nonzero_case)
        
        return self._vjp_direct_solve_get_output(pi_z, d_data_N)

    
    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        use_direct_solve: bool = False
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
        
        if use_direct_solve:
            return self._vjp_direct_solve(dx=dx, dy=dy, ds=ds)
        else:
            partial_d_data_Q_adjoint_gpu = ft.partial(_d_data_Q_adjoint_gpu,
                                                  P_rows=self.problem_structure.P_nonzero_rows,
                                                  P_cols=self.problem_structure.P_nonzero_cols,
                                                  P_csr_indices=self.problem_structure.P_csr_indices,
                                                  P_csr_indtpr=self.problem_structure.P_csr_indptr,
                                                  A_rows=self.problem_structure.A_nonzero_rows,
                                                  A_cols=self.problem_structure.A_nonzero_cols,
                                                  A_csr_indices=self.problem_structure.A_csr_indices,
                                                  A_csr_indtpr=self.problem_structure.A_csr_indptr,
                                                  n=self.problem_structure.n,
                                                  m=self.problem_structure.m)
            
            return self._vjp_common(dx=dx, dy=dy, ds=ds, produce_output=partial_d_data_Q_adjoint_gpu)