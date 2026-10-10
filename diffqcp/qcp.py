"""The derivative of a quadratic cone program's solution map: JVPs and VJPs.

`HostQCP` (CPU; P stored as its upper triangle, BCOO) and `DeviceQCP` (GPU;
full P, BCSR) hold a solved QCP and apply the derivative of its solution map
(section 3.1 of the paper) and its adjoint (section 3.2). Both share the
algorithm in `AbstractQCP`; they differ only in the sparse formats of the data
and of the returned data perturbations.

Notation follows the paper: `z = (x, y - s, 1)` is the solution embedding,
`Pi z = (x, Pi_{K^*}(y - s), 1)` its projection, and

    F = D_u Q(Pi z) DPi(z) - DPi(z) + I

the derivative of the normalized residual map at `z` (`z_N = 1`, so no
division is needed). F is singular, with null vector `z`, so both products are
computed from the gauge-fixed system `F' = F[:, :N-1]`; see `diffqcp.solvers`.
"""
from __future__ import annotations

import functools as ft
from abc import abstractmethod
from collections.abc import Callable
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import eval_shape
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator, IdentityLinearOperator

try:  # Optional GPU dependencies, only needed for `solve_method="nvmath-direct"`.
    from cupy import from_dlpack as cp_from_dlpack  # pyright: ignore[reportMissingImports]
    from cupyx.scipy.sparse import csr_matrix  # pyright: ignore[reportMissingImports]
    from nvmath.sparse.advanced import DirectSolver, DirectSolverAlgType  # pyright: ignore[reportMissingImports]
except ImportError:
    cp_from_dlpack = None
    csr_matrix = None
    DirectSolver = None
    DirectSolverAlgType = None

from diffqcp.linops import _BlockLinearOperator
from diffqcp.problem_data import (
    ObjMatrix,
    ObjMatrixCPU,
    ObjMatrixGPU,
    QCPStructure,
    QCPStructureCPU,
    QCPStructureGPU,
)
from diffqcp.qcp_derivs import _d_data_Q, _d_data_Q_adjoint_cpu, _d_data_Q_adjoint_gpu, _DuQ
from diffqcp.solvers import (
    AbstractDerivativeSolver,
    augmented_system,
    gauge_fixed,
    materialise_gauge_operator,
    resolve_solver,
)

SolutionPerturbation = tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]
"""`(dx, dy, ds)`: a perturbation of the primal, dual and slack solution."""


class AbstractQCP(eqx.Module):
    """A solved quadratic cone program, and the derivative of its solution map.

    The QCP is the primal-dual pair

        (P) minimize    (1/2) x^T P x + q^T x
            subject to  A x + s = b
                        s in K

        (D) maximize    -(1/2) x^T P x - b^T y
            subject to  P x + A^T y = -q
                        y in K^*,

    with data `(P, A, q, b)` (P symmetric positive semidefinite), a fixed
    cone K (a product of the cones in `diffqcp.cones`) with dual cone K^*,
    and a primal-dual solution `(x, y, s)`.

    The solution map takes the data to the solution. `jvp` applies its
    derivative to a data perturbation and `vjp` applies the adjoint to a
    solution perturbation. The sparsity patterns of P and A, the cone and the
    problem sizes come from `problem_structure`.
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

    def _form_atoms(self) -> tuple[Float[Array, " N"], AbstractLinearOperator, AbstractLinearOperator]:
        """The pieces shared by the JVP and VJP.

        **Returns:**

        `(pi_z, F, dproj_kstar_v)`: the projected solution embedding
        `Pi z = (x, Pi_{K^*}(y - s), 1)`, the N x N operator
        `F = D_u Q(Pi z) DPi(z) - DPi(z) + I`, and the Jacobian of the
        projection onto K^* at `v = y - s`.
        """
        proj_kstar_v, dproj_kstar_v = self.problem_structure.cone_projector(self.y - self.s)
        pi_z = jnp.concatenate([self.x, proj_kstar_v, jnp.array([1.0], dtype=self.x.dtype)])
        dpi_z = _BlockLinearOperator([IdentityLinearOperator(eval_shape(lambda: self.x)),
                                      dproj_kstar_v,
                                      IdentityLinearOperator(eval_shape(lambda: jnp.array([1.0])))])
        Px = self.P.mv(self.x)
        # D_u Q at Pi z. The x-block of Pi z is x itself, so `P @ x` is reused.
        DuQ_pi_z = _DuQ(P=self.P, Px=Px, xTPx=self.x @ Px, A=self.A, AT=self.AT, q=self.q,
                        b=self.b, x=self.x, tau=jnp.array(1.0, dtype=self.x.dtype),
                        n=self.problem_structure.n, m=self.problem_structure.m)
        F = (DuQ_pi_z @ dpi_z) - dpi_z + IdentityLinearOperator(eval_shape(lambda: pi_z))
        return pi_z, F, dproj_kstar_v

    def _jvp_common(
        self,
        dP: ObjMatrix,
        dA: Float[BCOO | BCSR, "m n"],
        dAT: Float[BCOO | BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
        solver: AbstractDerivativeSolver,
    ) -> SolutionPerturbation:
        """The JVP (paper section 3.1) with the gauge-fixed solve.

        Solves `F' d = -D_data Q(Pi z)[dP, dA, dq, db]` for `d`, sets
        `dz = (d, 0)`, and maps `dz` to `(dx, dy, ds)` with `Dphi(z)`.
        `dP` is the full symmetric perturbation, wrapped as an `ObjMatrix`.
        """
        pi_z, F, dproj_kstar_v = self._form_atoms()

        n, m = self.problem_structure.n, self.problem_structure.m
        pi_z_n, pi_z_m, pi_z_N = pi_z[:n], pi_z[n:n+m], pi_z[-1]
        d_data_N = _d_data_Q(x=pi_z_n, y=pi_z_m, tau=pi_z_N, dP=dP,
                             dA=dA, dAT=dAT, dq=dq, db=db)

        d = jax.lax.cond(jnp.all(d_data_N == 0),
                         lambda: jnp.zeros(self.problem_structure.N - 1, dtype=d_data_N.dtype),
                         lambda: solver.solve(gauge_fixed(F), -d_data_N))

        # dz = (d, 0), so the `- x dz_N` style terms of Dphi(z) dz vanish.
        dz_n, dz_m = d[:n], d[n:n+m]
        dproj_kstar_v_dz_m = dproj_kstar_v.mv(dz_m)
        dx = dz_n
        dy = dproj_kstar_v_dz_m
        ds = dproj_kstar_v_dz_m - dz_m
        return dx, dy, ds

    def _vjp_common(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        produce_output: Callable[..., Any],
        solver: AbstractDerivativeSolver,
    ) -> Any:
        """The VJP (paper section 3.2) with the gauge-fixed solve.

        Forms `dz = Dphi(z)^T (dx, dy, ds)`, takes the minimum-norm solution
        `w` of `F'^T w = -dz[:N-1]` (which also solves `F^T w = -dz`, since
        `dz` is orthogonal to the null vector `z` of F), and passes `Pi z` and
        `w` to `produce_output`, which applies the adjoint of
        `D_data Q(Pi z)` and builds the data perturbations in the caller's
        sparse format.
        """
        n, m = self.problem_structure.n, self.problem_structure.m
        pi_z, F, dproj_kstar_v = self._form_atoms()
        dz = jnp.concatenate([dx,
                              dproj_kstar_v.mv(dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])])

        w = jax.lax.cond(jnp.all(dz == 0),
                         lambda: jnp.zeros_like(dz),
                         lambda: solver.solve_transpose(gauge_fixed(F), -dz[:-1]))

        return produce_output(x=pi_z[:n], y=pi_z[n:n+m], tau=pi_z[-1],
                              w1=w[:n], w2=w[n:n+m], w3=w[-1])

    @abstractmethod
    def jvp(
        self,
        dP: Float[BCOO | BCSR, "n n"],
        dA: Float[BCOO | BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> SolutionPerturbation:
        """Apply the derivative of the solution map to a data perturbation.

        See the concrete classes for the sparse format `dP` and `dA` must use.
        """
        raise NotImplementedError

    @abstractmethod
    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> tuple[Float[BCOO | BCSR, "n n"], Float[BCOO | BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the solution map to a solution perturbation.

        See the concrete classes for the sparse format of the returned `dP` and `dA`.
        """
        raise NotImplementedError


class HostQCP(AbstractQCP):
    """A solved QCP whose derivative routines are laid out for the CPU.

    P is stored as its upper triangle and the sparse matrices are BCOO.

    **Arguments:**
    - `P`: BCOO, shape (n, n). The upper triangle (diagonal included) of the
        symmetric objective matrix.
    - `A`: BCOO, shape (m, n). The constraint matrix.
    - `q`: shape (n,). The linear objective.
    - `b`: shape (m,). The constraint offset.
    - `x`: shape (n,). The primal solution.
    - `y`: shape (m,). The dual solution.
    - `s`: shape (m,). The primal slack.
    - `problem_structure`: the `QCPStructureCPU` built from the sparsity
        patterns of `P` and `A` and the cone. `P` and `A` must have exactly
        those patterns (including the order of the stored entries).
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
        problem_structure: QCPStructureCPU,
    ):
        self.A, self.q, self.b = A, q, b
        self.AT = cast(BCOO, A.T)  # (`.T` is untyped in jax's stubs)
        self.x, self.y, self.s = x, y, s
        self.problem_structure = problem_structure
        self.P = self.problem_structure.form_obj(P)

    def jvp(
        self,
        dP: Float[BCOO, "n n"],
        dA: Float[BCOO, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> SolutionPerturbation:
        """Apply the derivative of the solution map to a data perturbation (paper section 3.1).

        **Arguments:**
        - `dP`: BCOO with the sparsity pattern of `P`, so only the upper
            triangle of the (symmetric) perturbation of P.
        - `dA`: BCOO with the sparsity pattern of `A`.
        - `dq`: shape (n,).
        - `db`: shape (m,).
        - `solve_method`: `"jax-lsmr"` (default, `LSMRSolver()`) or `"jax-lu"`
            (`DenseDirectSolver()`). Ignored when `solver` is given.
        - `solver`: an `AbstractDerivativeSolver` for the derivative system.

        **Returns:**

        `(dx, dy, ds)`, the first-order change in the solution.
        """
        return self._jvp_common(dP=self.problem_structure.form_obj(dP), dA=dA, dAT=cast(BCOO, dA.T), dq=dq, db=db,
                                solver=resolve_solver(solve_method, solver))

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> tuple[Float[BCOO, "n n"], Float[BCOO, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the solution map (paper section 3.2).

        **Arguments:**
        - `dx`: shape (n,). Perturbation (e.g. loss gradient) of the primal solution.
        - `dy`: shape (m,). Perturbation of the dual solution.
        - `ds`: shape (m,). Perturbation of the primal slack.
        - `solve_method`, `solver`: as in `jvp`.

        **Returns:**

        `(dP, dA, dq, db)`. `dP` and `dA` are BCOO with the sparsity patterns
        of `P` and `A`. `dP` is the gradient with respect to the *stored*
        upper-triangle values of P, so an off-diagonal entry accounts for both
        P_ij and P_ji.
        """
        produce_output = ft.partial(_d_data_Q_adjoint_cpu,
                                    P_rows=self.problem_structure.P_nonzero_rows,
                                    P_cols=self.problem_structure.P_nonzero_cols,
                                    A_rows=self.problem_structure.A_nonzero_rows,
                                    A_cols=self.problem_structure.A_nonzero_cols,
                                    n=self.problem_structure.n,
                                    m=self.problem_structure.m)
        return self._vjp_common(dx=dx, dy=dy, ds=ds, produce_output=produce_output,
                                solver=resolve_solver(solve_method, solver))


class DeviceQCP(AbstractQCP):
    """A solved QCP whose derivative routines are laid out for the GPU.

    P is stored in full and the sparse matrices are BCSR (whose matrix-vector
    products have CUDA kernels in JAX).

    **Arguments:**
    - `P`: BCSR, shape (n, n). The full symmetric objective matrix.
    - `A`: BCSR, shape (m, n). The constraint matrix.
    - `q`: shape (n,). The linear objective.
    - `b`: shape (m,). The constraint offset.
    - `x`: shape (n,). The primal solution.
    - `y`: shape (m,). The dual solution.
    - `s`: shape (m,). The primal slack.
    - `problem_structure`: the `QCPStructureGPU` built from the sparsity
        patterns of `P` and `A` and the cone. `P` and `A` must have exactly
        those patterns.
    """

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
        problem_structure: QCPStructureGPU,
    ):
        self.problem_structure = problem_structure
        self.P = ObjMatrixGPU(P)
        self.A, self.q, self.b = A, q, b
        self.AT = self.problem_structure.form_A_transpose(self.A)
        self.x, self.y, self.s = x, y, s

    def jvp(
        self,
        dP: Float[BCSR, "n n"],
        dA: Float[BCSR, "m n"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> SolutionPerturbation:
        """Apply the derivative of the solution map to a data perturbation (paper section 3.1).

        **Arguments:**
        - `dP`: BCSR with the sparsity pattern of `P`: the full symmetric
            perturbation, not just its upper triangle.
        - `dA`: BCSR with the sparsity pattern of `A`.
        - `dq`: shape (n,).
        - `db`: shape (m,).
        - `solve_method`: `"jax-lsmr"` (default, `LSMRSolver()`), `"jax-lu"`
            (`DenseDirectSolver()`), or `"nvmath-direct"` (a cuDSS factorization
            of the dense augmented system; needs `nvmath-python` and CuPy, and
            is not jittable). `solver` takes precedence over the first two.
        - `solver`: an `AbstractDerivativeSolver` for the derivative system.

        **Returns:**

        `(dx, dy, ds)`, the first-order change in the solution.
        """
        dP_op = ObjMatrixGPU(dP)
        dAT = eqx.filter_jit(self.problem_structure.form_A_transpose)(dA)
        if solve_method == "nvmath-direct":
            return self._jvp_nvmath(dP=dP_op, dA=dA, dAT=dAT, dq=dq, db=db)
        return self._jvp_common(dP=dP_op, dA=dA, dAT=dAT, dq=dq, db=db,
                                solver=resolve_solver(solve_method, solver))

    def vjp(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
        solve_method: str = "jax-lsmr",
        solver: AbstractDerivativeSolver | None = None,
    ) -> tuple[Float[BCSR, "n n"], Float[BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """Apply the adjoint of the derivative of the solution map (paper section 3.2).

        **Arguments:**
        - `dx`: shape (n,). Perturbation (e.g. loss gradient) of the primal solution.
        - `dy`: shape (m,). Perturbation of the dual solution.
        - `ds`: shape (m,). Perturbation of the primal slack.
        - `solve_method`, `solver`: as in `jvp`.

        **Returns:**

        `(dP, dA, dq, db)`. `dP` and `dA` are BCSR with the sparsity patterns
        of `P` and `A`. Each stored entry of `dP` is the gradient with respect
        to that entry of the symmetric P (P_ij and P_ji get the same value).
        """
        if solve_method == "nvmath-direct":
            return self._vjp_nvmath(dx=dx, dy=dy, ds=ds)
        return self._vjp_common(dx=dx, dy=dy, ds=ds, produce_output=self._adjoint_output,
                                solver=resolve_solver(solve_method, solver))

    def _adjoint_output(self, **kwargs: Any) -> tuple[
            Float[BCSR, "n n"], Float[BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """`_d_data_Q_adjoint_gpu` with this problem's sparsity patterns filled in."""
        st = self.problem_structure
        return _d_data_Q_adjoint_gpu(
            **kwargs,
            P_rows=st.P_nonzero_rows, P_cols=st.P_nonzero_cols,
            P_csr_indices=st.P_csr_indices, P_csr_indtpr=st.P_csr_indptr,
            A_rows=st.A_nonzero_rows, A_cols=st.A_nonzero_cols,
            A_csr_indices=st.A_csr_indices, A_csr_indtpr=st.A_csr_indptr,
            n=st.n, m=st.m,
        )

    # ─── "nvmath-direct": cuDSS on the dense augmented system ──────────────
    # Not exercised by CI (needs a GPU with nvmath-python and CuPy). The solve
    # runs on the host side of JAX, so these paths are not jittable; the jitted
    # pieces before and after the solve are split out.

    @eqx.filter_jit
    def _jvp_nvmath_form_atoms(
        self,
        dP: ObjMatrix,
        dA: Float[BCSR, "m n"],
        dAT: Float[BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
    ) -> tuple[Float[Array, " N"], AbstractLinearOperator, AbstractLinearOperator]:
        """`(-D_data Q(Pi z)[dP, dA, dq, db], F, dproj_kstar_v)`."""
        n, m = self.problem_structure.n, self.problem_structure.m
        pi_z, F, dproj_kstar_v = self._form_atoms()
        d_data_N = _d_data_Q(x=pi_z[:n], y=pi_z[n:n+m], tau=pi_z[-1], dP=dP,
                             dA=dA, dAT=dAT, dq=dq, db=db)
        return -d_data_N, F, dproj_kstar_v

    @eqx.filter_jit
    def _jvp_nvmath_get_output(
        self, dz: Float[Array, " N"], dproj_kstar_v: AbstractLinearOperator
    ) -> SolutionPerturbation:
        """`Dphi(z) dz`, split into `(dx, dy, ds)`."""
        n, m = self.problem_structure.n, self.problem_structure.m
        dz_n, dz_m, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dproj_kstar_v_dz_m = dproj_kstar_v.mv(dz_m)
        dx = dz_n - self.x * dz_N
        dy = dproj_kstar_v_dz_m - self.y * dz_N
        ds = dproj_kstar_v_dz_m - dz_m - self.s * dz_N
        return dx, dy, ds

    def _jvp_nvmath(
        self,
        dP: ObjMatrix,
        dA: Float[BCSR, "m n"],
        dAT: Float[BCSR, "n m"],
        dq: Float[Array, " n"],
        db: Float[Array, " m"],
    ) -> SolutionPerturbation:
        """JVP with the least-squares `d` from `K [res; d] = [r; 0]` (see `augmented_system`)."""
        _check_nvmath()
        r, F, dproj_kstar_v = self._jvp_nvmath_form_atoms(dP, dA, dAT, dq, db)
        N = self.problem_structure.N
        if jnp.all(r == 0):
            dz = jnp.zeros_like(r)
        else:
            K = _augmented_from_F(F)
            rhs = jnp.concatenate([r, jnp.zeros(N - 1, dtype=r.dtype)])
            sol = _cudss_solve(K, rhs)
            dz = jnp.concatenate([sol[N:], jnp.zeros(1, dtype=sol.dtype)])
        return self._jvp_nvmath_get_output(dz, dproj_kstar_v)

    @eqx.filter_jit
    def _vjp_nvmath_form_atoms(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
    ) -> tuple[Float[Array, " N"], AbstractLinearOperator, Float[Array, " N"]]:
        """`(-dz, F, Pi z)` with `dz = Dphi(z)^T (dx, dy, ds)`."""
        pi_z, F, dproj_kstar_v = self._form_atoms()
        dz = jnp.concatenate([dx,
                              dproj_kstar_v.mv(dy + ds) - ds,
                              - jnp.array([self.x @ dx + self.y @ dy + self.s @ ds])])
        return -dz, F, pi_z

    @eqx.filter_jit
    def _vjp_nvmath_get_output(
        self, pi_z: Float[Array, " N"], w: Float[Array, " N"]
    ) -> tuple[Float[BCSR, "n n"], Float[BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """The adjoint of `D_data Q(Pi z)` applied to `w`, as `(dP, dA, dq, db)`."""
        n, m = self.problem_structure.n, self.problem_structure.m
        return self._adjoint_output(x=pi_z[:n], y=pi_z[n:n+m], tau=pi_z[-1],
                                    w1=w[:n], w2=w[n:n+m], w3=w[-1])

    def _vjp_nvmath(
        self,
        dx: Float[Array, " n"],
        dy: Float[Array, " m"],
        ds: Float[Array, " m"],
    ) -> tuple[Float[BCSR, "n n"], Float[BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]]:
        """VJP with the minimum-norm `w` from `K [w; t] = [0; -dz[:N-1]]` (see `augmented_system`)."""
        _check_nvmath()
        minus_dz, F, pi_z = self._vjp_nvmath_form_atoms(dx, dy, ds)
        N = self.problem_structure.N
        if jnp.all(minus_dz == 0):
            w = jnp.zeros_like(minus_dz)
        else:
            K = _augmented_from_F(F)
            rhs = jnp.concatenate([jnp.zeros(N, dtype=minus_dz.dtype), minus_dz[:-1]])
            w = _cudss_solve(K, rhs)[:N]
        return self._vjp_nvmath_get_output(pi_z, w)


@eqx.filter_jit
def _augmented_from_F(F: AbstractLinearOperator) -> Float[Array, "2N-1 2N-1"]:
    """Dense augmented matrix of the gauge-fixed F' (see `diffqcp.solvers.augmented_system`)."""
    return augmented_system(materialise_gauge_operator(gauge_fixed(F)))


def _check_nvmath() -> None:
    if DirectSolver is None:
        raise ValueError("The `nvmath-direct` option can only be used when `nvmath-python` "
                         "is installed. Also check that CuPy is installed.")


def _cudss_solve(K: Float[Array, "k k"], rhs: Float[Array, " k"]) -> Float[Array, " k"]:
    """Solve `K sol = rhs` with nvmath's cuDSS direct solver (zero-copy via DLPack)."""
    assert DirectSolver is not None and DirectSolverAlgType is not None  # checked by `_check_nvmath`
    assert csr_matrix is not None and cp_from_dlpack is not None
    with DirectSolver(csr_matrix(cp_from_dlpack(K)), cp_from_dlpack(rhs)) as cudss:
        cudss.plan_config.reordering_algorithm = DirectSolverAlgType.ALG_1
        cudss.plan()
        cudss.factorize()
        sol = cudss.solve()
    return jax.dlpack.from_dlpack(sol)
