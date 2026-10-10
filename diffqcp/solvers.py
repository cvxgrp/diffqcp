"""Linear solvers for the derivative system of the QCP solution map.

The JVP and VJP both need a solve with the N x N matrix (N = n + m + 1)

    F = DQ(Pi z) DPi(z) - DPi(z) + I,

the derivative of the normalized residual map at the solution embedding
`z = (x, y - s, 1)`. **F is always singular**: the homogeneous embedding is
positively homogeneous of degree one, so `F z = 0`. LSMR copes (it returns
a least-squares solution), but a direct factorization of F breaks down (this
was the source of the exploding gradients from the "jax-lu" and nvmath paths).

The null direction is harmless because nothing downstream sees it:

- JVP: the map `dz -> (dx, dy, ds)` sends `z` to zero, so any solution of
  `F dz = r` gives the same answer.
- VJP: the right-hand side is orthogonal to `z`, so `F^T w = g` is consistent,
  and any two solutions differ by a vector the data adjoint sends to zero.

We therefore **fix the gauge** `dz_N = 0`: with E the embedding that appends a
zero, `F' = F E` is N x (N - 1) and (away from degenerate points) has full
column rank. The solver contract is:

- `solve(Fg, r)`: the least-squares solution `d = argmin ||F' d - r||`
  (exact when `r` is in range(F'), as it is for the JVP),
- `solve_transpose(Fg, g)`: the minimum-norm solution of `F'^T w = g`,

where `Fg` is `F'` as a `lineax` operator. Both are well posed when F' has
full column rank. At a degenerate point (e.g. no strict complementarity) F'
loses rank, the solution map is not differentiable, and the solvers return
the least-squares/minimum-norm answer.
"""
from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.linalg
import lineax as lx
from jaxtyping import Array, Bool, Float


class AbstractDerivativeSolver(eqx.Module):
    """Strategy for solving the gauge-fixed derivative system. See module docs."""

    @abstractmethod
    def solve(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N"]) -> Float[Array, " N-1"]:
        """Least-squares solution of `F' d = rhs` (used by the JVP)."""
        raise NotImplementedError

    @abstractmethod
    def solve_transpose(
        self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N-1"]
    ) -> Float[Array, " N"]:
        """Minimum-norm solution of `F'^T w = rhs` (used by the VJP)."""
        raise NotImplementedError


def _default_tol(dtype: jnp.dtype) -> float:
    """Default LSMR tolerance: 1e-12 in float64, 1e-6 otherwise."""
    # LSMR's stopping test is relative to ||F'|| ||d||; 1e-8 left 1e-4..1e-3
    # relative error in the derivative on moderately conditioned problems,
    # while 1e-12 costs only ~5-10% more iterations (see STATUS.md, Wave 5).
    return 1e-12 if jnp.dtype(dtype) == jnp.float64 else 1e-6


class LSMRSolver(AbstractDerivativeSolver):
    """Matrix-free LSMR (the default). Never materializes F.

    **Arguments:**
    - `rtol`, `atol`: LSMR stopping tolerances. `None` picks a default from the
      dtype: `1e-12` for float64, `1e-6` for float32.
    - `max_steps`: iteration cap; `None` uses lineax's default (10 x the
      smaller dimension). Not converging within it raises at runtime.
    """

    rtol: float | None = None
    atol: float | None = None
    max_steps: int | None = None

    def _lsmr(self, dtype: jnp.dtype) -> lx.LSMR:
        default = _default_tol(dtype)
        rtol = default if self.rtol is None else self.rtol
        atol = default if self.atol is None else self.atol
        # conlim=inf: F' is well conditioned away from degenerate points, and the
        # condition-number stop is not what we want to end the iteration on.
        return lx.LSMR(rtol=rtol, atol=atol, max_steps=self.max_steps, conlim=float("inf"))

    def solve(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N"]) -> Float[Array, " N-1"]:
        return lx.linear_solve(Fg, rhs, solver=self._lsmr(rhs.dtype)).value

    def solve_transpose(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N-1"]) -> Float[Array, " N"]:
        return lx.linear_solve(Fg.T, rhs, solver=self._lsmr(rhs.dtype)).value


def materialise_gauge_operator(Fg: lx.AbstractLinearOperator) -> Float[Array, "N N-1"]:
    """Dense N x (N - 1) matrix of `F'`, one matrix-vector product per column."""
    num_cols = Fg.in_size()
    dtype = Fg.in_structure().dtype
    return jax.vmap(Fg.mv, in_axes=1, out_axes=1)(jnp.eye(num_cols, dtype=dtype))


def augmented_system(Fp: Float[Array, "N N-1"]) -> Float[Array, "2N-1 2N-1"]:
    """The symmetric, nonsingular (when F' has full column rank) matrix

        K = [[I,    F'],
             [F'^T, 0 ]].

    - JVP: `K [res; d] = [r; 0]` gives the least-squares `d` (and `res = r - F' d`).
    - VJP: `K [w; t] = [0; g]` gives the minimum-norm `w` with `F'^T w = g`.

    One factorization of K serves both, and it is the form a sparse
    symmetric-indefinite direct solver (LDL^T, e.g. cuDSS) would factor. With
    sigma_i the singular values of F', the eigenvalues of K are 1 and
    (1 +- sqrt(1 + 4 sigma_i^2)) / 2, so cond(K) ~ max(sigma_1, 1) / sigma_min^2:
    between cond(F') and cond(F')^2, better than the normal equations when
    sigma_1 > 1. Scaling the identity block by ~sigma_min would bring it to
    ~sqrt(2) cond(F') (Bjorck); not done yet.
    """
    N, Nm1 = Fp.shape
    top = jnp.concatenate([jnp.eye(N, dtype=Fp.dtype), Fp], axis=1)
    bottom = jnp.concatenate([Fp.T, jnp.zeros((Nm1, Nm1), dtype=Fp.dtype)], axis=1)
    return jnp.concatenate([top, bottom], axis=0)


class DenseDirectSolver(AbstractDerivativeSolver):
    """Materializes F' and solves the augmented system with a dense LU.

    For small problems and as an accuracy reference; cost is O(N^3) time and
    O(N^2) memory.

    At a degenerate point F' is rank deficient and K is singular. The system
    is then still consistent, so LU can return *a* solution with a tiny
    residual that is nonetheless not the least-squares/minimum-norm one LSMR
    converges to; a residual check alone does not catch this. We therefore
    fall back to an SVD-based least-squares solve when either the smallest LU
    pivot is below `pivot_rtol` times the largest (default: 1000 machine
    epsilons; LU with partial pivoting is not rank revealing in general, but
    exact singularity shows up as an ~epsilon pivot), or the relative residual
    exceeds `residual_rtol`.
    """

    pivot_rtol: float | None = None
    residual_rtol: float = 1e-8

    def _solve_augmented(
        self, Fp: Float[Array, "N N-1"], top_rhs: Float[Array, " N"], bottom_rhs: Float[Array, " N-1"]
    ) -> tuple[Float[Array, " N"], Float[Array, " N-1"], Bool[Array, ""]]:
        """LU-solve `K [top; bottom] = [top_rhs; bottom_rhs]`; also report whether the pivots look nonsingular."""
        K = augmented_system(Fp)
        lu, piv = jax.scipy.linalg.lu_factor(K)
        sol = jax.scipy.linalg.lu_solve((lu, piv), jnp.concatenate([top_rhs, bottom_rhs]))
        pivots = jnp.abs(jnp.diag(lu))
        pivot_rtol = 1000 * jnp.finfo(Fp.dtype).eps if self.pivot_rtol is None else self.pivot_rtol
        well_posed = jnp.min(pivots) > pivot_rtol * jnp.max(pivots)
        N = Fp.shape[0]
        return sol[:N], sol[N:], well_posed

    def _accept(
        self, well_posed: Bool[Array, ""], residual: Float[Array, ""], rhs: Float[Array, " k"]
    ) -> Bool[Array, ""]:
        """Keep the LU solution if the pivots are fine and the relative residual is small."""
        rel_res = residual / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
        return well_posed & jnp.isfinite(rel_res) & (rel_res <= self.residual_rtol)

    def solve(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N"]) -> Float[Array, " N-1"]:
        Fp = materialise_gauge_operator(Fg)
        _, d, well_posed = self._solve_augmented(Fp, rhs, jnp.zeros(Fp.shape[1], dtype=rhs.dtype))
        ok = self._accept(well_posed, jnp.linalg.norm(Fp @ d - rhs), rhs)
        return jax.lax.cond(ok, lambda: d, lambda: jnp.linalg.lstsq(Fp, rhs)[0])

    def solve_transpose(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N-1"]) -> Float[Array, " N"]:
        Fp = materialise_gauge_operator(Fg)
        w, _, well_posed = self._solve_augmented(Fp, jnp.zeros(Fp.shape[0], dtype=rhs.dtype), rhs)
        ok = self._accept(well_posed, jnp.linalg.norm(Fp.T @ w - rhs), rhs)
        return jax.lax.cond(ok, lambda: w, lambda: jnp.linalg.lstsq(Fp.T, rhs)[0])


class ReducedDirectSolver(AbstractDerivativeSolver):
    """Materializes F' and LU-factors the square reduced system F-bar = F'[:N-1, :].

    F-bar is F without its last row and column. At an exact solution the left
    null vector of F is `Pi z = (x, y, 1)`, whose last entry is 1, so F's last
    row is a combination of the others and can be dropped. Away from degenerate
    points F-bar is nonsingular and cond(F-bar) is about cond(F') (within a
    factor `sqrt(1 + ||x||^2 + ||y||^2)`), whereas the augmented system that
    `DenseDirectSolver` factors has cond about cond(F')^2 / ||F'||.

    At a solution computed to finite accuracy, `Pi z` is a left null vector
    of F' only up to the solver's residual, so simply dropping the last row
    would add an error of order (residual x cond). Instead, the same LU gives
    the exact left null vector `l = (l-bar, 1)` of F' (`F-bar^T l-bar = -f_N`,
    with `f_N` the last row of F'), and

    - JVP: `d = F-bar^{-1} r_p[:N-1]` with `r_p = r - (l^T r / l^T l) l`, the
      least-squares solution of `F' d = r` (range(F') is the complement of l),
    - VJP: `w = (F-bar^{-T} g, 0) - c l`, with c chosen so that w is
      orthogonal to l: the minimum-norm solution of `F'^T w = g`,

    i.e. exactly what `DenseDirectSolver` and `LSMRSolver` return. One LU of
    the (N-1) x (N-1) F-bar serves both; cost is O(N^3) time and O(N^2) memory
    (about 1/8 of the flops of the augmented LU). Falls back to an SVD
    least-squares / minimum-norm solve on F' when the smallest LU pivot is
    below `pivot_rtol` times the largest (default 1000 machine epsilons) or the
    relative residual of the normal equations exceeds `residual_rtol`.
    """

    pivot_rtol: float | None = None
    residual_rtol: float = 1e-8

    def _factor(
        self, Fp: Float[Array, "N N-1"]
    ) -> tuple[tuple[Array, Array], Float[Array, " N"], Bool[Array, ""]]:
        """LU of F-bar, the left null vector l of F', and whether the pivots look nonsingular."""
        lu, piv = jax.scipy.linalg.lu_factor(Fp[:-1, :])
        pivots = jnp.abs(jnp.diag(lu))
        pivot_rtol = 1000 * jnp.finfo(Fp.dtype).eps if self.pivot_rtol is None else self.pivot_rtol
        l_bar = jax.scipy.linalg.lu_solve((lu, piv), -Fp[-1, :], trans=1)
        null = jnp.concatenate([l_bar, jnp.ones(1, dtype=Fp.dtype)])
        return (lu, piv), null, jnp.min(pivots) > pivot_rtol * jnp.max(pivots)

    def _accept(
        self, well_posed: Bool[Array, ""], residual: Float[Array, ""], rhs: Float[Array, " k"]
    ) -> Bool[Array, ""]:
        rel_res = residual / jnp.maximum(jnp.linalg.norm(rhs), jnp.finfo(rhs.dtype).tiny)
        return well_posed & jnp.isfinite(rel_res) & (rel_res <= self.residual_rtol)

    def solve(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N"]) -> Float[Array, " N-1"]:
        Fp = materialise_gauge_operator(Fg)
        lu_piv, null, well_posed = self._factor(Fp)
        rhs_p = rhs - null * (null @ rhs) / (null @ null)
        d = jax.scipy.linalg.lu_solve(lu_piv, rhs_p[:-1])
        # Least-squares optimality: F'^T (F' d - rhs) = 0.
        ok = self._accept(well_posed, jnp.linalg.norm(Fp.T @ (Fp @ d - rhs)), Fp.T @ rhs)
        return jax.lax.cond(ok, lambda: d, lambda: jnp.linalg.lstsq(Fp, rhs)[0])

    def solve_transpose(self, Fg: lx.AbstractLinearOperator, rhs: Float[Array, " N-1"]) -> Float[Array, " N"]:
        Fp = materialise_gauge_operator(Fg)
        lu_piv, null, well_posed = self._factor(Fp)
        w = jnp.concatenate([jax.scipy.linalg.lu_solve(lu_piv, rhs, trans=1), jnp.zeros(1, dtype=rhs.dtype)])
        w = w - null * (null @ w) / (null @ null)
        ok = self._accept(well_posed, jnp.linalg.norm(Fp.T @ w - rhs), rhs)
        return jax.lax.cond(ok, lambda: w, lambda: jnp.linalg.lstsq(Fp.T, rhs)[0])


def gauge_fixed(F: lx.AbstractLinearOperator) -> lx.AbstractLinearOperator:
    """`F' = F E`, where E appends a zero (fixing `dz_N = 0`)."""
    struct = F.in_structure()
    N = struct.shape[0]
    embed = lx.FunctionLinearOperator(
        lambda d: jnp.concatenate([d, jnp.zeros((1,), dtype=d.dtype)]),
        jax.ShapeDtypeStruct((N - 1,), struct.dtype),
    )
    return F @ embed


def resolve_solver(solve_method: str | None, solver: AbstractDerivativeSolver | None) -> AbstractDerivativeSolver:
    """The solver to use: `solver` if given, else the one named by `solve_method`.

    `"jax-lsmr"` (or `None`) -> `LSMRSolver()`, `"jax-lu"` -> `DenseDirectSolver()`,
    `"jax-lu-reduced"` -> `ReducedDirectSolver()`.
    `"nvmath-direct"` is handled by `DeviceQCP` before this is called.
    """
    if solver is not None:
        return solver
    if solve_method in (None, "jax-lsmr"):
        return LSMRSolver()
    if solve_method == "jax-lu":
        return DenseDirectSolver()
    if solve_method == "jax-lu-reduced":
        return ReducedDirectSolver()
    raise ValueError(
        f'Unknown solve_method "{solve_method}". Options: "jax-lsmr", "jax-lu", "jax-lu-reduced"'
        ' (and "nvmath-direct" on DeviceQCP), or pass `solver=`.'
    )
