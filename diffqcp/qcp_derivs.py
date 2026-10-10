"""Derivatives of the homogeneous embedding Q(u, data) of a QCP.

With `u = (x, y, tau)` and `data = (P, A, q, b)`,

    Q(u, data) = ( P x + A^T y + tau q,
                   -A x + tau b,
                   -(1/tau) x^T P x - q^T x - b^T y ).

This module has the two pieces the JVP and VJP need:

- `_DuQ`: the Jacobian `D_u Q(u)` as a lineax operator (and `_DuQAdjoint`,
  its transpose);
- `_d_data_Q`: the data JVP `D_data Q(u)[dP, dA, dq, db]`, and
  `_d_data_Q_adjoint_cpu` / `_d_data_Q_adjoint_gpu`: its adjoint, returning
  the data perturbations in the sparsity pattern of P and A.
"""
from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jax import ShapeDtypeStruct
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Float, Integer
from lineax import AbstractLinearOperator

from diffqcp.linops import _dense_from_mv
from diffqcp.problem_data import ObjMatrix


class _DuQAdjoint(AbstractLinearOperator):
    """The transpose `D_u Q(u)^T` of `_DuQ` (same fields)."""

    P: ObjMatrix
    Px: Float[Array, " n"]
    xTPx: Float[Array, ""]
    A: Float[BCOO | BCSR, "m n"]
    AT: Float[BCOO | BCSR, "n m"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    tau: Float[Array, ""]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def mv(self, dv: Float[Array, " n+m+1"]) -> Float[Array, " n+m+1"]:
        dv1: Float[Array, " n"] = dv[:self.n]
        dv2: Float[Array, " m"] = dv[self.n:-1]
        dv3: Float[Array, ""] = dv[-1]
        out1 = self.P.mv(dv1) - self.AT @ dv2 + ( -(2/self.tau) * self.Px - self.q) * dv3
        out2 = self.A @ dv1 - dv3 * self.b
        out3 = self.q @ dv1 + self.b @ dv2 + (1/self.tau**2) * dv3 * self.xTPx
        return jnp.concatenate([out1, out2, jnp.array([out3])])

    def as_matrix(self) -> Float[Array, "n+m+1 n+m+1"]:
        return _dense_from_mv(self)

    def transpose(self) -> _DuQ:
        return _DuQ(self.P, self.Px, self.xTPx, self.A, self.AT, self.q,
                    self.b, self.x, self.tau, self.n, self.m)

    def in_structure(self) -> ShapeDtypeStruct:
        return ShapeDtypeStruct(shape=(self.n + self.m + 1,),
                                dtype=self.A.dtype)

    def out_structure(self) -> ShapeDtypeStruct:
        return self.in_structure()


class _DuQ(AbstractLinearOperator):
    """The Jacobian `D_u Q(u)` of the homogeneous embedding at `u = (x, y, tau)`.

    `P`, `A`, `AT`, `q`, `b` are the problem data; `x` and `tau` the point;
    `Px = P x` and `xTPx = x^T P x` are precomputed. `n`, `m` are the sizes of
    x and y, so the operator is (n + m + 1) x (n + m + 1).
    """
    P: ObjMatrix
    Px: Float[Array, " n"]
    xTPx: Float[Array, ""]
    A: Float[BCOO | BCSR, "m n"]
    AT: Float[BCOO | BCSR, "n m"]
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    tau: Float[Array, ""]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def mv(self, du: Float[Array, " n+m+1"]) -> Float[Array, " n+m+1"]:
        dx, dy, dtau = du[:self.n], du[self.n:-1], du[-1]
        Pdx = self.P.mv(dx)
        out1 = Pdx + self.AT @ dy + dtau * self.q
        out2 = self.A @ (-dx) + dtau * self.b
        out3 = ((-2/self.tau) * self.x @ Pdx - self.q @ dx - self.b @ dy
                + (1/self.tau**2) * dtau * self.xTPx)
        return jnp.concatenate([out1, out2, jnp.array([out3])])

    def as_matrix(self) -> Float[Array, "n+m+1 n+m+1"]:
        return _dense_from_mv(self)

    def transpose(self) -> _DuQAdjoint:
        return _DuQAdjoint(self.P, self.Px, self.xTPx, self.A, self.AT, self.q,
                           self.b, self.x, self.tau, self.n, self.m)

    def in_structure(self) -> ShapeDtypeStruct:
        return ShapeDtypeStruct(shape=(self.n + self.m + 1,),
                                dtype=self.A.dtype)

    def out_structure(self) -> ShapeDtypeStruct:
        return self.in_structure()


@lx.is_symmetric.register(_DuQAdjoint)
def _(op: _DuQAdjoint) -> bool:
    return False


@lx.conj.register(_DuQAdjoint)
def _(op: _DuQAdjoint) -> _DuQAdjoint:
    return op


@lx.is_symmetric.register(_DuQ)
def _(op: _DuQ) -> bool:
    return False


@lx.conj.register(_DuQ)
def _(op: _DuQ) -> _DuQ:
    return op


def _d_data_Q(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    tau: Float[Array, ""],
    dP: ObjMatrix,
    dA: Float[BCOO | BCSR, "m n"],
    dAT: Float[BCOO | BCSR, "n m"],
    dq: Float[Array, " n"],
    db: Float[Array, " m"]
) -> Float[Array, " n+m+1"]:
    """The data JVP `D_data Q(u, data)[dP, dA, dq, db]` at `u = (x, y, tau)`.

    Q is affine in the data, so this is

        ( dP x + dA^T y + tau dq,
          -dA x + tau db,
          -(1/tau) x^T dP x - dq^T x - db^T y ).

    `dP` is the full symmetric perturbation of P (wrapped as an `ObjMatrix`,
    which expands a stored upper triangle on the CPU path), and `dAT` is the
    transpose of `dA`.
    """

    dPx = dP.mv(x)
    out1 = dPx + dAT @ y + tau * dq
    out2 = dA @ -x + tau * db
    out3 = -(1 / tau) * (x @ dPx) - dq @ x - db @ y

    return jnp.concatenate([out1, out2, jnp.array([out3])])


def _adjoint_values(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    tau: Float[Array, ""],
    w1: Float[Array, " n"],
    w2: Float[Array, " m"],
    w3: Float[Array, ""],
    P_rows: Integer[Array, "..."],
    P_cols: Integer[Array, "..."],
    A_rows: Integer[Array, "..."],
    A_cols: Integer[Array, "..."],
) -> tuple[Float[Array, " nnz_P"], Float[Array, " nnz_A"], Float[Array, " n"], Float[Array, " m"]]:
    """The adjoint of `_d_data_Q` applied to `w = (w1, w2, w3)`, at the nonzeros of P and A.

    For the entry (i, j) of a symmetric P the gradient is
    `(w1_i x_j + x_i w1_j) / 2 - (w3 / tau) x_i x_j`; for A it is
    `y_i w1_j - w2_i x_j`. `dq = tau w1 - w3 x` and `db = tau w2 - w3 y`.
    The `*_rows` / `*_cols` arrays give the (row, column) of each stored entry.
    """
    dP_values = (0.5 * ( w1[P_rows] * x[P_cols] + x[P_rows] * w1[P_cols] )
                 - (w3 / tau) * x[P_rows] * x[P_cols])
    dA_values = y[A_rows] * w1[A_cols] - w2[A_rows] * x[A_cols]
    dq = tau * w1 - w3 * x
    db = tau * w2 - w3 * y

    return (dP_values, dA_values, dq, db)


def _d_data_Q_adjoint_cpu(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    tau: Float[Array, ""],
    w1: Float[Array, " n"],
    w2: Float[Array, " m"],
    w3: Float[Array, ""],
    P_rows: Integer[Array, "..."],
    P_cols: Integer[Array, "..."],
    A_rows: Integer[Array, "..."],
    A_cols: Integer[Array, "..."],
    n: int,
    m: int
) -> tuple[
    Float[BCOO, "n n"], Float[BCOO, "m n"], Float[Array, " n"], Float[Array, " m"]
    ]:
    """The data VJP `D_data Q(u, data)^T w`, with dP and dA as BCOO (CPU path).

    `w = (w1, w2, w3)`; see `_adjoint_values`. On the CPU path `P` is stored as its upper triangle, and each stored
    off-diagonal value u_ij stands for both P_ij and P_ji. `_adjoint_values`
    returns the gradient with respect to a single entry of the symmetric
    matrix, so the gradient with respect to the stored value u_ij (i != j) is
    twice that.
    """
    dP_values, dA_values, dq, db = _adjoint_values(x, y, tau, w1, w2, w3,
                                                   P_rows, P_cols, A_rows, A_cols)
    dP_values = jnp.where(P_rows == P_cols, dP_values, 2 * dP_values)

    P_indices = jnp.stack([P_rows, P_cols], axis=1)
    dP = BCOO((dP_values, P_indices), shape=(n, n))
    A_indices = jnp.stack([A_rows, A_cols], axis=1)
    dA = BCOO((dA_values, A_indices), shape=(m, n))

    return (dP, dA, dq, db)


def _d_data_Q_adjoint_gpu(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    tau: Float[Array, ""],
    w1: Float[Array, " n"],
    w2: Float[Array, " m"],
    w3: Float[Array, ""],
    P_rows: Integer[Array, "..."],
    P_cols: Integer[Array, "..."],
    P_csr_indices: Integer[Array, "..."],
    P_csr_indtpr: Integer[Array, "..."],
    A_rows: Integer[Array, "..."],
    A_cols: Integer[Array, "..."],
    A_csr_indices: Integer[Array, "..."],
    A_csr_indtpr: Integer[Array, "..."],
    n: int,
    m: int
) -> tuple[
    Float[BCSR, "n n"], Float[BCSR, "m n"], Float[Array, " n"], Float[Array, " m"]
    ]:
    """The data VJP `D_data Q(u, data)^T w`, with dP and dA as BCSR (GPU path).

    `w = (w1, w2, w3)`; see `_adjoint_values`. P is stored in full, so each
    stored entry gets the gradient with respect to that entry of the symmetric
    matrix (equal for (i, j) and (j, i)). The CSR index arrays fix the
    sparsity patterns of the returned matrices; the parameter names keep the
    historical `indtpr` spelling of `indptr`.
    """
    dP_values, dA_values, dq, db = _adjoint_values(x, y, tau, w1, w2, w3,
                                                   P_rows, P_cols, A_rows, A_cols)

    dP = BCSR((dP_values, P_csr_indices, P_csr_indtpr), shape=(n, n))
    dA = BCSR((dA_values, A_csr_indices, A_csr_indtpr), shape=(m, n))

    return (dP, dA, dq, db)

