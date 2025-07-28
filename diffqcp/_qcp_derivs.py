from __future__ import annotations

from jax import ShapeDtypeStruct
import jax.numpy as jnp
import equinox as eqx
from lineax import AbstractLinearOperator
from jaxtyping import Array, Float, Integer
from jax.experimental.sparse import BCOO, BCSR

from diffqcp._problem_data import ObjMatrix, ConstrMatrix

# NOTE(quill): the last bit of that would fail since `dtau * self.q` would be 1D array * 2D array
#   So I guess the somewhat challenging aspect of this is the fact that the first two bits
#   in the expression are fine, so we don't actually want to vmap those...
# NOTE(quill): UPDATE. This is NOT TRUE. If in the batched case then `dtau` is also a 2D array!

class _DuQAdjoint(AbstractLinearOperator):
    P: ObjMatrix
    Px: Float[Array, " n"]
    xTPx: Float[Array, ""]
    A: ConstrMatrix
    AT: ConstrMatrix
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    tau: Float[Array, ""]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def mv(self, dv):
        dv1: Float[Array, " n"] = dv[:self.n]
        dv2: Float[Array, " m"] = dv[self.n:-1]
        dv3: Float[Array, ""] = dv[-1]
        out1 = self.P @ dv1 - self.AT @ dv2 + ( -(2/self.tau) * self.Px - self.q) * dv3
        out2 = self.A @ dv1 - dv3 * self.b
        out3 = self.q @ dv1 + self.b @ dv2 + (1/self.tau**2) * dv3 * self.xTPx
        return jnp.concatenate([out1, out2, out3])

    def as_matrix(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s `as_matrix` method is"
                                  + " not yet implemented.")
    
    def transpose(self) -> _DuQ:
        return _DuQ(self.P, self.Px, self.xTPx, self.A, self.AT, self.q,
                    self.b, self.x, self.tau, self.n, self.m)

    def in_structure(self):
        return ShapeDtypeStruct(shape=(self.n + self.m + 1,),
                                dtype=self.P.dtype)
    
    def out_structure(self):
        return self.in_structure


class _DuQ(AbstractLinearOperator):
    """
    NOTE(quill): we know at compile time if this is batched or not.
    """
    P: ObjMatrix
    Px: Float[Array, " n"]
    xTPx: Float[Array, ""]
    A: ConstrMatrix
    AT: ConstrMatrix
    q: Float[Array, " n"]
    b: Float[Array, " m"]
    x: Float[Array, " n"]
    tau: Float[Array, ""]
    n: int = eqx.field(static=True)
    m: int = eqx.field(static=True)

    def mv(self, du: Float[Array, " n+m+1"]):
        dx, dy, dtau = du[:self.n], du[self.n:-1], du[-1]
        out1 = self.P @ dx + self.AT @ dy + dtau * self.q
        out2 = -self.A @ dx + dtau * self.b
        out3 = ((-2/self.tau) * self.x @ (self.P @ dx) - self.q @ dx - self.b @ dy
                + (1/self.tau**2) * dtau + self.xTPx)
        return jnp.concatenate([out1, out2, out3])
    
    def as_matrix(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s `as_matrix` method is"
                                  + " not yet implemented.")
    
    def transpose(self) -> _DuQAdjoint:
        return _DuQAdjoint(self.P, self.Px, self.xTPx, self.A, self.AT, self.q,
                           self.b, self.x, self.tau, self.n, self.m)
    
    def in_structure(self):
        return ShapeDtypeStruct(shape=(self.n + self.m + 1,),
                                dtype=self.P.dtype)
    
    def out_structure(self):
        return self.in_structure

def _d_data_Q(
    x: Float[Array, " n"],
    y: Float[Array, " m"],
    tau: Float[Array, ""],
    dP: Float[Array | BCOO | BCSR, "n n"],
    dA: Float[Array | BCOO | BCSR, "m n"],
    dAT: Float[Array | BCOO, BCSR, "n m"],
    dq: Float[Array, " n"],
    db: Float[Array, " m"]
) -> Float[Array, " n+m+1"]:
    """The Jacobian-vector product D_dataQ(u, data)[data].

    More specifically, returns D_data Q(u, data)[d_data], where
    d_data = (dP, dA, dq, db), Q is the nonlinear homogeneous embedding
    and D_data is the derivative operator w.r.t. data = (P, A, q, b).

    u, dP, dA, dq, and db are the exact objects defined in the diffqcp paper.
    Specifically, note that dP should be the true perturbation to the matrix P,
    **not just the upper triangular part.**
    """
    
    dPx = dP @ x
    out1 = dPx + dAT @ y + tau * dq
    out2 = -dA @ x + tau * db
    out3 = -(1 / tau) * (x @ dPx) - dq @ x - db @ y

    return jnp.concatenate([out1, out2, jnp.array([out3])])

# NOTE(quill): what's going to happen when these get `jit`ted?


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
) -> tuple[Float[Array, "..."], Float[Array, "..."], Float[Array, " n"], Float[Array, " m"]]:
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
    """The vector-Jacobian product D_data(u, data)^T[w].
    """
    dP_values, dA_values, dq, db = _adjoint_values(x, y, tau, w1, w2, w3,
                                                   P_rows, P_cols, A_rows, A_cols)

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
    """The vector-Jacobian product D_data(u, data)^T[w]."""
    dP_values, dA_values, dq, db = _adjoint_values(x, y, tau, w1, w2, w3,
                                                   P_rows, P_cols, A_rows, A_cols)

    dP = BCSR((dP_values, P_csr_indices, P_csr_indtpr), shape=(n, n))
    dA = BCSR((dA_values, A_csr_indices, A_csr_indtpr), shape=(m, n))

    return (dP, dA, dq, db)

