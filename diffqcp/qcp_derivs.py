"""Non-cone derivative atoms composing the QCP solution map derivative.
"""
import torch
from torch import Tensor
import linops as lo
from jaxtyping import Float

from diffqcp.utils import sparse_tensor_transpose
from diffqcp.linops import _sLinearOperator

def Du_Q_efficient(
    u: Float[Tensor, "n+m+1"],
    P: Float[Tensor, "n n"] | lo.LinearOperator,
    A: Float[Tensor, "m n"],
    AT: Float[Tensor, "n m"],
    q: Float[Tensor, "n"],
    b: Float[Tensor, "m"]
) -> lo.LinearOperator:
    """Returns derivative of nonlinear homogeneous embedding w.r.t. u at u.

    Specifically, the derivative is returned as an abstract linear operator
    instantiated as a lo.LinearOperator. Mathematically, this is the
    object D_u Q(u, data), where D_u is the derivative operator with respect
    to u, Q is the nonlinear homogeneous embedding, and data = (P, A, q, b).

    Parameters
    ----------
    u : torch.Tensor
        The point the derivative of the nonlinear homogeneous embedding is evaluated at.
    P : torch.Tensor | lo.LinearOperator
        A PSD matrix. **Unlike the `P` parameter in the `compute_derivative` function,
        this `P` should be the full PSD matrix; not just its upper triangular part.**
    A : torch.Tensor
        A 2-D tensor whose number of rows match the size of P.shape[0] == P.shape[1].
    AT: torch.Tensor
        A^T
    q : torch.Tensor
        A 1-D tensor such that q.shape[0] == P.shape[0] == P.shape[1].
    b : torch.Tensor
        A 1D tensor such that b.shape[0] == A.shape[0].

    Returns
    -------
    lo.LinearOperator
        The derivative (w.r.t. u) of the nonlinear homogeneous embedding at u.
    """

    n = P.shape[0]
    m = A.shape[0]
    N = n + m + 1
    x, tau = u[:n], u[-1]

    Px = P @ x
    xT_P_x = x @ Px

    def mv(du: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
        dx, dy, dtau = du[:n], du[n:-1], du[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dx + AT @ dy + dtau * q
        out[n:-1] = -A @ dx + dtau * b
        out[-1] = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x

        return out

    def rv(dv: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dv1 - AT @ dv2 + ( -(2/tau)*Px - q )*dv3
        out[n:-1] = A @ dv1 - dv3 * b
        out[-1] = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return out

    return _sLinearOperator(N, N, mv, rv, device=A.device)


def Du_Q(
    u: Float[Tensor, "n+m+1"],
    P: Float[Tensor, "n n"] | lo.LinearOperator,
    A: Float[Tensor, "m n"],
    q: Float[Tensor, "n"],
    b: Float[Tensor, "m"]
) -> lo.LinearOperator:
    """Returns derivative of nonlinear homogeneous embedding w.r.t. u at u.

    Specifically, the derivative is returned as an abstract linear operator
    instantiated as a lo.LinearOperator. Mathematically, this is the
    object D_u Q(u, data), where D_u is the derivative operator with respect
    to u, Q is the nonlinear homogeneous embedding, and data = (P, A, q, b).

    Parameters
    ----------
    u : torch.Tensor
        The point the derivative of the nonlinear homogeneous embedding is evaluated at.
    P : torch.Tensor | lo.LinearOperator
        A PSD matrix. **Unlike the `P` parameter in the `compute_derivative` function,
        this `P` should be the full PSD matrix; not just its upper triangular part.**
    A : torch.Tensor
        A 2-D tensor whose number of rows match the size of P.shape[0] == P.shape[1].
    q : torch.Tensor
        A 1-D tensor such that q.shape[0] == P.shape[0] == P.shape[1].
    b : torch.Tensor
        A 1D tensor such that b.shape[0] == A.shape[0].

    Returns
    -------
    lo.LinearOperator
        The derivative (w.r.t. u) of the nonlinear homogeneous embedding at u.
    """

    n = P.shape[0]
    m = A.shape[0]
    N = n + m + 1
    x, tau = u[:n], u[-1]

    Px = P @ x
    xT_P_x = x @ Px
    AT = sparse_tensor_transpose(A) # If A was in csr format, AT is in csc format.

    def mv(du: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
        dx, dy, dtau = du[:n], du[n:-1], du[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dx + AT @ dy + dtau * q
        out[n:-1] = -A @ dx + dtau * b
        out[-1] = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x

        return out

    def rv(dv: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dv1 - AT @ dv2 + ( -(2/tau)*Px - q )*dv3
        out[n:-1] = A @ dv1 - dv3 * b
        out[-1] = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return out

    return _sLinearOperator(N, N, mv, rv, device=A.device)


def dData_Q_efficient(
    u: Float[Tensor, 'n+m+1'],
    dP: Float[Tensor, 'n n'] | lo.LinearOperator,
    dA: Float[Tensor, 'm n'],
    dAT: Float[Tensor, 'n m'],
    dq: Float[Tensor, 'n'],
    db: Float[Tensor, 'm']
) -> Float[Tensor, 'n+m+1']:
    """Jacobian-vector product of Q at (u, data) and d_data.

    More specifically, returns D_data Q(u, data)[d_data], where
    d_data = (dP, dA, dq, db), Q is the nonlinear homogeneous embedding
    and D_data is the derivative operator w.r.t. data = (P, A, q, b).

    u, dP, dA, dq, and db are the exact objects defined in the diffqcp paper.
    Specifically, note that dP should be the true perturbation to the matrix P,
    **not just the upper triangular part.**
    """
    n = dP.shape[0]
    N = n + dA.shape[0] + 1
    x, y, tau = u[:n], u[n:-1], u[-1]

    dP_x = dP @ x

    out = torch.empty(N, dtype=dA.dtype, device=dA.device)

    out[:n] = dP_x + dAT @ y + tau * dq
    out[n:-1] = -dA @ x + tau * db
    out[-1] = -(1/tau)* x @ dP_x - dq @ x - db @ y

    return out


def dData_Q(
    u: Float[Tensor, 'n+m+1'],
    dP: Float[Tensor, 'n n'] | lo.LinearOperator,
    dA: Float[Tensor, 'm n'],
    dq: Float[Tensor, 'n'],
    db: Float[Tensor, 'm']
) -> Float[Tensor, 'n+m+1']:
    """The Jacobian-vector product D_dataQ(u, data)[data].

    More specifically, returns D_data Q(u, data)[d_data], where
    d_data = (dP, dA, dq, db), Q is the nonlinear homogeneous embedding
    and D_data is the derivative operator w.r.t. data = (P, A, q, b).

    u, dP, dA, dq, and db are the exact objects defined in the diffqcp paper.
    Specifically, note that dP should be the true perturbation to the matrix P,
    **not just the upper triangular part.**

     Notes
    -----
    - TODO (quill): switch names with the above (dData_Q_efficient), then delete this one
    """
    n = dP.shape[0]
    N = n + dA.shape[0] + 1
    x, y, tau = u[:n], u[n:-1], u[-1]

    dP_x = dP @ x
    dAT = sparse_tensor_transpose(dA)

    out = torch.empty(N, dtype=dA.dtype, device=dA.device)

    out[:n] = dP_x + dAT @ y + tau * dq
    out[n:-1] = -dA @ x + tau * db
    out[-1] = -(1/tau)* x @ dP_x - dq @ x - db @ y

    return out


def dData_Q_adjoint_efficient(
    u: Float[Tensor, 'n+m+1'],
    w1: Float[Tensor, 'n'],
    w2: Float[Tensor, 'm'],
    w3: Float[Tensor, ''],
    P_rows: torch.Tensor,
    P_cols: torch.Tensor,
    Pcrow_indices: torch.Tensor,
    Pcol_indices: torch.Tensor,
    A_rows: torch.Tensor,
    A_cols: torch.Tensor,
    Acrow_indices: torch.Tensor,
    Acol_indices: torch.Tensor
) -> tuple[
        Float[Tensor, 'n n'], Float[Tensor, 'm n'], Float[Tensor, 'n'], Float[Tensor, 'm']
    ]:
    """The vector-Jacobian product D_dataQ(u, data)^T[w].
    
    """
    # so take in parameters which specify the entries to fill in.
    n = w1.shape[0]
    m = w2.shape[0]

    x = u[:n]
    y = u[n:-1]
    tau = u[-1]

    dP_values = (0.5 * ( w1[P_rows] * x[P_cols] + x[P_rows] * w1[P_cols] )
                 - (w3 / tau) * x[P_rows] * x[P_cols])
    dA_values = y[A_rows] * w1[A_cols] - w2[A_rows] * x[A_cols]

    # question: what happens when P is upper triangular
    #  -> should just work: we'll only get upper triangular values.
    dP = torch.sparse_csr_tensor(
        Pcrow_indices, Pcol_indices, dP_values, size=(n,n), dtype=u.dtype, device=u.device
    )
    dA = torch.sparse_csr_tensor(
        Acrow_indices, Acol_indices, dA_values, size=(m,n), dtype=u.dtype, device=u.device
    )

    dq = tau * w1 - w3 * x
    db = tau*w2 - w3 * y

    return (dP, dA, dq, db)

def dData_Q_adjoint(
    u: Float[Tensor, 'n+m+1'],
    w1: Float[Tensor, 'n'],
    w2: Float[Tensor, 'm'],
    w3: Float[Tensor, ''],
) -> tuple[
        Float[Tensor, 'n n'], Float[Tensor, 'm n'], Float[Tensor, 'n'], Float[Tensor, 'm']
    ]:
    """The vector-Jacobian product D_dataQ(u, data)^T[du].
    """
    # so take in parameters which specify the entries to fill in.
    n = w1.shape[0]
    m = w2.shape[0]

    w1 = w1.reshape((n, 1))
    w2 = w2.reshape((m, 1))

    x = u[:n].reshape((n, 1))
    y = u[n:-1].reshape((m, 1))
    tau = u[-1]

    dP = 0.5 * (x @ w1.T + w1 @ x.T)  - (w3 / tau) * x @ x.T
    dA = y @ w1.T - w2 @ x.T
    dq = (tau * w1 - w3 * x).squeeze()
    db = (tau*w2 - w3 * y).squeeze()

    return (dP, dA, dq, db)
    

