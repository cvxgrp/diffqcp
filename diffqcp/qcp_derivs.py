"""
Non-cone derivative atoms composing the QCP solution map derivative.
"""
import torch
import linops as lo

from diffqcp.cone_derivs import dpi
from diffqcp.utils import sparse_tensor_transpose
from diffqcp.linops import _sLinearOperator

def Du_Q(u: torch.Tensor,
         P: torch.Tensor | lo.LinearOperator,
         A: torch.Tensor,
         q: torch.Tensor,
         b: torch.Tensor
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

    # torch compile will handle conversion of n and N to GPU.
    n = P.shape[0]
    m = A.shape[0]
    N = n + m + 1
    x, tau = u[:n], u[-1]

    Px = P @ x
    xT_P_x = x @ Px
    AT = sparse_tensor_transpose(A) # If A was in csr format, AT is in csc format.

    def mv(du: torch.Tensor) -> torch.Tensor:
        dx, dy, dtau = du[:n], du[n:-1], du[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dx + AT @ dy + dtau * q
        out[n:-1] = -A @ dx + dtau * b
        out[-1] = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x

        return out

    def rv(dv: torch.Tensor) -> torch.Tensor:
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]
        out = torch.empty(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dv1 - AT @ dv2 + ( -(2/tau)*Px - q )*dv3
        out[n:-1] = A @ dv1 - dv3 * b
        out[-1] = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return out

    return _sLinearOperator(N, N, mv, rv)


def dData_Q(u: torch.Tensor,
            dP: torch.Tensor | lo.LinearOperator,
            dA: torch.Tensor,
            dq: torch.Tensor,
            db: torch.Tensor
) -> torch.Tensor:
    """Jacobian-vector product of Q at (u, data) and d_data.

    More specifically, returns D_data Q(u, data)[d_data], where
    d_data = (dP, dA, dq, db), Q is the nonlinear homogeneous embedding
    and D_data is the derivative operator w.r.t. data = (P, A, q, b).

    u, dP, dA, dq, and db are the exact objects defined in the diffqcp paper.
    Specifically, note that dP should be the true perturbation to the matrix P,
    **not just the upper triangular part.**

    Notes
    -----
    Potentially refactor into lo.LinearOperator once we have the adjoint.
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


def form_M(u: torch.Tensor,
           v: torch.Tensor,
           w: torch.Tensor,
           Dz_Q_Pi_z: lo.LinearOperator,
           cones: list[tuple[str, int | list[int]]]
) -> lo.LinearOperator:
    """Form the derivative composition M as given in diffqcp implementation section.
    """
    N = u.shape[0] + v.shape[0] + 1
    Dz_Pi_z = dpi(u, v, w, cones)
    M = (Dz_Q_Pi_z @ Dz_Pi_z) - Dz_Pi_z + lo.IdentityOperator(N)
    return M
