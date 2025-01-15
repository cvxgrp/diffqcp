"""
Non-cone derivative atoms composing the QCP solution map derivative.
"""
from typing import Tuple

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
    """Returns derivative of nonlinear homogeneous embedding map w.r.t. u.
    TODO: add more.
    P SHOULD NOT JUST BE UPPER TRIANGULAR PART
    """

    n = P.shape[0]
    m = A.shape[0]
    N = n + m + 1
    x, tau = u[:n], u[-1]

    Px = P @ x
    xT_P_x = x @ Px
    AT = sparse_tensor_transpose(A) # If A was in csr format, AT is in csc format.

    def mv(du: torch.Tensor) -> torch.Tensor:
        dx, dy, dtau = du[:n], du[n:-1], du[-1]
        out = torch.zeros(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dx + AT @ dy + dtau * q
        out[n:-1] = -A @ dx + dtau * b
        out[-1] = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x

        return out

    def rv(dv: torch.Tensor) -> torch.Tensor:
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]
        out = torch.zeros(N, dtype=A.dtype, device=A.device)

        out[0:n] = P @ dv1 - AT @ dv2 + ( -(2/tau)*Px - q )*dv3
        out[n:-1] = A @ dv1 - dv3 * b
        out[-1] = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return out

    Du_Q_op = _sLinearOperator(N, N, mv, rv)
    return Du_Q_op


def dData_Q(u: torch.Tensor,
            dP: torch.Tensor | lo.LinearOperator,
            dA: torch.Tensor,
            dq: torch.Tensor,
            db: torch.Tensor
) -> torch.Tensor:
    """
    Returns application of derivative of nonlinear homogeneous embedding w.r.t. the data
    to a data perturbation.
    TODO: add more.

    dP SHOULD NOT JUST BE UPPER TRIANGULAR PART

    Notes
    -----
    Potentially refactor into lo.LinearOperator once we have the adjoint.
    """
    n = dP.shape[0]
    x, y, tau = u[:n], u[n:-1], u[-1]

    dP_x = dP @ x
    dAT = sparse_tensor_transpose(dA)

    first_chunk = dP_x + dAT @ y + tau * dq
    second_chunk = -dA @ x + tau * db
    final_entry = -(1/tau)* x @ dP_x - dq @ x - db @ y

    return torch.cat((first_chunk,
                      second_chunk,
                      final_entry.unsqueeze(0)))


def form_M(u: torch.Tensor,
           v: torch.Tensor,
           w: torch.Tensor,
           Dz_Q_Pi_z: lo.LinearOperator,
           cones: list[Tuple[str, int | list[int]]]
) -> lo.LinearOperator:
    """Form the derivative composition M as given in diffqcp implementation section.

    TODO: add more.
    """
    Dz_Pi_z = dpi(u, v, w, cones)
    # Pyright doesn't like the following (since no type hints for __matmul__ ?),
    # but it is valid.
    M = (Dz_Q_Pi_z @ Dz_Pi_z) - Dz_Pi_z + lo.IdentityOperator(u.shape[0] + v.shape[0] + 1)
    return M
