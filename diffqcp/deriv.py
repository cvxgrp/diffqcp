from typing import List, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
import pylops as lo

from cones import dpi

def dQ(u: np.ndarray, P: csc_matrix, A: csc_matrix,
       q: np.ndarray, b: np.ndarray) -> sla.LinearOperator:
    n = P.shape[0]
    x, _, tau = u[:n], u[n:-1], u[-1]
    
    Px = P @ x
    xT_P_x = x @ Px

    def mv(du: np.ndarray):
        dx, dy, dtau = du[:n], du[n:-1], du[-1]

        first_chunk = P @ dx + A.T @ dy + dtau * q
        second_chunk = -A @ dx + dtau * q
        final_entry = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x
        
        return np.concatenate((first_chunk, second_chunk, [final_entry]))

    def rv(dv: np.ndarray):
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]

        first_chunk = P @ dv1 -A.T @ dv2 - (2/tau) * dv3 * Px - q
        second_chunk = A @ dv1 - dv3 * b
        final_entry = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return np.concatenate((first_chunk, second_chunk, [final_entry])) 

    return sla.LinearOperator((n, n), matvec=mv, rmatvec=rv)


def form_M(u: np.ndarray,
           v: np.ndarray,
           w: np.ndarray,
           DQ_Pi_z: sla.LinearOperator,
           cones: List[Tuple[str,
                         Union[int, List[int]]
                         ]
                    ]
) -> lo.LinearOperator:
    DPi_z = dpi(u, v, float(w[0]), cones)
    return DQ_Pi_z @ DPi_z - DPi_z + lo.Identity(u.size + v.size + w.size)


def dQ_wrt_D(u: np.ndarray, dP: csc_matrix,
             dA: csc_matrix, dq: np.ndarray,
             db: np.ndarray) -> np.ndarray:
    """
    Notes
    -----
    This `u` is different than the u from u, v, w = z
    """
    n = dP.shape[0]
    x, y, tau = u[:n], u[n:-1], u[-1]

    dP_x = dP @ x

    first_chunk = dP_x + dA.T @ y + tau * dq
    second_chunk = -dA @ x + tau * db
    final_entry = -(1/tau)* x @ dP_x - dq @ x - db @ y

    return np.concatenate((first_chunk, second_chunk, [final_entry]))
    

