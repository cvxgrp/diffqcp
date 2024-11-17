from typing import List, Tuple, Union

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
import pylops as lo

from diffqcp.cones import dpi

def Du_Q(u: np.ndarray,
         P: csc_matrix,
         A: csc_matrix,
         q: np.ndarray,
         b: np.ndarray
) -> lo.LinearOperator:
    n = P.shape[0]
    m = A.shape[0]
    N = n + m + 1
    x, _, tau = u[:n], u[n:-1], u[-1]
    
    Px = P @ x
    xT_P_x = x @ Px

    def mv(du: np.ndarray) -> np.ndarray:
        dx, dy, dtau = du[:n], du[n:-1], du[-1]

        first_chunk = P @ dx + A.T @ dy + dtau * q
        second_chunk = -A @ dx + dtau * b
        final_entry = -(2/tau) * x @ (P @ dx) - q @ dx - b @ dy \
                        + (1/tau**2) * dtau * xT_P_x
        
        return np.concatenate((first_chunk, second_chunk, [final_entry]))

    def rv(dv: np.ndarray) -> np.ndarray:
        dv1, dv2, dv3 = dv[:n], dv[n:-1], dv[-1]

        first_chunk = P @ dv1 -A.T @ dv2 - (2/tau) * dv3 * Px - q
        second_chunk = A @ dv1 - dv3 * b
        final_entry = q @ dv1 + b @ dv2 + (1/tau**2) * dv3 * xT_P_x

        return np.concatenate((first_chunk, second_chunk, [final_entry])) 

    return lo.aslinearoperator(sla.LinearOperator((N, N), matvec=mv, rmatvec=rv))


def dData_Q(u: np.ndarray,
            dP: csc_matrix,
            dA: csc_matrix,
            dq: np.ndarray,
            db: np.ndarray
) -> np.ndarray:
    n = dP.shape[0]
    x, y, tau = u[:n], u[n:-1], u[-1]

    dP_x = dP @ x

    first_chunk = dP_x + dA.T @ y + tau * dq
    second_chunk = -dA @ x + tau * db
    final_entry = -(1/tau)* x @ dP_x - dq @ x - db @ y

    return np.concatenate((first_chunk, second_chunk, [final_entry]))


def form_M(u: np.ndarray,
           v: np.ndarray,
           w: float,
           Dz_Q_Pi_z: lo.LinearOperator,
           cones: List[Tuple[str,
                             Union[int, List[int]]
                            ]
                       ]
) -> lo.LinearOperator:
    DPi_z = dpi(u, v, w, cones)
    # M = Dz_Q_Pi_z.__matmul__(DPi_z).__sub__(DPi_z).__add__()
    M = Dz_Q_Pi_z @ DPi_z - DPi_z + lo.Identity(u.size + v.size + 1)
    return (1/w)*M
    

class _qcpDerivative(lo.LinearOperator):
    """Applies derivative at (P, A, q, b) to perturbations dP, dA, dq, db 
    
    """

    def __init__(self,
                 M: lo.LinearOperator,
                 D_Pi_Kstar_v: lo.LinearOperator,
                 Pi_z: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 s = np.ndarray):
        self.M: lo.LinearOperator = M
        self.D_Pi_Kstar_v = D_Pi_Kstar_v
        self.Pi_z = Pi_z
        self.x, self.y, self.s = x, y, s
        
        self.dtype
        self.shape = ()

    
    def _matvec(self,
                dP: sparse.csc_matrix,
                dA: sparse.csc_matrix,
                dq: np.ndarray,
                db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parameters
        ---------
        dP :
            SciPy sparse matrix in CSC format representing a perturbation
            to `P` in the cone program. For this reason, it must have the
            same sparsity pattern as `P`.
        dA : 
            SciPy sparse matrix in CSC format representing a perturbation
            to `A` in the cone program. For this reason, it must have the
            same sparsity pattern as `A`.
        dq : numpy.ndarray
            A perturbation to `q` in the cone program.
        db : numpy.ndarray
            A perturbation to `b` in the cone program.

        Returns
        -------
        Numpy arrays dx, dy, ds, the
        """

        n = dP.shape[0]
        m = dA.shape[0]

        # ignore the z_N term since z_N == 1?
        dQ_D = dData_Q(self.Pi_z, dP, dA, dq, db)

        if np.allclose(dQ_D, 0):
            dz = np.zeros(dQ_D.size)
        else:
            dz = lo.lsqr(self.M, -dQ_D)

        du, dv, dw = np.split(dz, [n, n + m])
        dx = du - self.x * dw
        dy = self.D_Pi_Kstar_v._matvec(dv) - self.y * dw
        ds = self.D_Pi_Kstar_v._matvec(dv) - dv - self.s * dw
        return -dx, -dy, -ds

class qcpDerivative():

    def __init__(self,
                 M: lo.LinearOperator,
                 D_Pi_Kstar_v: lo.LinearOperator,
                 Pi_z: np.ndarray,
                 x: np.ndarray,
                 y: np.ndarray,
                 s = np.ndarray):
        self.M: lo.LinearOperator = M
        self.D_Pi_Kstar_v = D_Pi_Kstar_v
        self.Pi_z = Pi_z
        self.x, self.y, self.s = x, y, s
        
        # self.dtype
        # self.shape = ()