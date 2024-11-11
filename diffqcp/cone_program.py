from typing import Callable

import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as sla
import pylops as lo
import clarabel
from clarabel import DefaultSolution

import diffqcp.cones as cone_utils
from diffqcp import dQ, form_M, dQ_wrt_D


def compute_derivative(P: csc_matrix, A: csc_matrix,
                        q: np.ndarray, b: np.ndarray,
                        cone_dict, solution: DefaultSolution) -> Callable:
    x = np.array(solution.x)
    y = np.array(solution.y)
    s = np.array(solution.z)
    """Returns the derivative of a cone program as an abstract linear map.

    Given a solution (x, y, s) to convex cone program (with a quadratic objective),
    with primal-dual problems
        minimize    x^T P x + q^T x        minimize    -(1/2)x^T P x - b^T y
        subject to  Ax + s = b             subject to  A^T y + c = 0
                    s \in K                            y \in K^*

    with problem data P, A, q, b, this function returns a function that represents
    the application of the derivative (at P, A, q, b).

    Parameters
    ---------

    Returns
    -------

    Notes
    -----
    Add functionality to compute partials. (In practice we usually don't care about
    the derivative of the dual variable.)

    Create a dataclas D = (P, A, q, b) with dimensions
    
    """

    m, n = A.shape
    N = m + n + 1
    cones = cone_utils.parse_cone_dict(cone_dict)
    
    z = (x, y - s, np.array([1]))
    u, v, w, = z

    Pi_z = cone_utils.pi(z, cones)

    DQ_Pi_z: sla.LinearOperator = dQ(Pi_z, P, A, q, b)
    # note that aslinearoperator allows you to move between the lo and sla class
    M: lo.LinearOperator = form_M(u, v, w, DQ_Pi_z, cones)

    def derivative(dP: csc_matrix, dA: csc_matrix,
                   dq: np.ndarray, db: np.ndarray) -> lo.LinearOperator:
        """Applies derivative at (P, A, q, b) to perturbations dP, dA, dq, db
        
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

        dN_wrt_D = dQ_wrt_D(Pi_z, dP, dA, dq, db)
