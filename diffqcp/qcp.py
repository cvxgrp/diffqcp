"""
Exposes the function to be used to compute the derivative of a QCP.
"""
from typing import Dict, Union, List, Callable, Tuple

import numpy as np
from scipy.sparse import csc_matrix
import pylops as lo
from pylops.optimization.cls_leastsquares import lsqr
from clarabel import DefaultSolution

import diffqcp.cones as cone_utils
from diffqcp.qcp_deriv import Du_Q, form_M, dData_Q


def compute_derivative(P: csc_matrix,
                       A: csc_matrix,
                       q: np.ndarray,
                       b: np.ndarray,
                       cone_dict: Dict[str,
                                       Union[int, List[int]]
                                      ],
                       solution: DefaultSolution
) -> Callable[[csc_matrix,
               csc_matrix,
               np.ndarray,
               np.ndarray],
               Tuple[np.ndarray,
                     np.ndarray,
                     np.ndarray]]:
    """Returns the derivative of a cone program as an abstract linear map.

    Given a solution (x, y, s) to a quadratic convex cone program
    with primal-dual problems

        (P) minimize    (1/2)x^T P x + q^T x
            subject to  Ax + s = b
                        s in K

        (D) minimize    -(1/2)x^T P x - b^T y
            subject to  A^T y + c = 0
                        y in K^*

    with problem data P, A, q, b, this function returns a Linear Operator that represents
    the application of the derivative (at P, A, q, b).

    Parameters
    ---------
    P : A sparse SciPy matrix in CSC format. **Only the upper triangular part of P should be passed in.**
        Quadratic component of objective function.
    A :
        A sparse SciPy matrix in CSC format. The first block of rows
        must correspond to the zero cone, the next block ot the positive
        orthant, then the second-order cone, the PSD cone, the exponential
        cone, and finally the exponential dual cone. PSD matrix variables
        must be vectorized by scaling the off-diagonal entries by sqrt(2)
        and stacking the lower triangular part in column-major order.
        (TODO: WARNING still necessary?)
    q : np.ndarray
        Linear component of objective function.
    b : np.ndarray
        Cone program constraint offset.
    cone_dict :
        A dictionary with keys corresponding to cones and values
        corresponding to their dimension. The keys must be a subset of
            - diffqcp.ZERO
            - diffqcp.POS
            - diffqcp.SOC
            - diffqcp.PSD
            - diffqcp.EXP.
            - TODO: what about diffqcp.EXP_DUAL
        The values of diffqcp.ZERO and diffqcp.POS are scalars while
        the values of diffqcp.SOC, diffqcp.PSD, and diffqcp.EXP should
        be lists. A k-dimensional PSD cone corresponds to a k x k matrix
        variable; a value of k for diffcp.EXP corresponds to k / 3
        exponential cones. See SCS documentation for more details.
    solution :
        The DefaultSolution object returned by calling
        clarabel.DefaultSolver.solve().

    Returns
    -------
    Callable[[csc_matrix,
              csc_matrix,
              np.ndarray,
              np.ndarray
              ],
              Tuple[np.ndarray,
                    np.ndarray,
                    np.ndarray]
            ]
        The derivative of a primal-dual conic problem at (P, A, q, b)
        as an abstract linear operator.
    """

    x = np.array(solution.x)
    y = np.array(solution.z)
    s = np.array(solution.s)

    cones = cone_utils.parse_cone_dict(cone_dict)

    z = (x, y - s, np.array([1]))
    u, v, w, = z
    Pi_z = cone_utils.pi(z, cones)

    Dz_Q_Pi_z: lo.LinearOperator = Du_Q(Pi_z, P, A, q, b)
    # TODO?:somehow cache results for projecting onto cones -> most cones are self dual
    D_Pi_Kstar_v: lo.LinearOperator = cone_utils.dprojection(v, cones, dual=True)
    M: lo.LinearOperator = form_M(u, v, w[0], Dz_Q_Pi_z, cones)

    def derivative(dP: csc_matrix,
                   dA: csc_matrix,
                   dq: np.ndarray,
                   db: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, m = x.size, y.size

        dQ_D = dData_Q(Pi_z, dP, dA, dq, db)

        if np.allclose(dQ_D, 0):
            dz = np.zeros(dQ_D.size)
        else:
            dz = lsqr(M, dQ_D)[0]

        du, dv, dw = np.split(dz, [n, n + m])
        dx = du - x * dw
        dy = D_Pi_Kstar_v._matvec(dv) - y * dw
        ds = D_Pi_Kstar_v._matvec(dv) - dv - s * dw
        return dx, dy, ds

    return derivative
