"""
Exposes the function `compute_derivative`, which computes the derivative of a QCP.
"""
from typing import Callable

import numpy as np
from scipy.sparse import spmatrix
import torch
import linops as lo
from linops.lsqr import lsqr
import clarabel

from diffqcp.linops import SymmetricOperator, BlockDiag, ScalarOperator
import diffqcp.cones as cone_utils
from diffqcp.qcp_derivs import Du_Q, dData_Q
from diffqcp.utils import to_tensor, _convert_problem_data, _get_GPU_settings


def compute_derivative(P: torch.Tensor | spmatrix,
                       A: torch.Tensor | spmatrix,
                       q: torch.Tensor | np.ndarray,
                       b: torch.Tensor | np.ndarray,
                       cone_dict: dict[str, int | list[int]],
                       solution: tuple[torch.Tensor | np.ndarray | list[float],
                                       torch.Tensor | np.ndarray | list[float],
                                       torch.Tensor | np.ndarray | list[float]]
                                 | clarabel.DefaultSolution,
                       dtype: torch.dtype | None = None,
                       device: torch.device | None = None
) -> Callable[[torch.Tensor | spmatrix,
               torch.Tensor | spmatrix,
               torch.Tensor,
               torch.Tensor],
               tuple[torch.Tensor,
                     torch.Tensor,
                     torch.Tensor]]:
# ) -> lo.LinearOperator:
    r"""Returns the derivative of a cone program as an abstract linear map.

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
    P : torch.Tensor | scipy.sparse.spmatrix
        The quadratic component of the objective function.
        This parameter should only be the **upper triangular part** of the P
        in (P) and (D), and it should be either a scipy.sparse.spmatrix
        **or** a torch.Tensor. (See notes below for more information on the
        storage of P.)
    A : torch.Tensor | scipy.sparse.spmatrix
        A scipy.sparse.spmatrix **or** a torch.Tensor.
        (See notes below for more information on the
        storage of A.)
        The first block of rows must correspond to the zero cone, the next block
        to the positive orthant, then the second-order cone, the PSD cone, the exponential
        cone, and finally the exponential dual cone. PSD matrix variables
        must be vectorized by scaling the off-diagonal entries by sqrt(2)
        and stacking the lower triangular part in column-major order.
    q : torch.Tensor | np.ndarray
        Linear component of objective function.
    b : torch.Tensor | np.ndarray
        Cone program constraint offset.
    cone_dict : dict[str, int | list[int]]
        A dictionary with keys corresponding to cones and values
        corresponding to their dimension. The keys must be a subset of
            - diffqcp.ZERO
            - diffqcp.POS
            - diffqcp.SOC
            - diffqcp.PSD
            - diffqcp.EXP
            - diffqcp.EXP_DUAL
            - diffqcp.POW
        The values of diffqcp.ZERO, diffqcp.POS, diffqcp.EXP, diffqcp.EXP_DUAL are scalars
        while the values of diffqcp.SOC, diffqcp.PSD, and diffqcp.POW should
        be lists. A k-dimensional PSD cone corresponds to a k x k matrix
        variable; See SCS documentation for more details:
        (https://www.cvxgrp.org/scs/api/cones.html#cones).
    solution : tuple[torch.Tensor | np.ndarray | list[float],
                     torch.Tensor | np.ndarray | list[float],
                     torch.Tensor | np.ndarray | list[float]]
                | clarabel.DefaultSolution
        The primal-dual solution, (x^star, y^star, s^star), to the conic pair (P) and (D).
        This solution can be passed in as a tuple of torch.Tensors, np.ndarrays,
        or lists of floats or as a clarabel.DefaultSolution object.
        If provided as a tuple, provide the solution vectors according to
        (x^star, y^star, s^star).
    dtype : torch.dtype | None, optional
        Data type for tensors, by default torch.float32 (but see Notes below).
    device : torch.device | None, optional
        Device for tensors, by default None (but see Notes below).

    Returns
    -------
    Callable
        The derivative of a primal-dual conic problem at (P, A, q, b)
        as an abstract linear operator.

    Notes
    -----
    - Some value checks are done, but the function trusts that the provided P
    is the upper triangular part of a PSD matrix (i.e., checks on these properties
    are not done). The primal-dual solution, (x^star, y^star, s^star), is not
    checked in any way.
    - If the provided P is a torch tensor in sparse csr format, no data conversions
    will be performed. If P is a torch tensor stored in any other way (so
    according to any other sparsity format or just as a regular tensor), it will be
    internally converted to a tensor in sparse csr format.
    Similarly, while all scipy sparse matrices will be converted to a torch
    tensor, if the provided scipy sparse matrix is not in csr format, it will first
    be converted to such before being used to construct a sparse torch tensor in csr
    format.
    - The storage of A follows the same guidelines as the storage for P.
    - The dtype for tensors is chosen according to the following logic:
        - If a dtype parameter is provided, this dtype is used.
        - If a dtype parameter is not provided, but the parameter P is a torch.Tensor,
        then the dtype of P is used.
        - If a dtype parameter is not provided and P is not a torch.Tensor, then
        the dtype defaults to torch.float32.
    - The device for tensors is chosen according to the following logic:
        - If a device parameter is provided, tensor are stored and tensor computations
        are done on this device.
        - If a device parameter is not provided, but the parameter P is a torch.Tensor,
        then the device P is on will be the device used to store tensors and perform
        tensor computations.
        - If a device parameter is not provided and P is not a torch.Tensor, then
        the device defaults to None (i.e., the CPU).
    - TODO(quill): most likely will need to allow users to specify if they want to use
    sparse matrices...some DL/ML workflows may not support sparsity
    """

    DTYPE, DEVICE = _get_GPU_settings(P, dtype=dtype, device=device)

    P, A, q, b = _convert_problem_data(P, A, q, b, dtype=DTYPE, device=DEVICE)
    P_linop = SymmetricOperator(P.shape[0], P, DEVICE)

    if isinstance(solution, clarabel.DefaultSolution):
        x = solution.x
        y = solution.z
        s = solution.s
    else:
        x = solution[0]
        y = solution[1]
        s = solution[2]

    x = to_tensor(x, DTYPE, DEVICE)
    y = to_tensor(y, DTYPE, DEVICE)
    s = to_tensor(s, DTYPE, DEVICE)
    one = torch.tensor(1.0, dtype=DTYPE, device=DEVICE)

    n = x.shape[0]
    m = y.shape[0]

    cones : list[tuple[str, int | list[int]]] = cone_utils.parse_cone_dict(cone_dict)

    Pi_Kstar_v, D_Pi_Kstar_v = cone_utils.proj_and_dproj(y-s, cones, dual=True)
    Pi_z = torch.cat((x,
                      Pi_Kstar_v,
                      one.unsqueeze(-1)
                      ))
    DPi_z = BlockDiag([lo.IdentityOperator(n),
                       D_Pi_Kstar_v,
                       ScalarOperator(one)], device=DEVICE)

    Dz_Q_Pi_z: lo.LinearOperator = Du_Q(Pi_z, P_linop, A, q, b)
    M = (Dz_Q_Pi_z @ DPi_z) - DPi_z + lo.IdentityOperator(n + m + 1)

    def derivative(dP: torch.Tensor | spmatrix,
                   dA: torch.Tensor | spmatrix,
                   dq: torch.Tensor | np.ndarray,
                   db: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dP, dA, dq, db = _convert_problem_data(dP, dA, dq, db, dtype=DTYPE, device=DEVICE)
        dP_linop = SymmetricOperator(n, dP, DEVICE)

        # TODO: change dData_Q to return a linop and then move
        # the creation to outside derivative, and then
        # call operator here (this will require passing data as single tensor)
        d_DN = dData_Q(Pi_z, dP_linop, dA, dq, db)

        if torch.allclose(d_DN, torch.tensor(0, dtype=dtype, device=DEVICE)):
            dz = torch.zeros(d_DN.shape[0])
        else:
            dz = lsqr(M, -d_DN)

        dr, dw, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dx = dr - x * dz_N
        dy = D_Pi_Kstar_v @ dw - y * dz_N
        ds = D_Pi_Kstar_v @ dw - dw - s * dz_N
        return dx, dy, ds
    
    def adjoint(dx: torch.Tensor,
                dy: torch.Tensor,
                ds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    return derivative


# Callable[[csc_matrix,
#           csc_matrix,
#           np.ndarray,
#           np.ndarray
#           ],
#           tuple[np.ndarray,
#                 np.ndarray,
#                 np.ndarray]
#         ]
    # The derivative of a primal-dual conic problem at (P, A, q, b)
    # as an abstract linear operator.

# linops.LinearOperator
    # The derivative and adjoint of the conic pair at the solution, (x^star, y^star, s^star).
