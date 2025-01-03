"""
Exposes the function to be used to compute the derivative of a QCP.
"""
from typing import Dict, Tuple, Callable

import numpy as np
from scipy.sparse import csc_matrix
import torch
import linops as lo
from linops.lsqr import lsqr
import clarabel

import diffqcp.cones as cone_utils
from diffqcp.cone_derivs import dpi, dprojection
from diffqcp.qcp_derivs import Du_Q, form_M, dData_Q
from diffqcp.utils import to_tensor, to_sparse_csc_tensor, SymmetricOperator


def compute_derivative(P: torch.Tensor | csc_matrix,
                       A: torch.Tensor | csc_matrix,
                       q: torch.Tensor | np.ndarray,
                       b: torch.Tensor | np.ndarray,
                       cone_dict: Dict[str, int | list[int]],
                       solution: Tuple[torch.Tensor | np.ndarray | list[float],
                                       torch.Tensor | np.ndarray | list[float],
                                       torch.Tensor | np.ndarray | list[float]]
                                 | clarabel.DefaultSolution,
                       dtype: torch.dtype | None = None,
                       device: torch.device | None = None
) -> Callable[[torch.Tensor | csc_matrix,
               torch.Tensor | csc_matrix,
               torch.Tensor,
               torch.Tensor],
               Tuple[torch.Tensor,
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
    P : torch.Tensor | scipy.sparse.csc_matrix
        The quadratic component of the objective function.
        This parameter should only be the **upper triangular part** of the P
        in (P) and (D), and it should be either a scipy.sparse.csc_matrix
        **or** a torch.Tensor in torch.sparse_csc layout.
    A : torch.Tensor | scipy.sparse.csc_matrix
        A sparse matrix in CSC format, stored as a scipy.sparse_csc_matrix
        **or** a torch.Tensor in torch.sparse_csc_layout.
        The first block of rows must correspond to the zero cone, the next block
        to the positive orthant, then the second-order cone, the PSD cone, the exponential
        cone, and finally the exponential dual cone. PSD matrix variables
        must be vectorized by scaling the off-diagonal entries by sqrt(2)
        and stacking the lower triangular part in column-major order.
    q : torch.Tensor | np.ndarray
        Linear component of objective function.
    b : torch.Tensor | np.ndarray
        Cone program constraint offset.
    cone_dict : Dict[str, int | list[int]]
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
    solution : Tuple[torch.Tensor | np.ndarray | list[float],
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
    """

    if dtype is None and isinstance(P, torch.Tensor):
        DTYPE = P.dtype
    elif dtype is not None:
        assert isinstance(dtype, torch.dtype)
        DTYPE = dtype
    else:
        DTYPE = torch.float32

    if device is None and isinstance(P, torch.Tensor):
        DEVICE = P.device
    else:
        assert (device is None or isinstance(device, torch.device))
        DEVICE = device

    if isinstance(P, csc_matrix):
        P = to_sparse_csc_tensor(P, DTYPE, DEVICE)
    elif isinstance(P, torch.Tensor):
        if P.layout != torch.sparse_csc:
            raise ValueError("The provided P is a torch Tensor, "
                + " but it is not in sparse CSC format.")

        P = P.to(dtype=DTYPE, device=DEVICE)
    else:
        raise ValueError("P must be either a sparse torch Tensor"
            + " in CSC format, or a sparse scipy matrix"
            + " in CSC format. (And only the upper triangular"
            + " part of the mathematical P it represents"
            + " should be provided.)")

    if isinstance(A, csc_matrix):
        A = to_sparse_csc_tensor(A, DTYPE, DEVICE)
    elif isinstance(A, torch.Tensor):
        if A.layout != torch.sparse_csc:
            raise ValueError("The provided A is a torch Tensor, "
                + " but it is not in sparse CSC format.")
        A = A.to(dtype=DTYPE, device=DEVICE)
    else:
        raise ValueError("A must be either a sparse torch Tensor"
            + " in CSC format, or a sparse scipy matrix"
            + " in CSC format.")

    P_linop = SymmetricOperator(P.shape[0], P, DEVICE)

    q = to_tensor(q, DTYPE, DEVICE)
    b = to_tensor(b, DTYPE, DEVICE)

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

    n = x.shape[0]
    m = y.shape[0]

    cones = cone_utils.parse_cone_dict(cone_dict)

    z = (x, y - s, torch.tensor(1.0, dtype=DTYPE, device=DEVICE))
    u, v, w, = z
    Pi_z = cone_utils.pi(z, cones)

    Dz_Q_Pi_z: lo.LinearOperator = Du_Q(Pi_z, P_linop, A, q, b)
    # TODO?: Need to cache the following (this is 2nd repeat, essentially)
    D_Pi_Kstar_v: lo.LinearOperator = dprojection(v, cones, dual=True)
    M: lo.LinearOperator = form_M(u, v, w[0], Dz_Q_Pi_z, cones)

    def derivative(dP: torch.Tensor | csc_matrix,
                   dA: torch.Tensor | csc_matrix,
                   dq: torch.Tensor | np.ndarray,
                   db: torch.Tensor | np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # TODO: create a helper function to perform this functionality
        # since also done above
        if isinstance(dP, csc_matrix):
            dP = to_sparse_csc_tensor(dP, DTYPE, DEVICE)
        elif isinstance(dP, torch.Tensor):
            if dP.layout != torch.sparse_csc:
                raise ValueError("The provided P is a torch Tensor, "
                    + " but it is not in sparse CSC format.")

            dP = dP.to(dtype=DTYPE, device=DEVICE)
        else:
            raise ValueError("dP must be either a sparse torch Tensor"
                + " in CSC format, or a sparse scipy matrix"
                + " in CSC format. (And only the upper triangular"
                + " part of the mathematical dP it represents"
                + " should be provided.)")

        if isinstance(dA, csc_matrix):
            dA = to_sparse_csc_tensor(dA, DTYPE, DEVICE)
        elif isinstance(dA, torch.Tensor):
            if dA.layout != torch.sparse_csc:
                raise ValueError("The provided dA is a torch Tensor, "
                    + " but it is not in sparse CSC format.")
            dA = dA.to(dtype=DTYPE, device=DEVICE)
        else:
            raise ValueError("dA must be either a sparse torch Tensor"
                + " in CSC format, or a sparse scipy matrix"
                + " in CSC format.")

        dP_linop = SymmetricOperator(dP.shape[0], dP, DEVICE)

        dq = to_tensor(dq, DTYPE, DEVICE)
        db = to_tensor(db, DTYPE, DEVICE)

        # TODO: change dData_Q to return a linop and then move
        # the creation to outside derivative, and then
        # call operator here
        # NOTE: Requires passing in data as a single tensor
        dQ_D = dData_Q(Pi_z, dP_linop, dA, dq, db)

        if torch.allclose(dQ_D, torch.tensor(0, dtype=dtype, device=DEVICE)):
            dz = torch.zeros(dQ_D.shape[0])
        else:
            dz = lsqr(M, dQ_D)

        du, dv, dw = dz[:n], dz[n:n+m], dz[-1]
        dx = du - x * dw
        dy = D_Pi_Kstar_v @ dv - y * dw
        ds = D_Pi_Kstar_v @ dv - dv - s * dw
        return dx, dy, ds

    # TODO: create linear operator

    return derivative


# Callable[[csc_matrix,
#           csc_matrix,
#           np.ndarray,
#           np.ndarray
#           ],
#           Tuple[np.ndarray,
#                 np.ndarray,
#                 np.ndarray]
#         ]
    # The derivative of a primal-dual conic problem at (P, A, q, b)
    # as an abstract linear operator.

# linops.LinearOperator
    # The derivative and adjoint of the conic pair at the solution, (x^star, y^star, s^star).
