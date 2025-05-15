"""
TODO: functionality for efficiently evaluating adjoint applied to primal variable only.
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
from diffqcp.qcp_derivs import Du_Q_efficient, dData_Q_efficient, Du_Q, dData_Q, dData_Q_adjoint
from diffqcp.utils import to_tensor, _convert_problem_data, _get_GPU_settings
from diffqcp.problem_data import ProblemData


class QCP:
    """Quadratic Cone Program.

    Represents the quadratic (convex) cone program given by
    the primal-dual problems

        (P) minimize    (1/2)x^T P x + q^T x
            subject to  Ax + s = b
                        s in K

        (D) minimize    -(1/2)x^T P x - b^T y
            subject to  A^T y + q = 0
                        y in K^*,
    
    where P, A, q, b are mutable problem data, K and K^* are
    immutable problem data, and (x, y, s) are the optimization
    variables.

    (IN PROGRESS)

    Attributes 
    ----------
    data : ProblemData
        Holds P, A, AT, q, b, cones, and other (very) useful information
    x : torch.Tensor
        Primal solution.
    y : torch.Tensor
        Dual solution.
    s : torch.Tensor
        Primal slack variable.
    n : int
        Size (length) of the vector x.
    m : int
        Size (length) of the vectors y and s.
    N : int
        Size (length) of the embedding variable (`== n + m + 1`)
    dtype : torch.dtype
    device : torch.device
    reduce_fp_flops: bool


    Raises
    ------
    
    """

    __slots__ = ('dtype', 'device', 'data', '_x', '_y', '_s', 'n', 'm', 'N', 'reduce_fp_flops',
                    '_Pi_Kstar_v', '_D_Pi_kstar_v', '_Pi_z', '_Dpi_z', '_Dz_Q_Pi_z', '_F')
    
    def __init__(self,
                 P: torch.Tensor | spmatrix,
                 A: torch.Tensor | spmatrix,
                 q: torch.Tensor | np.ndarray | list[float],
                 b: torch.Tensor | np.ndarray | list[float],
                 x: torch.Tensor | np.ndarray | list[float],
                 y: torch.Tensor | np.ndarray | list[float],
                 s: torch.Tensor | np.ndarray | list[float],
                 cone_dict: dict[str, int | list[int]],
                 P_is_upper: bool,
                 dtype: torch.dtype | None = None,
                 device: torch.device | None = None,
                 reduce_fp_flops: bool = False
    ) -> None:
        """
        
        Parameters
        ----------
        P : torch.Tensor | scipy.sparse.spmatrix
            The quadratic component of the objective function.
        x : torch.Tensor | 
            The primal solution variable.
        
        """
        self.dtype, self.device = _get_GPU_settings(P, dtype=dtype, device=device)
        self.data = ProblemData(
            cone_dict=cone_dict, P=P, A=A, q=q, b=b, dtype=self.dtype, device=self.device, P_is_upper=P_is_upper
        )
        self._x = to_tensor(x, dtype=self.dtype, device=self.device)
        self._y = to_tensor(y, dtype=self.dtype, device=self.device)
        self._s = to_tensor(s, dtype=self.dtype, device=self.device)
        self.n = x.shape[0]
        self.m = y.shape[0]
        self.N = self.n + self.m + 1
        self.reduce_fp_flops = reduce_fp_flops
        if not reduce_fp_flops:
            self.form_atoms()

    # any chance I could JIT jvp or vjp...or at least certain parts?
    #   since problem data sizes will be fixed
    #   If I could use proper conditionals and JIT linops lsqr that would
    #   burn a lot of risk for head-to-head against diffcp.
    # PROBABLY SEE PERFORMANCE BEFORE JIT (will eventually JIT, but may not need to for paper experiment)
    
    def form_atoms(self) -> None:
        self._Pi_Kstar_v, self._D_Pi_kstar_v = cone_utils.proj_and_dproj(self._y - self._s, self.data.cones, dual=True)
        # --- jit this part? ----
        self._Pi_z = torch.cat((self._x,
                               self._Pi_Kstar_v,
                               torch.tensor(1.0, dtype=self.dtype, device=self.device)))
        self._Dpi_z = BlockDiag([lo.IdentityOperator(self.n),
                                self._D_Pi_kstar_v,
                                ScalarOperator(torch.tensor(1.0, dtype=self.dtype, device=self.device))],
                                device=self.device)
        self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)
        self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)
        # ----------------------
    
    def jvp(self,
            dP: torch.Tensor | spmatrix,
            dA: torch.Tensor | spmatrix,
            dq: torch.Tensor | np.ndarray,
            db: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reduce_fp_flops:
            self.form_atoms()

        # TODO (quill): data checks; see `problem_data.py`
        pass

    def vjp(self,
            dx: torch.Tensor | np.ndarray | list[float],
            dy: torch.Tensor | np.ndarray | list[float],
            ds: torch.Tensor | np.ndarray | list[float]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.reduce_fp_flops:
            self.form_atoms()
        pass
    
    def update(self,
               P: torch.Tensor | spmatrix,
               A: torch.Tensor | spmatrix,
               q: torch.Tensor | np.ndarray | list[float],
               b: torch.Tensor | np.ndarray | list[float],
               x: torch.Tensor | np.ndarray | list[float],
               y: torch.Tensor | np.ndarray | list[float],
               s: torch.Tensor | np.ndarray | list[float]
    ) -> None:
        self.data.P = P
        self.data.A = A
        self.data.q = q
        self.data.b = b
        self.update_solution(x, y, s)
    
    def update_data(self,
                    P: torch.Tensor | spmatrix,
                    A: torch.Tensor | spmatrix,
                    q: torch.Tensor | np.ndarray | list[float],
                    b: torch.Tensor | np.ndarray | list[float]
    ) -> None:
        # TODO (quill): make notes somewhere that
        #   1. Make note that setting `reduce_fp_flops = True` can reduce overall possible flop count reductions
        #       if it's possible to not redo all atoms.
        #   2. assuming a user wouldn't do update_data and update_solution, else you redo some computation (doesn't break things).
        self.data.P = P
        self.data.A = A
        self.data.q = q
        self.data.b = b
        
        if not self.reduce_fp_flops:
            self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)
            self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)


    def update_solution(self,
                        x: torch.Tensor | np.ndarray | list[float],
                        y: torch.Tensor | np.ndarray | list[float],
                        s: torch.Tensor | np.ndarray | list[float]
    ) -> None:
        # TODO: think about adding equality checks to previous values to see if we do need to recompute the projections?
        self._x = to_tensor(x, dtype=self.dtype, device=self.device)
        self._y = to_tensor(y, dtype=self.dtype, device=self.device)
        self._s = to_tensor(s, dtype=self.dtype, device=self.device)

        if not self.reduce_fp_flops:
            self.form_atoms()
    
    def _update_data_dependent_atoms(self):
        self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)
        self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)
    
    @property
    def P(self) -> torch.Tensor | SymmetricOperator:
        return self.data.P
    
    @P.setter
    def P(self, P):
        self.data.P = P
    
    @property
    def A(self) -> torch.Tensor | spmatrix:
        return self.data.A

    @A.setter
    def A(self, A: torch.Tensor | spmatrix) -> None:
        self.data.A = A
        self._update_data_dependent_atoms()

    @property
    def q(self) -> torch.Tensor | np.ndarray:
        return self.data.q

    @q.setter
    def q(self, q: torch.Tensor | np.ndarray) -> None:
        self.data.q = q
        self._update_data_dependent_atoms()

    @property
    def b(self) -> torch.Tensor | np.ndarray:
        return self.data.b

    @b.setter
    def b(self, b: torch.Tensor | np.ndarray) -> None:
        self.data.b = b
        self._update_data_dependent_atoms()

    @property
    def x(self) -> torch.Tensor:
        return self._x
    
    @x.setter
    def x(self, x):
        self._x = to_tensor(x, self.dtype, self.device)

        if not self.reduce_fp_flops:
            self._Pi_z = torch.cat((self._x,
                                self._Pi_Kstar_v,
                                torch.tensor(1.0, dtype=self.dtype, device=self.device)))
            self._Dpi_z = BlockDiag([lo.IdentityOperator(self.n),
                                    self._D_Pi_kstar_v,
                                    ScalarOperator(torch.tensor(1.0, dtype=self.dtype, device=self.device))],
                                    device=self.device)
            self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)
            self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)

    @property
    def y(self) -> torch.Tensor:
        return self._y
    
    @y.setter
    def y(self, y):
        self._y = to_tensor(y, self.dtype, self.device)

        if not self.reduce_fp_flops:
            self.form_atoms()

    @property
    def s(self) -> torch.Tensor:
        return self._s
    
    @s.setter
    def s(self, s):
        self._s = to_tensor(s, self.dtype, self.device)

        if not self.reduce_fp_flops:
            self.form_atoms()

# ====== (below) IN PROCESS OF BEING MOVED INTO THE ABOVE CLASS ======

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
    F = (Dz_Q_Pi_z @ DPi_z) - DPi_z + lo.IdentityOperator(n + m + 1)

    def derivative(dP: torch.Tensor | spmatrix,
                   dA: torch.Tensor | spmatrix,
                   dq: torch.Tensor | np.ndarray,
                   db: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        dP, dA, dq, db = _convert_problem_data(dP, dA, dq, db, dtype=DTYPE, device=DEVICE)
        dP_linop = SymmetricOperator(n, dP, DEVICE)

        d_DN = dData_Q(Pi_z, dP_linop, dA, dq, db)

        if torch.allclose(d_DN, torch.tensor(0, dtype=dtype, device=DEVICE)):
            dz = torch.zeros(d_DN.shape[0])
        else:
            dz = lsqr(F, -d_DN)

        dr, dw, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dx = dr - x * dz_N
        dy = D_Pi_Kstar_v @ dw - y * dz_N
        ds = D_Pi_Kstar_v @ dw - dw - s * dz_N
        return dx, dy, ds

    return derivative
