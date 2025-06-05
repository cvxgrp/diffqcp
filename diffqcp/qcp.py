"""
TODO: functionality for efficiently evaluating adjoint applied to primal variable only.
"""
from typing import Union

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import spmatrix, sparray
import torch
from torch import Tensor
import linops as lo
from linops.lsqr import lsqr
from jaxtyping import Float

from diffqcp.linops import SymmetricOperator, BlockDiag, ScalarOperator, _sLinearOperator
import diffqcp.cones as cone_utils
from diffqcp.qcp_derivs import Du_Q_efficient, dData_Q_efficient, dData_Q_adjoint_efficient
from diffqcp.utils import to_tensor, _get_GPU_settings, from_torch_csr_to_scipy_csc
from diffqcp.problem_data import ProblemData

# TODO (quill): check carefully where you use .to vs to_tensor

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

    Attributes 
    ----------
    data : ProblemData TODO (quill): do I want to expose this to the user or not? 
        Holds P, A, AT, q, b, cones, and other (very) useful information -> yes, remove this
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
    does_reduce_fp_flops: bool (immutable)
    """

    __slots__ = (
        'dtype',
        'device',
        'data',
        '_x',
        '_y',
        '_s',
        'n',
        'm',
        'N',
        '_reduce_fp_flops',
        '_atoms_computed',
        '_Pi_Kstar_v',
        '_D_Pi_kstar_v',
        '_Pi_z',
        '_Dpi_z',
        '_Dz_Q_Pi_z',
        '_F',
        '_FT'
    )
    
    def __init__(
        self,
        P: Float[Union[Tensor, sparray, spmatrix], 'n n'],
        A: Float[Union[Tensor, sparray, spmatrix], 'm n'],
        q: Float[Union[Tensor, np.ndarray], 'n'],
        b: Float[Union[Tensor, np.ndarray], 'm'],
        x: Float[Union[Tensor, np.ndarray], 'n'],
        y: Float[Union[Tensor, np.ndarray], 'm'],
        s: Float[Union[Tensor, np.ndarray], 'm'],
        cone_dict: dict[str, int | list[int]],
        P_is_upper: bool,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        reduce_fp_flops: bool = False
    ) -> None:
        """
        Parameters
        ----------
        P :
            The quadratic component of the objective function.
        x : 
            The primal solution.
        y :
            The dual solution.
        s :
            The primal slack solution.
        
        """
        self.dtype, self.device = _get_GPU_settings(P, dtype=dtype, device=device)
        self.data = ProblemData(
            cone_dict=cone_dict, P=P, A=A, q=q, b=b, dtype=self.dtype, device=self.device, P_is_upper=P_is_upper
        )
        self._x = to_tensor(x, dtype=self.dtype, device=self.device)
        self._y = to_tensor(y, dtype=self.dtype, device=self.device)
        self._s = to_tensor(s, dtype=self.dtype, device=self.device)
        self.n = self._x.shape[0]
        self.m = y.shape[0]
        self.N = self.n + self.m + 1
        self._reduce_fp_flops = reduce_fp_flops
        self._atoms_computed = False
        self.jvp_lsqr_residual: float | None = None
        self.vjp_lsqr_residual: float | None = None
        if not self._reduce_fp_flops:
            self._form_atoms()
    
    def _form_atoms(self) -> None:
        self._Pi_Kstar_v, self._D_Pi_kstar_v = cone_utils.proj_and_dproj(self._y - self._s, self.data.cones, dual=True)
        self._Pi_z = torch.cat((self._x,
                               self._Pi_Kstar_v,
                               torch.tensor(1.0, dtype=self.dtype, device=self.device).unsqueeze(-1)))
        self._Dpi_z = BlockDiag([lo.IdentityOperator(self.n),
                                self._D_Pi_kstar_v,
                                ScalarOperator(torch.tensor(1.0, dtype=self.dtype, device=self.device))],
                                device=self.device)
        self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)

        # TODO (quill): decouple and remove overhead
        #   Note: did this because the commented out bit below was causing dtype and device errors during the adjoint lsqr call.
        def mv(du: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
            dpi_z_du = self._Dpi_z @ du
            return self._Dz_Q_Pi_z @ dpi_z_du - dpi_z_du + du
        
        def rv(dv: Float[Tensor, 'n+m+1']) -> Float[Tensor, 'n+m+1']:
            return self._Dpi_z.T @ (self._Dz_Q_Pi_z.T @ dv) - self._Dpi_z.T @ dv + dv
        
        self._F = _sLinearOperator(self.N, self.N, mv, rv, device=self.device)

        # self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)
        # self._F.device = self.device
        # self._FT = self._F.T
        # self._FT.device = self.device
        self._atoms_computed = True
        self.jvp_lsqr_residual = None
        self.vjp_lsqr_residual = None
    
    def jvp(
        self,
        dP: Float[Union[Tensor, sparray, spmatrix], 'n n'],
        dA: Float[Union[Tensor, sparray, spmatrix], 'm n'],
        dq: Float[Union[Tensor, np.ndarray], 'n'],
        db: Float[Union[Tensor, np.ndarray], 'm'],
    ) -> tuple[
            Float[Tensor, 'n'], Float[Tensor, 'm'], Float[Tensor, 'm']
        ]:
        if not self._atoms_computed:
            self._form_atoms()

        dP, dA, dAT, dq, db = self.data.convert_perturbations(dP, dA, dq, db)
        n, m = self.n, self.m
        d_data_N = dData_Q_efficient(self._Pi_z, dP, dA, dAT, dq, db)
        if torch.allclose(d_data_N, torch.tensor(0, dtype=self.dtype, device=self.device)):
            dz = torch.zeros(d_data_N.shape[0], dtype=self.dtype, device=self.device)
        else:
            dz = lsqr(self._F, -d_data_N)
            self.jvp_lsqr_residual = torch.linalg.norm(self._F @ dz + d_data_N)**2
            # TODO (quill): make this save optional?
        
        dz_n, dz_m, dz_N = dz[:n], dz[n:n+m], dz[-1]
        dx = dz_n - self._x * dz_N
        D_Pi_K_star_v_dz_m = self._D_Pi_kstar_v @ dz_m
        dy = D_Pi_K_star_v_dz_m - self._y * dz_N
        ds = D_Pi_K_star_v_dz_m - dz_m - self._s * dz_N
        return dx, dy, ds

    def vjp(
        self,
        dx: Float[Union[Tensor, np.ndarray], 'n'],
        dy: Float[Union[Tensor, np.ndarray], 'm'],
        ds: Float[Union[Tensor, np.ndarray], 'm']
    ) -> tuple[
            Float[Tensor, 'n n'], Float[Tensor, 'm n'], Float[Tensor, 'n'], Float[Tensor, 'm']
        ]:
        """
        Returns dP and dA as `Tensor`s in `sparse_csr` layout.
        """
        if not self._atoms_computed:
            self._form_atoms()
        
        dx = to_tensor(dx, dtype=self.dtype, device=self.device)
        dy = to_tensor(dy, dtype=self.dtype, device=self.device)
        ds = to_tensor(ds, dtype=self.dtype, device=self.device)

        dz = torch.cat(
            (dx,
             self._D_Pi_kstar_v @ (dy + ds) - ds,
             - (self._x @ dx + self._y @ dy + self._s @ ds).unsqueeze(-1) )
        )

        if torch.allclose(dz, torch.tensor(0, dtype=self.dtype, device=self.device)):
            d_data_N = torch.zeros(dz.shape[0], dtype=self.dtype, device=self.device)
        else:
            d_data_N = lsqr(self._F.T, -dz)
            # TODO (quill): make this save optional?
            self.vjp_lsqr_residual = torch.linalg.norm(self._F.T @ d_data_N + dz)**2

        return dData_Q_adjoint_efficient(
            self._Pi_z, d_data_N[:self.n], d_data_N[self.n:self.n+self.m], d_data_N[-1],
            P_rows=self.data.P_rows, P_cols=self.data.P_cols, Pcrow_indices=self.data.Pcrow_indices,
            Pcol_indices=self.data.Pcol_indices, A_rows=self.data.A_rows, A_cols=self.data.A_cols,
            Acrow_indices=self.data.Acrow_indices, Acol_indices=self.data.Acol_indices
        )
    
    def update(
        self,
        P: Float[Union[Tensor, sparray, spmatrix], 'n n'],
        A: Float[Union[Tensor, sparray, spmatrix], 'm n'],
        q: Float[Union[Tensor, np.ndarray], 'n'],
        b: Float[Union[Tensor, np.ndarray], 'm'],
        x: Float[Union[Tensor, np.ndarray], 'n'],
        y: Float[Union[Tensor, np.ndarray], 'm'],
        s: Float[Union[Tensor, np.ndarray], 'm']
    ) -> None:
        self.update_data(P, A, q, b)
        self.update_solution(x, y, s)
    
    def update_data(
        self,
        P: Float[Union[Tensor, sparray, spmatrix], 'n n'],
        A: Float[Union[Tensor, sparray, spmatrix], 'm n'],
        q: Float[Union[Tensor, np.ndarray], 'n'],
        b: Float[Union[Tensor, np.ndarray], 'm'],
    ) -> None:
        """
        If changing data then will have to change solution, so don't do any
        atom forming.
        """
        self._atoms_computed = False
        self.data.P = P
        self.data.A = A
        self.data.q = q
        self.data.b = b
    
    def perturb_data(
        self,
        dP: Float[Union[Tensor, sparray, spmatrix], 'n n'],
        dA: Float[Union[Tensor, sparray, spmatrix], 'm n'],
        dq: Float[Union[Tensor, np.ndarray], 'n'],
        db: Float[Union[Tensor, np.ndarray], 'm']
    ) -> None:
        """
        A more efficient version of `update_data` that doesn't require
        forming `dP` when `P` is a linop.
        (Cannot currently add a `Tensor` in sparse CSR layout to a linop.)
        """
        self._atoms_computed = False
        self.data.perturb_P(dP)
        self.data.A = self.data.A + dA
        self.data.q = self.data.q + dq
        self.data.b = self.data.b + db
    
    def update_solution(
        self,
        x: Float[Union[Tensor, np.ndarray], 'n'],
        y: Float[Union[Tensor, np.ndarray], 'm'],
        s: Float[Union[Tensor, np.ndarray], 'm']
    ) -> None:
        self._atoms_computed = False
        self._x = to_tensor(x, dtype=self.dtype, device=self.device)
        self._y = to_tensor(y, dtype=self.dtype, device=self.device)
        self._s = to_tensor(s, dtype=self.dtype, device=self.device)

        if not self._reduce_fp_flops:
            self._form_atoms()
    
    # TODO (quill): add back `_reduce_fp_flops` functionality that exploits what actually
    #   needs to be updated.
    
    # def _update_data_dependent_atoms(self):
    #     # TODO (quill): update this to match the linop form in `_form_atoms`
    #     self._Dz_Q_Pi_z: lo.LinearOperator = Du_Q_efficient(self._Pi_z, self.data.P, self.data.A, self.data.AT, self.data.q, self.data.b)
    #     self._F = (self._Dz_Q_Pi_z @ self._Dpi_z) - self._Dpi_z + lo.IdentityOperator(self.N)
    
    @property
    def P(self) -> Float[Tensor, 'n n'] | SymmetricOperator:
        return self.data.P
    
    @P.setter
    def P(self, P: Float[Union[Tensor, sparray, spmatrix], 'n n']):
        self.data.P = P
        self._atoms_computed = False
    
    @property
    def A(self) -> Float[Tensor, 'm n']:
        return self.data.A

    @A.setter
    def A(self, A: Float[Union[Tensor, sparray, spmatrix], 'm n']) -> None:
        self.data.A = A
        self._atoms_computed = False

    @property
    def q(self) -> Float[Tensor, 'n']:
        return self.data.q

    @q.setter
    def q(self, q: Float[Union[Tensor, np.ndarray], 'n']) -> None:
        self.data.q = q
        self._atoms_computed = False

    @property
    def b(self) -> Float[Tensor, 'm']:
        return self.data.b

    @b.setter
    def b(self, b: Float[Union[Tensor, np.ndarray], 'm']) -> None:
        self.data.b = b
        self._atoms_computed = False

    @property
    def x(self) -> Float[Tensor, 'n']:
        return self._x
    
    @x.setter
    def x(self, x: Float[Union[Tensor, np.ndarray], 'n']):
        self._x = to_tensor(x, self.dtype, self.device)
        self._atoms_computed = False

    @property
    def y(self) -> Float[Tensor, 'm']:
        return self._y
    
    @y.setter
    def y(self, y: Float[Union[Tensor, np.ndarray], 'm']):
        self._y = to_tensor(y, self.dtype, self.device)
        self._atoms_computed = False

    @property
    def s(self) -> Float[Tensor, 'm']:
        return self._s
    
    @s.setter
    def s(self, s: Float[Union[Tensor, np.ndarray], 'm']):
        self._s = to_tensor(s, self.dtype, self.device)
        self._atoms_computed = False

    @property
    def does_reduce_fp_flops(self):
        return self._reduce_fp_flops
    
    def get_Acsc_cpu(self) -> Float[spmatrix, "m n"]:
        return from_torch_csr_to_scipy_csc(self.data.A)

    def get_Pcsc_cpu_upper(self) -> Float[spmatrix, "n n"]:
        if self.data.P_is_upper:
            return from_torch_csr_to_scipy_csc(self.data.P_upper)
        else:
            P = from_torch_csr_to_scipy_csc(self.data._P)
            return sparse.triu(P).tocsc()
