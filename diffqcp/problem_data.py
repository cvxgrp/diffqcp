import torch
import numpy as np
from scipy.sparse import spmatrix, sparray

import diffqcp.cones as cone_utils
from diffqcp.linops import SymmetricOperator
from diffqcp.utils import to_sparse_csr_tensor, to_tensor

class ProblemData:
    """Convenience class for storing QCP problem data and metadata.

    Most efficient when P and A are provided as CSR 
    (and continue to be provided this way during a "training cycle")

    Attributes
    ----------
    P : torch.Tensor | lo.LinearOperator
        The matrix P in the quadratic part of the objective function of a QCP.
    A : torch.Tensor
        The matrix A defininig the constraints of a QCP.
    q : torch.Tensor
        The vector q in the linear part of the objective function of a QCP.
    b : torch.Tensor
        The vecotr b on the RHS of the constraint equality of a QCP.
    cones : list[tuple[str, int | list[int]]]
        immutable.

    Notes
    -----
    - TODO (quill): for next phase, consider coalescing memory / how to efficiently store
        data matrices and cone vectors.
    """

    slots = (
        '_P',
        '_A',
        'q',
        'b',
        'cones',
        'P_is_upper',
        'dtype',
        'device',
        'P_original_nnz',
        'P_filtered_nnz',
        'P_nonzero_mask',
        'P_rows',
        'P_cols',
        'PT_perm',
        'P_diag_mask',
        'self.P_diag_indices',
        'Pcrow_indices_T',
        'Pcol_indices_T',
        'A_original_nnz',
        'A_filtered_nnz',
        'A_nonzero_mask',
        'A_rows',
        'A_cols',
        'AT_perm',
        'Acrow_indices_T',
        'Acol_indices_T'
    )
    
    def __init__(
        self,
        cone_dict: dict[str, int | list[int]],
        P: torch.Tensor | spmatrix | sparray,
        A: torch.Tensor | spmatrix | sparray,
        q: torch.Tensor | np.ndarray | list[float],
        b: torch.Tensor | np.ndarray | list[float],
        dtype: torch.dtype,
        device: torch.device,
        P_is_upper: bool = True,
    ) -> None:
        # for cone info, is there any more pre-computation
        # I can do?
        self.dtype = dtype
        self.device = device
        self.cones = cone_utils.parse_cone_dict(cone_dict)

        self.q = to_tensor(q, dtype=self.dtype, device=self.device)
        self.b = to_tensor(b, dtype=self.dtype, device=self.device)
        self.n = self.q.shape[0]
        self.m = self.b.shape[0]
        
        self.obj_matrix_init(P, P_is_upper)
        self.constr_matrix_init(A)
        

    def obj_matrix_init(
        self,
        P: torch.Tensor,
        P_is_upper: bool
    ) -> None:
        """
        Functionality:
        - Remove any potential explicit zeros from P
        - save the nonzero rows and column indices
        - if P_is_upper
            - store efficient ways of computing the diagonal and transpose of P
            - form P as a `linops.LinearOperator`
        """
        # TODO (quill): the following can be smarter.
        #   - Namely, fewer conversions
        #   - reusing attributes we may already know given `P`'s provided type.
        self.P_is_upper = P_is_upper
        if isinstance(P, spmatrix) or isinstance(P, sparray):
            if P.dtype == np.float64:
                dtype = torch.float64
            elif P.dtype == np.float32:
                dtype = torch.float32
            elif P.dtype == np.float16:
                dtype = torch.float16
            else:
                raise ValueError("`P`'s dtype must be a float16, 32, or 64."
                                    + f" Instead it is {P.dtype}")
            # if P is in this block then it is on CPU. Leave on CPU for now.
            P = to_sparse_csr_tensor(P, dtype=dtype, device=None)
        elif isinstance(P, torch.Tensor):
            # need to eliminate zeros
            P = P.to_sparse_csr() # returns self if P is already sparse csr
            P_coo = P.to_sparse_coo()
            # NOTE (quill): so long as `P` doesn't violate CSR's canonical form
            #   (e.g., have duplicate entries or unsorted indices), we can assume
            #   `P.values()`` have the same ordering as `P_coo.values()`
            self.P_original_nnz = P.values().shape[0]
            nonzero_mask = P_coo.values() != 0
            # indices have shape (2, num_nonzero), so each col
            # is the nonzero row and col, respectively
            indices = P_coo.indices()[:, nonzero_mask] # NOTE (quill): might need to coalesce
            values = P_coo.values()[nonzero_mask]
            self.P_nonzero_mask = to_tensor(nonzero_mask, dtype=torch.int64, device=self.device)
            self.P_filtered_nnz = values.shape[0]
            rows, cols = indices[0], indices[1]
            # copy the following to device, but keep original rows, cols for now
            #   (if we are using device)
            self.P_rows = torch.tensor(rows, dtype=torch.int64, device=self.device)
            self.P_cols = torch.tensor(cols, dtype=torch.int64, device=self.device)

            P_clean = torch.sparse_coo_tensor(indices, values, size=P.shape).coalesce()
            P = to_sparse_csr_tensor(P_clean).to(dtype=self.dtype, device=self.device)

            if self.P_is_upper:
                rows_T, cols_T = cols, rows
                transposed_idx = rows_T * P.shape[0] + cols_T # TODO (quill): check this
                sorted_perm = torch.argsort(transposed_idx)

                transposed_indices = torch.stack([rows_T[sorted_perm], cols_T[sorted_perm]], dim=0)
                self.PT_perm = to_tensor(sorted_perm, dtype=self.dtype, device=self.device)
                dummy_values = torch.ones_like(values, dtype=values.dtype)
                transposed_coo = torch.sparse_coo_tensor(
                    transposed_indices, dummy_values, size=(self.n, self.n)
                ).coalesce()
                transposed_csr = transposed_coo.to_sparse_csr()

                self.Pcrow_indices_T = transposed_csr.crow_indices()
                self.Pcol_indices_T = transposed_csr.col_indices()
                PT = self._P_transpose(values)
                self.Pcrow_indices_T = to_tensor(self.Pcrow_indices_T, dtype=self.dtype, device=self.device)
                self.Pcol_indices_T = to_tensor(self.Pcol_indices_T, dtype=self.dtype, device=self.device)

                diag_mask = rows == cols
                diag_indices = rows[diag_mask]
                diag_values = values[diag_mask]
                diag = torch.zeros(self.n, dtype=self.dtype, device=values.dtype)
                diag[diag_indices] = diag_values
                diag = to_tensor(diag, dtype=self.dtype, device=self.device)

                self.P_diag_mask = to_tensor(diag_mask, dtype=self.dtype, device=self.device)
                self.P_diag_indices = to_tensor(diag_indices, dtype=self.dtype, device=self.device)

                mv = lambda v : P @ v + PT @ v - diag * v
                self._P = SymmetricOperator(self.n, op=mv, device=self.device)
            else:
                self.PT_perm = None
                self.diag_mask = None
                self.Pcrow_indices_T = None
                self.Pcol_indices_T = None
                self._P = P
        else:
            raise ValueError("`P` must be a `scipy` `spmatrix` (or `sparray`) or a `torch.Tensor`."
                             + f" It is {type(A)}.")
        
    def _P_transpose(self, values: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
        """
        No checks since this is only accessed locally.
        uses `perm` when initializing `ProblemData` in case the data is on the device and then being
        moved to host since at this point `self.AT_perm` is already on host.
        """
        perm = self.AT_perm if perm is None else perm
        transposed_values = values[perm]
        transposed_values = to_tensor(transposed_values, dtype=self.dtype, device=transposed_values.device)
        return torch.sparse_csr_tensor(
            self.Pcrow_indices_T, self.Pcol_indices_T, transposed_values, size=(self.n, self.n), device=self.device
        )
    
    def constr_matrix_init(self, A: torch.Tensor | spmatrix) -> None:
        """
        Functionality:
        - Remove any potential explicit zeros from A
        - save the nonzero row and column indices
        - save an efficient way of creating A's transpose
        """
        if isinstance(A, spmatrix) or isinstance(A, sparray):
            # NOTE (quill): in case spmatrix is provided on first iteration of
            #   learning loop and then torch tensors from then on, we need
            #   the nonzero mask computed in the following if block.
            #   Consequently, just convert to torch CSR tensor and then
            #   Let next block handle setup.
            #   (I will leave commented out code for future reference.)
            
            # A = A.copy() # don't throw away user data
            # A.eliminate_zeros() # <- this operates in place; (future reference)
            
            if A.dtype == np.float64:
                dtype = torch.float64
            elif A.dtype == np.float32:
                dtype = torch.float32
            elif A.dtype == np.float16:
                dtype = torch.float16
            else:
                raise ValueError("`A`'s dtype must be a float16, 32, or 64."
                                    + f" Instead it is {A.dtype}")
            # if A is in this block then it is on CPU, so leave on CPU for now.
            A = to_sparse_csr_tensor(A, dtype=dtype, device=None)
            
            # === For reference ===
            # rows, cols = A.nonzero()
            # self.A_rows = torch.tensor(rows, dtype=torch.int64, device=self.device)
            # self.A_cols = torch.tensor(cols, dtype=torch.int64, device=self.device)
            
            # rows_T, cols_T = cols, rows
            # transposed_idx = rows_T * A.shape[0] + cols_T
            # sorted_perm = np.argsort(transposed_idx)
            # self.AT_perm = torch.tensor(sorted_perm, dtype=torch.int64, device=self.device)
            
        if isinstance(A, torch.Tensor):
            # None of the following throws away user data.
            A = A.to_sparse_csr() # returns self if A is already sparse csr.
            A_coo = A.to_sparse_coo()

            # NOTE (quill): so long as `A` doesn't violate CSR's canonical form
            #   (e.g., have duplicate entries or unsorted indices), we can assume
            #   `A.values()`` have the same ordering as `A_coo.values()`
            self.A_original_nnz = A.values().shape[0]
            nonzero_mask = A_coo.values() != 0
            # indices have shape (2, num_nonzero); so each col
            # is the nonzero row and col, respectively
            indices = A_coo.indices()[:, nonzero_mask] # NOTE (quill): might need to coalesce
            values = A_coo.values()[nonzero_mask]
            self.A_nonzero_mask = to_tensor(nonzero_mask, dtype=torch.int64, device=self.device)
            self.A_filtered_nnz = values.shape[0]
            rows, cols = indices[0], indices[1]
            # save since need rows and cols when constructing vector outer products
            self.A_rows = to_tensor(rows, dtype=torch.int64, device=self.device)
            self.A_cols = to_tensor(cols, dtype=torch.int64, device=self.device)

            # A_clean is on whatever device A was allocated on.
            A_clean = torch.sparse_coo_tensor(indices, values, size=A.shape).coalesce()
            self._A = A_clean.to_sparse_csr().to(dtype=self.dtype, device=self.device)

            rows_T, cols_T = cols, rows
            transposed_idx = rows_T * A.shape[0] + cols_T
            sorted_perm = torch.argsort(transposed_idx)
        else:
            raise ValueError("`P` must be a `scipy` `spmatrix` (or `sparray`) or a `torch.Tensor`."
                             + f" It is {type(A)}.")

        # Now create infrastructure for cheap transposes.
        transposed_indices = torch.stack([rows_T[sorted_perm], cols_T[sorted_perm]], dim=0)
        self.AT_perm = to_tensor(sorted_perm, dtype=torch.int64, device=self.device)
        # NOTE: we are staying on the device that A came to us on for all intermediate operations.
        dummy_values = torch.ones_like(values, dtype=values.dtype)
        transposed_coo = torch.sparse_coo_tensor(
            transposed_indices, dummy_values, size = (self.n, self.m)
        ).coalesce()
        transposed_csr = transposed_coo.to_sparse_csr()

        self.Acrow_indices_T = transposed_csr.crow_indices()
        self.Acol_indices_T = transposed_csr.col_indices()
        self._AT = self._A_transpose(self._A.values())
        self.Acrow_indices_T = to_tensor(self.Acrow_indices_T, dtype=self.dtype, device=self.device)
        self.Acol_indices_T = to_tensor(self.Acol_indices_T, dtype=self.dtype, device=self.device)
            
    def _A_transpose(self, values: torch.Tensor, perm: torch.Tensor | None = None) -> torch.Tensor:
        """
        supply with values of a csr array
        """
        perm = self.AT_perm if perm is None else perm
        # add check for number of nonnegative?
        transposed_values = values[perm]
        transposed_values = to_tensor(transposed_values, dtype=self.dtype, device=transposed_values.device)
        return torch.sparse_csr_tensor(
            self.Acrow_indices_T, self.Acol_indices_T, transposed_values, size=(self.n, self.m), device=self.device
        )

    @property
    def P(self) -> torch.Tensor | SymmetricOperator:
        return self._P
    
    @P.setter
    def P(self, P) -> None:
        if isinstance(P, spmatrix) or isinstance(P, sparray):
            P = to_sparse_csr_tensor(P, dtype=self.dtype, device=self.device)
        if isinstance(P, torch.Tensor):
            # remove explicit zeros
            P = P.to_sparse_csr() # returns self if P is already sparse csr
            values = P.values()
            values = to_tensor(values, dtype=self.dtype, device=self.device)
            if values.shape[0] == self.P_original_nnz:
                values = values[self.P_nonzero_mask]
            elif values.shape[0] != self.P_filtered_nnz:
                raise ValueError("The new `P` must have the same sparsity pattern as the"
                                 + " original or filtered `P`. Since the provided tensor doesn't"
                                 + " have the same number of nonzero elements as the"
                                 + " original or filtered `P`, its sparsity pattern differs.")

            if not self.P_is_upper:
                self._P = torch.sparse_csr_tensor(
                    self._P.crow_indices(), self._P.col_indices(), values=values, size=(self.n, self.n), device=self.device
            )
            else:
                PT = self._P_transpose(values)
                diag_values = values[self.P_diag_mask]
                diag = torch.zeros(self.n, dtype=self.dtype, device=self.dtype)
                diag[self.P_diag_indices] = diag_values
                mv = lambda v : P @ v + PT @ v - diag * v
                self._P = SymmetricOperator(self.n, op=mv, device=self.device)
        else:
            raise ValueError("P must be a `scipy` `spmatrix` (or `sparray`) or a `torch.Tensor`."
                             + f" It is {type(P)}.")
                    
    @property
    def A(self) -> torch.Tensor:
        return self._A
    
    @A.setter
    def A(self, A) -> None:
        if isinstance(A, spmatrix) or isinstance(A, sparray):
            A = to_sparse_csr_tensor(A, dtype=self.dtype, device=self.device)
        if isinstance(A, torch.Tensor):
            # need to remove explicit zeros
            A = A.to_sparse_csr() # returns self if A is already sparse csr.
            values = A.values()
            # assume A is on self.device when provided now
            values = to_tensor(values, dtype=self.dtype, device=self.device)
            if values.shape[0] == self.A_original_nnz:
                values = values[self.A_nonzero_mask] 
            elif values.shape[0] != self.A_filtered_nnz:
                raise ValueError("The new `A` must have the same sparsity pattern as the"
                                 + " original or filtered `A`. Since the provided tensor doesn't"
                                 + " have the same number of nonzero elements as the"
                                 + " original or filtered `A`, its sparsity pattern differs.")
        
            self._A = torch.sparse_csr_tensor(
                    self._A.crow_indices(), self._A.col_indices(), values=values, size=(self.m, self.n), device=self.device
                )
            self._AT = self._A_transpose(values)
        else:
            raise ValueError("A must be a `scipy` `spmatrix` (or `sparray`) or a `torch.Tensor`."
                             + f" It is {type(A)}.")

    @property
    def AT(self) -> torch.Tensor:
        return self._AT
    
    def convert_perturbations(
            self,
            dP: torch.Tensor | spmatrix | sparray,
            dA: torch.Tensor | spmatrix | sparray,
            dq: torch.Tensor,
            db: torch.Tensor
    ) -> tuple[torch.Tensor | SymmetricOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(dP, spmatrix) or isinstance(dP, sparray):
            dP = to_sparse_csr_tensor(dP, dtype=self.dtype, device=self.device)
        if isinstance(dP, torch.Tensor):
            dP = dP.to_sparse_csr() # returns self if P is already sparse csr
            values = dP.values()
            dP_nnz = values.shape[0]
            if dP_nnz != self.P_original_nnz and dP_nnz != self.P_filtered_nnz:
                raise ValueError("`dP` must have the same sparsity pattern as the"
                                 + " original or filtered `P`. Since the provided tensor doesn't"
                                 + " have the same number of nonzero elements as the"
                                 + " original or filtered `P`, its sparsity pattern differs.")
        dP = to_tensor(dP, dtype=self.dtype, device=self.device)

        if self.P_is_upper:
            values = dP.values()
            dPT = self._P_transpose(values)
            diag_values = values[self.P_diag_mask]
            diag = torch.zeros(self.n, dtype=self.dtype, device=self.dtype)
            diag[self.P_diag_indices] = diag_values
            mv = lambda v : dP @ v + dPT @ v - diag * v
            dP = SymmetricOperator(self.n, op=mv, device=self.device)

        if isinstance(dA, spmatrix) or isinstance(dA, sparray):
            dA = to_sparse_csr_tensor(dA, dtype=self.dtype, device=self.device)
        if isinstance(dA, torch.Tensor):
            dA = dA.to_sparse_csr() # returns self if P is already sparse csr
            values = dA.values()
            dA_nnz = values.shape[0]
            if dA_nnz != self.P_original_nnz and dA_nnz != self.P_filtered_nnz:
                raise ValueError("`dP` must have the same sparsity pattern as the"
                                 + " original or filtered `P`. Since the provided tensor doesn't"
                                 + " have the same number of nonzero elements as the"
                                 + " original or filtered `P`, its sparsity pattern differs.")
        dA = to_tensor(dA, dtype=self.dtype, device=self.device)
        dAT = self._A_transpose(dA.values())

        dq = to_tensor(dq, dtype=self.dtype, device=self.device)
        db = to_tensor(db, dtype=self.dtype, device=self.device)

        return dP, dA, dAT, dq, db