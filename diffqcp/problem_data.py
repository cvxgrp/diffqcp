import torch
import numpy as np
from scipy.sparse import spmatrix, sparray

import diffqcp.cones as cone_utils
from diffqcp.linops import SymmetricOperator
from diffqcp.utils import to_sparse_csr_tensor, to_tensor

class ProblemData:

    # TODO (quill): remove all nonzeros like done in diffcp
    # TODO (quill): for batching PSD cone, that functionality can probably just go
    #   in proj_and_dproj() -> instead of looping through PSD cones (or SOCs for that
    #   matter), pass to wrappers. Potentially useful to do parsing about whether this happens
    #   here (e.g., which indices in the cone list go with each other)?
    # TODO (quill): for next phase, need to think about coalescing memory / topics
    #   like that when considering how to hold data matrices and cone computations on a
    #   single vecotr.

    slots = (
        '_P',
        '_A',
        'q',
        'b',
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
        """
        # TODO (quill): make the following smarter
        #   -> create mappings for cheap conversions
        #   -> if already COO then don't convert to CSR yet.
        #   -> if spmatrix then use their built in utilities
        # TODO (quill): unit test by
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
            # if P is in this block then it is on CPU, so leave on CPU for now.
            P = to_sparse_csr_tensor(P, dtype=dtype, device=None)
        elif isinstance(P, torch.Tensor):
            # need to eliminate zeros
            # explicit zeros may be in there due to triangular structure?
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
            self.P_rows = torch.tensor(rows, dtype=torch.int64, device=self.device)
            self.P_cols = torch.tensor(cols, dtype=torch.int64, device=self.device)

            P_clean = torch.sparse_coo_tensor(indices, values, size=P.shape).coalesce()
            P = to_sparse_csr_tensor(P_clean).to(dtype=self.dtype, device=self.device)

            if self.P_is_upper:
                rows_T, cols_T = cols, rows
                transposed_idx = rows_T * P.shape[0] + cols_T # TODO check this
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

                diag_mask = rows == cols
                diag_indices = rows[diag_mask]
                diag_values = values[diag_mask]
                diag = torch.zeros(self.n, dtype=self.dtype, device=values.dtype)
                diag[diag_indices] = diag_values
                diag = to_tensor(diag, dtype=self.dtype, device=self.device)

                self.P_diag_mask = to_tensor(diag_mask, dtype=self.dtype, device=self.device)

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
    
    @property.setter
    def P(self, P) -> None:
        pass

    @property
    def A(self) -> torch.Tensor:
        return self._A
    
    @A.setter
    def A(self, A) -> None:
        if isinstance(A, spmatrix) or isinstance(A, sparray):
            self._A = to_sparse_csr_tensor(A, dtype=self.dtype, device=self.device)
        if isinstance(A, torch.Tensor):
            # need to remove explicit zeros
            A = A.to_sparse_csr() # returns self if A is already sparse csr.
            values = A.values()
            if values.shape[0] == self.A_original_nnz:
                values = values[self.A_nonzero_mask] # TODO this nonzero mask is on desired device...need a copy both places?
                # -> we always return dA on device specified by user, so then assume all updated As are on
                # this device
                values = to_tensor(values, dtype=self.dtype, device=self.device)
                self._A = torch.sparse_csr_tensor(
                    self._A.crow_indices(), self._A.col_indices(), values=values, size=(self.m, self.n), device=self.device
                )
                self._AT = self._A_transpose(values)
            elif values.shape[0] == self.A_filtered_nnz:
                self._A = torch.sparse_csr_tensor(
                    self._A.crow_indices(), self._A.col_indices(), values=values, size=(self.m, self.n), device=self.device
                )
                self._AT = self._A_transpose(values)
            else:
                raise ValueError("The new `A` must have the same sparsity pattern as the"
                                 + " original or filtered `A`. Since the provided tensor doesn't"
                                 + " have the same number of nonzero elements as the"
                                 + " original or filtered `A`, its sparsity pattern differs.")
        else:
            raise ValueError("A must be a `scipy` `spmatrix` (or `sparray`) or a `torch.Tensor`."
                             + f" It is {type(A)}.")

    @property
    def AT(self) -> torch.Tensor:
        return self._AT

    # TODO (quill): Create method which creates data objects (to hold d_data) that utilizes
    #   our knowledge about the fixed data
    #   -> called a class method or something like that?
    # static method?

class SparseHelper:
    def __init__(self):
        self.P = None
        self.P_diag_perm = None
        self.P_T_perm = None
        self.P_indices = None

        self.A = None
        self.A_T_perm = None
        self.A_indices = None

    def obj_matrix_init(self, P: torch.Tensor, P_is_upper: bool) -> None:
        """
        - Remove any potential explicit zeros from P
        - Save the nonzero rows and column indices
        - If P_is_upper:
            - Store efficient way to compute diagonal
            - Store efficient way to compute transpose
        """
        assert P.layout == torch.sparse_coo, "P must be in sparse COO format"

        # (quill) does converting tensor duplicate data?

        # Remove explicit zeros
        nonzero_mask = P.values() != 0
        P_clean = torch.sparse_coo_tensor(
            P.indices()[:, nonzero_mask],
            P.values()[nonzero_mask],
            size=P.shape
        ).coalesce()

        self.P = P_clean
        self.P_indices = P_clean.indices()

        if P_is_upper:
            # For transpose (symmetry: P^T = P but for strict upper-triangular, swap indices)
            perm = torch.argsort(
                P_clean.indices()[1] * P.shape[0] + P_clean.indices()[0]
            )
            self.P_T_perm = perm

            # For diagonal
            row, col = P_clean.indices()
            diag_mask = row == col
            diag_perm = torch.nonzero(diag_mask, as_tuple=True)[0]
            self.P_diag_perm = diag_perm

    def constr_matrix_init(self, A: torch.Tensor) -> None:
        """
        - Remove any potential explicit zeros from A
        - Save the nonzero rows and column indices
        - Save an efficient way of creating A's transpose
        """

        assert A.layout == torch.sparse_csr, "A must be in sparse CSR format"

        A_coo = A.to_sparse_coo()
        nonzero_mask = A_coo.values() != 0
        indices = A_coo.indices()[:, nonzero_mask]
        values = A_coo.values()[nonzero_mask]

        A_clean = torch.sparse_coo_tensor(indices, values, size=A.shape).coalesce()
        self.A = A_clean.to_sparse_csr()
        self.A_indices = A_clean.indices()

        row, col = indices[0], indices[1]
        row_t, col_t = col, row
        perm = torch.argsort(row_t * A.shape[0] + col_t)
        self.A_T_perm = perm

    def get_P_T(self) -> torch.Tensor:
        indices = self.P.indices()
        values = self.P.values()[self.P_T_perm]
        swapped_indices = torch.stack([indices[1], indices[0]], dim=0)[:, self.P_T_perm]
        return torch.sparse_coo_tensor(swapped_indices, values, size=self.P.shape)

    def get_P_diag(self) -> torch.Tensor:
        # need to put in an array with right length
        return self.P.values()[self.P_diag_perm]

    def get_A_T(self) -> torch.Tensor:
        indices = self.A.indices()
        values = self.A.values()[self.A_T_perm]
        swapped_indices = torch.stack([indices[1], indices[0]], dim=0)[:, self.A_T_perm]
        return torch.sparse_coo_tensor(swapped_indices, values, size=(self.A.shape[1], self.A.shape[0]))


# ===================
# Unit tests
# ===================
class TestSparseHelper(unittest.TestCase):
    def test_obj_matrix_init(self):
        helper = SparseHelper()
        indices = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 1, 2]])
        values = torch.tensor([1.0, 0.0, 3.0, 4.0, 5.0])
        P = torch.sparse_coo_tensor(indices, values, size=(3, 3))

        helper.obj_matrix_init(P, P_is_upper=True)

        P_T = helper.get_P_T()
        P_diag = helper.get_P_diag()

        self.assertTrue(torch.allclose(P_diag, torch.tensor([1.0, 3.0, 5.0])))
        self.assertEqual(P_T._indices().shape[1], helper.P._indices().shape[1])

    def test_constr_matrix_init(self):
        helper = SparseHelper()
        A_dense = torch.tensor([
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0]
        ])
        A = A_dense.to_sparse().to_sparse_csr()

        helper.constr_matrix_init(A)
        A_T = helper.get_A_T()

        self.assertEqual(A_T.shape, (3, 3))
        self.assertTrue(torch.allclose(
            A_T.to_dense(), A_dense.T
        ))

class SparseHelper:
    def __init__(self):
        self.P = None
        self.P_diag_perm = None
        self.P_T_perm = None
        self.P_indices = None

        self.A = None
        self.A_T_perm = None
        self.A_indices = None

    @dynamo.optimize("inductor")
    def obj_matrix_init(self, P: torch.Tensor, P_is_upper: bool) -> None:
        """
        - Remove any potential explicit zeros from P
        - Save the nonzero rows and column indices
        - If P_is_upper:
            - Store efficient way to compute diagonal
            - Store efficient way to compute transpose
        """
        assert P.layout == torch.sparse_coo, "P must be in sparse COO format"

        # Remove explicit zeros
        nonzero_mask = P.values() != 0
        P_clean = torch.sparse_coo_tensor(
            P.indices()[:, nonzero_mask].to("cuda"),
            P.values()[nonzero_mask].to("cuda"),
            size=P.shape,
            device="cuda"
        ).coalesce()

        self.P = P_clean
        self.P_indices = P_clean.indices()

        if P_is_upper:
            perm = torch.argsort(
                P_clean.indices()[1] * P.shape[0] + P_clean.indices()[0]
            )
            self.P_T_perm = perm

            row, col = P_clean.indices()
            diag_mask = row == col
            diag_perm = torch.nonzero(diag_mask, as_tuple=True)[0]
            self.P_diag_perm = diag_perm

    @dynamo.optimize("inductor")
    def constr_matrix_init(self, A: torch.Tensor) -> None:
        """
        - Remove any potential explicit zeros from A
        - Save the nonzero rows and column indices
        - Save an efficient way of creating A's transpose
        """
        assert A.layout == torch.sparse_csr, "A must be in sparse CSR format"

        A_coo = A.to_sparse_coo()
        nonzero_mask = A_coo.values() != 0
        indices = A_coo.indices()[:, nonzero_mask].to("cuda")
        values = A_coo.values()[nonzero_mask].to("cuda")

        A_clean = torch.sparse_coo_tensor(indices, values, size=A.shape, device="cuda").coalesce()
        self.A = A_clean.to_sparse_csr()
        self.A_indices = A_clean.indices()

        row, col = indices[0], indices[1]
        row_t, col_t = col, row
        perm = torch.argsort(row_t * A.shape[0] + col_t)
        self.A_T_perm = perm

    @dynamo.optimize("inductor")
    def get_P_T(self) -> torch.Tensor:
        indices = self.P.indices()
        values = self.P.values()[self.P_T_perm]
        swapped_indices = torch.stack([indices[1], indices[0]], dim=0)[:, self.P_T_perm]
        return torch.sparse_coo_tensor(swapped_indices, values, size=self.P.shape)

    @dynamo.optimize("inductor")
    def get_P_diag(self) -> torch.Tensor:
        return self.P.values()[self.P_diag_perm]

    @dynamo.optimize("inductor")
    def get_A_T(self) -> torch.Tensor:
        indices = self.A.indices()
        values = self.A.values()[self.A_T_perm]
        swapped_indices = torch.stack([indices[1], indices[0]], dim=0)[:, self.A_T_perm]
        return torch.sparse_coo_tensor(swapped_indices, values, size=(self.A.shape[1], self.A.shape[0]))