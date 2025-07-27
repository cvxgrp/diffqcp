from __future__ import annotations
from abc import abstractmethod

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
import equinox as eqx
from lineax import AbstractLinearOperator
from jaxtyping import Float, Integer, Array

from diffqcp.cones.canonical import ProductConeProjector
from diffqcp._helpers import _coo_to_csr_transpose_map, _TransposeCSRInfo

class QCPStructure(eqx.Module):
    cone_projector: ProductConeProjector
    # solver: AbstractLinearSolver = lineax. # set to LSMR; allow tolerance setting?


    def __init__(
        self,
        P: Float[Array | BCOO | BCSR, "*batch n n"],
        A: Float[Array | BCOO | BCSR, "*batch m n"],
        q: Float[Array, "*batch n"],
        b: Float[Array, "*batch m"],
        cones: dict[str, int | list[int] | list[float]]
    ):
        pass
    
    def obj_matrix_init(self):
        pass

    def constr_matrix_init(self):
        pass


class QCPStructureGPU(eqx.Module):
    """
    P is assumed to be the full matrix

    Whole class will be declared as `static` by `QCP` class
    """

    cone_projector: ProductConeProjector
    # TODO(quill): include the cone projector and linear solver?
    #   -> probably just continue developing and see if this decision is forced.
    is_batched: bool
    
    P_csr_indices: Integer[Array, "..."]
    P_csr_indptr: Integer[Array, "..."]
    P_nonzero_rows: Integer[Array, "..."]
    P_nonzero_cols: Integer[Array, "..."]
    
    A_csr_indices: Integer[Array, "..."]
    A_csr_indptr: Integer[Array, "..."]
    A_nonzero_rows: Integer[Array, "..."]
    A_nonzero_cols: Integer[Array, "..."]
    A_transpose_info: _TransposeCSRInfo

    def __init__(
        self, P: Float[BCSR, "*batch n n"], A: Float[BCSR, "*batch m n"]
    ):
        if not isinstance(P, BCSR):
            raise ValueError("The objective matrix `P` must be a `BCSR` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")
        # check if batched
        if P.n_batch == 0:
            self.is_batched = False
            self.obj_matrix_init(P)
        elif P.n_batch == 1:
            self.is_batched = True
            # Extract information from first matrix in the batch.
            # Strict requirement is that all matrices in the batch share
            # the same sparsity structure (holds via DPP, also maybe required by JAX?)
            self.obj_matrix_init(P[0])
        else:
            raise ValueError("The objective matrix `P` must have at most one batch dimension,"
                             + f" but the provided BCSR matrix has {P.n_batch} dimensions.")
        
        if not isinstance(A, BCSR):
            raise ValueError("The objective matrix `A` must be a `BCSR` JAX matrix,"
                             + f" but the provided `A` is a {type(A)}.")
        
        # NOTE(quill): could theoretically allow mismatch and broadcast
        #   (Just to keep in mind for the future; not needed now.)
        if A.n_batch != P.n_batch:
            raise ValueError(f"The objective matrix `P` has {P.n_batch} dimensions"
                             + f" while the constraint matrix `A` has {A.n_batch}"
                             + " dimensions. The batch dimensionality of `P` and `A`"
                             + " must match.")
        
        if self.is_batched:
            self.constr_matrix_init(A[0])
        else:
            self.constr_matrix_init(A)
        
    def obj_matrix_init(self, P: Float[BCSR, "n n"]):
        """
        Functionality:
            - Remove any explicit zeros from `P`.
            - Save these nonzero row and column indices
                (Needed for `diffqcp` `vjp` computation.)
        """
        
        P_coo = P.to_bcoo()
        # NOTE(quill): the following assumption is needed for the following
        #   manipulation to result in accurate metadata.
        #   If this error occurs more frequently than not, then it will probably
        #   be worth canonicalizing the data matrices by default.
        if P_coo.data != P.data:
            raise ValueError("The ordering of the data in `P_coo` and `P`"
                             + " (a BCSR matrix) does not match."
                             + " Please try to coerce `P` into canonical form.")
        
        self.P_csr_indices = P.indices
        self.P_csr_indptr = P.indptr
        
        self.P_nonzero_rows  = P_coo.indices[:, 0]
        self.P_nonzero_cols = P_coo.indices[:, 1]
        

    def constr_matrix_init(self, A: Float[BCSR, "m n"]):

        A_coo = A.to_bcoo()
        # NOTE(quill): see note in `obj_matrix_init`
        if A_coo.data != A.data:
            raise ValueError("The ordering of the data in `A_coo` and `A`"
                             + " (a BCSR matrix) does not match."
                             + " Please try to coerce `A` into canonical form.")
        
        self.A_csr_indices = A.indices
        self.A_csr_indptr = A.indptr
        
        self.A_nonzero_rows = A_coo.indices[:, 0]
        self.A_nonzero_cols = A_coo.indices[:, 1]

        # Create metadata for cheap transposes
        self.A_transpose_info = _coo_to_csr_transpose_map(A_coo)

    def form_A_transpose(self, A):
        pass


class QCPStructureCPU(eqx.Module):
    """
    `P` is assumed to be the upper triangular part of the matrix in the quadratic form.
    """
    
    cone_projector: ProductConeProjector
    is_batched: bool

    P_nonzero_rows: Integer[Array, "..."]
    P_nonzero_cols: Integer[Array, "..."]

    A_nonzero_rows: Integer[Array, "..."]
    A_nonzero_cols: Integer[Array, "..."]

    def __init__(
        self, P: Float[BCOO, "*batch n n"], A: Float[BCOO, "*batch n n"]
    ):
        if not isinstance(P, BCOO):
            raise ValueError("The objective matrix `P` must be a `BCOO` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")

        # check if batched
        if P.n_batch == 0:
            self.is_batched = False
            self.P_nonzero_rows = P.indices[:, 0]
            self.P_nonzero_cols = P.indices[:, 1]
        elif P.n_batch == 1:
            self.is_batched = True
            # Extract information from first matrix in the batch.
            # Strict requirement is that all matrices in the batch share
            # the same sparsity structure (holds via DPP, also maybe required by JAX?)
            self.P_nonzero_rows = P[0].indices[:, 0]
            self.P_nonzero_cols = P[0].indices[:, 1]
        else:
            raise ValueError("The objective matrix `P` must have at most one batch dimension,"
                             + f" but the provided BCOO matrix has {P.n_batch} dimensions.")

        if not isinstance(A, BCOO):
            raise ValueError("The objective matrix `A` must be a `BCOO` JAX matrix,"
                             + f" but the provided `A` is a {type(A)}.")
        
        # NOTE(quill): see note in `QCPStructureGPU`
        if A.n_batch != P.n_batch:
            raise ValueError(f"The objective matrix `P` has {P.n_batch} dimensions"
                             + f" while the constraint matrix `A` has {A.n_batch}"
                             + " dimensions. The batch dimensionality of `P` and `A`"
                             + " must match.")
        
        if self.is_batched:
            self.A_nonzero_rows = A[0].indices[:, 0]
            self.A_nonzero_cols = A[0].indices[:, 1]
        else:
            self.A_nonzero_rows = A.indices[:, 0]
            self.A_nonzero_cols = A.indices[:, 1]


class ObjMatrix(AbstractLinearOperator):
    P: Float[Array | BCOO | BCSR, "*batch n n"] # TODO(quill): follows abstract/final?

# NOTE(quill): are tags inherited?

class ObjMatrixCPU(ObjMatrix):
    P: Float[BCOO, "n n"]
    PT: Float[BCOO, "n n"]
    diag: Float[BCOO, " n-1"]

    def mv(self, vector):
        return self.P @ vector + self.PT @ vector - self.diag*vector
    
    def transpose(self):
        return self
    
    def as_matrix(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s `as_matrix` method is"
                                  + " not yet implemented.")
    
    def in_structure(self):
        pass

    def out_structure(self):
        pass


class ObjMatrixGPU(ObjMatrix):
    P: Float[BCSR, "n n"]


    def mv(self, vector):
        return self.P @ vector

