from __future__ import annotations
from abc import abstractmethod

import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
import equinox as eqx
from lineax import AbstractLinearOperator, AbstractLinearSolver, LSMR
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
    """

    cone_projector: ProductConeProjector
    
    P_csr_indices = Integer[Array, "..."]
    P_csr_indptr = Integer[Array, "..."]
    P_nonzero_rows = Integer[Array, "..."]
    P_nonzero_cols = Integer[Array, "..."]
    
    A_csr_indices = Integer[Array, "..."]
    A_csr_indptr = Integer[Array, "..."]
    A_nonzero_rows = Integer[Array, "..."]
    A_nonzero_cols = Integer[Array, "..."]
    A_transpose_info = _TransposeCSRInfo

    def __init__(
        self, P: Float[BCSR, "*batch n n"], A: Float[BCSR, "*batch m n"]
    ):
        if not isinstance(P, BCSR):
            raise ValueError("The objective matrix `P` must be a `BCSR` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")
        # check if batched
        if P.n_batch == 0:
            self.obj_matrix_init(P)
        elif P.n_batch == 1:
            # Extract information from first matrix in the batch.
            # Strict requirement is that all matrices in the batch share
            # the same sparsity structure (holds via DPP, also maybe required by JAX?)
            self.obj_matrix_init(P[0])
        else:
            raise ValueError("The objective matrix `P` must have at most one batch,"
                             + f" but the provided BCSR matrix has {P.n_batch} dimensions.")
        if not isinstance(A, BCSR):
            raise ValueError("The objective matrix `A` must be a `BCSR` JAX matrix,"
                             + f" but the provided `A` is a {type(A)}.")
        
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
        
        num_rows, num_cols = A.shape[0], A.shape[1]

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


class QCPStructureCPU(eqx.Module):
    """
    P is assumed to be <upper or lower> triangular
    """
    pass


class ObjMatrix(AbstractLinearOperator):
    P: Float[Array | BCOO | BCSR, "*batch n n"] # TODO(quill): follows abstract/final?
    is_batched: bool = eqx.field(static=True)

    def __init__(self):
        pass

# NOTE(quill): are tags inherited?

class ObjMatrixCPU(ObjMatrix):
    P: Float[BCOO, "*batch n n"]

    # def __init__(self, P:)

class ObjMatrixGPU(ObjMatrix):
    P: Float[BCSR, "*batch n n"]

# NOTE(quill): if dense then how to handle output of `vjp`

class ObjMatrixDense(ObjMatrix):
    P: Float[Array, "*batch n n"]

    def __init__(self, P):
        self.P = P

    def mv(self, vec):
        return self.P @ vec

"""
Ok, let's not overthink this.
We can instantiate the object 


Ooh, let's use that abstract/final pattern
"""    


class ConstrMatrix(AbstractLinearOperator):
    A: Float[Array | BCOO | BCSR, "*batch m n"]

    def __init__(self):
        pass

