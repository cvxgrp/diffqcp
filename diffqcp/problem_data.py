from __future__ import annotations

__all__ = [
    "QCPStructureCPU",
    "QCPStructureGPU",
    "QCPStructureLayers",
    "ObjMatrixGPU",
    "ObjMatrixCPU"
]

from abc import abstractmethod
from typing import TYPE_CHECKING

from jax import ShapeDtypeStruct
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
import equinox as eqx
import lineax as lx
from lineax import AbstractLinearOperator
from jaxtyping import Float, Integer, Bool, Array

if TYPE_CHECKING:
    from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg

from diffqcp.cones.canonical import ProductConeProjector
from diffqcp._helpers import _coo_to_csr_transpose_map, _TransposeCSRInfo

class QCPStructure(eqx.Module):

    # The following are needed for `_form_atoms` in `AbstractQCP`.
    n: eqx.AbstractVar[int]
    m: eqx.AbstractVar[int]
    N: eqx.AbstractVar[int]
    cone_projector: eqx.AbstractVar[ProductConeProjector]
    
    @abstractmethod
    def obj_matrix_init(self):
        pass
    
    @abstractmethod
    def constr_matrix_init(self):
        pass


class QCPStructureCPU(QCPStructure):
    """
    `P` is assumed to be the upper triangular part of the matrix in the quadratic form.
    """
    
    n: int
    m: int
    N: int
    cone_projector: ProductConeProjector
    is_batched: bool

    P_nonzero_rows: Integer[Array, "..."]
    P_nonzero_cols: Integer[Array, "..."]
    P_diag_mask: Bool[Array, "..."]
    P_diag_indices: Integer[Array, "..."]

    A_nonzero_rows: Integer[Array, "..."]
    A_nonzero_cols: Integer[Array, "..."]

    def __init__(
        self,
        P: Float[BCOO, "*batch n n"],
        A: Float[BCOO, "*batch n n"],
        cone_dims: dict[str, int | list[int] | list[float]],
        onto_dual: bool = True
    ):
        
        # NOTE(quill): checks on `cone_dims` done in `ProductConeProjector.__init__`
        self.cone_projector = ProductConeProjector(cone_dims, onto_dual=onto_dual)

        if not isinstance(P, BCOO):
            raise ValueError("The objective matrix `P` must be a `BCOO` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")

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
                             + f" but the provided BCOO matrix has {P.n_batch} dimensions.")

        if not isinstance(A, BCOO):
            raise ValueError("The objective matrix `A` must be a `BCOO` JAX matrix,"
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

        self.N = self.n + self.m + 1
    
    def obj_matrix_init(self, P: Float[BCOO, "n n"]):
        # TODO(quill): checks on P being upper triangular.
        #   (Might as well do since this structure is formed once.)
        self.n = jnp.shape(P)[0]
        self.P_nonzero_rows = P.indices[:, 0]
        self.P_nonzero_cols = P.indices[:, 1]
        self.P_diag_mask = P.indices[:, 0] == P.indices[:, 1]
        self.P_diag_indices = P.indices[:, 0][self.P_diag_mask]
        
    def constr_matrix_init(self, A: Float[BCOO, "m n"]):
        self.m = jnp.shape(A)[0]
        self.A_nonzero_rows = A.indices[:, 0]
        self.A_nonzero_cols = A.indices[:, 1]
            
    def form_obj(self, P_like: Float[BCOO, "n n"]) -> ObjMatrixCPU:
        diag_values = P_like.data[self.P_diag_mask]
        diag = jnp.zeros(self.n)
        diag = diag.at[self.P_diag_indices].set(diag_values)
        return ObjMatrixCPU(P_like, P_like.T, diag)


class QCPStructureGPU(QCPStructure):
    """
    P is assumed to be the full matrix
    """

    n: int
    m: int
    N: int
    cone_projector: ProductConeProjector
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
        self,
        P: Float[BCSR, "*batch n n"],
        A: Float[BCSR, "*batch m n"],
        cone_dims: dict[str, int | list[int] | list[float]],
        onto_dual: bool = True
    ):
        
        # NOTE(quill): checks on `cone_dims` done in `ProductConeProjector.__init__`
        self.cone_projector = ProductConeProjector(cone_dims, onto_dual=onto_dual)
        
        if not isinstance(P, BCSR):
            raise ValueError("The objective matrix `P` must be a `BCSR` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")
        # check if batched
        if P.n_batch == 0:
            self.is_batched = False
            self.obj_matrix_init(P)
        elif P.n_batch == 1:
            self.is_batched = True
            # NOTE(quill): see note in `QCPStructureCPU`
            self.obj_matrix_init(P[0])
        else:
            raise ValueError("The objective matrix `P` must have at most one batch dimension,"
                             + f" but the provided BCSR matrix has {P.n_batch} dimensions.")
        
        if not isinstance(A, BCSR):
            raise ValueError("The objective matrix `A` must be a `BCSR` JAX matrix,"
                             + f" but the provided `A` is a {type(A)}.")
        
        # NOTE(quill): see note in `QCPStructureCPU`
        if A.n_batch != P.n_batch:
            raise ValueError(f"The objective matrix `P` has {P.n_batch} dimensions"
                             + f" while the constraint matrix `A` has {A.n_batch}"
                             + " dimensions. The batch dimensionality of `P` and `A`"
                             + " must match.")
        
        if self.is_batched:
            self.constr_matrix_init(A[0])
        else:
            self.constr_matrix_init(A)

        self.N = self.n + self.m + 1
    
    def obj_matrix_init(self, P: Float[BCSR, "n n"]):
        self.n = jnp.shape(P)[0]
        P_coo = P.to_bcoo()
        # NOTE(quill): the following assumption is needed for the following
        #   manipulation to result in accurate metadata.
        #   If this error occurs more frequently than not, then it will probably
        #   be worth canonicalizing the data matrices by default.
        # NOTE(quill): must use `allclose` since `!=` compares if same data in memory.
        if not jnp.allclose(P_coo.data, P.data): 
            raise ValueError("The ordering of the data in `P_coo` and `P`"
                             + " (a BCSR matrix) does not match."
                             + " Please try to coerce `P` into canonical form.")
        
        self.P_csr_indices = P.indices
        self.P_csr_indptr = P.indptr
        
        self.P_nonzero_rows  = P_coo.indices[:, 0]
        self.P_nonzero_cols = P_coo.indices[:, 1]
        
    def constr_matrix_init(self, A: Float[BCSR, "m n"]):
        self.m = jnp.shape(A)[0]
        A_coo = A.to_bcoo()
        # NOTE(quill): see note in `obj_matrix_init`
        if not jnp.allclose(A_coo.data, A.data):
            raise ValueError("The ordering of the data in `A_coo` and `A`"
                             + " (a BCSR matrix) does not match."
                             + " Please try to coerce `A` into canonical form.")
        
        self.A_csr_indices = A.indices
        self.A_csr_indptr = A.indptr
        
        self.A_nonzero_rows = A_coo.indices[:, 0]
        self.A_nonzero_cols = A_coo.indices[:, 1]

        # Create metadata for cheap transposes
        self.A_transpose_info = _coo_to_csr_transpose_map(A_coo)

    def form_A_transpose(self, A_like: Float[BCSR, "m n"]) -> Float[BCSR, "n m"]:
        transposed_data = A_like.data[self.A_transpose_info.sorting_perm]
        return BCSR((transposed_data,
                     self.A_transpose_info.indices,
                     self.A_transpose_info.indptr),
                     shape=(self.n, self.m))


class QCPStructureLayers(QCPStructure):
    """Meant to be used with CVXPYlayers."""

    n: int
    m: int
    N: int
    cone_projector: ProductConeProjector
    is_batched: bool

    def __init__(
        self,
        prob: ParamConeProg,
        cone_dims: dict[str, int | list[int] | list[float]],
        onto_dual: bool = True
    ):
        
        self.cone_projector = ProductConeProjector(cone_dims, onto_dual=onto_dual)

#         # Now we need to obtain
#         constraint_structure = 


type ObjMatrix = ObjMatrixCPU | ObjMatrixGPU


class ObjMatrixCPU(AbstractLinearOperator):
    P: Float[BCOO, "n n"]
    PT: Float[BCOO, "n n"]
    diag: Float[BCOO, " n"]
    in_struc: ShapeDtypeStruct

    def __init__(
        self,
        P: Float[BCOO, "n n"],
        PT: Float[BCOO, "n n"],
        diag: Float[BCOO, " n"]
    ):
        self.P, self.PT, self.diag = P, PT, diag
        n = jnp.shape(P)[0]
        self.in_struc = ShapeDtypeStruct(shape=(n,),
                                         dtype=P.data.dtype)
    
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
        return self.in_structure()
    
class ObjMatrixGPU(AbstractLinearOperator):
    P: Float[BCSR, "n n"]
    in_struc: ShapeDtypeStruct

    def __init__(
        self,
        P: Float[BCSR, "n n"],
    ):
        self.P = P
        n = jnp.shape(P)[0]
        self.in_struc = ShapeDtypeStruct(shape=(n,),
                                         dtype=P.data.dtype)
    
    def mv(self, vector):
        return self.P @ vector
    
    def transpose(self):
        return self
    
    def as_matrix(self):
        raise NotImplementedError(f"{self.__class__.__name__}'s `as_matrix` method is"
                                  + " not yet implemented.")
    
    def in_structure(self):
        pass

    def out_structure(self):
        return self.in_structure()

@lx.is_symmetric.register(ObjMatrixCPU)
def _(op):
    return True

@lx.is_symmetric.register(ObjMatrixGPU)
def _(op):
    return True