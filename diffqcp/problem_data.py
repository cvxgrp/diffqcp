"""Problem structure and objective-matrix operators shared by `HostQCP` and `DeviceQCP`.

A `QCPStructure*` is built once per sparsity pattern (and cone) and reused
for every problem with that pattern, e.g. across the iterations of a learning
loop or the members of a CVXPYlayers batch:

- `QCPStructureCPU`: P given as its upper triangle, P and A as BCOO;
- `QCPStructureGPU`: P given in full, P and A as BCSR.

`ObjMatrixCPU` / `ObjMatrixGPU` wrap the stored P as a symmetric lineax
operator.
"""
from __future__ import annotations

__all__ = [
    "ObjMatrixCPU",
    "ObjMatrixGPU",
    "QCPStructureCPU",
    "QCPStructureGPU",
]

from abc import abstractmethod
from typing import Self, TypeAlias, cast

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jax import ShapeDtypeStruct
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Bool, Float, Integer
from lineax import AbstractLinearOperator

from diffqcp._helpers import _coo_to_csr_transpose_map, _TransposeCSRInfo
from diffqcp.cones.canonical import ProductConeProjector
from diffqcp.linops import _dense_from_mv


class QCPStructure(eqx.Module):
    """Sparsity patterns, sizes and cone of a family of QCPs.

    `n` and `m` are the sizes of x and y, `N = n + m + 1`, and
    `cone_projector` projects onto the cone K (or its dual, see the concrete
    classes' `onto_dual`).
    """

    n: eqx.AbstractVar[int]
    m: eqx.AbstractVar[int]
    N: eqx.AbstractVar[int]
    cone_projector: eqx.AbstractVar[ProductConeProjector]

    @abstractmethod
    def obj_matrix_init(self, P: BCOO | BCSR) -> None:
        """Record `n` and the sparsity pattern of the objective matrix `P`."""

    @abstractmethod
    def constr_matrix_init(self, A: BCOO | BCSR) -> None:
        """Record `m` and the sparsity pattern of the constraint matrix `A`."""


class QCPStructureCPU(QCPStructure):
    """Problem structure for `HostQCP`: P as its upper triangle, P and A as BCOO.

    **Arguments:**
    - `P`: BCOO, shape (n, n) or (batch, n, n). The upper triangle (diagonal
        included) of the objective matrix. Only its sparsity pattern (and the
        order of its stored entries) is used; with a batch axis, the first
        matrix's pattern is used and all must share it.
    - `A`: BCOO, shape (m, n) or (batch, m, n). The constraint matrix, with
        the same number of batch axes as `P`.
    - `cone_dims`: the cone K in SCS format, e.g.
        `{"z": 3, "l": 2, "q": [3, 4], "s": [2], "ep": 1, "p": [0.3, -0.6]}`
        (see `diffqcp/cones/CONVENTIONS.md`).
    - `onto_dual`: if `True` (default; what the derivative needs), the cone
        projector projects onto the dual cone K^*.

    Every stored entry's (row, column) is kept in `P_nonzero_rows` /
    `P_nonzero_cols` (likewise for A), in the order of `P.data`; data passed
    later (to `HostQCP`, as perturbations, or as values to
    `differentiable_solution`) must use that same order.
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
    # Positions in `P.data` of the diagonal entries. Integer (not boolean)
    # indexing so `form_obj` works when the structure is traced under `jit`.
    P_diag_positions: Integer[Array, "..."]

    A_nonzero_rows: Integer[Array, "..."]
    A_nonzero_cols: Integer[Array, "..."]

    def __init__(
        self,
        P: Float[BCOO, "*batch n n"],
        A: Float[BCOO, "*batch m n"],
        cone_dims: dict[str, int | list[int] | list[float]],
        onto_dual: bool = True,
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
            raise ValueError("The constraint matrix `A` must be a `BCOO` JAX matrix,"
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

    def obj_matrix_init(self, P: Float[BCOO, "n n"]) -> None:
        """Record `n`, the (row, col) of each stored entry of `P`, and where its diagonal is."""
        # TODO(quill): check that P is upper triangular (cheap: the structure is formed once).
        self.n = jnp.shape(P)[0]
        self.P_nonzero_rows = P.indices[:, 0]
        self.P_nonzero_cols = P.indices[:, 1]
        self.P_diag_mask = P.indices[:, 0] == P.indices[:, 1]
        self.P_diag_indices = P.indices[:, 0][self.P_diag_mask]
        self.P_diag_positions = jnp.nonzero(self.P_diag_mask)[0]

    def constr_matrix_init(self, A: Float[BCOO, "m n"]) -> None:
        """Record `m` and the (row, col) of each stored entry of `A`."""
        self.m = jnp.shape(A)[0]
        self.A_nonzero_rows = A.indices[:, 0]
        self.A_nonzero_cols = A.indices[:, 1]

    def form_obj(self, P_like: Float[BCOO, "n n"]) -> ObjMatrixCPU:
        """Wrap an upper triangle with this structure's pattern (P, or a perturbation of it)
        as the symmetric operator `ObjMatrixCPU`."""
        diag_values = P_like.data[self.P_diag_positions]
        diag = jnp.zeros(self.n)
        diag = diag.at[self.P_diag_indices].set(diag_values)
        return ObjMatrixCPU(P_like, cast(BCOO, P_like.T), diag)  # (`.T` is untyped in jax's stubs)


class QCPStructureGPU(QCPStructure):
    """Problem structure for `DeviceQCP`: P in full, P and A as BCSR.

    **Arguments:**
    - `P`: BCSR, shape (n, n) or (batch, n, n). The full symmetric objective
        matrix; only its sparsity pattern is used. With a batch axis, the
        first matrix's pattern is used and all must share it.
    - `A`: BCSR, shape (m, n) or (batch, m, n). The constraint matrix, with
        the same number of batch axes as `P`.
    - `cone_dims`: the cone K in SCS format (see `QCPStructureCPU`).
    - `onto_dual`: if `True` (default; what the derivative needs), the cone
        projector projects onto the dual cone K^*.

    `P` and `A` must be in canonical CSR form (sorted column indices, no
    duplicates), so that the order of their stored values matches their COO
    conversion; a `ValueError` is raised otherwise. The structure also
    precomputes the permutation that turns A's values into those of A^T in
    CSR form (`form_A_transpose`).
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
        onto_dual: bool = True,
    ):

        # NOTE(quill): checks on `cone_dims` done in `ProductConeProjector.__init__`
        self.cone_projector = ProductConeProjector(cone_dims, onto_dual=onto_dual)

        if not isinstance(P, BCSR):
            raise ValueError("The objective matrix `P` must be a `BCSR` JAX matrix,"
                             + f" but the provided `P` is a {type(P)}.")
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
            raise ValueError("The constraint matrix `A` must be a `BCSR` JAX matrix,"
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

    def obj_matrix_init(self, P: Float[BCSR, "n n"]) -> None:
        """Record `n`, the CSR index arrays of `P`, and the (row, col) of each stored entry."""
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

    def constr_matrix_init(self, A: Float[BCSR, "m n"]) -> None:
        """Record `m`, the CSR index arrays of `A` and of A^T, and the (row, col) of each stored entry."""
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
        """The transpose of a matrix with A's sparsity pattern, as BCSR (no sorting at call time)."""
        transposed_data = A_like.data[self.A_transpose_info.sorting_perm]
        return BCSR((transposed_data,
                     self.A_transpose_info.indices,
                     self.A_transpose_info.indptr),
                     shape=(self.n, self.m))


# `ObjMatrix` is defined below, after `ObjMatrixCPU` and `ObjMatrixGPU`.


class ObjMatrixCPU(AbstractLinearOperator):
    """The symmetric matrix `U + U^T - diag(U)` as a lineax operator, given its upper triangle U.

    **Arguments:**
    - `P`: BCOO, the upper triangle U.
    - `PT`: BCOO, U^T.
    - `diag`: shape (n,), the diagonal of U (subtracted so it is not counted twice).

    Build it with `QCPStructureCPU.form_obj`.
    """

    P: Float[BCOO, "n n"]
    PT: Float[BCOO, "n n"]
    diag: Float[Array, " n"]
    in_struc: ShapeDtypeStruct = eqx.field(static=True)

    def __init__(
        self,
        P: Float[BCOO, "n n"],
        PT: Float[BCOO, "n n"],
        diag: Float[Array, " n"]
    ):
        self.P, self.PT, self.diag = P, PT, diag
        n = jnp.shape(P)[0]
        self.in_struc = ShapeDtypeStruct(shape=(n,),
                                         dtype=P.data.dtype)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.P @ vector + self.PT @ vector - self.diag*vector

    def transpose(self) -> Self:
        return self

    def as_matrix(self) -> Float[Array, "n n"]:
        return _dense_from_mv(self)

    def in_structure(self) -> ShapeDtypeStruct:
        return self.in_struc

    def out_structure(self) -> ShapeDtypeStruct:
        return self.in_structure()

class ObjMatrixGPU(AbstractLinearOperator):
    """A full symmetric BCSR matrix as a lineax operator.

    **Arguments:**
    - `P`: BCSR, shape (n, n). Assumed symmetric (not checked).
    """

    P: Float[BCSR, "n n"]
    in_struc: ShapeDtypeStruct = eqx.field(static=True)

    def __init__(
        self,
        P: Float[BCSR, "n n"],
    ):
        self.P = P
        n = jnp.shape(P)[0]
        self.in_struc = ShapeDtypeStruct(shape=(n,),
                                         dtype=P.data.dtype)

    def mv(self, vector: Float[Array, " n"]) -> Float[Array, " n"]:
        return self.P @ vector

    def transpose(self) -> Self:
        return self

    def as_matrix(self) -> Float[Array, "n n"]:
        return _dense_from_mv(self)

    def in_structure(self) -> ShapeDtypeStruct:
        return self.in_struc

    def out_structure(self) -> ShapeDtypeStruct:
        return self.in_structure()


@lx.is_symmetric.register(ObjMatrixCPU)
def _(op: ObjMatrixCPU) -> bool:
    return True


@lx.is_symmetric.register(ObjMatrixGPU)
def _(op: ObjMatrixGPU) -> bool:
    return True


ObjMatrix: TypeAlias = ObjMatrixCPU | ObjMatrixGPU
"""Either objective-matrix operator."""
