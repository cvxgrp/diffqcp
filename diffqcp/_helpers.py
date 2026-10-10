"""Small internal helpers shared across `diffqcp`."""

import equinox as eqx
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jax.numpy import argsort, stack
from jaxtyping import Array, Float, Integer


def _to_int_list(v: np.ndarray) -> list[int]:
    """Convert an integer array to a list of Python ints.

    Used for split indices stored as static Equinox fields: static fields must
    be hashable Python values, not NumPy/JAX scalars.
    """
    return [int(val) for val in v]


class _TransposeCSRInfo(eqx.Module):
    """How to form A^T in CSR form from A's stored values.

    `A^T` has CSR index arrays `indices` / `indptr`, and its stored values are
    `A_values[sorting_perm]`.
    """

    indices: Integer[Array, "..."]
    indptr: Integer[Array, "..."]
    sorting_perm: Integer[Array, "..."]


def _coo_to_csr_transpose_map(mat: Float[BCOO, "_m _n"]) -> _TransposeCSRInfo:
    """Precompute `_TransposeCSRInfo` for a matrix with `mat`'s sparsity pattern.

    Transposing a COO matrix only swaps its index columns; the CSR form of the
    transpose also needs the values reordered row-major, which `sorting_perm` does.
    """
    num_rows = mat.shape[0]
    rowsT, colsT = mat.indices[:, 1], mat.indices[:, 0]
    transposed_val_ordering_unsorted = rowsT * num_rows + colsT # = cols * num_rows + rows
    sorting_perm = argsort(transposed_val_ordering_unsorted)
    transposed_indices = stack([rowsT[sorting_perm], colsT[sorting_perm]], axis=1)
    mat_transposed = BCOO((mat.data[sorting_perm], transposed_indices),
                          shape=(mat.shape[1], mat.shape[0]))
    mat_transposed_csr = BCSR.from_bcoo(mat_transposed)
    return _TransposeCSRInfo(indices=mat_transposed_csr.indices,
                             indptr=mat_transposed_csr.indptr,
                             sorting_perm=sorting_perm)

