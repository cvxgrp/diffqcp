"""Helper/Utility functions used """
from typing import TYPE_CHECKING

import numpy as np
from jax.numpy import argsort, stack
from jax.experimental.sparse import BCOO, BCSR
import equinox as eqx
from jaxtyping import Float, Integer, Array

def _to_int_list(v: np.ndarray) -> list[int]:
    """
    Utility function to ensure eqx.filter_{...} TODO(quill): finish

    Parameters
    ----------
    v : np.ndarray
        Should only contain intgers
    """
    return [int(val) for val in v]

class _TransposeCSRInfo(eqx.Module):
    indices: Integer[Array, "..."]
    indptr: Integer[Array, "..."]
    sorting_perm: Integer[Array, "..."]


def _coo_to_csr_transpose_map(mat: Float[BCOO, "_m _n"]) -> _TransposeCSRInfo:
    """
    we need `sorting_perm`, otherwise could just .T the BCOO array.
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

