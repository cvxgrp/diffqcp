"""Small JAX/scipy interop and tree-comparison helpers for tests.

The heavier `cvx.Problem`-generation and `QCPProbData` canonicalisation now
live in `tests/problems.py`; this module is just thin utilities.
"""
from __future__ import annotations

from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Float
from scipy.sparse import (
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    sparray,
    spmatrix,
)

CPU = jax.devices("cpu")[0]

SP: TypeAlias = spmatrix | sparray
SCSR: TypeAlias = csr_matrix | csr_array
SCSC: TypeAlias = csc_matrix | csc_array
SCOO: TypeAlias = coo_matrix | coo_array


def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def get_cpu_int(a: Float[Array, " 1"]) -> int:
    return int(jnp.squeeze(jax.device_put(a, device=CPU)))


def scoo_to_bcoo(coo_mat: SCOO) -> BCOO:
    """Convert a scipy COO sparse matrix to a JAX BCOO. Caller asserts canonical form."""
    row_indices = coo_mat.row
    col_indices = coo_mat.col
    indices = list(zip(row_indices, col_indices, strict=False))
    if len(indices) == 0:
        return BCOO.fromdense(jnp.zeros(coo_mat.shape))
    return BCOO((coo_mat.data, indices), shape=coo_mat.shape)


def scsr_to_bcsr(csr_mat: SCSR) -> BCSR:
    if len(csr_mat.data) > 0:
        return BCSR(
            (csr_mat.data, csr_mat.indices, csr_mat.indptr), shape=csr_mat.shape
        )
    return BCSR.fromdense(jnp.zeros(csr_mat.shape))


def get_zeros_like_coo(A: SCOO):
    return coo_array((np.zeros(A.size), A.nonzero()), shape=A.shape)


def get_zeros_like_csr(A: SCSR):
    return csr_array(
        (np.zeros(np.size(A.data)), A.indices, A.indptr), shape=A.shape, dtype=A.dtype
    )
