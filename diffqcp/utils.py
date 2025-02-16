"""
Utilities for manipulating torch tensors + the homogeneous, nonlinear embedding mapping.
"""

from numbers import Number

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, spmatrix, csr_matrix)

import torch
import linops as lo

def to_tensor(
    array: np.ndarray | torch.Tensor | list[Number],
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Convert a numpy array or torch tensor to a torch tensor.

    Parameters
    ----------
    array : np.ndarray | torch.Tensor | list[float]
        Input array or tensor.
    dtype : torch.dtype, optional
        Data type for the output tensor, by default torch.float32.
    device : torch.device, optional
        Device for the output tensor, by default None.

    Returns
    -------
    torch.Tensor
        Output tensor.

    Notes
    -----
    Taken from https://github.com/cvxgrp/randalo/blob/main/randalo/utils.py
    """

    if isinstance(array, np.ndarray):
        return torch.tensor(array, dtype=dtype, device=device)
    elif isinstance(array, torch.Tensor) or isinstance(array, list):
        return torch.as_tensor(array, dtype=dtype, device=device)
    else:
        raise ValueError("Input must be a numpy array, torch tensor, or list.")
    

def to_sparse_csc_tensor(sparse_array : spmatrix,
                         dtype: torch.dtype = torch.float32,
                         device: torch.device | None = None
) -> torch.Tensor:
    """Convert a scipy.sparse.spmatrix to a torch.sparse_csc_matrix.

    Parameters
    ----------
    sparse_array : spmatrix
        Input array
    dtype : torch.dtype, optional
        Data type for the output tensor, by default torch.float32.
    device : torch.device, optional
        Device for the output tensor, by default None.

    Returns
    -------
    torch.Tensor
        Output tensor in sparse_csc layout.

    Notes
    -----
    Oddly, when calling this function in diffqcp.test_utils.py on a
    scipy.sparse.csc_matrix, the sparse_array within this function was
    a scipy.sparse.coo_matrix. Hence the generic check and then casting to
    csc matrix.
    """

    if isinstance(sparse_array, sparse.spmatrix):

        if not isinstance(sparse_array, csc_matrix):
            sparse_array = csc_matrix(sparse_array)

        sparse_array = sparse_array if isinstance(sparse_array, csc_matrix) else csc_matrix(sparse_array)

        ccol_indices = torch.tensor(sparse_array.indptr, dtype=torch.int64, device=device)
        row_indices = torch.tensor(sparse_array.indices, dtype=torch.int64, device=device)
        values = torch.tensor(sparse_array.data, dtype=dtype, device=device)

        return torch.sparse_csc_tensor(
            ccol_indices=ccol_indices,
            row_indices=row_indices,
            values=values,
            size=sparse_array.shape,
            dtype=dtype,
            device=device,
        )

    else:
        raise ValueError("Input must be a scipy sparse matrix in CSC format")


def to_sparse_csr_tensor(sparse_array: spmatrix,
                         dtype: torch.dtype = torch.float32,
                         device: torch.device | None = None
) -> torch.Tensor:
    """Convert a scipy.sparse.spmatrix to a torch.sparse_csr_matrix.

    Parameters
    ----------
    sparse_array : spmatrix
        Input array.
    dtype : torch.dtype, optional
        Data type for the output tensor, by default torch.float32.
    device : torch.device, optional
        Device for the output tensor, by default None.

    Returns
    -------
    torch.Tensor
        Output tensor in sparse_csr layout.
    """

    if isinstance(sparse_array, spmatrix):

        sparse_array = sparse_array if isinstance(sparse_array, csr_matrix) else csr_matrix(sparse_array)

        crow_indices = torch.tensor(sparse_array.indptr, dtype=torch.int64, device=device)
        col_indices = torch.tensor(sparse_array.indices, dtype=torch.int64, device=device)
        values = torch.tensor(sparse_array.data, dtype=dtype, device=device)

        return torch.sparse_csr_tensor(
            crow_indices = crow_indices,
            col_indices = col_indices,
            values = values,
            size = sparse_array.shape,
            dtype = dtype,
            device = device
        )

    else:
        raise ValueError("Input must be a scipy sparse matrix.")
    

def sparse_csc_tensor_diag(X : torch.Tensor) -> torch.Tensor:
    """Extracts the diagonal of a square 2-D tensor.

    Parameters
    ----------
    X : torch.Tensor (in sparse csc format)
        A 2-D tensor (<=> len(X.shape) == 2).

    Returns
    -------
    torch.Tensor
        The 1-D diagonal tensor of X.
    """

    assert len(X.shape) == 2
    assert X.layout == torch.sparse_csc
    assert X.shape[0] == X.shape[1]

    n = X.shape[0]
    indptr = X.ccol_indices()
    indices = X.row_indices()
    values = X.values()

    diagonal = torch.zeros(n, dtype=X.dtype, device=X.device)

    for col in range(n):
        row_vals_start = indptr[col] # also start index of data vals for column
        row_vals_end = indptr[col+1] # also end index of data vals for column
        for i, row in enumerate(indices[row_vals_start:row_vals_end]):
            if row == col:
                diagonal[col] = values[row_vals_start + i]
            elif row > col:
                continue

    return diagonal


def sparse_csr_tensor_diag(X: torch.Tensor) -> torch.Tensor:
    """Extracts the diagonal of a square 2-D tensor.

    Parameters
    ----------
    X : torch.Tensor (in sparse csc format)
        A 2-D tensor (<=> len(X.shape) == 2).

    Returns
    -------
    torch.Tensor
        The 1-D diagonal tensor of X.
    """

    assert len(X.shape) == 2
    assert X.layout == torch.sparse_csr
    assert X.shape[0] == X.shape[1]

    n = X.shape[0]
    indptr = X.crow_indices()
    indices = X.col_indices()
    values = X.values()

    diagonal = torch.zeros(n, dtype=X.dtype, device=X.device)

    for row in range(n):
        col_vals_start = indptr[row] # also start index of data vals for row
        col_vals_end = indptr[row+1]
        for i, col in enumerate(indices[col_vals_start:col_vals_end]):
            if row == col:
                diagonal[row] = values[col_vals_start + i]
            elif row > col:
                continue

    return diagonal


def sparse_tensor_diag(X: torch.Tensor) -> torch.Tensor:
    """Extracts the diagonal of a square 2-D tensor.

    Parameters
    ----------
    X : torch.Tensor (in sparse csc or csr format)
        A 2-D tensor (<=> len(X.shape) == 2)

    Returns
    -------
    torch.Tensor
        The 1-D diagonal tensor of X.
    """
    assert len(X.shape) == 2
    assert (X.layout == torch.sparse_csr or X.layout == torch.sparse_csr)
    assert X.shape[0] == X.shape[1]

    if X.layout == torch.sparse_csr:
        return sparse_csr_tensor_diag(X)
    else:
        return sparse_csc_tensor_diag(X)


def sparse_tensor_transpose(X: torch.Tensor,
                            output_csr: bool=False
) -> torch.Tensor:
    """Return the transpose of a sparse 2-D tensor.

    Parameters
    ----------
    X : torch.Tensor (in sparse csc format or sparse csr format)
        A 2-D tensor (<=> len(X.shape) == 2).
    output_csr : bool, optional
        Whether the returned transpose should be a sparse csr tensor.
        If X is a sparse csc tensor, then X^T is already outputted
        as a sparse csr tensor.

    Returns
    --------
    torch.Tensor
        X^T. If X is in sparse csr format then X^T will by default be in
        sparse csc format (it can be cast to csr format for additional
        flops using the `output_csr` flag).
        If X is in sparse csc format then X^T will always be returned
        in sparse csr format.

    Notes
    -----
    BE ADVISED:
        if outputting X^T in sparse_csc format, be sure
        the operations performed with X^T are supported by
        torch.sparse_csc tensors. Such operations are
        - matrix-vector multiplication (X^T @ v),
        - matrix-matrix multiplication (X^T @ V),
        - scaling the matrix (alpha * X^T).
        An operation that is not supported is matrix-matrix
        addition (X^T + V).
    """

    assert len(X.shape) == 2
    assert (X.layout == torch.sparse_csc or
        X.layout == torch.sparse_csr)

    if X.layout == torch.sparse_csc:
        return torch.sparse_csr_tensor(
            crow_indices = X.ccol_indices(),
            col_indices = X.row_indices(),
            values = X.values(),
            size = (X.shape[1], X.shape[0]),
            dtype=X.dtype,
            device=X.device)
    else:
        out = torch.sparse_csc_tensor(
                ccol_indices = X.crow_indices(),
                row_indices = X.col_indices(),
                values = X.values(),
                size = (X.shape[1], X.shape[0]),
                dtype=X.dtype,
                device=X.device)
        if not output_csr:
            return out
        else:
            return out.to_sparse_csc()
        
def _get_GPU_settings(P: torch.Tensor,
                      dtype: torch.device | None,
                      device: torch.device | None
) -> tuple[torch.dtype, torch.device | None]:
    """Convenience method to reduce number of lines in `compute_derivative`.

    See the docstring of `compute_derivative` for the explanation of this
    method's logic.
    """
    if dtype is None and isinstance(P, torch.Tensor):
        DTYPE = P.dtype
    elif dtype is not None:
        assert isinstance(dtype, torch.dtype)
        DTYPE = dtype
    else:
        DTYPE = torch.float32

    if device is None and isinstance(P, torch.Tensor):
        DEVICE = P.device
    else:
        assert (device is None or isinstance(device, torch.device))
        DEVICE = device
    
    return DTYPE, DEVICE


def _convert_problem_data(P: torch.Tensor | spmatrix,
                          A: torch.Tensor | spmatrix,
                          q: torch.Tensor | np.ndarray,
                          b: torch.Tensor | np.ndarray,
                          dtype: torch.dtype,
                          device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience method to reduce number of lines in `compute_derivative`.
    """
    if isinstance(P, spmatrix):
        P = to_sparse_csr_tensor(P, dtype, device)
    elif isinstance(P, torch.Tensor):
        P = P.to_sparse_csr() if P.layout != torch.sparse_csr else P
        P = P.to(dtype=dtype, device=device)
    else:
        raise ValueError("P (and dP) must be a torch Tensor or a sparse scipy matrix."
            + " (And only the upper triangular part of the mathematical"
            + " P it represents should be provided.)")

    if isinstance(A, spmatrix):
        A = to_sparse_csr_tensor(A, dtype, device)
    elif isinstance(A, torch.Tensor):
        A = A.to_sparse_csr() if A.layout != torch.sparse_csr else A
        A = A.to(dtype=dtype, device=device)
    else:
        raise ValueError("A (and dA) must be a torch Tensor or a sparse scipy matrix.")

    q = to_tensor(q, dtype, device)
    b = to_tensor(b, dtype, device)

    return P, A, q, b


def Q(P: torch.Tensor | lo.LinearOperator,
      A: torch.Tensor,
      q: torch.Tensor,
      b: torch.Tensor,
      x: torch.Tensor,
      y: torch.Tensor,
      tau: torch.Tensor
) -> torch.Tensor:
    """Homogeneous embedding, nonlinear transform.

    See the diffqcp paper for the origin of this function.
    It is used throughout this repository for testing purposes only.
    """
    n = x.shape[0]
    N = n + y.shape[0] + 1
    AT = sparse_tensor_transpose(A)
    output = torch.empty(N, dtype = x.dtype, device=x.device)
    output[0:n] = P @ x + AT @ y + tau * q
    output[n:-1] = -A @ x + tau * b
    output[-1] = -(1/tau) * x @ (P @ x) - q @ x - b @ y

    return output
