from typing import Sequence, Callable
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
        raise ValueError("Input must be a numpy array or torch tensor")

def to_sparse_csc_tensor(sparse_array : csc_matrix,
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
            elif row > col : continue

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
            elif row > col : continue

    return diagonal

def sparse_tensor_transpose(X: torch.Tensor,
                            output_csr: bool=False
) -> torch.Tensor:
    """Return the transpose of a sparse 2-D torch tensor.

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
        return torch.sparse_csc_tensor(
            ccol_indices = X.crow_indices(),
            row_indices = X.col_indices(),
            values = X.values(),
            size = (X.shape[1], X.shape[0]),
            dtype=X.dtype,
            device=X.device)

class ScalarOperator(lo.LinearOperator):
    """A scalar linear operator.

    Not to be confused with a scalar, this operator
    maps 1-D tensors of length 1 to 1-D tensors of
    length 1.
    """
    supports_operator_matrix = True
    def __init__(self, num: torch.Tensor) -> None:
        """Initialize the ScalarOperator object.

        Parameters
        ----------
        num : torch.Tensor
            A **scalar tensor,** or equivalently, a zero-dimensional
            array.
        """
        assert len(num.shape) == 0
        # or (len(num.shape) == 1 and len(num) == )

        self._num = num

        self._shape = (1, 1)
        self._adjoint = self
        self.device = num.device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._num * v


class SymmetricOperator(lo.LinearOperator):
    """A symmetric linear operator.

    That is, this class can be used to create any linear
    operator L that satisfies L = L.T.

    See the constructor's docstring for implementation details.
    But broadly, a SymmetricOperator object can be created by
    providing the **upper triangular** part of a symmetric tensor,
    or providing a callable that defines how the operator maps
    vectors.
    """

    supports_operator_matrix = True
    def __init__(self,
                 n : int,
                 op: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
                 device: torch.device | None = None
    ) -> None:
        """Initialize the SymmetricOperator object.

        Parameters
        ----------
        n : int
            The number
        op : torch.Tensor | Callable[[torch.Tensor], torch.Tensor]
            Either the **upper triangular** part of a symmetric tensor
            in sparse_csc layout, **or** a function that accepts a single, 1-D
            torch tensor of length n and outputs a 1-D torch tensor of length n.
        device : torch.device, optional
            Default machine is the host. It is recommended to provide the device
            (loosely) the op is on.
        """
        if isinstance(op, torch.Tensor):
            assert len(op.shape) == 2

            diag = sparse_csc_tensor_diag(op)
            opT = sparse_csc_tensor_transpose(op)
            self._mv = lambda v : op @ v + opT @ v - diag * v
        else:
            self._mv = op

        self._shape = (n, n)
        self._adjoint = self
        self.device = device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._mv(v)


class BlockDiag(lo.LinearOperator):
    """Block-diagonal operator.

    Create a block-diagonal operator from a sequence of `linops.LinearOperator` objects.
    """
    def __init__(self,
                 ops: Sequence[lo.LinearOperator],
                 adjoint: lo.LinearOperator | None =None) -> None:
        """Initialize the BlockDiag object.

        Parameters
        ----------
        ops : Sequence[lo.LinearOperator]
            Linear operators to be stacked.
        adjoint : lo.LinearOperator | None, optional
            The adjoint of the block diagonal operator.
            There's no reason to provide this; it exists
            as a parameter purely so the `BlockDiag` object
            being created can create its own adjoint.
        """
        self._ops = ops
        m = 0
        n = 0
        self.supports_operator_matrix = True

        for op in ops:
            assert isinstance(op, lo.LinearOperator)

            if not op.supports_operator_matrix:
                self.supports_operator_matrix = False

            m += op.shape[0]
            n += op.shape[1]

        self._shape = (m, n)
        if adjoint is None:
            self._adjoint = BlockDiag([op.T for op in ops], self)
        else:
            self._adjoint = adjoint
        self.device = ops[0].device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        assert v.shape[0] == self._shape[1]

        out = torch.zeros(self.shape[0], device = self.device)
        i = 0
        j = 0

        for op in self._ops:
            out[i:i + op.shape[0]] = op._matmul_impl(v[j:j + op.shape[1]])
            i += op.shape[0]
            j += op.shape[1]

        return out


class _sLinearOperator(lo.LinearOperator):
    """Convenience class for creating scipy-like linops.
    """

    def __init__(
        self,
        n: int,
        m: int,
        mv: Callable[[torch.Tensor], torch.Tensor],
        rv: Callable[[torch.Tensor], torch.Tensor] | lo.LinearOperator | None = None,
        device : torch.device | None = None,
        supports_operator_matrix : bool = False
    ) -> None:
        """Initialize the _sLinearOperator object.

        Defines the linear operator L: R^n -> R^m.

        Parameters
        ----------
        n : int
            The length of the 1-D tensors the operator acts on.
        m : int
            The length of the 1-D tensors the operator outputs.
        mv : Callable[[torch.Tensor], torch.Tensor]
            Returns L @ v, where v is a 1-D tensor of length n and
            L is the operator
        rv : Callable[[torch.Tensor], torch.Tensor] | lo.LinearOperator | None, optional
            Returns L.T @ u, where u is a 1-D tensor of length m.
            Usually to create a `_sLinearOperator`, this parameter
            will be a `Callable`. When this is the case, the constructor
            create the adjoint of L by creating another `_sLinearOperator`
            where the `mv` parameter provided to that constructor will be
            this `rv` parameter.
            If a lo.LinearOperator is provided, this parameter will be the adjoint
            of the lo.LinearOperator created by this constructor.
            If this parameter is not provided, then the adjoint of L will be autogenerated
            by differentiating through the `mv` function.
        device : torch.device | None, optional
            Default is the host.
        supports_operator_matrix : bool, optional
            Whether mv handles matrix inputs correctly. Default is `False`.
        """
        self._shape = (n, m)
        self._mv = mv
        if isinstance(rv, lo.LinearOperator):
            self._adjoint = rv
        elif rv is not None:
            self._adjoint = _sLinearOperator(m, n, rv, self)
        # else we don't instantiate adjoint and make linops differentiate to find it.

        self.device = device
        self.supports_operator_matrix = supports_operator_matrix

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._mv(v)


def Q(P: torch.Tensor | lo.LinearOperator,
      A: torch.Tensor,
      q: torch.Tensor,
      b: torch.Tensor,
      x: torch.Tensor,
      y: torch.Tensor,
      tau: torch.Tensor
) -> torch.Tensor:
    """Homogeneous embedding, nonlinear transform.

    check if P is only upper part
    """
    n = x.shape[0]
    N = n + y.shape[0] + 1
    AT = sparse_csc_tensor_transpose(A)
    output = torch.zeros(N, dtype = x.dtype, device=x.device)
    output[0:n] = P @ x + AT @ y + tau * q
    output[n:-1] = -A @ x + tau * b
    output[-1] = -(1/tau) * x @ (P @ x) - q @ x - b @ y

    return output
