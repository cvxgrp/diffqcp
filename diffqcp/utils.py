from typing import Sequence, Callable

import numpy as np
from scipy.sparse import csc_matrix

import torch
import linops as lo

def to_tensor(
    array: np.ndarray | torch.Tensor | list[float],
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
    """Convert scipy.sparse.csc_matrix to torch.sparse_csc_matrix

    Parameters
    ----------
    sparse_array : csc_matrix
        Input array
    dtype : torch.dtype, optional
        Data type for the output tensor, by default torch.float32.
    device : torch.device, optional
        Device for the output tensor, by default None.

    Returns
    -------
    torch.Tensor
        Output tensor.
    """
    if isinstance(sparse_array, csc_matrix):
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


# where does num live and where does v live?
class Scalar(lo.LinearOperator):
    supports_operator_matrix = True
    def __init__(self, num: torch.Tensor) -> None:
        assert len(num.shape) == 0
        self._num = num

        self._shape = (1, 1)
        self._adjoint = self
        self.device = num.device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._num * v


class SymmetricOperator(lo.LinearOperator):
    """TODO: Add docstring
    substitute with self-adjoint
    """

    supports_operator_matrix = True
    def __init__(self,
                 n : int,
                 op: torch.Tensor | Callable[[torch.Tensor], torch.Tensor],
                 device: torch.device | None = None
    ) -> None:
        """
        U is the upper triangular part of a symmetrix matrix
        stored as torch.sparse_csc_matrix
        """
        # assert isinstance(U, torch.Tensor)
        # assert len(U.shape) == 2

        if isinstance(op, torch.Tensor):
            self._mv = lambda v : op @ v + op.T @ v - op.diagonal() @ v
        else:
            self._mv = op

        self._shape = (n, n)
        self._adjoint = self
        self.device = device

    def _matmul_impl(self, v: torch.Tensor) -> torch.Tensor:
        return self._mv(v)


class BlockDiag(lo.LinearOperator):
    """
    """
    def __init__(self, ops: Sequence[lo.LinearOperator], adjoint=None) -> None:
        self._ops = ops
        m = 0
        n = 0
        self.supports_operator_matrix = True

        for i, op in enumerate(ops):
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

        out = torch.zeros(10, device = self.device)
        i = 0
        j = 0

        for op in self._ops:
            out[i:i + op.shape[0]] = op._matmul_impl(v[j:j + op.shape[1]])
            i += op.shape[0]
            j += op.shape[1]

        return out


class _sLinearOperator(lo.LinearOperator):

    def __init__(
        self,
        n: int,
        m: int,
        mv: Callable[[torch.Tensor], torch.Tensor],
        rv: Callable[[torch.Tensor], torch.Tensor] | lo.LinearOperator,
        device : torch.device | None = None,
        supports_operator_matrix : bool = False
    ) -> None:
        self._shape = (n, m)
        self._mv = mv
        if isinstance(rv, lo.LinearOperator):
            self._adjoint = self
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
    first_chunk = P @ x + A.T @ y + tau * q
    second_chunk = -A @ x + tau * b
    final_entry = -(1/tau) * x @ (P @ x) - q @ x - b @ y
    output = torch.zeros(N, dtype = x.dtype, device=x.device)
    output[0:n] = first_chunk
    output[n:-1] = second_chunk
    output[-1] = final_entry
    return output
