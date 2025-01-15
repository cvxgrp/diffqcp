from typing import Callable, Sequence

import torch
import linops as lo

from diffqcp.utils import sparse_csr_tensor_diag, sparse_tensor_transpose

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

            diag = sparse_csr_tensor_diag(op)
            opT = sparse_tensor_transpose(op)
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
