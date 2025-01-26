"""
This suite of tests assumes that `test_utils` is passing.

Notes
-----
See the tests for the symmetric operators.
"""
import numpy as np
import pytest
import scipy.sparse as sparse
import torch
import linops as lo

from diffqcp.utils import (to_sparse_csr_tensor, to_tensor)
from diffqcp.linops import (ScalarOperator, SymmetricOperator, BlockDiag, _sLinearOperator)
from tests.utils import generate_problem_data

def test_scalar_operator():
    rng = torch.Generator().manual_seed(0)

    with pytest.raises(AssertionError):
        ScalarOperator(torch.tensor([1, 2, 3]))

    pos_scal = torch.tensor(1.23)
    neg_scal = torch.tensor(-3.21)

    pos_scal_op = ScalarOperator(pos_scal)
    neg_scal_op = ScalarOperator(neg_scal)

    v = torch.randn(1, generator=rng)

    assert torch.allclose(pos_scal_op@v, pos_scal*v)
    assert torch.allclose(neg_scal_op@v, neg_scal*v)
    assert torch.allclose(pos_scal_op.T@v, pos_scal*v)
    assert torch.allclose(neg_scal_op.T@v, neg_scal*v)


# TODO : add dtype as param for test
def test_symmetric_tensor():
    """
    For some 2-D torch tensor representing a symmetric matrix, there are numerical
    differences between the matrix vector products of this tensor and a 1-D tensor
    and the transpose of the 2-D tensor with the same 1-D tensor.
    This has a downstream effect on the `SymmetricOperator` objects.
    """
    np.random.seed(0)
    n = 100
    m = 20

    for _ in range(10):

        P, _, _, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P_upper = sparse.triu(P).tocsr()
        P_upper_tch = to_sparse_csr_tensor(P_upper)
        P_op = SymmetricOperator(n, P_upper_tch)

        x = np.random.randn(n)
        x_tch = to_tensor(x)
        P_x_tch = to_tensor(P @ x)
        PT_x_tch = to_tensor(P.T @ x)

        assert torch.allclose(P_x_tch, P_op @ x_tch, atol=1e-5, rtol=1e-5), "MV products not equal"
        assert torch.allclose(PT_x_tch, P_x_tch, atol=1e-5, rtol=1e-5), "MV transpose, products not equal for true values"
        assert torch.allclose(PT_x_tch, P_op.T @ x_tch, atol=1e-5, rtol=1e-5), "MV transpose, products not equal for operator"

def test_symmetric_callable():
    """
    Seeing the same numerical differences that arise in the `test_symmetric_tensor` test.
    """
    n = 100
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):

        X = torch.randn((n, n), generator=rng)
        X = (X + X.T)/2

        X_op = SymmetricOperator(n, lambda v : X @ v)
        v = torch.randn(n, generator=rng)

        assert torch.allclose(X_op @ v, X @ v)
        assert torch.allclose(X, X.T), "X is not symmetric"
        # diff = (X.T @ v) - (X @ v) # DEBUG
        # print(f"Max difference: {diff.abs().max()}")
        assert torch.allclose(X.T @ v, X @ v, atol=1e-5, rtol=1e-5), "MV products not equal for true values"
        assert torch.allclose(X_op.T @ v, X.T @ v, atol=1e-5, rtol=1e-5), "MV products not equal for operator"


def test_block_diag_operator():
    n = 10
    m = 5
    N = 2*n + 1
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):

        x = torch.randn(n, generator=rng)
        A = torch.randn((m, n), generator=rng)

        op1: lo.LinearOperator = lo.DiagonalOperator(x)
        op2 : lo.LinearOperator = lo.MatrixOperator(A)
        op3 : lo.LinearOperator = ScalarOperator(torch.tensor(2))
        block_op = BlockDiag([op1, op2, op3])

        v = torch.randn(N, generator=rng)
        x, y, tau = v[0:n], v[n:2*n], v[-1]

        out = torch.zeros(n+m+1)
        out[0:n] = op1 @ x
        out[n:n+m] = op2 @ y
        out[-1] = op3 @ tau.unsqueeze(0)

        u = torch.randn(n+m+1, generator=rng)
        out_transpose = torch.zeros(N)
        out_transpose[0:n] = op1.T @ u[0:n]
        out_transpose[n:2*n] = op2.T @ u[n:n+m]
        out_transpose[-1] = op3.T @ u[-1].unsqueeze(0)


        assert torch.allclose(out, block_op @ v)
        assert torch.allclose(out_transpose, block_op.T @ u)


def test_sLinearOperator():
    n = 50
    m = 20

    rng = torch.Generator().manual_seed(0)

    for _ in range(10):

        A = torch.randn((m, n), generator=rng)
        x = torch.randn(n, generator=rng)
        b = torch.randn(m, generator=rng)

        A_op = _sLinearOperator(n,
                                m,
                                lambda v : A @ v,
                                lambda u : A.T @ u,
                                supports_operator_matrix=True)

        assert torch.allclose(A @ x, A_op @ x)
        assert torch.allclose(A.T @ b, A_op.T @ b)
