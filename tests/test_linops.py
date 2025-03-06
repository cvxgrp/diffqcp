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

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda')]


@pytest.mark.parametrize("device", devices)
def test_scalar_operator(device):
    rng = torch.Generator(device=device).manual_seed(0)

    with pytest.raises(AssertionError):
        ScalarOperator(torch.tensor([1, 2, 3], device=device))

    pos_scal = torch.tensor(1.23, device=device)
    neg_scal = torch.tensor(-3.21, device=device)

    pos_scal_op = ScalarOperator(pos_scal)
    neg_scal_op = ScalarOperator(neg_scal)

    v = torch.randn(1, generator=rng, device=device)

    assert torch.allclose(pos_scal_op@v, pos_scal*v)
    assert torch.allclose(neg_scal_op@v, neg_scal*v)
    assert torch.allclose(pos_scal_op.T@v, pos_scal*v)
    assert torch.allclose(neg_scal_op.T@v, neg_scal*v)


@pytest.mark.parametrize("device", devices)
def test_symmetric_tensor(device):
    """Note that rtol can be increased if dtype is set to torch.float64.
    """
    np.random.seed(0)
    n = 100
    m = 20

    for _ in range(10):

        P, _, _, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P_upper = sparse.triu(P).tocsr()
        P_upper_tch = to_sparse_csr_tensor(P_upper, device=device)
        P_op = SymmetricOperator(n, P_upper_tch)

        x = np.random.randn(n)
        x_tch = to_tensor(x, device=device)
        P_x_tch = to_tensor(P @ x, device=device)
        PT_x_tch = to_tensor(P.T @ x, device=device)

        assert torch.allclose(P_x_tch, P_op @ x_tch, atol=1e-5, rtol=1e-5), "MV products not equal"
        assert torch.allclose(PT_x_tch, P_x_tch, atol=1e-5, rtol=1e-5), "MV transpose, products not equal for true values"
        assert torch.allclose(PT_x_tch, P_op.T @ x_tch, atol=1e-5, rtol=1e-5), "MV transpose, products not equal for operator"


@pytest.mark.parametrize("device", devices)
def test_symmetric_callable(device):
    n = 100
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):

        X = torch.randn((n, n), generator=rng, device=device)
        X = (X + X.T)/2

        X_op = SymmetricOperator(n, lambda v : X @ v, device=device)
        v = torch.randn(n, generator=rng, device=device)

        assert torch.allclose(X_op @ v, X @ v)
        assert torch.allclose(X, X.T), "X is not symmetric"
        assert torch.allclose(X.T @ v, X @ v, atol=1e-5, rtol=1e-5), "MV products not equal for true values"
        assert torch.allclose(X_op.T @ v, X.T @ v, atol=1e-5, rtol=1e-5), "MV products not equal for operator"


@pytest.mark.parametrize("device", devices)
def test_block_diag_operator(device):
    n = 10
    m = 5
    N = 2*n + 1
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):

        x = torch.randn(n, generator=rng, device=device)
        A = torch.randn((m, n), generator=rng, device=device)

        op1: lo.LinearOperator = lo.DiagonalOperator(x)
        op2 : lo.LinearOperator = lo.MatrixOperator(A)
        op3 : lo.LinearOperator = ScalarOperator(torch.tensor(2, device=device))
        block_op = BlockDiag([op1, op2, op3])

        v = torch.randn(N, generator=rng, device=device)
        x, y, tau = v[0:n], v[n:2*n], v[-1]

        out = torch.empty(n+m+1, device=device)
        out[0:n] = op1 @ x
        out[n:n+m] = op2 @ y
        out[-1] = op3 @ tau.unsqueeze(0)

        u = torch.randn(n+m+1, generator=rng, device=device)
        out_transpose = torch.empty(N, device=device)
        out_transpose[0:n] = op1.T @ u[0:n]
        out_transpose[n:2*n] = op2.T @ u[n:n+m]
        out_transpose[-1] = op3.T @ u[-1].unsqueeze(0)


        assert torch.allclose(out, block_op @ v)
        assert torch.allclose(out_transpose, block_op.T @ u)


@pytest.mark.parametrize("device", devices)
def test_sLinearOperator(device):
    n = 50
    m = 20

    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):

        A = torch.randn((m, n), generator=rng, device=device)
        x = torch.randn(n, generator=rng, device=device)
        b = torch.randn(m, generator=rng, device=device)

        A_op = _sLinearOperator(n,
                                m,
                                lambda v : A @ v,
                                lambda u : A.T @ u,
                                supports_operator_matrix=True)

        assert torch.allclose(A @ x, A_op @ x)
        assert torch.allclose(A.T @ b, A_op.T @ b)
