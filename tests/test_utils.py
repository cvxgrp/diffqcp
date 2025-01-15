import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix)
import torch
import pytest

from diffqcp import utils
from tests.utils import generate_problem_data

def test_to_tensor():
    """
    Taken from:
    https://github.com/cvxgrp/randalo/blob/main/test/test_utils.py
    """
    x = [1, 2, 3]
    x_np = np.array(x)
    assert x_np.dtype == np.dtype("int64")

    x_torch = torch.tensor(x)
    assert x_torch.dtype == torch.int64

    x_torch = x_torch.to(torch.float32)

    x_to_tensor = utils.to_tensor(x)
    assert torch.allclose(x_torch, x_to_tensor)
    assert x_to_tensor.dtype == torch.float32

    x_np_to_tensor = utils.to_tensor(x_np)
    assert torch.allclose(x_torch, x_np_to_tensor)
    assert x_np_to_tensor.dtype == torch.float32

    x_torch_to_tensor = utils.to_tensor(x_torch)
    assert torch.allclose(x_torch, x_torch_to_tensor)
    assert x_torch_to_tensor.dtype == torch.float32

    with pytest.raises(ValueError):
        utils.to_tensor(1)


def test_to_sparse_csc_tensor():
    np.random.seed(0)
    P, A, _, _ = generate_problem_data(10, 5, sparse.random, np.random.randn)

    assert P.dtype == np.dtype("float64")
    assert A.dtype == np.dtype("float64")

    P_tch = utils.to_sparse_csc_tensor(P)
    A_tch = utils.to_sparse_csc_tensor(A)

    assert P_tch.dtype == torch.float32
    assert A_tch.dtype == torch.float32

    x = np.random.randn(10)
    x_tch = utils.to_tensor(x)
    P_x = P @ x
    A_x = A @ x
    assert torch.allclose(P_tch @ x_tch, utils.to_tensor(P_x))
    assert torch.allclose(A_tch @ x_tch, utils.to_tensor(A_x))

    with pytest.raises(ValueError):
        utils.to_sparse_csc_tensor(P_tch)


def test_to_sparse_csr_tensor():
    np.random.seed(0)
    P, A, _, _ = generate_problem_data(10, 5, sparse.random, np.random.randn)

    assert P.dtype == np.dtype("float64")
    assert A.dtype == np.dtype("float64")

    P_tch = utils.to_sparse_csr_tensor(P)
    A_tch = utils.to_sparse_csr_tensor(A)

    assert P_tch.dtype == torch.float32
    assert A_tch.dtype == torch.float32

    x = np.random.randn(10)
    x_tch = utils.to_tensor(x)
    P_x = P @ x
    A_x = A @ x
    assert torch.allclose(P_tch @ x_tch, utils.to_tensor(P_x))
    assert torch.allclose(A_tch @ x_tch, utils.to_tensor(A_x))

    with pytest.raises(ValueError):
        utils.to_sparse_csc_tensor(P_tch)


def test_tch_csc_diagonal_extraction():
    np.random.seed(0)
    n = 100
    m = 20

    for _ in range(10):
        P, A, q, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P, A = csc_matrix(P), csc_matrix(A)
        P_tch = utils.to_sparse_csc_tensor(P)
        A_tch = utils.to_sparse_csc_tensor(A)
        X = torch.tensor(np.random.randn(n, m))
        q_tch = utils.to_tensor(q)

        with pytest.raises(AssertionError):
            utils.sparse_csc_tensor_diag(A_tch)

        with pytest.raises(AssertionError):
            utils.sparse_csc_tensor_diag(q_tch)

        with pytest.raises(AssertionError):
            utils.sparse_csc_tensor_diag(X)

        with pytest.raises(AssertionError):
            P_tch2 = utils.to_sparse_csr_tensor(P)
            utils.sparse_csc_tensor_diag(P_tch2)

        P_tch_diag = utils.sparse_csc_tensor_diag(P_tch)

        assert torch.allclose(utils.to_tensor(P.diagonal()), P_tch_diag)

def test_tch_csr_diagonal_extraction():
    np.random.seed(0)
    n = 100
    m = 20

    for _ in range(10):
        P, A, q, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P_tch = utils.to_sparse_csr_tensor(P)
        A_tch = utils.to_sparse_csr_tensor(A)
        X = torch.tensor(np.random.randn(n, m))
        q_tch = utils.to_tensor(q)

        with pytest.raises(AssertionError):
            utils.sparse_csr_tensor_diag(A_tch)

        with pytest.raises(AssertionError):
            utils.sparse_csr_tensor_diag(q_tch)

        with pytest.raises(AssertionError):
            utils.sparse_csr_tensor_diag(X)

        with pytest.raises(AssertionError):
            P_tch2 = utils.to_sparse_csc_tensor(P)
            utils.sparse_csr_tensor_diag(P_tch2)

        P_tch_diag = utils.sparse_csr_tensor_diag(P_tch)

        assert torch.allclose(utils.to_tensor(P.diagonal()), P_tch_diag)

def test_tch_csc_transpose():
    np.random.seed(0)
    n = 100
    m = 5

    for _ in range(10):
        P, A, _, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P = sparse.triu(P).tocsc()
        A = csc_matrix(A)

        P_tch = utils.to_sparse_csc_tensor(P)
        A_tch = utils.to_sparse_csc_tensor(A)
        PT = utils.sparse_tensor_transpose(P_tch)
        AT = utils.sparse_tensor_transpose(A_tch)

        assert torch.allclose(PT.to_dense().to(dtype=torch.float32),
                            torch.tensor(P.T.todense(), dtype=torch.float32))

        assert torch.allclose(AT.to_dense().to(dtype=torch.float32),
                            torch.tensor(A.T.todense(), dtype=torch.float32))

def test_tch_csr_transpose():
    np.random.seed(0)
    n = 100
    m = 5

    for _ in range(10):
        P, A, _, _ = generate_problem_data(n, m, sparse.random, np.random.randn)
        P = sparse.triu(P).tocsr()

        P_tch = utils.to_sparse_csr_tensor(P)
        A_tch = utils.to_sparse_csr_tensor(A)
        PT = utils.sparse_tensor_transpose(P_tch)
        AT = utils.sparse_tensor_transpose(A_tch)

        assert torch.allclose(PT.to_dense().to(dtype=torch.float32),
                            torch.tensor(P.T.todense(), dtype=torch.float32))

        assert torch.allclose(AT.to_dense().to(dtype=torch.float32),
                            torch.tensor(A.T.todense(), dtype=torch.float32))
