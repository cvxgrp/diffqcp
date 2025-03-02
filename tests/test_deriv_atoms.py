"""
Testing qcp derivative atom computations.

Notes
-----
- The `atol` parameters in the testing assertions
have been raised to the largest value that still
allows that respective test to pass.

- Setting u[-1] equal to 1 (which will always be tue at the solution),
yields more accurate derivative approximations (this makes sense when looking
at the nonlinear transform Q).
"""

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix
import torch
import linops as lo
from linops.lsqr import lsqr as lsqr2
import pytest

from diffqcp.qcp_derivs import Du_Q, dData_Q
from diffqcp.lsqr import lsqr
import tests.utils as utils
from diffqcp.utils import Q

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda')]

@pytest.mark.parametrize("device", devices)
def test_dData_Q_is_approximation(device):
    """Test implementation of DQ(u) w.r.t. Data.

    Taking D in S^n_+ x R^(n x m) x R^n x R^m and a small perturbation
    dD in S^n x R^(n x m) x R^n x R^m, test if

        Q(u, D + dD) - Q(u, D) approx. equal DQ(u, D)dD,

    where DQ(u, D)dD is computed using `diffqcp.qcp_deriv.dData_Q`.

    Notes
    -----
    Can also test with P in S^n (symmetric, but not PSD).
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64,
                                                                   device=device)

        u = torch.randn(N, generator=rng,dtype=torch.float64, device=device)
        u[-1] = torch.tensor(1, dtype=torch.float64, device=device) # always the case when differentiating at soln.

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P_op, A, q, b, x, y, tau)

        dP_upper = utils.get_random_like(P_upper, lambda n: np.random.normal(0, 1e-6, size=n))
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
        dq = torch.randn(q.shape[0], generator=rng)
        db = torch.randn(b.shape[0], generator=rng)
        dP_op, dA, dq, db = utils.convert_prob_data_to_torch(dP_upper, dA, dq, db, dtype=torch.float64, device=device)

        dQ = Q(P_op + dP_op, A + dA, q + dq, b + db, x, y, tau) - z

        assert torch.allclose(dQ, dData_Q(u, dP_op, dA, dq, db))


def test_Du_Q_is_approximation():
    """Test implementation of DQ(u) w.r.t. u.

    Taking u in R^N and a small perturbation du in R^N, test if

        Q(u + du, D) - Q(u, D) approx. equal DQ(u, D)du,

    where DQ(u, D) is obtained from `diffqcp.qcp_deriv.Du_Q`.

    Notes
    -----
    When attempted with atol=1e-6, test fails.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=10, high=20)
        m = np.random.randint(low=10, high=20)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64)

        u = torch.randn(N, generator=rng, dtype=torch.float64)
        u[-1] = torch.tensor(1, dtype=torch.float64) # always the case when differentiating at soln.

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P_op, A, q, b, x, y, tau)

        du = 1e-5*torch.randn(N, generator=rng, dtype=torch.float64)
        dx, dy, dtau = du[:n], du[n: -1], du[-1]
        dtau = dtau.unsqueeze(0)
        dQ = Q(P_op, A, q, b, x + dx, y + dy, tau + dtau) - z

        deriv_op = Du_Q(u, P_op, A, q, b)

        assert torch.allclose(dQ, deriv_op @ du)


# def test_Du_Q_T_is_approximation():
#     """Test adjoint implementation of DQ(u) w.r.t. u.

#     Taking u in R^N and a small perturbation du in R^N, test if

#         du approx. equal DQ(u, D)_T(Q(u + du, D) - Q(u, D)),

#     where DQ(u, D)_T is obtained from `diffqcp.qcp_deriv.Du_Q`.

#     Notes
#     -----
#     """
#     np.random.seed(0)
#     rng = torch.Generator().manual_seed(0)

#     for i in range(10):
#         m = np.random.randint(low=10, high=20)
#         n = m + np.random.randint(low=5, high=15)
#         N = n + m + 1

#         P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
#                                                                    m,
#                                                                    sparse.random,
#                                                                    np.random.randn,
#                                                                    dtype=torch.float64)

#         u = torch.randn(N, generator=rng, dtype=torch.float64)
#         u[-1] = torch.tensor(1.0, dtype=torch.float64) # always the case when differentiating at soln.
#         x, y, tau = u[:n], u[n: -1], u[-1]
#         z = Q(P_op, A, q, b, x, y, tau)

#         du = 1e-2*torch.randn(N, generator=rng, dtype=torch.float64)
#         dx, dy, dtau = du[:n], du[n: -1], du[-1]
#         dQ = Q(P_op, A, q, b, x + dx, y + dy, tau + dtau) - z

#         deriv_op = Du_Q(u, P_op, A, q, b)

#         assert torch.allclose(du, deriv_op.T @ dQ, atol=1e-6)


def test_Du_Q_is_linop():
    """Test if Du_Q is a linear operator.

    Another way of testing if Du_Q_T is implemented correctly.
    That is, instead of taking the output-input approximation
    validation approach used in `test_Du_Q_T_is_approximation`,
    this test uses `pylops.utils.dottest` to see if
    Du_Q_T is the adjoint of Du_Q.

    Notes
    -----
    See "Testing the operator" section of
    https://pylops.readthedocs.io/en/stable/adding.html#addingoperator
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64)
        u = torch.randn(N, generator=rng,dtype=torch.float64)
        u[-1] = torch.tensor(1, dtype=torch.float64) # always the case when differentiating at soln.

        deriv_op = Du_Q(u, P_op, A, q, b)

        # default dtype of dottest is float64
        assert utils.dottest(deriv_op)


def test_Du_Q_lsqr():
    """Test lsqr performance for Du_Q.

    For some u

    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64)
        u = torch.randn(N, generator=rng, dtype=torch.float64)
        u[-1] = torch.tensor(1.0, dtype=torch.float64) # always the case when differentiating at soln.

        deriv_op = Du_Q(u, P_op, A, q, b)

        x = 1e-6*torch.randn(N, generator=rng, dtype=torch.float64)
        xlsqr = lsqr(deriv_op, deriv_op @ x)

        print("x: ", x)
        print("xlsqr: ", xlsqr)

        # np.testing.assert_allclose(x, xlsqr, atol=1e-4)
        assert torch.allclose(x, xlsqr, atol=1e-8)

def test_Du_Q_lsqr2():
    """Test lsqr performance for Du_Q.

    For some u

    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64)
        u = torch.randn(N, generator=rng, dtype=torch.float64)
        u[-1] = torch.tensor(1.0, dtype=torch.float64) # always the case when differentiating at soln.

        deriv_op = Du_Q(u, P_op, A, q, b)

        x = 1e-6*torch.randn(N, generator=rng, dtype=torch.float64)
        xlsqr = lsqr2(deriv_op, deriv_op @ x)

        print("x: ", x)
        print("xlsqr: ", xlsqr)

        # np.testing.assert_allclose(x, xlsqr, atol=1e-4)
        assert torch.allclose(x, xlsqr, atol=1e-8)
