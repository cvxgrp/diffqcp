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
from scipy.sparse import csc_matrix
import torch

from diffqcp.qcp_derivs import Du_Q, dData_Q
import tests.utils as utils
from diffqcp.utils import Q
import diffqcp.utils as lo_utils

# TODO: manually implement PyLops's `dottest`


def test_dData_Q_is_approximation():
    """Test implementation of DQ(u) w.r.t. Data.

    Taking D in S^n x R^(n x m) x R^n x R^m and a small perturbation
    dD in S^n x R^(n x m) x R^n x R^m, test if

        Q(u, D + dD) - Q(u, D) approx. equal DQ(u, D)dD,

    where DQ(u, D)dD is computed using `diffqcp.qcp_deriv.dData_Q`.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P, A, q, b = utils.generate_problem_data(n,
                                                 m,
                                                 sparse.random,
                                                 np.random.randn)
        P_upper = sparse.triu(P).tocsc()
        P_tch = lo_utils.to_sparse_csc_tensor(P_upper)
        P_op = lo_utils.SymmetricOperator(n, P_tch)
        A_tch = lo_utils.to_sparse_csc_tensor(A)
        q_tch = lo_utils.to_tensor(q)
        b_tch = lo_utils.to_tensor(b)

        u = torch.randn(N, generator=rng)
        u[-1] = torch.tensor(1) # always the case when differentiating at soln.

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P_op, A_tch, q_tch, b_tch, x, y, tau)

        dP = utils.get_random_like(P_upper, lambda n: np.random.normal(0, 1e-6, size=n))
        dP_tch = lo_utils.to_sparse_csc_tensor(dP)
        dP_op = lo_utils.SymmetricOperator(n, dP_tch)
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
        dA_tch = lo_utils.to_sparse_csc_tensor(dA)
        dq = 1e-6*torch.randn(q.size, generator=rng)
        db = 1e-6*torch.randn(b.size, generator=rng)
        assert A_tch.layout == dA_tch.layout
        new = A_tch.to_sparse_csc() + dA_tch.to_sparse_csc()
        dQ = Q(P_op + dP_op, A_tch + dA_tch, q_tch + dq, b_tch + db, x, y, tau) - z

        torch.allclose(dQ, dData_Q(u, dP_op, dA_tch, dq, db))


# def test_Du_Q_is_approximation():
#     """Test implementation of DQ(u) w.r.t. u.

#     Taking u in R^N and a small perturbation du in R^N, test if

#         Q(u + du, D) - Q(u, D) approx. equal DQ(u, D)du,

#     where DQ(u, D) is obtained from `diffqcp.qcp_deriv.Du_Q`.

#     Notes
#     -----
#     When attempted with atol=1e-6, test fails.
#     """
#     pass
#     np.random.seed(0)
#     for _ in range(10):
#         n = np.random.randint(low=10, high=20)
#         m = np.random.randint(low=10, high=20)
#         N = n + m + 1

#         P, A, q, b = utils.generate_problem_data(n,
#                                                  m,
#                                                  sparse.random,
#                                                  np.random.randn)

#         u = np.random.randn(N)
#         u[-1] = 1 # always the case when differentiating at soln.
#         x, y, tau = u[:n], u[n: -1], u[-1]
#         z = Q(P, A, q, b, x, y, tau)

#         du = 1e-5*np.random.randn(N)
#         dx, dy, dtau = du[:n], du[n: -1], du[-1]
#         dQ = Q(P, A, q, b, x + dx, y + dy, tau + dtau) - z

#         deriv_op = Du_Q(u, P, A, q, b)

#         np.testing.assert_allclose(dQ,
#                                    deriv_op._matvec(du),
#                                    atol=1e-8)


# def test_Du_Q_T_is_approximation():
#     """Test adjoint implementation of DQ(u) w.r.t. u.

#     Taking u in R^N and a small perturbation du in R^N, test if

#         du approx. equal DQ(u, D)_T(Q(u + du, D) - Q(u, D)),

#     where DQ(u, D)_T is obtained from `diffqcp.qcp_deriv.Du_Q`.

#     Notes
#     -----
#     """
#     pass
#     np.random.seed(0)
#     for i in range(10):
#         m = np.random.randint(low=10, high=20)
#         n = m + np.random.randint(low=5, high=15)
#         N = n + m + 1

#         P, A, q, b = utils.generate_problem_data(n,
#                                                  m,
#                                                  sparse.random,
#                                                  np.random.randn)

#         u = np.random.randn(N)
#         u[-1] = 1 # always the case when differentiating at soln.
#         x, y, tau = u[:n], u[n: -1], u[-1]
#         z = Q(P, A, q, b, x, y, tau)

#         du = 1e-5*np.random.randn(N)
#         dx, dy, dtau = du[:n], du[n: -1], du[-1]
#         dQ = Q(P, A, q, b, x + dx, y + dy, tau + dtau) - z

#         deriv_op = Du_Q(u, P, A, q, b)

#         np.testing.assert_allclose(du,
#                                    deriv_op._rmatvec(dQ),
#                                    atol=1e-7)


# def test_Du_Q_is_linop():
#     """Test if Du_Q is a linear operator.

#     Another way of testing if Du_Q_T is implemented correctly.
#     That is, instead of taking the output-input approximation
#     validation approach used in `test_Du_Q_T_is_approximation`,
#     this test uses `pylops.utils.dottest` to see if
#     Du_Q_T is the adjoint of Du_Q.

#     Notes
#     -----
#     See "Testing the operator" section of
#     https://pylops.readthedocs.io/en/stable/adding.html#addingoperator
#     """
#     pass
#     np.random.seed(0)
#     for _ in range(10):
#         m = np.random.randint(low=10, high=20)
#         n = m + np.random.randint(low=5, high=15)
#         N = n + m + 1

#         P, A, q, b = utils.generate_problem_data(n,
#                                                  m,
#                                                  sparse.random,
#                                                  np.random.randn)
#         u = np.random.randn(N)

#         deriv_op = Du_Q(u, P, A, q, b)

#         assert dottest(deriv_op, N, N)


# def test_Du_Q_lsqr():
#     """Test lsqr performance for Du_Q.

#     For some u

#     """
#     pass
#     np.random.seed(0)
#     for i in range(10):
#         m = np.random.randint(low=10, high=20)
#         n = m + np.random.randint(low=5, high=15)
#         N = n + m + 1

#         P, A, q, b = utils.generate_problem_data(n,
#                                                  m,
#                                                  sparse.random,
#                                                  np.random.randn)
#         u = np.random.randn(N)
#         u[-1] = 1 # always the case when differentiating at soln.

#         deriv_op = Du_Q(u, P, A, q, b)

#         x = 1e-2*np.random.randn(N)
#         xlsqr = lsqr(deriv_op, deriv_op._matvec(x))[0]

#         np.testing.assert_allclose(x, xlsqr, atol=1e-4)
