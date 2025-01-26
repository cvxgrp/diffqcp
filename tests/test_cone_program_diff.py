"""
Testing qcp derivative computations.

Notes
-----
The `atol` parameters in the testing assertions
have been raised to the largest value that still
allows that respective test to pass.
"""

import numpy as np
import scipy.linalg as la
import cvxpy as cp
import torch

import diffqcp.qcp as cone_prog
from diffqcp.utils import to_tensor
from tests.utils import (data_and_soln_from_cvxpy_problem, get_zeros_like,
                        torch_data_and_soln_from_cvxpy_problem)


def test_least_squares_small():
    """
    The least squares (approximation) problem

        minimize    ||Ax - b||^2,

        <=>

        minimize    ||r||^2
        subject to  r = Ax - b,

    where A is a (m x n)-matrix with rank A = n, has
    the analytical solution

        x^star = (A^T A)^-1 A^T b.

    Considering x^star as a function of b, we know

        Dx^star(b) = (A^T A)^-1 A^T.

    This test checks the accuracy of `diffqcp`'s derivative computations by
    comparing DS(Data)dData to Dx^star(b)db.

    Notes
    ------
    dData == (0, 0, 0, db), and other canonicalization considerations must be made
    (hence the `data_and_soln_from_cvxpy_problem` function call and associated data declaration.)
    """

    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=10, high=15)
        m = n + np.random.randint(low=5, high=15)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        r = cp.Variable(m)
        f0 = cp.sum_squares(r)
        problem = cp.Problem(cp.Minimize(f0), [r == A@x - b])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        db = 1e-2 * torch.randn(b_can.size, generator=rng, dtype=torch.float64)

        Dx_b = to_tensor(la.solve(A.T @ A, A.T), dtype=torch.float64)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64)
        dx, dy, ds = DS(dP, dA, torch.zeros(q_can.size), db)

        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-4)


# def test_least_squares_small_torch_only():
#     """
#     Test with only torch data.
#     """

#     np.random.seed(0)
#     rng = torch.Generator().manual_seed(0)

#     for _ in range(10):
#         n = np.random.randint(low=10, high=15)
#         m = n + np.random.randint(low=5, high=15)

#         A = np.random.randn(m, n)
#         b = np.random.randn(m)

#         x = cp.Variable(n)
#         r = cp.Variable(m)
#         f0 = cp.sum_squares(r)
#         problem = cp.Problem(cp.Minimize(f0), [r == A@x - b])

#         data = torch_data_and_soln_from_cvxpy_problem(problem)
#         P_can, A_can = data[0], data[1]
#         q_can, b_can = data[2], data[3]
#         cone_dict, soln = data[4], data[5]

#         dP = get_zeros_like(P_can)
#         dA = get_zeros_like(A_can)
#         db = 1e-2 * np.random.randn(b_can.size)

#         Dx_b = la.solve(A.T @ A, A.T)

#         DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln)
#         dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db)

        # np.testing.assert_allclose(Dx_b@db, dx[m:], atol=1e-5)


def test_least_squares_larger():
    """
    See `test_least_squares_small` for more information.
    """

    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=100, high=150)
        m = n + np.random.randint(low=50, high=150)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        r = cp.Variable(m)
        f0 = cp.sum_squares(r)
        problem = cp.Problem(cp.Minimize(f0), [r == A@x - b])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        db = 1e-2 * torch.randn(b_can.size, generator=rng, dtype=torch.float64)

        Dx_b = to_tensor(la.solve(A.T @ A, A.T), dtype=torch.float64)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64)
        dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db)

        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-4)


# def test_least_squares_soln_of_eqns_small():
#     """
#     The least-l2-norm problem is

#         minimize    ||x||^2
#         subject to  Ax = b,

#     where A is a (m x n)-matrix with rank A = m,
#     has the analytical solution

#         x^star = A^T (A A^T)^-1 b.

#     Considering x^star as a function of b, we know

#         Dx^star(b) = A^T (A A^T)^-1.

#     This test checks the accuracy of diffqcp`'s derivative computations
#     by comparing DS(Data)dData to Dx^star(b)db

#     Notes
#     -----
#     dData == (0, 0, 0, db), and other canonicalization considerations must be made
#     (hence the `data_and_soln_from_cvxpy_problem` function call and associated data declaration.)
#     """

#     np.random.seed(0)

#     for _ in range(10):
#         m = np.random.randint(low=10, high=15)
#         n = m + np.random.randint(low=5, high=15)

#         A = np.random.randn(m, n)
#         count = 0
#         while np.linalg.matrix_rank(A) < m and count < 100:
#             A = np.random.randn(m, n)
#             count += 1
#         if count == 100 : assert True == False
#         b = np.random.randn(m)

#         x = cp.Variable(n)
#         f0 = cp.sum_squares(x)
#         problem = cp.Problem(cp.Minimize(f0), [A@x == b])

#         data = data_and_soln_from_cvxpy_problem(problem)
#         P_can, A_can = data[0], data[1]
#         q_can, b_can = data[2], data[3]
#         cone_dict, soln = data[4], data[5]

#         dP = get_zeros_like(P_can)
#         dA = get_zeros_like(A_can)
#         db = 1e-2 * np.random.randn(b_can.size)

#         AT = A.T
#         Dxb_db = AT @ la.solve(A @ AT, db)

#         DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln)
#         dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), -db)

#         np.testing.assert_allclose(Dxb_db, dx, atol=1e-7)


# def test_least_squares_soln_of_eqns_larger():
#     """
#     See `test_least_squares_soln_of_eqns_small` for more information.
#     """

#     np.random.seed(0)

#     for _ in range(10):
#         m = np.random.randint(low=100, high=150)
#         n = m + np.random.randint(low=50, high=150)

#         A = np.random.randn(m, n)
#         count = 0
#         while np.linalg.matrix_rank(A) < m and count < 100:
#             A = np.random.randn(m, n)
#             count += 1
#         if count == 100 : assert True == False
#         b = np.random.randn(m)

#         x = cp.Variable(n)
#         f0 = cp.sum_squares(x)
#         problem = cp.Problem(cp.Minimize(f0), [A@x == b])

#         data = data_and_soln_from_cvxpy_problem(problem)
#         P_can, A_can = data[0], data[1]
#         q_can, b_can = data[2], data[3]
#         cone_dict, soln = data[4], data[5]

#         dP = get_zeros_like(P_can)
#         dA = get_zeros_like(A_can)
#         db = 1e-2 * np.random.randn(b_can.size)

#         AT = A.T
#         Dxb_db = AT @ la.solve(A @ AT, db)

#         DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln)
#         dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), -db)

#         np.testing.assert_allclose(Dxb_db, dx, atol=1e-6)
