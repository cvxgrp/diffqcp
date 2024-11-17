import numpy as np
from scipy import sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla
import pylops as lo
import cvxpy as cp
import clarabel
import pytest

import diffqcp.qcp as cone_prog

def abstract_derivative_equality():
    pass


def dense_derivative_equality():
    pass

def test_least_squares(self):
    """

    minimize    ||r||^2
    subject to  r = Ax - b.

    <=>

    minimize    ||Ax - b||^2

    x^\star = (A^TA)^-1 A^T b
    
    """
    
    # Generate data

    np.random.seed(0)

    m, n = 10, 5

    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # Compute analytical solution and derivative

    Dx_b = la.solve(A.T @ A, A.T)
    absDx_b = sla.aslinearoperator(Dx_b)
    x_ls = Dx_b @ b
    f0_ls = la.norm(A @ x_ls - b)**2

    # Use CVXPY to generate canonical problem data for CLARABEL

    x = cp.Variable(n)
    r = cp.Variable(m)

    f0 = cp.sum_squares(r)

    problem = cp.Problem(cp.Minimize(f0), [r == A@x - b])

    probdata, _, _ = problem.get_problem_data(cp.CLARABEL)

    P_can, q_can = probdata['P'], probdata['c']
    A_can, b_can = probdata['A'], probdata['b']
    cone_dims = probdata['dims']
    
    clarabel_cones = [clarabel.ZeroConeT(cone_dims.zero)]

    solver = clarabel.DefaultSolver(P_can, q_can,
                               A_can, b_can,
                               clarabel_cones, clarabel.DefaultSettings())
    
    solution = solver.solve()

    # need a helper function to evaluate derivatives.



    # abstract this to a utility function at some point

    problem_solution

    # assert solution is as you think it should be (for sanity)

    # assert about the derivative.

def test_tikhonov_regularization(self):
    """
    
    minimize    ||s||^2 + ||x||^2
    subject to  Ax + s = b.

    x^\star = (A^TA + I)^-1 A^T b,

    considering the solution as a function of the data parameter b, i.e.,
        
        x^\star(b) = (A^TA + I)^-1 A^T b,

    the derivataive of the solution is just the matrix (A^TA + I)^-1 A^T.

    Steps:
    1. generate data A, b
    2. Compute the solution


    """
    pass