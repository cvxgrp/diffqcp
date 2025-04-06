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
import scipy.sparse as sparse
import cvxpy as cp
import torch
import pytest

import diffqcp.qcp as cone_prog
from diffqcp.utils import to_tensor
from tests.utils import data_and_soln_from_cvxpy_problem, get_zeros_like, get_random_like

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda')]

@pytest.mark.parametrize("device", devices)
def test_least_squares_small(device):
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
    rng = torch.Generator(device=device).manual_seed(0)

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
        assert b.size == b_can.size
        np.testing.assert_allclose(-b, b_can) # sanity check
        db = 1e-6 * torch.randn(b_can.size, generator=rng, dtype=torch.float64, device=device)

        Dx_b = to_tensor(la.solve(A.T @ A, A.T), dtype=torch.float64, device=device)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, torch.zeros(q_can.size, device=device), -db)
        
        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-8)


@pytest.mark.parametrize("device", devices)
def test_least_squares_larger(device):
    """
    See `test_least_squares_small` for more information.

    Notes
    -----
    db scaled by 1e-6 and .allclose with atol == 1e-8 also works here.
    """

    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

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
        assert b.size == b_can.size # sanity check
        np.testing.assert_allclose(-b, b_can)
        db = 1e-2 * torch.randn(b_can.size, generator=rng, dtype=torch.float64, device=device)

        Dx_b = to_tensor(la.solve(A.T @ A, A.T), dtype=torch.float64, device=device)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), -db)

        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_least_squares_soln_of_eqns_small(device):
    """
    The least-l2-norm problem is

        minimize    ||x||^2
        subject to  Ax = b,

    where A is a (m x n)-matrix with rank A = m,
    has the analytical solution

        x^star = A^T (A A^T)^-1 b.

    Considering x^star as a function of b, we know

        Dx^star(b) = A^T (A A^T)^-1.

    This test checks the accuracy of diffqcp`'s derivative computations
    by comparing DS(Data)dData to Dx^star(b)db

    Notes
    -----
    dData == (0, 0, 0, db), and other canonicalization considerations must be made
    (hence the `data_and_soln_from_cvxpy_problem` function call and associated data declaration.)
    """

    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):
        m = np.random.randint(low=10, high=15)
        n = m + np.random.randint(low=5, high=15)

        A = np.random.randn(m, n)
        count = 0
        while np.linalg.matrix_rank(A) < m and count < 100:
            A = np.random.randn(m, n)
            count += 1
        if count == 100:
            assert False
        b = np.random.randn(m)

        x = cp.Variable(n)
        f0 = cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(f0), [A@x == b])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        db = 1e-2 * torch.randn(b_can.size, generator=rng, dtype=torch.float64, device=device)
        db_np = db.cpu().numpy()
        
        AT = A.T
        Dxb_db = to_tensor(AT @ la.solve(A @ AT, db_np), dtype=torch.float64, device=device)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db) # leave zero creation with numpy to see if gpu conversion worked

        assert torch.allclose(Dxb_db, dx, atol=1e-5)


@pytest.mark.parametrize("device", devices)
def test_least_squares_soln_of_eqns_larger(device):
    """
    See `test_least_squares_soln_of_eqns_small` for more information.
    """

    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):
        m = np.random.randint(low=100, high=150)
        n = m + np.random.randint(low=50, high=150)

        A = np.random.randn(m, n)
        count = 0
        while np.linalg.matrix_rank(A) < m and count < 100:
            A = np.random.randn(m, n)
            count += 1
        if count == 100:
            assert False
        b = np.random.randn(m)

        x = cp.Variable(n)
        f0 = cp.sum_squares(x)
        problem = cp.Problem(cp.Minimize(f0), [A@x == b])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        db = 1e-2 * torch.randn(b_can.size, generator=rng, dtype=torch.float64, device=device)
        db_np = db.cpu().numpy()

        AT = A.T
        Dxb_db = to_tensor(AT @ la.solve(A @ AT, db_np), dtype=torch.float64, device=device)

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db)

        assert torch.allclose(Dxb_db, dx, atol=1e-5)


### More complicated examples ###

@pytest.mark.parametrize("device", devices)
def test_constrained_least_squares(device):
    np.random.seed(0)
    EPS = 1e-6

    num_optimal = 0
    for _ in range(10):
        n = np.random.randint(low=10, high=15)
        m = n + np.random.randint(low=5, high=15)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        f0 = cp.sum_squares(A@x - b)
        constrs = [x >= 0]
        problem = cp.Problem(cp.Minimize(f0), constrs)
        problem.solve()
        
        if problem.status != 'optimal':
            continue
        
        lambd: np.ndarray = constrs[0].dual_value

        S = [i for i, val in enumerate(x.value) if val > EPS]
        S_bar_len = n - len(S)
        I_s_bar = np.eye(n)
        I_s_bar = np.delete(I_s_bar, S, axis=1)
        A_hat = np.block([
            [A.T @ A, I_s_bar],
            [I_s_bar.T, np.zeros((S_bar_len, S_bar_len))]
        ])
        b_hat = np.hstack((A.T @ b, np.zeros(S_bar_len)))
        soln = la.solve(A_hat, b_hat)
        print("x_star from LS equation: ", soln[0:n])
        print("x_star from CVXPY: ", x.value)
        if not np.allclose(soln[0:n], x.value, atol=1e-6):
            continue
        num_optimal += 1
        print("lambda_{s_bar} from LS equation: ", soln[n:])
        print("lambda from CVXPY: ", lambd)

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        np.testing.assert_allclose(b, b_can[0:b.size])

        def dx_b_analytical(db) -> np.ndarray:
            dsoln = la.solve(A_hat, np.hstack((A.T @ db,
                                              np.zeros(S_bar_len))
                                              )
                            )
            dx_b = dsoln[0:n]
            return to_tensor(dx_b, dtype=torch.float64, device=device)

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        db = 1e-6 * np.random.randn(b.size)
        db_can = np.hstack((db, np.zeros(b_can.size - b.size)))

        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, torch.zeros(q_can.size, device=device), db_can)
        
        dx_b_analytic = dx_b_analytical(db)
        
        print("dx: ", dx[m:].to(device=None))
        print("analytical: ", dx_b_analytic.to(device=None))
        print("NUM OPTIMAL:", num_optimal)
        assert torch.allclose(dx_b_analytic, dx[m:], atol=1e-8)

    assert num_optimal == 10, "No derivative testing was actually performed."


@pytest.mark.parametrize("device", devices)
def test_dprojection_exp(device):
    x = cp.Variable()
    lam = cp.Parameter(1, nonneg=True)
    lam.value = np.ones(1)

    f0 = x + lam * (cp.log(1 + x) + cp.log(1 - x))
    problem = cp.Problem(cp.Maximize(f0))

    data = data_and_soln_from_cvxpy_problem(problem)
    P_can, A_can = data[0], data[1]
    q_can, b_can = data[2], data[3]
    cone_dict, soln = data[4], data[5]

    DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)

    dlam = 1e-6
    dP = get_zeros_like(P_can)
    # dA = get_random_like(A_can, lambda n: np.random.normal(0, 1e-6, size=n))
    dA = get_zeros_like(A_can)
    db = torch.zeros(b_can.shape[0])
    dq = torch.zeros(q_can.shape[0])
    dq[1] = -dlam
    dq[2] = -dlam

    dx, dy, ds = DS(dP, dA, dq, db)

    analytical = -1 + lam.value / np.sqrt(lam.value**2 + 1)

    print("dx: ", (dx[0]).item())
    print("analytical: ", analytical[0] * dlam)

    # assert abs(analytical[0] - (dx[0]).item()/dlam) < 1e-6
    np.testing.assert_allclose(analytical[0] * dlam, (dx[0]).item(), atol=1e-8)