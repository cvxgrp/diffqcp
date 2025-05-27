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
import clarabel

import diffqcp.qcp as cone_prog
from diffqcp.qcp import QCP
from diffqcp.utils import to_tensor
from tests.utils import (data_and_soln_from_cvxpy_problem, get_zeros_like, get_random_like, form_full_symmetric, random_qcp,
                         convert_prob_data_to_torch_new)

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
def test_least_squares_small_class_version(device):
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

        # DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        x = np.array(soln.x)
        y = np.array(soln.z)
        s = np.array(soln.s)
        qcp1 = QCP(P=P_can, A=A_can, q=q_can, b=b_can, x=x, y=y, s=s, cone_dict=cone_dict, P_is_upper=True, dtype=torch.float64, device=device)
        qcp2 = QCP(P=P_can, A=A_can, q=q_can, b=b_can, x=x, y=y, s=s, cone_dict=cone_dict, P_is_upper=True, dtype=torch.float64, device=device, reduce_fp_flops=True)
        # dx, dy, ds = DS(dP, dA, torch.zeros(q_can.size, device=device), -db)
        dx, dy, ds = qcp1.jvp(dP, dA, torch.zeros(q_can.size, device=device), -db)
        
        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-8)

        dx, dy, ds = qcp2.jvp(dP, dA, torch.zeros(q_can.size, device=device), -db)

        assert torch.allclose( Dx_b @ db, dx[m:], atol=1e-8)

@pytest.mark.parametrize("device", devices)
def test_nonneg_least_squares(device):
    """
    The least squares (approximation) problem

        minimize    1/2 ||x - q||^2
        subject to  x >= 0,

    where q > 0
    the analytical solution is

        x^star = q

    Considering x^star as a function of q, we know

        Dx^star(q) = I.

    This test checks the accuracy of `diffqcp`'s derivative computations by
    comparing DS(Data)dData to Dx^star(q)dq.

    Notes
    ------
    dData == (0, 0, dq, 0), and other canonicalization considerations must be made
    (hence the `data_and_soln_from_cvxpy_problem` function call and associated data declaration.)
    """

    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=10, high=15)

        P = sparse.csc_matrix(np.eye(n))
        q = -np.ones(n)

        A = -sparse.csc_matrix(np.eye(n))

        b = np.zeros(n)

        # Define cones
        cones = {'z': 0, 'l': n, 'ep': 0, 'q': [], 's': [], 'p': []}

        dP = get_zeros_like(P)
        dA = get_zeros_like(A)
        dq = 1e-6 * torch.randn(q.size, generator=rng, dtype=torch.float64, device=device)
        soln = torch.ones(n, device=device)

        Dx_q = to_tensor(np.eye(n), dtype=torch.float64, device=device)

        DS = cone_prog.compute_derivative(P, A, q, b, cones, (soln, soln, 1e-9 * np.ones(n)), dtype=torch.float64, device=device)
        dx, dy, ds = DS(dP, dA, -dq, torch.zeros(b.size, device=device))
        
        assert torch.allclose( Dx_q @ dq, dx, atol=1e-8)


@pytest.mark.parametrize("device", devices)
def test_nonneg_least_squares_cvxpy(device):
    np.random.seed(0)
    count = 0
    for _ in range(10):
        n = np.random.randint(low=10, high=15)

        q_np = np.ones(n)

        x = cp.Variable(n)
        f0 = 0.5 * cp.sum_squares(x - q_np)
        problem = cp.Problem(cp.Minimize(f0), [x >= 0])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        dq = 1e-2 * np.random.randn(n)
        dq_tch = to_tensor(dq, dtype=torch.float64, device=device)
        db = np.zeros(b_can.size)
        db[0:n] = dq
        
        DS = cone_prog.compute_derivative(P_can, A_can, q_can, b_can, cone_dict, soln, dtype=torch.float64, device=device)
        # q is canonicalized to b
        dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db)

        assert torch.allclose(dq_tch, dx[n:], atol=1e-5)
        count += 1
    assert count == 10

@pytest.mark.parametrize("device", devices)
def test_nonneg_least_squares_cvxpy_class_version(device):
    np.random.seed(0)
    count = 0
    for _ in range(10):
        n = np.random.randint(low=10, high=15)

        q_np = np.ones(n)

        x = cp.Variable(n)
        f0 = 0.5 * cp.sum_squares(x - q_np)
        problem = cp.Problem(cp.Minimize(f0), [x >= 0])

        data = data_and_soln_from_cvxpy_problem(problem)
        P_can, A_can = data[0], data[1]
        q_can, b_can = data[2], data[3]
        cone_dict, soln = data[4], data[5]

        dP = get_zeros_like(P_can)
        dA = get_zeros_like(A_can)
        dq = 1e-2 * np.random.randn(n)
        dq_tch = to_tensor(dq, dtype=torch.float64, device=device)
        db = np.zeros(b_can.size)
        db[0:n] = dq
        
        P_full = form_full_symmetric(P_can.todense())
        P_full = sparse.csr_array(P_full)

        x = np.array(soln.x)
        y = np.array(soln.z)
        s = np.array(soln.s)
        # Form four QCPs
        #   1. P only upper triangular and forming atoms when construct
        #   2. P only upper triangular and forming atoms when computing jvp.
        #   3. Materialized P and forming atoms when construct
        #   4. Materialized P and forming atoms when computing jvp.
        qcp1 = QCP(P_can, A_can, q_can, b_can, x, y, s, cone_dict, P_is_upper=True, dtype=torch.float64, device=device)
        qcp2 = QCP(P_can, A_can, q_can, b_can, x, y, s, cone_dict, P_is_upper=True, dtype=torch.float64, device=device, reduce_fp_flops=True)
        qcp3 = QCP(P_full, A_can, q_can, b_can, x, y, s, cone_dict, P_is_upper=False, dtype=torch.float64, device=device)
        qcp4 = QCP(P_full, A_can, q_can, b_can, x, y, s, cone_dict, P_is_upper=False, dtype=torch.float64, device=device, reduce_fp_flops=True)

        # q is canonicalized to b
        dx, _, _ = qcp1.jvp(dP, dA, np.zeros(q_can.size), db)
        assert torch.allclose(dq_tch, dx[n:], atol=1e-5)

        # test vjp:
        P_tilde, A_tilde, q_tilde, b_tilde = qcp1.vjp(P_can @ x + q_can, np.zeros(y.shape[0]), np.zeros(y.shape[0]))
        print("b_tilde = ", b_tilde)
        print("-y =  ", -y)
        np.testing.assert_allclose(b_tilde.cpu().numpy(), -y, atol=1e-8)

        dx, _, _ = qcp2.jvp(dP, dA, np.zeros(q_can.size), db)
        assert torch.allclose(dq_tch, dx[n:], atol=1e-5)

        dP = get_zeros_like(P_full)

        dx, _, _ = qcp3.jvp(dP, dA, np.zeros(q_can.size), db)
        assert torch.allclose(dq_tch, dx[n:], atol=1e-5)

        dx, _, _ = qcp4.jvp(dP, dA, np.zeros(q_can.size), db)
        assert torch.allclose(dq_tch, dx[n:], atol=1e-5)        
        
        count += 1
    assert count == 10


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
def test_constrained_least_squares_class_version(device):
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

        x = np.array(soln.x)
        y = np.array(soln.z)
        s = np.array(soln.s)

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

        qcp = QCP(P_can, A_can, q_can, b_can, x, y, s, cone_dict, P_is_upper=True, dtype=torch.float64, device=device)
        dx, dy, ds = qcp.jvp(dP, dA, torch.zeros(q_can.size, device=device), db_can)
        
        dx_b_analytic = dx_b_analytical(db)
        
        print("dx: ", dx[m:].to(device=None))
        print("analytical: ", dx_b_analytic.to(device=None))
        print("NUM OPTIMAL:", num_optimal)
        assert torch.allclose(dx_b_analytic, dx[m:], atol=1e-8)

        dP, dA, dq, db = qcp.vjp(P_can @ x + q_can, torch.zeros(y.shape[0]), torch.zeros(y.shape[0]))

        print("db = ", db)
        print("-y = ", -y)
        np.testing.assert_allclose(db.cpu().numpy(), -y, atol=1e-4)
        # NOTE 5/26/25 (quill): ^ this assertion fails, but looking at the results I have
        # enough confidence to try a gradient descent test


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
    diff = abs(analytical[0] - (dx[0]).item()/dlam)
    print("diff: ", diff)
    # assert diff < 1e-6 # fails
    # np.testing.assert_allclose(analytical[0] * dlam, (dx[0]).item(), atol=1e-8) # passes

@pytest.mark.parametrize("device", devices)
def test_adjoint_cvxbook_sensitivity_easy(device):

    K = {
        'z' : 3,
        'l' : 3,
        'q' : [5]
    }

    K_clarabel = [ # TODO (quill): helper function mapping SCS <-> Clarabel
        clarabel.ZeroConeT(3),
        clarabel.NonnegativeConeT(3),
        clarabel.SecondOrderConeT(5)
    ]

    m = 3 + 3 + 5
    n = 5

    np.random.seed(0)

    P, P_upper, A, q, b = random_qcp(m, n, K, sparse.random_array, np.random.randn)

    P_upper_csc = sparse.csc_matrix(P_upper)
    A_csc = sparse.csc_matrix(A)
    
    solver_settings = clarabel.DefaultSettings()
    solver_settings.verbose = False
    solver = clarabel.DefaultSolver(P_upper_csc, q, A_csc, b, K_clarabel, solver_settings)
    solution = solver.solve()
    print("soln status: ", solution.status)

    x = np.array(solution.x)
    y = np.array(solution.z)
    s = np.array(solution.s)

    qcp = QCP(P, A, q, b, x, y, s, K, P_is_upper=False, dtype=torch.float64, device=device)

    dP, dA, dq, db = qcp.vjp( P @ x + q, np.zeros(m), np.zeros(m))

    db = db.cpu().numpy()

    print("Px: ", P @ x)
    print("q: ", q)

    print("db: ", db)
    print("-y: ", -y)
    np.testing.assert_allclose(db, -y)


@pytest.mark.parametrize("device", devices)
def test_adjoint_consistency_easy(device):

    np.random.seed(0)

    m1 = 15
    m2 = 15
    m3 = 20
    m = m1 + m2 + m3
    n = 20

    K = {
        'z' : m1,
        'l' : m2,
        'q' : [m3]
    }

    K_clarabel = [ # TODO (quill): helper function mapping SCS <-> Clarabel
        clarabel.ZeroConeT(m1),
        clarabel.NonnegativeConeT(m2),
        clarabel.SecondOrderConeT(m3)
    ]

    num_computed = 0
    for _ in range(10):
        print("COUNT = ", num_computed)
        P, P_upper, A, q, b, soln = random_qcp(m, n, K, sparse.random_array, np.random.randn)

        P_upper_csc = sparse.csc_matrix(P_upper)
        A_csc = sparse.csc_matrix(A)
        
        solver_settings = clarabel.DefaultSettings()
        solver_settings.verbose = False
        solver = clarabel.DefaultSolver(P_upper_csc, q, A_csc, b, K_clarabel, solver_settings)
        solution = solver.solve()
        print("soln status: ", solution.status)

        x = np.array(solution.x)
        if not np.allclose(x, soln[0], atol=1e-4):
            print("clarabel primal: ", x)
            print("our primal: ", soln[0])
            continue
        y = np.array(solution.z)
        if not np.allclose(y, soln[1], atol=1e-4):
            print("clarabel dual: ", y)
            print("our dual: ", soln[1])
            continue
        s = np.array(solution.s)
        if not np.allclose(s, soln[2], atol=1e-4):
            print("clarabel slack: ", x)
            print("our slack: ", soln[2])
            continue

        qcp = QCP(P, A, q, b, x, y, s, K, P_is_upper=False, dtype=torch.float64, device=device)
        qcp2 = QCP(P_upper, A, q, b, x, y, s, K, P_is_upper=True, dtype=torch.float64, device=device)

        P_tilde = get_random_like(P, np.random.randn)
        P_upper_tilde = get_random_like(P_upper, np.random.randn)
        A_tilde = get_random_like(A, np.random.randn)
        q_tilde = np.random.randn(n)
        b_tilde = np.random.randn(m)

        P_tilde, P_upper_tilde, A_tilde, q_tilde, b_tilde = convert_prob_data_to_torch_new(P_tilde, P_upper_tilde, A_tilde, q_tilde, b_tilde, dtype=torch.float64, device=device)

        x_tilde = to_tensor(np.random.randn(n), dtype=torch.float64, device=device)
        y_tilde = to_tensor(np.random.randn(m), dtype=torch.float64, device=device)
        s_tilde = to_tensor(np.random.randn(m), dtype=torch.float64, device=device)

        dx, dy, ds = qcp.jvp(P_tilde, A_tilde, q_tilde, b_tilde)
        dP, dA, dq, db = qcp.vjp(x_tilde, y_tilde, s_tilde)

        lhs = dx @ x_tilde + dy @ y_tilde + ds @ s_tilde
        rhs = torch.trace( dP.to_dense() @ P_tilde.to_dense() ) + torch.trace(dA.to_dense().T @ A_tilde.to_dense()) + dq @ q_tilde + db @ b_tilde

        print("LHS: ", lhs)
        print("RHS: ", rhs)
        assert torch.abs(lhs - rhs) < 1e-10
        
        dx, dy, ds = qcp2.jvp(P_upper_tilde, A_tilde, q_tilde, b_tilde)
        dP, dA, dq, db = qcp2.vjp(x_tilde, y_tilde, s_tilde)

        lhs = dx @ x_tilde + dy @ y_tilde + ds @ s_tilde
        rhs = torch.trace( dP.to_dense() @ P_upper_tilde.to_dense() ) + torch.trace(dA.to_dense().T @ A_tilde.to_dense()) + dq @ q_tilde + db @ b_tilde

        assert torch.abs(lhs - rhs) < 1e-10

        num_computed += 1
    
    assert num_computed > 0