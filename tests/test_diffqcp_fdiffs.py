"""
Use finite differences to test JVPs of various QCPs.

Status (Feb. 2 2025)
- least squares: PASSING
- least norm: PASSING
- pure nonneg con (`test_nonneg_cone`): PASSING
- QCP with dim 20 SOC: passing for 1e-7 tol (1e-6 perturbation)
- QCP with intersection of nonneg cone, zero cone, SOC: PASSING
"""

import numpy as np
import torch
import cvxpy as cp
import clarabel

from tests.utils import data_and_soln_from_cvxpy_problem, get_random_like, get_zeros_like
from diffqcp import compute_derivative
from diffqcp import utils

NUM_TRIALS = 10

def _test_DS(prob: cp.Problem,
             tol: float = 1e-8,
             dtype: torch.dtype = torch.float64
) -> None:
    """
    Helper function to test Jacobian-vector products for QCP solution maps.

    Note the function itself will make `assert` statements; nothing is returned.
    """
    prob_data = data_and_soln_from_cvxpy_problem(prob)
    P_upper, A = prob_data[0], prob_data[1]
    q, b = prob_data[2], prob_data[3]
    cone_dict, soln, clarabel_cones = prob_data[4], prob_data[5], prob_data[6]

    DS = compute_derivative(P_upper, A, q, b, cone_dict,
                            solution=soln, dtype=dtype)

    dP_upper = get_random_like(P_upper, lambda n: np.random.normal(0, 1e-6, size=n))
    dA = get_random_like(A, lambda n: np.random.normal(0, 1e-6, size=n))
    dq = np.random.normal(0, 1e-6, size=q.size)
    db = np.random.normal(0, 1e-6, size=b.size)

    solver_settings = clarabel.DefaultSettings()
    solver_settings.verbose = False
    solver = clarabel.DefaultSolver(P_upper + dP_upper,
                                    q + dq,
                                    A + dA,
                                    b + db,
                                    clarabel_cones, solver_settings)
    delta_soln = solver.solve()

    delta_x = utils.to_tensor(delta_soln.x, dtype=dtype) - utils.to_tensor(soln.x, dtype=dtype)
    delta_y = utils.to_tensor(delta_soln.z, dtype=dtype) - utils.to_tensor(soln.z, dtype=dtype)
    delta_s = utils.to_tensor(delta_soln.s, dtype=dtype) - utils.to_tensor(soln.s, dtype=dtype)

    dx, dy, ds = DS(dP_upper, dA, dq, db)

    print(f"delta_x: {delta_x} \n dx: {dx} \n === ===")
    print(f"delta_y: {delta_y} \n dy: {dy} \n === ===")
    print(f"delta_s: {delta_s} \n ds: {ds} \n === ===")

    assert torch.allclose(delta_x, dx, atol=tol)
    assert torch.allclose(delta_y, dy, atol=tol)
    assert torch.allclose(delta_s, ds, atol=tol)


def test_least_squares():
    """
    See `test_least_squares_small` for more information.
    """

    np.random.seed(0)
    failed = 0
    for _ in range(NUM_TRIALS):
        n = np.random.randint(low=10, high=15)
        m = n + np.random.randint(low=5, high=15)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        r = cp.Variable(m)
        f0 = cp.sum_squares(r)
        prob = cp.Problem(cp.Minimize(f0), [r == A@x - b])
        prob.solve()
        if prob.status == 'optimal':
            _test_DS(prob)
        else:
            failed += 1
    
    if failed == NUM_TRIALS:
        assert False, "test_least_squares was never actually checked"


def test_least_norm():
    """
    Test DS(data)d_data for the problem

        minimize    ||x||^2
        subject to  Ax = b,

    where rank(A) = m.
    """
    np.random.seed(0)
    failed = 0
    for _ in range(NUM_TRIALS):
        m = np.random.randint(25, 75)
        n = m + np.random.randint(5, 30)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        
        x = cp.Variable(n)
        f0 = cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(f0), [A@x == b])
        prob.solve()
        if prob.status == 'optimal':
            _test_DS(prob)
        else:
            failed += 1
    
    if failed == NUM_TRIALS:
        assert False, "test_least_norm was never actually checked"


def test_ls_nonneg_cone():
    """
    Test DS(data)d_data for the problem

        minimize    ||Ax - b||^2
        subject to  x >= 0,

    where >= is componentwise and rank(A) = n.

    Notes
    -----
    This one never has all nonzero dual variables.
    """
    np.random.seed(0)
    failed = 0
    for _ in range(NUM_TRIALS):
        n = np.random.randint(25, 75)
        m = n + np.random.randint(5, 30)
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        
        x = cp.Variable(n)
        f0 = cp.sum_squares(A@x - b)
        constrs = [x >= 0]
        prob = cp.Problem(cp.Minimize(f0), constrs)
        prob.solve()
        print(f"DUAL VALUE: {constrs[0].dual_value}")
        if prob.status == 'optimal' and np.all(x.value > 0):
            _test_DS(prob)
        else:
            failed += 1
    
    if failed == NUM_TRIALS:
        assert False, "test_ls_nonneg_cone was never actually checked"


def test_nonneg_cone():
    """
    Test DS(data)d_data for the problem

        minimize    c^Tx + ||x||
        subject to  x >= 0,

    where >= is componentwise.
    """
    np.random.seed(0)
    failed = 0
    for _ in range(NUM_TRIALS):
        n = np.random.randint(25, 75)
        c = np.random.randn(n)
        
        x = cp.Variable(n)
        f0 = c @ x + cp.sum_squares(x)
        prob = cp.Problem(cp.Minimize(f0), [x >= 0])
        prob.solve()
        if prob.status == 'optimal':
            _test_DS(prob)
        else:
            failed += 1
    
    if failed == NUM_TRIALS:
        assert False, "test_nonneg_cone was never actually checked"


def test_socp_proj_deriv():
    """
    First iteration of the following fails, but the values are quite close.
    All iterations pass for tol=1e-7

    Looking closely at the difference betweens the deltas and differentials,
    this one is close enough to passing.
    """
    np.random.seed(0)
    failed = 0
    for i in range(NUM_TRIALS):
        n = np.random.randint(3, 30)
        x = np.random.randn(n)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [cp.norm(z[1:], 2) <= z[0]]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        if prob.status == 'optimal':
            print(f"NUM CHECKED: {i}")
            _test_DS(prob, tol=1e-8)
        else:
            failed += 1
    if failed == NUM_TRIALS:
        assert False, "test_socp_proj_deriv was never actually checked"


def test_socp():
    x = cp.Variable(shape=(3,))
    y = cp.Variable()
    soc = cp.constraints.second_order.SOC(y, x)
    constraints = [soc,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if prob.status == 'optimal':
        _test_DS(prob)
    else:
        assert False, "test_socp wasn't run because problem was infeasible"


def test_socp2():
    """Now with quadratic objective.
    """
    x = cp.Variable(shape=(3,))
    y = cp.Variable()
    soc = cp.constraints.second_order.SOC(y, x)
    constraints = [soc,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y <= 5]
    obj = cp.Minimize(3 * x[0]**2 + 2 * x[1] + x[2])
    prob = cp.Problem(obj, constraints)
    prob.solve()
    if prob.status == 'optimal':
        _test_DS(prob)
    else:
        assert False, "test_socp2 wasn't run because problem was infeasible"