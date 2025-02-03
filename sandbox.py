"""
Temporary file to tinker with QCP JVP comptutations.
"""

import numpy as np
import cvxpy as cp
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import torch
import clarabel
import scipy
import linops as lo
from torchsparsegradutils.utils.lsmr import lsmr

from diffqcp import compute_derivative
from diffqcp import utils
from tests.utils import data_and_soln_from_cvxpy_problem, get_random_like

def _test_DS(prob: cp.Problem,
             tol: float = 1e-8,
             dtype: torch.dtype = torch.float64
) -> None:
    """
    Helper function to test Jacobian-vector products for QCP solution maps.

    Note the function itself will make `assert` statements; nothing is returned.

    Notes
    -----
    If a 
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

    # assert torch.allclose(delta_x, dx, atol=tol)
    # assert torch.allclose(delta_y, dy, atol=tol)
    # assert torch.allclose(delta_s, ds, atol=tol)

def test_socp():
    x = cp.Variable(shape=(3,))
    y = cp.Variable()
    soc = cp.constraints.second_order.SOC(y, x)
    constraints = [soc,
                   x[0] + x[1] + 3 * x[2] >= 1.0,
                   y <= 5]
    obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
    prob = cp.Problem(obj, constraints)
    _test_DS(prob)

if __name__ == "__main__":
    # test_socp()
    rng = torch.Generator().manual_seed(0)
    diag_op = lo.DiagonalOperator(torch.randn(10, generator=rng, dtype=torch.float64))
    b = diag_op @ torch.randn(10)
    lsmr(diag_op, b, Armat=diag_op.T, n=10)