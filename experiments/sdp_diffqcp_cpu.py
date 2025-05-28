import time
from typing import Tuple
import os

import cvxpy as cvx
import numpy as np
import scipy.sparse as sparse
import torch
import clarabel

from diffqcp import QCP
from diffqcp.utils import to_tensor
from tests.utils import data_and_soln_from_cvxpy_problem, generate_problem_data_new, GradDescTestResult

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)

def randn_symm(n, random_array):
    A = random_array(n, n)
    return (A + A.T) / 2


def generate_problem(n, p, return_all=True):
    data = generate_problem_data_new(n=n, m=n, sparse_random_array=sparse.random_array,
                                     random_array=np.random.randn, P_psd=True)
    C = data[0].todense()
    
    data = generate_problem_data_new(n=n, m=n, sparse_random_array=sparse.random_array,
                                     random_array=np.random.randn, P_psd=True)
    # D = data[0].todense()

    As = [randn_symm(n, np.random.randn) for _ in range(p)]
    Bs = np.random.randn(p)

    X = cvx.Variable((n, n), PSD=True)
    # y = cvx.Variable(n)
    # objective = cvx.trace(C @ X) + cvx.quad_form(y, D, assume_PSD=True)
    objective = cvx.trace(C @ X)
    constraints = [cvx.trace(As[i] @ X) == Bs[i] for i in range(p)]
    prob = cvx.Problem(cvx.Minimize(objective), constraints)

    P, A, q, b, scs_cone_dict, soln, clarabel_cones = data_and_soln_from_cvxpy_problem(prob)

    x = np.array(soln.x)
    y = np.array(soln.z)
    s = np.array(soln.s)
    if return_all:
        return P, A, q, b, scs_cone_dict, clarabel_cones, x, y, s
    else:
        return x, y, s


def grad_desc_test(
    qcp: QCP,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor, 
    clarabel_cones,
    solver_settings,
    num_iter: int=100,
    step_size: float = 0.1,
    improvement_factor = 1e-2,
    verbose: bool = False,
) -> GradDescTestResult:
    curr_iter = 0
    optimal = False
    xk = x0
    yk = y0
    
    def f0(x, y) -> float:
        return 0.5 * torch.linalg.norm(x - x_target)**2 + 0.5 * torch.linalg.norm(y - y_target)

    if verbose:
        f0s = torch.zeros(num_iter, dtype=x_target.dtype, device=x_target.device)
    
    ds = torch.zeros(yk.shape[0], dtype=x_target.dtype, device=x_target.device)
    
    while curr_iter < num_iter:

        # will need to be smarter with data conversion during true experiment
        P_tch = qcp.data.materialize_P()
        P = sparse.csc_matrix(P_tch.cpu().numpy())
        A = sparse.csc_matrix(qcp.A.to_dense().cpu().numpy())
        q = qcp.q.cpu().numpy()
        b = qcp.b.cpu().numpy()
        solver = clarabel.DefaultSolver(P, q, A, b, clarabel_cones, solver_settings)
        solution = solver.solve()

        xk = to_tensor(solution.x, dtype=x_target.dtype, device=x_target.device)
        yk = to_tensor(solution.z, dtype=x_target.dtype, device=x_target.device)
        sk = to_tensor(solution.s, dtype=x_target.dtype, device=x_target.device)
        
        f0_k = f0(xk, yk)

        if verbose:
            f0s[curr_iter] = f0_k

        curr_iter += 1

        if curr_iter > 1 and (f0_k / f0s[0]) < improvement_factor:
            optimal = True
            break

        # TODO (quill) definitely need to fix `reduce_fp_flops` functionality

        qcp.update_solution(x = xk, y = yk, s = sk)

        d_theta = qcp.vjp(xk - x_target, yk - y_target, ds)

        # TODO (quill): need to allow updates that don't require forming the matrix.
        # shouldn't be too hard.
        Pk = P_tch - step_size * (d_theta[0].to_dense())
        Ak = qcp.A.to_dense() - step_size * (d_theta[1].to_dense())
        qk = qcp.q - step_size * d_theta[2]
        bk = qcp.b - step_size * d_theta[3]

        qcp.update_data(P=Pk, A=Ak, q=qk, b=bk)

    if verbose:
        f0_traj = f0s[0:curr_iter]
        del f0s
        return GradDescTestResult(passed=optimal, num_iterations=curr_iter, final_pt=xk, final_obj=f0_traj[-1].item(),
                                  verbose=True, obj_traj=f0_traj)



def main(n=3, p=3):
    
    x_star, y_star, s_star = generate_problem(n=n, p=p, return_all=False)
    x_star = to_tensor(x_star, dtype=torch.float64, device=None)
    y_star = to_tensor(y_star, dtype=torch.float64, device=None)

    P, A, q, b, scs_cone_dict, clarabel_cones, x0, y0, s0 = generate_problem(n=n, p=p, return_all=True)

    print("CLARABEL_CONES: ", clarabel_cones)
    
    mn_plus_m_plus_n = A.size + b.size + q.size
    n_plus_2n = q.size + 2 * b.size
    entries_in_derivative = mn_plus_m_plus_n * n_plus_2n
    print(f"""n={n}, p={p}, A.shape={A.shape}, nnz in A={A.nnz}, derivative={mn_plus_m_plus_n}x{n_plus_2n} ({entries_in_derivative} entries)""")

    qcp = QCP(
        P = P, A = A, q = q, b = b, x = x0, y = y0, s = s0, cone_dict = scs_cone_dict,
        P_is_upper = True, dtype=torch.float64
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = grad_desc_test(qcp, x_star, y_star, qcp.x, qcp.y, clarabel_cones, settings, verbose=True)
    save_path = os.path.join(results_dir, "plot.png")
    result.plot_obj_traj(save_path)
    

if __name__ == '__main__':
    np.random.seed(0)
    main()




