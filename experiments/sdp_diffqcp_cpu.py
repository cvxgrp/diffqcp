import os

import numpy as np
import scipy.sparse as sparse
import torch
from torch import Tensor
import clarabel
from jaxtyping import Float

from diffqcp import QCP
from diffqcp.utils import to_tensor
from tests.utils import GradDescTestResult
from cvx_problem_generator import generate_portfolio_problem, generate_least_squares_eq, generate_LS_problem, generate_sdp

results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(results_dir, exist_ok=True)


def grad_desc_test(
    qcp: QCP,
    x_target: torch.Tensor,
    y_target: torch.Tensor,
    s_target: torch.Tensor,
    x0: torch.Tensor,
    y0: torch.Tensor,
    s0: torch.Tensor,
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
    sk = s0
    
    def f0(x, y, s) -> Float[Tensor, ""]:
        return (0.5 * torch.linalg.norm(x - x_target)**2 + 0.5 * torch.linalg.norm(y - y_target)**2
                + 0.5 * torch.linalg.norm(s - s_target))

    if verbose:
        f0s = torch.zeros(num_iter, dtype=x_target.dtype, device=x_target.device)
    
    # ds = torch.zeros(yk.shape[0], dtype=x_target.dtype, device=x_target.device)
    
    while curr_iter < num_iter:

        # will need to be smarter with data conversion during true experiment
        P_tch_upper = qcp.data.materialize_P_upper()
        P_tch_upper_dense = P_tch_upper.to_dense()
        if curr_iter <= 2:
            print(P_tch_upper_dense)
            # save_path = os.path.join(results_dir, "SDP_p_upper")
            # np.save(save_path, P_tch_upper_dense.cpu().numpy())
            P_tch_upper_dense_np = P_tch_upper_dense.cpu().numpy()
            print("UPPER TRIANGULAR? ", np.allclose(P_tch_upper_dense_np, np.triu(P_tch_upper_dense.cpu().numpy())))
        P = sparse.csc_matrix(P_tch_upper_dense.cpu().numpy())
        A = sparse.csc_matrix(qcp.A.to_dense().cpu().numpy())
        q = qcp.q.cpu().numpy()
        b = qcp.b.cpu().numpy()
        solver = clarabel.DefaultSolver(P, q, A, b, clarabel_cones, solver_settings)
        solution = solver.solve()

        xk = to_tensor(solution.x, dtype=x_target.dtype, device=x_target.device)
        yk = to_tensor(solution.z, dtype=x_target.dtype, device=x_target.device)
        sk = to_tensor(solution.s, dtype=x_target.dtype, device=x_target.device)
        
        f0_k = f0(xk, yk, sk)

        if verbose:
            f0s[curr_iter] = f0_k

        curr_iter += 1

        if curr_iter > 1 and ((f0_k / f0s[0]) < improvement_factor or f0_k < 1e-3):
            optimal = True
            break

        qcp.update_solution(x = xk, y = yk, s = sk)

        d_theta = qcp.vjp(xk - x_target, yk - y_target, sk - s_target)

        Pk = P_tch_upper + (-step_size * (d_theta[0])) # allow to update P just by up
        Ak = qcp.A + (-step_size * d_theta[1])
        qk = qcp.q - step_size * d_theta[2]
        bk = qcp.b - step_size * d_theta[3]

        qcp.update_data(P=Pk, A=Ak, q=qk, b=bk)

    if verbose:
        f0_traj = f0s[0:curr_iter]
        del f0s
        return GradDescTestResult(passed=optimal, num_iterations=curr_iter, final_pt=xk, final_obj=f0_traj[-1].item(),
                                  verbose=True, obj_traj=f0_traj)



def sdp_test(n=3, p=3):
    
    x_star, y_star, s_star = generate_sdp(n=n, p=p, return_all=False)
    x_star = to_tensor(x_star, dtype=torch.float64, device=None)
    y_star = to_tensor(y_star, dtype=torch.float64, device=None)

    P, A, q, b, scs_cone_dict, clarabel_cones, x0, y0, s0 = generate_sdp(n=n, p=p, return_all=True)

    print("CLARABEL_CONES: ", clarabel_cones)
    
    mn_plus_m_plus_n = A.size + b.size + q.size
    n_plus_2n = q.size + 2 * b.size
    entries_in_derivative = mn_plus_m_plus_n * n_plus_2n
    P_dense = P.todense()

    print("P val original: ", P_dense)

    print("ORIGINAL IS UPPER TRIANGULAR:", np.allclose(P_dense, np.triu(P_dense)))
    
    print(f"""n={n}, p={p}, A.shape={A.shape}, nnz in A={A.nnz}, derivative={mn_plus_m_plus_n}x{n_plus_2n} ({entries_in_derivative} entries)""")

    qcp = QCP(
        P = P, A = A, q = q, b = b, x = x0, y = y0, s = s0, cone_dict = scs_cone_dict,
        P_is_upper = True, dtype=torch.float64
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = grad_desc_test(qcp, x_star, y_star, qcp.x, qcp.y, clarabel_cones, settings, verbose=True, num_iter=250)
    save_path = os.path.join(results_dir, "SDP_plot.png")
    result.plot_obj_traj(save_path)
    print(f"The initial loss was {result.obj_traj[0]} and the final loss was {result.final_obj}")
    if result.passed:
        print(f"The gradient descent test PASSED under an improvement factor requirement of {1 / 1e-2}")
    else:
        print(f"The gradient descent test FAILED under an improvement factor requirement of {1 / 1e-2}")
    

def ls_test(m=20, n=15):
    # TODO check to make sure the cones are the same?
    x_star, y_star, s_star = generate_LS_problem(m=m, n=n, return_all=False)
    x_star = to_tensor(x_star, dtype=torch.float64, device=None)
    y_star = to_tensor(y_star, dtype=torch.float64, device=None)

    P, A, q, b, scs_cone_dict, clarabel_cones, x0, y0, s0 = generate_LS_problem(m=m, n=n, return_all=True)

    P_dense = P.todense()

    print("P val original: ", P_dense)

    print("ORIGINAL IS UPPER TRIANGULAR:", np.allclose(P_dense, np.triu(P_dense)))
    
    print("CLARABEL_CONES: ", clarabel_cones)
    
    mn_plus_m_plus_n = A.size + b.size + q.size
    n_plus_2n = q.size + 2 * b.size
    entries_in_derivative = mn_plus_m_plus_n * n_plus_2n
    print("P val original: ", P.todense())
    print(f"""n={n}, m={m}, A.shape={A.shape}, nnz in A={A.nnz}, derivative={mn_plus_m_plus_n}x{n_plus_2n} ({entries_in_derivative} entries)""")

    qcp = QCP(
        P = P, A = A, q = q, b = b, x = x0, y = y0, s = s0, cone_dict = scs_cone_dict,
        P_is_upper = True, dtype=torch.float32
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = grad_desc_test(qcp, x_star, y_star, qcp.x, qcp.y, clarabel_cones, settings, verbose=True)
    save_path = os.path.join(results_dir, "LS_plot.png")
    result.plot_obj_traj(save_path)
    print(f"The initial loss was {result.obj_traj[0]} and the final loss was {result.final_obj}")
    if result.passed:
        print(f"The gradient descent test PASSED under an improvement factor requirement of {1 / 1e-2}")
    else:
        print(f"The gradient descent test FAILED under an improvement factor requirement of {1 / 1e-2}")

def ls_eq_test(m=20, n=10):
    # TODO check to make sure the cones are the same?
    x_star, y_star, s_star = generate_least_squares_eq(m=m, n=n, return_all=False)
    x_star = to_tensor(x_star, dtype=torch.float64, device=None)
    y_star = to_tensor(y_star, dtype=torch.float64, device=None)

    P, A, q, b, scs_cone_dict, clarabel_cones, x0, y0, s0 = generate_least_squares_eq(m=m, n=n, return_all=True)

    P_dense = P.todense()

    print("P val original: ", P_dense)

    print("ORIGINAL IS UPPER TRIANGULAR:", np.allclose(P_dense, np.triu(P_dense)))
    
    print("CLARABEL_CONES: ", clarabel_cones)

    print("SCS Cones: ", scs_cone_dict)
    
    mn_plus_m_plus_n = A.size + b.size + q.size
    n_plus_2n = q.size + 2 * b.size
    entries_in_derivative = mn_plus_m_plus_n * n_plus_2n
    print("P val original: ", P.todense())
    print(f"""n={n}, m={m}, A.shape={A.shape}, nnz in A={A.nnz}, derivative={mn_plus_m_plus_n}x{n_plus_2n} ({entries_in_derivative} entries)""")

    qcp = QCP(
        P = P, A = A, q = q, b = b, x = x0, y = y0, s = s0, cone_dict = scs_cone_dict,
        P_is_upper = True, dtype=torch.float64
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = grad_desc_test(qcp, x_star, y_star, s_star, qcp.x, qcp.y, qcp.s, clarabel_cones, settings, verbose=True, num_iter=300, step_size=.30)
    save_path = os.path.join(results_dir, "LS_eq_plot.png")
    result.plot_obj_traj(save_path)
    print(f"The initial loss was {result.obj_traj[0]} and the final loss was {result.final_obj}")
    if result.passed:
        print(f"The gradient descent test PASSED under an improvement factor requirement of {1 / 1e-2}")
    else:
        print(f"The gradient descent test FAILED under an improvement factor requirement of {1 / 1e-2}")

def portfolio_test(n=10):
    x_star, y_star, s_star = generate_portfolio_problem(n=n, return_all=False)
    x_star = to_tensor(x_star, dtype=torch.float64, device=None)
    y_star = to_tensor(y_star, dtype=torch.float64, device=None)

    P, A, q, b, scs_cone_dict, clarabel_cones, x0, y0, s0 = generate_portfolio_problem(n=n, return_all=True)

    P_dense = P.todense()

    print("P val original: ", P_dense)
    print("ORIGINAL IS UPPER TRIANGULAR:", np.allclose(P_dense, np.triu(P_dense)))
    print("CLARABEL_CONES: ", clarabel_cones)
    print("SCS Cones: ", scs_cone_dict)

    qcp = QCP(
        P = P, A = A, q = q, b = b, x = x0, y = y0, s = s0, cone_dict = scs_cone_dict,
        P_is_upper = True, dtype=torch.float64
    )

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    result = grad_desc_test(qcp, x_star, y_star, qcp.x, qcp.y, clarabel_cones, settings, verbose=True, step_size=0.25)
    save_path = os.path.join(results_dir, "portfolio_plot.png")
    result.plot_obj_traj(save_path)
    print(f"The initial loss was {result.obj_traj[0]} and the final loss was {result.final_obj}")
    if result.passed:
        print(f"The gradient descent test PASSED under an improvement factor requirement of {1 / 1e-2}")
    else:
        print(f"The gradient descent test FAILED under an improvement factor requirement of {1 / 1e-2}")

if __name__ == '__main__':
    np.random.seed(0)
    # sdp_test(n=10, p = 5)
    # sdp_test()
    # ls_test()
    # ls_test(m=100, n=40)
    ls_eq_test()
    # portfolio_test()




