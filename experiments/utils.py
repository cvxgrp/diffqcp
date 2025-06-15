from dataclasses import dataclass, field
from typing import Optional, Callable, Union
import time
import os
import json

import numpy as np
from scipy.sparse import spmatrix, sparray, save_npz
import torch
from torch import Tensor
import cvxpy as cvx
import matplotlib.pyplot as plt
from diffcp import solve_and_derivative
import clarabel
from jaxtyping import Float

from diffqcp import QCP
from diffqcp.utils import to_tensor
from tests.utils import data_and_soln_from_cvxpy_problem_quad, data_from_cvxpy_problem_linear

def _convert(obj):
    """Helper function for saving cone dictionaries"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

@dataclass
class DiffcpData:

    A : Float[Union[spmatrix, sparray], "m n"]
    c: Float[np.ndarray, "n"]
    b: Float[np.ndarray, "m"]
    cone_dict: dict[str, int | list[int]]


@dataclass
class GradDescTestResult:

    passed : bool
    num_iterations : int
    obj_traj : np.ndarray
    for_qcp: bool
    learning_time: float
    lsqr_residuals: np.ndarray | None = None
    improvement_factor: float = field(init=False)

    def __post_init__(self):
        self.improvement_factor = self.obj_traj[0] / self.obj_traj[-1]

    def plot_obj_traj(self, savepath: str) -> None:

        if self.obj_traj is None:
            raise ValueError("obj_traj is None. Cannot plot.")

        plt.figure(figsize=(8, 6))
        plt.plot(range(self.num_iterations), self.obj_traj, label="Objective Trajectory")
        # plt.plot(range(self.num_iterations), np.log(self.obj_traj), label="Objective Trajectory")
        # if self.lsqr_residuals is not None:
            # plt.plot(range(self.num_iterations), np.log(self.lsqr_residuals), label="LSQR residuals")
        plt.xlabel("num. iterations")
        # plt.ylabel("$f_0(p^{k}) = 0.5 \\| z(p) - z^{\\star} \\|^2$")
        plt.ylabel("Objective function")
        plt.legend()
        if self.for_qcp:
            plt.title(label="diffqcp")
        else:
            plt.title(label="diffcp")
        plt.savefig(savepath)
        plt.close()

    def save_result(self, savepath: str, experiment_name: str, experiment_count: int=0, verbose: bool=False) -> None:
        log_path = os.path.join(savepath, f"logs/experiment_{experiment_count}_log.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_content = []
        log_content.append(
            f"{"diffqcp" if self.for_qcp else "diffcp"} {experiment_name} learning experiment:\n"
            f"  Iterations: {self.num_iterations}\n"
            f"  Learning time: {self.learning_time}\n"
            f"  Improvement factor: {self.improvement_factor}\n"
            f"  Final loss: {self.obj_traj[-1]}\n"
        )

        if verbose:
            print(log_content)
            print("=== ===")
        
        with open(log_path, "a") as f:
            f.write("\n".join(log_content))
            f.write("\n")
            f.write("=== ===\n")

        if self.lsqr_residuals is not None:
            lsqr_dir = os.path.join(savepath, f"data/run_{experiment_count}/qcp")
            os.makedirs(lsqr_dir, exist_ok=True)
            np.save(os.path.join(lsqr_dir, "lsqr_residuals.npy"), self.lsqr_residuals)


@dataclass
class GradDescTestHelper:

    problem_generator: Callable[[], cvx.Problem]
    # passing criteria? (multiple?)
    verbose: bool = False
    dtype: torch.dtype = torch.float64
    save_dir: Optional[str] = None
    
    # post init attributes
    linear_cone_dict: dict[str, int | list[int]] = field(init=False)
    quad_cone_dict: dict[str, int | list[int]] = field(init=False)
    upper_P_qcp: QCP = field(init=False)
    full_P_qcp: QCP = field(init=False)
    quad_clarabel_cones: list = field(init=False)
    diffcp_cp: DiffcpData = field(init=False)
    target_x_qcp: Float[Tensor, "n_qcp"] = field(init=False)
    target_y_qcp: Float[Tensor, "m_qcp"] = field(init=False)
    target_s_qcp: Float[Tensor, "m_qcp"] = field(init=False)
    target_x_cp: Float[np.ndarray, "n_cp"] = field(init=False)
    target_y_cp: Float[np.ndarray, "m_cp"] = field(init=False)
    target_s_cp: Float[np.ndarray, "m_cp"] = field(init=False)
    reset_counter: int = field(default=0, init=False)

    def __post_init__(self):

        self._reset_problems()
    
    def _reset_problems(self):
        self.reset_counter += 1
        target_problem = self.problem_generator()
        initial_problem = self.problem_generator()
        self.P_full_qcp_has_descended: bool = False
        self.P_upper_qcp_has_descended: bool = False
        self.cp_has_descended: bool = False

        # grab target problem data for QCP and CP.
        qcp_data_and_soln = data_and_soln_from_cvxpy_problem_quad(target_problem)
        self.target_x_qcp = to_tensor(qcp_data_and_soln[5], dtype=self.dtype)
        self.target_y_qcp = to_tensor(qcp_data_and_soln[6], dtype=self.dtype)
        self.target_s_qcp = to_tensor(qcp_data_and_soln[7], dtype=self.dtype)
        cp_data = data_from_cvxpy_problem_linear(target_problem)
        target_x_cp, target_y_cp, target_s_cp, _, _ = solve_and_derivative(cp_data[0], cp_data[2], cp_data[1], cp_data[3], solve_method='CLARABEL')
        self.target_x_cp = target_x_cp
        self.target_y_cp = target_y_cp
        self.target_s_cp = target_s_cp

                # Save targets if save_dir is provided
        if self.save_dir is not None:
            os.makedirs(f"{self.save_dir}/data/run_{self.reset_counter}/qcp", exist_ok=True)
            os.makedirs(f"{self.save_dir}/data/run_{self.reset_counter}/cp", exist_ok=True)
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/x_target.npy"), self.target_x_qcp.cpu().numpy())
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/y_target.npy"), self.target_y_qcp.cpu().numpy())
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/s_target.npy"), self.target_s_qcp.cpu().numpy())
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/x_target.npy"), self.target_x_cp)
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/y_target.npy"), self.target_y_cp)
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/s_target.npy"), self.target_s_cp)
        
        # Now grab starting data for the learning problem
        qcp_data_and_soln = data_and_soln_from_cvxpy_problem_quad(initial_problem)
        Pfull, P_upper, A = qcp_data_and_soln[0], qcp_data_and_soln[1], qcp_data_and_soln[2]
        assert P_upper.nnz > 0
        assert P_upper.count_nonzero() > 0
        q, b = qcp_data_and_soln[3], qcp_data_and_soln[4]
        x, y, s = qcp_data_and_soln[5], qcp_data_and_soln[6], qcp_data_and_soln[7]
        scs_quad_cones, clarabel_quad_cones = qcp_data_and_soln[8], qcp_data_and_soln[9]
        if self.save_dir is not None:
            # Save as .npy for arrays, .npz for sparse matrices
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/q_initial.npy"), q)
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/b_initial.npy"), b)
            # Save sparse matrices as .npz as well
            if isinstance(P_upper, spmatrix) or isinstance(P_upper, sparray):
                save_npz(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/P_upper_initial.npz"), P_upper)
            if isinstance(A, spmatrix) or isinstance(A, sparray):
                save_npz(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/A_initial.npz"), A)
            with open(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/qcp/scs_cones.json"), "w") as f:
                json.dump(scs_quad_cones, f, default=_convert)
        self.quad_clarabel_cones = clarabel_quad_cones
        self.clarabel_solver_settings = clarabel.DefaultSettings()
        self.clarabel_solver_settings.verbose = False
        self.upper_P_qcp = QCP(P=P_upper, A=A, q=q, b=b, x=x, y=y, s=s, cone_dict=scs_quad_cones, P_is_upper=True, dtype=torch.float64)
        self.full_P_qcp = QCP(P=Pfull, A=A, q=q, b=b, x=x, y=y, s=s, cone_dict=scs_quad_cones, P_is_upper=False, dtype=torch.float64)
        cp_data = data_from_cvxpy_problem_linear(initial_problem)
        if self.save_dir is not None:
            # Save as .npy for arrays, .npz for sparse matrices
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/c_initial.npy"), cp_data[1])
            np.save(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/b_initial.npy"), cp_data[2])
            # Save sparse matrices as .npz as well
            if isinstance(A, spmatrix) or isinstance(A, sparray):
                save_npz(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/A_initial.npz"), cp_data[0])
            with open(os.path.join(self.save_dir, f"data/run_{self.reset_counter}/cp/scs_cones.json"), "w") as f:
                json.dump(cp_data[3], f, default=_convert)
        self.diffcp_cp = DiffcpData(A=cp_data[0], c=cp_data[1], b=cp_data[2], cone_dict=cp_data[3])
    
    def qcp_grad_desc(
        self,
        num_iter: int = 150,
        step_size: float = 0.15,
        improvement_factor: float = 1e-1, # 10x improvement
        fixed_tol: float = 1e-4,
        use_full_P: bool = True
    ) -> GradDescTestResult:

        if use_full_P:
            if self.P_full_qcp_has_descended:
                self._reset_problems()
            self.P_full_qcp_has_descended = True
            qcp = self.full_P_qcp

        else:
            if self.P_upper_qcp_has_descended:
                self._reset_problems()
            self.P_upper_qcp_has_descended = True
            qcp = self.upper_P_qcp
        
        curr_iter = 0
        optimal = False
        f0s = torch.zeros(num_iter+1, dtype=self.dtype)
        step_size = torch.tensor(step_size, dtype=self.dtype)
        lsqr_residuals = torch.zeros(num_iter, dtype=self.dtype)

        def f0(x: Float[Tensor, "n_qcp"], y: Float[Tensor, "m_qcp"], s: Float[Tensor, "m_qcp"]) -> Float[Tensor, ""]:
            return (0.5 * torch.linalg.norm(x - self.target_x_qcp)**2 + 0.5 * torch.linalg.norm(y - self.target_y_qcp)**2
                    + 0.5 * torch.linalg.norm(s - self.target_s_qcp)**2)
        
        # f0s[0] = f0(qcp.x, qcp.y, qcp.s)

        start_time = time.perf_counter()
        
        while curr_iter < num_iter:
            P_upper = qcp.get_Pcsc_cpu_upper()
            A = qcp.get_Acsc_cpu()
            q = qcp.q.cpu().numpy()
            b = qcp.b.cpu().numpy()
            
            solver = clarabel.DefaultSolver(P_upper, q, A, b, self.quad_clarabel_cones, self.clarabel_solver_settings)
            solution = solver.solve()

            xk = to_tensor(solution.x, dtype=self.dtype)
            yk = to_tensor(solution.z, dtype=self.dtype)
            sk = to_tensor(solution.s, dtype=self.dtype)

            f0k = f0(xk, yk, sk)

            f0s[curr_iter] = f0k
            curr_iter += 1

            if curr_iter > 1 and ((f0k / f0s[0]) < improvement_factor or f0k < fixed_tol):
                optimal = True
                break

            # add deriv and adjoint consistency checks
            # add feasibility checks
            #   what do you do once infeasible? quit?

            d_theta = qcp.vjp(xk - self.target_x_qcp, yk - self.target_y_qcp, sk - self.target_s_qcp)
            lsqr_residuals[curr_iter - 1] = qcp.vjp_lsqr_residual

            dP = -step_size * d_theta[0]
            dA = -step_size * d_theta[1]
            dq = -step_size * d_theta[2]
            db = -step_size * d_theta[3]
            qcp.perturb_data(dP, dA, dq, db)
            qcp.update_solution(xk, yk, sk)

        end_time = time.perf_counter()
        
        f0_traj = f0s[0:curr_iter]
        residuals = lsqr_residuals[0:curr_iter]
        del f0s
        del lsqr_residuals
        return GradDescTestResult(
                passed=optimal, num_iterations=curr_iter, obj_traj=f0_traj.cpu().numpy(),
                learning_time=(end_time - start_time), lsqr_residuals=residuals.cpu().numpy(), for_qcp=True
            )


    def cp_grad_desc(
        self,
        num_iter: int = 150,
        step_size: float = 0.15,
        improvement_factor: float = 1e-1, #10x improvement
        fixed_tol: float = 1e-3
    ) -> GradDescTestResult:
        
        if self.cp_has_descended:
            self._reset_problems()
            self.cp_has_descended = True

        curr_iter = 0
        optimal = False
        f0s = np.zeros(num_iter)
        
        def f0(x: Float[np.ndarray, "n_cp"], y: Float[np.ndarray, "m_cp"], s: Float[np.ndarray, "m_cp"]) -> float:
            return (0.5 * np.linalg.norm(x - self.target_x_cp)**2 + 0.5 * np.linalg.norm(y - self.target_y_cp)**2
                    + 0.5 * np.linalg.norm(s - self.target_s_cp)**2)

        start_time = time.perf_counter()
        
        while curr_iter < num_iter:
            
            A = self.diffcp_cp.A
            c = self.diffcp_cp.c
            b = self.diffcp_cp.b

            xk, yk, sk, _, DT = solve_and_derivative(A, b, c, self.diffcp_cp.cone_dict, solve_method='CLARABEL')

            f0k = f0(xk, yk, sk)

            f0s[curr_iter] = f0k
            curr_iter += 1

            if curr_iter > 1 and ((f0k / f0s[0]) < improvement_factor or f0k < fixed_tol):
                optimal = True
                break

            dA, db, dc = DT(xk - self.target_x_cp, yk - self.target_y_cp, sk - self.target_s_cp)

            self.diffcp_cp.A += -step_size * dA
            self.diffcp_cp.c += -step_size * dc
            self.diffcp_cp.b += -step_size * db

        end_time = time.perf_counter()
        
        f0_traj = f0s[0:curr_iter]
        del f0s
        return GradDescTestResult(
                passed=optimal, num_iterations=curr_iter, obj_traj=f0_traj, for_qcp=False, learning_time=(end_time - start_time)
            )

        

