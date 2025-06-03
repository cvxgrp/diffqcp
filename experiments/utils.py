from dataclasses import dataclass, field
from typing import Optional, Callable, Union

import numpy as np
from scipy.sparse import spmatrix, sparray
import torch
from torch import Tensor
import cvxpy as cvx
from diffcp import solve_and_derivative
import clarabel
from jaxtyping import Float

from diffqcp import QCP
from tests.utils import data_and_soln_from_cvxpy_problem_quad, data_from_cvxpy_problem_linear

@dataclass
class DiffcpData:

    A : Float[Union[spmatrix, sparray], "m n"]
    c: Float[np.ndarray, "n"]
    b: Float[np.ndarray, "m"]
    cone_dict: dict[str, int | list[int]]


@dataclass
class GradDescTestResult:

    num_iterations : int
    obj_traj : torch.Tensor
    # final_pt: 
    final_obj: float
    obj_traj 

@dataclass
class GradDescTestHelper:

    problem_generator: Callable[[], cvx.Problem]
    # passing criteria? (multiple?)
    verbose: bool = False
    dtype: torch.dtype

    # post init attributes
    linear_cone_dict: dict[str, int | list[int]] = field(init=False)
    quad_cone_dict: dict[str, int | list[int]] = field(init=False)
    upper_P_qcp: QCP = field(init=False)
    full_P_qcp: QCP = field(init=False)
    quad_clarabel_cones = field(init=False)
    diffcp_cp: DiffcpData = field(init=False)
    target_x_qcp: Float[Tensor, "n"] = field(init=False)
    target_y_qcp: Float[Tensor, "m"] = field(init=False)
    target_s_qcp: Float[Tensor, "m"] = field(init=False)
    target_x_cp: Float[Tensor, "n"] = field(init=False)
    target_y_cp: Float[Tensor, "m"] = field(init=False)
    target_s_cp: Float[Tensor, "m"] = field(init=False)

    def __post_init__(self):
        target_problem = self.problem_generator()
        initial_problem = self.problem_generator()
        
        # grab target problem information for QCP and CP.
        qcp_data_and_soln = data_and_soln_from_cvxpy_problem_quad(target_problem)
        self.target_x_qcp = qcp_data_and_soln[5]
        self.target_y_qcp = qcp_data_and_soln[6]
        self.target_s_qcp = qcp_data_and_soln[7]
        cp_data = data_from_cvxpy_problem_linear(initial_problem)
        self.target_x_cp, self.target_y_cp, self.target_s_cp, _, _ = solve_and_derivative(
                                                                        cp_data[0], cp_data[1], cp_data[2], cp_data[3]
                                                                    )

        # Now grab starting data for the learning problem
        qcp_data_and_soln = data_and_soln_from_cvxpy_problem_quad(initial_problem)
        Pfull, P_upper, A = qcp_data_and_soln[0], qcp_data_and_soln[1], qcp_data_and_soln[2]
        q, b = qcp_data_and_soln[3], qcp_data_and_soln[4]
        x, y, s = qcp_data_and_soln[5], qcp_data_and_soln[6], qcp_data_and_soln[7]
        scs_quad_cones, clarabel_quad_cones = qcp_data_and_soln[8]
        self.quad_clarabel_cones = clarabel_quad_cones
        self.clarabel_solver_settings = clarabel.DefaultSettings()
        self.upper_P_qcp = QCP(P_upper, A, q, b, x, y, s, scs_quad_cones, P_is_upper=True, dtype=torch.float64)
        self.full_P_qcp = QCP(Pfull, A, q, b, x, y, s, scs_quad_cones, P_is_upper=False, dtype=torch.float64)
        cp_data = data_from_cvxpy_problem_linear(initial_problem)
        self.diffcp_cp = DiffcpData(cp_data[0], cp_data[1], cp_data[2], cp_data[3])

    def qcp_grad_desc(
        num_iter: int = 150,
        step_size: float = 0.15,
        improvement_factor: float = 1e-2
    ):
        pass


