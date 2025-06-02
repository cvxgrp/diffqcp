from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
from scipy.sparse import spmatrix
import torch
from torch import Tensor
import cvxpy as cvx
from diffcp import solve_and_derivative
from jaxtyping import Float

from diffqcp import QCP
from tests.utils import data_and_soln_from_cvxpy_problem


@dataclass
class GradDescTestResult:

    num_iterations : int
    obj_traj : torch.Tensor
    # final_pt: 
    final_obj: float
    obj_traj 

@dataclass
class GradDescTestHelper:

    problem: cvx.Problem
    # passing criteria? (multiple?)
    verbose: bool = False

    # post init attributes
    linear_cone_dict: dict[str, int | list[int]] = field(init=False)
    quad_cone_dict: dict[str, int | list[int]] = field(init=False)
    qcp: QCP = field(init=False)

    def __post_init__(self):
        pass

    def full_P_desc(
        step_size, 
    ):
        pass


