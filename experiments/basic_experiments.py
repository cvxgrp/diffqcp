import os
from functools import partial

import torch

from experiments.utils import GradDescTestHelper
from experiments.cvx_problem_generator import generate_LS_problem

results_dir = os.path.join(os.path.dirname(__file__), "results")

if __name__ == "__main__":
    
    m = 20
    n = 15

    generator = partial(generate_LS_problem, m, n, return_problem_only=True)
    
    experiment1 = GradDescTestHelper(generator, dtype=torch.float64)
    experiment1_diffcp_result = experiment1.cp_grad_desc()
    save_path = os.path.join(results_dir, "diffcp_LS.png")
    experiment1_diffcp_result.plot_obj_traj(save_path)