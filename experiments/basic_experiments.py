import os
from functools import partial

import torch

from experiments.utils import GradDescTestHelper
from experiments.cvx_problem_generator import generate_LS_problem, generate_least_squares_eq

results_dir = os.path.join(os.path.dirname(__file__), "results")

if __name__ == "__main__":
    
    m = 20
    n = 10

    # generator = partial(generate_LS_problem, m, n, return_problem_only=True)
    
    # experiment1 = GradDescTestHelper(generator, dtype=torch.float64)
    # print("=== starting LS diffcp experiment ===")
    # experiment1_diffcp_result = experiment1.cp_grad_desc()
    # print("=== starting LS diffqcp experiment ===")
    # experiment1_diffqcp_result = experiment1.qcp_grad_desc()
    
    # save_path = os.path.join(results_dir, "diffcp_LS.png")
    # experiment1_diffcp_result.plot_obj_traj(save_path)

    # save_path = os.path.join(results_dir, "diffqcp_LS.png")
    # experiment1_diffqcp_result.plot_obj_traj(save_path)

    generator2 = partial(generate_least_squares_eq, m, n, return_all=False, return_problem_only=True)

    experiment2 = GradDescTestHelper(generator2, dtype=torch.float64)
    print("=== starting LS equality constrained diffcp experiment ===")
    experiment2_diffcp_result = experiment2.cp_grad_desc()
    print("=== starting LS equality constrained diffqcp experiment ===")
    experiment2_diffqcp_result = experiment2.qcp_grad_desc()
    
    save_path = os.path.join(results_dir, "diffcp_LS_eq.png")
    experiment2_diffcp_result.plot_obj_traj(save_path)

    save_path = os.path.join(results_dir, "diffqcp_LS_eq.png")
    experiment2_diffqcp_result.plot_obj_traj(save_path)

    # generator2 = partial(generate_least_squares_eq, m, n, return_problem_only=True)

    # experiment2 = GradDescTestHelper(generator, dtype=torch.float64)
    # print("=== starting LS equality constrained diffcp experiment ===")
    # experiment2_diffcp_result = experiment2.cp_grad_desc()
    # print("=== starting LS equality constrained diffqcp experiment ===")
    # experiment2_diffqcp_result = experiment2.qcp_grad_desc()
    
    # save_path = os.path.join(results_dir, "diffcp_LS_eq.png")
    # experiment2_diffcp_result.plot_obj_traj(save_path)

    # save_path = os.path.join(results_dir, "diffqcp_LS_eq.png")
    # experiment2_diffqcp_result.plot_obj_traj(save_path)