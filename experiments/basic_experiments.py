import os
from datetime import datetime
from functools import partial
import time

import torch

from experiments.utils import GradDescTestHelper
from experiments.cvx_problem_generator import (generate_LS_problem, generate_least_squares_eq,
                                               generate_group_lasso_logistic, generate_portfolio_problem,
                                               generate_sdp)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(os.path.dirname(__file__), f"results/results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)  # <-- Add this line
log_path = os.path.join(results_dir, f"experiment_log.txt")
log_content = []

if __name__ == "__main__":
    
    m = 20
    n = 10
    num_iter = 300
    step_size = 1e-5

    # === ===

    generator = partial(generate_LS_problem, m, n)
    
    experiment = GradDescTestHelper(generator, dtype=torch.float64)
    print("=== starting diffcp LS experiment ===")
    experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    print("=== starting diffqcp LS experiment ===")
    experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    # save results
    experiment_diffcp_result.save_result(log_path, experiment_name="Least Squares", verbose=True)
    experiment_diffqcp_result.save_result(log_path, experiment_name="Least Squares", verbose=True)
    diffcp_save_path = os.path.join(results_dir, "diffcp_LS.png")
    diffqcp_save_path = os.path.join(results_dir, "diffqcp_LS.png")
    experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # === ===

    generator = partial(generate_least_squares_eq, m, n)
    
    experiment = GradDescTestHelper(generator, dtype=torch.float64)
    print("=== starting diffcp LS eq experiment ===")
    experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    print("=== starting diffqcp LS eq experiment ===")
    experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    # save results
    experiment_diffcp_result.save_result(log_path, experiment_name="Least Squares eq.", verbose=True)
    experiment_diffqcp_result.save_result(log_path, experiment_name="Least Squares eq.", verbose=True)
    diffcp_save_path = os.path.join(results_dir, "diffcp_LS_eq.png")
    diffqcp_save_path = os.path.join(results_dir, "diffqcp_LS_eq.png")
    experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)


    # === ===

    generator = partial(generate_group_lasso_logistic, n=n, m=m)
    
    experiment = GradDescTestHelper(generator, dtype=torch.float64)
    print("=== starting diffcp logistic lasso experiment ===")
    experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    print("=== starting diffqcp logistic lasso experiment ===")
    experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    # save results
    experiment_diffcp_result.save_result(log_path, experiment_name="Logistic group lasso.", verbose=True)
    experiment_diffqcp_result.save_result(log_path, experiment_name="Logistic group lasso.", verbose=True)
    diffcp_save_path = os.path.join(results_dir, "diffcp_grp_lasso_logistic.png")
    diffqcp_save_path = os.path.join(results_dir, "diffqcp_grp_lasso_logistic.png")
    experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)
    

    # === ===

    generator = partial(generate_portfolio_problem, n=n)
    
    experiment = GradDescTestHelper(generator, dtype=torch.float64)
    print("=== starting diffcp portfolio experiment ===")
    experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    print("=== starting diffqcp portfolio experiment ===")
    experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    # save results
    experiment_diffcp_result.save_result(log_path, experiment_name="Portfolio", verbose=True)
    experiment_diffqcp_result.save_result(log_path, experiment_name="Portfolio", verbose=True)
    diffcp_save_path = os.path.join(results_dir, "diffcp_portfolio.png")
    diffqcp_save_path = os.path.join(results_dir, "diffqcp_portfolio.png")
    experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # === ===
    
    n = 20
    p = 10

    generator = partial(generate_sdp, n=n, p=p)
    
    experiment = GradDescTestHelper(generator, dtype=torch.float64)
    print("=== starting diffcp SDP experiment ===")
    experiment_diffcp_result = experiment.cp_grad_desc(step_size=1e-5, num_iter=num_iter)
    print("=== starting diffqcp SDP experiment ===")
    experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=1e-5, num_iter=num_iter)
    # save results
    experiment_diffcp_result.save_result(log_path, experiment_name="SDP", verbose=True)
    experiment_diffqcp_result.save_result(log_path, experiment_name="SDP", verbose=True)
    diffcp_save_path = os.path.join(results_dir, "diffcp_sdp.png")
    diffqcp_save_path = os.path.join(results_dir, "diffqcp_sdp.png")
    experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)