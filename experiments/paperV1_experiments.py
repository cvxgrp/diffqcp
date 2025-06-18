import os
from datetime import datetime
from functools import partial

import numpy as np
import torch

from experiments.utils import GradDescTestHelper
from experiments.cvx_problem_generator import (generate_robust_mvdr_beamformer, generate_kalman_smoother,
                                               generate_group_lasso, generate_portfolio_problem,
                                               generate_sdp, generate_least_squares_eq)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
results_dir = os.path.join(os.path.dirname(__file__), f"results/results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

if __name__ == "__main__":

    np.random.seed(13)
    
    m = 20
    n = 10
    num_iter = 500
    step_size = 1e-2
    num_loop_trials = 10

    # smaller parameters for testing
    # m = 20
    # n = 10
    # num_iter = 20
    # step_size = 1e-5
    # num_loop_trials = 3

    # TODO(quill): add try catch blocks?

    # === ===

    # generator = partial(generate_group_lasso, n=m, m=m)
    # exp_results_dir = os.path.join(results_dir, "group_lasso")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp group lasso experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     print(f"=== starting diffqcp group lasso experiment {i} / {num_loop_trials} ===")
    #     experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="group lasso", experiment_count=i, verbose=True)
    #     experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="group lasso", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    #     experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # # === ===

    # generator = partial(generate_portfolio_problem, n=20)
    # exp_results_dir = os.path.join(results_dir, "portfolio")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp portfolio experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     print(f"=== starting diffqcp portfolio experiment {i} / {num_loop_trials} ===")
    #     experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="portfolio", experiment_count=i, verbose=True)
    #     experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="portfolio", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    #     experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # # === ===
    
    # n = 20
    # p = 10

    # generator = partial(generate_sdp, n=n, p=p)
    # exp_results_dir = os.path.join(results_dir, "sdp")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):    
    #     print(f"=== starting diffcp quad SDP experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     print(f"=== starting diffqcp quad SDP experiment {i} / {num_loop_trials} ===")
    #     experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="SDP", experiment_count=i, verbose=True)
    #     experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="SDP", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    #     experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # # === ===

    # n=20

    # generator = partial(generate_robust_mvdr_beamformer, n=n)
    # exp_results_dir = os.path.join(results_dir, "robust_mvdr")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1):
    #     print(f"=== starting diffcp robust mvdr experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
    #     print(f"=== starting diffqcp robust mvdr experiment {i} / {num_loop_trials} ===")
    #     experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="robust mvdr", experiment_count=i, verbose=True)
    #     experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="robust mvdr", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    #     experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # === ===

    # w = np.random.randn(2,100)
    # v = np.random.randn(2,100)
    # generator = partial(generate_kalman_smoother, random_inputs=w, random_noise=v, n=100)
    # exp_results_dir = os.path.join(results_dir, "kalman")
    # os.makedirs(exp_results_dir, exist_ok=True)
    # exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    # os.makedirs(exp_results_dir_plots, exist_ok=True)
    # experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    # for i in range(1, num_loop_trials+1): 
    #     print(f"=== starting diffcp kalman experiment {i} / {num_loop_trials} ===")
    #     experiment_diffcp_result = experiment.cp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     print(f"=== starting diffqcp kalman experiment {i} / {num_loop_trials} ===")
    #     experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=step_size, num_iter=num_iter)
    #     # save results
    #     experiment_diffcp_result.save_result(exp_results_dir, experiment_name="kalman", experiment_count=i, verbose=True)
    #     experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="kalman", experiment_count=i, verbose=True)
    #     diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
    #     diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
    #     experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
    #     experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

    # === ===

    m=20
    n=10

    generator = partial(generate_least_squares_eq, m=m, n=n)
    exp_results_dir = os.path.join(results_dir, "LS_eq_small_LR")
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    os.makedirs(exp_results_dir_plots, exist_ok=True)
    experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    for i in range(1, num_loop_trials+1):
        print(f"=== starting diffcp LS eq small LR experiment {i} / {num_loop_trials} ===")
        experiment_diffcp_result = experiment.cp_grad_desc(step_size=1e-5, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
        print(f"=== starting diffqcp LS eq small LR experiment {i} / {num_loop_trials} ===")
        experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=1e-5, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
        # save results
        experiment_diffcp_result.save_result(exp_results_dir, experiment_name="LS eq small LR", experiment_count=i, verbose=True)
        experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="LS eq small LR", experiment_count=i, verbose=True)
        diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
        diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
        experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
        experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)

        # === ===

    m=20
    n=10

    generator = partial(generate_least_squares_eq, m=m, n=n)
    exp_results_dir = os.path.join(results_dir, "LS_eq_large_LR")
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_results_dir_plots = os.path.join(exp_results_dir, "plots")
    os.makedirs(exp_results_dir_plots, exist_ok=True)
    experiment = GradDescTestHelper(generator, dtype=torch.float64, save_dir=exp_results_dir)
    for i in range(1, num_loop_trials+1):
        print(f"=== starting diffcp LS eq large LR experiment {i} / {num_loop_trials} ===")
        experiment_diffcp_result = experiment.cp_grad_desc(step_size=1e-2, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
        print(f"=== starting diffqcp LS eq large LR experiment {i} / {num_loop_trials} ===")
        experiment_diffqcp_result = experiment.qcp_grad_desc(step_size=1e-2, num_iter=num_iter, improvement_factor=1e-5, fixed_tol=1e-9)
        # save results
        experiment_diffcp_result.save_result(exp_results_dir, experiment_name="LS eq large LR", experiment_count=i, verbose=True)
        experiment_diffqcp_result.save_result(exp_results_dir, experiment_name="LS eq large LR", experiment_count=i, verbose=True)
        diffcp_save_path = os.path.join(exp_results_dir_plots, f"diffcp_experiment_{i}.png")
        diffqcp_save_path = os.path.join(exp_results_dir_plots, f"diffqcp_experiment_{i}.png")
        experiment_diffcp_result.plot_obj_traj(diffcp_save_path)
        experiment_diffqcp_result.plot_obj_traj(diffqcp_save_path)
