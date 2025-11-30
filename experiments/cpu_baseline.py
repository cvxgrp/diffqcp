"""Clarabel+diffcp learning loop for paper experiment."""

import time
from dataclasses import dataclass, field
import os

import numpy as np
from numpy import ndarray
import scipy.linalg as la
from scipy.sparse import (spmatrix, sparray, csr_matrix,
                          csr_array, coo_matrix, coo_array,
                          csc_matrix, csc_array)
import cvxpy as cvx
from diffcp import solve_and_derivative
from jaxtyping import Float

import experiments.cvx_problem_generator as prob_generator
import patdb
import matplotlib.pyplot as plt

type SP = spmatrix | sparray
type SCSR = csr_matrix | csr_array
type SCSC = csc_matrix | csc_array
type SCOO = coo_matrix | coo_array

@dataclass
class CPProbData:
    """(linear) Cone Program (CP) problem data."""

    problem: cvx.Problem

    Acsc: SCSC = field(init=False)
    c: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    scs_cones: dict[int, int | list[int] | list[float]] = field(init=False)
    n: int = field(init=False)
    m: int = field(init=False)

    def __post_init__(self):
        
        probdata, _, _ = self.problem.get_problem_data(cvx.CLARABEL, ignore_dpp=True, solver_opts={'use_quad_obj': False})
        self.A = probdata["A"].tocsc()
        self.c, self.b = probdata["c"], probdata["b"]
        self.scs_cones = cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(probdata["dims"])
        self.n = np.size(self.c)
        self.m = np.size(self.b)


def f0(
    target_x: Float[ndarray, " n"],
    target_y: Float[ndarray, " m"],
    target_s: Float[ndarray, " m"],
    x: Float[ndarray, " n"],
    y: Float[ndarray, " m"],
    s: Float[ndarray, " m"]
) -> float:
    return (0.5 * la.norm(x - target_x)**2 + 0.5 * la.norm(y - target_y)**2
            + 0.5 * la.norm(s - target_s)**2)


def grad_desc(
    prob_data: CPProbData,
    target_x,
    target_y,
    target_s,
    num_iter: int=500,
    step_size: float = 1e-5
):
    curr_iter = 0
    losses = []

    while curr_iter < num_iter:

        xk, yk, sk, _, DT = solve_and_derivative(prob_data.A,
                                                 prob_data.b,
                                                 prob_data.c,
                                                 prob_data.scs_cones,
                                                 solve_method="CLARABEL")
        losses.append(f0(target_x, target_y, target_s, xk, yk, sk))

        dA, db, dc = DT(xk - target_x, yk - target_y, sk - target_s)

        prob_data.A += -step_size * dA
        prob_data.c += -step_size * dc
        prob_data.b += -step_size * db

        curr_iter += 1
    
    return losses


if __name__ == "__main__":
    
    np.random.seed(28)

    # SMALL
    m = 20
    n = 10
    # MEDIUM-ish
    # m = 200
    # n = 100
    # LARGE-ish
    # m = 2_000
    # n = 1_000
    start_time = time.perf_counter()
    # target_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    target_problem = prob_generator.generate_group_lasso_logistic(m=m, n=m)
    prob_data = CPProbData(target_problem)
    end_time = time.perf_counter()
    print("Time to generate the target problem and"
          + f" canonicalize it: {end_time - start_time} seconds")
    
    start_time = time.perf_counter()
    target_x, target_y, target_s, _, DT = solve_and_derivative(prob_data.A,
                                                               prob_data.b,
                                                               prob_data.c,
                                                               prob_data.scs_cones,
                                                               solve_method="CLARABEL")
    end_time = time.perf_counter()
    print("Time to solve target problem + precompute some derivative info:"
          + f" {end_time - start_time} seconds.")

    fake_x = 1e-3 * np.arange(np.size(prob_data.c), dtype=prob_data.c.dtype)
    fake_y = 1e-3 * np.arange(np.size(prob_data.b), dtype=prob_data.b.dtype)
    fake_s = 1e-3 * np.arange(np.size(prob_data.b), dtype=prob_data.b.dtype)

    start_time = time.perf_counter()
    _, _, _ = DT(fake_x - target_x,
                 fake_y - target_y,
                 fake_s - target_s)
    end_time = time.perf_counter()
    print("Time to do the main diffcp computations:"
          + f" {end_time - start_time}")
    
    start_time = time.perf_counter()
    # initial_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    initial_problem = prob_generator.generate_group_lasso_logistic(m=m, n=m)
    prob_data = CPProbData(initial_problem)
    end_time = time.perf_counter()
    print("Time to generate the initial (starting point) problem and"
          + f" canonicalize it: {end_time - start_time} seconds")
    print(f"Canonicalized n is: {prob_data.n}")
    print(f"Canonicalized m is: {prob_data.m}")

    # num_iter = 5
    # num_iter=100
    num_iter= 100
    print("starting loop:")
    start_time = time.perf_counter()
    losses = grad_desc(prob_data, target_x, target_y, target_s, num_iter)
    end_time = time.perf_counter()
    print("Learning loop time: ", end_time - start_time)
    print(f"Avg. iteration (solve + VJP) time: {(end_time - start_time) / num_iter}")
    print("starting loss: ", losses[0])
    print("final loss: ", losses[-1])

    plt.figure(figsize=(8, 6))
    plt.plot(range(num_iter), losses, label="Objective Trajectory")
    plt.xlabel("num. iterations")
    plt.ylabel("Objective function")
    plt.legend()
    plt.title(label="diffcp")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if n > 999:
        output_path = os.path.join(results_dir, "diffcp_logistic_lasso_large.svg")
    else:
        output_path = os.path.join(results_dir, "diffcp_logistic_lasso_small.svg")

    plt.savefig(output_path, format="svg")
    plt.close()

    

