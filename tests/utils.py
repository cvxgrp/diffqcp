"""
Helper functions for testing diffqcp derivative and derivative
atom computations.
"""
from typing import Dict, Tuple, List, Union, Callable

import numpy as np
from scipy.sparse import (csc_matrix, csr_matrix)
import cvxpy as cp
import clarabel
from clarabel import DefaultSolution

import diffqcp.utils as lo_utils

def generate_problem_data(n: int,
                          m: int,
                          sparse_randomness,
                          randomness
) -> Tuple[csc_matrix, csc_matrix, np.ndarray, np.ndarray]:
    P = sparse_randomness(n, n, density=0.4)
    P = (P + P.T) / 2
    P = csr_matrix(P)
    A = sparse_randomness(m, n, density=0.4)
    A = csr_matrix(A)

    q = randomness(n)
    b = randomness(m)

    return P, A, q, b


# def convert_problem_data(P: csc_matrix,
#                          A: csc_matrix,
#                          q: np.ndarray,
#                          b : np.ndarray
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     P = lo_u


# def convert_problem_data_true(P: csc_matrix,
#                               A: csc_matrix,
#                               q: np.ndarray,
#                               b : np.ndarray
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     pass

# stolen from diffcp/utils.py
def get_random_like(A: csc_matrix,
                    randomness: Callable[[int],
                                         np.ndarray]
) -> csc_matrix:
    """Generate a random sparse matrix with the same sparsity
    pattern as A, using the function `randomness`.

    `randomness` is a function that returns a random vector
    with a prescribed length.
    """
    rows, cols = A.nonzero()
    values = randomness(A.nnz)
    return csc_matrix((values, (rows, cols)), shape=A.shape)


def get_zeros_like(A: csc_matrix) -> csc_matrix:
    nonzeros = A.nonzero()
    data = np.zeros(A.size)
    return csc_matrix((data, nonzeros), shape=A.shape)


def data_and_soln_from_cvxpy_problem(problem: cp.Problem
) -> Tuple[csc_matrix,
           csc_matrix,
           np.ndarray,
           np.ndarray,
           Dict[str,
                Union[int, List[int]]
               ],
           DefaultSolution]:

    clarabel_probdata, _, _ = problem.get_problem_data(cp.CLARABEL)
    scs_probdata, _, _ = problem.get_problem_data(cp.SCS)

    P, q = scs_probdata['P'], scs_probdata['c']
    A, b = scs_probdata['A'], scs_probdata['b']

    clarabel_cones = cp.reductions.solvers.conic_solvers.clarabel_conif.dims_to_solver_cones(clarabel_probdata["dims"])
    scs_cone_dict = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(scs_probdata["dims"])

    solver_settings = clarabel.DefaultSettings()
    solver_settings.verbose = False
    solver = clarabel.DefaultSolver(P, q, A, b, clarabel_cones, solver_settings)
    soln = solver.solve()

    return P, A, q, b, scs_cone_dict, soln
