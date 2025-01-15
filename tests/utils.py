"""
Helper functions for testing diffqcp derivative and derivative
atom computations.
"""
from typing import Dict, Tuple, List, Union, Callable

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, spmatrix)
import cvxpy as cp
import clarabel
from clarabel import DefaultSolution
import linops as lo
import torch

import diffqcp.utils as qcp_utils
from diffqcp.linops import SymmetricOperator

def generate_problem_data(n: int,
                          m: int,
                          sparse_randomness,
                          randomness
) -> Tuple[csr_matrix, csc_matrix, np.ndarray, np.ndarray]:
    P = sparse_randomness(n, n, density=0.4)
    P = (P + P.T) / 2
    P = csr_matrix(P)
    A = sparse_randomness(m, n, density=0.4)
    A = csr_matrix(A)

    q = randomness(n)
    b = randomness(m)

    return P, A, q, b


def convert_prob_data_to_torch(P_upper: spmatrix,
                               A: spmatrix,
                               q: np.ndarray,
                               b: np.ndarray
) ->Tuple[lo.LinearOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = A.shape[1]
    P_tch = qcp_utils.to_sparse_csr_tensor(P_upper)
    P_op = SymmetricOperator(n, P_tch)
    A_tch = qcp_utils.to_sparse_csr_tensor(A)
    q_tch = qcp_utils.to_tensor(q)
    b_tch = qcp_utils.to_tensor(b)
    return P_op, A_tch, q_tch, b_tch


def generate_torch_problem_data(n: int,
                                m: int,
                                sparse_randomness,
                                randomness
) -> Tuple[csr_matrix, lo.LinearOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
    P, A, q, b = generate_problem_data(n, m, sparse_randomness, randomness)
    P_upper = sparse.triu(P).tocsr()
    P_op, A_tch, q_tch, b_tch = convert_prob_data_to_torch(P_upper, A, q, b)
    return P_upper, P_op, A_tch, q_tch, b_tch

# stolen from diffcp/utils.py
def get_random_like(A: csr_matrix | torch.Tensor,
                    randomness: Callable[[int],
                                         np.ndarray]
) -> csr_matrix:
    """Generate a random sparse matrix with the same sparsity
    pattern as A, using the function `randomness`.

    `randomness` is a function that returns a random vector
    with a prescribed length.
    """
    if isinstance(A, torch.Tensor):
        A = csr_matrix(A.to_dense())

    rows, cols = A.nonzero()
    values = randomness(A.nnz)
    return csr_matrix((values, (rows, cols)), shape=A.shape)


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
