"""
Helper functions for testing diffqcp derivative and derivative
atom computations.
"""
import math
from typing import Dict, Tuple, List, Union, Callable, Optional
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, spmatrix, sparray, csr_array)
import cvxpy as cp
import clarabel
from clarabel import DefaultSolution
import linops as lo
import torch
import matplotlib.pyplot as plt
from jaxtyping import Float

import diffqcp.utils as qcp_utils
from diffqcp.linops import SymmetricOperator
import diffqcp.cones as cone_utils

def get_transpose(
    A: csr_matrix | csr_array | torch.Tensor, return_tensor: bool=False,
    dtype: torch.dtype=torch.float64, device:torch.device | None = None
) -> csr_array:
    """Get the transpose of a sparse array. 

    Mainly helpful when A is a torch.Tensor.
    """
    if isinstance(A, torch.Tensor):
        A = csr_array(A.to_dense().cpu().numpy())

    AT = A.T
    if return_tensor:
        AT = qcp_utils.to_sparse_csr_tensor(AT, dtype=dtype, device=device)
    
    return AT

def get_sparse_information(X: torch.Tensor):
    """
    Parameters
    ----------
    X : torch.Tensor
        2D tensor in sparse_csr format.

    Returns
    --------
    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    rows: torch.Tensor
    cols: torch.Tensor
    """
    assert X.layout == torch.sparse_csr
    
    crow_indices = X.crow_indices()
    col_indices = X.col_indices()

    X_coo = X.to_sparse_coo()
    indices = X_coo.indices()
    rows, cols = indices[0], indices[1]

    return crow_indices, col_indices, rows, cols


def form_full_symmetric(P_upper: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    P_upper: np.ndarray
        Should be a dense 2D array that is only the upper triangular part of `P`.
    """
    P_diag = np.diag(P_upper, k=0)
    P_diag = np.diag(P_diag) # create diagonal matrix
    P = P_upper + P_upper.T - P_diag
    return P

def generate_problem_data_new(
    n: int, m: int, sparse_random_array: Callable[[tuple[int, int], int], sparray],
    random_array: Callable[[int], np.ndarray], density: float=0.2, P_psd: bool=False
) -> tuple[csr_array, csr_array, csr_array, np.ndarray, np.ndarray]:
    r"""Randomly generates theta in Theta (as defined in diffqcp paper).

    Optionally make P PSD.

    Parameters
    ----------
    n : int
        The number of opt. vars. <=> number of rows and columns of P in a QCP.
    m : int
        The number of equality constraints <=> number of rows of A in a QCP.
    sparse_random_array : Callable
        A function which when provided a tuple of integer arguments
        returns a sparse (random) 2D array that belongs to
        R^(tuple[0] x tuple[1]).
    random_array : Callable
        A function which when provided an integer argument returns
        a (random) 1D array of that integer length.
    density : float, optional
        A float between 0 and 0.25 (exclusive) that determines how sparse
        the matrices returned by `sparse_randomness` are. Default is 0.2.
        Note that for the sparse A matrix, the density is doubled. (So the
        default sparsity for A is 0.4).
    

    Returns
    -------
    csr_array
        A symmetric matrix with n columns.
    csr_array
        A 2D array that is the upper triangular part of the previous 2D array.
    csr_array
        A 2D array in R^{m x n}
    np.ndarray
        The linear part of the objective function of a QCP. 1D with size n.
    np.ndarray
        The offsets in the linear equality constraints of a QCP. 1D with size m.
    """
    assert isinstance(density, float)
    assert density > 0
    assert density < 0.25

    if not P_psd:
        upper_P = sparse.triu(sparse_random_array(shape=(n, n), density=density, format = 'coo'))
        upper_P = upper_P.todense()
        P = form_full_symmetric(upper_P)
        upper_P = csr_array(upper_P)
        P = csr_array(P)
    else:
        P = sparse_random_array(shape=(n, n), density=density, format = 'coo')
        P = P.todense()
        P = P @ P.T
        P += 0.1 * np.eye(n) 
        upper_P = csr_array(sparse.triu(P))
        P = csr_array(P)

    A = sparse_random_array(shape=(m, n), density=density*2)
    A = csr_array(A)

    q = random_array(n)
    b = random_array(m)

    return P, upper_P, A, q, b


def convert_prob_data_to_torch_new(
    P: csr_array, P_upper: csr_array, A: csr_array, q: np.ndarray, b: np.ndarray,
    dtype: torch.dtype=torch.float64, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    P = qcp_utils.to_sparse_csr_tensor(P, dtype=dtype, device=device)
    P_upper = qcp_utils.to_sparse_csr_tensor(P_upper, dtype=dtype, device=device)
    A = qcp_utils.to_sparse_csr_tensor(A, dtype=dtype, device=device)
    q = qcp_utils.to_tensor(q, dtype=dtype, device=device)
    b = qcp_utils.to_tensor(b, dtype=dtype, device=device)
    return P, P_upper, A, q, b

def generate_torch_problem_data_new(
    n: int, m: int, sparse_random_array: Callable[[int, int], sparray], random_array: Callable[[int], np.ndarray],
    density: float=0.2, dtype: torch.dtype=torch.float64, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Randomly generates theta in Theta (as defined in diffqcp paper) as torch Tensors.

    Parameters
    ----------
    n : int
        The number of opt. vars. <=> number of rows and columns of P in a QCP.
    m : int
        The number of equality constraints <=> number of rows of A in a QCP.
    sparse_random_array : Callable[[int, int], sparray]
        A function which when provided two integer arguments
        returns a sparse (random) matrix that belongs to
        R^(arg1 x arg2).
    random_array : Callable[[int], np.ndarray]
        A function which when provided an integer argument returns
        a (random) vector of that integer length.
    density : float, optional
        A float between 0 and 0.25 (exclusive) that determines how sparse
        the matrices returned by `sparse_randomness` are. Default is 0.2.
        Note that for the sparse A matrix, the density is doubled. (So the
        default sparsity for A is 0.4).
    dtype : torch.dtype, optional
        The desired data type of the torch tensors.
        Since this function is used for testing purposes, the
        default is set to torch.float64.
    device : torch.device | None, optional
        Device for tensors, by default None.

    Returns
    -------
    torch.Tensor
        A 2D tensor in S^n in sparse_csr layout.
    torch.Tensor
        A 2D tensor that is the upper triangular part of the previous 2D array. In sparse_csr layout.
    torch.Tensor
        A 2D tensor in R^{n x m}. In sparse_csr layout.
    torch.Tensor
        The linear part of the objective function of a QCP. 1D.
    torch.Tensor
        The offsets in the linear equality constraints of a QCP. 1D.
    """
    P, P_upper, A, q, b = generate_problem_data_new(n, m, sparse_random_array, random_array, density)
    AT = A.T
    P, P_upper, A, q, b = convert_prob_data_to_torch_new(P, P_upper, A, q, b, dtype, device)
    AT = qcp_utils.to_sparse_csr_tensor(AT, dtype=dtype, device=device)
    return P, P_upper, A, AT, q, b


def data_and_soln_from_cvxpy_problem(problem: cp.Problem,
) -> Tuple[
        Float[csc_matrix, "n n"], Float[csc_matrix, "m n"], Float[np.ndarray, "n"], Float[csc_matrix, "m"],
           Dict[str, Union[int, List[int]]], DefaultSolution
    ]:

    clarabel_probdata, _, _ = problem.get_problem_data(cp.CLARABEL)
    scs_probdata, _, _ = problem.get_problem_data(cp.SCS)

    q = scs_probdata['c']

    # test SCS vs. Clarabel solution?
    
    try:
        P = scs_probdata['P']
        P = sparse.triu(P).tocsc()
    except:
        P = np.zeros((q.size, q.size))
        P = sparse.triu(P).tocsc()

    A, b = scs_probdata['A'], scs_probdata['b']

    clarabel_cones = cp.reductions.solvers.conic_solvers.clarabel_conif.dims_to_solver_cones(clarabel_probdata["dims"])
    scs_cone_dict = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(scs_probdata["dims"])

    solver_settings = clarabel.DefaultSettings()
    solver_settings.verbose = False
    solver = clarabel.DefaultSolver(P, q, A, b, clarabel_cones, solver_settings)
    soln = solver.solve()

    P = P.tocsr()

    return P, A, q, b, scs_cone_dict, soln, clarabel_cones


def generate_problem_data(n: int,
                          m: int,
                          sparse_randomness: Callable[[int, int], spmatrix],
                          randomness: Callable[[int], np.ndarray],
                          density: float=0.2
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray]:
    r"""Randomly generate the problem objects for a QCP.

    These objects just belong to the sets they are defined
    to belong to; they do not incorporate any additional
    structure that is induced by the cone constraints.

    Parameters
    ----------
    n : int
        The number of opt. vars. <=> number of rows and columns of P in a QCP.
    m : int
        The number of equality constraints <=> number of rows of A in a QCP.
    sparse_randomness : Callable
        A function which when provided two integer arguments
        returns a sparse (random) matrix that belongs to
        R^(arg1 x arg2).
    randomness : Callable
        A function which when provided an integer argument returns
        a (random) vector of that integer length.
    density : float, optional
        A float between 0 and 0.5 (inclusive) that determines how sparse
        the matrices returned by `sparse_randomness` are. Default is 0.4.

    Returns
    -------
    csr_matrix
        The PSD P in the objective function of a QCP.
    csr_matrix
        The matrix A defining the equality constraints of a QCP.
    np.ndarray
        The linear part of the objective function of a QCP.
    np.ndarray
        The offsets in the linear equality constraints of a QCP.
    """
    assert isinstance(density, float)
    assert density > 0
    assert density < 0.25

    upper_P = sparse.triu(sparse_randomness(n, n, density=density, format = 'coo'))
    upper_P = upper_P.todense()
    P = form_full_symmetric(upper_P)
    upper_P = csr_array(upper_P)
    P = csr_matrix(P)

    A = sparse_randomness(m, n, density=density*2)
    A = csr_matrix(A)

    q = randomness(n)
    b = randomness(m)

    return P, A, q, b


def convert_prob_data_to_torch(P_upper: spmatrix,
                               A: spmatrix,
                               q: np.ndarray | torch.Tensor,
                               b: np.ndarray | torch.Tensor,
                               dtype: torch.dtype = torch.float32,
                               device: torch.device | None = None
) ->Tuple[lo.LinearOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Convert the scipy/numpy QCP problem objects to torch tensors.

    Returns the QCP problem objects as they are used by the QCP atom functions.

    Parameters
    ----------
    P : scipy.sparse.spmatrix
        The upper triangular part of the symmetric P in the objective function
        of a QCP.
    A : scipy.sparse.spmatrix
        The `A` matrix defining a QCP's equality constraints.
    q : np.ndarray
        The linear part of the objective function of a QCP.
    b : np.ndarray
        The offsets in the linear equality constraints of a QCP.
    dtype : torch.dtype, optional
        The desired data type of the torch tensors.
        Since this function is used for testing purposes, the
        default is set to torch.float64.
    device : torch.device | None, optional
        Device for tensors, by default None.

    Returns
    -------
    lo.LinearOperator
        An operator representing the symmetric P in the objective function of a
        QCP.
    torch.Tensor
        A, as a tensor in sparse csr format.
    torch.Tensor
        q, as a torch tensor.
    torch.Tensor
        b, as a torch tensor.
    """
    n = A.shape[1]
    P_upper_tch = qcp_utils.to_sparse_csr_tensor(P_upper, dtype=dtype, device=device)
    P_op = SymmetricOperator(n, P_upper_tch)
    A_tch = qcp_utils.to_sparse_csr_tensor(A, dtype=dtype, device=device)
    q_tch = qcp_utils.to_tensor(q, dtype=dtype, device=device)
    b_tch = qcp_utils.to_tensor(b, dtype=dtype, device=device)
    return P_op, A_tch, q_tch, b_tch


def generate_torch_problem_data(n: int,
                                m: int,
                                sparse_randomness: Callable[[int, int], spmatrix],
                                randomness: Callable[[int], np.ndarray],
                                density: float=0.2,
                                dtype: torch.dtype = torch.float32,
                                device: torch.device | None = None
) -> Tuple[csr_matrix, lo.LinearOperator, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Randomly generate the problem objects for a QCP as torch tensors.

    These objects just belong to the sets they are defined
    to belong to; they do not incorporate any additional
    structure that is induced by the cone constraints.

    Parameters
    ----------
    n : int
        The number of opt. vars. <=> number of rows and columns of P in a QCP.
    m : int
        The number of equality constraints <=> number of rows of A in a QCP.
    sparse_randomness : Callable
        A function which when provided two integer arguments
        returns a sparse (random) matrix that belongs to
        R^(arg1 x arg2).
    randomness : Callable
        A function which when provided an integer argument returns
        a (random) vector of that integer length.
    density : float, optional
        A float between 0 and 0.5 (inclusive) that determines how sparse
        the matrices returned by `sparse_randomness` are. Default is 0.4.
    dtype : torch.dtype, optional
        The desired data type of the torch tensors.
        Since this function is used for testing purposes, the
        default is set to torch.float64.
    device : torch.device | None, optional
        Device for tensors, by default None.

    Returns
    -------
    csr_matrix
        The upper triangular part of the PSD P in the objective function of a QCP.
    lo.LinearOperator
        An operator representing the symmetric P in the objective function of a
        QCP.
    torch.Tensor
        A, as a tensor in sparse csr format.
    torch.Tensor
        The linear part of the objective function of a QCP.
    torch.Tensor
        The offsets in the linear equality constraints of a QCP.
    """
    P, A, q, b = generate_problem_data(n, m, sparse_randomness, randomness, density)
    P_upper = sparse.triu(P).tocsr()
    P_op, A_tch, q_tch, b_tch = convert_prob_data_to_torch(P_upper, A, q, b, dtype, device)
    return P_upper, P_op, A_tch, q_tch, b_tch

# stolen from diffcp/utils.py
def get_random_like(
    A: csr_matrix | torch.Tensor, randomness: Callable[[int], np.ndarray]
) -> csr_matrix:
    """Generate a random sparse matrix with the same sparsity
    pattern as A, using the function `randomness`.

    `randomness` is a function that returns a random vector
    with a prescribed length.
    """
    if isinstance(A, torch.Tensor):
        A = csr_matrix(A.to_dense().cpu().numpy())

    rows, cols = A.nonzero()
    values = randomness(A.nnz)
    return csr_matrix((values, (rows, cols)), shape=A.shape)


def get_zeros_like(A: csr_matrix | torch.Tensor
) -> csr_matrix:

    if isinstance(A, torch.Tensor):
        A = csr_matrix(A.to_dense().to_cpu().numpy())

    nonzeros = A.nonzero()
    data = np.zeros(A.size)
    return csr_matrix((data, nonzeros), shape=A.shape)


def torch_data_and_soln_from_cvxpy_problem(
    problem: cp.Problem,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None
) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[
            str,
            Union[int, List[int]]
        ],
        DefaultSolution
    ]:
    result = data_and_soln_from_cvxpy_problem(problem)
    P, A, q, b, cone_dict, soln = result
    # note that P is the upper triangular part of the symmetric P
    # it represents.
    P_tch = qcp_utils.to_sparse_csr_tensor(P, dtype=dtype, device=device)
    A_tch = qcp_utils.to_sparse_csr_tensor(A, dtype=dtype, device=device)
    q_tch = qcp_utils.to_tensor(q, dtype=dtype, device=device)
    b_tch = qcp_utils.to_tensor(b, dtype=dtype, device=device)

    return P_tch, A_tch, q_tch, b_tch, cone_dict, soln


def dottest(Op: lo.LinearOperator,
            rtol: float = 1e-6,
            atol: float = 1e-21,
            dtype: torch.dtype = torch.float64
) -> bool:
    r"""Dot test.

    Generate random vectors u and v and perform the dot-test to verify
    the validity of forward and adjoint operators.

    Parameters
    ----------
    Op : lo.LinearOperator
        Linear operator to test.
    rtol : float, optional
        Relative dottest tolerance. Default 1e-6.
    atol : float, optional
        Absolute dottest tolerance. Default 1e-21.
    dtype: float, optional
        The datatype of u and v. Default is torch.float64.

    Returns
    -------
    bool
        Whether the provided operator is a linear operator.

    Notes
    -----
    The test is performed on whatever device the LinearOperator requires its vectors to be on.

    See https://pylops.readthedocs.io/en/stable/adding.html#addingoperator
    for the mathematical basis of this test, and
    https://github.com/PyLops/pylops/blob/6c92032519eb778e5f801abc88e8b0cbdafff8aa/pylops/utils/dottest.py#L10
    for the code which this implementation is based on.
    """
    device = Op.device
    rng = torch.Generator(device).manual_seed(0)

    for _ in range(10):

        n, m = Op.shape[0], Op.shape[1]

        u = torch.randn(n, generator=rng, dtype=dtype, device=device)
        v = torch.randn(m, generator=rng, dtype=dtype, device=device)

        is_linop = math.isclose( ((Op @ u) @ v).item(), (u @ (Op.T @ v)).item(), rel_tol=rtol, abs_tol=atol)

        if not is_linop:
            return False

    return True


@dataclass
class GradDescTestResult:
    passed : bool
    num_iterations : int
    final_pt : torch.Tensor
    final_obj: float
    verbose : Optional[bool] = False
    obj_traj : Optional[torch.Tensor] = None

    def plot_obj_traj(self, savepath: str | None = None):
        if self.obj_traj is None:
            raise ValueError("obj_traj is None. Cannot plot.")

        # Move obj_traj to CPU if it's on a device
        obj_traj_cpu = self.obj_traj.cpu().numpy() if self.obj_traj.is_cuda else self.obj_traj.numpy()

        plt.figure(figsize=(8, 6))
        plt.plot(range(len(obj_traj_cpu)), obj_traj_cpu, label="Objective Trajectory")
        plt.xlabel("k")
        # plt.ylabel("$f_0(p^{k}) = 0.5 \\| z(p) - z^{\\star} \\|^2$")
        plt.ylabel("Objective function")
        plt.legend()
        # plt.show()
        plt.savefig(savepath)


def grad_desc_test(
    f_and_Df: Callable[[torch.Tensor],
                        tuple[torch.Tensor, torch.Tensor | lo.LinearOperator]],
    p_target: torch.Tensor,
    p0: torch.Tensor,
    num_iter: int = 100,
    tol: float = 1e-6,
    step_size: float = 0.1,
    verbose: bool = False
) -> GradDescTestResult:
    """Gradient descent test specifically for projecting onto a cone.

    Given two distinct points p_target and p0, this function computes
            
            z^star = projection(p_target) = argmin_z { ||z - p_target||_2 | z in cone } ,

    and then attempts to solve the problem
    
        (1) minimize f0(p) = (1/2)||z(p) - z^star||_2^2,

    where z(p) = argmin_z { ||z - p||_2 | z in cone }. Solving (1) can be thought of
    as "learning" an equivalent projection problem to the one solved to find z^star.

    To attempt to solve (1) we use gradient descent. The gradient of f0 w.r.t. p is

        grad_f0(p) = Dz(p)^T (z(p) - z^star),
    
    where Dz(p)^T is the transpose of the Jacobian of the projection of p onto a cone. 
    
    Gradient descent steps:
    1. Descent direction. delta_p = - grad_f0(p)
    2. Update: p = p + step_size*delta_p.

    Parameters
    ----------
    f_and_Df : Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | lo.LinearOperator]]
        Given a point, returns the projection of that point onto a cone and the Jacobian of the
        projection onto the cone at that point.
    p_target : torch.Tensor
        The point used to generate z^star.
    p0: torch.Tensor
        A point != p_target to serve as the initial point when running
        gradient descent on (1).
    num_iter : int, optional
        The number of iterations to run gradient descent on (1) before terminating.
        Defaults to 100.
    tol: float, optional
        Gradient descent will terminate when f0(p) <= tol.
        Defaults to 1e-6.
    step_size: float, optional
        The step size in the gradient descent algorith.
        Must be greater than 0. Defaults to 0.1
    verbose: bool, optional
        Whether to record and return the objective function values
        associated with the gradient descent iterates.
        Defaults to False.

    Returns
    --------
    GradDescTestResult
    """
    z_star, _ = f_and_Df(p_target)
    curr_iter = 0    
    optimal = False
    f0 = lambda z_p : 0.5 * torch.linalg.norm(z_p - z_star)**2
    pk = p0

    if verbose:
        f0s = torch.zeros(num_iter, dtype=p_target.dtype, device=p_target.device)

    while curr_iter < num_iter:

        z_pk, Dz_pk = f_and_Df(pk)
        f0_pk = f0(z_pk)

        if verbose:
            f0s[curr_iter] = f0_pk

        curr_iter += 1
        
        if f0_pk < tol:
            optimal = True
            break

        delta_p = - Dz_pk.T @ (z_pk - z_star)
        pk += step_size * delta_p
    
    if verbose:
        f0_traj = f0s[0:curr_iter]
        del f0s
        return GradDescTestResult(passed=optimal, num_iterations=curr_iter,
                                     final_pt=pk, final_obj=f0_traj[-1].item(),
                                     verbose=True, obj_traj=f0_traj)
    
    return GradDescTestResult(passed=optimal, num_iterations=curr_iter,
                                 final_pt=pk, final_obj=f0_pk)


def random_qcp(m, n, cone_dict, random_sparse_array, random_array):
    """Returns the problem data of a random cone program.
    
    NOTE: The dual solution can be non-unique. Recall uniqueness is
        required to differentiate through a QCP.
    
    """
    cone_list = cone_utils.parse_cone_dict(cone_dict)

    print("cone list: ", cone_list)

    P, P_upper, A, q, b = generate_problem_data_new(n, m, random_sparse_array, random_array, P_psd=True)
    del q
    del b
    
    z = random_array(m)
    z_tch = qcp_utils.to_tensor(z)
    s_star_tch, _ = cone_utils.proj_and_dproj(z_tch, cone_list, dual=False)
    s_star = s_star_tch.cpu().numpy()
    y_star = s_star - z
    x_star = random_array(n)
    b = A @ x_star + s_star
    q = - P @ x_star - A.T @ y_star
    return P, P_upper, A, q, b, (x_star, y_star, s_star)
