from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import (spmatrix, sparray, csr_matrix,
                          csr_array, coo_matrix, coo_array,
                          csc_matrix, csc_array)
import cvxpy as cvx
import clarabel
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from jax.experimental.sparse import BCSR, BCOO

CPU = jax.devices("cpu")[0]

type SP = spmatrix | sparray
type SCSR = csr_matrix | csr_array
type SCSC = csc_matrix | csc_array
type SCOO = coo_matrix | coo_array

def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def get_cpu_int(a: Float[Array, " 1"]):
    return int(jnp.squeeze(jax.device_put(a, device=CPU)))


def scoo_to_bcoo(coo_mat: SCOO) -> BCOO:
    """just assume `coo_mat` is in correct form."""
    row_indices = coo_mat.row
    col_indices = coo_mat.col
    indices = list(zip(row_indices, col_indices))
    return BCOO((coo_mat.data, indices), shape=coo_mat.shape)


def scsr_to_bcsr(csr_mat: SCSR) -> BCSR:
    return BCSR((csr_mat.data, csr_mat.indices, csr_mat.indptr),
                shape=csr_mat.shape)
    

def quad_data_and_soln_from_qcp(problem: cvx.Problem, return_csr: bool = True):
    """
    note that we could grab the qcp problem data in a linear canonical form.
    """
    clarabel_probdata, _, _ = problem.get_problem_data(cvx.CLARABEL)

    Pfull = clarabel_probdata['P']
    P_upper = sparse.triu(Pfull).tocsc()
    A = clarabel_probdata['A']
    q = clarabel_probdata['c']
    b = clarabel_probdata['b']

    clarabel_cones = cvx.reductions.solvers.conic_solvers.clarabel_conif.dims_to_solver_cones(clarabel_probdata["dims"])
    scs_cone_dict = cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(clarabel_probdata["dims"])

    solver_settings = clarabel.DefaultSettings()
    solver_settings.verbose = False
    solver = clarabel.DefaultSolver(P_upper, q, A, b, clarabel_cones, solver_settings)
    soln = solver.solve()

    if return_csr:
        Pfull = Pfull.tocsr()
        P_upper = P_upper.tocsr()
        A = A.tocsr()
    else:
        Pfull = Pfull.tocoo()
        P_upper = P_upper.tocoo()
        A = A.tocoo()

    return Pfull, P_upper, A, q, b, np.array(soln.x), np.array(soln.z), np.array(soln.s), scs_cone_dict, clarabel_cones

quad_data_and_soln_from_qcp_coo = lambda prob: quad_data_and_soln_from_qcp(prob, return_csr=False)
quad_data_and_soln_from_qcp_csr = lambda prob: quad_data_and_soln_from_qcp(prob, return_csr=True)

@dataclass
class QCPProbData:

    problem: cvx.Problem

    Pcsc: SCSC = field(init=False)
    Pcsr: SCSR = field(init=False)
    Pcoo: SCOO = field(init=False)
    
    Pupper_csc: SCSC = field(init=False)
    Pupper_csr: SCSR = field(init=False)
    Pupper_coo: SCOO = field(init=False)
    
    Acsc: SCSC = field(init=False)
    Acsr: SCSR = field(init=False)
    Acoo: SCOO = field(init=False)
    
    q: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)
    
    n: np.ndarray = field(init=False)
    m: np.ndarray = field(init=False)
    
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)

    scs_cones: dict[int, int | list[int] | list[float]] = field(init=False)
    clarabel_cones: list = field(init=False)

    def __post_init__(self):
        """
        
        **Note**
        - `get_problem_data` seems to be returning CSR matrices/arrays.
        - for `P`, it returns the whole array, not just the upper triangular part.
            To check this consider the following example:

            ```python
            import cvxpy as cvx
            import numpy as np

            # Define problem data
            Q = np.array([[4.0, 1.0, 0.5],
                        [1.0, 3.0, 1.5],
                        [0.5, 1.5, 2.0]])  # Non-diagonal and symmetric

            c = np.array([-1.0, 0.0, 1.0])
            A = np.array([[1.0, 2.0, 3.0]])
            b = np.array([1.0])

            # Variable
            x = cvx.Variable(3)

            # Objective (standard QP form)
            objective = cvx.Minimize(0.5 * cvx.quad_form(x, Q) + c @ x)

            # Constraints
            constraints = [A @ x <= b]

            # Problem
            prob = cvx.Problem(objective, constraints)
            ```
        """
        clarabel_probdata, _, _ = self.problem.get_problem_data(cvx.CLARABEL, solver_opts={'use_quad_obj': True})

        self.Pcsr = clarabel_probdata["P"].tocsr()
        self.Pcsc = self.Pcsr.tocsc()
        self.Pcoo = self.Pcsr.tocoo()
        self.Pupper_csr = sparse.triu(self.Pcsr).tocsr()
        self.Pupper_csc = self.Pupper_csr.tocsc()
        self.Pupper_coo = self.Pupper_csr.tocoo()

        self.Acsr = clarabel_probdata["A"].tocsr()
        self.Acsc = self.Acsr.tocsc()
        self.Acoo = self.Acsr.tocoo()

        self.q = clarabel_probdata["c"]
        self.n = np.size(self.q)
        self.b = clarabel_probdata["b"]
        self.m = np.size(self.b)

        # NOTE(quill): that both reductions use the clarabel problem data
        #   (So we are getting clarabel canonical form cones, but in scs dict on second line down.)
        self.clarabel_cones = cvx.reductions.solvers.conic_solvers.clarabel_conif.dims_to_solver_cones(clarabel_probdata["dims"])
        self.scs_cones = cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(clarabel_probdata["dims"])

        solver_settings = clarabel.DefaultSettings()
        solver_settings.verbose = False
        solver = clarabel.DefaultSolver(self.Pupper_csc,
                                        self.q,
                                        self.Acsc,
                                        self.b,
                                        self.clarabel_cones,
                                        solver_settings)
        soln = solver.solve()
        self.x = np.array(soln.x)
        self.y = np.array(soln.z)
        self.s = np.array(soln.s)

def get_zeros_like_coo(A: SCOO):
    return coo_array((np.zeros(A.size), A.nonzero()), shape=A.shape)

def get_zeros_like_csr(A: SCSR):
    return csr_array((np.zeros(np.size(A.data)), A.indices, A.indptr), shape=A.shape, dtype=A.dtype)