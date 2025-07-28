import numpy as np
from scipy import sparse
import cvxpy as cvx
import clarabel
import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array

CPU = jax.devices("cpu")[0]

def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


def get_cpu_int(a: Float[Array, " 1"]):
    return int(jnp.squeeze(jax.device_put(a,
                                          device=CPU)))


def data_and_soln_from_quad_qcp(problem: cvx.Problem):
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

    Pfull = Pfull.tocsr()
    P_upper = P_upper.tocsr()
    A = A.tocsr()

    return Pfull, P_upper, A, q, b, np.array(soln.x), np.array(soln.z), np.array(soln.s), scs_cone_dict, clarabel_cones