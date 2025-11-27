import numpy as np
import diffcp
import cvxpy as cp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import clarabel
from diffqcp import HostQCP, QCPStructureCPU, DeviceQCP, QCPStructureGPU
from jax.experimental.sparse import BCOO, BCSR
from tests.helpers import QCPProbData, scoo_to_bcoo, scsr_to_bcsr
import patdb


# def scs_data_from_cvxpy_problem(problem):
#     data = problem.get_problem_data(cp.CLARABEL)[0]
#     cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
#                                                                                   "dims"])
#     # cone_dims = get_problem_data(cvx.CLARABEL, ignore_dpp=True, solver_opts={'use_quad_obj': True})
#     return data["A"], data["b"], data["c"], cone_dims

def scs_data_from_cvxpy_problem(problem):
    data = problem.get_problem_data(cp.SCS)[0]
    cone_dims = cp.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(data[
                                                                                  "dims"])
    return data["A"], data["b"], data["c"], cone_dims

def compare_gradients(A, b, C, n):
    Z = cp.Variable((n, n), symmetric=True)
    constraints = [Z >> 0]
    vec_Z = cp.vec(Z,order="F")  
    constraints.append(A @ vec_Z == b)
    objective = cp.Minimize(cp.trace(C @ Z))
    prob = cp.Problem(objective, constraints)

    A_scs, b_scs, c_scs, cone_dims = scs_data_from_cvxpy_problem(prob)

    # Solve with diffcp
    x, y, s, derivative, adjoint_derivative = diffcp.solve_and_derivative(
        A_scs, b_scs, c_scs, cone_dims)
    
    print(f"diffcp x: {x}")
    print(f"diffcp y: {y}")
    print(f"diffcp s: {s}")
    
    # Solve with diffqcp
    prob_data = QCPProbData(prob)  
    P_jax = scoo_to_bcoo(prob_data.Pupper_coo).astype(jnp.float64)
    # P_jax = scsr_to_bcsr(prob_data.Pcsr)
    A_jax = BCOO.fromdense(A_scs.todense()).astype(jnp.float64)
    # A_jax = scsr_to_bcsr(prob_data.Acsr)
    # A_jax = BCSR.fromdense(A_scs.todense())
    q_jax = jnp.array(c_scs, dtype=jnp.float64)
    b_jax = jnp.array(b_scs, dtype=jnp.float64)
    
    
    problem_structure = QCPStructureCPU(P_jax, A_jax, prob_data.scs_cones)
    # problem_structure = QCPStructureGPU(P_jax, A_jax, cone_dims)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.presolve_enable = False
    solver = clarabel.DefaultSolver(prob_data.Pupper_csc,
                                  prob_data.q,
                                  prob_data.Acsc,
                                  prob_data.b,
                                  prob_data.clarabel_cones,
                                  settings)
    solution = solver.solve()
    
    # x_sol = jnp.array(solution.x, dtype=jnp.float64)
    # y_sol = jnp.array(solution.z, dtype=jnp.float64)
    # s_sol = jnp.array(solution.s, dtype=jnp.float64)
    x_sol = jnp.array(x, dtype=jnp.float64)
    y_sol = jnp.array(y, dtype=jnp.float64)
    s_sol = jnp.array(s, dtype=jnp.float64)

    # patdb.debug()

    print(f"clarabel x: {x_sol}")
    print(f"clarabel y: {y_sol}")
    print(f"clarabel s: {s_sol}")
    
    qcp = HostQCP(P_jax, A_jax, q_jax, b_jax, x_sol, y_sol, s_sol, problem_structure)
    # qcp = DeviceQCP(P_jax, A_jax, q_jax, b_jax, x_sol, y_sol, s_sol, problem_structure)
    
    dx_seed = np.ones(len(x_sol))
    # diffcp gradients
    dA_diffcp, db_diffcp, dc_diffcp = adjoint_derivative(
        np.ones(len(x_sol), dtype=np.float64), np.zeros(y.size), np.zeros(s.size), atol=1e-8, btol=1e-8)
    
    # diffqcp gradients
    dx_seed_jax = jnp.array(dx_seed, dtype=jnp.float64)
    dP_diffqcp, dA_diffqcp, dq_diffqcp, db_diffqcp = qcp.vjp(
        dx_seed_jax, jnp.zeros_like(y_sol), jnp.zeros_like(s_sol),)
        # solve_method="jax-lsmr")
    
    # Compare
    x_diff = np.linalg.norm(np.array(x_sol) - x) / np.linalg.norm(x)
    y_diff = np.linalg.norm(np.array(y_sol) - y) / np.linalg.norm(y)
    s_diff = np.linalg.norm(np.array(s_sol) - s) / np.linalg.norm(s)
    db_diff = np.linalg.norm(np.array(db_diffqcp) - db_diffcp) / np.linalg.norm(db_diffcp)
    dA_diff = np.linalg.norm(np.array(dA_diffqcp.todense()) - dA_diffcp.todense()) / np.linalg.norm(dA_diffcp.todense())
    
    
    print("=== ===")
    print(f"x diff: {x_diff}")
    print(f"y diff: {y_diff}")
    print(f"s diff: {s_diff}")
    print(f"db diff: {db_diff}")
    print(f"dA_dff: {dA_diff}")
    print(f" db Diffcp: {db_diffcp}")
    print(f" db Diffqcp: {db_diffqcp}")
    
    # print(f" dA Diffcp: {dA_diffcp.todense()}")
    # print(f" dA Diffqcp: {dA_diffqcp.todense()}")

    patdb.debug()
    
    return dc_diffcp, np.array(dq_diffqcp), db_diffcp, np.array(db_diffqcp)

if __name__ == "__main__":
    n = 3
    A = np.array([[ 0.6098,  0.9365, -0.5868, -0.8762,  2.5231, -0.0405, -0.4607,  1.0639,
             -0.2023],
            [ 0.9633, -0.9168, -0.2633,  0.8062,  0.4248,  0.1785,  0.6259,  0.7198,
              0.8306],
            [ 0.5087, -1.5266, -1.9011,  0.2482, -0.0872, -0.3489,  0.7842, -0.9984,
              1.1860],
            [ 0.4137,  0.1841,  1.4228, -0.0176, -0.4991,  0.9396,  1.0793,  0.2924,
              0.0775],
            [ 0.0865,  0.8037,  0.9151, -0.7554,  0.0665, -0.0050, -0.4473,  0.9084,
              0.1129]])
    b = np.array([ 73.3233,  32.3913, -18.0499,  13.1492,  13.9454])
    C = np.array([[ 4.7208,  4.2384, -1.1487],
            [ 4.2384,  6.3524, -0.3547],
            [-1.1487, -0.3547,  3.0890]])
    P = None
    
    compare_gradients(A, b, C, n)
