import time
import os
from dataclasses import dataclass
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax.numpy.linalg as la
from jaxtyping import Float, Array
import equinox as eqx
from jax.experimental.sparse import BCOO
import clarabel
from scipy.sparse import (spmatrix, sparray,
                          csc_matrix, csc_array)
import patdb
import matplotlib.pyplot as plt

from diffqcp import HostQCP, QCPStructureCPU
import experiments.cvx_problem_generator as prob_generator
from tests.helpers import QCPProbData, scoo_to_bcoo

type SP = spmatrix | sparray
type SCSC = csc_matrix | csc_array

@dataclass
class SolverData:

    Pupper_csc: SCSC
    A: SCSC
    q: np.ndarray
    b: np.ndarray

def compute_loss(target_x, target_y, target_s, x, y, s):
    return (0.5 * la.norm(x - target_x)**2 + 0.5 * la.norm(y - target_y)**2
            + 0.5 * la.norm(s - target_s)**2)

@eqx.filter_jit
@eqx.debug.assert_max_traces(max_traces=1)
def make_step(
    qcp,
    target_x,
    target_y,
    target_s,
    step_size
) -> tuple[Float[Array, ""], Float[BCOO, "n n"], Float[BCOO, "m n"],
           Float[Array, " n"], Float[Array, " m"]]:
    loss = compute_loss(target_x, target_y, target_s, qcp.x, qcp.y, qcp.s)
    dP, dA, dq, db = qcp.vjp(qcp.x - target_x,
                             qcp.y - target_y,
                             qcp.s - target_s)
    # NOTE(quill): `__rmul__` is not implemented for BCSR; update only data vectors
    dP.data *= -step_size
    dA.data *= -step_size
    dq *= -step_size
    db *= -step_size

    return (loss, dP, dA, dq, db)

def grad_desc(
    Pk: Float[BCOO, "n n"],
    Ak: Float[BCOO, "m n"],
    qk: Float[Array, " n"],
    bk: Float[Array, " m"],
    target_x: Float[Array, " n"],
    target_y: Float[Array, " m"],
    target_s: Float[Array, " m"],
    qcp_problem_structure: QCPStructureCPU,
    data: QCPProbData,
    Pcoo_csc_perm: Float[np.ndarray, "..."],
    Acoo_csc_perm: Float[np.ndarray, "..."],
    clarabel_solver,
    num_iter: int = 100,
    step_size = 1e-5
):
    curr_iter = 0
    losses = []

    while curr_iter < num_iter:

        solution = clarabel_solver.solve()
        
        xk = jnp.array(solution.x)
        yk = jnp.array(solution.z)
        sk = jnp.array(solution.s)

        qcp = HostQCP(Pk, Ak, qk, bk, xk, yk, sk, qcp_problem_structure)
        loss, *dtheta_steps = make_step(qcp, target_x, target_y, target_s, step_size)
        losses.append(loss)

        dP_step, dA_step, dq_step, db_step = dtheta_steps
        dP_step_np = np.asarray(dP_step.data)
        dA_step_np = np.asarray(dA_step.data)
        Pk.data += dP_step.data
        data.Pupper_csc.data += dP_step_np[Pcoo_csc_perm]
        Ak.data += dA_step.data
        data.Acsc.data += dA_step_np[Acoo_csc_perm]
        qk += dq_step
        data.q += np.asarray(dq_step)
        bk += db_step
        data.b += np.asarray(db_step)
        
        solver.update(P=data.Pupper_csc, q=data.q, A=data.Acsc, b=data.b)

        curr_iter += 1

    return losses

if __name__ == "__main__":
    np.random.seed(28)
    
    m = 20
    n = 10
    target_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    prob_data_cpu = QCPProbData(target_problem)

    Pupper_coo_to_csc_order = np.lexsort((prob_data_cpu.Pupper_coo.row,
                                          prob_data_cpu.Pupper_coo.col))
    A_coo_to_csc_order = np.lexsort((prob_data_cpu.Acoo.row,
                                     prob_data_cpu.Acoo.col))
    
    cones = prob_data_cpu.clarabel_cones
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.presolve_enable = False

    solver = clarabel.DefaultSolver(prob_data_cpu.Pupper_csc,
                                    prob_data_cpu.q,
                                    prob_data_cpu.Acsc,
                                    prob_data_cpu.b,
                                    cones,
                                    settings)

    start_solve = time.perf_counter()
    solution = solver.solve()
    end_solve = time.perf_counter()
    print(f"Clarabel solve took: {end_solve - start_solve} seconds")
    
    target_x = jnp.array(solution.x)
    target_y = jnp.array(solution.z)
    target_s = jnp.array(solution.s)

    target_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    prob_data_cpu = QCPProbData(target_problem)

    P = scoo_to_bcoo(prob_data_cpu.Pupper_coo)
    A = scoo_to_bcoo(prob_data_cpu.Acoo)
    q = prob_data_cpu.q
    b = prob_data_cpu.b
    scs_cones = prob_data_cpu.scs_cones
    problem_structure = QCPStructureCPU(P, A, scs_cones)
    qcp_initial = HostQCP(P, A, q, b,
                          target_x, target_y, target_s,
                          problem_structure)
    fake_target_x = 1e-3 * jnp.arange(jnp.size(q), dtype=q.dtype)
    fake_target_y = 1e-3 * jnp.arange(jnp.size(b), dtype=b.dtype)
    fake_target_s = 1e-3 * jnp.arange(jnp.size(b), dtype=b.dtype)

    start_time = time.perf_counter()
    result = make_step(qcp_initial, fake_target_x, fake_target_y,
                       fake_target_s, step_size=1e-5)
    result[0].block_until_ready()
    end_time = time.perf_counter()
    # NOTE(quill): well, technically VJP + loss + step computations
    print("diffqcp VJP compile + compute took: ", end_time - start_time)

    # --- test compiled solve ---

    start_time = time.perf_counter()
    result = make_step(qcp_initial, fake_target_x, fake_target_y,
                       fake_target_s, step_size=1e-5)
    result[0].block_until_ready()
    end_time = time.perf_counter()
    print("Compiled diffqcp VJP compute took: ", end_time - start_time)

    # --- ---

    initial_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    prob_data_cpu = QCPProbData(initial_problem)

    cones = prob_data_cpu.clarabel_cones
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.presolve_enable = False

    solver = clarabel.DefaultSolver(prob_data_cpu.Pupper_csc,
                                    prob_data_cpu.q,
                                    prob_data_cpu.Acsc,
                                    prob_data_cpu.b,
                                    cones,
                                    settings)
    
    num_iter = 100

    start_time = time.perf_counter()
    losses = grad_desc(Pk=scoo_to_bcoo(prob_data_cpu.Pupper_coo),
                       Ak=scoo_to_bcoo(prob_data_cpu.Acoo),
                       qk = jnp.array(prob_data_cpu.q),
                       bk = jnp.array(prob_data_cpu.b),
                       target_x=target_x,
                       target_y=target_y,
                       target_s=target_s,
                       qcp_problem_structure=problem_structure,
                       data=prob_data_cpu,
                       Pcoo_csc_perm=Pupper_coo_to_csc_order,
                       Acoo_csc_perm=A_coo_to_csc_order,
                       clarabel_solver=solver, num_iter=num_iter)
    losses[0].block_until_ready()
    end_time = time.perf_counter()
    print(f"The learning loop time was {end_time - start_time} seconds")
    print(f"Avg. iteration (solve + VJP) time: {(end_time - start_time) / num_iter}")
    losses = jnp.stack(losses)
    losses = np.asarray(losses)

    plt.figure(figsize=(8, 6))
    plt.plot(range(num_iter), losses, label="Objective Trajectory")
    plt.xlabel("num. iterations")
    plt.ylabel("Objective function")
    plt.legend()
    plt.title(label="diffqcp")
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if prob_data_cpu.n > 99:
        output_path = os.path.join(results_dir, "diffqcp_cpu_probability_large.svg")
    else:
        output_path = os.path.join(results_dir, "diffqcp_cpu_probability_small.svg")
    plt.savefig(output_path, format="svg")
    plt.close()