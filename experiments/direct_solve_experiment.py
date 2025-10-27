"""
Experiment solving problems on the CPU and computing VJPs on the GPU.
"""
import time
import os
from dataclasses import dataclass
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.numpy.linalg as la
from jaxtyping import Float, Array
import equinox as eqx
from jax.experimental.sparse import BCSR
import clarabel
from scipy.sparse import (spmatrix, sparray,
                          csc_matrix, csc_array, triu)
import patdb
import matplotlib.pyplot as plt

from diffqcp import DeviceQCP, QCPStructureGPU
import experiments.cvx_problem_generator as prob_generator
from tests.helpers import QCPProbData, scsr_to_bcsr

type SP = spmatrix | sparray
type SCSC = csc_matrix | csc_array


def compute_loss(target_x, target_y, target_s, x, y, s):
    return (0.5 * la.norm(x - target_x)**2 + 0.5 * la.norm(y - target_y)**2
            + 0.5 * la.norm(s - target_s)**2) 


def _update_data(
    dP, dA, dq, db, Pdata, Adata, q, b, step_size
):
    new_Pdata = Pdata - step_size * dP.data
    new_Adata = Adata - step_size * dA.data
    new_q = q - step_size * dq
    new_b = b - step_size * db
    return new_Pdata, new_Adata, new_q, new_b


# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def make_step(
    qcp: DeviceQCP,
    target_x: Float[Array, " n"],
    target_y: Float[Array, " m"],
    target_s: Float[Array, " m"],
    Pdata: Float[Array, "..."],
    Adata: Float[Array, "..."],
    q: Float[Array, " n"],
    b: Float[Array, " m"],
    step_size: float
) -> tuple[Float[Array, ""], Float[Array, "..."], Float[Array, "..."],
           Float[Array, " n"], Float[Array, " m"]]:
    loss = eqx.filter_jit(compute_loss)(target_x, target_y, target_s, qcp.x, qcp.y, qcp.s)
    dP, dA, dq, db = qcp.vjp(qcp.x - target_x,
                             qcp.y - target_y,
                             qcp.s - target_s,
                             use_direct_solve=True)
    updated_data = eqx.filter_jit(_update_data)(dP, dA, dq, db, Pdata, Adata, q, b, step_size)
    return loss, *updated_data


def grad_desc(
    Pk: Float[BCSR, "n n"],
    Ak: Float[BCSR, "m n"],
    qk: Float[Array, " n"],
    bk: Float[Array, " m"],
    target_x: Float[Array, " n"],
    target_y: Float[Array, " m"],
    target_s: Float[Array, " m"],
    qcp_problem_structure: QCPStructureGPU,
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
        
        qcp = DeviceQCP(Pk, Ak, qk, bk, xk, yk, sk, qcp_problem_structure)
        
        loss, *new_data = make_step(qcp, target_x, target_y, target_s,
                                    Pk.data, Ak.data, qk, bk, step_size)
        losses.append(loss)

        Pk_data, Ak_data, qk, bk = new_data
        Pk.data, Ak.data = Pk_data, Ak_data
        # need to grap uppper part of P only
        data.Pcsc.data = np.asarray(Pk.data)[Pcoo_csc_perm]
        data.Pupper_csc = triu(data.Pcsr, format="csc")
        data.Acsc.data = np.asarray(Ak.data)[Acoo_csc_perm]
        data.q = np.asarray(qk)
        data.b = np.asarray(bk)
        
        solver.update(P=data.Pupper_csc, q=data.q, A=data.Acsc, b=data.b)

        curr_iter += 1

    return losses

if __name__ == "__main__":
    np.random.seed(28)
    
    # SMALL
    # m = 20
    # n = 10
    # MEDIUM-ish
    m = 200
    n = 100
    # LARGE-ish
    # m = 2_000
    # n = 1_000
    target_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    prob_data_cpu = QCPProbData(target_problem)

    # ensure validity of the following ordering permutations.
    np.testing.assert_allclose(prob_data_cpu.Pcoo.data,
                               prob_data_cpu.Pcsr.data)
    
    np.testing.assert_allclose(prob_data_cpu.Acoo.data,
                               prob_data_cpu.Acsr.data)

    P_coo_to_csc_order = np.lexsort((prob_data_cpu.Pcoo.row,
                                     prob_data_cpu.Pupper_coo.col))
    A_coo_to_csc_order = np.lexsort((prob_data_cpu.Acoo.row,
                                     prob_data_cpu.Acoo.col))
    
    cones = prob_data_cpu.clarabel_cones
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    # settings.presolve_enable = False

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

    P = scsr_to_bcsr(prob_data_cpu.Pcsr)
    A = scsr_to_bcsr(prob_data_cpu.Acsr)
    q = prob_data_cpu.q
    b = prob_data_cpu.b
    scs_cones = prob_data_cpu.scs_cones
    problem_structure = QCPStructureGPU(P, A, scs_cones)
    qcp_initial = DeviceQCP(P, A, q, b,
                          target_x, target_y, target_s,
                          problem_structure)
    fake_target_x = 1e-3 * jnp.arange(jnp.size(q), dtype=q.dtype)
    fake_target_y = 1e-3 * jnp.arange(jnp.size(b), dtype=b.dtype)
    fake_target_s = 1e-3 * jnp.arange(jnp.size(b), dtype=b.dtype)

    start_time = time.perf_counter()
    result = make_step(qcp_initial, fake_target_x, fake_target_y,
                       fake_target_s, P.data, A.data, q, b, step_size=1e-5)
    result[0].block_until_ready()
    end_time = time.perf_counter()
    # NOTE(quill): well, technically VJP + loss + step computations
    print("diffqcp VJP compile + compute took: ", end_time - start_time)

    # --- test compiled solve ---

    start_time = time.perf_counter()
    # with jax.profiler.trace("/home/quill/diffqcp/tmp/jax-trace", create_perfetto_link=True):
    result = make_step(qcp_initial, fake_target_x, fake_target_y,
                    fake_target_s, P.data, A.data, q, b, step_size=1e-5)
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
    
    num_iter = 1

    start_time = time.perf_counter()
    losses = grad_desc(Pk=scsr_to_bcsr(prob_data_cpu.Pcsr),
                       Ak=scsr_to_bcsr(prob_data_cpu.Acsr),
                       qk = jnp.array(prob_data_cpu.q),
                       bk = jnp.array(prob_data_cpu.b),
                       target_x=target_x,
                       target_y=target_y,
                       target_s=target_s,
                       qcp_problem_structure=problem_structure,
                       data=prob_data_cpu,
                       Pcoo_csc_perm=P_coo_to_csc_order,
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
        output_path = os.path.join(results_dir, "dsolve_probability_large.svg")
    else:
        output_path = os.path.join(results_dir, "dsolve_probability_small.svg")
    plt.savefig(output_path, format="svg")
    plt.close()