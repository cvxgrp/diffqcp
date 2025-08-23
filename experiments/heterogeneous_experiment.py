"""
Experiment solving problems on the GPU and computing VJPs on the CPU.
"""
from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/oxfordcontrol/Clarabel.jl.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

type CuVector = jl.CUDA.Cuvector
type CuSparseMatrixCSR = jl.CUDA.CUSPARSE.CuSparseMatrixCSR

import time
import os
from dataclasses import dataclass, field
import numpy as np
import jax
# TODO(quill): set JAX flags
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax.numpy.linalg as la
from jaxtyping import Float, Array
import cupy as cp
from cupy.sparse import csr_matrix
import equinox as eqx
import scipy.sparse as sparse
import patdb
from jax.experimental.sparse import BCOO

from diffqcp.qcp import HostQCP, QCPStructureCPU
import experiments.cvx_problem_generator as prob_generator
from tests.helpers import QCPProbData, scoo_to_bcoo
import matplotlib.pyplot as plt

def JuliaCuVector2CuPyArray(jl_arr) -> cp.ndarray:
    """Taken from https://github.com/cvxgrp/CuClarabel/blob/main/src/python/jl2py.py.
    """
    # Get the device pointer from Julia
    pDevice = jl.Int(jl.pointer(jl_arr))

    # Get array length and element type
    span = jl.size(jl_arr)
    dtype = jl.eltype(jl_arr)

    # Map Julia type to CuPy dtype
    if dtype == jl.Float64:
        dtype = cp.float64
    else:
        dtype = cp.float32

    # Compute memory size in bytes (assuming 1D vector)
    size_bytes = int(span[0] * cp.dtype(dtype).itemsize)

    # Create CuPy memory view from the Julia pointer
    mem = cp.cuda.UnownedMemory(pDevice, size_bytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, 0)

    # Wrap into CuPy ndarray
    arr = cp.ndarray(shape=span, dtype=dtype, memptr=memptr)
    return arr


def cpu_csr_to_cupy_csr(mat: sparse.csr_matrix) -> csr_matrix:
    # Ensure all arrays are 1-D
    data = cp.array(mat.data)
    indices = cp.array(mat.indices)
    indptr = cp.array(mat.indptr)
    return csr_matrix((data, indices, indptr), shape=mat.shape)


def cupy_csr_to_julia_csr(mat: csr_matrix) -> CuSparseMatrixCSR:
    shape = mat.shape
    m, n = shape[0], shape[1]
    nnz = mat.nnz
    if nnz == 0:
        mat_jl = jl.CuSparseMatrixCSR(jl.spzeros(m, n))
    else:
        data_ptr = int(mat.data.data.ptr)
        indices_ptr = int(mat.indices.data.ptr)
        indptr_ptr = int(mat.indptr.data.ptr)
        mat_jl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, m, n, nnz)
    return mat_jl


@dataclass
class SolverData:
    """
    NOTE(quill): Never use the indices of `Pcp` or `Acp` after creating Julia ptrs.
    """
    Pcp: csr_matrix
    Acp: csr_matrix
    qcp: cp.ndarray # q cupy, not QCP
    bcp: cp.ndarray

    Pjl: CuSparseMatrixCSR = field(init=False)
    Ajl: CuSparseMatrixCSR = field(init=False)
    qjl: CuVector = field(init=False)
    bjl: CuVector = field(init=False)

    def __post_init__(self):
        self.Pjl = cupy_csr_to_julia_csr(self.Pcp)
        self.Ajl = cupy_csr_to_julia_csr(self.Acp)        
        self.qjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(self.qcp.data.ptr), self.qcp.size)
        self.bjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(self.bcp.data.ptr), self.bcp.size)


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
    solver_data: SolverData,
    cuclarabel_solver,
    num_iter: int = 100,
    step_size = 1e-5
):
    curr_iter = 0
    losses = []

    while curr_iter < num_iter:

        jl.Clarabel.solve_b(cuclarabel_solver)
        
        xkcp = JuliaCuVector2CuPyArray(jl.solver.solution.x)
        xk = cp.asnumpy(xkcp)
        ykcp = JuliaCuVector2CuPyArray(jl.solver.solution.z)
        yk = cp.asnumpy(ykcp)
        skcp = JuliaCuVector2CuPyArray(jl.solver.solution.s)
        sk = cp.asnumpy(skcp)

        qcp = HostQCP(Pk, Ak, qk, bk, xk, yk, sk, qcp_problem_structure)
        loss, *dtheta_steps = make_step(qcp, target_x, target_y, target_s, step_size)
        losses.append(loss)

        dP_step, dA_step, dq_step, db_step = dtheta_steps
        Pk.data += dP_step.data
        Ak.data += dA_step.data
        qk += dq_step
        bk += db_step
        # update solver data
        solver_data.Pcp.data += cp.asarray(dP_step.data)
        solver_data.Acp.data += cp.asarray(dA_step.data)
        solver_data.qcp += cp.asarray(dq_step)
        solver_data.bcp += cp.asarray(db_step)
        
        # Also update solver
        jl.Clarabel.update_P_b(cuclarabel_solver, solver_data.Pjl)
        jl.Clarabel.update_A_b(cuclarabel_solver, solver_data.Ajl)
        jl.Clarabel.update_q_b(cuclarabel_solver, solver_data.qjl)
        jl.Clarabel.update_b_b(cuclarabel_solver, solver_data.bjl)

        curr_iter += 1

    return losses

if __name__ == "__main__":

    np.random.seed(28)

    # m = 20
    # n = 10
    m = 2_000
    n = 1_000
    # problem = prob_generator.generate_group_lasso(n=n, m=m) #TODO(quill): CuClarabel failing for this problem
    start_time = time.perf_counter()
    target_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    # target_problem = prob_generator.generate_LS_problem(m=m, n=n)
    prob_data_cpu = QCPProbData(target_problem)
    end_time = time.perf_counter()
    print("Time to generate the target problem,"
          + " canonicalize it, and solve it on the CPU:"
          + f" {end_time - start_time} seconds")
    
    # === Obtain target vectors + warm up GPU and JIT compile CuClarabel and diffqcp ===
    
    solver_data = SolverData(cpu_csr_to_cupy_csr(prob_data_cpu.Pcsr),
                             cpu_csr_to_cupy_csr(prob_data_cpu.Acsr),
                             cp.array(prob_data_cpu.q),
                             cp.array(prob_data_cpu.b))
    
    # Create Julia cone variables
    jl.zero_cone = prob_data_cpu.scs_cones["z"]
    jl.nonneg_cone = prob_data_cpu.scs_cones["l"]
    jl.soc = prob_data_cpu.scs_cones["q"]
    # Now use Julia variables in Julia code
    jl.seval("""
        cones = Dict(
             "f" => zero_cone,
             "l" => nonneg_cone,
             "q" => soc
        )
        settings = Clarabel.Settings(direct_solve_method = :cudss)
        settings.verbose = false
    """)
    # create CuClarabel solver
    jl.solver = jl.Clarabel.Solver(solver_data.Pjl, solver_data.qjl,
                                   solver_data.Ajl, solver_data.bjl,
                                   jl.cones, jl.settings)
    start_solve = time.perf_counter()
    jl.Clarabel.solve_b(jl.solver) # solve new problem w/o creating memory
    cp.cuda.Device().synchronize()
    end_solve = time.perf_counter()
    print(f"CuClarabel compile + solve took: {end_solve - start_solve} seconds")

    xcp = JuliaCuVector2CuPyArray(jl.solver.solution.x)
    ycp = JuliaCuVector2CuPyArray(jl.solver.solution.z)
    scp = JuliaCuVector2CuPyArray(jl.solver.solution.s)

    x_target = jnp.array(cp.asnumpy(xcp))
    y_target = jnp.array(cp.asnumpy(ycp))
    s_target = jnp.array(cp.asnumpy(scp))
    
    # --- Time compiled speedup ---
    start_solve = time.perf_counter()
    jl.Clarabel.solve_b(jl.solver) # solve new problem w/o creating memory
    cp.cuda.Device().synchronize()
    end_solve = time.perf_counter()
    print(f"Compiled CuClarabel solve took: {end_solve - start_solve} seconds")
    # --- ---

    P = scoo_to_bcoo(prob_data_cpu.Pupper_coo)
    A = scoo_to_bcoo(prob_data_cpu.Acoo)
    q = prob_data_cpu.q
    b = prob_data_cpu.b
    scs_cones = prob_data_cpu.scs_cones
    problem_structure = QCPStructureCPU(P, A, scs_cones)
    qcp_initial = HostQCP(P, A, q, b,
                          x_target, y_target, s_target,
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
    print("The result is on the: ", result[0].device)

    # --- ---

    # === Finished initialization ===

    # === Now get problem we'll actually use for LL ===

    start_time = time.perf_counter()
    initial_problem = prob_generator.generate_least_squares_eq(m=m, n=n)
    # initial_problem = prob_generator.generate_LS_problem(m=m, n=n)
    prob_data_cpu = QCPProbData(initial_problem)
    end_time = time.perf_counter()
    print("Time to generate the initial (starting point) problem,"
          + f" canonicalize it, and solve it on the cpu is: {end_time - start_time} seconds")
    print(f"Canonicalized n is: {prob_data_cpu.n}")
    print(f"Canonicalized m is: {prob_data_cpu.m}")

    # Put new data on GPU and create CuPy <-> Julia linking
    
    solver_data = SolverData(cpu_csr_to_cupy_csr(prob_data_cpu.Pcsr),
                             cpu_csr_to_cupy_csr(prob_data_cpu.Acsr),
                             cp.array(prob_data_cpu.q),
                             cp.array(prob_data_cpu.b))
    
    # Because problem is DPP-compliant, now just update existing solver object
    jl.Clarabel.update_P_b(jl.solver, solver_data.Pjl)
    jl.Clarabel.update_A_b(jl.solver, solver_data.Ajl)
    jl.Clarabel.update_q_b(jl.solver, solver_data.qjl)
    jl.Clarabel.update_b_b(jl.solver, solver_data.bjl)

    num_iter = 100

    start_time = time.perf_counter()
    losses = grad_desc(Pk=scoo_to_bcoo(prob_data_cpu.Pupper_coo),
                       Ak=scoo_to_bcoo(prob_data_cpu.Acoo),
                       qk = jnp.array(prob_data_cpu.q),
                       bk = jnp.array(prob_data_cpu.b),
                       target_x=x_target,
                       target_y=y_target,
                       target_s=s_target,
                       qcp_problem_structure=problem_structure,
                       solver_data=solver_data,
                       cuclarabel_solver=jl.solver,
                       num_iter=num_iter)
    cp.cuda.Device().synchronize()
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
    if prob_data_cpu.n > 999:
        output_path = os.path.join(results_dir, "hetero_probability_large.svg")
    else:
        output_path = os.path.join(results_dir, "hetero_probability_small.svg")
    plt.savefig(output_path, format="svg")
    plt.close()