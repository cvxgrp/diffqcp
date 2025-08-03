from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/cvxgrp/CuClarabel.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

type CuVector = jl.CUDA.Cuvector
type CuSparseMatrixCSR = jl.CUDA.CUSPARSE.CuSparseMatrixCSR

# TODO(quill): set JAX flags

import time
import os
from dataclasses import dataclass, field
import functools as ft
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
from jaxtyping import Float, Array
import cupy as cp
from cupy.sparse import csr_matrix
import equinox as eqx
import scipy.sparse as sparse
import patdb

from diffqcp.qcp import DeviceQCP, DeviceQCP
import experiments.cvx_problem_generator as prob_generator
from tests.helpers import QCPProbData

# what auxillary objects can I create to store the CuPy <-> Julia objects?

# will need helpers to do CuPy CSR <-> JAX BCSR
#   put in this function how to handle a 0 matrix

def JuliaCuVector2CuPyArray(jl_arr):
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


# def cpu_csr_to_cupy_csr(mat: sparse.csr_matrix) -> csr_matrix:
#     # TODO(quill): will this handle if there are no nonzeros?
#     print("data ndim: ", np.ndim(mat.data)) # data ndim:  1
#     print("indices ndim: ", np.ndim(mat.indices)) # indices ndim:  1
#     print("indptr ndim: ", np.ndim(mat.indptr)) # indptr ndim:  1
#     return csr_matrix((mat.data, mat.indices, mat.indptr),
#                       shape=mat.shape)


def cpu_csr_to_cupy_csr(mat: sparse.csr_matrix) -> csr_matrix:
    # Ensure all arrays are 1-D
    data = mat.data.flatten()
    indices = mat.indices.flatten()
    indptr = mat.indptr.flatten()
    patdb.debug()
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
        indptr_ptr = int(mat.indices.data.ptr)
        mat_jl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, m, n, nnz)
    return mat_jl


@dataclass
class SolverData:
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


@ft.partial(jax.jit, static_argnums=1)
def make_step(
    qcp_params,
    qcp_static,
    target_x, 
    target_y,
    target_s,
    step_size
):
    """
    Technically the target vectors and the step size are static,...but is this important
    to include?
    """
    qcp: DeviceQCP = eqx.combine(qcp_params, qcp_static)
    loss = compute_loss(target_x, target_y, target_s, qcp.x, qcp.y, qcp.s)
    dP, dA, dq, db = qcp.vjp(qcp.x - target_x,
                             qcp.y - target_y,
                             qcp.s - target_s)
    dP = -step_size * dP
    dA = -step_size * dA
    dq = -step_size * dq
    db = -step_size * db

    return (loss, dP, dA, dq, db)


def grad_desc(
    qcp: DeviceQCP,
    target_x: Float[Array, " n"],
    target_y: Float[Array, " m"],
    target_s: Float[Array, " m"],
    cuclarabel_solver,
    qcp_problem_structure,
    num_iter: int = 1000,
    step_size: float = 1e-5,
):
    # the solver should already be warmed up.
    # 
    # so at this point I should not
    # Need to solve the problem, so can collect (x, y, s)

    curr_iter = 0 # TODO(quill): determine iteration counting scheme
    
    while curr_iter < num_iter:

        jl.Clarabel.solve_b(cuclarabel_solver)

        # (x, y, s) Julia -> CuPy -> JAX

        # TODO(quill): do a simple LL and make sure this isn't causing
        #  unecessary recompilations
        qcp = DeviceQCP(P, A, q, b, x, y, s, qcp_problem_structure)
        # qcp = DeviceQCP()
        # qcp_params
        loss, *dtheta = make_step(qcp, target_x, target_y, target_s, step_size)
        # now update your Julia data
        #   need handle case where P is all zeros

        
        
if __name__ == "__main__":

    m = 20
    n = 10
    problem = prob_generator.generate_group_lasso(n, m)
    
    # grab problem data
    prob_data_cpu = QCPProbData(problem)
    # move data to CuPy arrays and create Julia ptrs.
    # patdb.debug()
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
    jl.Clarabel.solve_b(jl.solver) # solve new problem w/o creating memory

    
    # obtain xjl<-xcp, yjl<-ycp, sjl<-scp
    # we only care about these from the original problem
    
    # (make sure to copy the targets as necessary)
    # dlpack 
    # JIT compile `make_step`
    # 


    pass