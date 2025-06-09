from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/cvxgrp/CuClarabel.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

import time
import os
import torch
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from torch.utils.dlpack import to_dlpack
from cupy import from_dlpack

from diffqcp import QCP
from diffqcp.utils import to_tensor, to_sparse_csr_tensor
from tests.utils import data_from_cvxpy_problem_quad
from experiments.cvx_problem_generator import generate_group_lasso
from experiments.utils import GradDescTestResult

results_dir = os.path.join(os.path.dirname(__file__), "results")

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

def torch_csr_to_cupy_csr(X: torch.Tensor) -> csr_matrix:
    crow_indices = X.crow_indices()
    col_indices = X.col_indices()
    values = X.values()

    crow_cp = from_dlpack(to_dlpack(crow_indices))
    col_cp = from_dlpack(to_dlpack(col_indices))
    val_cp = from_dlpack(to_dlpack(values))

    assert crow_cp.__cuda_array_interface__['data'][0] == crow_indices.__cuda_array_interface__['data'][0]
    assert col_cp.__cuda_array_interface__['data'][0] == col_indices.__cuda_array_interface__['data'][0]
    assert val_cp.__cuda_array_interface__['data'][0] == values.__cuda_array_interface__['data'][0]

    # Build CuPy CSR matrix
    shape = X.shape
    Xcupy = csr_matrix((val_cp, col_cp, crow_cp), shape=shape)
    return Xcupy

def torch_to_jl(P, A, q, b, Pnnz, Annz):
    """returns Pjl, Ajl, qjl, bjl"""

    if Pnnz == 0:
        Pjl = jl.CuSparseMatrixCSR(jl.spzeros(n, n))
    else:
        Pcupy = torch_csr_to_cupy_csr(P)

        data_ptr    = int(Pcupy.data.data.ptr)
        indices_ptr = int(Pcupy.indices.data.ptr)
        indptr_ptr  = int(Pcupy.indptr.data.ptr)

        Pjl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, n, n, Pnnz)
    
    if Annz == 0:
        Ajl = jl.CuSparseMatrixCSR(jl.spzeros(m, n))
    else:
        Acupy = torch_csr_to_cupy_csr(A)
        # assert Acupy.indptr.__cuda_array_interface__['data'][0] == A.crow_indices().__cuda_array_interface__['data'][0]
        # assert Acupy.indices.__cuda_array_interface__['data'][0] == A.col_indices().__cuda_array_interface__['data'][0]
        # assert Acupy.data.__cuda_array_interface__['data'][0] == A.values().__cuda_array_interface__['data'][0]

        data_ptr    = int(Acupy.data.data.ptr)
        indices_ptr = int(Acupy.indices.data.ptr)
        indptr_ptr  = int(Acupy.indptr.data.ptr)

        Ajl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, m, n, Annz)
    
    qcupy = cp.asarray(q)
    # the following also passes
    # assert qcupy.__cuda_array_interface__['data'][0] == q.__cuda_array_interface__['data'][0]
    qjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(qcupy.data.ptr), qcupy.size)
    bcupy = cp.asarray(b)
    # assert bcupy.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]
    bjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bcupy.data.ptr), bcupy.size)
    # also return cupy arrays so they aren't GCed, which might break being able to point
    if Pnnz == 0 and Annz == 0:
        return Pjl, Ajl, qjl, bjl, qcupy, bcupy
    elif Pnnz == 0:
        return Pjl, Ajl, qjl, bjl, Acupy, qcupy, bcupy
    elif Annz == 0:
        return Pjl, Ajl, qjl, bjl, Pcupy, qcupy, bcupy
    else:
        return Pjl, Ajl, qjl, bjl, Pcupy, Acupy, qcupy, bcupy


def grad_desc(
    qcp: QCP,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_s: torch.Tensor,
    solver_jl,
    num_iter: int = 1000,
    step_size: float = 0.01,
    improvement_factor: float = 1e-2,
    fixed_tol: float = 1e-4,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None
):
    curr_iter = 0
    optimal = False
    f0s = torch.zeros(num_iter+1, dtype=dtype, device=device)
    step_size = torch.tensor(step_size, dtype=dtype, device=device)

    def f0(x: torch.Tensor, y: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
            return (0.5 * torch.linalg.norm(x - target_x)**2 + 0.5 * torch.linalg.norm(y - target_y)**2
                    + 0.5 * torch.linalg.norm(s - target_s)**2)
    
    data = torch_to_jl(qcp.P, qcp.A, qcp.q, qcp.b, qcp.data.P_filtered_nnz, qcp.data.A_filtered_nnz)
    Pjl, Ajl, qjl, bjl = data[0], data[1], data[2], data[3]
    # the cupy arrays should be in scope / not GCed since they are in the data tuple?
    
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    
    while curr_iter < num_iter:

        jl.Clarabel.update_P_b(solver_jl, Pjl)
        jl.Clarabel.update_A_b(solver_jl, Ajl)
        jl.Clarabel.update_q_b(solver_jl, qjl)
        jl.Clarabel.update_b_b(solver_jl, bjl)

        jl.Clarabel.solve_b(solver_jl)
        xcupy = JuliaCuVector2CuPyArray(solver_jl.solution.x)
        ycupy = JuliaCuVector2CuPyArray(solver_jl.solution.z)
        scupy = JuliaCuVector2CuPyArray(solver_jl.solution.s)

        xk = torch.as_tensor(xcupy, dtype=dtype, device=device)
        yk = torch.as_tensor(ycupy, dtype=dtype, device=device)
        sk = torch.as_tensor(scupy, dtype=dtype, device=device)

        qcp.update_solution(xk, yk, sk)

        f0k = f0(xk, yk, sk)
        f0s[curr_iter] = f0k
        curr_iter += 1

        if curr_iter > 1 and ((f0k / f0s[0]) < improvement_factor or f0k < fixed_tol):
            optimal = True
            break
        
        d_theta = qcp.vjp(xk - target_x, yk - target_y, sk - target_s)

        dP = -step_size * d_theta[0]
        dA = -step_size * d_theta[1]
        dq = -step_size * d_theta[2]
        db = -step_size * d_theta[3]

        ddata = torch_to_jl(dP, dA, dq, db, qcp.data.P_filtered_nnz, qcp.data.A_filtered_nnz)
        dPjl, dAjl, dqjl, dbjl = ddata[0], ddata[1], ddata[2], ddata[3]
        # d (cupy arrays) still in scope since returned

        Pjl = Pjl + dPjl
        Ajl = Ajl + dAjl
        qjl = qjl + dqjl
        bjl = bjl + dbjl

        # print("dA type: ", type(dA))
        # print("dA shape: ", dA.shape)
        # print("dA layout: ", dA.layout)

        # qcp.perturb_data(dP, dA, dq, db)

        # the following operate in place, so shouldn't have to repoint Julia data
        # qcp.data._P += dP
        # qcp.data._A += dA
        # qcp.data._AT = qcp.data._A_transpose(qcp.data_A.values())
        # qcp.data.q += dq
        # qcp.data.b += db

    torch.cuda.synchronize()
    end_time = time.perf_counter()
    print("Learning time: ", end_time - start_time)

    f0_traj = f0s[0:curr_iter]
    del f0s
    return GradDescTestResult(
            passed=optimal, num_iterations=curr_iter, obj_traj=f0_traj.cpu().numpy(), for_qcp=True
        )


if __name__ == '__main__':

    np.random.seed(0)

    # generate high-dimensional problem:
    #   - choose m = n since `generate_group_lasso` takes p = 10n
    #       as number of features.

    # m = n = 1000
    m = n = 25
    dtype = torch.float64
    device = torch.device('cuda')

    print("starting to build problem")

    target_problem = generate_group_lasso(n=n, m=m)

    print("built problem")

    print("extracting data from problem")
    qcp_data = data_from_cvxpy_problem_quad(target_problem)
    print("finished extracting data from problem")

    Pcpu, Acpu = qcp_data[0], qcp_data[2] # NOTE we take full `P` (not upper triangular)
    qcpu, bcpu = qcp_data[3], qcp_data[4]
    scs_cones, clarabel_cones = qcp_data[5], qcp_data[6]
    del qcp_data

    # Move data to device for `diffqcp` to access
    P = to_sparse_csr_tensor(Pcpu, dtype=dtype, device=device)
    A = to_sparse_csr_tensor(Acpu, dtype=dtype, device=device)
    q = to_tensor(qcpu, dtype=dtype, device=device)
    b = to_tensor(bcpu,dtype=dtype, device=device)

    n = P.shape[0]
    m = A.shape[0]

    # === ===
    # TODO(quill): can we just use `torch` directly?
    
    Pnnz = Pcpu.nnz
    Annz = Acpu.nnz

    if Pnnz == 0:
        Pjl = jl.CuSparseMatrixCSR(jl.spzeros(n, n))
    else:
        Pcupy = torch_csr_to_cupy_csr(P)
        assert Pcupy.indptr.__cuda_array_interface__['data'][0] == P.crow_indices().__cuda_array_interface__['data'][0]
        assert Pcupy.indices.__cuda_array_interface__['data'][0] == P.col_indices().__cuda_array_interface__['data'][0]
        assert Pcupy.data.__cuda_array_interface__['data'][0] == P.values().__cuda_array_interface__['data'][0]

        data_ptr    = int(Pcupy.data.data.ptr)
        indices_ptr = int(Pcupy.indices.data.ptr)
        indptr_ptr  = int(Pcupy.indptr.data.ptr)

        Pjl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, n, n, Pnnz)
    
    if Annz == 0:
        Ajl = jl.CuSparseMatrixCSR(jl.spzeros(m, n))
    else:
        Acupy = torch_csr_to_cupy_csr(A)
        assert Acupy.indptr.__cuda_array_interface__['data'][0] == A.crow_indices().__cuda_array_interface__['data'][0]
        assert Acupy.indices.__cuda_array_interface__['data'][0] == A.col_indices().__cuda_array_interface__['data'][0]
        assert Acupy.data.__cuda_array_interface__['data'][0] == A.values().__cuda_array_interface__['data'][0]

        data_ptr    = int(Acupy.data.data.ptr)
        indices_ptr = int(Acupy.indices.data.ptr)
        indptr_ptr  = int(Acupy.indptr.data.ptr)

        Ajl = jl.Clarabel.cupy_to_cucsrmat(jl.Float64, data_ptr, indices_ptr, indptr_ptr, m, n, Annz)
    
    qcupy = cp.asarray(q)
    # the following also passes
    assert qcupy.__cuda_array_interface__['data'][0] == q.__cuda_array_interface__['data'][0]
    qjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(qcupy.data.ptr), qcupy.size)
    bcupy = cp.asarray(b)
    assert bcupy.__cuda_array_interface__['data'][0] == b.__cuda_array_interface__['data'][0]
    bjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bcupy.data.ptr), bcupy.size)

    # === construct target problem ===

    # Assign Python variables to Julia variables
    jl.zero_cone = scs_cones['z']
    jl.nonneg_cone = scs_cones['l']
    jl.soc = scs_cones['q']
    jl.psd_cone = scs_cones['s']
    jl.exp_cones = scs_cones['ep']

    # Now use Julia variables in Julia code
    jl.seval('''
        cones = Dict(
            "f" => zero_cone,
            "l" => nonneg_cone,
            "q" => soc,
            "s" => psd_cone,
            "ep" => exp_cones
        )
        settings = Clarabel.Settings(direct_solve_method = :cudss)
    ''')
    jl.solver = jl.Clarabel.Solver(Pjl, qjl, Ajl, bjl, jl.cones, jl.settings)
    jl.Clarabel.solve_b(jl.solver) # solve new problem w/o creating memory

    xcupy = JuliaCuVector2CuPyArray(jl.solver.solution.x)
    ycupy = JuliaCuVector2CuPyArray(jl.solver.solution.z)
    scupy = JuliaCuVector2CuPyArray(jl.solver.solution.s)

    x_target = torch.as_tensor(xcupy, dtype=dtype, device=device)
    y_target = torch.as_tensor(ycupy, dtype=dtype, device=device)
    s_target = torch.as_tensor(scupy, dtype=dtype, device=device)

    print('starting to build initial learning problem.')
    initial_problem = generate_group_lasso(n=n, m=m)
    print('finished building initial learning problem.')

    print("extracting data from problem")
    qcp_data = data_from_cvxpy_problem_quad(target_problem)
    print("finished extracting data from problem")

    Pcpu, Acpu = qcp_data[0], qcp_data[2] # NOTE we take full `P` (not upper triangular)
    qcpu, bcpu = qcp_data[3], qcp_data[4]
    del qcp_data
    
    # Move data to device for `diffqcp` to access
    P = to_sparse_csr_tensor(Pcpu, dtype=dtype, device=device)
    A = to_sparse_csr_tensor(Acpu, dtype=dtype, device=device)
    q = to_tensor(qcpu, dtype=dtype, device=device)
    b = to_tensor(bcpu,dtype=dtype, device=device)
    
    qcp = QCP(P, A, q, b, cone_dict=scs_cones, P_is_upper=False, dtype=torch.float64, device=device, reduce_fp_flops=True)
    
    result = grad_desc(qcp, x_target, y_target, s_target, jl.solver, num_iter=20)
    
    save_path = os.path.join(results_dir, "diffqcp_paper_experiment.png")
    result.plot_obj_traj(save_path)
    
