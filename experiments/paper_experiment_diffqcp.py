from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/cvxgrp/CuClarabel.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

import time
from copy import deepcopy
import torch
import numpy as np
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from torch.utils.dlpack import to_dlpack
from cupy import from_dlpack

from diffqcp import QCP
from diffqcp.utils import to_tensor, to_sparse_csr_tensor
from tests.utils import data_from_cvxpy_problem_quad, data_from_cvxpy_problem_linear
from experiments.cvx_problem_generator import generate_group_lasso

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

def grad_desc():
    pass

if __name__ == '__main__':

    # generate high-dimensional problem:
    #   - choose m = n since `generate_group_lasso` takes p = 10n
    #       as number of features.

    # m = n = 1000
    m = n = 25
    dtype = torch.float64
    device = torch.device('cuda')

    print("starting to build problem")

    target_problem = generate_group_lasso(n=n, m=m)
    # initial_problem = generate_group_lasso(n=n, m=m)

    print("built problem")

    print("extracting data from problem")
    qcp_data = data_from_cvxpy_problem_quad(target_problem)
    print("finished extracting data from problem")

    Pcpu, Acpu = qcp_data[0], qcp_data[2] # NOTE we take full `P` (not upper triangular)
    qcpu, bcpu = qcp_data[3], qcp_data[4]
    scs_cones, clarabel_cones = qcp_data[5], qcp_data[6]

    # Move data to device for `diffqcp` to access
    P = to_sparse_csr_tensor(Pcpu, dtype=dtype, device=device)
    A = to_sparse_csr_tensor(Acpu, dtype=dtype, device=device)
    q = to_tensor(qcpu, dtype=dtype, device=device)
    b = to_tensor(bcpu,dtype=dtype, device=device)

    n = P.shape[0]
    m = A.shape[0]

    # === ===
    # Now create `cupy` pointers to data, then Julia pointers to data.
    # TODO(quill): can we just use `torch` directly?
    
    
    Pnnz = Pcpu.nnz # TODO(quill): need to ensure no explicit zeros like I do in `ProblemData`?
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
    # TODO(quill): remove `ed` from `scs_cones` if it exists
    #   Also make sure power cone has positive exponent
    # jl_cones = deepcopy(scs_cones)
    # jl_cones['f'] = jl_cones.pop('z')
    # jl.cones = jl_cones

    # zero_cone = scs_cones['z']
    # nonneg_cone = scs_cones['l']
    # soc = scs_cones['q']
    # psd_cone = scs_cones['s']
    # exp_cones = scs_cones['ep']

    # jl.seval('''
    #    cones = Dict("f" => {zero_cone}, "l" => {nonneg_cone}, "q" => {soc}, "s" => {psd_cone}, "ep" => {exp_cones})
    #    settings = Clarabel.Settings(direct_solve_method = :cudss) 
    # ''')

    # Assign Python variables to Julia variables
    jl.zero_cone = scs_cones['z']
    jl.nonneg_cone = scs_cones['l']
    jl.soc = scs_cones['q']
    jl.psd_cone = scs_cones['s']
    jl.exp_cones = scs_cones['ep']

    # Now use those Julia variables in your Julia code
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
    result = jl.seval("methods(Clarabel.Solver)")
    print(result)
    print("-> after which call")
    jl.solver = jl.Clarabel.Solver(Pjl, qjl, Ajl, bjl, jl.cones, jl.settings)
    jl.Clarabel.solve_b(jl.solver) # solve new problem w/o creating memory


    # should be able to do Pjl = jl.Clarabel.cupy ...
    # then jl.solver = jl.Clarabel.solver(Pjl, ...)
    

    # target_x_qcp = to_tensor(qcp_data_and_soln[5], dtype=dtype, device=device)
    # target_y_qcp = to_tensor(qcp_data_and_soln[6], dtype=dtype, device=device)
    # target_s_qcp = to_tensor(qcp_data_and_soln[7], dtype=dtype, device=device)