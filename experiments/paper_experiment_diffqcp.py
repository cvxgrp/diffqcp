from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/cvxgrp/CuClarabel.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

import time
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

    # m = n = 10_000
    # m = n = 1000
    m = n = 250
    dtype = torch.float64
    device = torch.device('cuda')

    print("starting to build problem")

    target_problem = generate_group_lasso(n=n, m=m)
    # initial_problem = generate_group_lasso(n=n, m=m)

    print("built problem")

    print("extracting data from problem")
    qcp_data = data_from_cvxpy_problem_quad(target_problem)
    print("finished extracting data from problem")

    Pcpu, Acpu = qcp_data[0], qcp_data[2] # note that we take the full P
    qcpu, bcpu = qcp_data[3], qcp_data[4]
    scs_cones, clarabel_cones = qcp_data[5], qcp_data[6]

    print("Pcpu nnz: ", Pcpu.nnz)

    P = to_sparse_csr_tensor(Pcpu, dtype=dtype, device=device)
    A = to_sparse_csr_tensor(Acpu, dtype=dtype, device=device)
    q = to_tensor(qcpu, dtype=dtype, device=device)
    b = to_tensor(bcpu,dtype=dtype, device=device)

    # Pcupy = torch_csr_to_cupy_csr(P) # need to handle when P is all zeros
    Acupy = torch_csr_to_cupy_csr(A)

    # TODO (quill): the following fail
    # assert Pcupy.indptr.__cuda_array_interface__['data'][0] == P.crow_indices().__cuda_array_interface__['data'][0]
    # assert Acupy.indptr.__cuda_array_interface__['data'][0] == A.crow_indices().__cuda_array_interface__['data'][0]
    
    # Pcupy = cp.asarray(P)
    # Acupy = cp.asarray()
    qcupy = cp.asarray(q)
    # the following passes
    assert qcupy.__cuda_array_interface__['data'][0] == q.__cuda_array_interface__['data'][0]
    # bcupy

    # bjl = jl.Clarabel.cupy_to_cuvector(jl.Float64, int(bpy.))


    # target_x_qcp = to_tensor(qcp_data_and_soln[5], dtype=dtype, device=device)
    # target_y_qcp = to_tensor(qcp_data_and_soln[6], dtype=dtype, device=device)
    # target_s_qcp = to_tensor(qcp_data_and_soln[7], dtype=dtype, device=device)