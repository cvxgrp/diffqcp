from juliacall import Main as jl
# jl.seval('import Pkg; Pkg.develop(url="https://github.com/cvxgrp/CuClarabel.git")')
jl.seval('using Clarabel, LinearAlgebra, SparseArrays')
# jl.seval('Pkg.add("CUDA")')
jl.seval('using CUDA, CUDA.CUSPARSE')

# TODO(quill): set JAX flags

import time
import os
from typing import Callable
import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import lineax as lx
from jaxtyping import Float, Array
import cupy as cp

from diffqcp.qcp import DeviceQCP, HostQCP

# what auxillary objects can I create to store the CuPy <-> Julia objects?

def compute_loss(target_x, target_y, target_s, x, y, s):
    return (0.5 * la.norm(x - target_x)**2 + 0.5 * la.norm(y - target_y)**2
            + 0.5 * la.norm(s - target_s)**2)

@jax.jit
def make_step(qcp: DeviceQCP, target_x, target_y, target_s, step_size):
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
        qcp = DeviceQCP()
        loss, *dtheta = make_step(qcp, target_x, target_y, target_s, step_size)
        # now update your Julia data
        #   need handle case where P is all zeros

        
        
if __name__ == "__main__":
    pass