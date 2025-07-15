from typing import Callable

import numpy as np
import cvxpy as cvx
from jax import vmap, jit
import jax.numpy as jnp
import jax.random as jr

import diffqcp.cones.canonical as cone_lib
from .helpers import tree_allclose

def _test_dproj_finite_diffs(
    projection_func: Callable, key_func, dim: int, num_batches: int = 0
):
    if num_batches > 0:
        x = jr.normal(key_func(), (num_batches, dim))
        dx = jr.normal(key_func(), (num_batches, dim))
        _projector = jit(vmap(projection_func))
    else:
        x = jr.normal(key_func(), dim)
        dx = jr.normal(key_func(), dim)
        _projector = projection_func

    dx = 1e-6 * dx

    proj_x, dproj_x = _projector(x)
    proj_x_plus_dx, _ = _projector(x + dx)
    
    dproj_x_fd = proj_x_plus_dx - proj_x
    dproj_x_dx = dproj_x.mv(dx)

    assert tree_allclose(dproj_x_dx, dproj_x_fd)
    

def test_zero_projector(getkey):
    n = 100
    num_batches = 10

    for dual in [True, False]:

        _zero_projector = cone_lib.ZeroConeProjector(is_dual=dual)
        zero_projector = jit(_zero_projector)
        batched_zero_projector = jit(vmap(_zero_projector))

        for _ in range(15):
            
            x = jr.normal(getkey(), n)
            
            proj_x, _ = zero_projector(x)
            truth = jnp.zeros_like(x) if not dual else x
            assert tree_allclose(truth, proj_x)
            _test_dproj_finite_diffs(zero_projector, getkey, dim=n, num_batches=0)

            # --- vmap ---
            x = jr.normal(getkey(), (num_batches, n))
            proj_x, _ = batched_zero_projector(x)
            truth = jnp.zeros_like(x) if not dual else x
            assert tree_allclose(truth, proj_x)
            _test_dproj_finite_diffs(_zero_projector, getkey, dim=n, num_batches=num_batches)


def test_nonnegative_projector(getkey):
    n = 100
    num_batches = 10

    _nn_projector = cone_lib.NonnegativeConeProjector()
    nn_projector = jit(_nn_projector)
    batched_nn_projector = jit(vmap(_nn_projector))

    for _ in range(15):

        x = jr.normal(getkey(), n)
        proj_x, _ = nn_projector(x)
        truth = jnp.maximum(x, 0)
        assert tree_allclose(truth, proj_x)
        _test_dproj_finite_diffs(nn_projector, getkey, dim=n, num_batches=0)
        
        x = jr.normal(getkey(), (num_batches, n))
        proj_x, _ = batched_nn_projector(x)
        truth = jnp.maximum(x, 0)
        assert tree_allclose(truth, proj_x)
        _test_dproj_finite_diffs(_nn_projector, getkey, dim=n, num_batches=10)


def test_soc_private_projector(getkey):
    n = 100
    num_batches = 10

    _soc_projector = cone_lib._SecondOrderConeProjector(dim=n)
    soc_projector = jit(_soc_projector)

    for _ in range(15):
        x_jnp = jr.normal(getkey(), n)
        x_np = np.array(x_jnp)
        z = cvx.Variable(n)
        objective = cvx.Minimize(cvx.sum_squares(z - x_np))
        constraints = [cvx.norm(z[1:], 2) <= z[0]]
        prob = cvx.Problem(objective, constraints)
        prob.solve(solver=cvx.CLARABEL)
        z_star_jnp = jnp.array(z.value)
        
        proj_x, _ = soc_projector(x_jnp)
        assert tree_allclose(proj_x, z_star_jnp)
        _test_dproj_finite_diffs(soc_projector, getkey, dim=n, num_batches=0)

        # NOTE(quill): to test a `batched_soc_projector`'s projecting functionality,
        #   will have to loop through the batched `x` and use each to solve the `cvxpy`
        #   problem. 

        _test_dproj_finite_diffs(_soc_projector, getkey, dim=n, num_batches=num_batches)


# def test_soc_projector(getkey):
#     dims = [10, 15, 30]
#     total_dim = sum(dims)
    
#     for _ in range(15):
#         x_jnp = jr.normal(getkey(), total_dim)
#         x_np = np.array(x_jnp)
#         z = cvx.Variable(total_dim)
#         z1, x1 = z[0:dims[0]], x_np[0:dims[0]]
#         z2, x2 = z[dims[0]:dims[1]], x_np[dims[0]:dims[1]]
#         z3, x3 = z[dims[1]:], x_np[dims[1]:]
#         objective = cvx.Minimize(cvx.sum_squares(z1 - x1) + cvx.sum_squares(z2 - x2)
#                                  + cvx.sum_squares(z3 - x3))
#         constraints = [cvx.norm(z1[1:], 2) <= z1[0],
#                        cvx.norm(z2[1:], 2) <= z2[0],
#                        cvx.norm(z3[1:], 3) <= z3[0]]
#         prob = cvx.Problem(objective, constraints)
#         prob.solve(solver=cvx.CLARABEL)
#         z_star_jnp = jnp.array(z.value)

#         soc_projector = cone_lib.SecondOrderConeProjector(dims=dims)
#         proj_x, _ = soc_projector(x_jnp)
#         assert tree_allclose(proj_x, z_star_jnp)