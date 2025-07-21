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
        # NOTE(quill): `jit`ing the following slows the check down
        #   since this is called in a loop, so we end up `jit`ing multiple times.
        #   Just doing it here to ensure it works.
        _projector = jit(vmap(projection_func))
    else:
        x = jr.normal(key_func(), dim)
        dx = jr.normal(key_func(), dim)
        _projector = projection_func

    dx = 1e-5 * dx

    proj_x, dproj_x = _projector(x)
    proj_x_plus_dx, _ = _projector(x + dx)
    
    dproj_x_fd = proj_x_plus_dx - proj_x    
    dproj_x_dx = dproj_x.mv(dx)
    assert dproj_x_dx is not None
    assert dproj_x_fd is not None
    assert tree_allclose(dproj_x_dx, dproj_x_fd)
    

def test_zero_projector(getkey):
    n = 100
    num_batches = 10

    for dual in [True, False]:

        _zero_projector = cone_lib.ZeroConeProjector(onto_dual=dual)
        zero_projector = jit(_zero_projector)
        batched_zero_projector = jit(vmap(_zero_projector))

        for _ in range(15):
            
            x = jr.normal(getkey(), n)
            
            proj_x, _ = zero_projector(x)
            truth = jnp.zeros_like(x) if not dual else x
            assert tree_allclose(truth, proj_x)
            _test_dproj_finite_diffs(zero_projector, getkey, dim=n, num_batches=0)

            # --- batched ---
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


def _proj_soc_via_cvxpy(x: np.ndarray) -> np.ndarray:
    n = x.size
    z = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum_squares(z - x))
    constraints = [cvx.norm(z[1:], 2) <= z[0]]
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver=cvx.SCS, eps=1e-10)
    return z.value


def test_soc_private_projector(getkey):
    n = 100
    num_batches = 10

    _soc_projector = cone_lib._SecondOrderConeProjector(dim=n)
    soc_projector = jit(_soc_projector)
    batched_soc_projector = jit(vmap(_soc_projector))

    for _ in range(15):
        x_jnp = jr.normal(getkey(), n)
        x_np = np.array(x_jnp)
        proj_x_solver = jnp.array(_proj_soc_via_cvxpy(x_np))
        
        proj_x, _ = soc_projector(x_jnp)
        assert tree_allclose(proj_x, proj_x_solver)
        _test_dproj_finite_diffs(soc_projector, getkey, dim=n, num_batches=0)

        # --- batched ---
        x_jnp = jr.normal(getkey(), (num_batches, n))
        x_np = np.array(x_jnp)
        proj_x, _ = batched_soc_projector(x_jnp)
        for i in range(num_batches):
            proj_x_solver = jnp.array(_proj_soc_via_cvxpy(x_np[i, :]))
            assert tree_allclose(proj_x[i, :], proj_x_solver)

        _test_dproj_finite_diffs(_soc_projector, getkey, dim=n, num_batches=num_batches)


def _test_soc_projector(dims, num_batches, keyfunc):
    total_dim = sum(dims)

    _soc_projector = cone_lib.SecondOrderConeProjector(dims=dims)
    soc_projector = jit(_soc_projector)
    batched_soc_projector = jit(vmap(_soc_projector))
    
    for _ in range(15):

        x_jnp = jr.normal(keyfunc(), total_dim)
        x_np = np.array(x_jnp)
        start = 0
        solns = []
        for dim in dims:
            end = start + dim
            solns.append(jnp.array(_proj_soc_via_cvxpy(x_np[start:end])))
            start = end
        proj_x_solver = jnp.concatenate(solns)
        proj_x, _ = soc_projector(x_jnp)
        assert tree_allclose(proj_x, proj_x_solver)
        _test_dproj_finite_diffs(soc_projector, keyfunc, dim=total_dim, num_batches=0)

        # --- batched ---
        x_jnp = jr.normal(keyfunc(), (num_batches, total_dim))
        x_np = np.array(x_jnp)
        proj_x, _ = batched_soc_projector(x_jnp)
        for i in range(num_batches):
            start = 0
            solns = []
            for dim in dims:
                end = start + dim
                solns.append(jnp.array(_proj_soc_via_cvxpy(x_np[i, start:end])))
                start = end
            proj_x_solver = jnp.concatenate(solns)
            assert tree_allclose(proj_x[i, :], proj_x_solver)

        _test_dproj_finite_diffs(_soc_projector, keyfunc, dim=total_dim, num_batches=num_batches)


def test_soc_projector_simple(getkey):
    dims = [10, 15, 30]
    num_batches = 10
    _test_soc_projector(dims, num_batches, getkey)


def test_soc_projector_hard(getkey):
    dims = [5, 5, 5, 3, 3, 4, 5, 2, 2]
    num_batches = 10
    _test_soc_projector(dims, num_batches, getkey)


def test_product_projector(getkey):
    """assumes that the other tests in this file pass."""
    zero_dim = 15
    nn_dim = 23
    soc_dims = [5, 5, 5, 3, 3, 4, 5, 2, 2]
    soc_total_dim = sum(soc_dims)
    total_dim = zero_dim + nn_dim + soc_total_dim
    num_batches = 10
    cones = {
        cone_lib.ZERO : zero_dim,
        cone_lib.NONNEGATIVE: nn_dim,
        cone_lib.SOC : soc_dims
    }

    _nn_projector = cone_lib.NonnegativeConeProjector()
    nn_projector = jit(_nn_projector)
    batched_nn_projector = jit(vmap(_nn_projector))

    _soc_projector = cone_lib.SecondOrderConeProjector(dims=soc_dims)
    soc_projector = jit(_soc_projector)
    batched_soc_projector = jit(vmap(_soc_projector))
    
    for dual in [True, False]:

        _zero_projector = cone_lib.ZeroConeProjector(onto_dual=dual)
        zero_projector = jit(_zero_projector)
        batched_zero_projector = jit(vmap(_zero_projector))

        _cone_projector = cone_lib.ProductConeProjector(cones, onto_dual=dual)
        cone_projector = jit(_cone_projector)
        batched_cone_projector = jit(vmap(_cone_projector))
    
        for _ in range(15):
            x = jr.normal(getkey(), total_dim)
            proj_x, _ = cone_projector(x)
            proj_x_zero, _ = zero_projector(x[0:zero_dim])
            proj_x_nn, _ = nn_projector(x[zero_dim:zero_dim+nn_dim])
            proj_x_soc, _ = soc_projector(x[zero_dim+nn_dim:zero_dim+nn_dim+soc_total_dim])
            proj_x_handmade = jnp.concatenate([proj_x_zero,
                                               proj_x_nn,
                                               proj_x_soc])
            assert tree_allclose(proj_x, proj_x_handmade)
            _test_dproj_finite_diffs(cone_projector, getkey, dim=total_dim, num_batches=0)

            # --- batched ---
            x = jr.normal(getkey(), (num_batches, total_dim))
            proj_x, _ = batched_cone_projector(x)
            proj_x_zero, _ = batched_zero_projector(x[:, 0:zero_dim])
            proj_x_nn, _ = batched_nn_projector(x[:, zero_dim:zero_dim+nn_dim])
            proj_x_soc, _ = batched_soc_projector(x[:, zero_dim+nn_dim:zero_dim+nn_dim+soc_total_dim])
            proj_x_handmade = jnp.concatenate([proj_x_zero,
                                               proj_x_nn,
                                               proj_x_soc], axis=-1)
            assert tree_allclose(proj_x, proj_x_handmade)
            _test_dproj_finite_diffs(cone_projector, getkey, dim=total_dim, num_batches=num_batches)


