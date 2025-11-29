from typing import Callable

import numpy as np
import cvxpy as cvx
from jax import vmap, jit
import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

import diffqcp.cones.canonical as cone_lib
from diffqcp.cones.exp import in_exp, in_exp_dual, ExponentialConeProjector
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
    soc_projector = eqx.filter_jit(_soc_projector)
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
    soc_projector = eqx.filter_jit(_soc_projector)
    batched_soc_projector = eqx.filter_jit(eqx.filter_vmap(_soc_projector))
    
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


def _proj_psd_via_cvxpy(x: np.ndarray) -> np.ndarray:
    """Project vectorized symmetric matrix x onto the PSD cone using CVXPY."""
    size = cone_lib.symm_dim_to_size(x.size)
    X = np.zeros((size, size), dtype=x.dtype)
    idxs = np.triu_indices(size)
    sqrt2 = np.sqrt(2.0)
    X[idxs] = x / sqrt2
    X = X + X.T
    diag = np.arange(size)
    X[diag, diag] /= sqrt2

    z = cvx.Variable((size, size), PSD=True)
    objective = cvx.Minimize(cvx.sum_squares(z - X))
    prob = cvx.Problem(objective)
    prob.solve(solver="SCS", eps=1e-10)
    Z_val = z.value
    vec = Z_val[idxs]
    off_diag = idxs[0] != idxs[1]
    vec[off_diag] *= sqrt2
    return vec


def _test_psd_projector(sizes, num_batches, keyfunc):
    total_size = sum([cone_lib.symm_size_to_dim(s) for s in sizes])

    _psd_projector = cone_lib.PSDConeProjector(sizes=sizes)
    psd_projector = eqx.filter_jit(_psd_projector)
    batched_psd_projector = eqx.filter_jit(eqx.filter_vmap(_psd_projector))
    
    for _ in range(10):
        x_jnp = jr.normal(keyfunc(), total_size)
        x_np = np.array(x_jnp)
        start = 0
        solns = []
        for size in sizes:
            end = start + cone_lib.symm_size_to_dim(size)
            solns.append(jnp.array(_proj_psd_via_cvxpy(x_np[start:end])))
            start = end
        proj_x_solver = jnp.concatenate(solns)
        proj_x, _ = psd_projector(x_jnp)
        assert tree_allclose(proj_x, proj_x_solver)

        # --- batched ---
        x_jnp = jr.normal(keyfunc(), (num_batches, total_size))
        x_np = np.array(x_jnp)
        proj_x, _ = batched_psd_projector(x_jnp)
        for i in range(num_batches):
            start = 0
            solns = []
            for size in sizes:
                end = start + cone_lib.symm_size_to_dim(size)
                solns.append(jnp.array(_proj_psd_via_cvxpy(x_np[i, start:end])))
                start = end
            proj_x_solver = jnp.concatenate(solns)
            assert tree_allclose(proj_x[i, :], proj_x_solver)


def test_psd_projector_simple(getkey):
    sizes = [3, 4, 10]
    num_batches = 5
    _test_psd_projector(sizes, num_batches, getkey)

def test_psd_projector_hard(getkey):
    sizes = [2, 3, 3, 4, 4, 2]
    num_batches = 5
    _test_psd_projector(sizes, num_batches, getkey)

def _proj_pow_via_cvxpy(x: np.ndarray, alphas: list[float]) -> np.ndarray:
    """Project x onto the product of 3D power cones with given alphas using CVXPY."""
    n = len(x)
    assert n % 3 == 0
    num_cones = n // 3
    var = cvx.Variable(n)
    constraints = []
    for i in range(num_cones):
        constraints.append(cvx.PowCone3D(var[3*i], var[3*i+1], var[3*i+2], alphas[i]))
    objective = cvx.Minimize(cvx.sum_squares(var - x))
    prob = cvx.Problem(objective, constraints)
    prob.solve(solver="SCS", eps=1e-10)
    return np.array(var.value)

def test_proj_pow():
    np.random.seed(0)
    n = 3
    alphas = np.random.uniform(low=0, high=1, size=15)
    for alpha in alphas:
        x = np.random.randn(n)
        proj_cvx = _proj_pow_via_cvxpy(x, [alpha])
        projector = cone_lib.PowerConeProjector([alpha], onto_dual=False)
        # this is not efficient since recompiling; just doing for testing.
        proj_jax, _ = eqx.filter_jit(projector)(jnp.array(x))
        proj_jax = np.array(proj_jax)
        print("proj_jax: ", proj_jax)
        print("proj_cvx: ", proj_cvx)
        assert np.allclose(proj_jax, proj_cvx, atol=1e-6, rtol=1e-7)

# def test_proj_pow_diffcpish():
#     # TODO(quill): test itself needs fixing
#     np.random.seed(0)
#     alphas1 = np.random.uniform(low=0.01, high=1, size=15)
#     alphas2 = np.random.uniform(low=0.01, high=1, size=15)
#     alphas3 = np.random.uniform(low=0.01, high=1, size=15)
#     for i in range(alphas1.shape[0]):
#         x = np.random.randn(9)
#         # primal
#         proj_cvx = _proj_pow_via_cvxpy(x, [alphas1[i], alphas2[i], alphas3[i]])
#         projector = cone_lib.PowerConeProjector([alphas1[i], alphas2[i], alphas3[i]], onto_dual=False)
#         proj_jax, _ = projector(jnp.array(x))
#         assert np.allclose(np.array(proj_jax), proj_cvx, atol=1e-4, rtol=1e-7)
#         # dual
#         proj_dual = cone_lib.PowerConeProjector([-alphas1[i], -alphas2[i], -alphas3[i]], onto_dual=False)
#         proj_cvx_dual = _proj_pow_via_cvxpy(-x, [-alphas1[i], -alphas2[i], -alphas3[i]])
#         proj_jax_dual, _ = proj_dual(jnp.array(x))
#         # Moreau: Pi_K^*(v) = v + Pi_K(-v)
#         assert np.allclose(np.array(proj_jax_dual), x + proj_cvx_dual, atol=1e-4)

def test_proj_pow_specific():
    n = 3
    x = np.array([1., 2., 3.])
    alpha = 0.6
    proj_cvx = _proj_pow_via_cvxpy(x, [alpha])
    projector = cone_lib.PowerConeProjector([alpha], onto_dual=False)
    proj_jax, _ = projector(jnp.array(x))
    proj_jax = np.array(proj_jax)
    print("proj_jax: ", proj_jax)
    print("proj_cvx:", proj_cvx)
    assert np.allclose(np.array(proj_jax), proj_cvx, atol=1e-6, rtol=1e-7)


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
    nn_projector = eqx.filter_jit(_nn_projector)
    batched_nn_projector = eqx.filter_jit(eqx.filter_vmap(_nn_projector))

    _soc_projector = cone_lib.SecondOrderConeProjector(dims=soc_dims)
    soc_projector = eqx.filter_jit(_soc_projector)
    batched_soc_projector = eqx.filter_jit(eqx.filter_vmap(_soc_projector))
    
    for dual in [True, False]:

        _zero_projector = cone_lib.ZeroConeProjector(onto_dual=dual)
        zero_projector = eqx.filter_jit(_zero_projector)
        batched_zero_projector = eqx.filter_jit(eqx.filter_vmap(_zero_projector))

        _cone_projector = cone_lib.ProductConeProjector(cones, onto_dual=dual)
        cone_projector = eqx.filter_jit(_cone_projector)
        batched_cone_projector = eqx.filter_jit(eqx.filter_vmap(_cone_projector))
    
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


def test_in_exp(getkey):
    in_vecs = [[0., 0., 1.], [-1., 0., 0.], [1., 1., 5.]]
    for vec in in_vecs:
        assert in_exp(jnp.array(vec))
    not_in_vecs = [[1., 0., 0.], [-1., -1., 1.], [-1., 0., -1.]]
    for vec in not_in_vecs:
        assert not in_exp(jnp.array(vec))


def test_in_exp_dual(getkey):
    in_vecs = [[0., 1., 1.], [-1., 1., 5.]]
    not_in_vecs = [[0., -1., 1.], [0., 1., -1.]]
    for vec in in_vecs:
        arr = jnp.array(vec)
        assert in_exp_dual(arr)
    for vec in not_in_vecs:
        arr = jnp.array(vec)
        assert not in_exp_dual(vec)

    
def test_proj_exp_scs(getkey):
    """test values ported from scs/test/problems/test_exp_cone.h
    """
    TOL = 1e-6

    vs = [jnp.array([1.0, 2.0, 3.0]),
          jnp.array([0.14814832, 1.04294573, 0.67905585]),
          jnp.array([-0.78301134, 1.82790084, -1.05417044]),
          jnp.array([1.3282585, -0.43277314, 1.7468072]),
          jnp.array([0.67905585, 0.14814832, 1.04294573]),
          jnp.array([0.50210027, 0.12314491, -1.77568921])]
    
    vp_true = [jnp.array([0.8899428, 1.94041881, 3.06957226]),
               jnp.array([-0.02001571, 0.8709169, 0.85112944]),
               jnp.array([-1.17415616, 0.9567094, 0.280399]),
               jnp.array([0.53160512, 0.2804836, 1.86652094]),
               jnp.array([0.38322814, 0.27086569, 1.11482228]),
               jnp.array([0, 0, 0])]
    vd_true = [jnp.array([-0., 2., 3.]),
               jnp.array([-0., 1.04294573, 0.67905585]),
               jnp.array([-0.68541419, 1.85424082, 0.01685653]),
               jnp.array([-0.02277033, -0.12164823, 1.75085347]),
               jnp.array([-0., 0.14814832, 1.04294573]),
               jnp.array([-0., 0.12314491, -0.])]
    
    primal_projector = ExponentialConeProjector(1, onto_dual=False)
    dual_projector = ExponentialConeProjector(1, onto_dual=True)
    
    for i in range(len(vs)):
        v = vs[i]
        vp, _ = primal_projector(v)
        vd, _ = dual_projector(v)
        # assert jnp.allclose(vp, vp_true[i], atol=TOL)
        assert jnp.allclose(vp, vp_true[i])
        # assert jnp.allclose(vd, vd_true[i], atol=TOL)
        assert jnp.allclose(vd, vd_true[i])

    # Now test batched
    vps, _ = vmap(primal_projector)(jnp.array(vs))
    vds, _ = vmap(dual_projector)(jnp.array(vs))

    for i in range(len(vs)):
        assert jnp.allclose(vps[i, :], vp_true[i])
        assert jnp.allclose(vds[i, :], vd_true[i])

    # now test integratated into diffqcp
    # vs = torch.cat(vs)
    # vp_true = torch.cat(vp_true)
    # vd_true = torch.cat(vd_true)
    # p = cone_lib.proj(vs, cones=[(cone_lib.EXP, 6)], dual=False)
    # pd = cone_lib.proj(vs, cones=[(cone_lib.EXP_DUAL, 6)], dual=False)
    # assert torch.allclose(p, vp_true, atol=TOL)
    # assert torch.allclose(pd, vd_true, atol=TOL)
    # p = cone_lib.proj(vs, cones=[(cone_lib.EXP_DUAL, 6)], dual=True)
    # pd = cone_lib.proj(vs, cones=[(cone_lib.EXP, 6)], dual=True)
    # assert torch.allclose(p, vp_true, atol=TOL)
    # assert torch.allclose(pd, vd_true, atol=TOL)