"""
Ported (projection-related) tests from diffcp/tests/test_cone_prog_diff.py.

More specifically, this file contains tests for
1. cone utility functions (all related to PSD projections),
2. the projections onto cones,
3. and the derivatives of projections onto cones.
"""
import cvxpy as cp
import numpy as np
import torch

import diffqcp.cones as cone_lib
from diffqcp.cone_derivs import _dprojection, dprojection
from diffqcp.pow_cone import proj_power_cone
import diffqcp.utils as utils

# ==== SOME CONE UTILITY TESTS ====

def test_symm_size_to_dim():
    assert cone_lib.symm_size_to_dim(10) == (10) * (10 + 1) / 2


def test_psd_dim():
    n = 4096
    assert cone_lib.symm_dim_to_size(cone_lib.symm_size_to_dim(n)) == n


def test_unvec_symm():
    np.random.seed(0)
    n = 5
    x = np.random.randn(n, n)
    x = x + x.T
    x = utils.to_tensor(x, dtype=torch.float64)
    assert torch.allclose(cone_lib.unvec_symm(cone_lib.vec_symm(x), n), x)


def test_vec_symm():
    rng = torch.Generator().manual_seed(0)
    n = 5
    x = torch.randn(cone_lib.symm_size_to_dim(n), generator=rng, dtype=torch.float64)
    np.testing.assert_allclose(cone_lib.vec_symm(cone_lib.unvec_symm(x, n)), x)

# ==== CONE PROJECTION TESTS ====

def test_proj_zero():
    """Test projection onto zero cone and its dual (the free cone).
    """
    rng = torch.Generator().manual_seed(0)
    n = 100
    for _ in range(10):
        x = torch.randn(n, generator=rng, dtype=torch.float64)
        assert torch.allclose(x, cone_lib._proj(x, cone_lib.ZERO, dual=True))
        assert torch.allclose(torch.zeros(n, dtype=torch.float64),
                              cone_lib._proj(x, cone_lib.ZERO, dual=False))
        

def test_proj_pos():
    """Test projection onto nonnegative cone, which is self-dual.
    """
    rng = torch.Generator().manual_seed(0)
    n = 100
    for _ in range(15):
        x = torch.randn(n, generator=rng, dtype=torch.float64)
        p = cone_lib._proj(x, cone_lib.POS, dual=False)
        assert torch.allclose(p, torch.maximum(x, torch.tensor(0)))
        assert torch.allclose(cone_lib._proj(x, cone_lib.POS, dual=True), p) # self-dual


def test_proj_soc():
    """Test projection onto SOC, which is self-dual.
    """
    np.random.seed(0)
    n = 100
    for _ in range(15):
        x = np.random.randn(n)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [cp.norm(z[1:], 2) <= z[0]]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS", eps=1e-10)
        z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
        p = cone_lib._proj(x_tch, cone_lib.SOC, dual=False)
        assert torch.allclose(p, z_star_tch)
        assert torch.allclose(p, cone_lib._proj(x_tch, cone_lib.SOC, dual=True))
        

def test_proj_psd():
    """Test projection onto PSD cone, which is self-dual.
    """
    np.random.seed(0)
    n = 10
    for _ in range(15):
        x = np.random.randn(n, n)
        x = x + x.T
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        x_vec = cone_lib.vec_symm(x_tch)
        z = cp.Variable((n, n), PSD=True)
        objective = cp.Minimize(cp.sum_squares(z - x))
        prob = cp.Problem(objective)
        prob.solve(solver="SCS", eps=1e-10)
        z_val = utils.to_tensor(z.value, dtype=torch.float64)
        p = cone_lib.unvec_symm(
            cone_lib._proj(x_vec, cone_lib.PSD, dual=False), n)
        assert torch.allclose(p, z_val, atol=1e-5, rtol=1e-5)
        assert torch.allclose(p, cone_lib.unvec_symm(
            cone_lib._proj(x_vec, cone_lib.PSD, dual=True), n))
        

def test_proj_pow():
    """Test projection onto POW cone, which is not self-dual.
    """
    np.random.seed(0)
    n = 3
    alphas = np.random.uniform(low=0, high=1, size=15)
    for alpha in alphas:
        x = np.random.randn(n)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(z - x))
        # constraints = [z[0]**alpha * z[1]**(1-alpha) >= cp.abs(z[2])]
        constraints = [cp.PowCone3D(z[0], z[1], z[2], alpha)]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS", eps=1e-10)
        z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
        p = cone_lib.proj(x_tch, cones=[(cone_lib.POW, [alpha])])
        assert torch.allclose(p, z_star_tch)
        

def test_projection():
    """Test projection onto cartesian product of atom cones.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    for _ in range(10):
        zero_dim = np.random.randint(1, 10)
        pos_dim = np.random.randint(1, 10)
        soc_dim = [np.random.randint(1, 10) for _ in range(
            np.random.randint(1, 10))]
        psd_dim = [np.random.randint(1, 10) for _ in range(
            np.random.randint(1, 10))]
        pow_alpha = [np.random.uniform(0, 1) for _ in range(np.random.randint(1, 10))]
        cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                 (cone_lib.SOC, soc_dim), (cone_lib.POW, pow_alpha),
                 (cone_lib.PSD, psd_dim)]
        size = zero_dim + pos_dim + sum(soc_dim) + sum([cone_lib.symm_size_to_dim(d) for d in psd_dim])\
                + 3*len(pow_alpha)
        x = torch.randn(size, generator=rng, dtype=torch.float64)
        for dual in [False, True]:
            proj = cone_lib.proj(x, cones, dual=dual)

            offset = 0
            assert torch.allclose(proj[:zero_dim], cone_lib._proj(x[:zero_dim], cone_lib.ZERO, dual=dual))
            offset += zero_dim

            assert torch.allclose(proj[offset:offset + pos_dim], cone_lib._proj(x[offset:offset + pos_dim], cone_lib.POS,
                                                                        dual=dual))
            offset += pos_dim

            for dim in soc_dim:
                assert torch.allclose(proj[offset:offset + dim], cone_lib._proj(x[offset:offset + dim], cone_lib.SOC,
                                                          dual=dual))
                offset += dim

            for alpha in pow_alpha:
                assert torch.allclose(proj[offset:offset+3], proj_power_cone(x[offset:offset + 3], alpha))
                offset += 3
            
            for dim in psd_dim:
                dim = cone_lib.symm_size_to_dim(dim)
                assert torch.allclose(proj[offset:offset + dim], cone_lib._proj(x[offset:offset + dim], cone_lib.PSD,
                                                                        dual=dual))
                offset += dim

# ==== DERIVATIVES OF PROJECTIONS TESTS ====

def _test_Dproj(cone: str,
                n: int,
                rgen: torch.Generator,
                x: torch.Tensor = None,
                dual: bool = False,
                tol: float = 1e-8
) -> None:
    """
    Helper function to test Jacobian-vector products for cone projections.

    Note the function itself will make `assert` statements; nothing is returned.
    """
    if x is None:
        x = torch.randn(n, generator=rgen, dtype=torch.float64)
    dx = 1e-6 * torch.randn(n, generator=rgen, dtype=torch.float64)

    proj_x = cone_lib._proj(x, cone, dual=dual)

    dproj_x = cone_lib._proj(x + dx, cone, dual=dual) - proj_x
    Dproj = _dprojection(x, cone, dual=dual)

    assert torch.allclose(Dproj @ dx, dproj_x, atol=tol)


def test_dproj_zero():
    """JVPs for Dproj onto zero and free cones.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        dim = np.random.randint(25, 75)
        _test_Dproj(cone_lib.ZERO, dim, rng, dual=True)
        _test_Dproj(cone_lib.ZERO, dim, rng, dual=False)


def test_dproj_pos():
    """JVPs for Dproj onto nonnegative cone.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        dim = np.random.randint(25, 75)
        _test_Dproj(cone_lib.POS, dim, rng, dual=True)
        _test_Dproj(cone_lib.POS, dim, rng, dual=False)


def test_dproj_soc():
    """JVPs for Dproj onto the SOC.
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        dim = np.random.randint(25, 75)
        _test_Dproj(cone_lib.SOC, dim, rng, dual=True)
        _test_Dproj(cone_lib.SOC, dim, rng, dual=False)


def test_dproj_psd():
    """JVPs for Dproj onto the PSD cone.

    TODO: not passing
    """
    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        size = np.random.randint(5, 15)
        dim = cone_lib.symm_size_to_dim(size)
        _test_Dproj(cone_lib.PSD, dim, rng, dual=True)
        _test_Dproj(cone_lib.PSD, dim, rng, dual=False)


def test_dprojection():
    """Test projection onto cartesian product of cones.
    """
    np.random.seed(0)
    for _ in range(10):
        zero_dim = np.random.randint(1, 10)
        pos_dim = np.random.randint(1, 10)
        soc_dim = [np.random.randint(1, 10) for _ in range(
            np.random.randint(1, 10))]
        # psd_dim = [np.random.randint(1, 10) for _ in range(
        #     np.random.randint(1, 10))]
        cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                 (cone_lib.SOC, soc_dim)] # (cone_lib.PSD, psd_dim)

        size = zero_dim + pos_dim + sum(soc_dim)
        # + sum([cone_lib.vec_psd_dim(d) for d in psd_dim])
        
        x = torch.randn(size, dtype=torch.float64)

        for dual in [False, True]:
            proj_x = cone_lib.proj(x, cones, dual=dual)
            dx = 1e-6 * torch.randn(size, dtype=torch.float64)
            dproj_x = cone_lib.proj(x + dx, cones, dual=dual) - proj_x
            Dproj = dprojection(x, cones, dual)
            
            assert torch.allclose(Dproj @ dx, dproj_x, atol=1e-8)