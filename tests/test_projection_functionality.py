"""
Ported (projection-related) tests from diffcp/tests/test_cone_prog_diff.py
"""

import cvxpy as cp
import numpy as np
import torch

import diffqcp.cones as cone_lib
import diffqcp.utils as utils

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
    torch.allclose(cone_lib.unvec_symm(cone_lib.vec_symm(x), n), x)


def test_vec_symm():
    rng = torch.Generator().manual_seed(0)
    n = 5
    x = torch.randn(cone_lib.symm_size_to_dim(n), generator=rng)
    np.testing.assert_allclose(cone_lib.vec_symm(cone_lib.unvec_symm(x, n)), x)


def test_proj_zero():
    """Test projection onto zero cone and its dual (the free cone).
    """
    rng = torch.Generator().manual_seed(0)
    n = 100
    for _ in range(10):
        x = torch.randn(n, generator=rng)
        assert torch.allclose(x, cone_lib._proj(x, cone_lib.ZERO, dual=True))
        assert torch.allclose(torch.zeros(n),
                              cone_lib._proj(x, cone_lib.ZERO, dual=False))
        

def test_proj_pos():
    """Test projection onto nonnegative cone, which is self-dual.
    """
    rng = torch.Generator().manual_seed(0)
    n = 100
    for _ in range(15):
        x = torch.randn(n, generator=rng)
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
        np.testing.assert_allclose(
            p, z_star_tch)
        np.testing.assert_allclose(
            p, cone_lib._proj(x_tch, cone_lib.SOC, dual=True))
        

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
        cones = [(cone_lib.ZERO, zero_dim), (cone_lib.POS, pos_dim),
                 (cone_lib.SOC, soc_dim)]
        size = zero_dim + pos_dim + sum(soc_dim)
        x = np.random.randn(size)
        x = torch.randn(size, generator=rng)
        for dual in [False, True]:
            proj = cone_lib.proj(x, cones, dual=dual)

            offset = 0
            torch.allclose(proj[:zero_dim], cone_lib._proj(x[:zero_dim], cone_lib.ZERO, dual=dual))
            offset += zero_dim

            torch.allclose(proj[offset:offset + pos_dim], cone_lib._proj(x[offset:offset + pos_dim], cone_lib.POS,
                                                                        dual=dual))
            offset += pos_dim

            for dim in soc_dim:
                torch.allclose(proj[offset:offset + dim], cone_lib._proj(x[offset:offset + dim], cone_lib.SOC,
                                                          dual=dual))
                offset += dim