"""
Projection (and projection JVP) tests.

More specifically, this file contains tests for
1. cone utility functions (all related to PSD projections),
2. the projections onto cones,
3. and the derivatives of projections onto cones.

Lots of ported tests from diffcp/tests/test_cone_prog_diff.py.

# TODO: add functionality to test these on device
"""
import cvxpy as cp
import numpy as np
import torch

import diffqcp.cones as cone_lib
from diffqcp.cone_derivs import _dprojection, dprojection
from diffqcp.pow_cone import proj_power_cone
from diffqcp.exp_cone import proj_exp_cone, in_exp, in_exp_dual
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


def test_in_exp():
    in_vecs = [[0, 0, 1], [-1, 0, 0], [1, 1, 5]]
    for vec in in_vecs:
        tensor = utils.to_tensor(vec, dtype=torch.float64)
        assert in_exp(tensor)
    not_in_vecs = [[1, 0, 0], [-1, -1, 1], [-1, 0, -1]]
    for vec in not_in_vecs:
        tensor = utils.to_tensor(vec, dtype=torch.float64)
        assert not in_exp(tensor)


def test_in_exp_dual():
    in_vecs = [[0, 1, 1], [-1, 1, 5]]
    not_in_vecs = [[0, -1, 1], [0, 1, -1]]
    for vec in in_vecs:
        tensor = utils.to_tensor(vec, dtype=torch.float64)
        assert in_exp_dual(tensor)
    for vec in not_in_vecs:
        tensor = utils.to_tensor(vec, dtype=torch.float64)
        assert not in_exp_dual(tensor)

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
        constraints = [cp.PowCone3D(z[0], z[1], z[2], alpha)]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS", eps=1e-10)
        z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
        p = cone_lib.proj(x_tch, cones=[(cone_lib.POW, [alpha])])
        assert torch.allclose(p, z_star_tch)


def test_proj_pow_diffcpish():
    """Modified from the exp cone test in diffcp.
    """
    np.random.seed(0)
    n = 3
    alphas1 = np.random.uniform(low=0, high=1, size=15)
    alphas2 = np.random.uniform(low=0, high=1, size=15)
    alphas3 = np.random.uniform(low=0, high=1, size=15)
    for i in range(alphas1.shape[0]):
        x = np.random.randn(9)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        var = cp.Variable(9)
        constr = [cp.PowCone3D(var[0], var[1], var[2], alphas1[i])]
        constr += [cp.PowCone3D(var[3], var[4], var[5], alphas2[i])]
        constr += [cp.PowCone3D(var[6], var[7], var[8], alphas3[i])]
        obj = cp.Minimize(cp.norm(var[0:3] - x[0:3]) +
                          cp.norm(var[3:6] - x[3:6]) +
                          cp.norm(var[6:9] - x[6:9]))
        prob = cp.Problem(obj, constr)
        prob.solve(solver="SCS")
        var_star = utils.to_tensor(var.value, dtype=torch.float64)
        p = cone_lib.proj(x_tch,
                          [(cone_lib.POW, [alphas1[i], alphas2[i], alphas3[i]])],
                          dual=False)
        assert torch.allclose(p, var_star, atol=1e-4, rtol=1e-7)

        var = cp.Variable(9)
        constr = [cp.PowCone3D(var[0], var[1], var[2], alphas1[i])]
        constr += [cp.PowCone3D(var[3], var[4], var[5], alphas2[i])]
        constr += [cp.PowCone3D(var[6], var[7], var[8], alphas3[i])]
        obj = cp.Minimize(cp.norm(var[0:3] + x[0:3]) +
                          cp.norm(var[3:6] + x[3:6]) +
                          cp.norm(var[6:9] + x[6:9]))
        prob = cp.Problem(obj, constr)
        prob.solve(solver="SCS")
        var_star = utils.to_tensor(var.value, dtype=torch.float64)
        p_dual = cone_lib.proj(x_tch,
                          [(cone_lib.POW, [-alphas1[i], -alphas2[i], -alphas3[i]])],
                          dual=False)
        assert torch.allclose(p_dual, x_tch + var_star, atol=1e-4)


def test_proj_pow_specific():
    n = 3
    x = np.array([1, 2, 3])
    x_tch = utils.to_tensor(x, dtype=torch.float64)
    alpha = 0.6
    z = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(z - x))
    constrs = [cp.PowCone3D(*z, alpha)]
    prob = cp.Problem(obj, constrs)
    prob.solve(solver="SCS")
    z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
    p = cone_lib.proj(x_tch, cones=[(cone_lib.POW, [alpha])])
    assert torch.allclose(p, z_star_tch)


def test_proj_exp():
    """Test projection onto EXP cone, which is not self-dual.
    """
    np.random.seed(0)
    n = 3
    for _ in range(15):
        x = np.random.randn(n)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(z - x))
        constraints = [cp.ExpCone(*z)]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS")
        z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
        p = cone_lib._proj(x_tch, cone=cone_lib.EXP, dual=False)
        assert torch.allclose(p, z_star_tch, atol=1e-4)


def test_proj_exp_dual():
    """Test projection onto EXP cone, which is not self-dual.
    """
    np.random.seed(0)
    n = 3
    for _ in range(15):
        x = np.random.randn(n)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        z = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(x + z))
        constraints = [cp.ExpCone(*z)]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver="SCS")
        z_star_tch = utils.to_tensor(z.value, dtype=torch.float64)
        # p = cone_lib._proj(x_tch, cone=cone_lib.EXP_DUAL, dual=False)
        p = proj_exp_cone(x_tch, primal=False)
        assert in_exp_dual(p), "p not in dual"
        assert in_exp_dual(z_star_tch + x_tch)
        assert torch.allclose(p, z_star_tch + x_tch, atol=1e-4)


def test_proj_exp_scs():
    """test values ported from scs/test/problems/test_exp_cone.h
    """
    TOL = torch.tensor(1e-6, dtype=torch.float64)

    vs = [torch.tensor([1, 2, 3], dtype=torch.float64),
          torch.tensor([0.14814832, 1.04294573, 0.67905585], dtype=torch.float64),
          torch.tensor([-0.78301134, 1.82790084, -1.05417044], dtype=torch.float64),
          torch.tensor([1.3282585, -0.43277314, 1.7468072], dtype=torch.float64),
          torch.tensor([0.67905585, 0.14814832, 1.04294573], dtype=torch.float64),
          torch.tensor([0.50210027, 0.12314491, -1.77568921], dtype=torch.float64)]
    
    vp_true = [torch.tensor([0.8899428, 1.94041881, 3.06957226], dtype=torch.float64),
               torch.tensor([-0.02001571, 0.8709169, 0.85112944], dtype=torch.float64),
               torch.tensor([-1.17415616, 0.9567094, 0.280399], dtype=torch.float64),
               torch.tensor([0.53160512, 0.2804836, 1.86652094], dtype=torch.float64),
               torch.tensor([0.38322814, 0.27086569, 1.11482228], dtype=torch.float64),
               torch.tensor([0, 0, 0], dtype=torch.float64)]
    vd_true = [torch.tensor([-0., 2., 3.], dtype=torch.float64),
               torch.tensor([-0., 1.04294573, 0.67905585], dtype=torch.float64),
               torch.tensor([-0.68541419, 1.85424082, 0.01685653], dtype=torch.float64),
               torch.tensor([-0.02277033, -0.12164823, 1.75085347], dtype=torch.float64),
               torch.tensor([-0., 0.14814832, 1.04294573], dtype=torch.float64),
               torch.tensor([-0., 0.12314491, -0.], dtype=torch.float64)]
    
    for i in range(len(vs)):
        v = vs[i]
        vp = proj_exp_cone(v, primal=True)
        vd = proj_exp_cone(v, primal=False)
        assert torch.allclose(vp, vp_true[i], atol=TOL)
        assert torch.allclose(vd, vd_true[i], atol=TOL)

    # now test integratated into diffqcp
    vs = torch.cat(vs)
    vp_true = torch.cat(vp_true)
    vd_true = torch.cat(vd_true)
    p = cone_lib.proj(vs, cones=[(cone_lib.EXP, 6)], dual=False)
    pd = cone_lib.proj(vs, cones=[(cone_lib.EXP_DUAL, 6)], dual=False)
    assert torch.allclose(p, vp_true, atol=TOL)
    assert torch.allclose(pd, vd_true, atol=TOL)
    p = cone_lib.proj(vs, cones=[(cone_lib.EXP_DUAL, 6)], dual=True)
    pd = cone_lib.proj(vs, cones=[(cone_lib.EXP, 6)], dual=True)
    assert torch.allclose(p, vp_true, atol=TOL)
    assert torch.allclose(pd, vd_true, atol=TOL)


def test_proj_exp_diffcp():
    """port from diffcp.
    """
    np.random.seed(0)
    for _ in range(15):
        x = np.random.randn(9)
        x_tch = utils.to_tensor(x, dtype=torch.float64)
        var = cp.Variable(9)
        constr = [cp.constraints.ExpCone(var[0], var[1], var[2])]
        constr += [cp.constraints.ExpCone(var[3], var[4], var[5])]
        constr += [cp.constraints.ExpCone(var[6], var[7], var[8])]
        obj = cp.Minimize(cp.norm(var[0:3] - x[0:3]) +
                          cp.norm(var[3:6] - x[3:6]) +
                          cp.norm(var[6:9] - x[6:9]))
        prob = cp.Problem(obj, constr)
        prob.solve(solver="SCS", eps=1e-12, max_iters=10_000)
        var_star = utils.to_tensor(var.value, dtype=torch.float64)
        p = cone_lib.proj(x_tch, [(cone_lib.EXP, 3)], dual=False)
        assert torch.allclose(p, var_star, atol=1e-4, rtol=1e-7)

        var = cp.Variable(9)
        constr = [cp.constraints.ExpCone(var[0], var[1], var[2])]
        constr.append(cp.constraints.ExpCone(var[3], var[4], var[5]))
        constr.append(cp.constraints.ExpCone(var[6], var[7], var[8]))
        obj = cp.Minimize(cp.norm(var[0:3] + x[0:3]) +
                          cp.norm(var[3:6] + x[3:6]) +
                          cp.norm(var[6:9] + x[6:9]))
        prob = cp.Problem(obj, constr)
        prob.solve(solver="SCS", eps=1e-12)
        var_star = utils.to_tensor(var.value, dtype=torch.float64)
        p_dual = cone_lib.proj(x_tch, [(cone_lib.EXP_DUAL, 3)], dual=False)
        assert torch.allclose(p_dual, x_tch + var_star, atol=1e-6)


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
        pow_alpha_neg = [-alpha for alpha in pow_alpha]
        exp_dim = np.random.randint(3, 18)
        cones = [(cone_lib.POW, pow_alpha_neg), (cone_lib.ZERO, zero_dim),
                 (cone_lib.POS, pos_dim), (cone_lib.EXP, exp_dim),
                 (cone_lib.SOC, soc_dim), (cone_lib.POW, pow_alpha),
                 (cone_lib.PSD, psd_dim), (cone_lib.EXP_DUAL, exp_dim)]
        size = zero_dim + pos_dim + sum(soc_dim) + sum([cone_lib.symm_size_to_dim(d) for d in psd_dim])\
                + 2*3*len(pow_alpha) + 2*3*exp_dim
        x = torch.randn(size, generator=rng, dtype=torch.float64)
        for dual in [False, True]:
            proj = cone_lib.proj(x, cones, dual=dual)

            offset = 0
            for alpha in pow_alpha_neg:
                v = -x[offset:offset + 3]
                assert torch.allclose(proj[offset:offset+3],
                                      x[offset:offset+3]+ proj_power_cone(v,-alpha))
                offset += 3
            
            assert torch.allclose(proj[offset:zero_dim], cone_lib._proj(x[offset:zero_dim], cone_lib.ZERO, dual=dual))
            offset += zero_dim

            assert torch.allclose(proj[offset:offset + pos_dim], cone_lib._proj(x[offset:offset + pos_dim], cone_lib.POS,
                                                                        dual=dual))
            offset += pos_dim

            dim = 3 * exp_dim
            assert torch.allclose(proj[offset:offset+dim],
                                  cone_lib.proj(x[offset:offset+dim], [(cone_lib.EXP, exp_dim)], dual=dual))
            offset += dim
            
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

            dim = 3 * exp_dim
            assert torch.allclose(proj[offset:offset+dim],
                                  cone_lib.proj(x[offset:offset+dim], [(cone_lib.EXP_DUAL, exp_dim)], dual=dual))
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