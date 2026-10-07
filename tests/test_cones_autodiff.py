"""Cross-check analytical cone Jacobians against `jax.jvp`.

Each `AbstractConeProjector` subclass implements `proj_dproj(x) -> (proj_x,
dproj_op)`, where `dproj_op` is the *analytical* Jacobian of the projection
written by hand. The same Jacobian could in principle be produced by JAX's
own autodiff over `proj` alone — and *should* match it exactly anywhere the
projection is smooth.

This file verifies that match. Failures here mean the analytical Jacobian
disagrees with what JAX would compute by autodiff'ing `proj`, which is
almost certainly a sign error or formula mistake in the analytical write-up.
The existing `test_cone_projectors.py` covers a complementary check (FD on
`proj_x_plus_dx - proj_x` against `dproj.mv(dx)`); together they pin the
Jacobians from two independent angles.

Cones with kinks (SOC at `||z|| = ±t`, PSD at the eigenvalue-zero hyperplane,
EXP/POW on their boundary surfaces) are non-differentiable on a measure-zero
set; we sample interior/exterior points to avoid those. EXP and POW are
checked against central finite differences instead of `jax.jvp` (their
projector internals contain `jax.lax.cond` branches with `1/r0`-type
singularities that produce NaNs in the "unused" branch under autodiff); see
the bottom of this file.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from diffqcp.cones import canonical as cone_lib

ATOL = 1e-8
RTOL = 1e-6


def _proj_only(projector):
    """Lift a projector into a callable that returns just `proj_x` (for jax.jvp)."""
    return lambda x: projector.proj_dproj(x)[0]


def _check_jvp_matches_dproj(projector, x, dx):
    """Assert `dproj.mv(dx) ≈ jax.jvp(proj, (x,), (dx,))[1]`."""
    _, dproj = projector.proj_dproj(x)
    analytical = dproj.mv(dx)
    _, autodiff = jax.jvp(_proj_only(projector), (x,), (dx,))
    np.testing.assert_allclose(
        np.asarray(analytical), np.asarray(autodiff), atol=ATOL, rtol=RTOL
    )


# ─── zero cone ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("onto_dual", [False, True])
def test_zero_cone_jacobian_matches_jvp(getkey, onto_dual):
    projector = cone_lib.ZeroConeProjector(onto_dual=onto_dual)
    for _ in range(5):
        x = jr.normal(getkey(), (50,))
        dx = jr.normal(getkey(), (50,))
        _check_jvp_matches_dproj(projector, x, dx)


# ─── nonnegative cone ────────────────────────────────────────────────────


def test_nonnegative_cone_jacobian_matches_jvp(getkey):
    projector = cone_lib.NonnegativeConeProjector()
    for _ in range(5):
        # Sample away from zero (kink at x_i = 0); sign decides which side.
        x_raw = jr.normal(getkey(), (50,))
        x = jnp.where(jnp.abs(x_raw) < 1e-2, 1.0 * jnp.sign(x_raw), x_raw)
        dx = jr.normal(getkey(), (50,))
        _check_jvp_matches_dproj(projector, x, dx)


# ─── second-order cone ────────────────────────────────────────────────────


def _soc_interior_point(key, dim):
    """Sample a point strictly inside the second-order cone."""
    z = jr.normal(key, (dim - 1,))
    # Make t = ||z|| + 1 so we are clearly interior (||z|| < t).
    t = jnp.linalg.norm(z) + 1.0
    return jnp.concatenate([jnp.array([t]), z])


def _soc_exterior_point(key, dim):
    """Sample a point outside the second-order cone (and outside its negation)."""
    z = jr.normal(key, (dim - 1,))
    # Set t small so ||z|| > |t|.
    t = jnp.array(0.1)
    return jnp.concatenate([jnp.array([t]), z])


def test_soc_interior_jacobian_matches_jvp(getkey):
    """Interior of SOC: projector is the identity; Jacobian = I."""
    projector = cone_lib._SecondOrderConeProjector(dim=10)
    for _ in range(5):
        x = _soc_interior_point(getkey(), dim=10)
        dx = jr.normal(getkey(), (10,))
        _check_jvp_matches_dproj(projector, x, dx)


def test_soc_proj_case_jacobian_matches_jvp(getkey):
    """Strict projection case (||z|| > |t|, t arbitrary sign)."""
    projector = cone_lib._SecondOrderConeProjector(dim=10)
    for _ in range(5):
        x = _soc_exterior_point(getkey(), dim=10)
        dx = jr.normal(getkey(), (10,))
        _check_jvp_matches_dproj(projector, x, dx)


# ─── PSD cone ─────────────────────────────────────────────────────────────


def _psd_interior_vec(key, size):
    """Vectorised symmetric matrix that's strictly positive-definite."""
    A = jr.normal(key, (size, size))
    sym = A @ A.T + size * jnp.eye(size)  # diagonally dominant ⇒ PD
    return cone_lib.vec_symm(sym)


def _psd_negative_vec(key, size):
    """Vectorised symmetric matrix that's strictly negative-definite."""
    return -_psd_interior_vec(key, size)


def test_psd_interior_jacobian_matches_jvp(getkey):
    """Strictly PD point: projector = identity; Jacobian = I."""
    size = 4
    dim = cone_lib.symm_size_to_dim(size)
    projector = cone_lib._PSDConeProjector(size=size, dim=dim)
    for _ in range(3):
        x = _psd_interior_vec(getkey(), size)
        dx = jr.normal(getkey(), (dim,))
        _check_jvp_matches_dproj(projector, x, dx)


def test_psd_negative_jacobian_matches_jvp(getkey):
    """Strictly ND point: projector = 0; Jacobian = 0."""
    size = 4
    dim = cone_lib.symm_size_to_dim(size)
    projector = cone_lib._PSDConeProjector(size=size, dim=dim)
    for _ in range(3):
        x = _psd_negative_vec(getkey(), size)
        dx = jr.normal(getkey(), (dim,))
        _check_jvp_matches_dproj(projector, x, dx)


def _psd_mixed_vec(key, size, num_pos):
    """Vectorised symmetric matrix with `num_pos` positive and `size - num_pos`
    negative eigenvalues, all at least 0.5 away from zero (away from the kink)."""
    k1, k2, k3 = jr.split(key, 3)
    Q, _ = jnp.linalg.qr(jr.normal(k1, (size, size)))
    pos = 0.5 + jr.uniform(k2, (num_pos,), maxval=2.0)
    neg = -(0.5 + jr.uniform(k3, (size - num_pos,), maxval=2.0))
    lambd = jnp.concatenate([neg, pos])
    return cone_lib.vec_symm(Q @ (lambd[:, None] * Q.T))


@pytest.mark.parametrize(
    "size,num_pos", [(2, 1), (3, 1), (3, 2), (4, 2), (5, 3), (6, 1), (6, 5)]
)
def test_psd_mixed_eigenvalues_jacobian_matches_jvp(getkey, size, num_pos):
    """Indefinite point: the only case where the PSD Jacobian is non-trivial.

    Regression test: the analytical Jacobian used an identity (instead of a
    block of ones) on the positive-eigenvalue block, which is only correct
    when exactly one eigenvalue is positive.
    """
    dim = cone_lib.symm_size_to_dim(size)
    projector = cone_lib._PSDConeProjector(size=size, dim=dim)
    for _ in range(3):
        x = _psd_mixed_vec(getkey(), size, num_pos)
        dx = jr.normal(getkey(), (dim,))
        _check_jvp_matches_dproj(projector, x, dx)


# ─── product projector ────────────────────────────────────────────────────


def test_product_projector_jacobian_matches_jvp(getkey):
    """Mixed zero + nonneg + SOC, with the SOC piece sampled in interior."""
    zero_dim = 7
    nn_dim = 9
    soc_dim = 5
    cones = {
        cone_lib.ZERO: zero_dim,
        cone_lib.NONNEGATIVE: nn_dim,
        cone_lib.SOC: [soc_dim],
    }
    projector = cone_lib.ProductConeProjector(cones, onto_dual=False)

    total_dim = zero_dim + nn_dim + soc_dim
    for _ in range(3):
        # Build a point with each block sampled appropriately.
        zero_part = jr.normal(getkey(), (zero_dim,))
        # Nonneg: avoid x_i = 0.
        nn_raw = jr.normal(getkey(), (nn_dim,))
        nn_part = jnp.where(jnp.abs(nn_raw) < 1e-2, 1.0 * jnp.sign(nn_raw), nn_raw)
        soc_part = _soc_interior_point(getkey(), dim=soc_dim)
        x = jnp.concatenate([zero_part, nn_part, soc_part])
        dx = jr.normal(getkey(), (total_dim,))
        _check_jvp_matches_dproj(projector, x, dx)


# ─── exponential and power cones (finite differences) ─────────────────────
#
# `jax.jvp` through these projectors hits NaNs in unused `lax.cond` branches,
# so these use central finite differences of the projection instead (the
# projections themselves are checked against Clarabel in
# `test_cone_projectors.py`). Points within ~1e-7 of a kink, detected by
# disagreeing one-sided differences, are skipped.


def _fd_check_cone(projector, dim, rng, num_points=200, h=1e-6, rtol=1e-5, max_bad_frac=0.0):
    """`max_bad_frac`: the exp projection is computed to a ~1e-8 tolerance, so
    with h = 1e-6 a small fraction of points show FD noise of order 1e-3."""
    proj = jax.jit(lambda v: projector(v)[0])
    jac_mv = jax.jit(lambda v, dv: projector(v)[1].mv(dv))
    checked = bad = 0
    for _ in range(num_points):
        scale = rng.choice([0.3, 1.0, 3.0])
        v = jnp.asarray(scale * rng.standard_normal(dim))
        dv = jnp.asarray(rng.standard_normal(dim))
        p0 = np.asarray(proj(v))
        fwd = (np.asarray(proj(v + h * dv)) - p0) / h
        bwd = (p0 - np.asarray(proj(v - h * dv))) / h
        if np.linalg.norm(fwd - bwd) > 1e-3 * max(np.linalg.norm(fwd), 1e-8):
            continue  # kink between v - h dv and v + h dv
        central = 0.5 * (fwd + bwd)
        analytical = np.asarray(jac_mv(v, dv))
        assert np.all(np.isfinite(analytical)), f"non-finite Jacobian at {np.asarray(v)}"
        checked += 1
        if not np.allclose(analytical, central, rtol=rtol, atol=1e-7):
            bad += 1
    assert checked >= 0.9 * num_points, f"only {checked}/{num_points} points away from kinks"
    assert bad <= max_bad_frac * checked, f"{bad}/{checked} points disagree with finite differences"


@pytest.mark.parametrize("onto_dual", [False, True])
def test_exp_jacobian_matches_finite_differences(onto_dual):
    from diffqcp.cones.exp import ExponentialConeProjector

    _fd_check_cone(
        ExponentialConeProjector(3, onto_dual=onto_dual), 9, np.random.default_rng(0), max_bad_frac=0.02
    )


def test_exp_jacobian_on_s_zero_face():
    """Points whose projection is (0, 0, t): small r > 0 next to a large
    negative s. Regression test: the general formula divided 0/0 here."""
    from diffqcp.cones.exp import ExponentialConeProjector

    projector = ExponentialConeProjector(1, onto_dual=False)
    for v in ([0.1089, -3.7565, 1.956], [0.0065, -0.2766, 0.1547], [0.028, -0.7137, 0.1323]):
        p, J = projector(jnp.asarray(v))
        np.testing.assert_allclose(np.asarray(p), [0.0, 0.0, v[2]], atol=1e-8)
        dv = jnp.asarray([0.3, -0.7, 1.1])
        np.testing.assert_allclose(np.asarray(J.mv(dv)), [0.0, 0.0, 1.1], atol=1e-10)


@pytest.mark.parametrize("onto_dual", [False, True])
@pytest.mark.parametrize(
    "alphas",
    [[0.5, 0.5], [0.2, 0.35], [0.7, 0.9], [-0.3, 0.6], [-0.8, -0.25]],
    ids=["half", "small", "large", "mixed_dual", "all_dual"],
)
def test_pow_jacobian_matches_finite_differences(alphas, onto_dual):
    """Regression test for the power cone: alpha/(1 - alpha) mix-ups in the
    Jacobian (invisible at alpha = 0.5), a wrong polar-cone membership test,
    and a sign error in the dual projection (negative alphas / onto_dual)."""
    from diffqcp.cones.pow import PowerConeProjector

    _fd_check_cone(PowerConeProjector(alphas, onto_dual=onto_dual), 3 * len(alphas), np.random.default_rng(1))


# ─── as_matrix ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "make_projector,dim",
    [
        (lambda: cone_lib.ZeroConeProjector(onto_dual=False), 5),
        (lambda: cone_lib.ZeroConeProjector(onto_dual=True), 5),
        (lambda: cone_lib.SecondOrderConeProjector([3, 4, 4]), 11),
        (lambda: cone_lib.PSDConeProjector([3, 2, 3]), 15),
        (lambda: cone_lib.ExponentialConeProjector(2, onto_dual=True), 6),
        (lambda: cone_lib.PowerConeProjector([0.3, -0.7], onto_dual=True), 6),
        (
            lambda: cone_lib.ProductConeProjector(
                {"z": 2, "l": 3, "q": [3], "s": [2], "ep": 1, "p": [0.4]}, onto_dual=True
            ),
            2 + 3 + 3 + 3 + 3 + 3,
        ),
    ],
    ids=["zero", "zero_dual", "soc", "psd", "exp_dual", "pow_dual", "product"],
)
def test_jacobian_as_matrix(make_projector, dim):
    """`as_matrix` agrees with `mv`, and projection Jacobians are symmetric."""
    rng = np.random.default_rng(0)
    projector = make_projector()
    v = jnp.asarray(rng.standard_normal(dim))
    dv = jnp.asarray(rng.standard_normal(dim))
    _, J = projector(v)
    M = np.asarray(J.as_matrix())
    assert M.shape == (dim, dim)
    np.testing.assert_allclose(M @ np.asarray(dv), np.asarray(J.mv(dv)), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(M, M.T, rtol=1e-8, atol=1e-10)
