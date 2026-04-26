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
left for Wave 3 — their projector internals contain `jax.lax.cond` branches
where one branch contains `1/r0`-type singularities that produce NaNs in the
"unused" branch under autodiff. Wave 3's cones cleanup addresses this.
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
