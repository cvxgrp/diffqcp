"""Tests for the gauge-fixed derivative solvers in `diffqcp.solvers`."""
from __future__ import annotations

import jax.numpy as jnp
import lineax as lx
import numpy as np
import pytest

from diffqcp import qcp as qcp_module
from diffqcp.solvers import (
    DenseDirectSolver,
    LSMRSolver,
    augmented_system,
    gauge_fixed,
    materialise_gauge_operator,
    resolve_solver,
)

from .test_diffcp_oracle import _case

STRUCTURE_PROBLEMS = ["lp", "portfolio", "sdp_5", "logistic_exp"]


def _dense_F(name):
    qcp = _case(name).qcp
    pi_z, F, _ = qcp._form_atoms()
    N = qcp.problem_structure.N
    F_dense = np.asarray(materialise_gauge_operator(F))  # works for any operator
    assert F_dense.shape == (N, N)
    return qcp, F, F_dense


@pytest.mark.parametrize("name", STRUCTURE_PROBLEMS)
def test_F_annihilates_embedding(name):
    """F z = 0 for z = (x, y - s, 1): the reason F is singular."""
    qcp, _, F_dense = _dense_F(name)
    z = np.concatenate([np.asarray(qcp.x), np.asarray(qcp.y - qcp.s), [1.0]])
    assert np.linalg.norm(F_dense @ z) <= 1e-8 * np.linalg.norm(F_dense) * np.linalg.norm(z)


@pytest.mark.parametrize("name", STRUCTURE_PROBLEMS)
def test_gauge_fixing_removes_null_direction(name):
    """F is numerically singular; F' = F E has full column rank."""
    _, F, F_dense = _dense_F(name)
    sv_F = np.linalg.svd(F_dense, compute_uv=False)
    sv_Fp = np.linalg.svd(np.asarray(materialise_gauge_operator(gauge_fixed(F))), compute_uv=False)
    assert sv_F[-1] / sv_F[0] < 1e-12
    assert sv_Fp[-1] / sv_Fp[0] > 1e-8


def _random_rank_deficient(N, rank, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N, rank)) @ rng.standard_normal((rank, N - 1))


@pytest.mark.parametrize("solver", [LSMRSolver(), DenseDirectSolver()], ids=["lsmr", "dense"])
def test_solvers_on_full_rank_system(solver):
    rng = np.random.default_rng(0)
    N = 12
    Fp = rng.standard_normal((N, N - 1))
    op = lx.MatrixLinearOperator(jnp.asarray(Fp))
    # JVP: consistent system F' d = r.
    d_true = rng.standard_normal(N - 1)
    d = solver.solve(op, jnp.asarray(Fp @ d_true))
    np.testing.assert_allclose(np.asarray(d), d_true, rtol=1e-8, atol=1e-10)
    # VJP: minimum-norm solution of F'^T w = g.
    g = rng.standard_normal(N - 1)
    w = solver.solve_transpose(op, jnp.asarray(g))
    np.testing.assert_allclose(np.asarray(w), np.linalg.pinv(Fp.T) @ g, rtol=1e-8, atol=1e-10)


def test_dense_direct_falls_back_when_rank_deficient():
    """At a degenerate point F' loses rank; the augmented LU is singular and
    the solver must return the least-squares / minimum-norm answer instead."""
    N = 12
    Fp = _random_rank_deficient(N, rank=N - 3, seed=1)
    op = lx.MatrixLinearOperator(jnp.asarray(Fp))
    rng = np.random.default_rng(2)
    r = Fp @ rng.standard_normal(N - 1)
    g = Fp.T @ rng.standard_normal(N)
    solver = DenseDirectSolver()
    d = solver.solve(op, jnp.asarray(r))
    w = solver.solve_transpose(op, jnp.asarray(g))
    assert np.all(np.isfinite(np.asarray(d))) and np.all(np.isfinite(np.asarray(w)))
    np.testing.assert_allclose(np.asarray(d), np.linalg.pinv(Fp) @ r, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(w), np.linalg.pinv(Fp.T) @ g, rtol=1e-6, atol=1e-8)


def test_augmented_system_is_symmetric():
    Fp = jnp.asarray(np.random.default_rng(3).standard_normal((7, 6)))
    K = augmented_system(Fp)
    assert K.shape == (13, 13)
    np.testing.assert_array_equal(np.asarray(K), np.asarray(K.T))


def test_resolve_solver_legacy_strings():
    assert isinstance(resolve_solver("jax-lsmr", None), LSMRSolver)
    assert isinstance(resolve_solver(None, None), LSMRSolver)
    assert isinstance(resolve_solver("jax-lu", None), DenseDirectSolver)
    custom = LSMRSolver(rtol=1e-9)
    assert resolve_solver("jax-lu", custom) is custom
    with pytest.raises(ValueError, match="Unknown solve_method"):
        resolve_solver("cholesky", None)


def test_device_qcp_vjp_defaults_to_lsmr():
    """cvxpylayers calls `DeviceQCP.vjp(dx, dy, ds)` with no solve_method; the
    default must not be the dense LU path."""
    import inspect

    sig = inspect.signature(qcp_module.DeviceQCP.vjp)
    assert sig.parameters["solve_method"].default == "jax-lsmr"
