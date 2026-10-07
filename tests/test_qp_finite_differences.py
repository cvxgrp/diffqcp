"""JVP of the solution map vs. central finite differences, for QPs (P != 0).

`diffcp` cannot serve as an oracle when P != 0 (see `test_diffcp_oracle.py`),
so here we perturb the data along a random direction, re-solve with Clarabel
at tight tolerance, and compare `(x(+h) - x(-h)) / 2h` against `qcp.jvp`.

This is only meaningful where the solution map is differentiable (active set
locally constant); the fixtures are random, strictly feasible problems, and
the step is small enough not to change the active set.
"""
from __future__ import annotations

import clarabel
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from jax.experimental.sparse import BCOO

from diffqcp.solvers import DenseDirectSolver, LSMRSolver

from .problems import QCPProbData, generate_dense_qp, generate_portfolio_problem
from .test_qcp_adjoint import _build_host_qcp

STEP = 1e-6
RTOL = 1e-6


def _clarabel_solve(Pupper: sp.csc_matrix, q, A: sp.csc_matrix, b, cones):
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-13
    settings.tol_ktratio = 1e-12
    soln = clarabel.DefaultSolver(Pupper, q, A, b, cones, settings).solve()
    assert str(soln.status) == "Solved"
    return np.array(soln.x), np.array(soln.z), np.array(soln.s)


PROBLEMS = {
    "dense_qp": lambda: generate_dense_qp(n=8, m=12, rng_or_seed=0),
    "portfolio": lambda: generate_portfolio_problem(15, 0),
}


@pytest.mark.parametrize("solver", [LSMRSolver(), DenseDirectSolver()], ids=["lsmr", "dense"])
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_qp_jvp_matches_finite_differences(name, solver):
    pd = QCPProbData(PROBLEMS[name]())
    assert not pd.scs_cones.get("s"), "Clarabel-order data below assumes no PSD cones"
    qcp = _build_host_qcp(pd)

    rng = np.random.default_rng(1)
    U = pd.Pupper_coo
    A = pd.Acoo
    dP_vals = rng.standard_normal(U.nnz)
    dA_vals = rng.standard_normal(A.nnz)
    dq = rng.standard_normal(pd.n)
    db = rng.standard_normal(pd.m)

    def perturbed(t):
        Pu = sp.csc_matrix(sp.coo_matrix((U.data + t * dP_vals, (U.row, U.col)), shape=U.shape))
        Ap = sp.csc_matrix(sp.coo_matrix((A.data + t * dA_vals, (A.row, A.col)), shape=A.shape))
        return _clarabel_solve(Pu, pd.q + t * dq, Ap, pd.b + t * db, pd.clarabel_cones)

    plus, minus = perturbed(STEP), perturbed(-STEP)
    fd = [(p - m) / (2 * STEP) for p, m in zip(plus, minus, strict=True)]

    dP = BCOO((jnp.asarray(dP_vals), jnp.stack([jnp.asarray(U.row), jnp.asarray(U.col)], axis=1)), shape=U.shape)
    dA = BCOO((jnp.asarray(dA_vals), jnp.stack([jnp.asarray(A.row), jnp.asarray(A.col)], axis=1)), shape=A.shape)
    jvp = qcp.jvp(dP, dA, jnp.asarray(dq), jnp.asarray(db), solver=solver)

    for label, actual, expected in zip(("dx", "dy", "ds"), jvp, fd, strict=True):
        rel = np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected)
        assert rel < RTOL, f"{label}: {rel:.2e}"
