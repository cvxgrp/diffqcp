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
import cvxpy as cvx
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scs
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from jax.experimental.sparse import BCOO

from diffqcp import HostQCP, QCPStructureCPU
from diffqcp.solvers import DenseDirectSolver, LSMRSolver

from .problems import (
    QCPProbData,
    generate_dense_qp,
    generate_portfolio_problem,
    generate_pow_projection_problem,
)
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


def _scs_solve(P, A, b, c, cones):
    solver = scs.SCS(
        dict(P=sp.csc_matrix(P), A=sp.csc_matrix(A), b=b, c=c), cones,
        eps_abs=1e-12, eps_rel=1e-12, max_iters=500_000, acceleration_lookback=0, verbose=False,
    )
    sol = solver.solve()
    assert sol["info"]["status"] == "solved", sol["info"]["status"]
    return sol["x"], sol["y"], sol["s"]


def test_power_cone_jvp_matches_finite_differences():
    """Power cones have no other oracle (diffcp lacks them). Clarabel cannot
    solve these tightly enough (its solutions satisfy Pi_{K*}(y - s) = y only
    to ~1e-6), so this uses SCS at eps=1e-12, which reaches ~1e-12, both for
    the point the derivative is taken at and for the finite differences."""
    data, _, _ = generate_pow_projection_problem(9, 0).get_problem_data(cvx.SCS)
    cones = dims_to_solver_dict(data["dims"])
    assert cones["p"], "expected power cones"
    A = sp.csc_matrix(data["A"])
    b, c = data["b"], data["c"]
    m, n = A.shape
    P = sp.triu(sp.csc_matrix(data["P"])).tocsc()  # SCS and HostQCP take the upper triangle

    def to_bcoo(M, values=None):
        M = M.tocoo()
        vals = M.data if values is None else values
        return BCOO((jnp.asarray(vals), jnp.stack([jnp.asarray(M.row), jnp.asarray(M.col)], axis=1)), shape=M.shape)

    x, y, s = _scs_solve(P, A, b, c, cones)
    P_b, A_b = to_bcoo(P), to_bcoo(A)
    qcp = HostQCP(P_b, A_b, jnp.asarray(c), jnp.asarray(b), jnp.asarray(x), jnp.asarray(y), jnp.asarray(s),
                  QCPStructureCPU(P_b, A_b, cones))

    rng = np.random.default_rng(1)
    Pc, Ac = P.tocoo(), A.tocoo()
    dP_vals, dA_vals = rng.standard_normal(Pc.nnz), rng.standard_normal(Ac.nnz)
    dc, db = rng.standard_normal(n), rng.standard_normal(m)
    dP = sp.csc_matrix((dP_vals, (Pc.row, Pc.col)), shape=P.shape)
    dA = sp.csc_matrix((dA_vals, (Ac.row, Ac.col)), shape=A.shape)

    h = 1e-5
    plus = _scs_solve(P + h * dP, A + h * dA, b + h * db, c + h * dc, cones)
    minus = _scs_solve(P - h * dP, A - h * dA, b - h * db, c - h * dc, cones)
    fd = [(p - q) / (2 * h) for p, q in zip(plus, minus, strict=True)]

    jvp = qcp.jvp(to_bcoo(P, dP_vals), to_bcoo(A, dA_vals), jnp.asarray(dc), jnp.asarray(db))
    for label, actual, expected in zip(("dx", "dy", "ds"), jvp, fd, strict=True):
        rel = np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected)
        assert rel < 1e-5, f"{label}: {rel:.2e}"
