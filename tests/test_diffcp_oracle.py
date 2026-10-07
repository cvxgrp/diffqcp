"""Cross-check `diffqcp` against `diffcp` as an independent oracle.

`diffcp` (Agrawal et al., 2019) differentiates the solution map of cone
programs, i.e. CQPs with `P = 0`. Its linear-solve modes do not support a
quadratic objective, so this file only covers `P = 0`; QPs are covered by the
closed-form tests in `test_qcp_*_analytical.py`.

For each problem we
1. canonicalize with CVXPY into SCS-format data `(A, b, c, cones)` with no
   quadratic objective (`use_quad_obj=False`),
2. let `diffcp` solve it with SCS at tight tolerance and build its derivative
   in `dense` mode, our reference,
3. build a `HostQCP` at the same `(x, y, s)` with `P = 0`, and
4. compare JVPs and VJPs on random perturbations.

The two libraries share the SCS vectorization convention for PSD blocks, so no
permutation is needed here (contrast `QCPProbData.scs_ordered`).

Why SCS rather than Clarabel: the reference is only as good as the point it
is evaluated at. SCS at eps=1e-10 satisfies Pi_{K*}(y - s) = y to ~1e-15,
Clarabel's interior-point solution only to ~1e-6. Also, `diffcp`'s Clarabel
path (diffcp 1.1.6) returns wrong solutions for PSD cones (primal residual
~0.8), apparently a bug in its SCS<->Clarabel PSD row permutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cvx
import diffcp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from jax.experimental.sparse import BCOO

from diffqcp import HostQCP, QCPStructureCPU
from diffqcp.solvers import DenseDirectSolver, LSMRSolver

from . import problems

# Observed agreement with the default solver is 1e-7..1e-14 relative.
RTOL = 1e-6


def _lp(seed: int) -> cvx.Problem:
    rng = np.random.default_rng(seed)
    m, n = 20, 10
    A = rng.standard_normal((m, n))
    b = A @ rng.standard_normal(n) + rng.random(m)  # strictly feasible
    c = -A.T @ rng.random(m)  # dual feasible => bounded
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(c @ x), [A @ x <= b])


PROBLEMS = {
    "lp": lambda: _lp(0),
    "least_squares_soc": lambda: problems.generate_LS_problem(20, 10, 0),
    "portfolio": lambda: problems.generate_portfolio_problem(15, 0),
    "group_lasso_soc": lambda: problems.generate_group_lasso(8, 12, 0),
    "sdp_3": lambda: problems.generate_feasible_sdp(3, 2, rng_or_seed=0),
    "sdp_5": lambda: problems.generate_feasible_sdp(5, 3, rng_or_seed=0),
    "logistic_exp": lambda: problems.generate_group_lasso_logistic(10, 3, 0),
}

SOLVERS = {
    "lsmr": LSMRSolver(),
    "dense": DenseDirectSolver(),
}


@dataclass
class _Case:
    A: sp.csc_matrix
    b: np.ndarray
    c: np.ndarray
    cones: dict[str, Any]
    x: np.ndarray
    y: np.ndarray
    s: np.ndarray
    D: Any
    DT: Any
    qcp: HostQCP
    A_bcoo: BCOO
    P_zero: BCOO


_CACHE: dict[str, _Case] = {}


def _case(name: str) -> _Case:
    if name in _CACHE:
        return _CACHE[name]
    prob = PROBLEMS[name]()
    data, _, _ = prob.get_problem_data(cvx.SCS, solver_opts={"use_quad_obj": False})
    assert data.get("P") is None, "expected a pure cone program"
    A = sp.csc_matrix(data["A"])
    b, c = np.asarray(data["b"], dtype=float), np.asarray(data["c"], dtype=float)
    cones = dims_to_solver_dict(data["dims"])
    # diffcp mutates `A`, so hand it a copy.
    x, y, s, D, DT = diffcp.solve_and_derivative(
        A.copy(), b.copy(), c.copy(), cones, mode="dense", solve_method="SCS", verbose=False,
        eps_abs=1e-10, eps_rel=1e-10, max_iters=200_000,
    )
    m, n = A.shape
    A_coo = A.tocoo()
    A_bcoo = BCOO(
        (jnp.asarray(A_coo.data), jnp.stack([jnp.asarray(A_coo.row), jnp.asarray(A_coo.col)], axis=1)),
        shape=(m, n),
    )
    P_zero = BCOO.fromdense(jnp.zeros((n, n)))
    qcp = HostQCP(
        P_zero, A_bcoo, jnp.asarray(c), jnp.asarray(b),
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(s),
        QCPStructureCPU(P_zero, A_bcoo, cones),
    )
    _CACHE[name] = _Case(A, b, c, cones, x, y, s, D, DT, qcp, A_bcoo, P_zero)
    return _CACHE[name]


def _rel(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12))


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_solution_is_optimal(name):
    """Guard: the oracle is meaningless at a non-optimal point (e.g. an
    unboundedness certificate). At a solution, Pi_{K*}(y - s) = y."""
    case = _case(name)
    proj, _ = case.qcp.problem_structure.cone_projector(jnp.asarray(case.y - case.s))
    assert _rel(np.asarray(proj), case.y) < 1e-9


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_jvp_matches_diffcp(name, solver):
    case = _case(name)
    rng = np.random.default_rng(1)
    A_coo = case.A.tocoo()
    dA_vals = rng.standard_normal(A_coo.nnz)
    db = rng.standard_normal(case.b.size)
    dc = rng.standard_normal(case.c.size)

    expected = case.D(sp.csc_matrix((dA_vals, (A_coo.row, A_coo.col)), shape=case.A.shape), db, dc)

    dA = BCOO((jnp.asarray(dA_vals), case.A_bcoo.indices), shape=case.A_bcoo.shape)
    actual = case.qcp.jvp(case.P_zero, dA, jnp.asarray(dc), jnp.asarray(db), solver=SOLVERS[solver])

    for label, a, e in zip(("dx", "dy", "ds"), actual, expected, strict=True):
        assert _rel(np.asarray(a), e) < RTOL, label


@pytest.mark.parametrize("solver", list(SOLVERS))
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_vjp_matches_diffcp(name, solver):
    case = _case(name)
    rng = np.random.default_rng(2)
    dx = rng.standard_normal(case.c.size)
    dy = rng.standard_normal(case.b.size)
    ds = rng.standard_normal(case.b.size)

    e_dA, e_db, e_dc = case.DT(dx, dy, ds)
    _, a_dA, a_dq, a_db = case.qcp.vjp(
        jnp.asarray(dx), jnp.asarray(dy), jnp.asarray(ds), solver=SOLVERS[solver]
    )

    A_coo = case.A.tocoo()
    e_dA_vals = np.asarray(e_dA.tocsr()[A_coo.row, A_coo.col]).ravel()
    a_dA_vals = np.asarray(a_dA.todense())[A_coo.row, A_coo.col]
    assert _rel(a_dA_vals, e_dA_vals) < RTOL, "dA"
    assert _rel(np.asarray(a_db), e_db) < RTOL, "db"
    assert _rel(np.asarray(a_dq), e_dc) < RTOL, "dc"
