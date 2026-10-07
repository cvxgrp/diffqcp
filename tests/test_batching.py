"""Batching contract: `vmap` at the boundary.

Every operator inside `diffqcp` is written for a single problem; batching is
done by `jax.vmap` over the public entry points:

- many perturbations, one problem: `vmap` over the inputs of `qcp.jvp` /
  `qcp.vjp`;
- many problems sharing one sparsity pattern: `vmap` over the `data` and
  `solution` arguments of `differentiable_solution` (the structure is shared
  and not batched).

Each test compares the batched result with a Python loop over the batch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO

from diffqcp import DenseDirectSolver, LSMRSolver, differentiable_solution

from .problems import QCPProbData, generate_dense_qp, generate_feasible_sdp
from .test_autodiff import _cpu_inputs, _make_solve
from .test_qcp_adjoint import _build_host_qcp

B = 4

PROBLEMS = {
    "dense_qp": lambda: generate_dense_qp(n=6, m=9, rng_or_seed=0),
    "sdp": lambda: generate_feasible_sdp(3, 2, rng_or_seed=0),
}


def _max_abs_diff(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()) if a.size else 0.0


@pytest.mark.parametrize("name", list(PROBLEMS))
def test_vmap_over_perturbations(name):
    pd = QCPProbData(PROBLEMS[name]())
    qcp = _build_host_qcp(pd)
    structure, data, _ = _cpu_inputs(pd)
    rng = np.random.default_rng(0)

    def zeros_like_pattern(values, rows, cols, shape):
        return BCOO((jnp.zeros_like(values), jnp.stack([rows, cols], axis=1)), shape=shape)

    dP = zeros_like_pattern(data[0], structure.P_nonzero_rows, structure.P_nonzero_cols, (pd.n, pd.n))
    dA = zeros_like_pattern(data[1], structure.A_nonzero_rows, structure.A_nonzero_cols, (pd.m, pd.n))
    dq = jnp.asarray(rng.standard_normal((B, pd.n)))
    db = jnp.asarray(rng.standard_normal((B, pd.m)))

    batched = jax.jit(jax.vmap(lambda q_, b_: qcp.jvp(dP, dA, q_, b_)))(dq, db)
    for i in range(B):
        for k, single in enumerate(qcp.jvp(dP, dA, dq[i], db[i])):
            assert _max_abs_diff(batched[k][i], single) < 1e-9

    dx = jnp.asarray(rng.standard_normal((B, pd.n)))
    dy = jnp.asarray(rng.standard_normal((B, pd.m)))
    ds = jnp.asarray(rng.standard_normal((B, pd.m)))
    batched_vjp = jax.jit(jax.vmap(qcp.vjp))(dx, dy, ds)
    for i in range(B):
        single = qcp.vjp(dx[i], dy[i], ds[i])
        assert _max_abs_diff(batched_vjp[0].data[i], single[0].data) < 1e-9
        assert _max_abs_diff(batched_vjp[1].data[i], single[1].data) < 1e-9
        assert _max_abs_diff(batched_vjp[2][i], single[2]) < 1e-9
        assert _max_abs_diff(batched_vjp[3][i], single[3]) < 1e-9


@pytest.mark.parametrize("solver", [LSMRSolver(), DenseDirectSolver()], ids=["lsmr", "dense"])
def test_vmap_over_problems(solver):
    """A batch of distinct problems (perturbed data, each solved) sharing one
    sparsity pattern: vmap(grad) over (data, solution) matches a loop."""
    pd = QCPProbData(PROBLEMS["dense_qp"]())
    structure, data, _ = _cpu_inputs(pd)
    rng = np.random.default_rng(1)
    solve = _make_solve(
        pd,
        (structure.P_nonzero_rows, structure.P_nonzero_cols, structure.A_nonzero_rows, structure.A_nonzero_cols),
    )
    datas = [
        (
            data[0] * (1.0 + 0.1 * rng.random()),
            data[1] + 0.05 * jnp.asarray(rng.standard_normal(data[1].shape)),
            data[2] + 0.3 * jnp.asarray(rng.standard_normal(pd.n)),
            data[3],
        )
        for _ in range(B)
    ]
    solutions = [solve(d) for d in datas]
    batched_data = tuple(jnp.stack([d[k] for d in datas]) for k in range(4))
    batched_solutions = tuple(jnp.stack([s[k] for s in solutions]) for k in range(3))
    c = jnp.asarray(rng.standard_normal(pd.n))

    def loss(d, s):
        return c @ differentiable_solution(d, s, structure, solver)[0]

    batched = jax.jit(jax.vmap(jax.grad(loss)))(batched_data, batched_solutions)
    for i in range(B):
        single = jax.grad(loss)(datas[i], solutions[i])
        for k in range(4):
            assert _max_abs_diff(batched[k][i], single[k]) < 1e-9
