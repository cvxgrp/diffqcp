"""`diffqcp.differentiable_solution`: jax.grad through a conic solve."""
from __future__ import annotations

import clarabel
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from diffqcp import QCPStructureCPU, QCPStructureGPU, differentiable_solution, make_differentiable

from .helpers import scoo_to_bcoo, scsr_to_bcsr
from .problems import QCPProbData, generate_dense_qp
from .test_qcp_adjoint import _build_host_qcp


@pytest.fixture(scope="module")
def prob_data():
    return QCPProbData(generate_dense_qp(n=6, m=9, rng_or_seed=0))


def _cpu_inputs(pd):
    P = scoo_to_bcoo(pd.Pupper_coo)
    A = scoo_to_bcoo(pd.scs_ordered()[0])
    structure = QCPStructureCPU(P, A, pd.scs_cones)
    data = (P.data, A.data, jnp.asarray(pd.q), jnp.asarray(pd.b))
    solution = (jnp.asarray(pd.x), jnp.asarray(pd.y), jnp.asarray(pd.s))
    return structure, data, solution


def test_grad_matches_vjp(prob_data):
    """grad of a linear loss <c, (x, y, s)> is exactly vjp(c)."""
    structure, data, solution = _cpu_inputs(prob_data)
    rng = np.random.default_rng(0)
    cx, cy, cs = (jnp.asarray(rng.standard_normal(v.shape)) for v in solution)

    def loss(data):
        x, y, s = differentiable_solution(data, solution, structure)
        return cx @ x + cy @ y + cs @ s

    grads = jax.grad(loss)(data)
    dP, dA, dq, db = _build_host_qcp(prob_data).vjp(cx, cy, cs)
    for g, e in zip(grads, (dP.data, dA.data, dq, db), strict=True):
        np.testing.assert_allclose(np.asarray(g), np.asarray(e), rtol=1e-10, atol=1e-12)


def _make_solve(pd, structure_rows_cols):
    """A JAX-callable Clarabel solve (via pure_callback) for this sparsity pattern."""
    P_rows, P_cols, A_rows, A_cols = (np.asarray(v) for v in structure_rows_cols)
    n, m = pd.n, pd.m

    def _solve_np(P_values, A_values, q, b):
        # Default tolerances: asking Clarabel for ~1e-13 on this problem ends in
        # InsufficientProgress at a poor point (complementarity ~4e-3).
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        P = sp.csc_matrix((np.asarray(P_values), (P_rows, P_cols)), shape=(n, n))
        A = sp.csc_matrix((np.asarray(A_values), (A_rows, A_cols)), shape=(m, n))
        soln = clarabel.DefaultSolver(P, np.asarray(q), A, np.asarray(b), pd.clarabel_cones, settings).solve()
        if str(soln.status) != "Solved":
            raise RuntimeError(f"Clarabel status {soln.status}")
        return np.array(soln.x), np.array(soln.z), np.array(soln.s)

    out_shapes = tuple(jax.ShapeDtypeStruct((k,), jnp.float64) for k in (n, m, m))

    def solve(data):
        return jax.pure_callback(_solve_np, out_shapes, *data)

    return solve


def test_end_to_end_grad_matches_finite_differences(prob_data):
    """loss(data) = ||x*(data)||^2 with a real solve in the loop; jit(grad)
    agrees with a central difference of the loss along a random direction."""
    pd = prob_data
    structure, data, _ = _cpu_inputs(pd)
    s_ = structure
    solve = _make_solve(pd, (s_.P_nonzero_rows, s_.P_nonzero_cols, s_.A_nonzero_rows, s_.A_nonzero_cols))

    differentiable_solve = make_differentiable(solve, structure)

    def loss(data):
        x, y, s = differentiable_solve(data)
        return jnp.sum(x**2) + 0.5 * jnp.sum(y**2)

    grads = jax.jit(jax.grad(loss))(data)

    rng = np.random.default_rng(1)
    direction = tuple(jnp.asarray(rng.standard_normal(v.shape)) for v in data)
    h = 1e-6
    plus = tuple(v + h * d for v, d in zip(data, direction, strict=True))
    minus = tuple(v - h * d for v, d in zip(data, direction, strict=True))
    fd = (loss(plus) - loss(minus)) / (2 * h)
    analytical = sum(jnp.sum(g * d) for g, d in zip(grads, direction, strict=True))
    np.testing.assert_allclose(float(analytical), float(fd), rtol=1e-5)


def test_gpu_structure_on_cpu(prob_data):
    """The BCSR/`QCPStructureGPU` path also works under jax.grad (run on CPU)."""
    pd = prob_data
    A_scs, b, y, s = pd.scs_ordered()
    P = scsr_to_bcsr(pd.Pcsr)
    A = scsr_to_bcsr(A_scs.tocsr())
    structure = QCPStructureGPU(P, A, pd.scs_cones)
    data = (P.data, A.data, jnp.asarray(pd.q), jnp.asarray(b))
    solution = (jnp.asarray(pd.x), jnp.asarray(y), jnp.asarray(s))

    def loss(data):
        x, *_ = differentiable_solution(data, solution, structure)
        return jnp.sum(x)

    grads = jax.grad(loss)(data)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)
    # Gradient wrt q is solution-map structure independent: compare with CPU path.
    cpu_structure, cpu_data, cpu_solution = _cpu_inputs(pd)
    cpu_grads = jax.grad(lambda d: jnp.sum(differentiable_solution(d, cpu_solution, cpu_structure)[0]))(cpu_data)
    np.testing.assert_allclose(np.asarray(grads[2]), np.asarray(cpu_grads[2]), rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(np.asarray(grads[3]), np.asarray(cpu_grads[3]), rtol=1e-8, atol=1e-10)
