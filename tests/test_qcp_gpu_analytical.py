"""Closed-form Jacobian check for diffqcp.jvp on a least-squares CQP (GPU path).

Mirror of `test_qcp_cpu_analytical.py` against `DeviceQCP` / `QCPStructureGPU`.
The test runs CPU-only when no GPU is present (BCSR's `mv` works on CPU too).
The `nvmath-direct` path is only exercised when both nvmath-python is
importable *and* JAX's first device is a GPU.

See `test_qcp_cpu_analytical.py` for the math motivation; tolerance reasoning
is identical.
"""
from __future__ import annotations

import cvxpy as cvx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.linalg as la

try:
    from nvmath.sparse.advanced import DirectSolver
except ImportError:
    DirectSolver = None

from diffqcp import DeviceQCP, QCPStructureGPU

from .helpers import get_zeros_like_csr, scsr_to_bcsr
from .problems import QCPProbData


def _ls_problem(A: np.ndarray, b: np.ndarray) -> cvx.Problem:
    m, n = A.shape
    x = cvx.Variable(n)
    r = cvx.Variable(m)
    return cvx.Problem(cvx.Minimize(cvx.sum_squares(r)), [r == A @ x - b])


def _build_device_qcp_and_perturbations(rng, getkey):
    n = int(rng.integers(low=10, high=15))
    m = n + int(rng.integers(low=5, high=15))

    A_orig = rng.standard_normal((m, n))
    b_orig = rng.standard_normal(m)
    problem = _ls_problem(A_orig, b_orig)
    data = QCPProbData(problem)

    P = scsr_to_bcsr(data.Pcsr)
    A_bcsr = scsr_to_bcsr(data.Acsr)
    q = jnp.asarray(data.q)
    b_canon = jnp.asarray(data.b)
    x = jnp.asarray(data.x)
    y = jnp.asarray(data.y)
    s = jnp.asarray(data.s)

    np.testing.assert_allclose(np.asarray(b_canon), -b_orig, atol=1e-12)

    structure = QCPStructureGPU(P, A_bcsr, data.scs_cones)
    qcp = DeviceQCP(P, A_bcsr, q, b_canon, x, y, s, structure)

    dP = scsr_to_bcsr(get_zeros_like_csr(data.Pcsr))
    dA = scsr_to_bcsr(get_zeros_like_csr(data.Acsr))
    dq = jnp.zeros_like(q)
    db = 1e-6 * jr.normal(getkey(), shape=(jnp.size(b_canon),))

    Dx_b = jnp.asarray(la.solve(A_orig.T @ A_orig, A_orig.T))
    true_dx = Dx_b @ db

    return qcp, (dP, dA, dq, -db), true_dx, m


def test_least_squares_jvp_db_gpu_lsmr(getkey):
    rng = np.random.default_rng(0)
    for _ in range(10):
        qcp, jvp_inputs, true_dx, m = _build_device_qcp_and_perturbations(rng, getkey)
        dx, _, _ = qcp.jvp(*jvp_inputs, solve_method="jax-lsmr")
        np.testing.assert_allclose(np.asarray(dx[m:]), np.asarray(true_dx), atol=1e-9)


def test_least_squares_jvp_db_gpu_direct(getkey):
    """Direct (LU/cuDSS) solver path. Runs on GPU if available; else CPU LU only.

    Note: `nvmath-direct` is GPU-only. When no GPU is present we exercise the
    `jax-lu` (lineax LU on the dense materialised F) path only.
    """
    jax_gpu_enabled = jax.devices()[0].platform == "gpu"
    solvers: list[str] = ["jax-lu"]
    if DirectSolver is not None and jax_gpu_enabled:
        solvers.append("nvmath-direct")

    for solve_method in solvers:
        rng = np.random.default_rng(0)
        for _ in range(10):
            qcp, jvp_inputs, true_dx, m = _build_device_qcp_and_perturbations(rng, getkey)
            dx, _, _ = qcp.jvp(*jvp_inputs, solve_method=solve_method)
            # Direct solves now factor the nonsingular augmented system of the
            # gauge-fixed F' rather than the singular F (Wave 5), so they meet
            # the same tolerance as LSMR.
            np.testing.assert_allclose(np.asarray(dx[m:]), np.asarray(true_dx), atol=1e-9)
