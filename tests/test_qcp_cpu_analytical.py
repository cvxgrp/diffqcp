"""Closed-form Jacobian check for diffqcp.jvp on a least-squares CQP (CPU).

Plain least-squares  min ||A x - b||²  with full-rank `A` (m by n, m >= n) has
the unique optimum  x*(b) = (Aᵀ A)⁻¹ Aᵀ b, so its Jacobian w.r.t. `b` is the
constant matrix  D x*(b) = (Aᵀ A)⁻¹ Aᵀ. We perturb only `b` (zero `dP`,
`dA`, `dq`) and assert that diffqcp's `jvp` recovers `D x*(b) · db` exactly.

Complements `test_qcp_adjoint.py`, which checks JVP/VJP duality (relative
correctness) but not absolute correctness — this file pins the absolute
answer against a closed form.
"""
from __future__ import annotations

import cvxpy as cvx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.linalg as la

from diffqcp import HostQCP, QCPStructureCPU

from .helpers import get_zeros_like_coo, scoo_to_bcoo
from .problems import QCPProbData

# Note: this test deliberately calls `qcp.jvp` un-jitted. Wrapping the call in
# `eqx.filter_jit` triggers a boolean-indexing trace error inside
# `QCPStructureCPU.form_obj`, where `P_diag_mask` (a Bool field) is read as a
# dynamic boolean index. Wave 4's problem-data unification fixes this by
# marking the mask as `eqx.field(static=True)`. Until then, we test
# correctness eagerly and let Wave 4 reinstate the compiled path.


def _ls_problem(A: np.ndarray, b: np.ndarray) -> cvx.Problem:
    m, n = A.shape
    x = cvx.Variable(n)
    r = cvx.Variable(m)
    return cvx.Problem(cvx.Minimize(cvx.sum_squares(r)), [r == A @ x - b])


def test_least_squares_jvp_db_cpu(getkey):
    rng = np.random.default_rng(0)

    for _ in range(10):
        n = int(rng.integers(low=10, high=15))
        m = n + int(rng.integers(low=5, high=15))

        A_orig = rng.standard_normal((m, n))
        b_orig = rng.standard_normal(m)
        problem = _ls_problem(A_orig, b_orig)
        data = QCPProbData(problem)

        Pupper = scoo_to_bcoo(data.Pupper_coo)
        A_bcoo = scoo_to_bcoo(data.Acoo)
        q = jnp.asarray(data.q)
        b_canon = jnp.asarray(data.b)
        x = jnp.asarray(data.x)
        y = jnp.asarray(data.y)
        s = jnp.asarray(data.s)

        # CVXPY's canonical form negates b: data.b == -b_orig (with the residual
        # variable convention used here). Confirms our perturbation direction.
        np.testing.assert_allclose(np.asarray(b_canon), -b_orig, atol=1e-12)

        structure = QCPStructureCPU(Pupper, A_bcoo, data.scs_cones)
        qcp = HostQCP(Pupper, A_bcoo, q, b_canon, x, y, s, structure)

        dP = scoo_to_bcoo(get_zeros_like_coo(data.Pupper_coo))
        dA = scoo_to_bcoo(get_zeros_like_coo(data.Acoo))
        dq = jnp.zeros_like(q)
        db = 1e-6 * jr.normal(getkey(), shape=(jnp.size(b_canon),))

        # Closed-form Jacobian for least-squares.
        Dx_b = jnp.asarray(la.solve(A_orig.T @ A_orig, A_orig.T))
        true_dx = Dx_b @ db

        # `db` lives in the *canonical* b-space, so feed -db (sign flip from above).
        dx, _, _ = qcp.jvp(dP, dA, dq, -db)

        # Canonicalisation prepends m residual vars; the original n variables
        # are at the tail of `dx`.
        # With the gauge-fixed system and LSMR at its 1e-12 default (Wave 5)
        # the error is far below 1e-9; at the old hard-coded 1e-8 it floored
        # at ~1e-8 absolute.
        np.testing.assert_allclose(np.asarray(dx[m:]), np.asarray(true_dx), atol=1e-9)
