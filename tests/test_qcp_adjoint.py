"""JVP/VJP adjoint-identity tests for the QCP solution map.

For any linear operator `J: U → V` with adjoint `J*: V → U`, the duality
identity

    <v, J(u)>_V  =  <J*(v), u>_U

must hold for every `(u, v)`. Applied to `diffqcp`'s solution-map derivative,
`u = (dP, dA, dq, db)` is a problem-data perturbation and `v = (dx, dy, ds)`
is a perturbation in solution space. So:

    <(dx, dy, ds), qcp.jvp(dP, dA, dq, db)>
        =  <qcp.vjp(dx, dy, ds), (dP, dA, dq, db)>

If the JVP and VJP are implemented correctly, both inner products must agree
up to LSMR tolerance — and crucially, this holds *without* having to re-solve
the perturbed CQP (so the test is fast and doesn't depend on FD step size).

This is the same canonical adjoint check used in lineax's tests; it catches
sign errors, transposed matrices, and wrong adjoint formulas instantly.

Tolerance note: with the gauge-fixed system and LSMR's default 1e-12
tolerance (Wave 5) the identity holds to ~1e-10 here; before, LSMR at a
hard-coded 1e-8 left ~1e-4 relative contamination and this test used 1e-3.
"""
from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax.experimental.sparse import BCOO, BCSR

from diffqcp import DeviceQCP, HostQCP, QCPStructureCPU, QCPStructureGPU

from .helpers import scoo_to_bcoo, scsr_to_bcsr
from .problems import QCPProbData, generate_dense_qp, generate_least_squares_eq

ATOL = 1e-8
RTOL = 1e-8


def _sparse_inner(a: BCOO | BCSR, b: BCOO | BCSR) -> jnp.ndarray:
    """Inner product of two sparse matrices that share the same sparsity pattern.

    `qcp.vjp`'s declared return type is BCSR (abstract base), while the
    `HostQCP` runtime returns BCOO; both expose `.data` in the same order
    as the perturbation we passed in, so the inner product is well-defined.
    Wave 4 will harmonise the declared and runtime types.
    """
    return jnp.sum(a.data * b.data)


def _make_perturbations(prob_data: QCPProbData, key):
    """Random `(dP, dA, dq, db)` matching `(P, A)` sparsity, plus `(dx, dy, ds)`.

    `dP` shares P's upper-triangular sparsity; `dA` shares A's full sparsity.
    """
    P_upper = scoo_to_bcoo(prob_data.Pupper_coo)
    A = scoo_to_bcoo(prob_data.scs_ordered()[0])

    k1, k2, k3, k4, k5, k6, k7 = jr.split(key, 7)
    dP_data = jr.normal(k1, P_upper.data.shape, dtype=P_upper.data.dtype)
    dA_data = jr.normal(k2, A.data.shape, dtype=A.data.dtype)
    dP = BCOO((dP_data, P_upper.indices), shape=P_upper.shape)
    dA = BCOO((dA_data, A.indices), shape=A.shape)
    dq = jr.normal(k3, (prob_data.n,))
    db = jr.normal(k4, (prob_data.m,))

    dx = jr.normal(k5, (prob_data.n,))
    dy = jr.normal(k6, (prob_data.m,))
    ds = jr.normal(k7, (prob_data.m,))
    return (dP, dA, dq, db), (dx, dy, ds)


def _build_host_qcp(prob_data: QCPProbData) -> HostQCP:
    P_upper = scoo_to_bcoo(prob_data.Pupper_coo)
    A_scs, b, y, s = prob_data.scs_ordered()  # PSD rows into diffqcp's SCS order
    A = scoo_to_bcoo(A_scs)
    structure = QCPStructureCPU(P_upper, A, prob_data.scs_cones)
    return HostQCP(
        P_upper,
        A,
        jnp.asarray(prob_data.q),
        jnp.asarray(b),
        jnp.asarray(prob_data.x),
        jnp.asarray(y),
        jnp.asarray(s),
        structure,
    )


# Problem fixtures: name, generator (callable taking a numpy rng/seed), and a
# jax PRNG seed offset for the perturbations. Keep individual problems small;
# the goal is correctness, not stress.
PROBLEM_GENERATORS = [
    pytest.param(
        lambda seed: generate_least_squares_eq(m=12, n=6, rng_or_seed=seed),
        id="least_squares_eq_12x6",
    ),
    pytest.param(
        lambda seed: generate_least_squares_eq(m=20, n=10, rng_or_seed=seed),
        id="least_squares_eq_20x10",
    ),
    # The fixtures above have a diagonal P; this one has a dense P, which
    # exercises the off-diagonal (factor-of-two) part of the CPU dP adjoint.
    pytest.param(
        lambda seed: generate_dense_qp(n=8, m=12, rng_or_seed=seed),
        id="dense_qp_8x12",
    ),
]


@pytest.mark.parametrize("generator", PROBLEM_GENERATORS)
def test_jvp_vjp_adjoint_identity_cpu(generator):
    problem = generator(seed=0)
    prob_data = QCPProbData(problem)
    qcp = _build_host_qcp(prob_data)

    key = jr.PRNGKey(0)
    (dP, dA, dq, db), (dx, dy, ds) = _make_perturbations(prob_data, key)

    # Forward: qcp.jvp maps data perturbations → solution perturbations.
    jvp_dx, jvp_dy, jvp_ds = qcp.jvp(dP, dA, dq, db)

    # Backward: qcp.vjp maps solution perturbations → data perturbations.
    vjp_dP, vjp_dA, vjp_dq, vjp_db = qcp.vjp(dx, dy, ds)

    # Inner products on each side of the adjoint identity.
    forward = (
        jnp.sum(dx * jvp_dx)
        + jnp.sum(dy * jvp_dy)
        + jnp.sum(ds * jvp_ds)
    )
    backward = (
        _sparse_inner(dP, vjp_dP)
        + _sparse_inner(dA, vjp_dA)
        + jnp.sum(dq * vjp_dq)
        + jnp.sum(db * vjp_db)
    )

    np.testing.assert_allclose(
        np.asarray(forward), np.asarray(backward), atol=ATOL, rtol=RTOL
    )


@pytest.mark.parametrize("generator", PROBLEM_GENERATORS)
def test_jvp_vjp_adjoint_identity_device(generator):
    """Same identity for `DeviceQCP`, which stores the *full* symmetric P.

    Perturbations to P must be symmetric (they perturb a symmetric matrix),
    so dP is drawn as a symmetric matrix restricted to P's sparsity pattern.
    """
    prob_data = QCPProbData(generator(seed=0))
    A_scs, b, y, s = prob_data.scs_ordered()
    P = scsr_to_bcsr(prob_data.Pcsr)
    A = scsr_to_bcsr(A_scs.tocsr())
    qcp = DeviceQCP(
        P, A, jnp.asarray(prob_data.q), jnp.asarray(b),
        jnp.asarray(prob_data.x), jnp.asarray(y), jnp.asarray(s),
        QCPStructureGPU(P, A, prob_data.scs_cones),
    )

    rng = np.random.default_rng(0)
    n, m = prob_data.n, prob_data.m
    R = rng.standard_normal((n, n))
    P_rows, P_cols = np.asarray(qcp.problem_structure.P_nonzero_rows), np.asarray(qcp.problem_structure.P_nonzero_cols)
    dP = BCSR(
        (jnp.asarray((R + R.T)[P_rows, P_cols]), P.indices, P.indptr), shape=P.shape
    )
    dA = BCSR((jnp.asarray(rng.standard_normal(A.data.shape)), A.indices, A.indptr), shape=A.shape)
    dq, db = jnp.asarray(rng.standard_normal(n)), jnp.asarray(rng.standard_normal(m))
    dx, dy, ds = (jnp.asarray(rng.standard_normal(k)) for k in (n, m, m))

    jvp_dx, jvp_dy, jvp_ds = qcp.jvp(dP, dA, dq, db)
    vjp_dP, vjp_dA, vjp_dq, vjp_db = qcp.vjp(dx, dy, ds)

    forward = jnp.sum(dx * jvp_dx) + jnp.sum(dy * jvp_dy) + jnp.sum(ds * jvp_ds)
    backward = (
        _sparse_inner(dP, vjp_dP) + _sparse_inner(dA, vjp_dA)
        + jnp.sum(dq * vjp_dq) + jnp.sum(db * vjp_db)
    )
    np.testing.assert_allclose(np.asarray(forward), np.asarray(backward), atol=ATOL, rtol=RTOL)
