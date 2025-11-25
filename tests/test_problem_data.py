"""Mainly testing adjoint
"""

from diffqcp import QCPStructureCPU, QCPStructureGPU, QCPStructureLayers
import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR

def _make_upper_tri_bcoo(n, rng):
    M = rng.standard_normal(n)
    M = np.triu(M)  # keep upper triangular (including diag)
    Md = jnp.array(M)
    return BCOO.fromdense(Md), Md

def _make_dense_bcsr(m, n, rng):
    M = rng.standard_normal((m, n))
    Md = jnp.array(M)
    return BCSR.fromdense(Md), Md

def test_qcpstructurecpu_obj_matrix_and_mv():
    rng = np.random.default_rng(0)
    n = 8
    m = 5

    P_bcoo, P_upper = _make_upper_tri_bcoo(n, rng)
    A_bcoo = BCOO.fromdense(jnp.array(rng.standard_normal((m, n))))

    s = QCPStructureCPU(P_bcoo, A_bcoo, {})

    # form ObjMatrixCPU and check mv equals full symmetric multiplication
    obj = s.form_obj(P_bcoo)
    v = jnp.array(rng.standard_normal(n))

    # full symmetric matrix = P_upper + P_upper.T - diag(P_upper)
    full_sym = P_upper + P_upper.T - jnp.diag(jnp.diag(P_upper))
    res_expected = full_sym @ v
    res_actual = obj.mv(v)

    assert jnp.allclose(res_actual, res_expected, atol=1e-12, rtol=1e-12)

    # metadata checks
    nz_rows = np.array(s.P_nonzero_rows)
    nz_cols = np.array(s.P_nonzero_cols)
    # positions reported should match nonzero positions of the upper triangular matrix
    mask = (np.triu(P_upper) != 0)
    rows, cols = np.where(np.asarray(mask))
    assert set(zip(rows.tolist(), cols.tolist())) == set(zip(nz_rows.tolist(), nz_cols.tolist()))

def test_qcpstructuregpu_A_transpose_inner_product():
    rng = np.random.default_rng(1)
    n = 10
    m = 7

    # P can be simple full matrix for obj init
    P_dense = jnp.array(rng.standard_normal((n, n)))
    P_bcsr = BCSR.fromdense(P_dense)

    A_bcsr, A_dense = _make_dense_bcsr(m, n, rng)

    s = QCPStructureGPU(P_bcsr, A_bcsr, {})

    # Make random vectors x (n,) and y (m,)
    x = jnp.array(rng.standard_normal((n,)))
    y = jnp.array(rng.standard_normal((m,)))

    # compute <y, A x>
    Ax = A_bcsr @ x
    left = y @ Ax

    # form transpose via structure and compute <x, A^T y>
    A_T = s.form_A_transpose(A_bcsr)
    ATy = A_T @ y
    right = x @ ATy

    assert jnp.allclose(left, right, atol=1e-12, rtol=1e-12)

    # additionally, check that form_A_transpose produces a BCSR whose dense equals A.T
    assert jnp.allclose(jnp.array(A_T.todense()), jnp.array(A_dense.T), atol=1e-12)

