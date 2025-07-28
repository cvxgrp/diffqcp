import jax
import jax.numpy as jnp
import jax.random as jr
import cvxpy as cvx

from diffqcp import HostQCP, DeviceQCP
from .helpers import get_cpu_int, data_and_soln_from_quad_qcp

# TODO(quill): configure so don't run GPU tests when no GPU present
#   => does require utilizing BCOO vs. BCSR matrices, so probably
#   have to create different tests.

CPU = lambda x : jax.device_put(x, device=jax.devices("cpu")[0])
GPU = lambda x : jax.device_put(x, device=jax.devices("gpu")[0])

def test_least_squares_cpu(getkey):
    """
    The least squares (approximation) problem

        minimize    ||Ax - b||^2,

        <=>

        minimize    ||r||^2
        subject to  r = Ax - b,

    where A is a (m x n)-matrix with rank A = n, has
    the analytical solution

        x^star = (A^T A)^-1 A^T b.

    Considering x^star as a function of b, we know

        Dx^star(b) = (A^T A)^-1 A^T.

    This test checks the accuracy of `diffqcp`'s derivative computations by
    comparing DS(Data)dData to Dx^star(b)db.

    **Notes:**
    - `dData == (0, 0, 0, db)`, and other canonicalization considerations must be made
    (hence the `data_and_soln_from_cvxpy_problem` function call and associated data declaration.)
    """

    for _ in range(10):
        # NOTE(quill): ideally don't initialize random data on GPU
        n = jr.randint(getkey(), shape=1, minval=10, maxval=15)
        m = n + jr.randint(getkey(), shape=1, minval=5, high=25)

        n_cpu = get_cpu_int(n)
        m_cpu = get_cpu_int(m)

        A = jr.normal(getkey(), (m, n))
        A_cpu = CPU(A)
        b = jr.normal(getkey(), m)
        b_cpu = CPU(b)

        x = cvx.Variable(n_cpu)
        r = cvx.Variable(m_cpu)
        f0 = cvx.sum_squares(r)
        problem = cvx.Problem(cvx.Minimize(f0), [r == A_cpu @ x - b_cpu])

