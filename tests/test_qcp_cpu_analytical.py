import time

import numpy as np
import jax
jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import scipy.linalg as la
import jax.numpy.linalg as jla
import jax.random as jr
import cvxpy as cvx
import patdb
import equinox as eqx

from diffqcp import  HostQCP, QCPStructureCPU
from .helpers import (quad_data_and_soln_from_qcp_coo as quad_data_and_soln_from_qcp,
                      scoo_to_bcoo, QCPProbData, get_zeros_like_coo)

# TODO(quill): configure so don't run GPU tests when no GPU present
#   => does require utilizing BCOO vs. BCSR matrices, so probably
#   have to create different tests.

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

    # TODO(quill): update the testing to follow best practices

    np.random.seed(0)

    for _ in range(10):
        np.random.seed(0)
        n = np.random.randint(low=10, high=15)
        m = n + np.random.randint(low=5, high=15)
        # n = np.random.randint(low=1_000, high=1_500)
        # m = n + np.random.randint(low=500, high=1_000)

        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cvx.Variable(n)
        r = cvx.Variable(m)
        f0 = cvx.sum_squares(r)
        problem = cvx.Problem(cvx.Minimize(f0), [r == A@x - b])

        data = QCPProbData(problem)

        P = scoo_to_bcoo(data.Pcoo)
        Pupper = scoo_to_bcoo(data.Pupper_coo)
        A_orig = A
        A = scoo_to_bcoo(data.Acoo)
        q = jnp.array(data.q)
        b_orig = b
        b = jnp.array(data.b)
        x = jnp.array(data.x)
        y = jnp.array(data.y)
        s = jnp.array(data.s)

        qcp_struc = QCPStructureCPU(Pupper, A, data.scs_cones)
        qcp = HostQCP(P, A, q, b, x, y, s, qcp_struc)

        print("N = ", qcp_struc.N)
        print("n = ", qcp_struc.n)
        print("m = ", qcp_struc.m)

        dP = get_zeros_like_coo(data.Pupper_coo)
        dP = scoo_to_bcoo(dP)
        dA = get_zeros_like_coo(data.Acoo)
        dA = scoo_to_bcoo(dA)
        assert b_orig.size == b.size
        np.testing.assert_allclose(-b_orig, b) # sanity check
        db = 1e-6 * jr.normal(getkey(), shape=jnp.size(b))
        dq = jnp.zeros_like(q)

        Dx_b = jnp.array(la.solve(A_orig.T @ A_orig, A_orig.T))

        # start = time.perf_counter()
        # dx, dy, ds = qcp.jvp(dP, dA, dq, -db)
        # tol = jnp.abs(dx)
        # end = time.perf_counter()
        # print(f"compile + solve time = {end - start}..")
        
        true_result = Dx_b @ db

        # patdb.debug()

        # assert jnp.allclose(true_result, dx[m:], atol=1e-8)

        # assert False # DEBUG

        def is_array_and_dtype(dtype):
            def _predicate(x):
                return isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, dtype)
            return _predicate

        # Partition qcp into (traced, static) components
        qcp_traced, qcp_static = eqx.partition(qcp, is_array_and_dtype(jnp.floating))

        # Partition inputs similarly
        jvp_inputs = (dP, dA, dq, -db)
        inputs_traced, inputs_static = eqx.partition(jvp_inputs, is_array_and_dtype(jnp.floating))

        # Define a wrapper that takes only the traced inputs
        def jvp_wrapped(qcp_traced, inputs_traced):
            # Recombine with the static parts
            qcp_full = eqx.combine(qcp_traced, qcp_static)
            inputs_full = eqx.combine(inputs_traced, inputs_static)
            return qcp_full.jvp(*inputs_full)

        # Compile it
        jvp_compiled = eqx.filter_jit(jvp_wrapped)

        # print out static vs traced inputs
        
        # Call it
        start = time.perf_counter()
        dx, dy, ds = jvp_compiled(qcp_traced, inputs_traced)
        tol = np.asarray(dx)
        end = time.perf_counter()
        print(f"compile + solve time = {end - start}..")

        start = time.perf_counter()
        dx, dy, ds = jvp_compiled(qcp_traced, inputs_traced)
        tol = np.asarray(dx)
        end = time.perf_counter()
        print(f"solve only time = {end - start}..")
        
        # dx, dy, ds = jvp(dP, dA, dq, -db)

        true_result = Dx_b @ db

        print("true result shape: ", jnp.shape(true_result))
        print("dx shape: ", jnp.shape(dx[m:]))

        patdb.debug()
        
        assert jnp.allclose(true_result, dx[m:], atol=1e-8)

        assert False

        # assert False
        
        # with jax.log_compiles():
        
        #     start = time.perf_counter()
        #     jvp = jax.jit(qcp.jvp)
        #     end = time.perf_counter()
        #     print("compilation_time was: ", end - start)

        #     # patdb.debug()

        #     start = time.perf_counter()
        #     dx, dy, ds = jvp(dP, dA, dq, -db)
        #     end = time.perf_counter()
        #     print("solve + jit time was: ", end - start)

        #     start = time.perf_counter()
        #     dx, dy, ds = jvp(dP, dA, dq, -db)
        #     end = time.perf_counter()
        #     print("solve time was: ", end - start)

        # assert jnp.allclose(Dx_b @ db, dx[m:], atol=1e-8)

        # assert False

        # time here (remember to `block_until_ready`)

        # for loop perturbations; make sure jvp is only compiled once.