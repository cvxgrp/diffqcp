import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.experimental.sparse as jsparse
import equinox as eqx

def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)


# Parameters
m, n = 25, 25
num_trials = 500
density = 0.05

# Generate random sparse matrix and vector
rng = np.random.default_rng(123)
A = sp.random(m, n, density=density, format="csr", random_state=rng)
A = A.toarray()
B = sp.random(m, n, density=density, format="csr", random_state=rng)
B = B.toarray()
C = sp.random(m, n, density=density, format="csr", random_state=rng)
C = C.toarray()
x = rng.standard_normal(n)

cpu_device = jax.devices("cpu")[0]
A = jax.device_put(A, device=cpu_device)
A_coo = jsparse.BCOO.fromdense(A)
A_csr = jsparse.BCSR.fromdense(A)
B = jax.device_put(B, device=cpu_device)
B_coo = jsparse.BCOO.fromdense(B)
B_csr = jsparse.BCSR.fromdense(B)
C = jax.device_put(C, device=cpu_device)
C_coo = jsparse.BCOO.fromdense(C)
C_csr = jsparse.BCSR.fromdense(C)
x = jax.device_put(x, device=cpu_device)

coo_batched = jsparse.bcoo_concatenate(
    [A_coo[None], B_coo[None], C_coo[None]], dimension=0
)
print("batched coo shape: ", coo_batched)
csr_batched = jsparse.bcsr_concatenate(
    [A_csr[None], B_csr[None], C_csr[None]], dimension=0
)
print("batched csr shape: ", csr_batched)

# on the CPU, CSR matvecs fall back to COO computations.
A_dense_result = A @ x
A_coo_result = A_coo @ x
A_csr_result = A_csr @ x
assert tree_allclose(A_dense_result, A_coo_result)
assert tree_allclose(A_dense_result, A_csr_result)

B_dense_result = B @ x
B_coo_result = B_coo @ x
B_csr_result = B_csr @ x
assert tree_allclose(B_dense_result, B_coo_result)
assert tree_allclose(B_dense_result, B_csr_result)

C_dense_result = C @ x
C_coo_result = C_coo @ x
C_csr_result = C_csr @ x
assert tree_allclose(C_dense_result, C_coo_result)
assert tree_allclose(C_dense_result, C_csr_result)

batched_coo_result = coo_batched @ x
print("(broadcasting) batched_coo_shape: ", batched_coo_result.shape)
assert tree_allclose(A_dense_result, batched_coo_result[0])
assert tree_allclose(B_dense_result, batched_coo_result[1])
assert tree_allclose(C_dense_result, batched_coo_result[2])

batched_csr_result = csr_batched @ x
assert tree_allclose(A_dense_result, batched_csr_result[0])
assert tree_allclose(B_dense_result, batched_csr_result[1])
assert tree_allclose(C_dense_result, batched_csr_result[2])

# A_coo.transpose() # this works
# A_csr.transpose() # not implemented

# === now test when x has a batch dimension ===

vec1_key, vec2_key, vec3_key = jr.split(jr.PRNGKey(0), 3)
vec1 = jr.normal(vec1_key, shape=(n,))
vec2 = jr.normal(vec2_key, shape=(n,))
vec3 = jr.normal(vec3_key, shape=(n,))

Av1 = A @ vec1
Bv2 = B @ vec2
Cv3 = C @ vec3

batched_vec = jnp.stack([vec1, vec2, vec3])
batched_mv = jax.vmap(lambda A_i, x_i: A_i @ x_i, in_axes=(0, 0))

batched_coo_result = jsparse.bcoo_dot_general(
    coo_batched, batched_vec, dimension_numbers=(([2], [1]), ([0], [0]))
)
batched_coo_result2 = batched_mv(coo_batched, batched_vec)
assert tree_allclose(Av1, batched_coo_result[0])
assert tree_allclose(Bv2, batched_coo_result[1])
assert tree_allclose(Cv3, batched_coo_result[2])
assert tree_allclose(Av1, batched_coo_result2[0])
assert tree_allclose(Bv2, batched_coo_result2[1])
assert tree_allclose(Cv3, batched_coo_result2[2])

batched_csr_result = jsparse.bcsr_dot_general(
    csr_batched, batched_vec, dimension_numbers=(([2], [1]), ([0], [0]))
)
batched_csr_result2 = batched_mv(csr_batched, batched_vec)
assert tree_allclose(Av1, batched_csr_result[0])
assert tree_allclose(Bv2, batched_csr_result[1])
assert tree_allclose(Cv3, batched_csr_result[2])
assert tree_allclose(Av1, batched_csr_result2[0])
assert tree_allclose(Bv2, batched_csr_result2[1])
assert tree_allclose(Cv3, batched_csr_result2[2])