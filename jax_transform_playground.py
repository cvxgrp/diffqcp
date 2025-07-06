# `uv run playground.py` vs `uv run python playground.py`

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp
import jax.numpy.linalg as la
import jax.random as jr
import lineax as lx
import equinox as eqx
from jaxtyping import Array, Float, PyTree
from diffqcp.linops import BlockOperator, _to_2d_symmetric_psd_func_op

def tree_allclose(x, y, *, rtol=1e-5, atol=1e-8):
    return eqx.tree_equal(x, y, typematch=True, rtol=rtol, atol=atol)

# === test PyTree functionality ===
def soc_proj_dproj(x: PyTree[Array]):
    n = x.shape[-1] # TODO(quill): what kind of filter needed to get list of these?
    t, z = x[..., 0], x[..., 1:]
    norm_z = la.norm(z)

    proj_x = 0.5 * (1 + t / norm_z) * jnp.concatenate([jnp.array([norm_z]), z])
    unit_z = z / norm_z
    
    def mv(dx: PyTree[Array]) -> PyTree[Array]:
        dt, dz = dx[..., 0], dx[..., 1:n]
        first_entry = jnp.array([dt*norm_z + z @ dz])
        second_chunk = dt*z + (t + norm_z)*dz \
                        - t * unit_z * (unit_z @ dz)
        output = jnp.concatenate([first_entry, second_chunk])
        return (1.0 / (2.0 * norm_z)) * output
    
    # NOTE(quill): let's just try third case for now and see if passses through JAX API boundary
    # in_structure = jax.eval_shape(mv)
    y = 1.0 * jnp.arange(0, n)
    in_structure = jax.eval_shape(lambda: y)
    dproj = lx.FunctionLinearOperator(mv, input_structure=in_structure, tags=[lx.symmetric_tag, lx.positive_semidefinite_tag])
    return proj_x, dproj
    
def f1(p, dp):
    proj_x, dproj_x = soc_proj_dproj(p)
    return proj_x + dproj_x.mv(dp)

p = jnp.array([3.0, 3.0, 4.0])
p2 = jnp.array([2.0, 6.0, 6.0])
tree = (p, p2)
dp = jnp.array([.001, .002, .001])
# proj_x, dproj_x = soc_proj_dproj(p)
f1_upgraded = jax.jit(jax.vmap(f1))
# print(f1(p, dp))
p3 = jnp.stack((p, p2))
dp3 = jnp.stack((dp, dp))
print("regular call p: ", f1(p, dp))
print("regular call p2: ", f1(p2, dp))
print("batched call: ", f1_upgraded(p3, dp3))


# Now try `vmap`ing a function that returns a linear operator
# This works!

def make_operator(A: jnp.ndarray) -> lx.AbstractLinearOperator:
    return lx.MatrixLinearOperator(A)

A_batch = jnp.stack([jnp.eye(3), 2*jnp.eye(3), 3*jnp.eye(3)])  # (B, 3, 3)
ops = jax.vmap(make_operator)(A_batch)  # Tuple of MatrixLinearOperators
print("batched matrix ops: ", ops)
print("batched matrix ops type: ", type(ops))
print("batched matrix eval shape ", jax.eval_shape(ops.mv, jnp.arange(9).reshape((3, 3))))
# new_op = _to_2d_symmetric_psd_func_op(ops, jnp.reshape(jnp.arange(9), (3, 3)))
# probably have to use `einsum` to make multiplications work.
# print("new op mv", new_op.mv(jnp.arange(9)))
print("batched matrix ops shape: ", ops.out_structure())


# === try nonnegative cone implementation ===

# TODO(quill): return tuple or Pytree?
def proj_dproj_nonnegative(x: Float[Array, " n"]):
    proj_x = jnp.maximum(x, 0)
    dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
    return proj_x, dproj_x

projs, Dprojs = jax.jit(jax.vmap(proj_dproj_nonnegative))(jnp.stack([p, p2]))
dps = jnp.stack([dp, dp])
print("batched projections: ", projs)
print("Batched ops: ", Dprojs)
print("Batched jvp: ", Dprojs.mv(dps))

ops = [lx.MatrixLinearOperator(jnp.eye(3)), lx.MatrixLinearOperator(2 * jnp.eye(2))]
block_op = BlockOperator(ops)

def _some_op(linop, vec):
    return linop.mv(vec) + la.norm(vec)**2

# print(jax.jit(block_op.mv(jnp.ones(5))))
# print(jax.jit(block_op.mv)(jnp.ones(5)))

_vec = jnp.arange(5)
# some_op = eqx.filter_jit(_some_op)
some_op = jax.jit(_some_op)
print(some_op(lx.MatrixLinearOperator(5*jnp.eye(5)), _vec))
print("first one success")
print(some_op(block_op, _vec))