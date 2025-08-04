import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx

from diffqcp._linops import _ZeroOperator, _BlockLinearOperator

from .helpers import tree_allclose

# TODO(quill): remove the Zero and Scalar operators.

def test_zero_operator(getkey):
    dim_in = 5
    dim_out = 3
    num_batches = 10

    # === same dimension in and out ===
    x_in = jr.normal(getkey(), dim_in)
    y_out = jr.normal(getkey(), dim_in)
    zero_op = _ZeroOperator(x_in, y_out)
    assert tree_allclose(zero_op.mv(x_in), jnp.zeros_like(y_out))

    # vmap
    y_outs = jax.vmap(zero_op.mv)(jr.normal(getkey(), (num_batches, dim_in)))
    assert tree_allclose(y_outs, jnp.zeros((num_batches, dim_in)))

    # === different dimension in than out ===
    y_out = jr.normal(getkey(), dim_out)
    zero_op = _ZeroOperator(x_in, y_out)
    assert tree_allclose(zero_op.mv(x_in), jnp.zeros_like(y_out))

    # vmap
    y_outs = jax.vmap(zero_op.mv)(jr.normal(getkey(), (num_batches, dim_in)))
    assert tree_allclose(y_outs, jnp.zeros((num_batches, dim_out)))


def test_block_operator(getkey):
    # test `mv`
    # test `.transpose.mv`
    # test `in_structure` and `out_structure
    # test under vmap`
    n = 10
    m = 5

    x = jr.normal(getkey(), n)
    A = jr.normal(getkey(), (m, n))
    op1 = lx.DiagonalLinearOperator(x)
    op2 = lx.MatrixLinearOperator(A)
    _fn = lambda y: A.T @ y
    in_struc_fn = lambda: jnp.arange(m, dtype=x.dtype)
    op3 = lx.FunctionLinearOperator(_fn, input_structure=jax.eval_shape(in_struc_fn))
    ops = [op1, op2, op3]
    block_op = _BlockLinearOperator(ops)

    in_dim = out_dim = 2 * n + m
    assert block_op.in_size() == in_dim
    assert block_op.out_size() == out_dim
    assert block_op.in_structure().shape == (in_dim,)
    assert block_op.out_structure().shape == (out_dim,)

    v = jr.normal(getkey(), in_dim)
    out1 = op1.mv(v[0:n])
    out2 = op2.mv(v[n:2*n])
    out3 = op3.mv(v[2*n:2*n+m])
    out_correct = jnp.concatenate([out1, out2, out3])
    assert tree_allclose(out_correct, block_op.mv(v))

    # --- test vmap ---
    
    v = jr.normal(getkey(), (5, in_dim))
    out1 = jax.vmap(op1.mv)(v[:, 0:n])
    out2 = jax.vmap(op2.mv)(v[:, n:2*n])
    out3 = jax.vmap(op3.mv)(v[:, 2*n:2*n+m])
    out_correct = jnp.concatenate([out1, out2, out3], axis=1)
    assert tree_allclose(out_correct, jax.vmap(block_op.mv)(v))

    # === test transpose ===
    
    u = jr.normal(getkey(), out_dim)
    out1 = op1.transpose().mv(u[0:n])
    out2 = op2.transpose().mv(u[n:n+m])
    out3 = op3.transpose().mv(u[n+m:2*n+m])
    out_correct = jnp.concatenate([out1, out2, out3])
    assert tree_allclose(out_correct, block_op.transpose().mv(u))

    # --- test vmap ---

    u = jr.normal(getkey(), (5, out_dim))
    out1 = jax.vmap(op1.transpose().mv)(u[:, 0:n])
    out2 = jax.vmap(op2.transpose().mv)(u[:, n:n+m])
    out3 = jax.vmap(op3.transpose().mv)(u[:, n+m:2*n+m])
    out_correct = jnp.concatenate([out1, out2, out3], axis=1)
    assert tree_allclose(out_correct, jax.vmap(block_op.transpose().mv)(u))

# TODO(quill): will need to create a wrapper function so I can batch ops on top of each other
# NOTE(quill): a `vmap` test in `lineax` does this.

# TODO(quill): add test to ensure BlockOperator is symmetric if its blocks are symmetric.
#   (skipping for now since this is irrelevant to `diffqcp`.)