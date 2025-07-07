import jax
import jax.numpy as jnp
import jax.random as jr
import lineax as lx
import pytest

from diffqcp._linops import _ZeroOperator, _ScalarOperator, _BlockOperator

from .helpers import tree_allclose

# Tests
# - Block Operator
#   - All blocks are symmetric (`_BlockOperator` should then be symmetric)
#   - `vmap` (So instead of applying `_BlockOperator` to (d,), apply `vmap(_BlockOperator.mv)` to (B, d).)

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
    

def test_scalar_operator():
    pos_scal = jnp.array(1.23)
    neg_scal = jnp.array(-3.21)
    pos_scal_op = _ScalarOperator(pos_scal)
    neg_scal_op = _ScalarOperator(neg_scal)

    v = jnp.array([2.0])

    assert tree_allclose(pos_scal_op.mv(v), pos_scal * v)
    assert tree_allclose(neg_scal_op.mv(v), neg_scal * v)

    vs = jnp.reshape(jnp.arange(5.0), (5, 1))

    assert tree_allclose(jax.vmap(pos_scal_op.mv)(vs), pos_scal * vs)
    assert tree_allclose(jax.vmap(neg_scal_op.mv)(vs), neg_scal * vs)

    # So what is my desired behavior in BlockOperator that is required for `diffqcp`
    #   => I'm going to place a single `ScalarOperator` at the end of my block linop.
    #   then, when I want to 


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
    op4 = _ScalarOperator(1.5)
    ops = [op1, op2, op3, op4]
    block_op = _BlockOperator(ops)

    in_dim = out_dim = 2 * n + m + 1
    assert block_op.in_size() == in_dim
    assert block_op.out_size() == out_dim
    assert block_op.in_structure().shape == (in_dim,)
    assert block_op.out_structure().shape == (out_dim,)

    v = jr.normal(getkey(), in_dim)
    out1 = op1.mv(v[0:n])
    out2 = op2.mv(v[n:2*n])
    out3 = op3.mv(v[2*n:2*n+m])
    out4 = op4.mv(jnp.array([v[-1]]))
    out_correct = jnp.concatenate([out1, out2, out3, out4])
    assert tree_allclose(out_correct, block_op.mv(v))

    # --- test vmap ---
    
    v = jr.normal(getkey(), (5, in_dim))
    out1 = jax.vmap(op1.mv)(v[:, 0:n])
    out2 = jax.vmap(op2.mv)(v[:, n:2*n])
    out3 = jax.vmap(op3.mv)(v[:, 2*n:2*n+m])
    out4 = jax.vmap(op4.mv)(jnp.reshape(v[:, -1], (5, 1)))
    out_correct = jnp.concatenate([out1, out2, out3, out4], axis=1)
    assert tree_allclose(out_correct, jax.vmap(block_op.mv)(v))

    # === test transpose ===
    
    u = jr.normal(getkey(), out_dim)
    out1 = op1.transpose().mv(u[0:n])
    out2 = op2.transpose().mv(u[n:n+m])
    out3 = op3.transpose().mv(u[n+m:2*n+m])
    out4 = op4.mv(jnp.array([u[-1]]))
    out_correct = jnp.concatenate([out1, out2, out3, out4])
    assert tree_allclose(out_correct, block_op.transpose().mv(u))

    # --- test vmap ---

    u = jr.normal(getkey(), (5, out_dim))
    out1 = jax.vmap(op1.transpose().mv)(u[:, 0:n])
    out2 = jax.vmap(op2.transpose().mv)(u[:, n:n+m])
    out3 = jax.vmap(op3.transpose().mv)(u[:, n+m:2*n+m])
    out4 = jax.vmap(op4.transpose().mv)(jnp.reshape(u[:, -1], (5, 1)))
    out_correct = jnp.concatenate([out1, out2, out3, out4], axis=1)
    assert tree_allclose(out_correct, jax.vmap(block_op.transpose().mv)(u))

# def test_block_diag_operator(device):
#     n = 10
#     m = 5
#     N = 2*n + 1
#     rng = torch.Generator(device=device).manual_seed(0)

#     for _ in range(10):

#         x = torch.randn(n, generator=rng, device=device)
#         A = torch.randn((m, n), generator=rng, device=device)

#         op1: lo.LinearOperator = lo.DiagonalOperator(x)
#         op2 : lo.LinearOperator = lo.MatrixOperator(A)
#         op3 : lo.LinearOperator = ScalarOperator(torch.tensor(2, device=device))
#         block_op = BlockDiag([op1, op2, op3], device=device)

#         v = torch.randn(N, generator=rng, device=device)
#         x, y, tau = v[0:n], v[n:2*n], v[-1]

#         out = torch.empty(n+m+1, device=device)
#         out[0:n] = op1 @ x
#         out[n:n+m] = op2 @ y
#         out[-1] = op3 @ tau.unsqueeze(0)

#         u = torch.randn(n+m+1, generator=rng, device=device)
#         out_transpose = torch.empty(N, device=device)
#         out_transpose[0:n] = op1.T @ u[0:n]
#         out_transpose[n:2*n] = op2.T @ u[n:n+m]
#         out_transpose[-1] = op3.T @ u[-1].unsqueeze(0)


#         assert torch.allclose(out, block_op @ v)
#         assert torch.allclose(out_transpose, block_op.T @ u)

# TODO(quill): will need to create a wrapper function so I can batch ops on top of each other
# NOTE(quill): a `vmap` test in `lineax` does this.

# TODO(quill): add test to ensure BlockOperator is symmetric if its blocks are symmetric.
#   (skipping for now since this is irrelevant to `diffqcp`.)