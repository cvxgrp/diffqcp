"""General (i.e., not cone-specific) linear operators that are not implemented in `lineax`.

Note that these operators were purposefully made "private" since they are solely implemented
to support functionality required by `diffqcp`. They **should not** be accessed as if they
were true atoms implemented in `lineax`.
"""

import numpy as np
from jax import ShapeDtypeStruct, eval_shape, vmap
import jax.numpy as jnp
import lineax as lx
import equinox as eqx
from jaxtyping import Array, Integer, Float

from diffqcp._helpers import _to_int_list


def _ZeroOperator(
    x: Float[Array, " _d1"], y: Float[Array, " _d2"] | None = None
) -> lx.AbstractLinearOperator:
    in_struc = eval_shape(lambda: x)
    if y is None:
        out_struc = in_struc
    else:
        out_struc = eval_shape(lambda: y)
    # NOTE(quill): safe to multiply by 0.0 (so a float), since we're assuming the linops are arrays of floats.
    return 0.0 * lx.IdentityLinearOperator(in_struc, out_struc)


def _ScalarOperator(alpha: float) -> lx.AbstractLinearOperator:
    # `_ScalarOperator.in_structure.shape == (1,)`
    return alpha * lx.IdentityLinearOperator(eval_shape(lambda: jnp.arange(1.0)))


class _BlockLinearOperator(lx.AbstractLinearOperator):
    """Represents a block matrix (without explicitly forming zeros).

    TODO(quill): Support operating on PyTrees (clearly the way I handle `input_structure`
        and `output_structure` isn't compatible with PyTrees.)
    """

    blocks: list[lx.AbstractLinearOperator]
    num_blocks: int
    # _in_sizes: list[int]
    # _out_sizes: list[int]
    # NOTE(quill): either use the non-static defined `split_indices` along with `eqx.filter_{...}`,
    #   or use regular JAX function transforms with `split_indices` declared as static.
    #   I'm personally a fan of the explicit declaration, but it seems that this is not the
    #   suggested approach: https://github.com/patrick-kidger/equinox/issues/154.
    #   (It is worth noting that `lineax` itself does use explicit static declarations, such as
    #       in `PyTreeLinearOperator`.)
    # split_indices: list[int]
    split_indices: list[int] = eqx.field(static=True)

    def __init__(
        self,
        blocks: list[lx.AbstractLinearOperator]
    ):
        """
        Parameters
        ----------
        `blocks`: list[lx.AbstractLinearOperator]
        """
        self.blocks = blocks
        self.num_blocks = len(blocks)

        in_sizes = [block.in_size() for block in self.blocks]
        # NOTE(quill): `int(idx)` is needed else `eqx.filter_{...}` doesn't filter out these indices
        #   (Since I've declared `split_indices` as static this isn't necessary, but there's no true cost
        #       to keeping.)
        self.split_indices = _to_int_list(np.cumsum(in_sizes[:-1]))
    
    def mv(self, x):
        chunks = jnp.split(x, self.split_indices, axis=-1)
        results = [op.mv(xi) for op, xi in zip(self.blocks, chunks)]
        return jnp.concatenate(results, axis=-1)
    
    def as_matrix(self):
        """uses output dtype

        not meant to be efficient.
        """
        # dtype = self.blocks[0].out_structure().dtype
        # zeros_block = jnp.zeros((self._out_size, self._in_size), dtype=dtype)
        # n, m = 0, 0
        # for i in range(self.num_blocks):
        #     ni, mi = self._in_sizes[i], self._out_sizes[i]
        #     zeros_block.at[m:m+mi, n:n+ni].set(self.blocks[i].as_matrix())
        #     n += ni
        #     m += mi
        raise NotImplementedError("`_BlockLinearOperator`'s `as_matrix` is not implemented.")

    def transpose(self):
        return _BlockLinearOperator([block.T for block in self.blocks])
    
    def in_structure(self):
        if len(self.blocks[0].in_structure().shape) == 2:
            num_batches = self.blocks[0].in_structure().shape[0]
            idx = 1
        else:
            num_batches = 0
            idx = 0
        in_size = 0
        for block in self.blocks:
            in_size += block.in_structure().shape[idx]
        dtype = self.blocks[0].in_structure().dtype
        in_shape = (num_batches, in_size) if num_batches > 0 else (in_size,)
        return ShapeDtypeStruct(shape=in_shape, dtype=dtype)

    def out_structure(self):
        if len(self.blocks[0].out_structure().shape) == 2:
            num_batches = self.blocks[0].out_structure().shape[0]
            idx = 1
        else:
            num_batches = 0
            idx = 0
        out_size = 0
        for block in self.blocks:
            out_size += block.out_structure().shape[idx]
        dtype = self.blocks[0].out_structure().dtype
        in_shape = (num_batches, out_size) if num_batches > 0 else (out_size,)
        return ShapeDtypeStruct(shape=in_shape, dtype=dtype)
    
@lx.is_symmetric.register(_BlockLinearOperator)
def _(op):
    return all(lx.is_symmetric(block) for block in op.blocks)


# TODO(quill): delete the following
def _to_2D_symmetric_psd_func_op(A: lx.AbstractLinearOperator, v: Float[Array, "B n"]) -> lx.FunctionLinearOperator:
    """Collapse a batch of 2D operators.

    Helper function that takes a batch of AbstractLinearOperators that map 1D Arrays to 1D Arrays
    and wraps it in a `FunctionLinearOperator` so that

    Helper function that accepts 
    
    Parameters
    ----------
    `A` : lx.AbstractLinearOperator
        Each operator within the batch is assumed to be symmetric and positive semidefinite.
    This is a flopless 
    """
    
    in_shape = jnp.shape(v)
    v_dim = jnp.ndim(v)

    if v_dim != 2:
        raise ValueError("`_to_2D_symmetric_psd_func_op` is meant to wrap around"
                         + " linear operators that map from 2D arrays to 2D arrays"
                         + " but the provided vector the operator supposedly operates"
                         + f" on is {v_dim}D.")
    
    def mv(dx: Float[Array, "*batch Bn"]):
        dx_dim = jnp.ndim(dx)
        if dx_dim == 2:
            # in this case the first dimension is batch dimension
            #   should reshape to be (batch, B, n)
            dx_shape = jnp.shape(dx)
            dx = jnp.reshape(dx, (dx_shape[0], in_shape[0], in_shape[1]))
            out = A.mv(dx)
            return jnp.reshape(out, dx_shape)
        elif dx_dim == 1:
            dx = jnp.reshape(dx, in_shape)
            out = A.mv(dx)
            return jnp.ravel(out)
        else:
            raise ValueError("The functional linear operator that wraps around"
                             + " batched SOC Jacobians espects a 1D or 2D input"
                             + f" perturbation, but receieved a {dx_dim}D input.")
            
    in_structure = ShapeDtypeStruct(shape=(in_shape[0]*in_shape[1],), dtype=v.dtype)
    return lx.FunctionLinearOperator(mv, input_structure=in_structure, tags=[lx.symmetric_tag, lx.positive_semidefinite_tag])
