import numpy as np
from jax import ShapeDtypeStruct, eval_shape
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float, Integer

from diffqcp._helpers import _to_int_list

class BlockOperator(lx.AbstractLinearOperator):

    blocks: list[lx.AbstractLinearOperator]
    num_blocks: int
    _in_sizes: list[int]
    _out_sizes: list[int]
    split_indices: Integer[list, "..."]

    def __init__(
        self,
        blocks: list[lx.AbstractLinearOperator]
    ):
        """
        assumption is that block i maps (n_i,) -> (m_i,)
        no support for pytrees
        """
        self.blocks = blocks
        self.num_blocks = len(blocks)
        self._in_sizes = [block.in_size() for block in self.blocks]
        self._out_sizes = [block.out_size() for block in self.blocks]
        # NOTE(quill): `int(idx)` is needed else `eqx.filter_{...}` doesn't filter out these indices
        self.split_indices = _to_int_list(np.cumsum(self._in_sizes[:-1]))
    
    def mv(self, x):
        chunks = jnp.split(x, self.split_indices)
        results = [op.mv(xi) for op, xi in zip(self.blocks, chunks)]
        return jnp.concatenate(results)
    
    def as_matrix(self):
        """uses output dtype

        not meant to be efficient.
        """
        dtype = self.blocks[0].out_structure().dtype
        zeros_block = jnp.zeros((self._out_size, self._in_size), dtype=dtype)
        n, m = 0, 0
        for i in range(self.num_blocks):
            ni, mi = self._in_sizes[i], self._out_sizes[i]
            zeros_block.at[m:m+mi, n:n+ni].set(self.blocks[i].as_matrix())
            n += ni
            m += mi

    def transpose(self):
        return BlockOperator([block.T for block in self.blocks])
    
    def in_structure(self):
        return ShapeDtypeStruct(shape=(self.in_size(),), dtype=self.blocks[0].in_structure().dtype)

    def out_structure(self):
        return ShapeDtypeStruct(shape=(self.out_size(),), dtype=self.blocks[0].out_structure().dtype)
    
    def in_size(self) -> int:
        return sum(self._in_sizes)

    def out_size(self) -> int:
        return sum(self._out_sizes)
    
@lx.is_symmetric.register(BlockOperator)
def _(op):
    # You can define symmetry however you want; here's a simple version:
    return all(lx.is_symmetric(block) for block in op.blocks)

# def _as_symmetric_psd_func_op(A: lx.AbstractLinearOperator, v: Array) -> lx.FunctionLinearOperator:
def _to_2d_symmetric_psd_func_op(A: lx.AbstractLinearOperator, v: Array) -> lx.FunctionLinearOperator:
    """FLOPless
    
    Assumed `A` is symmetric.

    """
    
    A_shape_dtype = eval_shape(A.mv, v)
    A_shape = A_shape_dtype.shape

    def mv(dx: Array):
        # dx is 1D
        dx = jnp.reshape(dx, (A_shape[0], A_shape[-1]))
        out = A.mv(dx)
        return jnp.ravel(out)
    
    in_structure = ShapeDtypeStruct(shape=(A_shape[0]*A_shape[-1],), dtype=A_shape_dtype.dtype)
    return lx.FunctionLinearOperator(mv, input_structure=in_structure, tags=[lx.symmetric_tag, lx.positive_semidefinite_tag])
