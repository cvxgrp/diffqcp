"""General (i.e., not cone-specific) linear operators that are not implemented in `lineax`.

Note that these operators were purposefully made "private" since they are solely implemented
to support functionality required by `diffqcp`. They **should not** be accessed as if they
were true atoms implemented in `lineax`.
"""

import numpy as np
from jax import ShapeDtypeStruct, eval_shape
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


class _BlockOperator(lx.AbstractLinearOperator):
    """Represents a block matrix (without explicitly forming zeros).

    TODO(quill): Support operating on PyTrees (clearly the way I handle `input_structure`
        and `output_structure` isn't compatible with PyTrees.)
    """

    blocks: list[lx.AbstractLinearOperator]
    num_blocks: int
    _in_sizes: list[int]
    _out_sizes: list[int]
    # NOTE(quill): either use the non-static defined `split_indices` along with `eqx.filter_{...}`,
    #   or use regular JAX function transforms with `split_indices` declared as static.
    #   I'm personally a fan of the explicit declaration, but it seems that this is not the
    #   suggested approach: https://github.com/patrick-kidger/equinox/issues/154.
    #   (It is worth noting that `lineax` itself does use explicit static declarations, such as
    #       in `PyTreeLinearOperator`.)
    # split_indices: Integer[list, "..."]
    split_indices: Integer[list, "..."] = eqx.field(static=True)

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
        self._in_sizes = [block.in_size() for block in self.blocks]
        self._out_sizes = [block.out_size() for block in self.blocks]
        # NOTE(quill): `int(idx)` is needed else `eqx.filter_{...}` doesn't filter out these indices
        #   (Since I've declared `split_indices` as static this isn't necessary, but there's no true cost
        #       to keeping.)
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
        return _BlockOperator([block.T for block in self.blocks])
    
    def in_structure(self):
        return ShapeDtypeStruct(shape=(self.in_size(),), dtype=self.blocks[0].in_structure().dtype)

    def out_structure(self):
        return ShapeDtypeStruct(shape=(self.out_size(),), dtype=self.blocks[0].out_structure().dtype)
    
    def in_size(self) -> int:
        return sum(self._in_sizes)

    def out_size(self) -> int:
        return sum(self._out_sizes)
    
@lx.is_symmetric.register(_BlockOperator)
def _(op):
    return all(lx.is_symmetric(block) for block in op.blocks)


def _to_2D_symmetric_psd_func_op(A: lx.AbstractLinearOperator, v: Float[Array, "_B _d"]) -> lx.FunctionLinearOperator:
    """Collapse a batch of 2D operators.

    Helper function that takes a batch of AbstractLinearOperators that map 1D Arrays to 1D Arrays
    and wraps it in a `FunctionLinearOperator` so that

    TODO(quill): finish docstring
    NOTE / TODO (quill): This function does not currently work when `MatrixLinearOperator`s
    are stacked on top of one another (see the NOTE in `jax_transform_playground.py`).
    I'm unsure if this method will also fail on other linops, but I'm assuming it will.
    For the `MatrixLinearOperator` the problem is that its `mv` definition uses `jnp.matmul`, so
    when a batch of vectors is supplied to the function, a batch of matrix-matrix multiplications
    are performed (as opposed to applying each matrix operator to a single vector). I need to consider
    how I can specify to only apply `mv` to a single vector at a time.
    
    Parameters
    ----------
    `A` : lx.AbstractLinearOperator
        Each operator within the batch is assumed to be symmetric and positive semidefinite.
    This is a flopless 
    """
    
    A_shape_dtype = eval_shape(A.mv, v)
    A_shape = A_shape_dtype.shape

    def mv(dx: Float[Array, " _Bd"]):
        dx = jnp.reshape(dx, (A_shape[0], A_shape[-1]))
        out = A.mv(dx)
        return jnp.ravel(out)
    
    breakpoint()
    
    in_structure = ShapeDtypeStruct(shape=(A_shape[0]*A_shape[-1],), dtype=A_shape_dtype.dtype)
    return lx.FunctionLinearOperator(mv, input_structure=in_structure, tags=[lx.symmetric_tag, lx.positive_semidefinite_tag])
