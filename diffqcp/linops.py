"""General (i.e., not cone-specific) linear operators that are not implemented in `lineax`.

Note that these operators were purposefully made "private" since they are solely implemented
to support functionality required by `diffqcp`. They **should not** be accessed as if they
were true atoms implemented in `lineax`.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jax import ShapeDtypeStruct

from diffqcp._helpers import _to_int_list


def _dense_from_mv(op: lx.AbstractLinearOperator):
    """`as_matrix` for an unbatched operator: one `mv` per column.

    Intended for testing and small problems. Operators whose leaves carry a
    batch axis (from `vmap`ping a projector) have no single matrix; materialise
    them inside the `vmap` instead.
    """
    struct = op.in_structure()
    if len(struct.shape) != 1:
        raise NotImplementedError(
            f"{type(op).__name__}.as_matrix is only defined for unbatched (1D) operators,"
            f" but its input structure has shape {struct.shape}."
        )
    return jax.vmap(op.mv, in_axes=1, out_axes=1)(jnp.eye(struct.shape[0], dtype=struct.dtype))


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
    # TODO(quill): make this a JAX array so goes onto device.

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
        results = [op.mv(xi) for op, xi in zip(self.blocks, chunks, strict=False)]
        return jnp.concatenate(results, axis=-1)

    def as_matrix(self):
        return _dense_from_mv(self)

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

@lx.conj.register(_BlockLinearOperator)
def _(op):
    return op
