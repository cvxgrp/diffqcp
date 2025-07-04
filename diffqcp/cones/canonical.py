import numpy as np
from jax import vmap
import jax.numpy as jnp
import jax.numpy.linalg as la
import lineax as lx
from lineax import AbstractLinearOperator
import equinox as eqx
from abc import abstractmethod
from jaxtyping import Array, Float

from diffqcp._helpers import _to_int_list
from diffqcp.linops import _to_2d_symmetric_psd_func_op, BlockOperator

# TODO(quill): determine if we want to make these public--easier to work with the "magic keys"
#   -> consequential action item: remove the prepended underscore.
_ZERO = "z"
_NONNEGATIVE = "l"
_SOC = "q"
_PSD = "s"
_EXP = "ep"
_EXP_DUAL = "ed"
_POW = 'p'
# Note we don't define a POW_DUAL cone as we stick with SCS convention
# and use -alpha to create a dual power cone.

# The ordering of CONES matches SCS.
CONES = [_ZERO, _NONNEGATIVE, _SOC, _PSD, _EXP, _EXP_DUAL, _POW]

def _parse_cone_dict(cone_dict: dict[str, int | list[int]]
) -> list[tuple[str, int | list[int]]]:
    """Parses SCS-style cone dictionary.

    Parameters
    ----------
    cone_dict : dict[str, int | list[int]]
        A dictionary with keys corresponding to cones and values
        corresponding to their dimension (either integers or lists of integers;
        see the docstring for `compute_derivative` in `qcp.py`).

    Returns
    -------
    list[tuple[str, int | list[int]]]
        A list of two-tuples where the first entry in a tuple is a
        key from the provided dictionary and the second entry in
        that tuple is the corresponding dictionary value.
    """
    return [(cone, cone_dict[cone]) for cone in CONES if cone in cone_dict]


# def _parse_cone_list_for_batches(cones: list[])


def _group_cones_in_order(dims: list[int] | list[float]) -> list[list[int] | list[float]]:
    """Group consecutive same-sized cones while preserving order.
    
    For a list of cone dimensions (so for a specific cone), returns a 
    
    """
    if not isinstance(dims, list):
        raise ValueError(f"`dims` must be a `list`, but a {type(dims)} was provided.")
    
    groups = [[dims[0]]]
    for d in dims[1:]:
        if d == groups[-1][-1]:
            groups[-1].append(d)
        else:
            groups.append([d])

    return groups

def _collect_cone_batch_info(groups: list[list[int] | list[float]]) -> list[tuple[int, int]]:
    """
    Returns a list of tuples such that for the ith group in groups,
    the 0th element in the ith tuple is the dimension of the cone for the ith
    group and the 1st element in the tuple is the number of those 
    """
    dims_batches = []
    for group in groups:
        dims_batches.append((group[0], group[1]))
    return dims_batches


class ConeProjector(eqx.Module):

    # is_dual: AbstractVar[bool]
    is_dual: bool
    dims: list[int]

    @abstractmethod
    def proj_dproj(self, x: Float[Array, " d"]) -> tuple[Float[Array, " d"], AbstractLinearOperator]:
        pass

class ZeroConeProjector(ConeProjector):
    is_dual: bool

    def proj_dproj(self, x: Float[Array, " d"]) -> tuple[Float[Array, " d"], AbstractLinearOperator]:
        pass

class NonnegativeConeProjector(ConeProjector):
    is_dual: bool

    # TODO(quill): return tuple or PyTree...what does `jaxtyping` say?
    def proj_dproj(self, x: Float[Array, " d"]) -> tuple[Float[Array, " d"], AbstractLinearOperator]:
        proj_x = jnp.maximum(x, 0)
        dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
        return proj_x, dproj_x


class SecondOrderConeProjector(ConeProjector):
    is_dual: bool
    dims: list[int]
    dims_batches: list[list[int]]

    def __init__(self, is_dual: bool, dims: list[int]):
        self.dims = dims
        self.dims_batches = _collect_cone_batch_info(_group_cones_in_order(dims))
        # self.split_indices = _to_int_list(np.cumsum(dims[:-1]))

    def _proj_dproj(self, x):
        # TODO(quill):in progress
        n = x.shape[0]
        t, z = x[0], x[1:]
        norm_z = la.norm(z)

        proj_x = 0.5 * (1 + t / norm_z) * jnp.concatenate([jnp.array([norm_z]), z])
        unit_z = z / norm_z
    
    def proj_dproj(self, x: Float[Array, " d"]) -> tuple[Float[Array, " d"], AbstractLinearOperator]:
        projs, dproj_ops = [], []
        start_idx = 0
        for dim_batch in self.dims_batches:
            dim = dim_batch[0]
            num_batches = dim_batch[1]
            slice_size = dim*num_batches
            xi = x[start_idx:start_idx+slice_size]
            if num_batches == 1:
                proj_x, dproj_x = self._proj_dproj(xi)
            else:
                xi = jnp.reshape(xi, (num_batches, dim))
                proj_xi, dproj_xi = vmap(self._proj_dproj)(xi)
                proj_x = jnp.ravel(xi)
                dproj_x = _to_2d_symmetric_psd_func_op(dproj_xi, xi)
            projs.append(proj_x)
            dproj_ops.append(dproj_x)
            start_idx += slice_size
        
        return jnp.concatenate(projs), BlockOperator(dproj_ops)
        
            

class ProductConeProjector(ConeProjector):
    pass

def _construct_product_cone(
    cones: dict[str, int | list[int] | list[float]],
    dual: bool = False
) -> ConeProjector:
    pass