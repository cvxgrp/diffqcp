"""
TODO(quill): unimportant for `diffqcp` purposes, but allow the cone ops to work on `PyTrees`.
TODO(quill): add ability to compute `proj` or `dproj` (i.e., don't have to compute both)
    -> again, unimportant for `diffqcp`, but would be nice if you want to provide a JAX cone
    projection library.
"""

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import lineax as lx
from lineax import AbstractLinearOperator
import equinox as eqx
from abc import abstractmethod
from jaxtyping import Array, Float

from diffqcp._linops import _BlockOperator

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

if jax.config.jax_enable_x64:
    EPS = 1e-12
else:
    EPS = 1e-6

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
        dims_batches.append((group[0], len(group)))
    return dims_batches


class ConeProjector(eqx.Module):

    # is_dual: AbstractVar[bool]
    # NOTE(quill): list of floats needed for the power cone
    # NOTE(quill): might need to make `dims` (and `is_dual`) static?
    dims: int | list[int] | list[float]
    is_dual: bool | None = None

    @abstractmethod
    def proj_dproj(self, x: Float[Array, " _d"]) -> tuple[Float[Array, " _d"], AbstractLinearOperator]:
        pass

    def __call__(self, x: Float[Array, " _d"]):
        return self.proj_dproj(x)


class _ZeroConeProjectorJacobian(lx.AbstractLinearOperator):
    # NOTE(quill): this Jacobian operator already works on arbitrarily-dimensioned arrays.
    #   i.e., it already can operate on batches of points to project.
    x: Float[Array, "..."]
    onto_dual: bool = eqx.field(static=True) # NOTE(quill): known at compile time

    def __init__(self, x: Float[Array, "..."], onto_dual: bool):
        # self.shape_dtype = jax.eval_shape(lambda: x)
        self.x = x # NOTE(quill): this is hacky; see if you can store shape w/o storing array.
        self.onto_dual = onto_dual
    
    def mv(self, dx: Float[Array, "..."]):
        if not self.onto_dual:
            return jnp.zeros_like(dx)
        else:
            return dx
        
    def as_matrix(self):
        raise NotImplementedError("`_ZeroConeProjectorJacobian`'s `as_matrix` method is"
                             + " yet implemented.")

    def transpose(self):
        # NOTE(quill): while the projector is not self-dual, the Jacobian of the
        #   projection in either case is symmetric.
        return self

    def in_structure(self):
        # return self.shape_dtype
        return jax.eval_shape(lambda: self.x)
    
    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_ZeroConeProjectorJacobian)
def _(op):
    return True


class ZeroConeProjector(ConeProjector):
    is_dual: bool = False

    def proj_dproj(self, x: Float[Array, " _d"]) -> tuple[Float[Array, " _d"], AbstractLinearOperator]:
        if self.is_dual:
            return (x, _ZeroConeProjectorJacobian(x=x, onto_dual=True))
        else:
            return (jnp.zeros_like(x), _ZeroConeProjectorJacobian(x=x, onto_dual=False))


class NonnegativeConeProjector(ConeProjector):

    def proj_dproj(self, x: Float[Array, " _d"]) -> tuple[Float[Array, " _d"], AbstractLinearOperator]:
        proj_x = jnp.maximum(x, 0)
        dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
        return proj_x, dproj_x


class _ProjSecondOrderConeJacobian(lx.AbstractLinearOperator):
    """
    Jacobian operator of the projection onto the second-order cone when
    `norm(z) >= abs(t)` for x = (t, z) in R^{n + 1}.
    """
    t: Float[Array, "*B 1"]
    z: Float[Array, " *B n-1"]
    unit_z: Float[Array, "*B n-1"]
    norm_z: float
    # TODO(quill): get rid of `n` and move other possible Jacobians here like with NN cone.
    n: int = eqx.field(static=True)

    def _mv_batched(self, dx: Float[Array, "B n"]):
        dt, dz = dx[..., 0], dx[..., 1:]
        batched_inner = lambda x, y : jnp.sum(x * y, axis=1)
        first_entries = dt[:, jnp.newaxis] * self.norm_z + batched_inner(self.z, dz)
        second_chunks = (dt[:, jnp.newaxis] * self.z + (self.t[:, jnp.newaxis] + self.norm_z)
                         - self.t[:, jnp.newaxis] * self.unit_z * batched_inner(self.unit_z, dz))
        output = jnp.column_stack([first_entries, second_chunks])
        return (1.0 / (2.0 * self.norm_z)) * output

    def _mv_single(self, dx: Float[Array, " n"]):
        dt, dz = dx[0], dx[1:]
        first_entry = jnp.array([dt * self.norm_z + self.z @ dz])
        second_chunk = (dt * self.z + (self.t + self.norm_z)
                        - self.t * self.unit_z * (self.unit_z @ dz))
        output = jnp.concatenate([first_entry, second_chunk])
        return (1.0 / (2.0 * self.norm_z)) * output
    
    def _mv(self, dx: Float[Array, "*B n"]):
        pass
    
    def mv(self, dx: Float[Array, "*B n"]):
        """
        So if I don't use `jax.lax.cond` then when this is traced 
        """

        dx_num_dims = len(jnp.shape(dx))
        z_num_dims = len(jnp.shape(self.z))

        if dx_num_dims != z_num_dims:
            raise ValueError(f"Dimension mismatch between ")

        if dx_num_dims == 1:
            return self._mv()

        def proj_case():
            dx_num_dims = len(jnp.shape(dx))
            if dx_num_dims == 1:
                return self._mv(dx)
            elif dx_num_dims == 2:
                return self._mv_batched(dx)
            else:
                raise ValueError("The input to `_ProjSecondOrderConeJacobian.mv` must be"
                                + " a 1D or 2D array.")
            
        result = jax.lax.cond(self.norm_z <= self.t + EPS,
                              identity_case,
                              lambda: jax.lax.cond(norm_z <= -t,
                                                   zero_case,
                                                   proj_case,
                                                   operand=None),
                               operand=None)
        return result

    def as_matrix(self):
        # TODO(quill): `raise ValueError`?
        return jax.vmap(self.mv, in_axis=1, out_axes=1)(jnp.eye(self.n))
    
    def transpose(self):
        return self
    
    def in_structure(self):
        # TODO(quill): I do need to properly implement this.
        orginal_pt = jnp.concatenate([self.t, self.z])
        return 
    
    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_ProjSecondOrderConeJacobian)
def _(op):
    return True
    

class _SecondOrderConeProjector(ConeProjector):
    dims: int # TODO(quill): determine if to use static or not
    is_dual: bool

    def __check_init__(self):
        if not isinstance(self.dims, int):
            raise ValueError("The private `eqx.Module` `_SecondOrderConeProjector`"
                             + " expects `dims` to be an integer,"
                             + f" but received a {type(self.dims)}")
    
    def proj_dproj(self, x):
        n = self.dims
        t, z = x[0], x[1:]
        norm_z = jnp.maximum(jla.norm(z), EPS) # safe norm
        unit_z = z / norm_z
        dproj_x = _ProjSecondOrderConeJacobian(t, z, unit_z, norm_z, n)

        def identity_case():
            return x
        
        def zero_case():
            return jnp.zeros_like(x)
        
        def proj_case():
            return 0.5 * (1 + t / norm_z) * jnp.concatenate([jnp.array([norm_z]), z])
        
        proj_x = jax.lax.cond(norm_z <= t + EPS,
                              identity_case,
                              lambda: jax.lax.cond(norm_z <= -t,
                                                   zero_case,
                                                   proj_case,
                                                   operand=None),
                               operand=None)
        
        return proj_x, dproj_x


class SecondOrderConeProjector(ConeProjector):
    dims: list[int]
    dims_batches: list[tuple[int, int]]
    projectors: list[_SecondOrderConeProjector]

    def __init__(self, dims: list[int], is_dual=None):
        self.dims = dims
        # NOTE(quill): `_collect_cone_batch_info` will only return tuples with 0th element as dtype int.
        self.dims_batches = _collect_cone_batch_info(_group_cones_in_order(dims))
        self.projectors = [_SecondOrderConeProjector(dims=dim_batch[0]) for dim_batch in self.dims_batches]
    
    def proj_dproj(self, x: Float[Array, "*B _d"]) -> tuple[Float[Array, "*B _d"], AbstractLinearOperator]:
        projs, dproj_ops = [], []
        start_idx = 0
        for i, dim_batch in enumerate(self.dims_batches):
            projector = self.projectors[i]
            dim = dim_batch[0]
            num_batches = dim_batch[1]
            slice_size = dim*num_batches
            xi = x[start_idx:start_idx+slice_size]
            if num_batches == 1:
                proj_x, dproj_x = projector(xi)
            else:
                xi = jnp.reshape(xi, (num_batches, dim))
                proj_xi, dproj_x = jax.vmap(projector)(xi)
                proj_x = jnp.ravel(proj_xi)
                # dproj_x = _to_2D_symmetric_psd_func_op(dproj_xi, xi)
            projs.append(proj_x)
            dproj_ops.append(dproj_x)
            start_idx += slice_size
        
        return jnp.concatenate(projs), _BlockOperator(dproj_ops)
        
            
class ProductConeProjector(ConeProjector):
    pass


def _construct_product_cone(
    cones: dict[str, int | list[int] | list[float]],
    dual: bool = False
) -> ConeProjector:
    pass


SecondOrderConeProjector.__init__.__doc__ = r"""
Parameters
----------
- `dims`: list[int]
- `is_dual`: bool, optional
    Whether the projection should be onto the SOC or the dual of the SOC.
"""