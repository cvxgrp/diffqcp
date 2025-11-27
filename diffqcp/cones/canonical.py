"""Functionality for projecting onto cones and forming the Jacobians of these projections.

NOTE(quill): Main tech debt
- Not implementing `as_matrix` for the Jacobian Operators

NOTE(quill): Design patterns/decisions
- Trying to follow the abstract/final pattern as described here: https://docs.kidger.site/equinox/pattern/
- The `mv` methods of the projector Jacobians needed to handle the case where their defining data is
    2D as this occurs when `vmap`ing the `proj_dproj` of instantiations of `AbstractConeProjector`s.
    (Remembering that `eqx.Module`s are PyTrees, and unless otherwise specified by `out_axes` in
    `vmap`, a 0th batch index will be added to all arrays.)
- It seemed necessary to add `jax.lax.cond` within the SOC Jacobian operator vs. instantiating
    the class with a flag like `case1`, `case2`, or `case3`. TODO(quill): interrogate this reasoning
    then type it up.

TODO(quill): unimportant for `diffqcp` purposes, but allow the cone ops to work on `PyTrees`.
TODO(quill): add ability to compute `proj` or `dproj` (i.e., don't have to compute both)
    -> again, unimportant for `diffqcp`, but would be nice if you want to provide a JAX cone
    projection library.
"""
from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import lineax as lx
from lineax import AbstractLinearOperator
import equinox as eqx
from jaxtyping import Array, Float

from .abstract_projector import AbstractConeProjector
from .pow import PowerConeProjector
from diffqcp.linops import _BlockLinearOperator
from diffqcp._helpers import _to_int_list

ZERO = "z"
NONNEGATIVE = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"
POW = 'p'
# Note we don't define a POW_DUAL cone as we stick with SCS convention
# and use -alpha to create a dual power cone.

# The ordering of _CONES matches SCS.
_CONES = [ZERO, NONNEGATIVE, SOC, PSD, EXP, EXP_DUAL, POW]

if jax.config.jax_enable_x64:
    EPS = 1e-12
else:
    EPS = 1e-6


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


class _ZeroConeProjectorJacobian(lx.AbstractLinearOperator):
    # NOTE(quill): this Jacobian operator already works on arbitrarily-dimensioned arrays.
    #   i.e., it already can operate on batches of points to project.
    x: Float[Array, "*B n"]
    onto_dual: bool = eqx.field(static=True) # NOTE(quill): known at compile time

    def __init__(self, x: Float[Array, "*B n"], onto_dual: bool):
        # self.shape_dtype = jax.eval_shape(lambda: x) # NOTE(quill): this didn't work with `vmap`
        self.x = x # NOTE(quill): this is hacky; see if you can store shape w/o storing array.
        self.onto_dual = onto_dual
    
    def mv(self, dx: Float[Array, "*B n"]):
        if not self.onto_dual:
            return jnp.zeros_like(dx)
        else:
            return dx
        
    def as_matrix(self):
        raise NotImplementedError("`_ZeroConeProjectorJacobian`'s `as_matrix` method is"
                             + " yet implemented.")

    def transpose(self) -> lx.AbstractLinearOperator:
        # NOTE(quill): while the projector is not self-dual, the Jacobian of the
        #   projection in either case is symmetric.
        return self

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)
    
    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_ZeroConeProjectorJacobian)
def _(op):
    return True


class ZeroConeProjector(AbstractConeProjector):
    
    onto_dual: bool

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        if self.onto_dual:
            return (x, _ZeroConeProjectorJacobian(x=x, onto_dual=True))
        else:
            return (jnp.zeros_like(x), _ZeroConeProjectorJacobian(x=x, onto_dual=False))


class NonnegativeConeProjector(AbstractConeProjector):

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        proj_x = jnp.maximum(x, 0)
        dproj_x = lx.DiagonalLinearOperator(0.5 * (jnp.sign(x) + 1.0))
        return proj_x, dproj_x


def _soc_jacobian_one_dimensional_mv(
    dx: Float[Array, " n"], t: Float[Array, " 1"], z: Float[Array, " n-1"], unit_z: Float[Array, " n-1"], norm_z: Float[Array, " 1"]
) -> Float[Array, " n"]:
    """
    NOTE(quill): I separated this from `_ProjSecondOrderConeJacobian` so that I didn't have to do anything "hacky"
        to `vmap`.
    """
    
    def identity_case():
        return dx
    
    def zero_case():
        return jnp.zeros_like(dx)
    
    def proj_case():
        dt, dz = dx[0], dx[1:]
        first_entry = jnp.array([dt * norm_z + z @ dz])
        second_chunk = (dt * z + (t + norm_z)*dz
                        - t * unit_z * (unit_z @ dz))
        output = jnp.concatenate([first_entry, second_chunk])
        return (1.0 / (2.0 * norm_z)) * output 
    
    return jax.lax.cond(norm_z <= t + EPS,
                              identity_case,
                              lambda: jax.lax.cond(norm_z <= -t,
                                                   zero_case,
                                                   proj_case))


class _ProjSecondOrderConeJacobian(lx.AbstractLinearOperator):
    """Jacobian operator of the projection onto the second-order cone."""
    t: Float[Array, "*B 1"]
    z: Float[Array, " *B n-1"]
    unit_z: Float[Array, "*B n-1"]
    norm_z: Float[Array, ""]
    x: Float[Array, "*B n"]
    # _shape_dtype: jax.ShapeDtypeStruct = eqx.field(static=True)
    # _ndim: int = eqx.field(static=True)

    def __init__(
        self, t: Float[Array, "*B 1"], z: Float[Array, " *B n-1"], unit_z: Float[Array, "*B n-1"], norm_z: Float[Array, ""], x: Float[Array, "*B n"]
    ):
        self.t, self.z = t, z
        self.unit_z, self.norm_z = unit_z, norm_z
        self.x = x
        # self._shape_dtype = jax.eval_shape(lambda: x)
        # self._ndim = jnp.ndim(x)
    
    def mv(self, dx: Float[Array, "*B n"]):
        dx_num_dims = jnp.ndim(dx)
        z_num_dims = jnp.ndim(self.z)

        if dx_num_dims != z_num_dims:
            raise ValueError("Dimension mismatch between the supplied vector `dx`"
                             + " and the dimension of the Jacobian operator's arrays."
                             + f" `dx` is {dx_num_dims}D while the operator's"
                             + f" arrays are {z_num_dims}D.")
        elif dx_num_dims == 1:
            return _soc_jacobian_one_dimensional_mv(dx, self.t, self.z, self.unit_z, self.norm_z)
        elif dx_num_dims == 2:
            return eqx.filter_vmap(_soc_jacobian_one_dimensional_mv)(dx, self.t, self.z, self.unit_z, self.norm_z)
        elif dx_num_dims == 3:
            # third case is needed when we batch projections and have multiple SOCs with the same dimension
            return eqx.filter_vmap(eqx.filter_vmap(_soc_jacobian_one_dimensional_mv))(dx, self.t, self.z, self.unit_z, self.norm_z)
        else:
            raise ValueError(f"The vector `dx` must be 1D or 2D. The supplied vector is {dx_num_dims}D.")

    def as_matrix(self):
        raise NotImplementedError("`_ProjSecondOrderConeJacobian`'s `as_matrix` is not implemented.")
    
    def transpose(self):
        return self
    
    def in_structure(self):
        return jax.eval_shape(lambda: self.x)
    
    def out_structure(self):
        # symmetric
        return self.in_structure()
    
@lx.is_symmetric.register(_ProjSecondOrderConeJacobian)
def _(op):
    return True
    

class _BatchedProjSecondOrderJacobian(lx.AbstractLinearOperator):
    
    batched_jacobians: _ProjSecondOrderConeJacobian
    original_point: Float[Array, "*batch Bn"]
    original_point_two_d_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self, batched_jacobians: _ProjSecondOrderConeJacobian, original_point: Float[Array, "*batch Bn"]
    ):
        self.batched_jacobians = batched_jacobians
        self.original_point = original_point
        self.original_point_two_d_shape = jnp.shape(original_point)
        
    def mv(self, dx: Float[Array, "*batch Bn"]) -> Float[Array, "*batch Bn"]:
        dx_dim = jnp.ndim(dx)
        if dx_dim == 2:
            # `jnp.ndim(original_point)` should equal 3
            # in this case the first dimension is batch dimension
            #   should reshape to be (batch, B, n)
            dx_shape = jnp.shape(dx)
            dx = jnp.reshape(dx, (dx_shape[0],
                                  self.original_point_two_d_shape[0],
                                  self.original_point_two_d_shape[1]))
            out = self.batched_jacobians.mv(dx)
            return jnp.reshape(out, dx_shape)
        elif dx_dim == 1:
            dx = jnp.reshape(dx, self.original_point_two_d_shape)
            out = self.batched_jacobians.mv(dx)
            return jnp.ravel(out)
        else:
            raise ValueError("The functional linear operator that wraps around"
                             + " batched SOC Jacobians espects a 1D or 2D input"
                             + f" perturbation, but receieved a {dx_dim}D input.")
        
    def as_matrix(self):
        raise NotImplementedError("`_BatchedProjSecondOrderJacobian`'s `as_matrix` method is"
                             + " not yet implemented.")
    
    def transpose(self) -> lx.AbstractLinearOperator:
        return self
    
    def in_structure(self):
        curr_shape_dtype = jax.eval_shape(lambda: self.original_point)
        curr_shape = curr_shape_dtype.shape
        curr_dtype = curr_shape_dtype.dtype
        if len(curr_shape_dtype.shape) == 3:
            return jax.ShapeDtypeStruct(shape=(curr_shape[0],
                                               curr_shape[1]* curr_shape[2]),
                                        dtype=curr_dtype)
        else:
            # Making the assumption no error elsewhere...
            return jax.ShapeDtypeStruct(shape=(curr_shape[0]*curr_shape[1],),
                                        dtype=curr_dtype)
    
    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_BatchedProjSecondOrderJacobian)
def _(op):
    return True


class _SecondOrderConeProjector(AbstractConeProjector):
    dim: int # TODO(quill): determine if to use static or not
    # TODO(quill; updated): determine whether to keep dim or not
    #   Won't want/need to unless I end up wanting all projectors to keep track of `dim`
    #   they are projecting onto.

    def __check_init__(self):
        if not isinstance(self.dim, int):
            raise ValueError("The private `eqx.Module` `_SecondOrderConeProjector`"
                             + " expects `dims` to be an integer,"
                             + f" but received a {type(self.dims)}")
    
    def proj_dproj(self, x):
        t, z = x[0], x[1:]
        norm_z = jnp.maximum(jla.norm(z), EPS) # safe norm
        unit_z = z / norm_z
        dproj_x = _ProjSecondOrderConeJacobian(t, z, unit_z, norm_z, x)

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
                                                   proj_case))
        
        return proj_x, dproj_x


class SecondOrderConeProjector(AbstractConeProjector):
    dims: list[int] = eqx.field(static=True)
    dims_batches: list[tuple[int, int]] = eqx.field(static=True)
    projectors: list[_SecondOrderConeProjector]

    def __init__(self, dims: list[int]):
        self.dims = dims
        # NOTE(quill): `_collect_cone_batch_info` will only return tuples with 0th element as dtype int.
        self.dims_batches = _collect_cone_batch_info(_group_cones_in_order(dims))
        self.projectors = [_SecondOrderConeProjector(dim=dim_batch[0]) for dim_batch in self.dims_batches]
    
    def proj_dproj(self, x: Float[Array, "*B n"]) -> tuple[Float[Array, "*B n"], AbstractLinearOperator]:
        projs, dproj_ops = [], []
        start_idx = 0
        # NOTE(quill): the following should be unrolled when (JIT) compiled
        for i, dim_batch in enumerate(self.dims_batches):
            projector = self.projectors[i]
            dim = dim_batch[0]
            num_batches = dim_batch[1]
            slice_size = dim*num_batches
            xi = x[start_idx:start_idx+slice_size]
            if num_batches == 1:
                proj_x, dproj_x = projector(xi)
            else:
                # TODO(quill): ensure this is ordering xi the correct way
                xi = jnp.reshape(xi, (num_batches, dim))
                proj_xi, dproj_xi = eqx.filter_vmap(projector)(xi)
                proj_x = jnp.ravel(proj_xi)
                dproj_x = _BatchedProjSecondOrderJacobian(dproj_xi, xi)
            projs.append(proj_x)
            dproj_ops.append(dproj_x)
            start_idx += slice_size
        
        return jnp.concatenate(projs), _BlockLinearOperator(dproj_ops)


def symm_size_to_dim(size: int) -> int:
    return int(size * (size + 1) // 2)


def symm_dim_to_size(dim: int) -> int:
    return int((jnp.sqrt(8 * dim + 1) - 1) // 2)


def jax_symm_dim_to_size(dim: int) -> jnp.ndarray:
    return jnp.floor_divide(jnp.sqrt(8 * dim + 1) - 1, 2).astype(jnp.int32)


def vec_symm(X: jnp.ndarray) -> jnp.ndarray:
    """Vectorize a symmetric matrix X (per SCS convention)."""
    assert X.ndim == 2, "vec_symm requires that X is a 2-D array."

    size = X.shape[0]
    row_idx, col_idx = jnp.triu_indices(size)

    # Grab upper-triangular elements (equivalent to lower-triangular in column-major order)
    vec = X[row_idx, col_idx]

    # Scale off-diagonals by sqrt(2)
    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=X.dtype))
    scale = jnp.where(row_idx == col_idx, 1.0, sqrt2)

    return vec * scale


def unvec_symm(x: Float[Array, " d"], size: int) -> Float[Array, "k k"]:
    sqrt2 = jnp.sqrt(jnp.array(2.0, dtype=x.dtype))
    X = jnp.zeros((size, size), dtype=x.dtype)
    idxs = jnp.triu_indices(size)
    X = X.at[idxs].set(x / sqrt2)
    X = X + X.T
    diag = jnp.arange(size)
    X = X.at[diag, diag].set(X[diag, diag] / sqrt2)
    return X


def form_B_block(v1: Float[Array, " n"], v2: Float[Array, " m"]) -> Float[Array, "m n"]:
    v1 = jnp.expand_dims(v1, 0).repeat(v2.shape[0], axis=0)
    block = v1 - jnp.expand_dims(v2, 1)
    block = v1 / block
    return block


def _psd_jacobian_one_dimensional_mv(dx, lambd, Q, B, size):

    def identity_case():
        return dx
    
    def zero_case():
        return jnp.zeros_like(dx)
    
    def proj_case():
        dX = unvec_symm(dx, size)
        out = dX @ Q
        out = Q.T @ out
        out = B * out
        out = out @ Q.T
        out = Q @ out
        return vec_symm(out)
    
    return jax.lax.cond(lambd[0] >= 0,
                        identity_case,
                        lambda: jax.lax.cond(lambd[-1] < 0,
                                             zero_case,
                                             proj_case))


class _ProjPSDConeJacobian(AbstractLinearOperator):
    lambd: Float[Array, " k"]
    Q: Float[Array, "k k"]
    B: Float[Array, "k k"]
    size: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    x: Float[Array, " d"]

    def __init__(self, lambd, Q, B, size, dim, x):
        self.lambd = lambd
        self.Q = Q
        self.B = B
        self.size = size
        self.dim = dim
        self.x = x

    def mv(self, dx: Float[Array, "*B d"]):
        dx_num_dims = jnp.ndim(dx)
        x_num_dims = jnp.ndim(self.x)

        if dx_num_dims != x_num_dims:
            raise ValueError("Dimension mismatch between the supplied vector `dx`"
                             + " and the dimension of the Jacobian operator's arrays."
                             + f" `dx` is {dx_num_dims}D while the operator's"
                             + f" arrays are {x_num_dims}D.")
        elif dx_num_dims == 1:
            return _psd_jacobian_one_dimensional_mv(dx, self.lambd, self.Q, self.B, self.size)
        elif dx_num_dims == 2:
            return eqx.filter_vmap(_psd_jacobian_one_dimensional_mv)(dx, self.lambd, self.Q, self.B, self.size)
        elif dx_num_dims == 3:
            # third case is needed when we batch projections and have multiple PSDs with the same dimension
            return eqx.filter_vmap(eqx.filter_vmap(_psd_jacobian_one_dimensional_mv))(dx, self.lambd, self.Q, self.B, self.size)
        else:
            raise ValueError(f"The vector `dx` must be 1D or 2D. The supplied vector is {dx_num_dims}D.")

    def as_matrix(self):
        raise NotImplementedError("PSD Jacobian as_matrix not implemented.")

    def transpose(self):
        return self

    def in_structure(self):
        return jax.eval_shape(lambda: self.x)

    def out_structure(self):
        # symmetric
        return self.in_structure()

@lx.is_symmetric.register(_ProjPSDConeJacobian)
def _(op):
    return True

class _BatchedProjPSDConeJacobian(AbstractLinearOperator):
    batched_jacobians: _ProjPSDConeJacobian
    original_point: Float[Array, "*batch Bd"]
    original_point_two_d_shape: tuple[int, int] = eqx.field(static=True)

    def __init__(self, batched_jacobians, original_point):
        self.batched_jacobians = batched_jacobians
        self.original_point = original_point
        self.original_point_two_d_shape = jnp.shape(original_point)

    def mv(self, dx: Float[Array, "*batch Bd"]):
        dx_dim = jnp.ndim(dx)
        if dx_dim == 2:
            dx_shape = jnp.shape(dx)
            dx = jnp.reshape(dx, (dx_shape[0], self.original_point_two_d_shape[0], self.original_point_two_d_shape[1]))
            out = self.batched_jacobians.mv(dx)
            return jnp.reshape(out, dx_shape)
        elif dx_dim == 1:
            dx = jnp.reshape(dx, self.original_point_two_d_shape)
            out = self.batched_jacobians.mv(dx)
            return jnp.ravel(out)
        else:
            raise ValueError("Batched PSD Jacobian expects 1D or 2D input.")

    def as_matrix(self):
        raise NotImplementedError("Batched PSD Jacobian as_matrix not implemented.")

    def transpose(self):
        return self

    def in_structure(self):
        curr_shape_dtype = jax.eval_shape(lambda: self.original_point)
        curr_shape = curr_shape_dtype.shape
        curr_dtype = curr_shape_dtype.dtype
        if len(curr_shape_dtype.shape) == 3:
            return jax.ShapeDtypeStruct(shape=(curr_shape[0], curr_shape[1] * curr_shape[2]), dtype=curr_dtype)
        else:
            return jax.ShapeDtypeStruct(shape=(curr_shape[0] * curr_shape[1],), dtype=curr_dtype)

    def out_structure(self):
        return self.in_structure()

@lx.is_symmetric.register(_BatchedProjPSDConeJacobian)
def _(op):
    return True


def form_B_block_full(lam_pos_full: jnp.ndarray,
                      lam_neg_full: jnp.ndarray,
                      pos_mask: jnp.ndarray,
                      neg_mask: jnp.ndarray,
                      eps: float = 1e-12) -> jnp.ndarray:
    """
    Shape-stable replacement for form_B_block(lam_pos, lam_neg).
    - lam_pos_full, lam_neg_full: length-n arrays where entries not in the mask are zeroed.
    - pos_mask, neg_mask: boolean masks of length n (pos_mask True => pos eigen)
    Returns an (n, n) matrix whose nonzeros live only at rows where neg_mask and cols where pos_mask.
    The original form_B_block returned a (n_pos, n_neg) block computed as:
        v1 = expand(v1)  # repeated rows
        block = v1 - expand(v2)
        block = v1 / block
    which is equivalent to block_ij = lam_pos[j] / (lam_pos[j] - lam_neg[i]).
    """
    # Broadcast to (n_rows, n_cols): rows use lam_neg, cols use lam_pos
    # row i, col j -> lam_pos[j] / (lam_pos[j] - lam_neg[i])
    # Add tiny eps to avoid exact divide-by-zero (adjust/remove if you require exact)
    denom = (lam_pos_full[None, :] - lam_neg_full[:, None]) + eps
    block = lam_pos_full[None, :] / denom  # shape (n, n)

    # Keep only top-right support: rows = neg (<= k), cols = pos (> k)
    tr_mask = (neg_mask[:, None] & pos_mask[None, :])
    return jnp.where(tr_mask, block, 0.0)


class _PSDConeProjector(AbstractConeProjector):
    size: int
    dim: int

    def proj_dproj(self, x: jnp.ndarray):
        """
        Assumes x is the vectorized symmetric input (SCS ordering) and returns (proj_x, dproj_obj).
        """
        # Reconstruct symmetric matrix
        X = unvec_symm(x, self.size)

        # eigh (returns ascending eigenvalues)
        lambd, Q = jnp.linalg.eigh(X)  # (n,), (n,n)

        def identity_case():
            B = jnp.zeros((self.size, self.size), dtype=x.dtype)
            return x, _ProjPSDConeJacobian(lambd, Q, B, self.size, self.dim, x)

        def zero_case():
            B = jnp.zeros((self.size, self.size), dtype=x.dtype)
            return jnp.zeros_like(x), _ProjPSDConeJacobian(lambd, Q, B, self.size, self.dim, x)

        def general_case():
            # PSD projection
            lambd_pos = jnp.clip(lambd, min=0.0)
            proj_X = Q @ (lambd_pos[..., None] * Q.T)
            proj_x = vec_symm(proj_X)

            # Find last negative index k
            idx = jnp.arange(self.size)
            neg_mask = lambd < 0
            k = jnp.max(jnp.where(neg_mask, idx, -1))  # scalar index

            # Build shape-stable masks
            neg_mask_full = idx <= k
            pos_mask_full = ~neg_mask_full

            # Create padded eigenvalue vectors (zeros where mask false)
            lam_neg_full = jnp.where(neg_mask_full, lambd, 0.0)
            lam_pos_full = jnp.where(pos_mask_full, lambd, 0.0)

            # Bottom-right identity block (on indices i>k)
            eye = jnp.eye(self.size, dtype=x.dtype)
            br_mask = (pos_mask_full[:, None] & pos_mask_full[None, :])
            B_br = jnp.where(br_mask, eye, 0.0)

            # Top-right block (shape-stable full (n,n) matrix but with support only at rows<=k, cols>k)
            B_tr_full = form_B_block_full(lam_pos_full, lam_neg_full, pos_mask_full, neg_mask_full)

            # Final B: bottom-right identity + top-right + its transpose (symmetry)
            B = B_br + B_tr_full + B_tr_full.T

            return proj_x, _ProjPSDConeJacobian(lambd, Q, B, self.size, self.dim, x)

        # Choose which case applies:
        # - if smallest eigenvalue >= 0: identity (already PSD)
        # - elif largest eigenvalue < 0: zero case (all negative -> projection zero)
        # - else general_case
        result_proj, result_dproj = jax.lax.cond(
            lambd[0] >= 0,
            identity_case,
            lambda: jax.lax.cond(
                lambd[-1] < 0,
                zero_case,
                general_case
            )
        )

        return result_proj, result_dproj


class PSDConeProjector(AbstractConeProjector):
    sizes: list[int] = eqx.field(static=True)
    dims: list[int] = eqx.field(static=True)
    size_batches: list[tuple[int, int]] = eqx.field(static=True)
    dim_batches: list[tuple[int, int]] = eqx.field(static=True)
    projectors: list[_PSDConeProjector]

    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.dims = [symm_size_to_dim(s) for s in sizes]
        self.size_batches = _collect_cone_batch_info(_group_cones_in_order(sizes))
        self.dim_batches = _collect_cone_batch_info(_group_cones_in_order(self.dims)) # NOTE(quill): this is lazy
        self.projectors = [_PSDConeProjector(size=size_batch[0], dim=symm_size_to_dim(size_batch[0])) for size_batch in self.size_batches]
    
    def proj_dproj(self, x: Float[Array, "*B n"]) -> tuple[Float[Array, "*B n"], AbstractLinearOperator]:
        projs, dproj_ops = [], []
        start_idx = 0
        # NOTE(quill): the following should be unrolled when (JIT) compiled
        for i, dim_batch in enumerate(self.dim_batches):
            projector = self.projectors[i]
            dim = dim_batch[0]
            num_batches = dim_batch[1]
            slice_size = dim*num_batches
            xi = x[start_idx:start_idx+slice_size]
            if num_batches == 1:
                proj_x, dproj_x = projector(xi)
            else:
                # TODO(quill): ensure this is ordering xi the correct way
                xi = jnp.reshape(xi, (num_batches, dim))
                proj_xi, dproj_xi = eqx.filter_vmap(projector)(xi)
                proj_x = jnp.ravel(proj_xi)
                dproj_x = _BatchedProjPSDConeJacobian(dproj_xi, xi)
            projs.append(proj_x)
            dproj_ops.append(dproj_x)
            start_idx += slice_size
        
        return jnp.concatenate(projs), _BlockLinearOperator(dproj_ops)
        
            
class ProductConeProjector(AbstractConeProjector):
    projectors: list[AbstractConeProjector]
    dims: list[int] = eqx.field(static=True)
    split_indices: list[int] = eqx.field(static=True)
    # onto_dual: bool = eqx.field(static=True)
    
    def __init__(self, cones: dict[str, int | list[int] | list[float]], onto_dual: bool=False):
        projectors = []
        dims = []
        for cone_key in _CONES:
            if cone_key not in cones:
                continue
            val = cones[cone_key]
            if cone_key == ZERO:
                # Zero cone: val is an int (number of zeros)
                projectors.append(ZeroConeProjector(onto_dual=onto_dual))
                dims.append(val)
            elif cone_key == NONNEGATIVE:
                # Nonnegative cone: val is an int (number of nonnegatives)
                projectors.append(NonnegativeConeProjector())
                dims.append(val)
            elif cone_key == SOC:
                # SOC: val is a list of ints (dimensions of each SOC block)
                if len(val) > 0:
                    projectors.append(SecondOrderConeProjector(val))
                    dims.append(sum(val))
            elif cone_key == EXP:
                if val > 0:
                    raise ValueError("The exponential cone is not yet supported.")
            elif cone_key == EXP_DUAL:
                if val > 0:
                    raise ValueError("The dual exponential cone is not yet supported.")
            elif cone_key == POW:
                if len(val) > 0:
                    projectors.append(PowerConeProjector(val, onto_dual=onto_dual))
                    dims.append(3 * np.size(val))
            elif cone_key == PSD:
                if len(val) > 0:
                    projectors.append(PSDConeProjector(val))
                    dims.append(sum([symm_size_to_dim(s) for s in val]))
            else:
                raise ValueError(f"The cone corresponding to cone key: {cone_key}"
                                 + " is not known.")
        self.projectors = projectors
        self.dims = dims
        self.split_indices = _to_int_list(np.cumsum(dims[:-1]))

    def proj_dproj(self, x):
        chunks = jnp.split(x, self.split_indices, axis=-1)
        
        projs, dproj_ops = [], []
        for chunk, projector in zip(chunks, self.projectors):
            proj_xi, dproj_xi = projector(chunk)
            projs.append(proj_xi)
            dproj_ops.append(dproj_xi)

        # NOTE(quill): when `vmap`ping, the concatenation of `projs` will work
        #   as desired, but the `mv` of `_BlockLinearOperator` now needs to know
        #   how to handle 2D input AND the attributes (leaves) of the operator
        #   now have a batch dimension.
        return jnp.concatenate(projs, axis=-1), _BlockLinearOperator(dproj_ops)