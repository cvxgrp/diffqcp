"""Projection onto (products of) PSD cones (self-dual), SCS vectorization.

See `CONVENTIONS.md` for the vectorization and the Jacobian formula.
"""
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from diffqcp.linops import _BlockLinearOperator, _dense_from_mv

from ._grouping import _collect_cone_batch_info, _group_cones_in_order
from .abstract_projector import AbstractConeProjector


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
        return _dense_from_mv(self)

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
    original_point_two_d_shape: tuple[int, ...] = eqx.field(static=True)

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
        return _dense_from_mv(self)

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


def form_B_block_full(lam_pos_full: Array,
                      lam_neg_full: Array,
                      pos_mask: Array,
                      neg_mask: Array,
                      eps: float = 1e-12) -> Array:
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
            # (`cast`: jnp.where's stubs include the 1-argument tuple form.)
            lam_neg_full = cast(Array, jnp.where(neg_mask_full, lambd, 0.0))
            lam_pos_full = cast(Array, jnp.where(pos_mask_full, lambd, 0.0))

            # Bottom-right block of ones (on indices i, j > k). This must be all
            # ones, not the identity: the two only coincide when there is a
            # single positive eigenvalue.
            br_mask = (pos_mask_full[:, None] & pos_mask_full[None, :])
            B_br = jnp.where(br_mask, 1.0, 0.0).astype(x.dtype)

            # Top-right block (shape-stable full (n,n) matrix but with support only at rows<=k, cols>k)
            B_tr_full = form_B_block_full(lam_pos_full, lam_neg_full, pos_mask_full, neg_mask_full)

            # Final B: bottom-right ones + top-right + its transpose (symmetry)
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
