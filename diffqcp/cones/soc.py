"""Projection onto (products of) second-order cones (self-dual).

Each block is `(t, z)` with `t` scalar; consecutive cones of equal dimension are
projected together with one `vmap`.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import lineax as lx
from jaxtyping import Array, Float
from lineax import AbstractLinearOperator

from diffqcp.linops import _BlockLinearOperator, _dense_from_mv

from ._grouping import _collect_cone_batch_info, _group_cones_in_order
from .abstract_projector import AbstractConeProjector

EPS = 1e-12 if jax.config.read("jax_enable_x64") else 1e-06


def _soc_jacobian_one_dimensional_mv(
    dx: Float[Array, " n"],
    t: Float[Array, ""],
    z: Float[Array, " n-1"],
    unit_z: Float[Array, " n-1"],
    norm_z: Float[Array, ""],
) -> Float[Array, " n"]:
    """Apply the Jacobian of the SOC projection at the point `(t, z)` to `dx`.

    Identity if `||z|| <= t` (inside the cone), zero if `||z|| <= -t` (inside the
    polar cone), and otherwise
    `(1 / (2 ||z||)) [[||z||, z^T], [z, (t + ||z||) I - t u u^T]] dx` with
    `u = z / ||z||`. Kept separate from the operator class so it can be `vmap`ped.
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
    """Jacobian of the projection onto one second-order cone, at `x = (t, z)`.

    Stores the pieces of `x` needed by `_soc_jacobian_one_dimensional_mv`. When
    built under `vmap` (several same-sized cones, and possibly an outer batch
    axis), every array carries the corresponding leading axes and `mv` accepts a
    2D or 3D `dx` to match.

    **Arguments:**
    - `t`: the first (scalar) entry of `x`.
    - `z`: the remaining `n - 1` entries of `x`.
    - `unit_z`: `z / ||z||` (with a safeguarded norm).
    - `norm_z`: the safeguarded norm `max(||z||, EPS)`.
    - `x`: the point itself; used for `in_structure`.
    """

    t: Float[Array, "*B"]
    z: Float[Array, "*B n-1"]
    unit_z: Float[Array, "*B n-1"]
    norm_z: Float[Array, "*B"]
    x: Float[Array, "*B n"]

    def __init__(
        self,
        t: Float[Array, "*B"],
        z: Float[Array, "*B n-1"],
        unit_z: Float[Array, "*B n-1"],
        norm_z: Float[Array, "*B"],
        x: Float[Array, "*B n"],
    ):
        self.t, self.z = t, z
        self.unit_z, self.norm_z = unit_z, norm_z
        self.x = x

    def mv(self, dx: Float[Array, "*B n"]) -> Float[Array, "*B n"]:
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

    def as_matrix(self) -> Float[Array, "n n"]:
        return _dense_from_mv(self)

    def transpose(self) -> "_ProjSecondOrderConeJacobian":
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        return jax.eval_shape(lambda: self.x)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        # symmetric
        return self.in_structure()

@lx.is_symmetric.register(_ProjSecondOrderConeJacobian)
def _(op: _ProjSecondOrderConeJacobian) -> bool:
    return True


class _BatchedProjSecondOrderJacobian(lx.AbstractLinearOperator):
    """Jacobian of the projection onto `num` consecutive SOCs of the same dimension.

    Wraps the `vmap`ped `_ProjSecondOrderConeJacobian` (whose arrays have a leading
    axis of length `num`) so that it acts on the flat `num * n` slice of the cone
    vector: `mv` reshapes `dx` to `(num, n)`, applies each cone's Jacobian, and
    flattens the result. If the projector itself was `vmap`ped over an outer batch,
    `dx` is 2D, `(batch, num * n)`.

    **Arguments:**
    - `batched_jacobians`: the per-cone Jacobians, built under `vmap`.
    - `original_point`: the points that were projected, shape `(num, n)`.
    """

    batched_jacobians: _ProjSecondOrderConeJacobian
    original_point: Float[Array, "*batch num n"]
    original_point_two_d_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self, batched_jacobians: _ProjSecondOrderConeJacobian, original_point: Float[Array, "num n"]
    ):
        self.batched_jacobians = batched_jacobians
        self.original_point = original_point
        self.original_point_two_d_shape = jnp.shape(original_point)

    def mv(self, dx: Float[Array, "*batch num*n"]) -> Float[Array, "*batch num*n"]:
        dx_dim = jnp.ndim(dx)
        if dx_dim == 2:
            # `jnp.ndim(self.original_point)` equals 3 in this case: the first
            # axis is the outer batch axis, so reshape `dx` to (batch, num, n).
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

    def as_matrix(self) -> Float[Array, "num*n num*n"]:
        return _dense_from_mv(self)

    def transpose(self) -> lx.AbstractLinearOperator:
        return self

    def in_structure(self) -> jax.ShapeDtypeStruct:
        curr_shape_dtype = jax.eval_shape(lambda: self.original_point)
        curr_shape = curr_shape_dtype.shape
        curr_dtype = curr_shape_dtype.dtype
        if len(curr_shape_dtype.shape) == 3:
            return jax.ShapeDtypeStruct(shape=(curr_shape[0],
                                               curr_shape[1]* curr_shape[2]),
                                        dtype=curr_dtype)
        else:
            # Unbatched: `original_point` is 2D, (num, n).
            return jax.ShapeDtypeStruct(shape=(curr_shape[0]*curr_shape[1],),
                                        dtype=curr_dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()

@lx.is_symmetric.register(_BatchedProjSecondOrderJacobian)
def _(op: _BatchedProjSecondOrderJacobian) -> bool:
    return True


class _SecondOrderConeProjector(AbstractConeProjector):
    """Projector onto a single second-order cone of dimension `dim`.

    See `SecondOrderConeProjector` for the cone. `proj_dproj` takes a 1D `x` of
    length `dim`; it is `vmap`ped to project several cones at once.

    **Arguments:**
    - `dim`: the dimension `n` of the cone (a Python `int`).
    """

    dim: int
    # TODO(quill): determine whether to keep `dim` (and whether it should be
    #   static). Only needed if all projectors end up tracking their dimension.

    def __check_init__(self) -> None:
        if not isinstance(self.dim, int):
            raise ValueError("The private `eqx.Module` `_SecondOrderConeProjector`"
                             + " expects `dims` to be an integer,"
                             + f" but received a {type(self.dim)}")

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], _ProjSecondOrderConeJacobian]:
        """Project `x = (t, z)` onto the second-order cone.

        **Returns:**

        A 2-tuple of the projection (`x` if `||z|| <= t`, `0` if `||z|| <= -t`,
        otherwise `((1 + t / ||z||) / 2) (||z||, z)`) and its Jacobian at `x` as a
        `_ProjSecondOrderConeJacobian`.
        """
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
    """Projector onto a product of second-order (Lorentz) cones.

    Each cone of dimension `n` is

        K_soc = {(t, z) in R x R^(n-1) : ||z||_2 <= t},

    stored in the cone vector with `t` first. The second-order cone is self-dual,
    so there is no `onto_dual` flag. The input is the `q` block of the cone vector:
    the cones concatenated in the order given by `dims`.

    **Arguments:**
    - `dims`: the dimension of each cone, in order (the `q` entry of the SCS cone
        dictionary).
    """

    dims: list[int] = eqx.field(static=True)
    dims_batches: list[tuple[int, int]] = eqx.field(static=True)
    projectors: list[_SecondOrderConeProjector]

    def __init__(self, dims: list[int]):
        self.dims = dims
        # NOTE(quill): `_collect_cone_batch_info` will only return tuples with 0th element as dtype int.
        self.dims_batches = _collect_cone_batch_info(_group_cones_in_order(dims))
        self.projectors = [_SecondOrderConeProjector(dim=dim_batch[0]) for dim_batch in self.dims_batches]

    def proj_dproj(self, x: Float[Array, " n"]) -> tuple[Float[Array, " n"], AbstractLinearOperator]:
        """Project `x` onto the product of second-order cones.

        **Arguments:**
        - `x`: a 1D array of length `sum(dims)`. To project a batch of points,
            `vmap` this method; the returned operator then accepts the matching 2D
            input.

        **Returns:**

        A 2-tuple of the projection (same shape as `x`) and its Jacobian, a block
        diagonal `_BlockLinearOperator` with one block per group of consecutive
        same-sized cones.
        """
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
                # Row `j` of `xi` is the `j`-th cone of the group (row-major
                # reshape of the contiguous slice).
                xi = jnp.reshape(xi, (num_batches, dim))
                proj_xi, dproj_xi = eqx.filter_vmap(projector)(xi)
                proj_x = jnp.ravel(proj_xi)
                dproj_x = _BatchedProjSecondOrderJacobian(dproj_xi, xi)
            projs.append(proj_x)
            dproj_ops.append(dproj_x)
            start_idx += slice_size

        return jnp.concatenate(projs), _BlockLinearOperator(dproj_ops)
