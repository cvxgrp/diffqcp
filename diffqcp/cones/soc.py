"""Projection onto (products of) second-order cones (self-dual)."""
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
        return _dense_from_mv(self)

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
    original_point_two_d_shape: tuple[int, ...] = eqx.field(static=True)

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
        return _dense_from_mv(self)

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
                             + f" but received a {type(self.dim)}")

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
