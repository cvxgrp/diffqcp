"""
Notation:
- `x` is the 
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float, Bool

from ._abstract_projector import AbstractConeProjector

if jax.config.jax_enable_x64:
    TOL = 1e-12
else:
    TOL = 1e-6

MAX_ITER = 20

def _pow_calc_xi(
    ri: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """x_i from eq 4. from Hien paper"""
    x = 0.5 * (x + jnp.sqrt(x*x + 4. * alpha * (abs_z - ri) * ri))
    return jnp.maximum(x, TOL)


def _gi(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """gi from diffqcp paper."""
    return 2. * _pow_calc_xi(ri, xi, abs_z, alpha) - xi


def _pow_calc_f(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """Phi from Hien paper."""
    return xi**alpha * yi**(1.-alpha) - ri


def _pow_calc_dxi_dr(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """
    `xi` is an iterate toward the projection of `x` or `y` in `(x, y, z)` toward
    the first element or second element, respectively, in `proj(v)`.
    """
    return alpha * (abs_z - 2.0 * ri) / (2.0 * xi - x)


def _pow_calc_fp(
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    dxidri: Float[Array, ""],
    dyidri: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    alphac = 1 - alpha
    return xi**alpha + yi**alphac * (alpha * dxidri / xi + alphac * dyidri / yi) - 1


def _in_cone(
    x: Float[Array, ""],
    y: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> bool:
    return jnp.logical_and(x >= 0,
                           jnp.logical_and(y >= 0,
                                           TOL + x**alpha * y**(1-alpha) >= abs_z))


def _in_polar_cone(
    u: Float[Array, ""],
    v: Float[Array, ""],
    abs_w: Float[Array, ""],
    alpha: Float[Array, ""]
) -> bool:
    return jnp.logical_and(u <= 0,
                           jnp.logical_and(v <= 0,
                                           TOL + jnp.pow(-u, alpha) * jnp.pow(-v, 1. - alpha) >=
                                            abs_w * alpha**alpha + jnp.pow(1. - alpha, 1. - alpha)))


def _proj_dproj(
    v: Float[Array, " 3"],
    alpha: Float[Array, ""]
) -> tuple[Float[Array, " 3"], Float[Array, "3 3"]]:
    """
    probably need to return a matrix linear operator as derivative?
    """
    x, y, z = v
    abs_z = jnp.abs(z)

    def identity_case():
        return (
            v, jnp.eye(3, dtype=x.dtype)
        )

    def zero_case():
        return (
            jnp.zeros_like(v), jnp.zeros((3, 3), dtype=x.dtype)
        )

    def z_zero_case():
        J = jnp.zeros((3, 3), dtype=v.dtype)
        J = J.at[0, 0].set(0.5 * (jnp.sign(x) + 1.0))
        J = J.at[1, 1].set(0.5 * (jnp.sign(y) + 1.0))

        def case1():  # (x > 0 and y0 < 0 and a_device > 0.5) or (y0 > 0 and x < 0 and alpha < 0.5)
            return 1.0

        def case2():  # (x > 0 and y < 0 and alpha < 0.5) or (y > 0 and x < 0 and alpha > 0.5)
            return 0.0

        def case3():  # a_device == 0.5 and x0 > 0 and y0 < 0
            return x / (2 * jnp.abs(y) + x)

        def case4():
            return y / (2 * jnp.abs(x) + y)

        cond1 = ((x > 0) & (y < 0) & (alpha > 0.5)) | ((y > 0) & (x < 0) & (alpha < 0.5))
        cond2 = ((x > 0) & (y < 0) & (alpha < 0.5)) | ((y > 0) & (x < 0) & (alpha > 0.5))
        cond3 = (alpha == 0.5) & (x > 0) & (y < 0)

        J22 = jax.lax.cond(
            cond1, case1,
            lambda: jax.lax.cond(
                cond2, case2,
                lambda: jax.lax.cond(
                    cond3, case3,
                    case4
                )
            )
        )

        J = J.at[2, 2].set(J22)
        proj_v = jnp.array([jnp.maximum(x, 0), jnp.maximum(y, 0), 0.0], dtype=v.dtype)
        return proj_v, J
        
    
    def solve_case():

        jax.debug.print("IN solve CASE")
        
        def _solve_while_body(loop_state):
            # NOTE(quill): we're purposefully using both `i` and `j`.
            #   The former (which is in the function names) is denoting
            #   an element in a vector while the latter is being used to denote
            #   an interation count.
            loop_state["xj"] = _pow_calc_xi(loop_state["rj"], x, abs_z, alpha)
            loop_state["yj"] = _pow_calc_xi(loop_state["rj"], y, abs_z, 1.0 - alpha)
            fj = _pow_calc_f(loop_state["xj"], loop_state["yj"], loop_state["rj"], alpha)
            
            dxdr = _pow_calc_dxi_dr(loop_state["rj"], loop_state["xj"], x, abs_z, alpha)
            dydr = _pow_calc_dxi_dr(loop_state["rj"], loop_state["yj"], y, abs_z, 1.-alpha)
            fp = _pow_calc_fp(loop_state["xj"], loop_state["yj"], dxdr, dydr, alpha)

            loop_state["rj"] = jnp.maximum(loop_state["rj"] - fj / fp, 0)
            loop_state["rj"] = jnp.minimum(loop_state["rj"], abs_z)

            loop_state["itn"] += 1
            loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 2, loop_state["istop"])
            loop_state["istop"] = jax.lax.select(jnp.abs(fj) <= TOL, 1, loop_state["istop"])

            return loop_state

        def condfun(loop_state):
            return loop_state["istop"] == 0

        loop_state = {
            "xj": 0,
            "yj": 0,
            "rj": abs_z / 2,
            "istop": 0,
            "itn": 0
        }

        loop_state = jax.lax.while_loop(condfun, _solve_while_body, loop_state)

        r_star = loop_state["rj"]
        x_star = loop_state["xj"]
        y_star = loop_state["yj"]
        z_star = jax.lax.cond(z < 0, lambda: -r_star, lambda: r_star)
        proj_v = jnp.array([x_star, y_star, z_star])
        a = alpha
        aa = a * a
        ac = 1 - alpha
        acac = ac * ac

        two_r = 2 * r_star
        sign_z = jnp.sign(z)
        gx = _gi(r_star, x, abs_z, a)
        gy = _gi(r_star, y, abs_z, a)
        frac_x = (a * x) / gx
        frac_y = (ac * y) / gy
        T = - (frac_x + frac_y)
        L = 2 * abs_z - two_r

        gxgy = gx * gy
        rL = r_star * L
        J = jnp.zeros((3, 3), dtype=x.dtype)
        J = J.at[0, 0].set(0.5 + x / (2 * gx) + (aa * (abs_z - two_r) * rL) / (gx * gx))
        J = J.at[1, 1].set(0.5 + y / (2 * gy) + (acac * (abs_z - two_r) * rL) / (gy * gy))
        J = J.at[2, 2].set(r_star / abs_z + (r_star / abs_z) * T * L)
        J = J.at[0, 1].set(rL * acac * (abs_z - two_r) / gxgy)
        J = J.at[1, 0].set(J[0, 1])
        J = J.at[0, 2].set(sign_z * a * rL / gx)
        J = J.at[2, 0].set(J[0, 2])
        J = J.at[1, 2].set(sign_z * ac * rL / gy)
        J = J.at[2, 1].set(J[1, 2])

        jax.debug.print("solve answer: {sa}", sa=proj_v)
        jax.debug.print("loop state: {ls}", ls=loop_state)
        return proj_v, J

    return jax.lax.cond(_in_cone(x, y, abs_z, alpha),
                        identity_case,
                        lambda: jax.lax.cond(
                            _in_polar_cone(x, y, abs_z, alpha),
                            zero_case,
                            lambda: jax.lax.cond(
                                abs_z <= TOL,
                                z_zero_case,
                                solve_case
                            )))


def _pow_cone_jacobian_mv(
    dx: Float[Array, "num_cones 3"],
    jacobians: Float[Array, "num_cones 3 3"],
    is_dual: Bool[Array, " num_cones"],
    num_cones: int,
):
    # num cones could be 1.
    dx_batch = jnp.reshape(dx, (num_cones, 3))
    Jdx = eqx.filter_vmap(lambda y: jacobians @ y,
                            in_axes=(0, 0), out_axes=0)
    mv_dual = dx_batch - Jdx
    mv = jnp.where(is_dual[:, None], mv_dual, Jdx)
    return jnp.ravel(mv)


class _PowerConeJacobianOperator(lx.AbstractLinearOperator):

    batched_jacobians: Float[Array, "*num_batches num_cones 3 3"]
    is_dual: Bool[Array, " num_cones"]
    num_cones: int

    def __init__(
        self,
        batched_jacobians: Float[Array, "*num_batches num_cones 3 3"],
        is_dual: Bool[Array, " num_cones"],
    ):
        self.batched_jacobians = batched_jacobians
        self.is_dual = is_dual
        self.num_cones = jnp.size(is_dual)

    def mv(self, dx: Float[Array, "*batch num_cones*3"]):
        # cases:
        # 1. single power cone -> 1D input
        # 2. multiple power cones -> 1D input
        # 3. batch over single -> 2D input
        # 4. batch over many -> 2D input
        ndim = jnp.ndim(dx)
        if ndim == 1:
            return _pow_cone_jacobian_mv(dx, self.batched_jacobians, self.is_dual, self.num_cones)
        else:
            return eqx.filter_vmap(_pow_cone_jacobian_mv,
                                   in_axes=(0, 0, None, None),
                                   out_axes=0)

    def as_matrix(self):
        raise NotImplementedError("Power Cone Jacobian `as_matrix` not implemented.")
    
    def transpose(self):
        return self

    def in_structure(self):
        ndim = jnp.ndim(self.batched_jacobians)
        shape = jnp.shape(self.batched_jacobians)
        # will we always have `ndim >= 3` via my handling of input in `proj_dproj`?
        if ndim == 2:
            # single Jacobian
            return jax.ShapeDtypeStruct(shape=(3,),
                                        dtype=self.batched_jacobians.dtype)
        elif ndim == 3:
            # either a batch of single power cone or 
            # many power cones
            return jax.ShapeDtypeStruct(shape=(shape[0] * 3,),
                                        dtype=self.batched_jacobians.dtype)
        elif ndim == 4:
            # batch of many power cones.
            # initialization in case of many power cones has 3D `batched_jacobians`,
            # then if we batch we have 4D.
            return jax.ShapeDtypeStruct(shape=(shape[0],
                                               shape[1] * 3),
                                        dtype=self.batched_jacobians.dtype)

    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_PowerConeJacobianOperator)
def _(op):
    return True

class PowerConeProjector(AbstractConeProjector):

    alphas: Float[Array, " num_cones"]
    num_cones: int
    alphas_abs: Float[Array, " num_cones"]
    signs: Float[Array, " num_cones"]
    is_dual: Bool[Array, " num_cones"]
    
    def __init__(self, alphas: list[float], onto_dual: bool):

        self.alphas = jnp.array(alphas)
        self.num_cones = jnp.size(self.alphas)
        self.is_dual = self.alphas < 0
        self.signs = jnp.where(self.alphas < 0, -1.0, 1.0)
        if onto_dual:
            self.signs = -1.0 * self.signs
            self.is_dual = jnp.logical_not(self.is_dual)
        self.alphas_abs = jnp.abs(self.alphas)

    def proj_dproj(self, x):
        batch = jnp.reshape(x, (self.num_cones, 3))
        # negate points being projected onto dual
        batch = batch * self.signs[:, None]

        proj_primal, jacs = eqx.filter_vmap(_proj_dproj, in_axes=(0, 0), out_axes=0)(batch, self.alphas_abs)

        # via Moreau: Pi_K^*(v) = v + Pi_K(-v)
        proj_dual = batch + proj_primal

        proj = jnp.where(self.is_dual[:, None], proj_dual, proj_primal)

        return jnp.ravel(proj), _PowerConeJacobianOperator(jacs, self.is_dual)