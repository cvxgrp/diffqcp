"""
Projection onto exponential cone and the derivative of the projection.

The projection routines are from

    Projection onto the exponential cone: a univariate root-finding problem,
        by Henrik A. Fridberg, 2021.

And even more specifically, the routines are a port from SCS's implementation
of Fridberg's routines (see exp_cone.c).

The derivative of the projection is a port from https://github.com/cvxgrp/diffcp/blob/master/cpp/src/cones.cpp.
"""

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import equinox as eqx
import lineax as lx
from jaxtyping import Float, Array

from diffqcp.cones.abstract_projector import AbstractConeProjector

EXP_CONE_INF_VALUE = ...
EXP_CONE_INF_VALUE = ...
MAX_ITER = 40
EPS = ...
TOL = ...
CONE_THRESH = ...

def _is_finite(x: Float[Array, " "]) -> bool:
    return jnp.abs(x) < EXP_CONE_INF_VALUE


def _clip(
    x: Float[Array, "..."],
    l: Float[Array, " "],
    u: Float[Array, " "]
):
    return jnp.maximum(l, jnp.minimum(u, x))


def hfun(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
):
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = jnp.exp(rho)
    expnegrho = jnp.exp(-rho)

    f = (((rho - 1)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -
        (rho * (rho - 1) + 1) * t0)
    
    return f


def hfun_and_grad_hfun(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
):
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = jnp.exp(rho)
    expnegrho = jnp.exp(-rho)
    
    f = (((rho - 1)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -
        (rho * (rho - 1) + 1) * t0)
    df = (rho * r0 + s0) * exprho + (r0 - (rho - 1) * s0) * expnegrho - (2 * rho - 1) * t0
    return (f, df)


def pomega(rho: Float[Array, " "]) -> Float[Array, " "]:
    val = jnp.exp(rho) / (rho * (rho - 1) + 1.0)
    return jnp.where(rho < 2.0, jnp.minimum(val, jnp.exp(2.0) / 3.0), val)


def domega(rho: Float[Array, " "]) -> Float[Array, " "]:
    val = -jnp.exp(-rho) / (rho * (rho - 1) + 1.0)
    return jnp.where(rho > -1.0, jnp.maximum(val, -jnp.exp(1.0) / 3.0), val)


def ppsi(v: Float[Array, " "]) -> Float[Array, " "]:
    r0, s0 = v[0], v[1]
    sqrt_arg = r0 * r0 + s0 * s0 - r0 * s0
    sqrt_term = jnp.sqrt(sqrt_arg)

    psi1 = (r0 - s0 + sqrt_term) / r0
    psi2 = -s0 / (r0 - s0 - sqrt_term)

    psi = jnp.where(r0 > s0, psi1, psi2)
    return ((psi - 1.0) * r0 + s0) / (psi * (psi - 1.0) + 1.0)


def proj_primal_exp_cone_heuristic(v: Float[Array, " "]) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Computes heuristic (cheap) projection onto EXP cone.
    
    :param v: Point to (heuristically) project onto the (primal) exponential cone.
    :type v: Float[Array, " "]
    :return: Heuristic projection and distance between this projection and provided point.
    :rtype: tuple[Array, Array]
    """
    r0, s0, t0 = v[0], v[1], v[2]
    
    vp = jnp.empty_like(v)
    vp[0] = jnp.minimum(r0, 0)
    vp[1] = 0
    vp[2] = jnp.maximum(t0, 0)

    dist = jla.norm(v - vp)

    def non_interior_case():
        return vp, dist

    def interior_case():
        tp = jnp.maximum(t0, s0 * jnp.exp(r0 / s0))
        newdist = tp - t0

        def new_dist_case():
            vp[0] = r0
            vp[1] = s0
            vp[2] = tp
            
            return vp, newdist
        
        return jax.lax.cond(newdist < dist,
                            new_dist_case,
                            lambda: vp, dist)

    vp, dist = jax.lax.cond(s0 > 0,
                            interior_case,
                            non_interior_case)

    return vp, dist


def proj_polar_exp_cone_heuristic(v: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Computes heuristic (cheap) projection onto polar EXP cone.
    
    :param v: 1D array of size three to heuristically project onto EXP cone.
    :type v: jax.Array
    :return: Description
    :rtype: tuple[Array, Array]
    """
    r0, s0, t0 = v[0], v[1], v[2]

    vd = jnp.empty_like(v)
    vd[0] = 0
    vd[1] = jnp.minimum(s0, 0)
    vd[2] = jnp.minimum(t0, 0)
    dist = jla.norm(v - vd)

    def non_interior_case():
        return vd, dist

    def interior_case():
        td = jnp.maximum(t0, -r0 * jnp.exp(s0 / r0 - 1))
        newdist = t0 - td

        def new_dist_case():
            vd[0] = r0
            vd[1] = s0
            vd[2] = td
            
            return vd, newdist
        
        return jax.lax.cond(newdist < dist,
                            new_dist_case,
                            lambda: vd, dist)

    vd, dist = jax.lax.cond(r0 > 0,
                            interior_case,
                            non_interior_case)


def exp_search_bracket(
    v: Float[Array, "3"],
    pdist: Float[Array, " "],
    ddist: Float[Array, " "]
) -> tuple[Float[Array, " "], Float[Array, " "]]:
    pass


def root_search_binary(
    v: Float[Array, "3"],
    xl: Float[Array, " "],
    xh: Float[Array, " "],
    x: Float[Array, "3"]
) -> Float[Array, "3"]:
    """
    Docstring for root_search_binary
    
    :param v: Description
    :type v: Float[Array, "3"]
    :param xl: Description
    :type xl: Float[Array, " "]
    :param xh: Description
    :type xh: Float[Array, " "]
    :param x: Description
    :type x: Float[Array, "3"]
    :return: Description
    :rtype: Array
    """

    def _binary_search_body(loop_state):
        f = hfun(loop_state["v"], loop_state["x"])

        loop_state = jax.lax.cond(
            f < 0,
            lambda st: {**st, "xl": x},
            lambda st: {**st, "xu": x},
            loop_state
        )

        # binary search step
        x_plus = 0.5 * (loop_state["xl"] + loop_state["xu"])

        loop_state["itn"] += 1
        loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 2, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(jnp.abs(x_plus - x) <= EPS * jnp.maximum(1))


    def condfun(loop_state):
        return loop_state["istop"] == 0
    
    loop_state = {
        "v": v,
        "x": x,
        "xl": xl,
        "xh": xh,
        "itn": 0,
        "istop": 0
    }

    # while loop continues while `condfun == True`, so
    # break when istop != 0.
    jax.lax.while_loop()


def root_search_newton(
    v: Float[Array, "3"],
    xl: Float[Array, " "],
    xu: Float[Array, " "],
    x: Float[Array, " "]
) -> Float[Array, "3"]:
    
    def _newton_body():
        f, df = hfun_and_grad_hfun(v, x)

    def condfun():
        pass

    loop_state = {
        "itn": 0,
        "istop": 0,
        "x": x,
    }


def proj_sol_primal_exp_cone(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Project point onto EXP cone using root of Fridberg function.
    
    :param v: Point to project onto the EXP cone.
    :type v: Float[Array, "3"]
    :param rho: Root of the Fridberg function.
    :type rho: Float[Array, " "]
    :return: The projection of `v` onto the EXP cone and the distance between the point and projection.
    :rtype: tuple[Array, Array]
    """
    
    vp = jnp.empty_like(v)
    linrho = (rho - 1) * v[0] + v[1]
    exprho = jnp.exp(rho)

    def case1():
        quadrho = rho * (rho - 1) + 1
        vp[0] = rho * linrho / quadrho
        vp[1] = linrho / quadrho
        vp[2] = exprho * linrho / quadrho
        dist = jla.norm(vp - v)
        return vp, dist

    def case2():
        vp[0] = 0
        vp[1] = 0
        vp[2] = EXP_CONE_INF_VALUE
        dist = EXP_CONE_INF_VALUE
        return vp, dist

    return jax.lax.cond(linrho > 0 and _is_finite(exprho),
                        case1,
                        case2)


def proj_sol_polar_exp_cone(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Project point onto polar EXP cone using root of Fridberg function.
    
    :param v: Point to project onto the polar EXP cone.
    :type v: Float[Array, "3"]
    :param rho: Root of the Fridberg function.
    :type rho: Float[Array, " "]
    :return: Projection of `v` onto the polar cone and the distance between the point and projection.
    :rtype: tuple[Array, Array]
    """
    
    vd = jnp.empty_like(v)
    linrho = v[0] - rho * v[1]
    exprho = jnp.exp(-rho)

    def case1():
        quadrho = rho * (rho - 1) + 1
        lrho_div_qrho = linrho / quadrho
        vd[0] = lrho_div_qrho
        vd[1] = (1 - rho) * lrho_div_qrho
        vd[2] = -exprho * lrho_div_qrho
        dist = jla.norm(vd - v)
        return vd, dist

    def case2():
        vd[0] = 0
        vd[1] = 0
        vd[2] = -EXP_CONE_INF_VALUE
        dist = EXP_CONE_INF_VALUE
        return vd, dist

    return jax.lax.cond(linrho > 0 and _is_finite(exprho),
                        case1,
                        case2)


def in_exp(v: Float[Array, "3"]) -> bool:
    """Whether `v` is in the EXP cone.
    
    :param v: Point in R^3.
    :type v: Float[Array, "3"]
    :return: Whether `v` is in the EXP cone.
    :rtype: bool
    """
    x, y, z = v[0], v[1], v[2]
    return ((x <= 0 and jnp.abs(y) <= CONE_THRESH and z >= 0)
            or y > 0 and y * jnp.exp(x / y) - z <= CONE_THRESH)


def in_exp_dual(z: Float[Array, "3"]) -> bool:
    """Whether `z` is in the dual EXP cone.
    
    :param z: Point in R^3.
    :type z: Float[Array, "3"]
    :return: Whether `z` is in the dual EXP cone.
    :rtype: bool
    """
    u, v, w = z[0], z[1], z[2]
    return (jnp.abs(u) <= CONE_THRESH and v >= 0 and w >= 0
            or (u < 0 and -u * jnp.exp(v / u) - jnp.exp(1) * w <= CONE_THRESH))


def _proj_exp(v: Float[Array, "3"], onto_dual: bool = False) -> Float[Array, "3"]:
    """Project `v` onto the exponential cone (or its dual). 
    
    :param v: Point in R^3 to project.
    :type v: jax.Array
    :param onto_dual: Whether `v` should be projected onto the EXP cone or its dual.
    :type onto_dual: bool
    :return: Projection of `v` onto the primal or dual EXP cone.
    :rtype: Float[Array, "3"]
    """

    # `onto_dual` is static
    if onto_dual:
        v = -1 * v

    vp, pdist = proj_primal_exp_cone_heuristic(v)
    vd, ddist = proj_polar_exp_cone_heuristic(v)

    err = jnp.abs(vp[0] + vd[0] - v[0])
    err = jnp.maximum(err, jnp.abs(vp[1] + vd[1] - v[1]))
    err = jnp.maximum(err, jnp.abs(vp[2] + vd[2] - v[2]))

    # skip root search if presolve rules apply
    # or optimality conditions are satisfied
    opt = (v[1] <= 0 and v[0] <= 0)
    opt |= jnp.minimum(pdist, ddist) <= TOL
    opt |= err <= TOL and vp @ vd <= TOL
    
    def heuristic_not_optimal():

        xl, xh = exp_search_bracket(v, pdist, ddist)
        rho = root_search_newton(v, xl, xh, 0.5 * (xl + xh))

        def _proj_onto_primal():
            v_hat, dist_hat = proj_sol_primal_exp_cone(v, rho)
            return jax.lax.cond(dist_hat < pdist,
                                v_hat,
                                vp)

        def _proj_onto_dual():
            v_hat, dist_hat = proj_sol_polar_exp_cone(v, rho)
            return jax.lax.cond(dist_hat < ddist,
                                -v_hat,
                                -vd)

        return jax.lax.cond(onto_dual,
                            _proj_onto_dual,
                            _proj_onto_primal)
    
    return jax.lax.cond(opt,
                        lambda: jax.lax.cond(onto_dual, lambda: -vd, vp),
                        heuristic_not_optimal)


def _dproj_exp(
    v: Float[Array, "3"],
    proj_v: Float[Array, "3"],
    onto_dual: bool = False
) -> Float[Array, "3 3"]:
    pass


class _ExponentialConeJacobianOperator(lx.AbstractLinearOperator):
    pass


class ExponentialConeProjector(AbstractConeProjector):

    num_cones: int = eqx.field(static=True)
    onto_dual: bool = eqx.field(static=True)

    def __init__(self, num_cones: int, onto_dual: bool):
        self.num_cones = num_cones
        self.onto_dual = onto_dual

    def proj_dproj(self, x):
        xs = jnp.reshape(x, (self.num_cones, 3))
        
        projs = eqx.filter_vmap(_proj_exp, in_axes=(0, None))(xs)
        jacs = eqx.filter_vmap(_dproj_exp, in_axes=(0, 0, None))(xs, projs, self.onto_dual)

        # It seems like `_ExponentialConeJacobianOperator` may be unecessary, but it also could be
        #   due to potential batching.
        return jnp.ravel(projs), _ExponentialConeJacobianOperator(jacs)