"""
Projection onto exponential cone and the derivative of the projection.

The projection routines are from

    Projection onto the exponential cone: a univariate root-finding problem,
        by Henrik A. Fridberg, 2021.

And even more specifically, the routines are a port from SCS's implementation
of Fridberg's routines (see exp_cone.c).

The derivative of the projection is taken from https://github.com/cvxgrp/diffcp/blob/master/cpp/src/cones.cpp.
"""
import math

import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import equinox as eqx
import lineax as lx
from jaxtyping import Float, Integer, Array

from diffqcp.cones.abstract_projector import AbstractConeProjector

EXP_CONE_INF_VALUE = 1e15
CONE_THRESH = 1e-6

if jax.config.jax_enable_x64:
    EXP_CONE_INF_VALUE = math.sqrt(EXP_CONE_INF_VALUE)
    CONE_THRESH = 0.001


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


def dpsi(v: Float[Array, "3"]):
    r0, s0 = v[0], v[1]

    def case1():
        return (r0 - jnp.sqrt(r0*r0 + s0*s0 - r0*s0)) / s0
    
    def case2():
        return (r0 - s0) / (r0 + jnp.sqrt(r0*r0 + s0*s0 - r0*s0))

    psi = jax.lax.cond(s0 > r0,
                       case1,
                       case2)
    return (r0 - psi*s0) / (psi * (psi - 1.0) + 1.0)


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
    """Generate upper and lower search bounds for root of `hfun`.

    :param v: Point in R^3 being projected.
    :type v: Float[Array, "3"]
    :param pdist: Distance between point and its heuristic projection onto EXP cone.
    :type pdist: Float[Array, " "]
    :param ddist: Distance between point and its heuristic projection onto polar EXP cone.
    :type ddist: Float[Array, " "]
    :return: Lower and upper `hfun` search bounds, respectively.
    :rtype: tuple[Float[Array, " "], Float[Array, " "]]
    """
    EPS = 1e-12
    if not jax.config.enable_x64:
        EPS = math.sqrt(EPS)

    r0, s0, t0 = v[0], v[1], v[2]
    baselow = -EXP_CONE_INF_VALUE
    baseupr = EXP_CONE_INF_VALUE
    low = baselow
    upr = baseupr

    s0m = jnp.minimum(s0, 0.0)
    Dp = jnp.sqrt(jnp.maximum(0.0, pdist * pdist - s0m * s0m))
    r0m = jnp.minimum(r0, 0.0)
    Dd = jnp.sqrt(jnp.maximum(0.0, ddist * ddist - r0m * r0m))

    # t0 > 0  / t0 < 0 branch
    def tpos(_):
        curbnd = jnp.log(t0 / ppsi(v))
        return jnp.maximum(low, curbnd), upr

    def tneg(_):
        curbnd = -jnp.log(-t0 / dpsi(v))
        return low, jnp.minimum(upr, curbnd)

    low, upr = jax.lax.cond(t0 > 0.0, tpos, lambda _: jax.lax.cond(t0 < 0.0, tneg, lambda _: (low, upr), operand=None), operand=None)

    # r0 > 0 branch
    def rpos(_):
        baselow2 = 1.0 - s0 / r0
        low2 = jnp.maximum(low, baselow2)
        tpu = jnp.maximum(EPS, jnp.minimum(Dd, Dp + t0))
        curbnd = jnp.maximum(low2, baselow2 + tpu / r0 / pomega(low2))
        upr2 = jnp.minimum(upr, curbnd)
        return low2, upr2

    low, upr = jax.lax.cond(r0 > 0.0, rpos, lambda _: (low, upr), operand=None)

    # s0 > 0 branch
    def spos(_):
        baseupr2 = r0 / s0
        upr2 = jnp.minimum(upr, baseupr2)
        tdl = -jnp.maximum(EPS, jnp.minimum(Dp, Dd - t0))
        curbnd = jnp.minimum(upr2, baseupr2 - tdl / s0 / domega(upr2))
        low2 = jnp.maximum(low, curbnd)
        return low2, upr2

    low, upr = jax.lax.cond(s0 > 0.0, spos, lambda _: (low, upr), operand=None)

    # guarantee valid bracket
    low = _clip(jnp.minimum(low, upr), baselow, baseupr)
    upr = _clip(jnp.maximum(low, upr), baselow, baseupr)

    # if bracket endpoints different, verify signs and possibly collapse to the closer endpoint
    def adjust_bracket(_):
        fl = hfun(v, low)
        fu = hfun(v, upr)
        def collapse(_):
            pick = jnp.where(jnp.abs(fl) < jnp.abs(fu), low, upr)
            return pick, pick
        return jax.lax.cond(fl * fu > 0.0, collapse, lambda _: (low, upr), operand=None)

    low, upr = jax.lax.cond(low != upr, adjust_bracket, lambda _: (low, upr), operand=None)

    return low, upr


def root_search_binary(
    v: Float[Array, "3"],
    xl: Float[Array, " "],
    xu: Float[Array, " "],
    x: Float[Array, " "],
    perform_search: Integer[Array, " "] = jnp.array(0)
) -> Float[Array, " "]:
    """Binary search method for finding root of `hfun`.
    
    :param v: Point in R^3 being projected.
    :type v: Float[Array, "3"]
    :param xl: Lower search bound for the root of `hfun`.
    :type xl: Float[Array, " "]
    :param xu: Upper search bound for the root of `hfun`.
    :type xu: Float[Array, " "]
    :param x: The intial guess for the root of `hfun`. (A scalar.)
    :type x: Float[Array, " "]
    :param perform_search: Whether the binary search should be performed or not.
        If `perform_search == 0`, the binary search is performed; otherwise it is not.
        This parameter is included to avoid unnecessarily binary searching after
        the Newton search due to wrapping the newton method in `vmap`.
        (See https://kidger.site/thoughts/torch2jax/)
    :type perform_search: Integer[Array, " "], optional, Default to 0.
    :return: The root of `hfun`. (A scalar.)
    :rtype: Float[Array, " "]
    """

    EPS = 1e-12
    MAX_ITER = 40
    
    if not jax.config.jax_enable_x64:
        EPS = math.sqrt(EPS)

    def _binary_search_body(loop_state):

        f = hfun(v, loop_state["x"])

        loop_state = jax.lax.cond(
            f < 0,
            lambda st: {**st, "xl": x},
            lambda st: {**st, "xu": x},
            loop_state
        )

        # binary search step
        x_plus = 0.5 * (loop_state["xl"] + loop_state["xu"])

        # Termination coding:
        # 1: max iterations reached
        # 2: within tolerance
        tol_reached = (jnp.abs(x_plus - x) <= EPS) or (x_plus == loop_state["xl"]) or (x_plus == loop_state["xu"])
        loop_state["itn"] += 1
        loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 1, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(tol_reached, 2, loop_state["istop"])

        # Only commit new_x into x if we will continue (no stop triggered this iter).
        cont = loop_state["istop"] == 0
        loop_state["x"] = jax.lax.select(cont, x_plus, x)

        return loop_state

    def condfun(loop_state):
        return loop_state["istop"] == 0
    
    loop_state = {
        "x": x,
        "xl": xl,
        "xh": xu,
        "itn": 0,
        "istop": perform_search
    }

    # while loop continues while `condfun == True`, so
    # break when istop != 0.
    loop_state = jax.lax.while_loop(condfun, _binary_search_body, loop_state)

    return loop_state["x"]


def root_search_newton(
    v: Float[Array, "3"],
    xl: Float[Array, " "],
    xu: Float[Array, " "],
    x: Float[Array, " "],
    perform_search: Integer[Array, " "] = jnp.array(0)
) -> Float[Array, "3"]:
    """Univariate damped Newton method for finding the root of `hfun`.

    :param v: The point in R^3 being projected.
    :type v: Float[Array, "3"]
    :param xl: A lower bound on the root.
    :type xl: Float[Array, " "]
    :param xu: An upper bound on the root.
    :type xu: Float[Array, " "]
    :param x: Initial guess of the root.
    :type x: Float[Array, " "]
    :param perform_search: Whether the Newton search should be performed or not.
        If `perform_search == 0`, the newton search is performed; otherwise it is not.
        This parameter is included to avoid unnecessarily Newton searching
        due to wrapping the `_proj_exp` in `vmap`.
        (See https://kidger.site/thoughts/torch2jax/)
    :type perform_search: Integer[Array, " "], optional, Default to 0.
    :return: The root of `hfun` (a scalar).
    :rtype: Float[Array, " "]
    """

    EPS = 1e-15
    DFTOL = 1e-13
    MAX_ITER = 20
    LODAMP = 0.05
    HIDAMP = 0.95
    if not jax.config.jax_enable_x64:
        EPS = math.sqrt(EPS)
        DFTOL = math.sqrt(DFTOL)
    
    def _newton_body():

        f, df = hfun_and_grad_hfun(v, loop_state["x"])

        ftol_reached = jnp.abs(f) <= EPS
        
        loop_state["xl"], loop_state["xu"] = jax.lax.cond(f < 0,
                                                          lambda _: (loop_state["x"], loop_state["xu"]),
                                                          lambda _: (loop_state["xl"], loop_state["x"]))
        
        xu_new, xl_new, brk = jax.lax.cond(loop_state["xu"] <= loop_state["xl"],
                                           lambda _: (0.5 * (loop_state["xu"] + loop_state["xl"]), loop_state["xu"], True),
                                           lambda _: (loop_state["xu"], loop_state["xl"], False))
        
        loop_state["xl"] = xl_new
        loop_state["xu"] = xu_new
        
        non_finite_brk = jnp.logical_or(jnp.logical_not(_is_finite(f)), df < DFTOL)

        # Newton step
        x_plus = loop_state["x"] - f / df
        
        tol_reached = jnp.abs(loop_state["x_plus"] - loop_state["x"]) <= EPS * jnp.maximum(1, jnp.abs(x_plus))
        
        def _case_high():
            return jnp.minimum(LODAMP * loop_state["x"] + HIDAMP * loop_state["xu"], loop_state["xu"])

        def _case_low():
            return jnp.maximum(LODAMP * loop_state["x"] + HIDAMP * loop_state["xl"], loop_state["xl"])
        
        new_x = jax.lax.cond(x_plus >= xu,
                             _case_high,
                             lambda _: jax.lax.cond(x_plus <= xl,
                                                    _case_low,
                                                    lambda _: x_plus))

        # Termination coding:
        # 1: max iterations reached
        # 2: `f` within `EPS` of a root
        # 3: `x_plus` within (roughly) `EPS` of `x` (converging).
        # 4: nonfinite values
        # 5: lower bound oversteps upper bound.
        # NOTE(quill): the order of the following does matter. Whichever should cause a break
        # in a Newton step first should come last. For instance, in the original code, if `ftol_reached`
        # then the remainder of the `_newton_body` is not executed. Thus since the last `select`
        # will overwrite the previous `select`s if the conditional is true, we place
        # the `ftol_reached` `select` last.
        loop_state["itn"] += 1
        loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 1, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(tol_reached, 3, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(non_finite_brk, 4, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(brk, 5, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(ftol_reached, 2, loop_state["istop"])

        cont = loop_state["istop"] == 0
        loop_state["x"] = jax.lax.select(cont, new_x, loop_state["x"])

        return loop_state

    def condfun():
        return loop_state["istop"] == 0

    loop_state = {
        "itn": 0,
        "istop": perform_search,
        "x": x,
        "x_plus": x,
        "xl": xl,
        "xu": xu,
    }

    loop_state = jax.lax.while_loop(condfun, _newton_body, loop_state)

    do_binary = jax.lax.select(loop_state["itn"] < MAX_ITER, 1, 0)

    return jax.lax.cond(loop_state["itn"] < MAX_ITER,
                        lambda _: _clip(loop_state["x"], loop_state["xl"], loop_state["xu"]),
                        lambda _: root_search_binary(v, loop_state["xl"], loop_state["xu"], loop_state["x"], do_binary))


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

    TOL = 1e-8

    if not jax.config.jax_enable_x64:
        TOL = 1e-4

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

    perform_search = jax.lax.select(opt, 1, 0)
    
    def heuristic_not_optimal():

        # NOTE(quill): while we protect against doing a Newton root search if the heuristic
        #   projection is optimal, we stil do many other ops.

        xl, xh = exp_search_bracket(v, pdist, ddist)
        rho = root_search_newton(v, xl, xh, 0.5 * (xl + xh), perform_search)

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
                        lambda _: jax.lax.cond(onto_dual, lambda: -vd, vp),
                        heuristic_not_optimal)


def _dproj_exp(
    v: Float[Array, "3"],
    proj_v: Float[Array, "3"],
    onto_dual: bool = False
) -> Float[Array, "3 3"]:
    """Form the Jacobian of the projection onto the exponential cone or its dual.
    
    :param v: Point in R^3.
    :type v: Float[Array, "3"]
    :param proj_v: The projection of `v` onto the EXP cone or its dual.
    :type proj_v: Float[Array, "3"]
    :param onto_dual: Whether `v` was projected onto the EXP cone or its dual.
    :type onto_dual: bool
    :return: The Jacobian of the projection onto the dual cone or its 
    """
    
    if onto_dual:
        v = -v

    # branch for both negative special-case (returns 3x3)
    def _both_negative(_):
        J = jnp.zeros((3, 3), dtype=v.dtype)
        J = J.at[0, 0].set(1.0)

        def _case1(_):
            return J.at[2, 2].set(1.0)

        return jax.lax.cond(v[2] >= 0.0, _case1, lambda _: J, operand=None)

    # branch for the "general" case (keeps original 4x4 construction)
    def _general_case(_):
        r = proj_v[0]
        s = proj_v[1]
        s = jax.lax.cond(s == 0.0, lambda _: jnp.abs(r), lambda _: s, operand=None)
        l = proj_v[2] - v[2]
        alpha = jnp.exp(r / s)
        beta = l * r / (s * s) * alpha
        J = jnp.zeros((4, 4), dtype=v.dtype)

        J = J.at[0, 0].set(alpha)
        J = J.at[0, 1].set((( -r + s) / s) * alpha)
        J = J.at[0, 2].set(-1.0)
        # 0,3 remains 0

        J = J.at[1, 0].set(1.0 + (l / s) * alpha)
        J = J.at[1, 1].set(-beta)
        # 1,2 remains 0
        J = J.at[1, 3].set(alpha)

        J = J.at[2, 0].set(-beta)
        J = J.at[2, 1].set(1.0 + beta * (r / s))
        # 2,2 remains 0
        J = J.at[2, 3].set((1.0 - (r / s)) * alpha)

        # 3,0 and 3,1 remain 0
        J = J.at[3, 2].set(1.0)
        J = J.at[3, 3].set(-1.0)

        # NOTE(quill): obviously it's best to avoid explicitly inverting matrices,
        # but in this 3x3 case I'll just be lazy for now.
        # (Do note that clicking through `inv` shows that we are just computing J_inv = solve(J, eye(3)).)
        J_inv = jla.inv(J)

        return J_inv

    # top-level nested conditional using lax.cond for JIT compatibility
    J = jax.lax.cond(
        in_exp(v),
        lambda _: jnp.identity(3, dtype=v.dtype),
        lambda _: jax.lax.cond(
            in_exp_dual(-v),
            lambda __: jnp.zeros((3, 3), dtype=v.dtype),
            lambda __: jax.lax.cond(
                (v[0] < 0.0) and (v[1] < 0.0),
                _both_negative,
                _general_case
            )))
    
    if onto_dual:
        # So this leaves us with Dproj_k_star(v)[dv] = dv - Dproj_k(-v)[dv]
        J = jnp.eye(3, dtype=v.dtype) - J
    
    return J


def _exp_cone_jacobian_mv(
    dx: Float[Array, "num_cones 3"],
    jacobians: Float[Array, "num_cones 3 3"],
):
    num_cones = jnp.shape(jacobians)[0]
    dx = jnp.reshape(dx, (num_cones, 3))
    Jdx = eqx.filter_vmap(lambda jac, y: jac @ y, in_axes=0, out_axes=0)(jacobians, dx)
    return jnp.ravel(Jdx)


class _ExponentialConeJacobianOperator(lx.AbstractLinearOperator):

    jacobians: Float[Array, "*num_batches num_cones 3 3"]

    def __check_init__(self):
        ndim = jnp.ndim(self.jacobians)
        if ndim not in [3, 4]:
            raise ValueError("The `jacobians` argument provided to the `_ExponentialConeJacobianOperator` "
                             f"is {ndim}D, but it must be 3D or 4D.")

    
    def mv(self, dx: Float[Array, "*num_batches num_cones*3"]):
        
        ndim = jnp.ndim(dx)

        if ndim == 1:
            if jnp.ndim(self.jacobians) == 4:
                raise ValueError("Batched Exponential cone Jacobians cannot be applied to a 1D input.")
            
            return _exp_cone_jacobian_mv(dx, self.jacobians, jnp.shape(self.jacobians)[0])
        elif ndim == 2:
            return eqx.filter_vmap(_exp_cone_jacobian_mv,
                                   in_axes=(0, 0), out_axes=(0))(dx, self.jacobians)
        else:
            raise ValueError("The `_ExponentialConeJacobianOperator` can only be applied to 1D or 2D inputs "
                             f"but the provided vector is {ndim}D.")

    def as_matrix(self):
        raise NotImplementedError("Exponential Cone Jacobian `as_matrix` not implemented.")
    
    def tranpose(self):
        return self
    
    def in_structure(self):

        ndim = jnp.ndim(self.jacobians)
        shape = jnp.shape(self.jacobians)
        dtype = self.jacobians.dtype

        if ndim == 3:
            # non-batched case
            return jax.ShapeDtypeStruct(shape=(shape[0] * 3,),
                                        dtype=dtype)
        elif ndim == 4:
            # batched case
            return jax.ShapeDtypeStruct(shape=(shape[0], shape[1] * 3),
                                        dtype=dtype)

    def out_structure(self):
        return self.in_structure()
    
@lx.is_symmetric.register(_ExponentialConeJacobianOperator)
def _(op):
    return True
        

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