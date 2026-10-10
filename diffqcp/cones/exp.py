"""Projection onto (products of) exponential cones, or their duals, and its derivative.

The projection routines are from

    Projection onto the exponential cone: a univariate root-finding problem,
        by Henrik A. Friberg, 2021.

More specifically, they are a port of SCS's implementation of Friberg's
routines (see SCS's `exp_cone.c`). The projection is found from a root `rho` of
the univariate function `hfun`.

The derivative of the projection is taken from https://github.com/cvxgrp/diffcp/blob/master/cpp/src/cones.cpp.
"""
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.numpy.linalg as jla
import lineax as lx
from jaxtyping import Array, Bool, Float, Integer

from diffqcp.cones.abstract_projector import AbstractConeProjector
from diffqcp.linops import _dense_from_mv

EXP_CONE_INF_VALUE = 1e15
CONE_THRESH = 1e-6

if jax.config.read("jax_enable_x64"):
    EXP_CONE_INF_VALUE = math.sqrt(EXP_CONE_INF_VALUE)
    CONE_THRESH = 0.001


def _is_finite(x: Float[Array, " "]) -> Bool[Array, ""]:
    """Whether `|x|` is below the large sentinel value `EXP_CONE_INF_VALUE`."""
    return jnp.abs(x) < EXP_CONE_INF_VALUE


def _clip(
    x: Float[Array, "..."],
    l: Float[Array, " "] | float,
    u: Float[Array, " "] | float,
) -> Float[Array, "..."]:
    """Clip `x` elementwise to the interval `[l, u]`."""
    return jnp.maximum(l, jnp.minimum(u, x))


def hfun(
    v: Float[Array, "3"],
    rho: Float[Array, " "] | float,
) -> Float[Array, " "]:
    """Friberg's univariate function whose root `rho` determines the projection of `v`.

    `h(rho) = ((rho - 1) r0 + s0) exp(rho) - (r0 - rho s0) exp(-rho)
    - (rho (rho - 1) + 1) t0`, for `v = (r0, s0, t0)`.
    """
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = jnp.exp(rho)
    expnegrho = jnp.exp(-rho)

    f = (((rho - 1)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -
        (rho * (rho - 1) + 1) * t0)

    return f


def hfun_and_grad_hfun(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
) -> tuple[Float[Array, " "], Float[Array, " "]]:
    """Return `hfun(v, rho)` and its derivative with respect to `rho`."""
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = jnp.exp(rho)
    expnegrho = jnp.exp(-rho)

    f = (((rho - 1)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -
        (rho * (rho - 1) + 1) * t0)
    df = (rho * r0 + s0) * exprho + (r0 - (rho - 1) * s0) * expnegrho - (2 * rho - 1) * t0
    return (f, df)


def pomega(rho: Float[Array, " "]) -> Float[Array, " "]:
    """Primal-side helper `exp(rho) / (rho (rho - 1) + 1)` for `exp_search_bracket`.

    Clipped to at most `exp(2) / 3` for `rho < 2` (port of SCS's `pomega`).
    """
    val = jnp.exp(rho) / (rho * (rho - 1) + 1.0)
    return jnp.where(rho < 2.0, jnp.minimum(val, jnp.exp(2.0) / 3.0), val)


def domega(rho: Float[Array, " "]) -> Float[Array, " "]:
    """Dual-side helper `-exp(-rho) / (rho (rho - 1) + 1)` for `exp_search_bracket`.

    Clipped to at least `-exp(1) / 3` for `rho > -1` (port of SCS's `domega`).
    """
    val = -jnp.exp(-rho) / (rho * (rho - 1) + 1.0)
    return jnp.where(rho > -1.0, jnp.maximum(val, -jnp.exp(1.0) / 3.0), val)


def ppsi(v: Float[Array, "3"]) -> Float[Array, " "]:
    """Primal-side helper for the bracket on the root of `hfun` when `t0 > 0`.

    Port of SCS's `ppsi`; only `r0, s0 = v[0], v[1]` are used.
    """
    r0, s0 = v[0], v[1]
    sqrt_arg = r0 * r0 + s0 * s0 - r0 * s0
    sqrt_term = jnp.sqrt(sqrt_arg)

    psi1 = (r0 - s0 + sqrt_term) / r0
    psi2 = -s0 / (r0 - s0 - sqrt_term)

    psi = jnp.where(r0 > s0, psi1, psi2)
    return ((psi - 1.0) * r0 + s0) / (psi * (psi - 1.0) + 1.0)


def dpsi(v: Float[Array, "3"]) -> Float[Array, " "]:
    """Dual-side helper for the bracket on the root of `hfun` when `t0 < 0`.

    Port of SCS's `dpsi`; only `r0, s0 = v[0], v[1]` are used.
    """
    r0, s0 = v[0], v[1]

    def case1():
        return (r0 - jnp.sqrt(r0*r0 + s0*s0 - r0*s0)) / s0

    def case2():
        return (r0 - s0) / (r0 + jnp.sqrt(r0*r0 + s0*s0 - r0*s0))

    psi = jax.lax.cond(s0 > r0,
                       case1,
                       case2)
    return (r0 - psi*s0) / (psi * (psi - 1.0) + 1.0)


def proj_primal_exp_cone_heuristic(v: Float[Array, "3"]) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Compute a heuristic (cheap) projection of `v` onto the exponential cone.

    The candidate is the better of `(min(r0, 0), 0, max(t0, 0))` and, when
    `s0 > 0`, `(r0, s0, max(t0, s0 exp(r0 / s0)))`.

    **Arguments:**
    - `v`: the point `(r0, s0, t0)` to (heuristically) project.

    **Returns:**

    The heuristic projection and its distance to `v`.
    """
    r0, s0, t0 = v[0], v[1], v[2]

    vp = jnp.empty_like(v)
    vp = jnp.array([
        jnp.minimum(r0, 0),
        0.0,
        jnp.maximum(t0, 0)
    ], dtype=v.dtype)

    dist = jla.norm(v - vp)

    def non_interior_case():
        return vp, dist

    def interior_case():
        tp = jnp.maximum(t0, s0 * jnp.exp(r0 / s0))
        newdist = tp - t0

        def new_dist_case():
            vp = jnp.array([r0, s0, tp], dtype=v.dtype)

            return vp, newdist

        return jax.lax.cond(newdist < dist,
                            new_dist_case,
                            lambda: (vp, dist))

    vp, dist = jax.lax.cond(s0 > 0,
                            interior_case,
                            non_interior_case)

    return vp, dist


def proj_polar_exp_cone_heuristic(v: Float[Array, "3"]) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Compute a heuristic (cheap) projection of `v` onto the polar exponential cone.

    The candidate is the better of `(0, min(s0, 0), min(t0, 0))` and, when
    `r0 > 0`, `(r0, s0, min(t0, -r0 exp(s0 / r0 - 1)))`.

    **Arguments:**
    - `v`: the point `(r0, s0, t0)` to (heuristically) project.

    **Returns:**

    The heuristic projection onto the polar cone and its distance to `v`.
    """
    r0, s0, t0 = v[0], v[1], v[2]

    vd = jnp.empty_like(v)
    vd = jnp.array([
        0.0,
        jnp.minimum(s0, 0.0),
        jnp.minimum(t0, 0)
    ], dtype=v.dtype)
    dist = jla.norm(v - vd)

    def non_interior_case():
        return vd, dist

    def interior_case():
        td = jnp.minimum(t0, -r0 * jnp.exp(s0 / r0 - 1.0))
        newdist = t0 - td

        def new_dist_case():
            vd = jnp.array([r0, s0, td], dtype=v.dtype)

            return vd, newdist

        return jax.lax.cond(newdist < dist,
                            new_dist_case,
                            lambda: (vd, dist))

    vd, dist = jax.lax.cond(r0 > 0,
                            interior_case,
                            non_interior_case)

    return vd, dist


def exp_search_bracket(
    v: Float[Array, "3"],
    pdist: Float[Array, " "],
    ddist: Float[Array, " "]
) -> tuple[Float[Array, " "], Float[Array, " "]]:
    """Generate lower and upper search bounds for the root of `hfun`.

    **Arguments:**
    - `v`: the point in R^3 being projected.
    - `pdist`: distance between `v` and its heuristic projection onto the
        exponential cone.
    - `ddist`: distance between `v` and its heuristic projection onto the polar
        exponential cone.

    **Returns:**

    The lower and upper bounds, respectively. If the signs of `hfun` at the two
    ends do not bracket a root, both are set to the endpoint with the smaller
    `|hfun|`.
    """
    EPS = 1e-12
    if not jax.config.read("jax_enable_x64"):
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
    perform_search: Integer[Array, " "] = jnp.array(0)  # noqa: B008  (jnp scalar constant)
) -> Float[Array, " "]:
    """Binary search for the root of `hfun`.

    **Arguments:**
    - `v`: the point in R^3 being projected.
    - `xl`: lower search bound for the root.
    - `xu`: upper search bound for the root.
    - `x`: the initial guess for the root.
    - `perform_search`: the search runs only if this is `0`; any other value
        returns `x` unchanged. This avoids needlessly searching when the call is
        `vmap`ped, since `vmap` turns a `lax.cond` into a select that evaluates
        both branches (see https://kidger.site/thoughts/torch2jax/).

    **Returns:**

    The (approximate) root of `hfun`, a scalar.
    """

    EPS = 1e-12
    MAX_ITER = 40

    if not jax.config.read("jax_enable_x64"):
        EPS = math.sqrt(EPS)

    def _binary_search_body(loop_state):

        f = hfun(v, loop_state["x"])

        loop_state = jax.lax.cond(
            f < 0,
            lambda st: {**st, "xl": loop_state["x"]},
            lambda st: {**st, "xu": loop_state["x"]},
            loop_state
        )

        # binary search step
        x_plus = 0.5 * (loop_state["xl"] + loop_state["xu"])

        # Termination coding:
        # 1: max iterations reached
        # 2: within tolerance
        tol_reached = jnp.logical_or(jnp.abs(x_plus - loop_state["x"]) <= EPS, x_plus == loop_state["xl"])
        tol_reached = jnp.logical_or(tol_reached, x_plus == loop_state["xu"])
        loop_state["itn"] += 1
        loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 1, loop_state["istop"])
        loop_state["istop"] = jax.lax.select(tol_reached, 2, loop_state["istop"])

        # Only commit new_x into x if we will continue (no stop triggered this iter).
        cont = loop_state["istop"] == 0
        loop_state["x"] = jax.lax.select(cont, x_plus, loop_state["x"])

        return loop_state

    def condfun(loop_state):
        return loop_state["istop"] == 0

    loop_state = {
        "x": x,
        "xl": xl,
        "xu": xu,
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
    perform_search: Integer[Array, " "] = jnp.array(0)  # noqa: B008  (jnp scalar constant)
) -> Float[Array, " "]:
    """Univariate damped Newton method for the root of `hfun`.

    Falls back to `root_search_binary` if Newton's method uses up its iteration
    budget.

    **Arguments:**
    - `v`: the point in R^3 being projected.
    - `xl`: a lower bound on the root.
    - `xu`: an upper bound on the root.
    - `x`: the initial guess for the root.
    - `perform_search`: the search runs only if this is `0`; any other value
        returns `x` (clipped to `[xl, xu]`). This avoids needlessly searching
        when `_proj_exp_and_polar` is `vmap`ped, since `vmap` turns a `lax.cond`
        into a select that evaluates both branches
        (see https://kidger.site/thoughts/torch2jax/).

    **Returns:**

    The (approximate) root of `hfun`, a scalar.
    """

    EPS = 1e-15
    DFTOL = 1e-13
    MAX_ITER = 20
    LODAMP = 0.05
    HIDAMP = 0.95
    if not jax.config.read("jax_enable_x64"):
        EPS = math.sqrt(EPS)
        DFTOL = math.sqrt(DFTOL)

    def _newton_body(loop_state):

        f, df = hfun_and_grad_hfun(v, loop_state["x"])

        ftol_reached = jnp.abs(f) <= EPS

        loop_state["xl"], loop_state["xu"] = jax.lax.cond(f < 0.0,
                                                          lambda: (loop_state["x"], loop_state["xu"]),
                                                          lambda: (loop_state["xl"], loop_state["x"]))

        xu_new, xl_new, brk = jax.lax.cond(loop_state["xu"] <= loop_state["xl"],
                                           lambda: (0.5 * (loop_state["xu"] + loop_state["xl"]), loop_state["xu"], True),
                                           lambda: (loop_state["xu"], loop_state["xl"], False))

        loop_state["xl"] = xl_new
        loop_state["xu"] = xu_new

        non_finite_brk = jnp.logical_or(jnp.logical_not(_is_finite(f)), df < DFTOL)

        # Newton step
        x_plus = loop_state["x"] - f / df

        tol_reached = jnp.abs(x_plus - loop_state["x"]) <= EPS * jnp.maximum(1, jnp.abs(x_plus))

        def _case_high():
            return jnp.minimum(LODAMP * loop_state["x"] + HIDAMP * loop_state["xu"], loop_state["xu"])

        def _case_low():
            return jnp.maximum(LODAMP * loop_state["x"] + HIDAMP * loop_state["xl"], loop_state["xl"])

        new_x = jax.lax.cond(x_plus >= loop_state["xu"],
                             _case_high,
                             lambda: jax.lax.cond(x_plus <= loop_state["xl"],
                                                    _case_low,
                                                    lambda: x_plus))

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

    def condfun(loop_state):
        return loop_state["istop"] == 0

    loop_state = {
        "itn": 0,
        "istop": perform_search,
        "x": x,
        "xl": xl,
        "xu": xu,
    }

    loop_state = jax.lax.while_loop(condfun, _newton_body, loop_state)

    do_binary = jax.lax.select(loop_state["itn"] < MAX_ITER, 1, 0)
    do_binary = jax.lax.select(perform_search.astype(jnp.bool), perform_search, do_binary)

    return jax.lax.cond(loop_state["itn"] < MAX_ITER,
                        lambda: _clip(loop_state["x"], loop_state["xl"], loop_state["xu"]),
                        lambda: root_search_binary(v, loop_state["xl"], loop_state["xu"], loop_state["x"], do_binary))


def proj_sol_primal_exp_cone(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Recover the projection of `v` onto the exponential cone from a root of `hfun`.

    **Arguments:**
    - `v`: the point to project.
    - `rho`: a root of Friberg's function `hfun(v, .)`.

    **Returns:**

    The candidate projection and its distance to `v`. If it is not well defined
    (`(rho - 1) r0 + s0 <= 0` or `exp(rho)` overflows), returns
    `(0, 0, EXP_CONE_INF_VALUE)` with distance `EXP_CONE_INF_VALUE` so the caller
    keeps its heuristic projection.
    """

    linrho = (rho - 1) * v[0] + v[1]
    exprho = jnp.exp(rho)

    def case1():
        quadrho = rho * (rho - 1) + 1
        vp = jnp.array([
            rho * linrho / quadrho,
            linrho / quadrho,
            exprho * linrho / quadrho
        ], dtype=v.dtype)
        dist = jla.norm(vp - v)
        return vp, dist

    def case2():
        vp = jnp.array([0.0, 0.0, EXP_CONE_INF_VALUE], dtype=v.dtype)
        dist = EXP_CONE_INF_VALUE
        return vp, dist

    return jax.lax.cond(jnp.logical_and(linrho > 0, _is_finite(exprho)),
                        case1,
                        case2)


def proj_sol_polar_exp_cone(
    v: Float[Array, "3"],
    rho: Float[Array, " "]
) -> tuple[Float[Array, "3"], Float[Array, " "]]:
    """Recover the projection of `v` onto the polar exponential cone from a root of `hfun`.

    **Arguments:**
    - `v`: the point to project.
    - `rho`: a root of Friberg's function `hfun(v, .)`.

    **Returns:**

    The candidate projection onto the polar cone and its distance to `v`. If it is
    not well defined (`r0 - rho s0 <= 0` or `exp(-rho)` overflows), returns
    `(0, 0, -EXP_CONE_INF_VALUE)` with distance `EXP_CONE_INF_VALUE` so the
    caller keeps its heuristic projection.
    """

    linrho = v[0] - rho * v[1]
    exprho = jnp.exp(-rho)

    def case1():
        quadrho = rho * (rho - 1) + 1
        lrho_div_qrho = linrho / quadrho
        vd = jnp.array([
            lrho_div_qrho,
            (1 - rho) * lrho_div_qrho,
            -exprho * lrho_div_qrho
        ], dtype=v.dtype)
        dist = jla.norm(vd - v)
        return vd, dist

    def case2():
        vd = jnp.array([
            0.0, 0.0, -EXP_CONE_INF_VALUE
        ], dtype=v.dtype)
        dist = EXP_CONE_INF_VALUE
        return vd, dist

    return jax.lax.cond(jnp.logical_and(linrho > 0, _is_finite(exprho)),
                        case1,
                        case2)


def in_exp(v: Float[Array, "3"]) -> Bool[Array, ""]:
    """Whether `v = (x, y, z)` is in the exponential cone, up to `CONE_THRESH`.

    Tests `y > 0 and y exp(x / y) - z <= CONE_THRESH`, or the closure face
    `x <= 0, |y| <= CONE_THRESH, z >= 0`.
    """
    x, y, z = v[0], v[1], v[2]
    # NOTE(quill): yes, all the parentheses are necessary.
    #   Otherwise we get:
    #   `TypeError: and does not accept dtype float64 at position 0.`
    #   `Accepted dtypes at position 0 are subtypes of integer, bool`
    return (((x <= 0) & (jnp.abs(y) <= CONE_THRESH) & (z >= 0))
            | (y > 0) & (y * jnp.exp(x / y) - z <= CONE_THRESH))


def in_exp_dual(z: Float[Array, "3"]) -> Bool[Array, ""]:
    """Whether `z = (u, v, w)` is in the dual exponential cone, up to `CONE_THRESH`.

    K_exp^* = closure{(u, v, w) : u < 0, -u exp(v / u) <= e w}; the test is
    `u < -CONE_THRESH and -u exp(v / (u + CONE_THRESH)) - e w <= CONE_THRESH`,
    or the closure face `|u| <= CONE_THRESH, v >= 0, w >= 0`.
    """
    u, v, w = z[0], z[1], z[2]
    return ((jnp.abs(u) <= CONE_THRESH) & (v >= 0) & (w >= 0)
            | ((u < -CONE_THRESH) & (-u * jnp.exp(v / (u + CONE_THRESH)) - jnp.exp(1) * w <= CONE_THRESH)))


def _proj_exp_and_polar(v: Float[Array, "3"]) -> tuple[Float[Array, "3"], Float[Array, "3"]]:
    """Project `v` onto the exponential cone and onto its polar cone.

    Starts from the cheap heuristic projections and, unless they are already
    optimal, refines them with Friberg's root search on `hfun`.

    To use this subroutine for projecting onto the dual cone, negate `v` before
    passing it and then negate the polar projection output
    (Pi_{K^*}(v) = -Pi_{K^o}(-v), with K^o the polar cone).

    **Arguments:**
    - `v`: the point in R^3 to project.

    **Returns:**

    The projections of `v` onto the exponential cone and onto its polar cone,
    respectively.
    """

    TOL = 1e-8

    if not jax.config.read("jax_enable_x64"):
        TOL = 1e-4

    vp, pdist = proj_primal_exp_cone_heuristic(v)
    vd, ddist = proj_polar_exp_cone_heuristic(v)

    err = jnp.abs(vp[0] + vd[0] - v[0])
    err = jnp.maximum(err, jnp.abs(vp[1] + vd[1] - v[1]))
    err = jnp.maximum(err, jnp.abs(vp[2] + vd[2] - v[2]))

    # skip root search if presolve rules apply
    # or optimality conditions are satisfied
    opt = jnp.logical_and(v[1] <= 0, v[0] <= 0)
    opt = jnp.logical_or(opt, jnp.minimum(pdist, ddist) <= TOL)
    opt = jnp.logical_or(opt, jnp.logical_and(err <= TOL, vp @ vd <= TOL))

    # NOTE(quill): we pass this integer select to `root_search_newton`
    #   otherwise when `vmap`ping `_proj_exp_and_polar` we are liable
    #   to perform Newton (and binary) searches even when `opt == True`.
    perform_search = jax.lax.select(opt, 1, 0)

    def heuristic_not_optimal():

        # NOTE(quill): while we protect against doing a Newton root search if the heuristic
        #   projection is optimal, we still do many other ops.

        xl, xu = exp_search_bracket(v, pdist, ddist)
        rho = root_search_newton(v, xl, xu, 0.5 * (xl + xu), perform_search)

        def _proj_onto_primal():
            v_hat, dist_hat = proj_sol_primal_exp_cone(v, rho)
            return jax.lax.cond(dist_hat < pdist,
                                lambda: v_hat,
                                lambda: vp)

        def _proj_onot_polar():
            v_hat, dist_hat = proj_sol_polar_exp_cone(v, rho)
            return jax.lax.cond(dist_hat < ddist,
                                lambda: v_hat,
                                lambda: vd)

        return (_proj_onto_primal(), _proj_onot_polar())

    return jax.lax.cond(opt,
                        lambda: (vp, vd),
                        heuristic_not_optimal)


def _dproj_exp(
    v: Float[Array, "3"],
    proj_v: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    """Form the 3 x 3 Jacobian of the projection onto the exponential cone at `v`.

    The regions are: `v` in the cone (identity), `v` in the polar cone (zero),
    `r < 0` and `s < 0` (projection `(r, 0, max(t, 0))`), the projection on the
    face `s = 0`, and the general case, which reads the Jacobian off the inverse
    of a 4 x 4 matrix (as in diffcp).

    To use this subroutine for the derivative of the projection onto the dual
    cone,

    1. negate `v` before passing it to this function,

    2. pass as `proj_v` the projection of the negated `v` onto the (primal)
    exponential cone,

    3. and use `I - J` for the result `J`
    (since Pi_{K^*}(v) = v + Pi_K(-v)).

    **Arguments:**
    - `v`: the point in R^3 being projected.
    - `proj_v`: the projection of `v` onto the exponential cone
        (**not** onto the dual or polar cone).

    **Returns:**

    The Jacobian of the projection onto the exponential cone, evaluated at `v`.
    """
    def _both_negative():
        J = jnp.zeros((3, 3), dtype=v.dtype)
        J = J.at[0, 0].set(1.0)

        def _case1():
            return J.at[2, 2].set(1.0)

        return jax.lax.cond(v[2] > 0.0, _case1, lambda: J)

    def _general_case():
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
        # but in this 4x4 case I'll just be lazy for now.
        # (Do note that clicking through `inv` shows that we are just computing J_inv = solve(J, eye(4)).)
        J_inv = jla.inv(J)

        return J_inv[0:3, 1:4]

    def _on_s_zero_face():
        # The projection lies on the face {(r, 0, t): r <= 0, t >= 0}, where it
        # is (min(r, 0), 0, max(t, 0)) locally. This happens on a set of
        # positive measure that includes points with r > 0 (small r next to a
        # large negative s), so it cannot be detected from the signs of v alone.
        # The general formula divides by s there (0/0 -> NaN).
        return jnp.diag(jnp.array([v[0] < 0.0, False, v[2] > 0.0], dtype=v.dtype))

    J = jax.lax.cond(
        in_exp(v),
        lambda: jnp.identity(3, dtype=v.dtype),
        lambda: jax.lax.cond(
            in_exp_dual(-v),
            lambda: jnp.zeros((3, 3), dtype=v.dtype),
            lambda: jax.lax.cond(
                (v[0] < 0.0) & (v[1] < 0.0) & (jnp.logical_not(jnp.allclose(v[2], 0.0))),
                _both_negative,
                lambda: jax.lax.cond(
                    proj_v[1] == 0.0,
                    _on_s_zero_face,
                    _general_case,
                ))))

    return J


def _exp_cone_jacobian_mv(
    dx: Float[Array, " num_cones*3"],
    jacobians: Float[Array, "num_cones 3 3"],
) -> Float[Array, " num_cones*3"]:
    """Apply the block diagonal matrix with 3 x 3 blocks `jacobians` to the flat `dx`."""
    num_cones = jnp.shape(jacobians)[0]
    dx = jnp.reshape(dx, (num_cones, 3))
    Jdx = eqx.filter_vmap(lambda jac, y: jac @ y, in_axes=(0, 0), out_axes=0)(jacobians, dx)
    return jnp.ravel(Jdx)


class _ExponentialConeJacobianOperator(lx.AbstractLinearOperator):
    """Jacobian of the projection onto a product of exponential cones (or their duals).

    Block diagonal with one dense 3 x 3 block per cone, acting on the flat
    `3 * num_cones` vector. If the projector was `vmap`ped over a batch,
    `jacobians` is 4D and `mv` takes a 2D `(batch, 3 * num_cones)` input.

    **Arguments:**
    - `jacobians`: the per-cone 3 x 3 Jacobians, shape `(num_cones, 3, 3)` or
        `(batch, num_cones, 3, 3)`.
    """

    jacobians: Float[Array, "*num_batches num_cones 3 3"]

    def __check_init__(self) -> None:
        ndim = jnp.ndim(self.jacobians)
        if ndim not in [3, 4]:
            raise ValueError("The `jacobians` argument provided to the `_ExponentialConeJacobianOperator` "
                             f"is {ndim}D, but it must be 3D or 4D.")


    def mv(self, dx: Float[Array, "*num_batches num_cones*3"]) -> Float[Array, "*num_batches num_cones*3"]:

        ndim = jnp.ndim(dx)

        if ndim == 1:
            if jnp.ndim(self.jacobians) == 4:
                raise ValueError("Batched Exponential cone Jacobians cannot be applied to a 1D input.")

            return _exp_cone_jacobian_mv(dx, self.jacobians)
        elif ndim == 2:
            return eqx.filter_vmap(_exp_cone_jacobian_mv,
                                   in_axes=(0, 0), out_axes=(0))(dx, self.jacobians)
        else:
            raise ValueError("The `_ExponentialConeJacobianOperator` can only be applied to 1D or 2D inputs "
                             f"but the provided vector is {ndim}D.")

    def as_matrix(self) -> Float[Array, "num_cones*3 num_cones*3"]:
        return _dense_from_mv(self)

    def transpose(self) -> "_ExponentialConeJacobianOperator":
        return self

    # The implicit `None` path is unreachable: `__check_init__` ensures `jacobians` is 3D or 4D.
    def in_structure(self) -> jax.ShapeDtypeStruct:  # pyright: ignore[reportReturnType]

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

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()

@lx.is_symmetric.register(_ExponentialConeJacobianOperator)
def _(op: _ExponentialConeJacobianOperator) -> bool:
    return True


class ExponentialConeProjector(AbstractConeProjector):
    """Projector onto a product of exponential cones, or of their duals.

    Each cone is

        K_exp = closure{(r, s, t) : s > 0, s exp(r / s) <= t},

    and its dual is

        K_exp^* = closure{(u, v, w) : u < 0, -u exp(v / u) <= e w}.

    The input is `num_cones` consecutive `(r, s, t)` triples (the `ep` or `ed`
    block of the cone vector). Dual projections use Moreau's decomposition,
    Pi_{K^*}(v) = v + Pi_K(-v), with Jacobian I - DPi_K(-v).

    **Arguments:**
    - `num_cones`: the number of exponential cones.
    - `onto_dual`: if `True`, project onto K_exp^* instead of K_exp.
    """

    num_cones: int = eqx.field(static=True)
    # NOTE(quill): `onto_dual` being static is what allows us to
    #   use regular Python control flow with this flag throughout
    #   this file.
    onto_dual: bool = eqx.field(static=True)

    def __init__(self, num_cones: int, onto_dual: bool):
        self.num_cones = num_cones
        self.onto_dual = onto_dual

    def proj_dproj(self, x: Float[Array, " num_cones*3"]) -> tuple[Float[Array, " num_cones*3"], _ExponentialConeJacobianOperator]:
        """Project `x` onto the product of exponential (or dual exponential) cones.

        **Arguments:**
        - `x`: a 1D array of length `3 * num_cones`. Higher-dimensional inputs
            raise an error; to project a batch, `vmap` this method (the returned
            operator then accepts the matching 2D input).

        **Returns:**

        A 2-tuple of the projection (same shape as `x`) and its Jacobian, an
        `_ExponentialConeJacobianOperator`.
        """

        ndimx = jnp.ndim(x)
        if ndimx > 1:
            raise ValueError("Only 1D arrays can be passed to `proj_dproj` "
                             f"but a {ndimx}D array was provided. "
                             "To operate on higher-dimensional arrays, wrap "
                             "`proj_dproj` in `vmap`.")

        xs = jnp.reshape(x, (self.num_cones, 3))

        if self.onto_dual:
            xs = -xs

        primal_projs, polar_projs = eqx.filter_vmap(_proj_exp_and_polar, in_axes=0, out_axes=(0, 0))(xs)
        jacs = eqx.filter_vmap(_dproj_exp, in_axes=(0, 0))(xs, primal_projs)

        if self.onto_dual:
            projs = -polar_projs
            # So this leaves us with Dproj_k_star(v)[dv] = dv - Dproj_k(-v)[dv]
            jacs = eqx.filter_vmap(lambda jac: jnp.eye(3, dtype=xs.dtype) - jac, in_axes=0, out_axes=0)(jacs)

        else:
            projs = primal_projs

        return jnp.ravel(projs), _ExponentialConeJacobianOperator(jacs)
