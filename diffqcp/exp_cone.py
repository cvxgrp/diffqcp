"""
Projection onto exponential cone and the derivative of the projection.

The projection routines are from

    Projection onto the exponential cone: a univariate root-finding problem,
        by Henrik A. Fridberg, 2021.

And even more specifically, the routines are a port from SCS's implementation
of Fridberg's routines (see exp_cone.c).

The derivative of the projection is a port from https://github.com/cvxgrp/diffcp/blob/master/cpp/src/cones.cpp.
"""
import torch

EXP_CONE_INF_VALUE = 1e15

def _is_finite(x: float) -> bool:
    EXP_CONE_INF_VALUE_DEV = torch.tensor(EXP_CONE_INF_VALUE, dtype=x.dtype, device=x.device)
    return torch.abs(x) < EXP_CONE_INF_VALUE_DEV


def _clip(x: torch.Tensor,
          l: torch.Tensor,
          u: torch.Tensor
) -> torch.Tensor:
    return torch.maximum(l, torch.minimum(u, x))


def hfun_and_grad_hfun(v: torch.Tensor,
                       rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    v: torch.Tensor
        1D tensor of size 3. Point to evaluate the (Fridberg) function and gradient at.
    rho: torch.Tensor
        A scalar tensor.

    Returns
    -------
    torch.Tensor (scalar)
        f(v), the Fridberg function at v.
    torch.Tensor (scalar)
        Df(v)^T, the adjoint (the gradient) of the Fridberg function at v.
    """
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = torch.exp(rho)
    expnegrho = -torch.exp(rho)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    
    f = ((rho - one)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -\
        (rho * (rho - one) + one) * t0
    df = (rho * r0 + s0) * exprho + (r0 - (rho - one) * s0) * expnegrho - (2 * rho - one) * t0
    return (f, df)


def hfun(v: torch.Tensor,
         rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameters
    ----------
    v: torch.Tensor
        1D tensor of size 3. Point to evaluate the (Fridberg) function and gradient at.
    rho: torch.Tensor
        A scalar tensor.

    Returns
    -------
    torch.Tensor (scalar)
        f(v), the Fridberg function at v.
    """
    r0, s0, t0 = v[0], v[1], v[2]
    exprho = torch.exp(rho)
    expnegrho = -torch.exp(rho)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    
    f = ((rho - one)*r0 + s0) * exprho - (r0 - rho * s0) * expnegrho -\
        (rho * (rho - one) + one) * t0
    
    return f


def pomega(rho: torch.Tensor) -> torch.Tensor:
    one = torch.tensor(1, dtype=rho.dtype, device=rho.device)
    two = torch.tensor(2, dtype=rho.dtype, device=rho.device)
    val = torch.exp(rho) / (rho * (rho - one) + one)

    if rho < two:
        val = torch.minimum(val, torch.exp(two) / 3.0)
    
    return val


def domega(rho: torch.Tensor) -> torch.Tensor:
    one = torch.tensor(1, dtype=rho.dtype, device=rho.device)
    val = -torch.exp(-rho) / (rho * (rho - one) + one)

    if rho > -1:
        val = torch.maximum(val, -torch.exp(one) / 3.0)
    
    return val


def ppsi(v: torch.Tensor) -> torch.Tensor:
    r0, s0 = v[0], v[1]

    if r0 > s0:
        psi = (r0 - s0 + torch.sqrt(r0*r0 + s0*s0 - r0*s0)) / r0
    else:
        psi = -s0 / (r0 - s0 - torch.sqrt(r0*r0 + s0*s0 - r0*s0))
    
    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    return ((psi - one) * r0 + s0) / (psi * (psi - one) + one)


def dpsi(v: torch.Tensor) -> torch.Tensor:
    r0, s0 = v[0], v[1]

    if s0 > r0:
        psi = (r0 - torch.sqrt(r0*r0 + s0*s0 - r0*s0)) / s0
    else:
        psi = (r0 - s0) / (r0 + torch.sqrt(r0*r0 + s0*s0 - r0*s0))

    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    return (r0 - psi*s0) / (psi * (psi - one) + one)


def exp_search_bracket(v: torch.Tensor,
                       pdist: torch.Tensor,
                       ddist: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate upper and lower search bounds for root of hfun.

    Returns
    -------
    torch.Tensor (a scalar)
        The lower bound for the root of hfun.
    torch.Tensor (a scalar)
        The upper bound for the root of hfun.
    """
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    EXP_CONE_INF_VALUE_DEV = torch.tensor(EXP_CONE_INF_VALUE,
                                          dtype=v.dtype,
                                          device=v.device)
    EPS = torch.tensor(1e-12, dtype=torch.dtype, device=torch.device)

    r0, s0, t0 = v[0], v[1], v[2]
    baselow = -EXP_CONE_INF_VALUE_DEV
    baseupr = EXP_CONE_INF_VALUE_DEV
    low = -EXP_CONE_INF_VALUE_DEV
    upr = EXP_CONE_INF_VALUE_DEV

    s0m = torch.minimum(s0, zero)
    Dp = torch.sqrt(pdist*pdist - s0m*s0m)
    r0m = torch.minimum(r0, zero)
    Dd = torch.sqrt(ddist*ddist - r0m*r0m)

    if t0 > zero:
        curbnd = torch.log(t0 / ppsi(v))
        low = torch.maximum(low, curbnd)
    elif t0 < zero:
        curbnd = -torch.log(-t0, dpsi(v))
        upr = torch.minimum(upr, curbnd)

    if r0 > zero:
        baselow = 1 - s0/r0
        low = torch.maximum(low, baselow)
        tpu = torch.maximum(EPS, torch.minimum(Dd, Dp + t0))
        curbnd = torch.maximum(low, baselow + tpu / r0 / pomega(low))
        upr = torch.minimum(upr, curbnd)
    
    if s0 > zero:
        baseupr = r0 / s0
        upr = torch.minimum(upr, baseupr)
        tdl = -torch.maximum(EPS, torch.minimum(Dp, Dd - t0))
        curbnd = torch.minimum(upr, baseupr - tdl / s0 / domega(upr))
        low = torch.maximum(low, curbnd)

    # guarantee valid bracket
    # TODO: these two lines might not be necessary (follow what SCS does)
    low = _clip(torch.minimum(low, upr), baselow, baseupr)
    upr = _clip(torch.maximum(low, upr), baselow, baseupr)

    if (low != upr):
        fl = hfun(v, low)
        fu = hfun(v, upr)

        if fl * fu > zero:
            if torch.abs(fl) < torch.abs(fu):
                upr = low
            else:
                low = upr
    
    return (low, upr)


def proj_primal_exp_cone_heuristic(v: torch.Tensor,
                                   vp: torch.Tensor
) -> torch.Tensor:
    """Computes heuristic (cheap) projection onto EXP cone.

    **Modifies vp in place** as the heuristic projection.

    Returns
    -------
    float (as a torch.Tensor)
        ||v - vp||_2
    """
    r0, s0, t0 = v[0], v[1], v[2]
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    # perspective boundary
    vp[0] = torch.minimum(r0, zero)
    vp[1] = torch.tensor(0, dtype=v.dtype, device=v.device)
    vp[2] = torch.maximum(t0, zero)
    dist = torch.linalg.norm(v - vp)

    # perspective interior
    if s0 > zero:
        tp = torch.maximum(t0, s0 * torch.exp(r0 / s0))
        newdist = tp - t0
        if newdist < dist:
            vp[0] = r0
            vp[1] = s0
            vp[2] = tp
            dist = newdist
    
    return dist


def proj_polar_exp_cone_heuristic(v: torch.Tensor,
                                  vd: torch.Tensor
) -> torch.Tensor:
    """Computes heuristic (cheap) projection onto polar EXP cone.

    **Modifies vd in place** as the heuristic projection.

    Returns
    -------
    float (as a torch.Tensor)
        ||v - vd||_2
    """
    r0, s0, t0 = v[0], v[1], v[2]
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    
    # perspective boundary
    vd[0] = torch.tensor(0, dtype=v.dtype, device=v.device)
    vd[1] = torch.minimum(s0, zero)
    vd[2] = torch.minimum(t0, zero)
    dist = torch.linalg.norm(v - vd)

    # perspective interior
    if r0 > zero:
        one = torch.tensor(1, dtype=v.dtype, device=v.device)
        td = torch.minimum(t0, -r0 * torch.exp(s0/r0 - one))
        newdist = t0 - td
        if newdist < dist:
            vd[0] = r0
            vd[1] = s0
            vd[2] = td
            dist = newdist

    return dist


def root_search_binary(v: torch.Tensor,
                       xl: torch.Tensor,
                       xh: torch.Tensor,
                       x: torch.Tensor
) -> torch.Tensor:
    """Binary search method for finding root of hfun.

    Parameters
    ----------
    v: torch.Tensor
        The point (in R^3) being projected onto the exponential cone.
    xl: torch.Tensor (scalar)
        Lower search bound for the root.
    xh: torch.Tensor (scalar)
        Upper search bound for the root.
    x: torch.Tensor
        The initial guess for the root.
    
    Returns
    -------
    torch.Tensor
        The root of hfun.
    """
    EPS = torch.tensor(1e-12, dtype=v.dtype, device=v.device) # loosened from newton, since expensive
    MAXITER = 40
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    point5 = torch.tensor(0.5, dtype=v.dtype, device=v.device)

    i = 0
    while i < MAXITER:
        f = hfun(v, x)
        
        if f < zero:
            xl = x
        else:
            xu = x

        # binary search step
        x_plus = point5 * (xl + xu)
        if (torch.abs(x_plus - x) <= EPS * torch.maximum(one, torch.abs(x_plus))
            or x_plus == x or x_plus == xu):
            break
        
        x = x_plus
        i += 1
    
    return x_plus


def root_search_newton(v: torch.Tensor,
                       xl: torch.Tensor,
                       xu: torch.Tensor,
                       x: torch.Tensor
) -> torch.Tensor:
    """Univariate Newton method for finding a root of hfun.

    Parameters
    ----------
    v: torch.Tensor
        The point (in R^3) being projected onto the exponential cone.
    xl: torch.Tensor (scalar)
        Lower search bound for the root.
    xh: torch.Tensor (scalar)
        Upper search bound for the root.
    x: torch.Tensor
        The initial guess for the root.
    
    Returns
    -------
    torch.Tensor
        The root of hfun.
    """
    EPS = torch.tensor(1e-15, dtype=v.dtype, device=v.device)
    DFTOL = torch.tensor(1e-13, dtype=v.dtype, device=v.device)
    MAXITER = 20
    LODAMP = torch.tensor(0.5, dtype=v.dtype, device=v.device)
    HIDAMP = torch.tensor(0.95, dtype=v.dtype, device=v.device)
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)

    i = 0
    while i < MAXITER:
        f, df = hfun_and_grad_hfun(v, x)

        if torch.abs(f) <= EPS:
            break

        if f < zero:
            xl = x
        else:
            xu = x
        
        if xu <= xl:
            xu = 0.5 * (xu + xl)
            xl = xu
            break

        if (not _is_finite(f) or df < DFTOL):
            break

        # Newton step
        x_plus = x - f / df

        if torch.abs(x_plus - x) <= EPS * torch.maximum(one, torch.abs(x_plus)):
            break

        if x_plus >= xu:
            x = torch.minimum(LODAMP * x + HIDAMP * xu, xu)
        elif x_plus <= xl:
            x = torch.maximum(LODAMP * x + HIDAMP * xl, xl)
        else:
            x = x_plus
        
        i += 1
    
    # Newton method converged
    if i < MAXITER:
        return _clip(x, xl, xu)
    else:
        return root_search_binary(v, xl, xu, x)
    

def proj_sol_primal_exp_cone(v: torch.Tensor,
                             rho: torch.Tensor,
                             vp: torch.Tensor
) -> torch.Tensor:
    """Convert from rho to primal projection.

    **vp is modified in place** to be the primal projection.

    Returns
    -------
    float (as a torch.Tensor)
        ||v - vp||_2
    """
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)
    
    linrho = (rho - one) * v[0] + v[1]
    exprho = torch.exp(rho)
    
    if linrho > zero and _is_finite(exprho):
        quadrho = rho * (rho - one) + one
        vp[0] = rho * linrho / quadrho
        vp[1] = linrho / quadrho
        vp[2] = exprho * linrho / quadrho
        dist = torch.linalg.norm(vp - v)
    else:
        vp[0] = torch.tensor(0, dtype=v.dtype, device=v.device)
        vp[1] = torch.tensor(0, dtype=v.dtype, device=v.device)
        vp[2] = torch.tensor(EXP_CONE_INF_VALUE, dtype=v.dtype, device=v.device)
        dist = torch.tensor(EXP_CONE_INF_VALUE, dtype=v.dtype, device=v.device)
    
    return dist


def proj_sol_polar_exp_cone(v: torch.Tensor,
                            rho: torch.Tensor,
                            vd: torch.Tensor
) -> torch.Tensor:
    """Convert from rho to polar projection.

    **vd is modified in place** to be the polar projection.

    Returns
    -------
    float (as a torch.Tensor)
        ||v - vd||_2
    """
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    one = torch.tensor(1, dtype=v.dtype, device=v.device)

    linrho = v[0] - rho * v[1]
    exprho = torch.exp(-rho)
    
    if linrho > zero and _is_finite(exprho):
        quadrho = rho * (rho - one) + one
        lrho_div_qrho = linrho / quadrho
        vd[0] = lrho_div_qrho
        vd[1] = (one - rho) * lrho_div_qrho
        vd[2] = -exprho * lrho_div_qrho
        dist = torch.linalg.norm(vd - v)
    else:
        vd[0] = torch.tensor(0, dtype=v.dtype, device=v.device)
        vd[1] = torch.tensor(0, dtype=v.dtype, device=v.device)
        vd[2] = torch.tensor(-EXP_CONE_INF_VALUE, dtype=v.dtype, device=v.device)
        dist = torch.tensor(EXP_CONE_INF_VALUE, dtype=v.dtype, device=v.device)
    
    return dist
    

def proj_exp(v: torch.Tensor,
             primal: bool
) -> torch.Tensor:
    """Project v onto the exponential cone (or its dual).

    Parameters
    ----------
    v: 1D torch.Tensor
        The point to project.
    primal: bool
        Whether to project v onto the exponential cone or the dual exponential cone.
    
    Returns
    -------
    1D torch.Tensor
        The projection.
    """
    TOL = torch.tensor(1e-8, dtype=v.dtype, device=v.device)
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)
    vp = torch.empty_like(v)
    vd = torch.empty_like(v)
    v_hat = torch.empty_like(v)

    if not primal:
        v *= -1
    
    pdist = proj_primal_exp_cone_heuristic(v, vp)
    ddist = proj_polar_exp_cone_heuristic(v, vd)

    err = torch.abs(vp[0] + vd[0] - v[0])
    err = torch.maximum(err, torch.abs(vp[1] + vd[1] - v[1]))
    err = torch.maximum(err, torch.abs(vp[2] + vd[2] - v[2]))

    # skip root search if presovle rules apply 
    # or optimality conditions are satisfied
    opt = (v[1] <= zero and v[0] <= zero)
    opt |= (torch.minimum(pdist, ddist) <= TOL)
    opt |= (err <= TOL and vp @ vd <= TOL)
    if opt:
        if primal:
            return vp
        # else negate projection onto polar cone for projection onto dual
        return -vd

    xl, xh = exp_search_bracket(v, pdist, ddist)
    rho = root_search_newton(v, xl, xh, 0.5 * (xl + xh))

    if primal:
        dist_hat = proj_sol_primal_exp_cone(v, rho, v_hat)
        if dist_hat <= pdist:
            vp[...] = v_hat
        return vp

    dist_hat = proj_sol_polar_exp_cone(v, rho, v_hat)
    if dist_hat <= ddist:
        vd[...] = v_hat
    # else negate projection onto polar cone for projection onto dual
    return -vd