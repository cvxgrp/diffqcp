"""
Projection onto the power cone and (TODO next) the derivative of the projection.

Basically just a port from https://github.com/cvxgrp/scs/blob/master/src/cones.c
"""
import torch

POW_CONE_TOL = 1e-9
POW_CONE_MAX_ITERS = 20

def pow_calc_x_i(r: torch.Tensor,
                 x_i: torch.Tensor,
                 z0: torch.Tensor,
                 alpha_i: torch.Tensor
) -> torch.Tensor:
    """x_i from eq 4. from Hien paper.
    """
    x = 0.5 * (x_i + torch.sqrt(x_i*x_i + 4 * alpha_i * (z0 - r)*r))
    return torch.maximum(x, torch.tensor(1e-12, dtype=x_i.dtype, device=x_i.device))


def g_i(r: torch.Tensor,
        x_i: torch.Tensor,
        z0: torch.Tensor,
        alpha_i: torch.Tensor
) -> torch.Tensor:
    """g_i from diffqcp paper.
    """
    return 2 * pow_calc_x_i(r, x_i, z0, alpha_i) - x_i


def d():
    pass

def pow_calc_f(x: torch.Tensor,
               y: torch.Tensor,
               r: torch.Tensor,
               alpha: torch.Tensor,
               alphac: torch.Tensor
) -> torch.Tensor:
    """Phi from Hien paper.
    """
    return torch.pow(x, alpha) * torch.pow(y, alphac) - r


def pow_calc_dx_i_dr(x: torch.Tensor,
                     x0: torch.Tensor,
                     z0: torch.Tensor,
                     r: torch.Tensor,
                     alpha: torch.Tensor
) -> torch.Tensor:
    two = torch.tensor(2, dtype=x.dtype, device=x.device)
    return alpha * (z0 - two * r) / (two * x - x0)


def pow_calc_fp(x: torch.Tensor,
                y: torch.Tensor,
                dxdr: torch.Tensor,
                dydr: torch.Tensor,
                alpha: torch.Tensor,
                alphac: torch.Tensor
) -> torch.Tensor:
    return torch.pow(x, alpha) * torch.pow(y, alphac) * (alpha * dxdr / x + alphac * dydr / y) - 1


def in_K_pow(alpha: torch.Tensor,
             alphac: torch.Tensor,
             x0: torch.Tensor,
             y0: torch.Tensor,
             z0: torch.Tensor,
             tol: torch.Tensor
) -> bool:
    return (x0 >= 0 and y0 >= 0 and
        tol + torch.pow(x0, alpha) * torch.pow(y0, alphac) >= z0)


def in_K_pow_polar(alpha: torch.Tensor,
                   alphac: torch.Tensor,
                   x0: torch.Tensor,
                   y0: torch.Tensor,
                   z0: torch.Tensor,
                   tol: torch.Tensor
) -> bool:
    return (x0 <= 0 and y0 <= 0 and
        tol + torch.pow(-x0, alpha) * torch.pow(-y0, alphac) >=
            z0 * torch.pow(alpha, alpha) * torch.pow(alphac, alphac))


def construct_z_not_zero_Jacobian():
    pass


def construct_z_zero_Jacobian():
    pass


def proj_dproj_power_cone(v: torch.Tensor,
                          alpha: float | torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Projection onto 3D power cone.

    The 3D power cone is defined as
    
        K_pow(alpha) = {(x, y, z) | x^alpha * y^(1-alpha) >= |z|, x, y >= 0}.

    This function computes the projection of v in R^3 onto K_pow(alpha). 
    More specifically, it is the function P_power: R^3 to R^3 given by

        P_power(v) = argmin      ||w - v||_2
                     subject to  w in K_pow(alpha).

    Parameters
    ----------
    v : torch.Tensor
        The point in R^3 to project onto the power cone.
    alpha : torch.Tensor
        Parameter defining a specific power cone.

    Returns
    -------
    torch.Tensor
        The output of P_power(v) <=> the projection of v onto the 3D power cone.
    torch.Tensor
        The Jacobian of P_power at v <=> DP_power(v) in R^{3 x 3}
    """
    POW_CONE_TOL_DEV = torch.tensor(POW_CONE_TOL, dtype=v.dtype, device=v.device)

    x0, y0, z0 = v
    z0 = torch.abs(z0)
    a_device = torch.as_tensor(alpha, dtype=v.dtype, device=v.device)
    ac_device = 1 - a_device
    zero = torch.tensor(0, dtype=v.dtype, device=v.device)

    if in_K_pow(a_device, ac_device, x0, y0, z0, POW_CONE_TOL_DEV):
        return (v, torch.eye(3, dtype=v.dtype, device=v.device))
    
    if in_K_pow_polar(a_device, ac_device, x0, y0, z0, POW_CONE_TOL_DEV):
        return (torch.zeros(3, dtype=v.dtype, device=v.device), torch.zeros((3, 3), dtype=v.dtype, device=v.device))

    x = torch.tensor(0, dtype=v.dtype, device=v.device)
    y = torch.tensor(0, dtype=v.dtype, device=v.device)
    r = z0 / 2

    for _ in range(POW_CONE_MAX_ITERS):
        x = pow_calc_x_i(r, x0, z0, a_device)
        y = pow_calc_x_i(r, y0, z0, ac_device)

        f = pow_calc_f(x, y, r, a_device, ac_device)

        if torch.abs(f) < POW_CONE_TOL_DEV:
            break

        dxdr = pow_calc_dx_i_dr(x, x0, z0, r, a_device)
        dydr = pow_calc_dx_i_dr(y, y0, z0, r, ac_device)
        fp = pow_calc_fp(x, y, dxdr, dydr, a_device, ac_device)

        r = torch.maximum(r - f / fp, zero)
        r = torch.minimum(r, z0)

    out = torch.empty_like(v)
    out[0] = x
    out[1] = y
    out[2] = -r if v[2] < 0 else r
    
    abs_z = torch.abs(z0)
    if abs_z > POW_CONE_TOL_DEV:
        two_r = 2 * r
        sign_z = torch.sign(z0)
        gx = g_i(r, x0, z0, a_device)
        gy = g_i(r, y0, z0, ac_device)
        frac_x = (a_device * x0) / gx
        frac_y = (ac_device * y0) / gy
        T = - ( frac_x + frac_y )
        L = 2 * abs_z - two_r
        L /= abs_z + (abs_z - two_r) * (frac_x + frac_y)
        J = torch.empty((3, 3), dtype=v.dtype, device=v.device)
        a_device_squared = a_device*a_device
        alpha_alphac = a_device - a_device_squared
        gxgy = gx*gy
        rL = r * L
        J[0, 0] = 0.5 + x0/(2*gx) + ((a_device_squared) * (abs_z - two_r) * rL ) / (gx*gx)
        J[1, 1] = 0.5 + y0/(2*gy) + ((ac_device*ac_device) * (abs_z - two_r) * rL ) / (gy*gy)
        J[2, 2] = r/abs_z + (r/abs_z) * T * L
        J[0, 1] = rL*alpha_alphac*(abs_z - two_r) / gxgy
        J[1, 0] = J[0, 1]
        J[0, 2] = sign_z * a_device * rL / gx
        J[2, 0] = J[0, 2]
        J[1, 2] = sign_z * ac_device * rL / gy
        J[2, 1] = J[1, 2]
    # TODO: add other case 
    # if z0 
    return (out, J)


