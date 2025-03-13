import torch
import linops as lo

from diffqcp.linops import ScalarOperator, BlockDiag, SymmetricOperator
from diffqcp.cones import (ZERO, POS, SOC, PSD, EXP, EXP_DUAL, POW, CONES,
                           symm_size_to_dim, vec_symm, unvec_symm)

def _dprojection_psd(x: torch.Tensor) -> lo.LinearOperator:
    """Returns the derivative of the projection onto the PSD cone at x.

    Parameters
    ----------
    x : torch.Tensor
        A vectorized PSD matrix with dimension == len(x).

    Returns
    -------
    lo.LinearOperator
        The derivative of the projection onto the PSD cone at x as an
        abstract linear map.

    Notes
    -----
    - see BMB'18 for derivative and its derivation
    - TODO: cache Q and lambd from projection
    - TODO: optimize computation for GPU
    """
    assert len(x.shape) == 1, "PSD projection: x must be vectorized."

    dim = x.shape[0]
    X = unvec_symm(x)
    lambd, Q = torch.linalg.eigh(X)
    zero = torch.tensor(0, dtype=x.dtype, device=x.device)

    # eigenvalues assorted in ascending order
    if lambd[0] >= zero:
        return lo.IdentityOperator(dim)

    k = -1
    i = 0
    while i < lambd.shape[0]:
        if lambd[i] < 0:
            k += 1
        else:
            break
        i += 1

    def mv(dx: torch.Tensor) -> torch.Tensor:
        Q_T_DX_Q = Q.T @ unvec_symm(dx) @ Q

        # Hadamard product w/o forming B matrix
        # So Q_T_DX_Q becomes (B hadamard Q_T_DX_Q) after double for loop.
        for i in range(Q_T_DX_Q.shape[0]):
            for j in range(Q_T_DX_Q.shape[1]):
                if i <= k and j <= k:
                    Q_T_DX_Q[i, j] = 0
                elif i > k and j <= k:
                    lambda_i_pos = torch.maximum(lambd[i], zero)
                    lambda_j_neg = -torch.minimum(lambd[j], zero)
                    Q_T_DX_Q[i, j] *= lambda_i_pos / (lambda_j_neg + lambda_i_pos)
                elif i <= k and j > k:
                    lambda_i_neg = -torch.minimum(lambd[i], zero)
                    lambd_j_pos = torch.maximum(lambd[j], zero)
                    Q_T_DX_Q[i, j] *= lambd_j_pos / (lambda_i_neg + lambd_j_pos)

        DPiX_DX = Q @ Q_T_DX_Q @ Q.T
        return vec_symm(DPiX_DX)

    return SymmetricOperator(dim, mv, device=x.device)

def _dprojection_soc(x: torch.Tensor) -> lo.LinearOperator:
    """Returns the derivative of the projection onto the SOC at x.

    Parameters
    ----------
    x : torch.Tensor
        Where the derivative of the projection onto the SOC is evaluated.
        The length of x is the dimension of the SOC that the projection
        is onto.

    Returns
    -------
    lo.LinearOperator
        The derivative of the projection onto the SOC at x as an
        abstract linear map.

    Notes
    ------
    - See BMB'18 for derivative
    """
    n = x.shape[0]
    t, z = x[0], x[1:]
    norm_z = torch.norm(z)
    if (norm_z <= t):
        return lo.IdentityOperator(n)
    elif (norm_z <= -t):
        return lo.ZeroOperator((n, n))
    else:
        unit_z = z / norm_z

        def mv(dx: torch.Tensor) -> torch.Tensor:
            dt, dz = dx[0], dx[1:dx.shape[0]]
            first_entry = dt*norm_z + z @ dz
            second_chunk = dt*z + (t + norm_z)*dz \
                            - t * unit_z * (unit_z @ dz)
            output = torch.empty_like(dx)
            output[0] = first_entry
            output[1:] = second_chunk
            return (1.0 / (2.0 * norm_z)) * output

        return SymmetricOperator(x.shape[0], mv, device=x.device)


def _dprojection_pos(x: torch.Tensor) -> lo.LinearOperator:
    """TODO: add docstring"""
    return lo.DiagonalOperator(0.5 * (torch.sign(x).to(dtype=x.dtype, device=x.device) + 1.0))


def _dprojection_zero(x: torch.Tensor, dual: bool) -> lo.LinearOperator:
    """TODO: add docstring; dual cone is free cone"""
    n = x.shape[0]
    return lo.IdentityOperator(n) if dual else lo.ZeroOperator((n, n))


def _dprojection(x: torch.Tensor,
                 cone : str,
                 dual: bool=False
) -> lo.LinearOperator:
    """The derivative of the projection onto cone at x.
    """
    if cone == EXP_DUAL:
        cone = EXP
        dual = not dual

    if cone == ZERO:
        return _dprojection_zero(x, dual)
    elif cone == POS:
        return _dprojection_pos(x)
    elif cone == SOC:
        return _dprojection_soc(x)
    elif cone == PSD:
        return _dprojection_psd(x)
    else:
        raise NotImplementedError("%s not implemented" % cone)


def dprojection(x: torch.Tensor,
                cones: list[tuple[str, int | list[int]]],
                dual=False
) -> lo.LinearOperator:
    """Returns the derivative of the projection of x onto a convex cone (or its dual).

    Parameters
    ----------
    x : torch.Tensor
        The tensor to evaluate the derivative of the projection at.
    cones : list[tuple[str, int | list[int]]]
        A list of cones in the format specified in the docstrings of `proj`
        in `cones.py`, whose cartesian product is the cone this function
        returns the derivative of the projection onto at x.
    dual : bool, optional
        Whether the projection of x is onto the cone or dual cone.
        Default is True <=> project x onto the cone.

    Returns
    -------
    lo.LinearOperator
        The derivative of the projection onto a convex cone (or its dual) at x.

    Notes
    -----
    - TODO: This function can certainly be rewritten to utilize GPU parallelization
    (this will look similar to however `proj` is rewritten).
    - TODO: still need to support EXP, EXP_DUAL, and POW JVPs
    """
    ops = []
    offset = 0
    for cone, sz in cones:
        assert cone in CONES, f"{cone} is not a known cone."
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for cone_dim in sz:
            if cone == PSD:
                cone_dim = symm_size_to_dim(cone_dim)
            elif cone == EXP or cone == EXP_DUAL:
                cone_dim *= 3
            elif cone == POW:
                cone_dim *= 3

            ops.append(_dprojection(x[offset:offset + cone_dim], cone, dual=dual))
            offset += cone_dim

    return BlockDiag(ops)


def dpi(u: torch.Tensor,
        v : torch.Tensor,
        w: torch.Tensor,
        cones: list[tuple[str, int | list[int]]]
) -> lo.LinearOperator:
    """Derivative of the projection of z = (u, v , w) onto R^n x K^* x R_+.

    Parameters
    ----------
    u : torch.Tensor
        The derivative of the projection onto R^n is evaluated at u.
    v : torch.Tensor
        The derivative of the projection onto K^* is evaluated at v.
    w : torch.Tensor
        The derivative of the projection onto R_+ is evaluated at w.
    cones : list[tuple[str, int | list[int]]]
        A list of cones in the format specified in the docstrings of `proj`
        in `cones.py`. The derivative of the projection onto K^* is computed
        at v.

    Returns
    -------
    lo.LinearOperator
        The derivative of the projection onto R^n x K^* x R_+ at x.
    """
    assert len(u.shape) == 1
    assert len(v.shape) == 1
    assert len(w.shape) == 0

    scale_val = torch.where(w >= 0, torch.tensor(1, dtype=w.dtype, device=w.device),
                                    torch.tensor(0, dtype=w.dtype, device=w.device))

    ops = [lo.IdentityOperator(u.shape[0]),
           dprojection(v, cones, dual=True),
           ScalarOperator(scale_val)]

    return BlockDiag(ops)
