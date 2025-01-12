from typing import Tuple

import torch
import linops as lo

from diffqcp.utils import ScalarOperator, BlockDiag, SymmetricOperator
from diffqcp.cones import (ZERO, POS, SOC, PSD, EXP, EXP_DUAL, symm_size_to_dim,
    vec_symm, unvec_symm)

def _dprojection_psd(x: torch.Tensor) -> lo.LinearOperator:
    """Returns the derivative of the projection onto the PSD cone at x.

    Returns
    -------
    lo.LinearOperator
        The derivative of the projectoin onto the PSD cone at x as an
        abstract linear map.

    Notes
    -----
    - see BMB'18 for derivative and its derivation
    - TODO: cache Q and lambd from projection
    """
    dim = x.shape[0]
    X = unvec_symm(x)
    lambd, Q = torch.linalg.eigh(X)
    zero = torch.tensor(0, dtype=x.dtype, device=x.device)

    # eigenvalues assorted in ascending order
    if lambd[0] >= zero : return lo.IdentityOperator(dim)

    k = (lambd < zero).sum().item() - 1

    def mv(dx: torch.Tensor) -> torch.Tensor:
        Q_T_DX_Q = Q.T @ unvec_symm(dx) @ Q

        i, j = torch.meshgrid(torch.arange(Q_T_DX_Q.shape[0], device=dx.device),
                              torch.arange(Q_T_DX_Q.shape[1], device=dx.device),
                              indexing='ij')

        i_le_k_j_le_k = (i <= k) & (j <= k)
        i_gt_k_j_le_k = (i > k) & (j <= k)
        i_le_k_j_gt_k = (i <= k) & (j > k)

        lambda_i_pos = torch.clamp(lambd[i], min=zero)
        lambda_i_neg = -torch.clamp(lambd[i], max=zero)
        lambda_j_pos = torch.clamp(lambd[j], min=zero)
        lambda_j_neg = -torch.clamp(lambd[i], max=zero)

        lambda_i_pos = lambda_i_pos[i_gt_k_j_le_k]
        lambda_j_pos = lambda_j_pos[i_le_k_j_gt_k]

        Q_T_DX_Q[i_le_k_j_le_k] = 0
        Q_T_DX_Q[i_gt_k_j_le_k] *= lambda_i_pos / (lambda_j_neg[i_gt_k_j_le_k] + lambda_i_pos)
        Q_T_DX_Q[i_le_k_j_gt_k] *= lambda_j_pos / (lambda_i_neg[i_le_k_j_gt_k] + lambda_j_pos)

        # Hadamard product w/o forming B matrix
        # So Q_T_DX_Q becomes (B hadamard Q_T_DX_Q) after double for loop.
        # for i in range(Q_T_DX_Q.shape[0]):
        #     for j in range(Q_T_DX_Q.shape[1]):
        #         if i <= k and j <= k:
        #             Q_T_DX_Q[i, j] = 0
        #         elif k > k and j <= k:
        #             lambda_i_pos = torch.maximum(lambd[i], zero)
        #             lambda_j_neg = -torch.minimum(lambd[j], zero)
        #             Q_T_DX_Q[i, j] *= lambda_i_pos / (lambda_j_neg + lambda_i_pos)
        #         elif i <= k and j > k:
        #             lambda_i_neg = -torch.minimum(lambd[i], zero)
        #             lambd_j_pos = torch.maximum(lambd[j], zero)
        #             Q_T_DX_Q[i, j] *= lambd_j_pos / (lambda_i_neg + lambd_j_pos)

        DPiX_DX = Q @ Q_T_DX_Q @ Q.T
        return vec_symm(DPiX_DX)

    return SymmetricOperator(dim, mv, device=x.device)

def _dprojection_soc(x: torch.Tensor) -> lo.LinearOperator:
    """Returns the derivative of the projection onto the SOC at x.

    Parameters
    ----------
    x : torch.Tensor
        The point

    Returns
    -------
    lo.LinearOperator
    The derivative of the projectoin onto the SOC at x as an
    abstract linear map.

    Notes
    ------
    See BMB'18 for derivative
    """
    n = x.shape[0]
    t, z = x[0], x[1:]
    norm_z = torch.norm(z)
    if (norm_z <= t):
        return lo.IdentityOperator(n)
    elif (norm_z <= -t):
        return lo.ZeroOperator(n)
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
    return lo.DiagonalOperator(0.5 * (torch.sign(x) + 1.0))

def _dprojection_zero(x: torch.Tensor, dual: bool) -> lo.LinearOperator:
    """TODO: add docstring; dual cone is free cone"""
    n = x.shape[0]
    return lo.IdentityOperator(n) if dual else lo.ZeroOperator(n)

def _dprojection(x: torch.Tensor,
                 cone : str,
                 dual: bool=False
) -> lo.LinearOperator:
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
    elif cone == EXP:
        raise NotImplementedError("%s not implemented" % cone)
    else:
        raise NotImplementedError("%s not implemented" % cone)

def dprojection(x: torch.Tensor,
                cones: list[Tuple[str, int | list[int]]],
                dual=False
) -> lo.LinearOperator:
    """TODO: add docstring"""
    ops = []
    offset = 0
    for cone, sz in cones:
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for cone_dim in sz:
            if cone == PSD:
                cone_dim = symm_size_to_dim(cone_dim)
            elif cone == EXP or cone == EXP_DUAL:
                cone_dim *= 3

            ops.append(_dprojection(x[offset:offset + cone_dim], cone, dual=dual))
            offset += cone_dim

    return BlockDiag(ops)

def dpi(u: torch.Tensor,
        v : torch.Tensor,
        w: torch.Tensor,
        cones: list[Tuple[str, int | list[int]]]
) -> lo.LinearOperator:
    """Derivative of the projection of z onto R^n x K^* x R_+
    TODO: finish docstring
    Notes
    -----
    allow for batch dimension?
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
