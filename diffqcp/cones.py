from typing import Tuple, List, Union

import numpy as np
import scipy.sparse as sparse
import scipy.linalg as la
import scipy.sparse.linalg as sla
import pylops as lo

from diffqcp.proj_exp_cone import project_exp_cone
from diffqcp.utils import Scalar

# need to check for alternative to distutils, which was deprecated starting in Python 3.12 

ZERO = "z"
POS = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"

# The ordering of CONES matches SCS.
CONES = [ZERO, POS, SOC, PSD, EXP, EXP_DUAL]

# ------ The following taken from diffcp (but add type hints, etc.) ------

def parse_cone_dict(cone_dict):
    """Parses SCS-style cone dictionary."""
    return [(cone, cone_dict[cone]) for cone in CONES if cone in cone_dict]


def vec_psd_dim(dim):
    return int(dim * (dim + 1) / 2)


def psd_dim(size):
    return int(np.sqrt(2 * size))


def unvec_symm(x, dim):
    """Returns a dim-by-dim symmetric matrix corresponding to `x`.

    `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
          X21 X22 ... X2k
          ...
          Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """
    X = np.zeros((dim, dim))
    # triu_indices gets indices of upper triangular matrix in row-major order
    col_idx, row_idx = np.triu_indices(dim)
    X[(row_idx, col_idx)] = x
    X = X + X.T
    X /= np.sqrt(2)
    X[np.diag_indices(dim)] = np.diagonal(X) * np.sqrt(2) / 2
    return X


def vec_symm(X):
    """Returns a vectorized representation of a symmetric matrix `X`.

    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """
    X = X.copy()
    X *= np.sqrt(2)
    X[np.diag_indices(X.shape[0])] = np.diagonal(X) / np.sqrt(2)
    col_idx, row_idx = np.triu_indices(X.shape[0])
    return X[(row_idx, col_idx)]


def _proj(x: np.ndarray,
          cone,
          dual=False
) -> np.ndarray:
    if cone == EXP_DUAL:
        cone = EXP
        dual = not dual
    
    if cone == ZERO:
        return x if dual else np.zeros(x.shape)
    elif cone == POS:
        return np.maximum(x, 0)
    elif cone == SOC:
        t = x[0]
        z = x[1:]
        norm_z = np.linalg.norm(z, 2)
        if norm_z <= t or np.isclose(norm_z, t, atol=1e-8):
            return x
        elif norm_z <= -t:
            return np.zeros(x.shape)
        else:
            return 0.5 * (1 + t / norm_z) * np.append(norm_z, z)
    elif cone == PSD:
        dim = psd_dim(x.size)
        X = unvec_symm(x, dim)
        lambd, Q = np.linalg.eigh(X)
        return vec_symm(Q @ sparse.diags(np.maximum(lambd, 0)) @ Q.T)
    elif cone == EXP:
        num_cones = int(x.size / 3)
        out = np.zeros(x.size)
        offset = 0
        for _ in range(num_cones):
            x_i = x[offset:offset + 3]
            if dual:
                x_i = x_i * -1
            out[offset:offset + 3] = project_exp_cone(x_i);
            offset += 3
        # via Moreau: Pi_K*(x) = x + Pi_K(-x)
        return x + out if dual else out
    else:
        raise NotImplementedError("%s not implemented" % cone)


def proj(x,
         cones,
         dual=False
) -> np.ndarray:
    projection = np.zeros(x.shape)
    offset = 0
    for cone, sz in cones:
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for dim in sz:
            if cone == PSD:
                dim = vec_psd_dim(dim)
            elif cone == EXP or cone == EXP_DUAL:
                dim *= 3
            
            projection[offset:offset+dim] = _proj(x[offset:offset+dim],
                                                  cone,
                                                  dual=dual)
            offset += dim
    return projection

def pi(z: Tuple[np.ndarray,
                np.ndarray,
                np.ndarray],
       cones: List[Tuple[str,
                         Union[int, List[int]]
                         ]
                    ]
) -> np.ndarray:
    """Projection onto R^n x K^* x R_+
    """
    u, v, w = z
    return np.concatenate(
        [u, proj(v, cones, dual=True), np.maximum(w, 0)]
    )


# ======= DERIVATIVES =======

def _dprojection_soc(x: np.ndarray) -> lo.LinearOperator:
    n = x.size
    t, z = x[0], x[1:]
    norm_z = la.norm(z)
    if (norm_z <= t):
        return lo.Identity(n)
    elif (norm_z <= -t):
        return lo.Zero(n)
    else:
        unit_z = z / norm_z
        
        def mv(dx: np.ndarray) -> np.ndarray:
            dt, dz = dx[0], dx[1:dx.size]
            first_entry = dt*norm_z + z @ dz
            second_chunk = dt*z + (t + norm_z)*dz \
                            - t * unit_z * (unit_z @ dz)
            output = np.concatenate(([first_entry], second_chunk))
            return (1.0 / (2 * norm_z)) * output
        
        return lo.aslinearoperator(sla.LinearOperator((n, n), matvec=mv, rmatvec=mv))

def _dprojection_pos(x: np.ndarray) -> lo.LinearOperator:
    return lo.Diagonal(0.5 * (np.sign(x) + 1))

def _dprojection_zero(x: np.ndarray, dual: bool) -> lo.LinearOperator:
    """dual cone is free cone"""
    n = x.size
    return lo.Identity(n) if dual else lo.Zero(n)

def _dprojection(x: np.ndarray,
                 cone,
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
        return _dprojection(x)
    elif cone == PSD:
        raise NotImplementedError("%s not implemented" % cone)
    elif cone == EXP:
        raise NotImplementedError("%s not implemented" % cone)

def dprojection(x: np.ndarray,
                cones: List[Tuple[str,
                                  Union[int, List[int]]
                                 ]
                           ],
                dual=False
) -> lo.LinearOperator:
    ops = []
    offset = 0
    # TODO: create a cone iterator or something to consolidate?
    for cone, sz in cones:
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for dim in sz:
            if cone == PSD:
                dim = vec_psd_dim(dim)
            elif cone == EXP or cone == EXP_DUAL:
                dim *= 3
            
            ops.append(_dprojection(x[offset:offset + dim], cone, dual=dual))
            offset += dim

    return lo.BlockDiag(ops)

def dpi(u: np.ndarray,
        v: np.ndarray,
        w: float,
        cones: List[Tuple[str,
                         Union[int, List[int]]
                         ]
                    ]
) -> lo.LinearOperator:
    """Derivative of the projection of z onto R^n x K^* x R_+
    """

    def gt_0(t):
        return 1.0 if t >= 0.0 else 0.0

    ops = [lo.Identity(u.size),
           dprojection(v, cones, dual=True),
           Scalar(gt_0(w))
           ]
    return lo.BlockDiag(ops) # note this successfully ran in a jupyter notebook