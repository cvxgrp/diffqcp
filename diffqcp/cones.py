"""
Cone utility functions and projections.
The derivative of these projections and the exponential cone projection are in separate files.
"""
import math

import torch

from diffqcp.pow_cone import proj_power_cone
from diffqcp.exp_cone import proj_exp_cone

# TODO: need to check for alternative to distutils, which was deprecated starting in Python 3.12

ZERO = "z"
POS = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"
POW = 'p'
# Note we don't define a POW_DUAL cone as we stick with SCS convention
# and use -alpha to create a dual power cone.

# The ordering of CONES matches SCS.
CONES = [ZERO, POS, SOC, PSD, EXP, EXP_DUAL, POW]

def parse_cone_dict(cone_dict: dict[str, int | list[int]]
) -> list[tuple[str, int | list[int]]]:
    """Parses SCS-style cone dictionary.

    Parameters
    ----------
    cone_dict : dict[str, int | list[int]]
        A dictionary with keys corresponding to cones and values
        corresponding to their dimension (either integers or lists of integers;
        see the docstring for `compute_derivative` in `qcp.py`).

    Returns
    -------
    list[tuple[str, int | list[int]]]
        A list of two-tuples where the first entry in a tuple is a
        key from the provided dictionary and the second entry in
        that tuple is the corresponding dictionary value.
    """
    return [(cone, cone_dict[cone]) for cone in CONES if cone in cone_dict]


def symm_size_to_dim(size: int) -> int:
    """Returns dimension of a size-by-size symmetric matrix.

    Equivalently
    - returns the number of elements in the vectorization of a matrix in S^size
    - returns the number of elements in the vectorization from a matrix in the
    size-dimensional PSD cone.

    Parameters
    ----------
    size : int
        The number of columns (equivalently, rows) of a symmetric matrix.

    Returns
    -------
    int
        The dimension of X in S^size.
    """
    return int(size * (size + 1) / 2)


def symm_dim_to_size(dim: int) -> int:
    """Returns the number of columns in a symmetric matrix from its dimension.

    Equivalently,
    - returns the dimension of the PSD cone from the number of elements in the vectorization
    of a matrix in that cone.

    Parameters
    ---------
    dim : int
        The dimension of a symmetric matrix. Or, equivalently,
        the number of entries in the vectorization of a symmetric matrix.

    Returns
    -------
    int
        The number of columns (equivalently, rows) in a symmetric matrix with dimension dim.
    """
    return int((math.sqrt(8 * dim + 1) - 1) / 2)


def vec_symm(X: torch.Tensor) -> torch.Tensor:
    """Returns a vectorized representation of a symmetric `X`.

    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)

    Parameters
    ----------
    X : torch.Tensor
        The k-by-k symmetric to compute vectorization of.

    Returns
    -------
    torch.Tensor
        The vectorized representation of the symmetric `X` per SCS.

    Notes
    -----
    According to the SCS documentation, https://www.cvxgrp.org/scs/api/cones.html#sdcone,
    the vectorized representation of `X` is constructed from the lower triangular elements
    placed in column-major order.
    `torch.triu_indices(size, size)` outputs the upper triangular indices of a size x size
    matrix in row-major order. This is equivalent to extracting the lower triangular
    indices of a dim x dim matrix in column-major order.
    """
    assert len(X.shape) == 2, "vec_symm requires that X is a 2-D tensor."

    sqrt2 = torch.sqrt(torch.tensor(2, dtype=X.dtype, device=X.device))
    size = X.shape[0]

    row_idx, col_idx = torch.triu_indices(size, size, device=X.device)
    vec = X[row_idx, col_idx]
    vec[row_idx != col_idx] *= sqrt2
    return vec


def unvec_symm(x: torch.Tensor,
               size: int = 0
) -> torch.Tensor:
    """Returns the size-by-size symmetric matrix from its vectorized form `x`.

    `x` is a vector of length size*(size + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
          X21 X22 ... X2k
          ...
          Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)

    Parameters
    ----------
    x : torch.Tensor
        A vectorized symmetric matrix of length size*(size + 1)/2.
    size : int, optional
        The dimension of the PSD-cone the symmetric matrix being reconstructed
        belongs to. Equivalently, the number of rows and columns the returned
        symmetric matrix will have.

    Returns
    -------
    torch.Tensor
        A size-by-size symmetric matrix.
    """
    assert len(x.shape) == 1
    size = size if size > 0 else symm_dim_to_size(x.shape[0])

    sqrt2 = torch.sqrt(torch.tensor(2, dtype=x.dtype, device=x.device))
    X = torch.zeros((size, size), dtype=x.dtype, device=x.device)
    row_idx, col_idx = torch.triu_indices(size, size, device=x.device)
    X[row_idx, col_idx] = x / sqrt2
    X = X + X.T
    diag_indices = torch.arange(size, device=x.device)
    X[diag_indices, diag_indices] /= sqrt2
    return X


def _proj(x: torch.Tensor,
          cone: str,
          dual=False
) -> torch.Tensor:
    """Project x onto an "atom" cone or its dual.

    Notes
    -----
    - TODO: cache the eigendecomposition computed when projecting onto PSD cones.
    """
    if cone == EXP_DUAL:
        cone = EXP
        dual = not dual

    if cone == ZERO:
        return x if dual else torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    elif cone == POS:
        return torch.maximum(x, torch.tensor(0, dtype=x.dtype, device=x.device))
    elif cone == SOC:
        t = x[0]
        z = x[1:]
        norm_z = torch.linalg.norm(z, 2)
        #TODO: lower tolerance to account for dtype being torch.float32 by default?
        if norm_z <= t or torch.isclose(norm_z, t, atol=1e-8):
            return x
        elif norm_z <= -t:
            return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        else:
            return 0.5 * (1 + t / norm_z) * torch.cat((norm_z.unsqueeze(0), z))
    elif cone == PSD:
        size = symm_dim_to_size(x.shape[0])
        X = unvec_symm(x, size)
        lambd, Q = torch.linalg.eigh(X)
        lambd.clamp_(min=0)
        PiX = Q @ (lambd.unsqueeze(-1) * Q.T)
        return vec_symm(PiX)
    elif cone == EXP:
        num_cones = int(x.shape[0] / 3)
        out = torch.empty_like(x)
        offset = 0
        for _ in range(num_cones):
            x_i = x[offset:offset+3]
            out[offset:offset+3] = proj_exp_cone(x_i, primal=not dual)
            offset += 3
        return out
    else:
        raise NotImplementedError("%s not implemented" % cone)


def proj(x,
         cones: list[tuple[str, int | list[int]]],
         dual=False
) -> torch.Tensor:
    """Projects x onto a (convex) cone, or its dual cone.

    Parameters
    ----------
    x : torch.Tensor
        The tensor to be projected.
    cones : list[tuple[str, int | list[int]]]
        The list of cones that x will be projected onto
        the cartesian product of.
        Specifically, cones should be a list of two-tuples, where
        the first element is the key for a known (convex) "atom" cone
        (see the docstring for `compute_derivative` in `qcp.py`)
        and the second element is the dimensionality of the cone.
        The second element in the tuple being a list corresponds to
        there being len(list) of those cones to project onto.
    dual : bool, optional
        Whether to project x onto the cone or dual cone.
        Default is True <=> project x onto the cone (not the dual).

    Returns
    -------
    torch.Tensor
        x projected onto cones.

    Notes
    -----
    - TODO: This function can certainly be rewritten to utilize GPU parallelization.
    (see the mixed parallel computing strategy proposed in CuClarabel.)
    """
    projection = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    offset = torch.tensor(0, dtype=torch.int32, device=x.device)
    for cone, sz in cones:
        assert cone in CONES, f"{cone} is not a known cone."
        sz = sz if isinstance(sz, (tuple, list)) else (sz,)
        if sum(sz) == 0:
            continue
        for cone_dim in sz:

            if cone == POW:
                # cone_dim is now actually the alpha defining K_pow(alpha)
                if cone_dim < 0:
                    # dual case
                    # via Moreau: Pi_K^*(v) = v + Pi_K(-v)
                    projection[offset:offset+3] = x[offset:offset+3] + proj_power_cone(-x[offset:offset+3], -cone_dim)
                else:
                    # primal case
                    projection[offset:offset+3] = proj_power_cone(x[offset:offset+3], cone_dim)
                
                offset += 3
                continue
            
            if cone == EXP or cone == EXP_DUAL:
                cone_dim *= 3
            elif cone == PSD:
                cone_dim = symm_size_to_dim(cone_dim)

            projection[offset:offset+cone_dim] = _proj(x[offset:offset+cone_dim],
                                                       cone,
                                                       dual=dual)
            offset += cone_dim

    return projection


def pi(z: tuple[torch.Tensor,
                torch.Tensor,
                torch.Tensor],
       cones: list[tuple[str, int | list[int]]]
) -> torch.Tensor:
    """Projection onto R^n x K^* x R_+.
    
    Parameters
    ----------
    z : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        The point to project onto R^n x K^* x R_+.
        The second element in the tuple must have a length
        corresponding to the dimensionality of K^*.
    cones : list[tuple[str, int | list[int]]]
        A list of cones in the format specified in the docstrings of `proj`
        in `cones.py`. The tensor z[1] will be projected onto the dual of
        each cone (so the dual of the cartesian product of the cones). 
    
    Returns
    -------
    torch.Tensor
        The projection of z onto R^n x K^* x R_+.
    """
    u, v, w = z
    n = u.shape[0]
    out = torch.empty(n + v.shape[0] + 1, dtype=u.dtype, device=u.device)
    out[0:n] = u
    out[n:-1] = proj(v, cones, dual=True) # TODO: cache this!!!
    out[-1] = torch.maximum(w, torch.tensor(0.0, dtype=w.dtype, device=w.device))
    return out
