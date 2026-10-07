"""Cone keys, the product-cone projector, and re-exports of the per-cone projectors.

Conventions (ordering, PSD vectorization, dual cones) are in `CONVENTIONS.md`.
Design notes:
- Projectors follow the abstract/final pattern
  (https://docs.kidger.site/equinox/pattern/).
- The Jacobian operators' `mv` methods handle 2D (and 3D) inputs because
  `vmap`ping `proj_dproj` gives operators whose leaves carry a batch axis.
  Replacing this with 1D operators + `vmap` at the boundary is deferred to the
  batching wave (it changes how batched operators are applied).
"""
from typing import cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from diffqcp._helpers import _to_int_list
from diffqcp.linops import _BlockLinearOperator

from ._grouping import _collect_cone_batch_info, _group_cones_in_order
from .abstract_projector import AbstractConeProjector
from .exp import ExponentialConeProjector
from .nonneg import NonnegativeConeProjector
from .pow import PowerConeProjector
from .psd import (
    PSDConeProjector,
    _PSDConeProjector,
    form_B_block,
    form_B_block_full,
    jax_symm_dim_to_size,
    symm_dim_to_size,
    symm_size_to_dim,
    unvec_symm,
    vec_symm,
)
from .soc import SecondOrderConeProjector, _SecondOrderConeProjector
from .zero import ZeroConeProjector

# The per-cone projectors live in `zero.py`, `nonneg.py`, `soc.py`, `psd.py`,
# `exp.py` and `pow.py`; they are re-exported here for backward compatibility.
__all__ = [
    "EXP",
    "EXP_DUAL",
    "NONNEGATIVE",
    "POW",
    "PSD",
    "SOC",
    "ZERO",
    "ExponentialConeProjector",
    "NonnegativeConeProjector",
    "PSDConeProjector",
    "PowerConeProjector",
    "ProductConeProjector",
    "SecondOrderConeProjector",
    "ZeroConeProjector",
    "_PSDConeProjector",
    "_SecondOrderConeProjector",
    "_collect_cone_batch_info",
    "_group_cones_in_order",
    "form_B_block",
    "form_B_block_full",
    "jax_symm_dim_to_size",
    "symm_dim_to_size",
    "symm_size_to_dim",
    "unvec_symm",
    "vec_symm",
]

ZERO = "z"
NONNEGATIVE = "l"
SOC = "q"
PSD = "s"
EXP = "ep"
EXP_DUAL = "ed"
POW = 'p'
# Note we don't define a POW_DUAL cone as we stick with SCS convention
# and use -alpha to create a dual power cone.

# The ordering of _CONES matches SCS.
_CONES = [ZERO, NONNEGATIVE, SOC, PSD, EXP, EXP_DUAL, POW]

EPS = 1e-12 if jax.config.read("jax_enable_x64") else 1e-06


class ProductConeProjector(AbstractConeProjector):
    projectors: list[AbstractConeProjector]
    dims: list[int] = eqx.field(static=True)
    split_indices: list[int] = eqx.field(static=True)

    def __init__(self, cones: dict[str, int | list[int] | list[float]], onto_dual: bool=False):
        projectors = []
        dims = []
        for cone_key in _CONES:
            if cone_key not in cones:
                continue
            val = cones[cone_key]
            if cone_key == ZERO:
                # Zero cone: val is an int (number of zeros)
                projectors.append(ZeroConeProjector(onto_dual=onto_dual))
                dims.append(int(cast(int, val)))
            elif cone_key == NONNEGATIVE:
                # Nonnegative cone: val is an int (number of nonnegatives)
                projectors.append(NonnegativeConeProjector())
                dims.append(int(cast(int, val)))
            elif cone_key == SOC:
                # SOC: val is a list of ints (dimensions of each SOC block)
                soc_dims = [int(d) for d in cast(list, val)]
                if len(soc_dims) > 0:
                    projectors.append(SecondOrderConeProjector(soc_dims))
                    dims.append(sum(soc_dims))
            elif cone_key == EXP:
                # EXP cone: `val` is the (integer) number of cones
                num_exp = int(cast(int, val))
                if num_exp > 0:
                    projectors.append(ExponentialConeProjector(num_exp, onto_dual=onto_dual))
                    dims.append(3 * num_exp)
            elif cone_key == EXP_DUAL:
                # dual EXP cone: `val` is the (integer) number of cones
                num_exp_dual = int(cast(int, val))
                if num_exp_dual > 0:
                    projectors.append(ExponentialConeProjector(num_exp_dual, onto_dual=not onto_dual))
                    dims.append(3 * num_exp_dual)
            elif cone_key == POW:
                # Power cone: val is a list of floats in (-1, 1), which are the defining alphas.
                #   val[i] < 0 corresponds to projecting onto the dual exponential cone with
                #   abs(val[i]) as the defining alpha.
                alphas = [float(a) for a in cast(list, val)]
                if len(alphas) > 0:
                    projectors.append(PowerConeProjector(alphas, onto_dual=onto_dual))
                    dims.append(3 * len(alphas))
            elif cone_key == PSD:
                # PSD cone: val is a list of matrix sizes
                sizes = [int(k) for k in cast(list, val)]
                if len(sizes) > 0:
                    projectors.append(PSDConeProjector(sizes))
                    dims.append(sum([symm_size_to_dim(k) for k in sizes]))
            else:
                raise ValueError(f"The cone corresponding to cone key: {cone_key}"
                                 + " is not known.")
        self.projectors = projectors
        self.dims = dims
        self.split_indices = _to_int_list(np.cumsum(dims[:-1]))

    def proj_dproj(self, x):
        chunks = jnp.split(x, self.split_indices, axis=-1)

        projs, dproj_ops = [], []
        for chunk, projector in zip(chunks, self.projectors, strict=False):
            proj_xi, dproj_xi = projector(chunk)
            projs.append(proj_xi)
            dproj_ops.append(dproj_xi)

        # NOTE(quill): when `vmap`ping, the concatenation of `projs` will work
        #   as desired, but the `mv` of `_BlockLinearOperator` now needs to know
        #   how to handle 2D input AND the attributes (leaves) of the operator
        #   now have a batch dimension.
        return jnp.concatenate(projs, axis=-1), _BlockLinearOperator(dproj_ops)
