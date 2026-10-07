"""Row-ordering conventions for the cone vector.

`diffqcp` uses the SCS convention for vectorizing a symmetric matrix in a PSD
cone block: the lower triangle stacked column-major (equivalently, the upper
triangle row-major), with off-diagonal entries scaled by sqrt(2). This is the
same convention `diffcp` uses.

Clarabel (and CVXPY's Clarabel interface) instead stacks the *upper* triangle
column-major. The cone blocks appear in the same order in both formats; only
the entries inside each PSD block are permuted. Problem data (rows of `A`, `b`)
and solutions (`y`, `s`) produced in Clarabel's format must be permuted into
SCS order before being handed to `diffqcp`, otherwise the PSD projection is
evaluated at the wrong point and the derivatives are silently wrong.
"""

import numpy as np


def _psd_block_clarabel_to_scs(size: int) -> np.ndarray:
    """Permutation `p` such that `v_scs = v_clarabel[p]` for one `size x size` block."""
    # Position of entry (row, col), row <= col, in Clarabel's upper-triangle,
    # column-major vector.
    upper_colmajor = {}
    idx = 0
    for col in range(size):
        for row in range(col + 1):
            upper_colmajor[(row, col)] = idx
            idx += 1
    # SCS walks the lower triangle column-major: (col, col), (col + 1, col), ...
    # Entry (i, j) with i >= j lives at (j, i) in the upper triangle.
    return np.array(
        [upper_colmajor[(col, row)] for col in range(size) for row in range(col, size)],
        dtype=np.int64,
    )


def clarabel_to_scs_permutation(cone_dims: dict) -> np.ndarray:
    """Row permutation taking a Clarabel-ordered cone vector to SCS order.

    **Arguments:**
    - `cone_dims`: SCS-style cone dictionary (as produced by
      `cvxpy.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict`),
      with keys among `"z"`, `"l"`, `"q"`, `"s"`, `"ep"`, `"ed"`, `"p"`.

    **Returns:**

    An integer array `p` of length `m` such that `v_scs = v_clarabel[p]`. Apply
    it to the rows of `A` and to `b`, `y` and `s`. It is the identity when there
    are no PSD cones.
    """
    offset = int(cone_dims.get("z", 0)) + int(cone_dims.get("l", 0)) + int(sum(cone_dims.get("q", [])))
    pieces = [np.arange(offset, dtype=np.int64)]
    for size in cone_dims.get("s", []):
        block = size * (size + 1) // 2
        pieces.append(offset + _psd_block_clarabel_to_scs(size))
        offset += block
    tail = 3 * int(cone_dims.get("ep", 0)) + 3 * int(cone_dims.get("ed", 0)) + 3 * len(cone_dims.get("p", []))
    pieces.append(np.arange(offset, offset + tail, dtype=np.int64))
    return np.concatenate(pieces)
