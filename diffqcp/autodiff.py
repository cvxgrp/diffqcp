"""`jax.grad`-compatible entry point.

`HostQCP.vjp` / `DeviceQCP.vjp` apply the adjoint derivative by hand. To make
JAX's autodiff flow through a conic solve, wrap the (externally computed)
solution with `differentiable_solution`:

```python
data = (P_values, A_values, q, b)  # values in `structure`'s sparsity pattern
x, y, s = my_solver(jax.lax.stop_gradient(data))  # solved at this data
x, y, s = diffqcp.differentiable_solution(data, (x, y, s), structure)
loss = f(x, y, s)
jax.grad(...)                      # gradients reach P_values, A_values, q, b
```

The forward pass returns `(x, y, s)` unchanged; the solver is not called
here. The backward pass applies `vjp` at that solution. It is the caller's
job to make sure `(x, y, s)` actually solves the problem with this data, e.g.
by calling the solver through `jax.pure_callback` in the same function.
`(x, y, s)` are treated as constants: no gradient flows into them, so the
solver's inputs should be wrapped in `jax.lax.stop_gradient` (JAX would
otherwise try to differentiate the solver itself, which fails for
`jax.pure_callback`). `make_differentiable` does both steps for you.
"""
from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR
from jaxtyping import Array, Float

from diffqcp.problem_data import QCPStructureCPU, QCPStructureGPU
from diffqcp.qcp import DeviceQCP, HostQCP
from diffqcp.solvers import AbstractDerivativeSolver, LSMRSolver

Data = tuple[Float[Array, " nnz_P"], Float[Array, " nnz_A"], Float[Array, " n"], Float[Array, " m"]]
Solution = tuple[Float[Array, " n"], Float[Array, " m"], Float[Array, " m"]]


def _build_qcp(data: Data, solution: Solution, structure: QCPStructureCPU | QCPStructureGPU) -> HostQCP | DeviceQCP:
    P_values, A_values, q, b = data
    x, y, s = solution
    n, m = structure.n, structure.m
    if isinstance(structure, QCPStructureCPU):
        P = BCOO((P_values, jnp.stack([structure.P_nonzero_rows, structure.P_nonzero_cols], axis=1)), shape=(n, n))
        A = BCOO((A_values, jnp.stack([structure.A_nonzero_rows, structure.A_nonzero_cols], axis=1)), shape=(m, n))
        return HostQCP(P, A, q, b, x, y, s, structure)
    if isinstance(structure, QCPStructureGPU):
        P = BCSR((P_values, structure.P_csr_indices, structure.P_csr_indptr), shape=(n, n))
        A = BCSR((A_values, structure.A_csr_indices, structure.A_csr_indptr), shape=(m, n))
        return DeviceQCP(P, A, q, b, x, y, s, structure)
    raise TypeError(f"Unsupported problem structure {type(structure).__name__}.")


@eqx.filter_custom_vjp
def differentiable_solution(
    data: Data,
    solution: Solution,
    structure: QCPStructureCPU | QCPStructureGPU,
    solver: AbstractDerivativeSolver | None = None,
) -> Solution:
    """Return `solution` with the QCP solution map's derivative attached.

    **Arguments:**
    - `data`: `(P_values, A_values, q, b)`. `P_values` / `A_values` are the
      nonzero values of P and A in the order of `structure`'s sparsity pattern
      (for `QCPStructureCPU`, P is the upper triangle; for `QCPStructureGPU`,
      the full symmetric matrix in CSR order).
    - `solution`: `(x, y, s)`, a primal-dual solution at `data`.
    - `structure`: the problem structure the sparsity pattern came from.
    - `solver`: derivative solver; defaults to `LSMRSolver()`.

    **Returns:** `(x, y, s)`, unchanged, differentiable with respect to `data`.
    """
    del data, structure, solver
    return solution


@differentiable_solution.def_fwd
def _fwd(perturbed, data, solution, structure, solver=None):
    del perturbed, data, structure, solver
    return solution, None


@differentiable_solution.def_bwd
def _bwd(residuals, grad_obj, perturbed, data, solution, structure, solver=None):
    del residuals, perturbed
    x, y, s = solution
    dx, dy, ds = (jnp.zeros_like(v) if g is None else g for g, v in zip(grad_obj, (x, y, s), strict=True))
    qcp = _build_qcp(data, solution, structure)
    dP, dA, dq, db = qcp.vjp(dx, dy, ds, solver=LSMRSolver() if solver is None else solver)
    return (dP.data, dA.data, dq, db)


def make_differentiable(
    solve: Callable[[Data], Solution],
    structure: QCPStructureCPU | QCPStructureGPU,
    solver: AbstractDerivativeSolver | None = None,
) -> Callable[[Data], Solution]:
    """Turn a (non-differentiable) conic solver into a differentiable function.

    `solve(data) -> (x, y, s)` may be anything JAX can call, e.g. a host solver
    wrapped in `jax.pure_callback`. The returned function solves at `data` and
    attaches the solution map's derivative, so `jax.grad` / `jax.vjp` of any
    function of its output reach `data = (P_values, A_values, q, b)`.
    """

    def differentiable_solve(data: Data) -> Solution:
        solution = solve(jax.lax.stop_gradient(data))
        return differentiable_solution(data, solution, structure, solver)

    return differentiable_solve
