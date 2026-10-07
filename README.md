<h1 align='center'>diffqcp: Differentiating through conic quadratic programs</h1>

`diffqcp` is a [JAX](https://docs.jax.dev/en/latest/) library to form the derivative of the solution map to a conic quadratic program (CQP) with respect to the CQP problem data as an abstract linear operator and to compute Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) with this operator.
The implementation is based on the derivations in our paper (see below) and computes
these products implicitly via projections onto cones and sparse linear system solves.
Our approach therefore differs from libraries that compute JVPs and VJPs by unrolling algorithm iterates.
We directly exploit the underlying structure of CQPs.

**Features include**:
- Hardware acclerated: JVPs and VJPs can be computed on CPUs, GPUs, and (theoretically) TPUs.
- Support for many canonical classes of convex optimization problems including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - and semidefinite programs (SDPs).
- Support for convex optimization problems constrained to the product of exponential
and power cones (as well as their duals).

# Conic quadratic programs

A conic quadratic program is given by the primal and dual problems

```math
\begin{equation*}
    \begin{array}{lll}
        \text{(P)} \quad &\text{minimize} \; & (1/2)x^T P x + q^T x  \\
        &\text{subject to} & Ax + s = b  \\
        & & s \in \mathcal{K},
    \end{array}
    \qquad
    \begin{array}{lll}
         \text{(D)} \quad  &\text{maximize} \; & -(1/2)x^T P x -b^T y  \\
        &\text{subject to} & Px + A^T y = -q \\
        & & y \in \mathcal{K}^*,
    \end{array}
\end{equation*}
```
where $`x \in \mathbf{R}^n`$ is the *primal* variable, $`y \in \mathbf{R}^m`$ is the *dual* variable, and $`s \in \mathbf{R}^m`$ is the primal *slack* variable. The problem data are $`P\in \mathbf{S}_+^{n}`$, $`A \in \mathbf{R}^{m \times n}`$, $`q \in \mathbf{R}^n`$, and $`b \in \mathbf{R}^m`$. We assume that $`\mathcal K \subseteq \mathbf{R}^m`$ is a nonempty, closed, convex cone with dual cone $`\mathcal{K}^*`$.

`diffqcp` currently supports CQPs whose cone is the Cartesian product of the zero cone, the positive orthant, second-order cones, positive semidefinite cones,
exponential cones, dual exponential cones, power cones, and dual power cones.
For more information about these cones, see the appendix of our paper.

# Usage

`diffqcp` is meant to be used as a CVXPYlayers backend --- it is not designed to be a stand-alone
library.
Nonetheless, here is how it use it.
(Note that while we'll specify different CPU and a GPU configurations,
all modules are CPU and GPU compatible--we just recommend the following
as JAX's `BCSR` arrays do have CUDA backends for their `mv` operations while the `BCOO` arrays do not.)

For both of the following problems, we'll use the following objects:

```python
import cvxpy as cvx

problem = cvx.Problem(...)
prob_data, _, _ = problem.get_problem_data(cvx.CLARABEL, solver_opts={'use_quad_obj': True})
scs_cones = cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(prob_data["dims"])

x, y, s = ... # canonicalized solutions to `problem`
```

> **PSD cones:** `diffqcp` expects PSD blocks vectorized in SCS order (lower
> triangle, column-major). Data from `get_problem_data(cvx.CLARABEL)` (and
> solutions from Clarabel) use the upper triangle instead. If the problem has
> PSD constraints, permute the rows of `A` and the entries of `b`, `y`, `s`
> first:
>
> ```python
> from diffqcp import clarabel_to_scs_permutation
> perm = clarabel_to_scs_permutation(scs_cones)
> A, b, y, s = A[perm], b[perm], y[perm], s[perm]
> ```

## Optimal CPU approach

If computing JVPs and VJPs on a CPU, we recommend using the `equinox.Module`s `HostQCP` and `QCPStructureCPU` as demonstrated in the following pseudo-example.

```python
from diffqcp import HostQCP, QCPStructureCPU
from jax.experimental.sparse import BCOO
from jaxtyping import Array

P: BCOO = ... # Only the upper triangular part of the CQP matrix P
A: BCOO = ...
q: Array = ...
b: Array = ...

problem_structure = QCPStructureCPU(P, A, scs_cones)
qcp = HostQCP(P, A, q, b, x, y, s, problem_structure)

# Compute JVPs

dP: BCOO ... # Same sparsity pattern as `P`
dA: BCOO = ... # Same sparsity pattern as `A`
db: Array = ...
dq: Array = ...

dx, dy, ds = qcp.jvp(dP, dA, dq, db)

# Compute VJPs
# `dP`, `dA` will be BCOO arrays, `dq`, `db` just Arrays
dP, dA, dq, db = qcp.vjp(f1(x), f2(y), f3(s)) 
```

## Optimal GPU approach

If computing JVPs and VJPs on a GPU, we recommend using the `equinox.Module`s `QCPStructureGPU` and `DeviceQCP`.

```python
from diffqcp import DeviceQCP, QCPStructureGPU
from jax.experimental.sparse import BCSR
from jaxtyping import Array

P: BCSR = ... # The entirety of the CQP matrix P
A: BCSR = ...
q: Array = ...
b: Array = ...

problem_structure = QCPStructureGPU(P, A, scs_cones)
qcp = DeviceQCP(P, A, q, b, x, y, s, problem_structure)

# Compute JVPs

dP: BCSR ... # Same sparsity pattern as `P`
dA: BCSR = ... # Same sparsity pattern as `A`
db: Array = ...
dq: Array = ...

dx, dy, ds = qcp.jvp(dP, dA, dq, db)

# Compute VJPs
# `dP`, `dA` will be BCSR arrays, `dq`, `db` just Arrays
dP, dA, dq, db = qcp.vjp(f1(x), f2(y), f3(s)) 
```

## Using `jax.grad`

`jvp` / `vjp` apply the derivative by hand. To let JAX's autodiff flow through a
conic solve instead, wrap your (non-differentiable) solver with
`make_differentiable`. Problem data are passed as the *values* of P and A in the
sparsity pattern of the problem structure:

```python
import jax
from diffqcp import QCPStructureCPU, make_differentiable

structure = QCPStructureCPU(P, A, scs_cones)       # fixes the sparsity pattern

def solve(data):                                   # any JAX-callable solver, e.g.
    P_values, A_values, q, b = data                # Clarabel via jax.pure_callback
    ...
    return x, y, s

differentiable_solve = make_differentiable(solve, structure)

def loss(data):
    x, y, s = differentiable_solve(data)
    return jax.numpy.sum(x ** 2)

grads = jax.jit(jax.grad(loss))((P.data, A.data, q, b))
```

For a solution you already have, `differentiable_solution(data, (x, y, s), structure)`
attaches the derivative without calling a solver (see `diffqcp/autodiff.py`).

## Selecting solvers

As detailed in our paper, the JVPs and VJPs are computed via a linear system solve
with an N x N matrix F (N = n + m + 1). F is always singular: the homogeneous
embedding has a one-dimensional null space spanned by the solution embedding
`z = (x, y - s, 1)`, which the derivative never sees. `diffqcp` fixes this "gauge"
by setting the last component of the solution to zero, which leaves an
N x (N - 1) system of full column rank (away from degenerate points where the
solution map is not differentiable). See `diffqcp/solvers.py` for details.

Pass a solver object via the `solver` argument of `jvp` / `vjp`:

```python
from diffqcp.solvers import LSMRSolver, DenseDirectSolver

dx, dy, ds = qcp.jvp(dP, dA, dq, db, solver=LSMRSolver(rtol=1e-12, atol=1e-12))
```

- `LSMRSolver` (default): matrix-free LSMR via `lineax`. Tolerances default to
  `1e-12` in float64 and `1e-6` in float32.
- `DenseDirectSolver`: materializes the gauge-fixed matrix and solves a symmetric
  augmented system with a dense LU, falling back to an SVD least-squares solve at
  degenerate points. Intended for small problems and as an accuracy reference.

The older string options are still accepted through `solve_method`:
`"jax-lsmr"` (= `LSMRSolver()`), `"jax-lu"` (= `DenseDirectSolver()`), and, on
`DeviceQCP`, `"nvmath-direct"` (cuDSS via `nvmath-python`, on the dense augmented
system).

**Future direction:** keep the augmented system sparse (CSR) for the cuDSS path
instead of materializing it densely.

# Installation

| Platform        | Instructions                            |
|-----------------|-----------------------------------------|
| CPU             | `pip install diffqcp`                   |
| NVIDIA GPU      | `pip install "diffqcp[gpu]"`            |

Note that `diffqcp[gpu]` is currently packaged with version 12 of CUDA. Moreover,
if your system supports version 13 of CUDA, install the CPU version of `diffqcp`
and then `pip install -U jax[cuda13]`. Optionally, if you want access to the cuDSS
solvers, also `pip install "cupy-cuda13x` and `nvmath-python[cu12]`. (Although
note that we're unsure how `nvmath-python[cu12]` will interact with the version
13s of the other packages.)

# Citation


[arXiv:2508.17522 [math.OC]](https://arxiv.org/abs/2508.17522)
```
@misc{healey2025differentiatingquadraticconeprogram,
      title={Differentiating Through a Quadratic Cone Program}, 
      author={Quill Healey and Parth Nobel and Stephen Boyd},
      year={2025},
      eprint={2508.17522},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2508.17522}, 
}
```

# Next steps

`diffqcp` is still in development! WIP features and improvements include:
- Batched problem computations.
- Not forming dense $F$ when using direct solver methods.
- Consider JAX's [`spsolve`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html#jax.experimental.sparse.linalg.spsolve).
- Better performance benchmarking / regression testing.
- Migration of tests from our [torch branch](https://github.com/cvxgrp/diffqcp/tree/torch-implementation).

## See also

**Core dependencies** (`diffqcp` makes essential use of the following libraries)
- [Equinox](https://github.com/patrick-kidger/equinox): Neural networks and everything not already in core JAX (via callable `PyTree`s).
- [Lineax](https://github.com/patrick-kidger/lineax): Linear solvers.

**Related** 
- [CVXPYlayers](https://github.com/cvxpy/cvxpylayers): Construct differentiable convex optimization layers using [CVXPY](https://github.com/cvxpy/cvxpy/). (`diffqcp` is a backend for CVXPYlayers.)
- [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel): The GPU implemenation of the second-order CQP solver, Clarabel.
- [SCS](https://github.com/cvxgrp/scs): A first-order CQP solver that has an optional GPU-accelerated backend.
- [diffcp](https://github.com/cvxgrp/diffcp): A (Python with C-bindings) library for differentiating through (linear) cone programs.
