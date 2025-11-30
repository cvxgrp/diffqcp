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

# Quadratic cone programs

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

## Selecting solvers

As detailed in our paper, the JVPs and VJPs are computed via a linear system solve.
For this solve, `diffqcp` provides three options:
- LSMR via `lineax`, an indirect method that does not materialize the coefficient matrix.
- LU via `lineax`, a direct method that materializes the dense coefficient matrix.
- A direct method via `nvmath-python` / `cuDSS` that materializes the dense coefficient matrix.

The default solve method is `lineax`'s LU. The LSMR solve is not packaged by default as it is not
currently in a released `lineax` version, but it is accessible if you build `diffqcp` from source.
To switch between the solvers, provide `jax-lsmr`, `jax-lu`, or `nvmath-direct` (as strings) to the optional
`solve_method` parameter of an `AbstractQCP`'s `jvp` and `vjp` methods. 

**Future direction:** We're currently working on materializing the coefficient matrix as a sparse array, not a dense matrix. The `lineax` LU method would still require forming the dense matrix,
but the cuDSS backed-solve already accepts sparse arrays in CSR layout.

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
- Re-incorporate the LSMR solver when `lineax` has a new release.
- Consider JAX's [`spsolve`](https://docs.jax.dev/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html#jax.experimental.sparse.linalg.spsolve).
- Provide options to linear system solvers.
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
