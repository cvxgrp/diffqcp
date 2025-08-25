<h1 align='center'>diffqcp: Differentiating through quadratic cone programs</h1>

`diffqcp` is a [JAX](https://docs.jax.dev/en/latest/) library that enables forming the derivative of the solution map to a quadratic cone program (QCP) with respect to the QCP problem data as an abstract linear operator and computing Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) with this operator.
The implementation is based on the derivations in our paper (see below) and computes
these products implicitly via projections onto cones and sparse linear system solves.
Our approach therefore differs from other libraries that compute JVPs and VJPs via
automatic differentiation of unrolled solution algorithm iterates as we exploit
the underlying structure of QCPs.

**Features include**:
- Hardware acclerated: JVPs and VJPs can be computed on CPUs, GPUs, and (theoretically) TPUs.
- Support for all canonical classes of convex optimization problems including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - and semidefinite programs (SDPs).

## Quadratic cone programs

A quadratic cone program is given by the primal and dual problems

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

`diffqcp` currently supports QCPs whose cone is the Cartesian product of the zero cone, the positive orthant, second-order cones, and positive semidefinite cones. Support for exponential and power cones (and their dual cones) is in development.
For more information about these cones, see the appendix of our paper.

## Usage

**We view `diffqcp` as an enabling library.** Specifically, `diffqcp` is
being integrated as a backend to CVXPYlayers--it is not designed to be a stand-alone
library. Nonetheless, here is how it can be most optimally used. (Note that while we'll specify
a CPU and a GPU approach, all modules are CPU and GPU compatible--we just recommend the following
as JAX's `BCSR` arrays do have CUDA backends for their `mv` operations while the `BCOO` arrays do not.)

For both of the following problems, we'll use the following objects:

```python
import cvxpy as cvx

problem = cvx.Problem(...)
prob_data, _, _ = problem.get_problem_data(cvx.CLARABEL, solver_opts={'use_quad_obj': True})
scs_cones = cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(prob_data["dims"])

x, y, s = ... # canonicalized solutions to `problem`
```

### Optimal CPU approach

If computing JVPs and VJPs on a CPU, we recommend using the `equinox.Module`s `HostQCP` and `QCPStructureCPU` as demonstrated in the following pseudo-example.

```python
from diffqcp import HostQCP, QCPStructureCPU
from jax.experimental.sparse import BCOO
from jaxtyping import Array

P: BCOO = ... # Only the upper triangular part of the QCP matrix P
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

### Optimal GPU approach

If computing JVPs and VJPs on a GPU, we recommend using the `equinox.Module`s `QCPStructureGPU` and `DeviceQCP`.

```python
from diffqcp import DeviceQCP, QCPStructureGPU
from jax.experimental.sparse import BCSR
from jaxtyping import Array

P: BCSR = ... # The entirety of the QCP matrix P
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
# `dP`, `dA` will be BCOO arrays, `dq`, `db` just Arrays
dP, dA, dq, db = qcp.vjp(f1(x), f2(y), f3(s)) 
```

## Citation

## Next steps

`diffqcp` is still in development! WIP features and improvements include:
- Support for the exponential cone, the power cone, and their dual cones.
- Batched problem computations.
- Migration of tests from our [torch branch](https://github.com/cvxgrp/diffqcp/tree/torch-implementation).
- Heuristic JVP and VJP computations when the solution map of a QCP is non-differentiable.

## See also

**Core dependencies** (`diffqcp` makes essential use of the following libraries)
- [Equinox](https://github.com/patrick-kidger/equinox): Neural networks and everything not already in core JAX (via callable `PyTree`s).
- [Lineax](https://github.com/patrick-kidger/lineax): Linear solvers.

**Related** 
- [CVXPYlayers](https://github.com/cvxpy/cvxpylayers): Construct differentiable convex optimization layers using [CVXPY](https://github.com/cvxpy/cvxpy/). (WIP: `diffqcp` is being added as a backend for CVXPYlayers.)
- [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel): The GPU implemenation of the second-order QCP solver, Clarabel.
- [SCS](https://github.com/cvxgrp/scs): A first-order QCP solver that has an optional GPU-accelerated backend.
- [diffcp](https://github.com/cvxgrp/diffcp): A (Python with C-bindings) library for differentiating through (linear) cone programs.
