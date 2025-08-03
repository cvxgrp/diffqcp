<h1 align='center'>diffqcp: Differentiating through quadratic cone programs</h1>

`diffqcp` is a [JAX](https://docs.jax.dev/en/latest/) library that enables forming the derivative of the solution map to a quadratic cone program (QCP) with respect to the QCP problem data as an abstract linear operator and computing Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) with this operator.

Discuss
- implicit differentiation approach to argmin differentiation (exploiting mathematical structure)
- DPP (relevant for batched problems)
- Automatic differentiation.

**Features include**:
- Hardware acclerated: JVPs and VJPs can be computed on CPUs, GPUs, and (theoretically) TPUs.
- Support for all canonical classes of convex optimization problems including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - and semidefinite programs (SDPs).
- Heuristic JVP and VJP computations when the solution map of a QCP is non-differentiable. (TODO(quil): ensure this is still the case with JAX and lineax)
- Batched JVP and VJP computations(i.e. ...)
- Batched problem computions (so yes, this means that )
- Distributed

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

`diffqcp` currently supports QCPs whose cone is the Cartesian product of the zero cone, positive orthant, second-order cones, and positive semidefinite cones. Support for exponential and power cones (and their dual cones) is in development (see the Todos below).
For more information about these cones, see the appendix of our paper.

## Citation

## See also: 

**Enabling libraries:**
- [Equinox](https://github.com/patrick-kidger/equinox): Neural networks and everything not already in core JAX. (Callable PyTrees.)
- [Lineax](https://github.com/patrick-kidger/lineax): Linear solvers.

**Related** TODO(quill): finish this
- [CVXPYlayers](https://github.com/cvxpy/cvxpylayers) make note about CVXPY
- [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel)
- [SCS](https://github.com/cvxgrp/scs) now supports GPU computations
- [diffcp](https://github.com/cvxgrp/diffcp)


## TODOs:

Note that after failing to achieve desired performance with a torch-backed implementation (branch [here](https://github.com/cvxgrp/diffqcp))

Furthermore

**Functionality**
- Support for the exponential (and dual exponential) cone. (Just requires re-implementing the PyTorch version in JAX following best practices as found in `lineax` or `optimistix`.)
- Support for the power (and dual power) cone. (Same approach as for exponential cone.)
- Can `HostQCP` and `DeviceQCP` be combined?
    - Only difference is the use of `BCOO` arrays for the CPU "optimized" verion vs. `BCSR` arrays for the GPU "optimized" version
    - Overall the architecture of the library can be improved.
- Finish batched problem functionality
    - The cone `proj_dproj` methods already support this functionality
- ensure `vmap` works over `jvp` and `vjp`
- Allow factoring-based solves
    - requires `as_matrix` to be implemented for all custom `lineax.AbstractLinearOperator`s.
    - Would need to have non-sparse returning atom functions.
- more explicit host and device array placement (right now have to use flag.)
- differentiable?
- Upgrade the cone library so that it can stand alone (i.e.)

**Integration**
- CVXPYlayers

**Testing**
- most of the testing exists in the torch branch, so need to port over key tests (i.e., not tests that just validate functionality that I know exists, but tests that ensure future change don't break anything)
