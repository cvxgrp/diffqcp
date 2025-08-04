<h1 align='center'>diffqcp: Differentiating through quadratic cone programs</h1>

`diffqcp` is a [JAX](https://docs.jax.dev/en/latest/) library that enables forming the derivative of the solution map to a quadratic cone program (QCP) with respect to the QCP problem data as an abstract linear operator and computing Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs) with this operator.

TODO(quill): (briefly) Discuss
- implicit differentiation approach to argmin differentiation (exploiting mathematical structure)
- DPP (relevant for batched problems)
- Automatic differentiation.

**Features include**:
- Hardware acclerated: JVPs and VJPs can be computed on CPUs, GPUs, and (theoretically) TPUs.
- Support for all canonical classes of convex optimization problems including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - and semidefinite programs (SDPs). TODO(quill): implement before release...should be easy

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

`diffqcp` currently supports QCPs whose cone is the Cartesian product of the zero cone, the positive orthant, second-order cones, and positive semidefinite cones. Support for exponential and power cones (and their dual cones) is in development (see the TODOs below).
For more information about these cones, see the appendix of our paper.

## Citation

## See also

**Core dependencies** (`diffqcp` makes essential use of the following libraries)
- [Equinox](https://github.com/patrick-kidger/equinox): Neural networks and everything not already in core JAX (via callable `PyTree`s).
- [Lineax](https://github.com/patrick-kidger/lineax): Linear solvers.

**Related** 
- [CVXPYlayers](https://github.com/cvxpy/cvxpylayers): Construct differentiable convex optimization layers using [CVXPY](https://github.com/cvxpy/cvxpy/). (WIP: `diffqcp` is being added as a backend for CVXPYlayers.)
- [CuClarabel](https://github.com/oxfordcontrol/Clarabel.jl/tree/CuClarabel): The GPU implemenation of the second-order QCP solver, Clarabel.
- [SCS](https://github.com/cvxgrp/scs): A first-order QCP solver that has an optional GPU-accelerated backend.
- [diffcp](https://github.com/cvxgrp/diffcp): A (Python with C-bindings) library for differentiating through (linear) cone programs.


## TODOs:

After failing to achieve desired performance with a torch-backed implementation (branch [here](https://github.com/cvxgrp/diffqcp/tree/torch-implementation)), this JAX implementation of `diffqcp` was rapidly developed. Consequently, there is some tech debt:

**Functionality**
- **TODO(quill)--important**: Heuristic JVP and VJP computations when the solution map of a QCP is non-differentiable (`lineax` just fails if LSMR doesn't converge, whereas our torch version and `diffcp` just return the last iterate).
- Support for the exponential (and dual exponential) cone. (Just requires re-implementing the PyTorch version in JAX following best practices as found in `lineax` or `optimistix`.)
- Support for the power (and dual power) cone. (Same approach as for exponential cone.)
- Batched JVP and VJP computations (via `vmap`--should just work since we can already `jit`)
- Batched problem computions--*i.e.*, constructing *derivatives* of solution *maps* to a batch of DPP-compliant problems. (so yes, `diffqcp` is aiming to support multi-level batching: you can batch compute JVPs and VJPs over a batch of problems.)
    - The cone `proj_dproj` methods already support this functionality
- Can `HostQCP` and `DeviceQCP` be combined?
    - Only difference is the use of `BCOO` arrays for the CPU "optimized" verion vs. `BCSR` arrays for the GPU "optimized" version
    - Other architecture improvements? (Be sure to add performance regression tests before making large changes.)
- Allow factor-solve based JVPs and VJPs
    - requires `as_matrix` to be implemented for all custom `lineax.AbstractLinearOperator`s.
    - Would need to have non-sparse returning atom functions.
- Similarly, allow for changing the tolerance of the LSMR solve.
- more explicit host and device array placement (right now have to use flag to specify whether to use single or double precision.)
- Differentiable? (*i.e.*, what happns if we use `jax`'s auto-diff functionality--would this computation correspond to anything meaningful?)
- Clean up the cone library so it can stand alone (*i.e.*, it can be a JAX library for projecting onto convex cones and computing derivatives of these projections)
    - so will require separate `proj` and `dproj` methods,
    - plus just cleaner abstractions,
    - and removal of tech debt
- See if `diffqcp` just works for distributed computations out of the box

**Testing**
- Most of the testing exists in the torch branch, so need to port over key tests--*i.e.*, not tests that were just initial (research) validation tests, but tests that ensure future change don't break anything.
