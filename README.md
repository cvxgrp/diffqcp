<h1 align='center'>diffqcp: Differentiating through quadratic cone programs</h1>

`diffqcp` is a Python, [torch](https://github.com/pytorch/pytorch)-backed library that enables forming the derivative
of the solution map to a quadratic cone program (QCP) with respect
to the QCP problem data as an abstract linear operator and computing
Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs)
with this operator.

mention somewhere implicit differentiation / exploiting mathematical
structure.

Existing features include:
- GPU compatible.
- Support for all canonical classes of convex optimization problems, including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - Semidefinite programs
- Support for any of the aforementioned programs that also has variables constrained to the exponential cone or power cone.
- heuristic computation when problem is not differentiable

In development features will include:
- batched JVP and VJP
- JIT compilation
- integration into/with other frameworks for solving and canonicalization
- distributed
- more parallel computations (cones)
- refactored cones

## Installation

- not going to be releasedon PyPi until more feature complete (still in 0. version), but if you want to use,
we recommend using the package manager uv and doing ...

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
where the problem data is $`P\in \mathbf{S}_+^{n}`$, $`A \in \mathbf{R}^{m \times n}`$, $`q \in \mathbf{R}^n`$,
$`b \in \mathbf{R}^m`$, and the nonempty, closed, convex cone $`\mathcal K \subseteq \mathbf{R}^m`$ with dual cone $`\mathcal{K}^*`$.
However, `diffqcp` considers $`\mathcal K`$ as fixed when computing derivatives.

Supported cones are

refer to the appendix in our paper

## Usage



### Quick example

```python
import torch
from diffqcp import QCP

qcp = QCP(P, A, q, b,
          x, y, s,
          cone_dict = K)
dP, dA, dq, db = qcp.vjp(dx, dy, ds)
```

## Citation


## General
(Update May 11 2025)

## TODOS

### Next steps
1. More gradient descent testing/debugging, particularly for power cone.
2. Implement the adjoint for QCPs (just need to consider adding in sparsity to already implemented atom). **Quill**
3. Add exponential cone info to paper appendix. **Quill**
4. Paper edits. 

### smaller items
- remove `matplotlib` from dependencies
- remember need to be careful finding adjoint of $DQ$ w.r.t. data; inner product for symmetric matrices (vectorization)
- fix PSD finite differences test (need to pass in a symmetric matrix)

## Running the code

`uv` is being used manage the `diffqcp` project. See `uv`'s [documentation](https://docs.astral.sh/uv/) for more information on how to use the tool, but once the software has been [installed](https://docs.astral.sh/uv/getting-started/installation/), you can use the codebase in the following ways.

1. Create a `.py` script in the main directory (so in the directory that contains the `diffqcp` and `tests` directory; **assume this is the directory in question going forward**) and then on the command line enter
```zsh
uv run python <script_name>.py
```
2. Run unit tests via `pytest` using the command
```zsh
uv run pytest <typical pytest CLI arguments>
```

If these commands are not working as expected, execute the command `uv sync` and try again (albeit supposedly this is happening automatically whenever `run` is used).