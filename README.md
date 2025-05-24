<h1 align='center'>diffqcp: Differentiating through quadratic cone programs</h1>

`diffqcp` is a Python, (for now) [torch](https://github.com/pytorch/pytorch)-backed
(prototype) library that enables forming the derivative
of the solution map to a quadratic cone program (QCP) with respect
to the QCP problem data as an abstract linear operator and computing
Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs)
with this operator.

`diffqcp` is an implicit differentiation approach to argmin differentiation (exploiting mathematical structure)

Existing features include:
- **GPU compatible**: `diffqcp` can compute JVPs and VJPs on either CPUs or GPUs.
- Support for all canonical classes of convex optimization problems, including
    - linear programs (LPs),
    - quadratic programs (QPs),
    - second-order cone programs (SOCPs),
    - and semidefinite programs.
- Support for any of the aforementioned programs that also has variables constrained to the exponential cone or power cone. (More broadly, `diffqcp` supports any convex optimization problem that can be written as a QCP whose cone is the Cartesian product of eight canonical cones. See the "Quadratic cone programs" section below.)
- Heuristic JVP and VJP computations when the solution map of a QCP is non-differentiable.

In development features include:
- Batched JVPs and VJPs.
- JIT compilation.
- Integration with frameworks that
    - Canonicalize arbitrary convex optimization problems to QCPs.
    - Solve QCPs (on CPUs or GPUs).
- Distributed JVPs and VJPs.
- A more modern implementation, which will include a better interface 
- Better implementation of projecting onto cones (more parallel compuations and a better interface).

## Installation

- not going to be released on PyPi until more feature complete (still in 0. version), but if you want to use,
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
where $`x \in \mathbf{R}^n`$ is the *primal* variable, $`y \in \mathbf{R}^m`$ is the *dual* variable, and $`s \in \mathbf{R}^m`$ is the primal *slack* variable. The problem data are $`P\in \mathbf{S}_+^{n}`$, $`A \in \mathbf{R}^{m \times n}`$, $`q \in \mathbf{R}^n`$, and $`b \in \mathbf{R}^m`$. We assume that $`\mathcal K \subseteq \mathbf{R}^m`$ is a nonempty, closed, convex cone with dual cone $`\mathcal{K}^*`$.

`diffqcp` supports QCPs whose cone is the Cartesian product of the zero cone, positive orthant, second-order cone, positive semidefinite cone, exponential cone, dual exponential cone, power cone, and dual power cone. For more information about these cones, see the appendix of our paper.

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