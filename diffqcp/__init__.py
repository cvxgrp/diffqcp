"""diffqcp: derivatives of the solution map of a quadratic cone program, in JAX.

Public API:

- `HostQCP`, `QCPStructureCPU`: JVPs and VJPs on the CPU (P as its upper triangle, BCOO).
- `DeviceQCP`, `QCPStructureGPU`: JVPs and VJPs on the GPU (full P, BCSR).
- `differentiable_solution`, `make_differentiable`: `jax.grad` through a conic solve.
- `LSMRSolver`, `DenseDirectSolver`, `ReducedDirectSolver`, `AbstractDerivativeSolver`: derivative-system solvers.
- `clarabel_to_scs_permutation`: reorder PSD rows from Clarabel to SCS order.
"""

from .autodiff import differentiable_solution as differentiable_solution
from .autodiff import make_differentiable as make_differentiable
from .cones.ordering import clarabel_to_scs_permutation as clarabel_to_scs_permutation
from .problem_data import QCPStructureCPU as QCPStructureCPU
from .problem_data import QCPStructureGPU as QCPStructureGPU
from .qcp import DeviceQCP as DeviceQCP
from .qcp import HostQCP as HostQCP
from .solvers import AbstractDerivativeSolver as AbstractDerivativeSolver
from .solvers import DenseDirectSolver as DenseDirectSolver
from .solvers import LSMRSolver as LSMRSolver
from .solvers import ReducedDirectSolver as ReducedDirectSolver
