import lineax as lx
from lineax import AbstractLinearOperator
import equinox as eqx
from equinox import AbstractVar
from abc import abstractmethod
from jaxtyping import Array, Float

# TODO(quill): determine if we want to make these public--easier to work with the "magic keys"
#   -> consequential action item: remove the prepended underscore.
_ZERO = "z"
_NONNEGATIVE = "l"
_SOC = "q"
_PSD = "s"
_EXP = "ep"
_EXP_DUAL = "ed"
_POW = 'p'
# Note we don't define a POW_DUAL cone as we stick with SCS convention
# and use -alpha to create a dual power cone.

# The ordering of CONES matches SCS.
CONES = [_ZERO, _NONNEGATIVE, _SOC, _PSD, _EXP, _EXP_DUAL, _POW]

class Cone(eqx.Module):
    is_dual: AbstractVar[bool]

    @abstractmethod
    def proj_dproj(x: Float[Array, "d"]) -> tuple[Float[Array, "d"], AbstractLinearOperator]:
        pass

class ZeroCone(Cone):
    is_dual: bool

    def proj_dproj(x: Float[Array, "d"]) -> tuple[Float[Array, "d"], AbstractLinearOperator]:
        pass

class NonnegativeCone(Cone):
    is_dual: bool

    def proj_dproj(x: Float[Array]):
        pass