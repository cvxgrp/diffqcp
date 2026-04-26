from abc import abstractmethod

import equinox as eqx
from lineax import AbstractLinearOperator
from jaxtyping import Float, Array

class AbstractConeProjector(eqx.Module):

    @abstractmethod
    def proj(self, x: Float[Array, " n"]) -> Float[Array, " n"]:
        """Project a point onto a convex cone.
        
        :param x: Description
        :type x: Float[Array, " n"]
        :return: Description
        :rtype: Array
        """
        raise NotImplementedError
    
    @abstractmethod
    def dproj(self, x: Float[Array, " n"]) -> AbstractLinearOperator:
        """Return the derivative of the projection onto a convex cone at x.
        
        :param x: Description
        :type x: Float[Array, " n"]
        :return: Description
        :rtype: AbstractLinearOperator
        """
        raise NotImplementedError
    
    @abstractmethod
    def proj_dproj(self, x: Float[Array, " _n"]) -> tuple[Float[Array, " _n"], AbstractLinearOperator]:
        """Project x onto the cone and return the Jacobian operator of the projection at this point.

        Do not provide a default implementation in accordance with abstract/final design pattern.
        
        Purpose of this combined method is computational efficiency:
        
        :param x: Description
        :type x: Float[Array, " _n"]
        :return: Description
        :rtype: tuple[Array, AbstractLinearOperator]
        """
        raise NotImplementedError

    def __call__(self, x: Float[Array, " _n"]):
        return self.proj_dproj(x)