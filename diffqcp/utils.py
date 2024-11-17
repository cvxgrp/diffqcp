import numpy as np
import pylops as lo

class Scalar(lo.LinearOperator):
    """
    
    see documentation for how to test the operator
    
    version in diffcp cpp file can take in a vector for _matvec
    
    this successfully ran in a Jupyter notebook
    """

    def __init__(self, num: float):
        self.num = num
        super().__init__(dtype=np.dtype(float),
                         shape=(1, 1))
    
    
    def _matvec(self, x: float) -> float:
        return self.num*x

    
    def _rmatvec(self, x: np.ndarray) -> float:
        return self._matvec(x)