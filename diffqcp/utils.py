import numpy as np
from scipy.sparse import csc_matrix
import pylops as lo

class Scalar(lo.LinearOperator):
    """TODO: Add docstring
    """

    def __init__(self, num: float):
        self.num = num
        super().__init__(dtype=np.dtype(float),
                         shape=(1, 1))


    def _matvec(self, x: float) -> float:
        return self.num*x


    def _rmatvec(self, x: np.ndarray) -> float:
        return self._matvec(x)

def Q(P: csc_matrix,
      A: csc_matrix,
      q: np.ndarray,
      b: np.ndarray,
      x: np.ndarray,
      y: np.ndarray,
      tau: float
) -> np.ndarray:
    """Homogeneous embedding, nonlinear transform.
    """
    first_chunk = P @ x + A.T @ y + tau * q
    second_chunk = -A @ x + tau * b
    final_entry = -(1/tau) * x @ (P @ x) - q @ x - b @ y
    return np.concatenate((first_chunk,
                           second_chunk,
                           [final_entry]))
