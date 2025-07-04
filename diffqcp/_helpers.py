"""Helper/Utility functions used """
from typing import Sequence
import numpy as np

def _to_int_list(v: np.ndarray) -> list[int]:
    """
    Utility function to ensure eqx.filter_{...} TODO(quill): finish

    Parameters
    ----------
    v : np.ndarray
        Should only contain intgers
    """
    return [int(val) for val in v]