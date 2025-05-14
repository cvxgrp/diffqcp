import torch
import numpy as np
from scipy.sparse import spmatrix

import diffqcp.cones as cone_utils
from diffqcp.linops import SymmetricOperator

class Data:

    def __init__(self,
                 cone_dict: dict[str, int | list[int]],
                 P: torch.Tensor | spmatrix,
                 A: torch.Tensor | spmatrix,
                 q: torch.Tensor | np.ndarray | list[float],
                 b: torch.Tensor | np.ndarray | list[float],
                 dtype: torch.dtype,
                 device: torch.device,
                 P_is_upper: bool = True,
    ) -> None:
        # for cone info, is there any more pre-computation
        # I can do?
        self.cones = cone_utils.parse_cone_dict(cone_dict)
        
        self.obj_matrix_init(P, P_is_upper)
        self.constr_matrix_init(A)
        self.q = q
        self.b = b

    def obj_matrix_init(self,
                        P: torch.Tensor,
                        P_is_upper: bool
    ) -> None:
        # TODOS (quill):
        #   -- save row, col info needed to compute efficient vector outer products
        #   -- save diagonal info needed if want to use only upper triangular part of P
        #       That said, I do think saving whole matrix makes greater since for GPU
        #       computations (or when we care more about reducing flops than memory).
        self.P_is_upper = P_is_upper
    
    def constr_matrix_init(self, A: torch.Tensor | spmatrix) -> None:
        # TODOS (quill):
        #   -- save row, col info needed to compute efficient vector outer products
        #   -- this ^ can be part of computing A^T, plus additional info so that forming
        #       future A^Ts is super efficient
        pass

    @property
    def P(self) -> torch.Tensor | SymmetricOperator:
        pass
    
    @property.setter
    def P(self) -> None:
        pass

    # TODO (quill): how do I want to handle the data matrices?
    #   specifically, don't want to be overwriting data, but also
    #   don't want to keep extra data around.
    #   At least as long as data is being updated then keeping
    #   around isn't a problem.