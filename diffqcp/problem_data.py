import torch
import numpy as np
from scipy.sparse import spmatrix

import diffqcp.cones as cone_utils
from diffqcp.linops import SymmetricOperator

class ProblemData:

    # TODO (quill): remove all nonzeros like done in diffcp
    # TODO (quill): make sure you know where data about sparse tensor
    #   is stored (e.g., indices on gpu or cpu?)
    # TODO (quill): for batching PSD cone, that functionality can probably just go
    #   in proj_and_dproj() -> instead of looping through PSD cones (or SOCs for that
    #   matter), pass to wrappers. Potentially useful to do parsing about whether this happens
    #   here (e.g., which indices in the cone list go with each other)?
    # TODO (quill): for next phase, need to think about coalescing memory / topics
    #   like that when considering how to hold data matrices and cone computations on a
    #   single vecotr.

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
        
        # TODO (quill): can I somehow take advantage of things if P = 0? Use ZeroOperator?
        self.obj_matrix_init(P, P_is_upper)
        self.constr_matrix_init(A)
        # actually, probably need to do additional to_tensor work on q and b.
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

    # TODO (quill): Create method which creates data objects (to hold d_data) that utilizes
    #   our knowledge about the fixed data
    #   -> called a class method or something like that?