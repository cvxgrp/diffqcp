import torch
import numpy as np
import scipy.sparse as sparse

from diffqcp.problem_data import ProblemData

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda')]

# Put problem data here to use?

# What do I need to test here?
# For objective matrix

# Generate a sparse, symmetric matrix
# Generate a sparse m x n matrix
# Generate p and q arrays
# Just use an empty cone dictionary
# One probdata that uses full P and one that does not.


@pytest.mark.parametrize("device", devices)
def test_obj_matrix_init(self):
    # helper = SparseHelper()
    indices = torch.tensor([[0, 0, 1, 2, 2], [0, 2, 1, 1, 2]])
    values = torch.tensor([1.0, 0.0, 3.0, 4.0, 5.0])
    P = torch.sparse_coo_tensor(indices, values, size=(3, 3))



    helper.obj_matrix_init(P, P_is_upper=True)

    P_T = helper.get_P_T()
    P_diag = helper.get_P_diag()

    self.assertTrue(torch.allclose(P_diag, torch.tensor([1.0, 3.0, 5.0])))
    self.assertEqual(P_T._indices().shape[1], helper.P._indices().shape[1])

@pytest.mark.parametrize("device", devices)
def test_constr_matrix_init(self):
    helper = SparseHelper()
    A_dense = torch.tensor([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])
    A = A_dense.to_sparse().to_sparse_csr()

    helper.constr_matrix_init(A)
    A_T = helper.get_A_T()

    self.assertEqual(A_T.shape, (3, 3))
    self.assertTrue(torch.allclose(
        A_T.to_dense(), A_dense.T
    ))