import torch
import numpy as np
import scipy.sparse as sparse
import pytest

from diffqcp.problem_data import ProblemData
import tests.utils as utils
import diffqcp.utils as diffqcp_utils

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    devices += [torch.device('cuda')]

# Generate a sparse, symmetric matrix
# Generate a sparse m x n matrix
# Generate p and q arrays
# Just use an empty cone dictionary
# One probdata that uses full P and one that does not.


@pytest.mark.parametrize("device", devices)
def test_prob_data_full_P_scipy(device):
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    m = np.random.randint(low=10, high=20)
    n = m + np.random.randint(low=5, high=15)
    N = n + m + 1

    P, _, A, q, b = utils.generate_problem_data_new(n, m, sparse.random_array,
                                                        np.random.randn)
    
    K = {} # just use an empty cone dict. `parse_cone_dict` will just return empty list.

    data = ProblemData(K, P, A, q, b, dtype=torch.float64, device=device, P_is_upper=False)

    assert torch.allclose(diffqcp_utils.to_tensor(P.todense(), torch.float64, device=device),
                          data.P.to_dense())
    assert torch.allclose(diffqcp_utils.to_tensor(A.todense(), torch.float64, device=device),
                          data.A.to_dense())

    indices = P.nonzero()
    P_rows, P_cols = indices[0], indices[1]
    assert torch.allclose(diffqcp_utils.to_tensor(P_rows, dtype=torch.int64, device=device),
                          data.P_rows)
    assert torch.allclose(diffqcp_utils.to_tensor(P_cols, dtype=torch.int64, device=device),
                          data.P_cols)
    assert torch.allclose(diffqcp_utils.to_tensor(P.indptr, dtype=torch.int64, device=device),
                          data.Pcrow_indices)
    assert torch.allclose(diffqcp_utils.to_tensor(P.indices, dtype=torch.int64, device=device),
                          data.Pcol_indices)
    
    indices = A.nonzero()
    A_rows, A_cols = indices[0], indices[1]
    assert torch.allclose(diffqcp_utils.to_tensor(A_rows, dtype=torch.int64, device=device),
                          data.A_rows)
    assert torch.allclose(diffqcp_utils.to_tensor(A_cols, dtype=torch.int64, device=device),
                          data.A_cols)
    assert torch.allclose(diffqcp_utils.to_tensor(A.indptr, dtype=torch.int64, device=device),
                          data.Acrow_indices)
    assert torch.allclose(diffqcp_utils.to_tensor(A.indices, dtype=torch.int64, device=device),
                          data.Acol_indices)
    
    assert torch.allclose(diffqcp_utils.to_tensor(A.T.todense(), dtype=torch.float64, device=device),
                          data.AT.to_dense())
    
    print("A T INFO: ", data.AT)
    
    # now create new problem and make the checks again

    new_A = utils.get_random_like(A, randomness = lambda n: np.random.normal(0, 1e-6, size=n))
    new_P = utils.get_random_like(P, randomness = lambda n: np.random.normal(0, 1e-6, size=n))

    data.A = new_A
    data.P = new_P
    
    assert torch.allclose(diffqcp_utils.to_tensor(new_P.todense(), torch.float64, device=device),
                          data.P.to_dense())
    assert torch.allclose(diffqcp_utils.to_tensor(new_A.todense(), torch.float64, device=device),
                          data.A.to_dense())
    
    indices = new_P.nonzero()
    P_rows, P_cols = indices[0], indices[1]
    assert torch.allclose(diffqcp_utils.to_tensor(P_rows, dtype=torch.int64, device=device),
                          data.P_rows)
    assert torch.allclose(diffqcp_utils.to_tensor(P_cols, dtype=torch.int64, device=device),
                          data.P_cols)
    assert torch.allclose(diffqcp_utils.to_tensor(new_P.indptr, dtype=torch.int64, device=device),
                          data.Pcrow_indices)
    assert torch.allclose(diffqcp_utils.to_tensor(new_P.indices, dtype=torch.int64, device=device),
                          data.Pcol_indices)
    
    indices = new_A.nonzero()
    A_rows, A_cols = indices[0], indices[1]
    assert torch.allclose(diffqcp_utils.to_tensor(A_rows, dtype=torch.int64, device=device),
                          data.A_rows)
    assert torch.allclose(diffqcp_utils.to_tensor(A_cols, dtype=torch.int64, device=device),
                          data.A_cols)
    assert torch.allclose(diffqcp_utils.to_tensor(new_A.indptr, dtype=torch.int64, device=device),
                          data.Acrow_indices)
    assert torch.allclose(diffqcp_utils.to_tensor(new_A.indices, dtype=torch.int64, device=device),
                          data.Acol_indices)
    
    print("NEW AT INFO: ", data.AT)
    
    print(data.AT.to_dense())
    
    assert torch.allclose(diffqcp_utils.to_tensor(new_A.T.todense(), dtype=torch.float64, device=device),
                          data.AT.to_dense())
    
    P, _, A, q, b = utils.generate_problem_data_new(n, m, sparse.random_array,
                                                        np.random.randn, density=0.1)
    
    with pytest.raises(ValueError):
        # make sure we get failure if sparsity changes
        data.A = A
    with pytest.raises(ValueError):
        # make sure we get failure if sparsity changes
        data.P = P
    

@pytest.mark.parametrize("device", devices)
def test_prob_data_full_P_torch(device):
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    m = np.random.randint(low=10, high=20)
    n = m + np.random.randint(low=5, high=15)
    N = n + m + 1

    P, _, A, AT, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                np.random.randn, dtype=torch.float64,
                                                                device=device)
    
    K = {} # just use an empty cone dict. `parse_cone_dict` will just return empty list?

    data = ProblemData(K, P, A, q, b, dtype=torch.float64, device=device, P_is_upper=False)