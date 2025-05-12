import numpy as np
import torch

from diffqcp.cones import _proj_dproj_psd, _proj_dproj_soc, symm_size_to_dim
from diffqcp.pow_cone import proj_dproj_power_cone
from tests.utils import grad_desc_test

if __name__ == '__main__':

    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    n = np.random.randint(10, 20)
    # dim = symm_size_to_dim(n)
    dim = 3
    
    p_target = torch.randn(dim, generator=rng, dtype=torch.float64)
    p0 = torch.randn(dim, generator=rng, dtype=torch.float64)

    p_and_dp = lambda p : proj_dproj_power_cone(p, alpha=0.95)

    result = grad_desc_test(p_and_dp, p_target, p0, verbose=True, num_iter=500)

    print("was solved: ", result.passed)
    print("final obj val: ", result.final_obj)
    result.plot_obj_traj()
