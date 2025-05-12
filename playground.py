import numpy as np
import torch

from diffqcp.cones import _proj_dproj_psd, _proj_dproj_soc, symm_size_to_dim, vec_symm
from diffqcp.pow_cone import proj_dproj_power_cone
from tests.utils import grad_desc_test
from diffqcp.utils import to_tensor

if __name__ == '__main__':

    np.random.seed(0)
    rng = torch.Generator().manual_seed(0)

    # ==== SOC testing ====

    n = np.random.randint(50, 100)

    p_target = torch.randn(n, generator=rng, dtype=torch.float64)
    p0 = torch.randn(n, generator=rng, dtype=torch.float64)

    result = grad_desc_test(_proj_dproj_soc, p_target, p0, verbose=True, num_iter=350)

    # print("initial point: ", p_target)
    # print("found point: ", result.final_pt)
    print("SOC grad test was passed: ", result.passed)
    print("final obj val for SOC grad test: ", result.final_obj)
    result.plot_obj_traj()

    
    # ==== Power Cone testing ====
    
    n = 3
    
    p_target = torch.randn(n, generator=rng, dtype=torch.float64)
    p0 = torch.randn(n, generator=rng, dtype=torch.float64)

    p_and_dp = lambda p : proj_dproj_power_cone(p, alpha=0.75)

    result = grad_desc_test(p_and_dp, p_target, p0, verbose=True, num_iter=350)

    # print("initial point: ", p_target)
    # print("found point: ", result.final_pt)
    print("POW grad test was passed: ", result.passed)
    print("final obj val for POW grad test: ", result.final_obj)
    result.plot_obj_traj()
    
    # ==== PSD testing ====
    
    x = np.random.randn(n, n)
    x = x + x.T
    x_tch = to_tensor(x, dtype=torch.float64)
    p_target = vec_symm(x_tch)
    
    # repeat for starting point
    x = np.random.randn(n, n)
    x = x + x.T
    x_tch = to_tensor(x, dtype=torch.float64)
    p0 = vec_symm(x_tch)

    result = grad_desc_test(_proj_dproj_psd, p_target, p0, verbose=True, num_iter=500)

    # print("initial point: ", p_target)
    # print("found point: ", result.final_pt)
    print("PSD grad test was solved: ", result.passed)
    print("final obj val for PSD grad test: ", result.final_obj)
    result.plot_obj_traj()
