import time
import os

import numpy as np
from scipy import sparse
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import lineax as lx
import equinox as eqx
import clarabel
import matplotlib.pyplot as plt
import patdb

import experiments.cvx_problem_generator as prob_generator
from tests.helpers import QCPProbData, scoo_to_bcoo, scsr_to_bcsr
from diffqcp import DeviceQCP, QCPStructureGPU

def _get_dense_mat(mat: lx.AbstractLinearOperator, N):
    mv = lambda vec: mat.mv(vec)
    mm = jax.vmap(mv, in_axes=1, out_axes=1)
    return mm(jnp.eye(N))


def get_info(
    prob_data: QCPProbData,
    experiment_name: str,
    plot: bool = True
):
    print(f"=== Starting experiment {experiment_name} ===")
    cones = prob_data.clarabel_cones
    settings = clarabel.DefaultSettings()
    settings.verbose = False

    solver = clarabel.DefaultSolver(prob_data.Pupper_csc,
                                    prob_data.q,
                                    prob_data.Acsc,
                                    prob_data.b,
                                    cones,
                                    settings)
    
    start_solve = time.perf_counter()
    solution = solver.solve()
    end_solve = time.perf_counter()
    
    if str(solution.status) != "Solved":
        print(f"Experiment: {experiment_name} could not be solved. "
              f"Clarabel returned with status: {solution.status}")
        return False
    
    print(f"Clarabel solve took: {end_solve - start_solve} seconds")

    target_x = jnp.array(solution.x)
    target_y = jnp.array(solution.z)
    target_s = jnp.array(solution.s)
    
    P = scsr_to_bcsr(prob_data.Pcsr)
    A = scsr_to_bcsr(prob_data.Acsr)
    q = prob_data.q
    b = prob_data.b
    scs_cones = prob_data.scs_cones

    print(f"P shape: {prob_data.Pcsr.shape}")
    print(f"P nnz: {prob_data.Pcsr.count_nonzero()}")
    print(f"A shape: {prob_data.Acsr.shape}")
    print(f"A nnz: {prob_data.Acsr.count_nonzero()}")
    print("cones: ", prob_data.clarabel_cones)
    
    problem_structure = QCPStructureGPU(P, A, scs_cones)
    qcp = DeviceQCP(P, A, q, b,
                    target_x, target_y, target_s,
                    problem_structure)
    
    atoms = qcp._form_atoms()
    F = atoms[1]

    F_dense = np.asarray(_get_dense_mat(F, problem_structure.N))
    total_elements = F_dense.size
    num_zeros = np.count_nonzero(F_dense == 0)
    cond2 = np.linalg.cond(F_dense)
    print("F condition number:", cond2)

    # sparsity = num_zeros / total_elements
    sparsity = np.count_nonzero(F_dense) / total_elements
    print(f"Sparsity: {sparsity}")

    rows, cols = F_dense.shape
    # Larger matrices -> choose plotting strategy
    markersize = max(1, 5000 / max(rows, cols))
    nnz = np.count_nonzero(F_dense)
    total_elements = rows * cols

    # thresholds (tune as needed)
    MAX_SPARSEY_SIZE = int(1e6)     # max total elements to safely call plt.spy on
    MAX_SCATTER_POINTS = int(200_000)  # cap number of scatter points when sampling

    results_dir = os.path.join(os.path.dirname(__file__), "results/sparsity_experiments")
    output_path = os.path.join(results_dir, f"{experiment_name}.png")

    if plot:
    
        if total_elements <= MAX_SPARSEY_SIZE and nnz <= MAX_SCATTER_POINTS:
            # small enough -> full spy
            plt.figure(figsize=(6, 6))
            plt.spy(F_dense, markersize=markersize)
            plt.title(f"$F$ operator ({rows} x {cols}) sparsity pattern")
            plt.xlabel("Columns")
            plt.ylabel("Rows")
            plt.savefig(output_path, format="png")
            plt.close()
        else:
            # large -> sample nonzeros and scatter (fast & memory-light)
            nz = np.argwhere(F_dense != 0)
            num_nz = nz.shape[0]
            if num_nz == 0:
                plt.figure(figsize=(6, 6))
                plt.text(0.5, 0.5, "All zeros", ha="center", va="center")
                plt.title(f"$F$ operator ({rows} x {cols}) - all zeros")
                plt.savefig(output_path, format="png")
                plt.close()
            else:
                if num_nz > MAX_SCATTER_POINTS:
                    rng = np.random.default_rng(0)
                    idx = rng.choice(num_nz, size=MAX_SCATTER_POINTS, replace=False)
                    sample = nz[idx]
                else:
                    sample = nz
                plt.figure(figsize=(6, 6))
                # note: argwhere returns (row, col)
                plt.scatter(sample[:, 1], sample[:, 0], s=1, marker=".", alpha=0.6)
                plt.gca().invert_yaxis()
                plt.title(f"$F$ operator ({rows} x {cols}) sparsity sample ({min(num_nz, MAX_SCATTER_POINTS)} pts)")
                plt.xlabel("Columns")
                plt.ylabel("Rows")
                plt.xlim(-0.5, cols - 0.5)
                plt.ylim(rows - 0.5, -0.5)
                plt.savefig(output_path, format="png")
                plt.close()

    return True


if __name__ == "__main__":

    np.random.seed(28)

    ls_problem_medium = prob_generator.generate_LS_problem(m=200, n=100)
    ls_problem_medium_data = QCPProbData(ls_problem_medium)
    get_info(ls_problem_medium_data, "LS_medium", plot=False)

    ls_problem_large = prob_generator.generate_LS_problem(m=2000, n=1000)
    ls_problem_large_data = QCPProbData(ls_problem_large)
    get_info(ls_problem_large_data, "LS_large", plot=False)

    port_problem_small = prob_generator.generate_portfolio_problem(n=200)
    port_problem_small_data = QCPProbData(port_problem_small)
    get_info(port_problem_small_data, "portfolio_small", plot=False)

    port_problem_medium = prob_generator.generate_portfolio_problem(n=1_000)
    port_problem_medium_data = QCPProbData(port_problem_medium)
    get_info(port_problem_medium_data, "portfolio_medium", plot=False)
    
    max_attempts = 100
    
    # attempt_num = 0
    # while attempt_num < max_attempts:
    #     # sdp_medium = prob_generator.generate_sdp(n=30, p=30)
    #     sdp_medium = prob_generator.generate_feasible_sdp(n=5, p=5)
    #     # sdp_medium = prob_generator.generate_sdp(n=3, p=3)
    #     sdp_medium_data = QCPProbData(sdp_medium)
    #     if get_info(sdp_medium_data, "sdp_medium"):
    #         break
    #     attempt_num += 1

    # attempt_num = 0
    # while attempt_num < max_attempts:
    #     sdp_large = prob_generator.generate_sdp(n=300, p=5)
    #     sdp_large_data = QCPProbData(sdp_large)
    #     get_info(sdp_large_data, "sdp_large", plot=False)
    #     attempt_num += 1

    # attempt_num = 0
    while attempt_num < max_attempts:
        pow_medium = prob_generator.generate_pow_projection_problem(n=33)
        pow_medium_data = QCPProbData(pow_medium)
        if get_info(pow_medium_data, "pow_medium", plot=False):
            break
        attempt_num += 1

    attempt_num = 0
    while attempt_num < max_attempts:
        pow_large = prob_generator.generate_pow_projection_problem(n=333)
        pow_large_data = QCPProbData(pow_large)
        if get_info(pow_large_data, "pow_large", plot=False):
            break
        attempt_num += 1