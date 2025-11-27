import numpy as np
import cvxpy as cvx
import clarabel
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from diffqcp import QCPStructureCPU, QCPStructureGPU, HostQCP, DeviceQCP
from tests.helpers import QCPProbData, scoo_to_bcoo, scsr_to_bcsr

def compare_gradients(A, b, C, n):
    Z = cvx.Variable((n, n), symmetric=True)
    constraints = [Z >> 0]
    vec_Z = cvx.vec(Z, order="F")
    constraints.append(A @ vec_Z == b)
    objective = cvx.Minimize(cvx.trace(C @ Z))
    prob = cvx.Problem(objective, constraints)

    prob_data = QCPProbData(prob)

    settings = clarabel.DefaultSettings()
    settings.verbose = False
    solver = clarabel.DefaultSolver(prob_data.Pupper_csc,
                                    prob_data.q,
                                    prob_data.Acsc,
                                    prob_data.b,
                                    prob_data.clarabel_cones,
                                    settings)
    solution = solver.solve()

    host_qcp = HostQCP(
        P=scoo_to_bcoo(prob_data.Pupper_coo),
        A=scoo_to_bcoo(prob_data.Acoo),
        q=jnp.array(prob_data.q),
        b=jnp.array(prob_data.b),
        x=jnp.array(solution.x),
        y=jnp.array(solution.z),
        s=jnp.array(solution.s),
        problem_structure=QCPStructureCPU(
            P=scoo_to_bcoo(prob_data.Pupper_coo),
            A=scoo_to_bcoo(prob_data.Acoo),
            cone_dims=prob_data.scs_cones
        )
    )
    device_qcp = DeviceQCP(
        P=scsr_to_bcsr(prob_data.Pcsr),
        A=scsr_to_bcsr(prob_data.Acsr),
        q=jnp.array(prob_data.q),
        b=jnp.array(prob_data.b),
        x=jnp.array(solution.x),
        y=jnp.array(solution.z),
        s=jnp.array(solution.s),
        problem_structure=QCPStructureGPU(
            P=scsr_to_bcsr(prob_data.Pcsr),
            A=scsr_to_bcsr(prob_data.Acsr),
            cone_dims=prob_data.scs_cones
        )
    )

    dx = jnp.ones(len(solution.x))

    _, dA_host, dq_host, db_host = host_qcp.vjp(
        dx, jnp.zeros_like(host_qcp.y), jnp.zeros_like(host_qcp.s)
    )

    _, dA_device, dq_device, db_device = device_qcp.vjp(
        dx, jnp.zeros_like(device_qcp.y), jnp.zeros_like(device_qcp.s)
    )

    db_diff = jnp.linalg.norm(jnp.array(db_host) - db_device) / jnp.linalg.norm(db_host)
    dA_diff = jnp.linalg.norm(jnp.array(dA_host.todense()) - dA_device.todense()) / jnp.linalg.norm(dA_host.todense())
    
    print(f"db diff: {db_diff}")
    print(f"dA_dff: {dA_diff}")

if __name__ == "__main__":
    n = 3
    A = np.array([[ 0.6098,  0.9365, -0.5868, -0.8762,  2.5231, -0.0405, -0.4607,  1.0639,
             -0.2023],
            [ 0.9633, -0.9168, -0.2633,  0.8062,  0.4248,  0.1785,  0.6259,  0.7198,
              0.8306],
            [ 0.5087, -1.5266, -1.9011,  0.2482, -0.0872, -0.3489,  0.7842, -0.9984,
              1.1860],
            [ 0.4137,  0.1841,  1.4228, -0.0176, -0.4991,  0.9396,  1.0793,  0.2924,
              0.0775],
            [ 0.0865,  0.8037,  0.9151, -0.7554,  0.0665, -0.0050, -0.4473,  0.9084,
              0.1129]])
    b = np.array([ 73.3233,  32.3913, -18.0499,  13.1492,  13.9454])
    C = np.array([[ 4.7208,  4.2384, -1.1487],
            [ 4.2384,  6.3524, -0.3547],
            [-1.1487, -0.3547,  3.0890]])
    P = None
    
    compare_gradients(A, b, C, n)