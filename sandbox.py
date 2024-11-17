import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import pylops as lo
import cvxpy as cp
import clarabel

from diffqcp.cones import parse_cone_dict, pi
from diffqcp.qcp import compute_derivative

# cone_dict = {
#     'z': 3,
#     'l': 3,
#     'q': [5]
# }

# cones = parse_cone_dict(cone_dict)

# for cone, sz in cones:
#     print(cone)
#     sz = sz if isinstance(sz, (tuple, list)) else (sz,)
#     print(sz)

np.random.seed(0)

m, n = 10, 5

A = np.random.randn(m, n)
b = np.random.randn(m)

# Compute analytical solution and derivative

Dx_b = la.solve(A.T @ A, A.T)
absDx_b = sla.aslinearoperator(Dx_b)
x_ls = Dx_b @ b
f0_ls = la.norm(A @ x_ls - b)**2

x = cp.Variable(n)
r = cp.Variable(m)

f0 = cp.sum_squares(r)

problem = cp.Problem(cp.Minimize(f0), [r == A@x - b])

probdata, _, _ = problem.get_problem_data(cp.CLARABEL)

P_can, q_can = probdata['P'], probdata['c']
A_can, b_can = probdata['A'], probdata['b']
cone_dims = probdata['dims']

clarabel_cones = [clarabel.ZeroConeT(cone_dims.zero)]
solver_settings = clarabel.DefaultSettings()
solver_settings.verbose = False

solver = clarabel.DefaultSolver(P_can, q_can,
                            A_can, b_can,
                            clarabel_cones, solver_settings)

solution = solver.solve()

cone_dict = {
    'z' : cone_dims.zero
}
cones = parse_cone_dict(cone_dict)

DS = compute_derivative(P_can, A_can, q_can, b_can, cone_dict, solution)

nonzeros = P_can.nonzero()
# data = 1e-4 * np.random.randn(P_can.size)
data = np.zeros(P_can.size)
dP = sparse.csc_matrix((data, nonzeros), shape=P_can.shape)
nonzeros = A_can.nonzero()
# data = 1e-4 * np.random.randn(A_can.size)
data = np.zeros(A_can.size)
dA = sparse.csc_matrix((data, nonzeros), shape=A_can.shape)
# dq = 1e-4 * np.random.randn(q_can.size)
db = 1e-4 * np.random.randn(b_can.size)
dx, dy, ds = DS(dP, dA, np.zeros(q_can.size), db)

print(dx)
print(dy)
print(ds)

print("=== CHECK ===")

print(Dx_b@db)
print(dx[m:])

# z = (np.array(solution.x),
#      np.array(solution.z) - np.array(solution.s),
#      np.array([1]))

# Pi_z = pi(z, cones)

# print(f"Pi_z shape: {Pi_z.shape}")

# print(f"{A_can.shape[0] + A_can.shape[1] + 1}")
