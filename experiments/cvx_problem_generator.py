import numpy as np
import cvxpy as cvx
import scipy.sparse as sparse

from tests.utils import data_and_soln_from_cvxpy_problem, generate_problem_data_new

def randn_symm(n, random_array):
    A = random_array(n, n)
    return (A + A.T) / 2

def generate_portfolio_problem(n, return_all, return_problem_only):
    mu = np.random.randn(n)
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    w = cvx.Variable(n)
    gamma = cvx.Parameter(nonneg=True)
    gamma.value = 3.43046929e+01
    ret = mu.T @ w
    risk = cvx.quad_form(w, Sigma)
    problem = cvx.Problem(cvx.Maximize(ret - gamma * risk), [cvx.sum(w) == 1, w >= 0])

    data = data_and_soln_from_cvxpy_problem(problem)
    P, A, q, b = data[0], data[1], data[2], data[3]
    scs_cone_dict, soln, clarabel_cones = data[4], data[5], data[6]

    x = np.array(soln.x)
    y = np.array(soln.z)
    s = np.array(soln.s)

    if return_all:
        return P, A, q, b, scs_cone_dict, clarabel_cones, x, y, s
    else:
        return x, y, s
    
def generate_least_squares_eq(m, n, return_all, return_problem_only: bool = False):
    """Generate a conic problem with unique solution."""
    assert m >= n
    x = cvx.Variable(n)
    b = np.random.randn(m)
    A = np.random.randn(m, n)
    assert np.linalg.matrix_rank(A) == n
    objective = cvx.pnorm(A @ x - b, 1)
    constraints = [x >= 0, cvx.sum(x) == 1.0]
    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    if return_problem_only:
        return problem
    
    data = data_and_soln_from_cvxpy_problem(problem)
    P, A, q, b = data[0], data[1], data[2], data[3]
    scs_cone_dict, soln, clarabel_cones = data[4], data[5], data[6]

    x = np.array(soln.x)
    y = np.array(soln.z)
    s = np.array(soln.s)

    if return_all:
        return P, A, q, b, scs_cone_dict, clarabel_cones, x, y, s
    else:
        return x, y, s
    
def generate_LS_problem(m, n, return_all=True, return_problem_only: bool=False):
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cvx.Variable(n)
    r = cvx.Variable(m)
    f0 = cvx.sum_squares(r)
    problem = cvx.Problem(cvx.Minimize(f0), [r == A@x - b])

    if return_problem_only:
        return problem

    data = data_and_soln_from_cvxpy_problem(problem)
    P, A, q, b = data[0], data[1], data[2], data[3]
    scs_cone_dict, soln, clarabel_cones = data[4], data[5], data[6]

    x = np.array(soln.x)
    y = np.array(soln.z)
    s = np.array(soln.s)

    if return_all:
        return P, A, q, b, scs_cone_dict, clarabel_cones, x, y, s
    else:
        return x, y, s
    
def generate_sdp(n, p, return_all=True, return_problem_only: bool=False):
    data = generate_problem_data_new(n=n, m=n, sparse_random_array=sparse.random_array,
                                     random_array=np.random.randn, P_psd=True)
    C = data[0].todense()
    
    # data = generate_problem_data_new(n=n, m=n, sparse_random_array=sparse.random_array,
    #                                  random_array=np.random.randn, P_psd=True)
    # D = data[0].todense()

    As = [randn_symm(n, np.random.randn) for _ in range(p)]
    Bs = np.random.randn(p)

    X = cvx.Variable((n, n), PSD=True)
    y = cvx.Variable(n)
    # objective = cvx.trace(C @ X) + cvx.quad_form(y, D, assume_PSD=True)
    objective = cvx.trace(C @ X)
    constraints = [cvx.trace(As[i] @ X) == Bs[i] for i in range(p)]
    prob = cvx.Problem(cvx.Minimize(objective), constraints)

    if return_problem_only:
        return prob

    P, A, q, b, scs_cone_dict, soln, clarabel_cones = data_and_soln_from_cvxpy_problem(prob)

    x = np.array(soln.x)
    y = np.array(soln.z)
    s = np.array(soln.s)
    if return_all:
        return P, A, q, b, scs_cone_dict, clarabel_cones, x, y, s
    else:
        return x, y, s
    
def sigmoid(z):
  return 1/(1 + np.exp(-z))



def generate_group_lasso(n: int, m: int) -> cvx.Problem:
    X = np.random.randn(m, 10 * n)
    true_beta = np.zeros(10 * n)
    true_beta[:10 * n // 100] = 1.0
    y = np.round(sigmoid(X @ true_beta + np.random.randn(m)*0.5)) 

    beta = cvx.Variable(10 * n)
    lambd = 0.1
    loss = -cvx.sum(cvx.multiply(y, X @ beta) - cvx.logistic(X @ beta))
    reg = lambd * cvx.sum( cvx.norm( beta.reshape((-1, 10), 'C'), axis=1 ) )

    prob = cvx.Problem(cvx.Minimize(loss + reg))

    return prob