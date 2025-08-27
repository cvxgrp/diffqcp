import numpy as np
import cvxpy as cvx
import scipy.sparse as sparse
import scipy.linalg as la

def randn_symm(n, random_array):
    A = random_array(n, n)
    return (A + A.T) / 2


def generate_sdp(n, p) -> cvx.Problem:
    """
    Taken from https://www.cvxpy.org/examples/basic/sdp.html.
    """
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cvx.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cvx.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cvx.Problem(cvx.Minimize(cvx.trace(C @ X)),
                    constraints)
    return prob


def generate_portfolio_problem(n) -> cvx.Problem:
    mu = cvx.Parameter(n)
    mu.value = np.random.randn(n)
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T @ Sigma
    Sigma_sqrt = cvx.Parameter((n, n))
    w = cvx.Variable((n, 1))
    gamma = 3.43046929e+01 # fix the risk-aversion parameter.
    ret = mu.T @ w
    # risk = cvx.quad_form(w, Sigma)
    risk = cvx.sum_squares(Sigma_sqrt @ w)
    Sigma_sqrt.value = la.sqrtm(Sigma)
    problem = cvx.Problem(cvx.Maximize(ret - gamma * risk), [cvx.sum(w) == 1, w >= 0])

    return problem
    

def generate_least_squares_eq(m, n) -> cvx.Problem:
    """Generate a conic problem with unique solution.
    Taken from diffcp.
    """
    assert m >= n
    x = cvx.Variable(n)
    b = cvx.Parameter(m)
    b.value = np.random.randn(m)
    A = cvx.Parameter((m, n))
    A.value = np.random.randn(m, n)
    assert np.linalg.matrix_rank(A.value) == n
    # objective = cvx.pnorm(A @ x - b, 2)
    objective = cvx.sum_squares(A@x - b)
    constraints = [x >= 0, cvx.sum(x) == 1.0]
    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    assert problem.is_dpp()
    return problem
    

def generate_LS_problem(m, n) -> cvx.Problem:
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cvx.Variable(n)
    r = cvx.Variable(m)
    f0 = cvx.sum_squares(r)
    problem = cvx.Problem(cvx.Minimize(f0), [r == A@x - b])
    return problem

    
def sigmoid(z):
  return 1/(1 + np.exp(-z))

def generate_group_lasso_logistic(n: int, m: int) -> cvx.Problem:
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

def generate_group_lasso(n: int, m: int) -> cvx.Problem:
    X = cvx.Parameter((m, 10*n))
    X.value = np.random.randn(m, 10 * n)
    true_beta = np.zeros(10 * n)
    true_beta[:10 * n // 100] = 1.0
    y = X @ true_beta + np.random.randn(m)*0.5

    beta = cvx.Variable(10 * n)
    lambd = cvx.Parameter(pos=True)
    lambd.value = 0.1
    loss = cvx.sum_squares(y - X @ beta)
    reg = lambd * cvx.sum( cvx.norm( beta.reshape((-1, 10), 'C'), axis=1 ) )

    prob = cvx.Problem(cvx.Minimize(loss + reg))

    assert prob.is_dpp()

    return prob

def generate_robust_mvdr_beamformer(n: int) -> cvx.Problem:
    """`n` is the number of sensors."""

    w = cvx.Variable((n, 1), complex=True)

    Sigma = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    Sigma = Sigma @ Sigma.conj().T + 0.1 * np.eye(n) # Make Hermitian PSD
    Sigma_sqrt = cvx.Parameter((n, n), complex=True)
    Sigma_sqrt.value = la.sqrtm(Sigma)

    a_hat = 5 * np.random.randn(n) + 1j * np.random.randn(n) # Fake array manifold/response
    P = cvx.Parameter((n, n), complex=True) # uncertainty matrix
    P.value = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # f0 = cvx.real(cvx.sum_squares(Sigma_sqrt @ w))
    f0 = cvx.sum_squares(Sigma_sqrt @ w)
    obj = cvx.Minimize(f0)

    gamma = 1.0   # desired signal constraint
    delta = 0.5   # uncertainty size

    constraints = [
        cvx.real(a_hat.conj().T @ w) >= gamma,
        cvx.norm(P.conj().T @ w, 2) <= delta
    ]

    prob = cvx.Problem(obj, constraints)
    assert prob.is_dpp()
    return prob


def generate_kalman_smoother(
    random_inputs, random_noise, T: int=5, n: int=100
) -> cvx.Problem:
    """
    `n` is number of time steps.
    `T` is time horizon.

    Largely taken from: https://www.cvxpy.org/examples/applications/robust_kalman.html.
    """
    _, delt = np.linspace(0,T,n,endpoint=True, retstep=True)
    gamma = .05 # damping, 0 is no damping

    A = np.zeros((4,4))
    B = np.zeros((4,2))
    C = np.zeros((2,4))

    A[0,0] = 1
    A[1,1] = 1
    A[0,2] = (1-gamma*delt/2)*delt
    A[1,3] = (1-gamma*delt/2)*delt
    A[2,2] = 1 - gamma*delt
    A[3,3] = 1 - gamma*delt

    B[0,0] = delt**2/2
    B[1,1] = delt**2/2
    B[2,0] = delt
    B[3,1] = delt

    C[0,0] = 1
    C[1,1] = 1

    x = np.zeros((4,n+1))
    x[:,0] = [0,0,0,0]
    y = np.zeros((2,n))

    # generate random input and noise vectors
    w = random_inputs
    v = random_noise

    # simulate the system forward in time
    for t in range(n):
        y[:,t] = C @ x[:,t] + v[:,t]
        x[:,t+1] = A @ x[:,t] + B @ w[:,t]

    x = cvx.Variable(shape=(4, n+1))
    w = cvx.Variable(shape=(2, n))
    v = cvx.Variable(shape=(2, n))

    tau = cvx.Parameter(pos=True)
    tau.value = np.random.uniform(0.1, 5)

    obj = cvx.sum_squares(w) + tau*cvx.sum_squares(v)
    obj = cvx.Minimize(obj)

    constr = []
    for t in range(n):
        constr += [ x[:,t+1] == A@x[:,t] + B@w[:,t] ,
                    y[:,t]   == C@x[:,t] + v[:,t]  ]

    prob = cvx.Problem(obj, constr)
    assert prob.is_dpp()
    return prob