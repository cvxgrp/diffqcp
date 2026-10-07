"""Canonical CQP fixtures used by tests (and shared with experiments/).

A "fixture" here is a deterministic construction of a `cvxpy.Problem` plus
helpers to canonicalise it (`QCPProbData`) into the data Clarabel and
`diffqcp` consume. Each generator takes either a numpy `Generator` or a
seed so call sites are reproducible.

Wave 2 introduced this module by consolidating the prior
`experiments/cvx_problem_generator.py` and the problem-data dataclass that
lived in `tests/helpers.py`. Experiments and tests both import from here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import clarabel
import cvxpy as cvx
import numpy as np
import scipy.linalg as la
from scipy import sparse

from diffqcp import clarabel_to_scs_permutation

# Generator type — accept a seed, an existing rng, or None (uses default rng).
Rng = np.random.Generator


def _as_rng(rng_or_seed: Rng | int | None) -> Rng:
    if isinstance(rng_or_seed, np.random.Generator):
        return rng_or_seed
    return np.random.default_rng(rng_or_seed)


# ─────────────────────────────────────────────────────────────────────────────
# Problem generators
#
# Each function returns a fresh `cvx.Problem`. They are deterministic given
# `rng_or_seed`. Helpers from here (or `QCPProbData` below) canonicalise into
# the form Clarabel/diffqcp consume.
# ─────────────────────────────────────────────────────────────────────────────


def randn_symm(n: int, rng: Rng) -> np.ndarray:
    A = rng.standard_normal((n, n))
    return (A + A.T) / 2


def generate_sdp(n: int, p: int, rng_or_seed: Rng | int | None = None) -> cvx.Problem:
    """SDP from https://www.cvxpy.org/examples/basic/sdp.html.

    Note: with random data this is frequently infeasible or unbounded, so do not
    use it where an optimal solution is required; use `generate_feasible_sdp`.
    """
    rng = _as_rng(rng_or_seed)
    C = randn_symm(n, rng)
    A = [randn_symm(n, rng) for _ in range(p)]
    b = [float(rng.standard_normal()) for _ in range(p)]

    X = cvx.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cvx.trace(A[i] @ X) == b[i] for i in range(p)]
    return cvx.Problem(cvx.Minimize(cvx.trace(C @ X)), constraints)


def generate_feasible_sdp(
    n: int, p: int, rank: int = 3, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    """SDP that is strictly primal *and* dual feasible, so an optimum is attained.

    Primal: a known full-rank X* satisfies the equality constraints.
    Dual: C = sum_i lambda_i A_i + S with S positive definite, so (lambda, S) is
    strictly dual feasible. Without the dual construction a random C makes the
    problem unbounded and the "solution" a certificate of unboundedness.
    """
    del rank  # historical; not used currently
    rng = _as_rng(rng_or_seed)
    Z = rng.standard_normal((n, n))
    X_star = Z @ Z.T  # PSD, full rank
    A = [randn_symm(n, rng) for _ in range(p)]
    b = [float(np.trace(Ai @ X_star)) for Ai in A]
    lambd = rng.standard_normal(p)
    W = rng.standard_normal((n, n))
    C = sum(li * Ai for li, Ai in zip(lambd, A, strict=True)) + W @ W.T + np.eye(n)

    X = cvx.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    constraints += [cvx.trace(A[i] @ X) == b[i] for i in range(p)]
    return cvx.Problem(cvx.Minimize(cvx.trace(C @ X)), constraints)


def generate_portfolio_problem(n: int, rng_or_seed: Rng | int | None = None) -> cvx.Problem:
    rng = _as_rng(rng_or_seed)
    mu = cvx.Parameter(n)
    mu.value = rng.standard_normal(n)
    Sigma = rng.standard_normal((n, n))
    Sigma = Sigma.T @ Sigma
    Sigma_sqrt = cvx.Parameter((n, n))
    Sigma_sqrt.value = la.sqrtm(Sigma)
    w = cvx.Variable((n, 1))
    gamma = 3.43046929e01
    ret = mu.T @ w
    risk = cvx.sum_squares(Sigma_sqrt @ w)
    return cvx.Problem(cvx.Maximize(ret - gamma * risk), [cvx.sum(w) == 1, w >= 0])


def generate_least_squares_eq(
    m: int, n: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    """Least-squares with simplex constraint — strongly convex, unique soln."""
    assert m >= n
    rng = _as_rng(rng_or_seed)
    x = cvx.Variable(n)
    b = cvx.Parameter(m)
    b.value = rng.standard_normal(m)
    A = cvx.Parameter((m, n))
    A.value = rng.standard_normal((m, n))
    assert np.linalg.matrix_rank(A.value) == n
    objective = cvx.sum_squares(A @ x - b)
    constraints = [x >= 0, cvx.sum(x) == 1.0]
    problem = cvx.Problem(cvx.Minimize(objective), constraints)
    assert problem.is_dpp()
    return problem


def generate_dense_qp(
    n: int, m: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    """Inequality-constrained QP with a dense objective matrix.

    CVXPY canonicalizes `sum_squares` with auxiliary variables, so the other
    QP fixtures all have a *diagonal* P; this one exercises off-diagonal P.
    """
    rng = _as_rng(rng_or_seed)
    M = rng.standard_normal((n, n))
    Q = M @ M.T + 0.1 * np.eye(n)
    c = rng.standard_normal(n)
    G = rng.standard_normal((m, n))
    h = G @ rng.standard_normal(n) + rng.random(m)  # strictly feasible
    x = cvx.Variable(n)
    return cvx.Problem(cvx.Minimize(0.5 * cvx.quad_form(x, Q) + c @ x), [G @ x <= h])


def generate_LS_problem(m: int, n: int, rng_or_seed: Rng | int | None = None) -> cvx.Problem:
    """Plain least-squares with auxiliary residual variable."""
    rng = _as_rng(rng_or_seed)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    x = cvx.Variable(n)
    r = cvx.Variable(m)
    f0 = cvx.sum_squares(r)
    return cvx.Problem(cvx.Minimize(f0), [r == A @ x - b])


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def generate_group_lasso_logistic(
    m: int, n: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    rng = _as_rng(rng_or_seed)
    X = rng.standard_normal((m, 10 * n))
    true_beta = np.zeros(10 * n)
    true_beta[: 10 * n // 100] = 1.0
    y = np.round(_sigmoid(X @ true_beta + rng.standard_normal(m) * 0.5))

    beta = cvx.Variable(10 * n)
    lambd = 0.1
    loss = -cvx.sum(cvx.multiply(y, X @ beta) - cvx.logistic(X @ beta))
    reg = lambd * cvx.sum(cvx.norm(beta.reshape((-1, 10), "C"), axis=1))
    return cvx.Problem(cvx.Minimize(loss + reg))


def generate_group_lasso(
    m: int, n: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    rng = _as_rng(rng_or_seed)
    X = cvx.Parameter((m, 10 * n))
    X.value = rng.standard_normal((m, 10 * n))
    true_beta = np.zeros(10 * n)
    true_beta[: 10 * n // 100] = 1.0
    y = X @ true_beta + rng.standard_normal(m) * 0.5

    beta = cvx.Variable(10 * n)
    lambd = cvx.Parameter(pos=True)
    lambd.value = 0.1
    loss = cvx.sum_squares(y - X @ beta)
    reg = lambd * cvx.sum(cvx.norm(beta.reshape((-1, 10), "C"), axis=1))
    prob = cvx.Problem(cvx.Minimize(loss + reg))
    assert prob.is_dpp()
    return prob


def generate_robust_mvdr_beamformer(
    n: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    rng = _as_rng(rng_or_seed)
    w = cvx.Variable((n, 1), complex=True)

    Sigma = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Sigma = Sigma @ Sigma.conj().T + 0.1 * np.eye(n)
    Sigma_sqrt = cvx.Parameter((n, n), complex=True)
    Sigma_sqrt.value = la.sqrtm(Sigma)

    a_hat = 5 * rng.standard_normal(n) + 1j * rng.standard_normal(n)
    P = cvx.Parameter((n, n), complex=True)
    P.value = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))

    f0 = cvx.sum_squares(Sigma_sqrt @ w)
    obj = cvx.Minimize(f0)

    gamma = 1.0
    delta = 0.5

    constraints = [
        cvx.real(a_hat.conj().T @ w) >= gamma,
        cvx.norm(P.conj().T @ w, 2) <= delta,
    ]
    prob = cvx.Problem(obj, constraints)
    assert prob.is_dpp()
    return prob


def generate_kalman_smoother(
    random_inputs: np.ndarray,
    random_noise: np.ndarray,
    T: int = 5,
    n: int = 100,
    rng_or_seed: Rng | int | None = None,
) -> cvx.Problem:
    """Kalman smoother. `random_inputs`/`random_noise` provided by the caller
    so this generator is fully deterministic.
    """
    rng = _as_rng(rng_or_seed)
    _, delt = np.linspace(0, T, n, endpoint=True, retstep=True)
    gamma = 0.05

    A = np.zeros((4, 4))
    B = np.zeros((4, 2))
    C = np.zeros((2, 4))

    A[0, 0] = 1
    A[1, 1] = 1
    A[0, 2] = (1 - gamma * delt / 2) * delt
    A[1, 3] = (1 - gamma * delt / 2) * delt
    A[2, 2] = 1 - gamma * delt
    A[3, 3] = 1 - gamma * delt

    B[0, 0] = delt**2 / 2
    B[1, 1] = delt**2 / 2
    B[2, 0] = delt
    B[3, 1] = delt

    C[0, 0] = 1
    C[1, 1] = 1

    x_hist = np.zeros((4, n + 1))
    x_hist[:, 0] = [0, 0, 0, 0]
    y_hist = np.zeros((2, n))

    w_in = random_inputs
    v_in = random_noise

    for t in range(n):
        y_hist[:, t] = C @ x_hist[:, t] + v_in[:, t]
        x_hist[:, t + 1] = A @ x_hist[:, t] + B @ w_in[:, t]

    x = cvx.Variable(shape=(4, n + 1))
    w = cvx.Variable(shape=(2, n))
    v = cvx.Variable(shape=(2, n))

    tau = cvx.Parameter(pos=True)
    tau.value = float(rng.uniform(0.1, 5))

    obj = cvx.sum_squares(w) + tau * cvx.sum_squares(v)
    obj = cvx.Minimize(obj)

    constr = []
    for t in range(n):
        constr += [
            x[:, t + 1] == A @ x[:, t] + B @ w[:, t],
            y_hist[:, t] == C @ x[:, t] + v[:, t],
        ]
    prob = cvx.Problem(obj, constr)
    assert prob.is_dpp()
    return prob


def generate_pow_projection_problem(
    n: int, rng_or_seed: Rng | int | None = None
) -> cvx.Problem:
    """Project x onto product of 3D power cones with random alphas."""
    assert n % 3 == 0
    rng = _as_rng(rng_or_seed)
    x_target = rng.standard_normal(n)
    num_cones = n // 3
    var = cvx.Variable(n)
    constraints = []
    for i in range(num_cones):
        alpha = max(float(rng.random()), 0.01)
        constraints.append(cvx.PowCone3D(var[3 * i], var[3 * i + 1], var[3 * i + 2], alpha))
    objective = cvx.Minimize(cvx.sum_squares(var - x_target))
    return cvx.Problem(objective, constraints)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical problem data: cvx.Problem → matrices + Clarabel solution
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QCPProbData:
    """Canonical-form data for a CQP, plus the Clarabel solution."""

    problem: cvx.Problem

    # Symmetric quadratic objective matrix, three formats.
    Pcsc: sparse.csc_matrix | sparse.csc_array = field(init=False)
    Pcsr: sparse.csr_matrix | sparse.csr_array = field(init=False)
    Pcoo: sparse.coo_matrix | sparse.coo_array = field(init=False)
    # Upper-triangular slice — the form `diffqcp` (CPU path) wants.
    Pupper_csc: sparse.csc_matrix | sparse.csc_array = field(init=False)
    Pupper_csr: sparse.csr_matrix | sparse.csr_array = field(init=False)
    Pupper_coo: sparse.coo_matrix | sparse.coo_array = field(init=False)

    Acsc: sparse.csc_matrix | sparse.csc_array = field(init=False)
    Acsr: sparse.csr_matrix | sparse.csr_array = field(init=False)
    Acoo: sparse.coo_matrix | sparse.coo_array = field(init=False)

    q: np.ndarray = field(init=False)
    b: np.ndarray = field(init=False)

    n: int = field(init=False)
    m: int = field(init=False)

    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)
    s: np.ndarray = field(init=False)
    status: str = field(init=False)
    scs_perm: np.ndarray = field(init=False)

    scs_cones: dict = field(init=False)
    clarabel_cones: list = field(init=False)

    def __post_init__(self) -> None:
        clarabel_probdata, _, _ = self.problem.get_problem_data(
            cvx.CLARABEL, ignore_dpp=True, solver_opts={"use_quad_obj": True}
        )

        self.q = clarabel_probdata["c"]
        self.n = int(np.size(self.q))
        self.b = clarabel_probdata["b"]
        self.m = int(np.size(self.b))

        if "P" in clarabel_probdata:
            self.Pcsr = clarabel_probdata["P"].tocsr()
            self.Pcsc = self.Pcsr.tocsc()
            self.Pcoo = self.Pcsr.tocoo()
            self.Pupper_csr = sparse.triu(self.Pcsr).tocsr()
            self.Pupper_csc = self.Pupper_csr.tocsc()
            self.Pupper_coo = self.Pupper_csr.tocoo()
        else:
            P = np.zeros((self.n, self.n))
            self.Pcsr = sparse.csr_matrix(P)
            self.Pcsc = self.Pcsr.tocsc()
            self.Pcoo = self.Pcsr.tocoo()
            self.Pupper_csr = self.Pcsr.copy()
            self.Pupper_csc = self.Pcsc.copy()
            self.Pupper_coo = self.Pcoo.copy()

        self.Acsr = clarabel_probdata["A"].tocsr()
        self.Acsc = self.Acsr.tocsc()
        self.Acoo = self.Acsr.tocoo()

        self.clarabel_cones = (
            cvx.reductions.solvers.conic_solvers.clarabel_conif.dims_to_solver_cones(
                clarabel_probdata["dims"]
            )
        )
        self.scs_cones = (
            cvx.reductions.solvers.conic_solvers.scs_conif.dims_to_solver_dict(
                clarabel_probdata["dims"]
            )
        )

        solver_settings = clarabel.DefaultSettings()
        solver_settings.verbose = False
        solver = clarabel.DefaultSolver(
            self.Pupper_csc, self.q, self.Acsc, self.b, self.clarabel_cones, solver_settings
        )
        soln = solver.solve()
        self.status = str(soln.status)
        self.x = np.array(soln.x)
        self.y = np.array(soln.z)
        self.s = np.array(soln.s)

        # Rows above are in Clarabel order (PSD blocks upper-triangular,
        # column-major); `diffqcp` wants SCS order. See `scs_ordered`.
        self.scs_perm = clarabel_to_scs_permutation(self.scs_cones)

    def scs_ordered(self) -> tuple[sparse.coo_matrix, np.ndarray, np.ndarray, np.ndarray]:
        """`(A, b, y, s)` with rows permuted into the SCS order `diffqcp` expects.

        Identical to the stored Clarabel-ordered data unless there are PSD cones.
        """
        perm = self.scs_perm
        A = sparse.coo_matrix(self.Acsr[perm])
        return A, self.b[perm], self.y[perm], self.s[perm]
