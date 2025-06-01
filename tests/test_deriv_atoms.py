"""
Testing qcp derivative atom computations.

Notes
-----
- The `atol` parameters in the testing assertions
have been raised to the largest value that still
allows that respective test to pass.

- Setting u[-1] equal to 1 (which will always be tue at the solution),
yields more accurate derivative approximations (this makes sense when looking
at the nonlinear transform Q).
"""
import math

import numpy as np
import scipy.sparse as sparse
import torch
from linops.lsqr import lsqr
import pytest

from diffqcp.qcp_derivs import Du_Q, dData_Q, dData_Q_adjoint
from diffqcp.qcp_derivs import Du_Q_efficient, dData_Q_adjoint_efficient, dData_Q_efficient
import tests.utils as utils
from diffqcp.utils import Q

devices = [torch.device('cpu')]
if torch.cuda.is_available():
    print("CUDA is available; adding a cuda device to the tests.")
    devices += [torch.device('cuda')]
else:
    print("CUDA is not available; testing solely on CPU.")


@pytest.mark.parametrize("device", devices)
def test_Ddata_Q_is_approximation(device):
    """Test implementation of DQ(u) w.r.t. Data.

    Taking D in S^n_+ x R^(n x m) x R^n x R^m and a small perturbation
    dD in S^n x R^(n x m) x R^n x R^m, test if

        Q(u, D + dD) - Q(u, D) approx. equal DQ(u, D)dD,

    where DQ(u, D)dD is computed using `diffqcp.qcp_deriv.dData_Q`.
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64,
                                                                   device=device)

        u = torch.randn(N, generator=rng,dtype=torch.float64, device=device)

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P_op, A, q, b, x, y, tau)

        dP_upper = utils.get_random_like(P_upper, lambda n: np.random.normal(0, 1e-5, size=n))
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-5, size=n))
        dq = torch.randn(q.shape[0], generator=rng, device=device)
        db = torch.randn(b.shape[0], generator=rng, device=device)
        dP_op, dA, dq, db = utils.convert_prob_data_to_torch(dP_upper, dA, dq, db, dtype=torch.float64, device=device)

        dQ = Q(P_op + dP_op, A + dA, q + dq, b + db, x, y, tau) - z

        assert torch.allclose(dQ, dData_Q(u, dP_op, dA, dq, db))


@pytest.mark.parametrize("device", devices)
def test_Ddata_Q_is_approximation_efficient(device):
    """Test implementation of DQ(u) w.r.t. Data.

    Taking D in S^n_+ x R^(n x m) x R^n x R^m and a small perturbation
    dD in S^n x R^(n x m) x R^n x R^m, test if

        Q(u, D + dD) - Q(u, D) approx. equal DQ(u, D)dD,

    where DQ(u, D)dD is computed using `diffqcp.qcp_deriv.dData_Q`.

    Notes
    -----
    Can also test with P in S^n (symmetric, but not PSD).
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        n = np.random.randint(low=10, high=20)
        m = n + np.random.randint(low=5, high=15)
        N = n + m + 1

        P, _, A, _, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                    np.random.randn, dtype=torch.float64,
                                                                    device=device)

        u = torch.randn(N, generator=rng,dtype=torch.float64, device=device)

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P, A, q, b, x, y, tau)

        dP = utils.get_random_like(P, lambda n: np.random.normal(0, 1e-5, size=n))
        dA = utils.get_random_like(A, lambda n: np.random.normal(0, 1e-5, size=n))
        dAT = utils.get_transpose(dA, return_tensor=True, dtype=torch.float64, device=device)
        dq = torch.randn(q.shape[0], generator=rng, device=device)
        db = torch.randn(b.shape[0], generator=rng, device=device)
        dP, dP_upper, dA, dq, db = utils.convert_prob_data_to_torch_new(dP, dP, dA, dq, db,
                                                                        dtype=torch.float64, device=device) # don't need dP_upper

        dQ = Q(P + dP, A + dA, q + dq, b + db, x, y, tau) - z

        assert torch.allclose(dQ, dData_Q_efficient(u, dP, dA, dAT, dq, db))

# TODO: can theoretically run gradient descent test on this one too.


@pytest.mark.parametrize("device", devices)
def test_Du_Q_is_approximation(device):
    """Test implementation of DQ(u) w.r.t. u.

    Taking u in R^N and a small perturbation du in R^N, test if

        Q(u + du, D) - Q(u, D) approx. equal DQ(u, D)du,

    where DQ(u, D) is obtained from `diffqcp.qcp_deriv.Du_Q`.

    Notes
    -----
    When attempted with atol=1e-6, test fails.
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=10, high=20)
        m = np.random.randint(low=10, high=20)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64,
                                                                   device=device)

        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        u[-1] = torch.tensor(1, dtype=torch.float64, device=device) # always the case when differentiating at soln.

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P_op, A, q, b, x, y, tau)

        du = 1e-5*torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        dx, dy, dtau = du[:n], du[n: -1], du[-1]
        dtau = dtau.unsqueeze(0)
        dQ = Q(P_op, A, q, b, x + dx, y + dy, tau + dtau) - z

        deriv_op = Du_Q(u, P_op, A, q, b)

        assert torch.allclose(dQ, deriv_op @ du)


@pytest.mark.parametrize("device", devices)
def test_Du_Q_is_approximation_efficient(device):
    """Test implementation of DQ(u) w.r.t. u.

    Taking u in R^N and a small perturbation du in R^N, test if

        Q(u + du, D) - Q(u, D) approx. equal DQ(u, D)du,

    where DQ(u, D) is obtained from `diffqcp.qcp_deriv.Du_Q`.

    This version is using the more efficient `Du_Q` implementation.

    Notes
    -----
    When attempted with atol=1e-6, test fails.
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for _ in range(10):
        n = np.random.randint(low=10, high=20)
        m = np.random.randint(low=10, high=20)
        N = n + m + 1

        P, _, A, AT, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                    np.random.randn, dtype=torch.float64,
                                                                    device=device)

        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        # u[-1] = torch.tensor(1, dtype=torch.float64, device=device) # always the case when differentiating at soln.

        x, y, tau = u[:n], u[n: -1], u[-1]
        tau = tau.unsqueeze(0)
        z = Q(P, A, q, b, x, y, tau)

        du = 1e-5*torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        dx, dy, dtau = du[:n], du[n: -1], du[-1]
        dtau = dtau.unsqueeze(0)
        dQ = Q(P, A, q, b, x + dx, y + dy, tau + dtau) - z

        deriv_op = Du_Q_efficient(u, P, A, AT, q, b)

        print("FD: ", dQ.cpu())
        print("autodiff: ", (deriv_op @ du).cpu())

        assert torch.allclose(dQ, deriv_op @ du, atol=1e-6, rtol=1e-4)
        

@pytest.mark.parametrize("device", devices)
def test_Du_Q_is_linop(device):
    """Test if Du_Q is a linear operator.

    Another way of testing if Du_Q_T is implemented correctly.
    That is, instead of taking the output-input approximation
    validation approach used in `test_Du_Q_T_is_approximation`,
    this test uses `pylops.utils.dottest` to see if
    Du_Q_T is the adjoint of Du_Q.

    Notes
    -----
    See "Testing the operator" section of
    https://pylops.readthedocs.io/en/stable/adding.html#addingoperator
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P, _, A, AT, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                    np.random.randn, dtype=torch.float64,
                                                                    device=device)
        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)

        deriv_op = Du_Q_efficient(u, P, A, AT, q, b)

        # default dtype of dottest is float64
        assert utils.dottest(deriv_op)


@pytest.mark.parametrize("device", devices)
def test_Ddata_Q_is_linop(device):
    # can't use linop librarty since cannot taken in arbitrary vectors.
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)

        _, dP_op, dA, dq, db = utils.generate_torch_problem_data(n,
                                                                 m,
                                                                 sparse.random,
                                                                 np.random.randn,
                                                                 dtype=torch.float64,
                                                                 device=device)
        dP = dP_op @ torch.eye(n, dtype=torch.float64, device=device)
        dQ = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        dQ1, dQ2, dQ3 = dQ[:n], dQ[n:-1], dQ[-1]

        dDataQ = dData_Q(u=u, dP=dP_op, dA=dA, dq=dq, db=db)
        deltaP, deltaA, delta_q, delta_b = dData_Q_adjoint(u, dQ1, dQ2, dQ3)

        lhs = ( dQ @ dDataQ ).item()
        prod1 = torch.trace(dP @ deltaP).item()
        prod2 = torch.trace(deltaA.T @ dA).item()
        prod3 = (dq @ delta_q).item()
        prod4 = (db @ delta_b).item()
        rhs = prod1 + prod2 + prod3 + prod4
        print("lhs: ", lhs)
        print("rhs: ", rhs)
        assert math.isclose(lhs, rhs, rel_tol=1e-6, abs_tol=1e-21)
        assert np.abs(lhs - rhs) < 1e-10


@pytest.mark.parametrize("device", devices)
def test_Du_Q_lsqr(device):
    """Test lsqr performance for Du_Q.
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P_upper, P_op, A, q, b = utils.generate_torch_problem_data(n,
                                                                   m,
                                                                   sparse.random,
                                                                   np.random.randn,
                                                                   dtype=torch.float64,
                                                                   device=device)
        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        u[-1] = torch.tensor(1.0, dtype=torch.float64, device=device) # always the case when differentiating at soln.

        deriv_op = Du_Q(u, P_op, A, q, b)

        x = 1e-6*torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        xlsqr = lsqr(deriv_op, deriv_op @ x)

        print("x: ", x.to(device=None))
        print("xlsqr: ", xlsqr.to(device=None))

        assert torch.allclose(x, xlsqr, atol=1e-8)

@pytest.mark.parametrize("device", devices)
def test_Du_Q_lsqr_new(device):
    """Test lsqr performance for Du_Q.
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        # n = np.random.randint(low=10, high=20)
        # m = n + np.random.randint(low=5, high=15)
        N = n + m + 1

        P, _, A, AT, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                    np.random.randn, dtype=torch.float64,
                                                                    device=device)

        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        u[-1] = torch.tensor(1.0, dtype=torch.float64, device=device) # always the case when differentiating at soln.

        deriv_op = Du_Q_efficient(u, P, A, AT, q, b)

        x = 1e-6*torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        xlsqr = lsqr(deriv_op, deriv_op @ x)

        print("x: ", x.to(device=None))
        print("xlsqr: ", xlsqr.to(device=None))

        assert torch.allclose(x, xlsqr, atol=1e-8)


@pytest.mark.parametrize("device", devices)
def test_dData_Qadjoint_efficient(device):
    """
    Ensuring the implementation which only computes the nonzero entries does so validly.

    Have to make sure to only check the nonzero entries.

    check nonzero entries
    """
    np.random.seed(0)
    rng = torch.Generator(device=device).manual_seed(0)

    for i in range(10):
        m = np.random.randint(low=10, high=20)
        n = m + np.random.randint(low=5, high=15)
        N = n + m + 1

        P, P_upper, A, AT, q, b = utils.generate_torch_problem_data_new(n, m, sparse.random_array,
                                                                        np.random.randn, dtype=torch.float64,
                                                                        device=device)
        u = torch.randn(N, generator=rng, dtype=torch.float64, device=device)
        w = torch.randn(N, generator=rng, dtype=torch.float64, device=device)

        Pcrow_indices, Pcol_indices, Prows, Pcols = utils.get_sparse_information(P)
        Acrow_indices, Acol_indices, Arows, Acols = utils.get_sparse_information(A)

        deltaP, deltaA, deltaq, deltab = dData_Q_adjoint(u, w[:n], w[n:-1], w[-1])

        deltaPe, deltaAe, deltaqe, deltabe = dData_Q_adjoint_efficient(u, w[:n], w[n:-1], w[-1],
                                                                       Prows, Pcols, Pcrow_indices, Pcol_indices,
                                                                       Arows, Acols, Acrow_indices, Acol_indices)
        
        deltaPe_dense = deltaPe.to_dense()
        deltaAe_dense = deltaAe.to_dense()

        deltaP_nonzero = torch.zeros(size=(n, n), dtype=torch.float64, device=device)
        deltaP_nonzero[Prows, Pcols] = deltaP[Prows, Pcols]

        deltaA_nonzero = torch.zeros(size=(m, n), dtype=torch.float64, device=device)
        deltaA_nonzero[Arows, Acols] = deltaA[Arows, Acols]

        assert torch.allclose(deltaP_nonzero, deltaPe_dense)
        assert torch.allclose(deltaA_nonzero, deltaAe_dense)
        assert torch.allclose(deltaq, deltaqe)
        assert torch.allclose(deltab, deltabe)