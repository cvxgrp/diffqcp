"""Accuracy and speed of the derivative (VJP) of a QCP solution map.

Methods, all applied to the same canonical problem data (P, A, q, b, cones):

- diffqcp `LSMRSolver` (matrix-free, the default),
- diffqcp `DenseDirectSolver` (dense LU of the augmented system K of F'),
- diffqcp `ReducedDirectSolver` (dense LU of the square reduced system F-bar),
- diffcp (`mode="lsqr"`, its default, and `mode="dense"`), solving with Clarabel.
  diffcp's implicit derivative ignores P (its embedding matrix is the LP one),
  so it only runs on the P = 0 families (lp, socp, exp),
- Moreau (CPU, JAX interface, `diff_method="exact"`).

Accuracy: the relative error of the data gradient `(dP, dA, dq, db)` for a
random cotangent `(dx, dy, ds)`, against a reference computed *at the same
library's own solution* with diffqcp's formula and a dense SVD least-squares
solve. This separates the derivative's error from the forward solve's (the
forward error against a tight Clarabel solve is reported separately). In the
`conditioning` sweep the reference instead solves the float64 F' in 50-digit
arithmetic (mpmath).

Speed: median wall time of one VJP after warm-up (jitted for diffqcp; the
backward pass only for Moreau, which reuses state from its forward solve; the
`adjoint_derivative` call for diffcp). float64.

GPU: with `DIFFQCP_BENCH_DEVICE=device` (and `DIFFQCP_PLATFORM=cuda,cpu`) the
diffqcp methods run as `DeviceQCP` (full P, BCSR) on the default JAX device,
plus `solve_method="nvmath-direct"` (cuDSS) when nvmath-python is installed,
and Moreau runs on CUDA when `moreau.device_available("cuda")`. diffcp always
runs on the CPU. References are always computed on the CPU with `HostQCP`, so
GPU results compare directly with CPU ones. Each JSON row records the devices.

    python experiments/derivative_benchmark.py run <out.json> [families] [sizes]
    python experiments/derivative_benchmark.py conditioning <out.json>
    python experiments/derivative_benchmark.py report <run.json> <conditioning.json> <out.png>
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", os.environ.get("DIFFQCP_PLATFORM", "cpu"))

import clarabel
import cvxpy as cvx
import diffcp
import equinox as eqx
import jax.numpy as jnp
import moreau
import moreau.jax
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from jax.experimental.sparse import (
    BCOO,
    BCSR,
)

from diffqcp import (
    DenseDirectSolver,
    DeviceQCP,
    HostQCP,
    LSMRSolver,
    QCPStructureCPU,
    QCPStructureGPU,
    ReducedDirectSolver,
)
from diffqcp.solvers import AbstractDerivativeSolver, gauge_fixed, materialise_gauge_operator

DENSE_MAX_N = 3200  # dense methods (and the SVD reference) only up to this N
REPS = 5
# "host": HostQCP (P upper triangle, BCOO). "device": DeviceQCP (full P, BCSR) on the default JAX device.
QCP_KIND = os.environ.get("DIFFQCP_BENCH_DEVICE", "host")
assert QCP_KIND in ("host", "device"), QCP_KIND
CPU = jax.devices("cpu")[0]
MOREAU_DEVICE = "cuda" if QCP_KIND == "device" and moreau.device_available("cuda") else "cpu"
FAMILIES = ("lp", "socp", "exp", "qp", "pow")
P_ZERO_FAMILIES = ("lp", "socp", "exp")
SIZES = {  # family -> size parameter per size class
    "lp": {"S": 40, "M": 200, "L": 600, "XL": 2500},
    "socp": {"S": 30, "M": 150, "L": 500, "XL": 2000},
    "exp": {"S": 10, "M": 50, "L": 160, "XL": 600},
    "qp": {"S": 40, "M": 200, "L": 600, "XL": 2500},
    "pow": {"S": 10, "M": 60, "L": 200, "XL": 800},
}


# ─── problems ─────────────────────────────────────────────────────────────


def make_problem(family: str, k: int, seed: int = 0) -> cvx.Problem:
    rng = np.random.default_rng(seed)
    if family == "lp":  # min c^T x, G x <= h, box
        n, m = k, 2 * k
        x = cvx.Variable(n)
        G = sp.random(m, n, density=min(1.0, 8 / n), random_state=seed) + sp.eye(m, n)
        h = G @ rng.uniform(-0.5, 0.5, n) + rng.uniform(0.1, 1.0, m)
        return cvx.Problem(cvx.Minimize(rng.standard_normal(n) @ x), [G @ x <= h, x >= -1, x <= 1])
    if family == "socp":  # min c^T x, ||A_i x + b_i|| <= d_i, box
        n, nc = k, max(2, k // 4)
        x = cvx.Variable(n)
        x0 = rng.uniform(-0.5, 0.5, n)  # strictly feasible point
        cons = [x >= -1, x <= 1]
        for _ in range(nc):
            Ai = sp.random(4, n, density=min(1.0, 6 / n), random_state=rng.integers(1 << 30))
            bi = rng.standard_normal(4)
            radius = np.linalg.norm(Ai @ x0 + bi) + 0.5 + rng.uniform()
            # `cvx.SOC` directly: `norm(.) <= d` adds an epigraph variable that is
            # not unique when the cone is inactive (a degenerate problem).
            cons.append(cvx.SOC(cvx.Constant(radius), Ai @ x + bi))
        return cvx.Problem(cvx.Minimize(rng.standard_normal(n) @ x), cons)
    if family == "exp":  # l1-regularized logistic regression
        m, p = 4 * k, k
        X = rng.standard_normal((m, p)) / np.sqrt(p)
        y = (rng.uniform(size=m) < 0.5).astype(float)
        beta = cvx.Variable(p)
        z = X @ beta
        return cvx.Problem(cvx.Minimize(cvx.sum(cvx.logistic(z)) - y @ z + 0.05 * cvx.norm1(beta)))
    if family == "qp":  # min 1/2 x^T M x + c^T x, G x <= h
        n, m = k, 2 * k
        B = sp.random(n, n, density=min(1.0, 4 / n), random_state=seed)
        M = (B @ B.T + 0.1 * sp.eye(n)).tocsc()
        G = sp.random(m, n, density=min(1.0, 8 / n), random_state=seed + 1) + sp.eye(m, n)
        h = G @ rng.uniform(-0.5, 0.5, n) + rng.uniform(0.1, 1.0, m)
        x = cvx.Variable(n)
        return cvx.Problem(cvx.Minimize(0.5 * cvx.quad_form(x, M, assume_PSD=True) + rng.standard_normal(n) @ x),
                           [G @ x <= h])
    if family == "pow":  # project a point onto a product of 3D power cones, plus a linear constraint
        alphas = rng.uniform(0.15, 0.85, k)
        v = cvx.Variable(3 * k)
        cons = [cvx.PowCone3D(v[3 * i], v[3 * i + 1], v[3 * i + 2], float(a)) for i, a in enumerate(alphas)]
        cons.append(cvx.sum(v) == 1.0)
        return cvx.Problem(cvx.Minimize(0.5 * cvx.sum_squares(v - rng.standard_normal(3 * k))), cons)
    raise ValueError(family)


class Canon:
    """Clarabel-format data of a CVXPY problem: P (upper triangle) and A in COO order."""

    def __init__(self, problem: cvx.Problem, column_scale: np.ndarray | None = None):
        data, _, _ = problem.get_problem_data(cvx.CLARABEL, solver_opts={"use_quad_obj": True})
        self.q = np.asarray(data["c"], dtype=float)
        self.b = np.asarray(data["b"], dtype=float)
        self.n, self.m = self.q.size, self.b.size
        P = data.get("P")
        P = sp.csc_matrix((self.n, self.n)) if P is None else sp.csc_matrix(P)
        A = sp.csc_matrix(data["A"])
        if column_scale is not None:  # x = D x': an equivalent but worse-scaled problem
            D = sp.diags(column_scale)
            P, A, self.q = (D @ P @ D).tocsc(), (A @ D).tocsc(), column_scale * self.q
        self.P_up = sp.triu(P).tocoo()
        self.A = A.tocoo()
        self.scs_cones = dims_to_solver_dict(data["dims"])
        assert not self.scs_cones.get("s") and not self.scs_cones.get("ed")
        self.N = self.n + self.m + 1

    @property
    def has_P(self) -> bool:
        return self.P_up.nnz > 0

    def clarabel_cones(self) -> list:
        c = self.scs_cones
        cones = []
        if c.get("z"):
            cones.append(clarabel.ZeroConeT(c["z"]))
        if c.get("l"):
            cones.append(clarabel.NonnegativeConeT(c["l"]))
        cones += [clarabel.SecondOrderConeT(d) for d in c.get("q", [])]
        cones += [clarabel.ExponentialConeT() for _ in range(c.get("ep", 0))]
        cones += [clarabel.PowerConeT(a) for a in c.get("p", [])]
        return cones

    def solve_clarabel(self, tol: float = 1e-10):
        s = clarabel.DefaultSettings()
        s.verbose = False
        s.tol_gap_abs = s.tol_gap_rel = s.tol_feas = tol
        s.max_iter = 400
        P = sp.csc_matrix((self.P_up.data, (self.P_up.row, self.P_up.col)), shape=(self.n, self.n))
        sol = clarabel.DefaultSolver(P, self.q, sp.csc_matrix(self.A), self.b, self.clarabel_cones(), s).solve()
        if str(sol.status) != "Solved" and tol < 1e-8:  # e.g. InsufficientProgress: retry at the default
            return self.solve_clarabel(tol=1e-8)
        return str(sol.status), np.array(sol.x), np.array(sol.z), np.array(sol.s)


# ─── diffqcp ──────────────────────────────────────────────────────────────


class SVDSolver(AbstractDerivativeSolver):
    """Reference: dense least-squares / minimum-norm solves via SVD (`jnp.linalg.lstsq`)."""

    def solve(self, Fg, rhs):
        return jnp.linalg.lstsq(materialise_gauge_operator(Fg), rhs)[0]

    def solve_transpose(self, Fg, rhs):
        return jnp.linalg.lstsq(materialise_gauge_operator(Fg).T, rhs)[0]


def build_qcp(c: Canon, x, y, s) -> HostQCP:
    idx = lambda M: jnp.stack([jnp.asarray(M.row), jnp.asarray(M.col)], axis=1)  # noqa: E731
    P = BCOO((jnp.asarray(c.P_up.data), idx(c.P_up)), shape=(c.n, c.n))
    A = BCOO((jnp.asarray(c.A.data), idx(c.A)), shape=(c.m, c.n))
    structure = QCPStructureCPU(P, A, c.scs_cones)
    return HostQCP(P, A, jnp.asarray(c.q), jnp.asarray(c.b), jnp.asarray(x), jnp.asarray(y), jnp.asarray(s),
                   structure)


@eqx.filter_jit
def _vjp(qcp, dx, dy, ds, solver):
    dP, dA, dq, db = qcp.vjp(dx, dy, ds, solver=solver)
    return jnp.concatenate([dP.data, dA.data, dq, db])


class DeviceLayout:
    """Full-P / A CSR layout for `DeviceQCP`, and the maps from its gradients to our (COO) order."""

    def __init__(self, c: Canon):
        n, m = c.n, c.m
        up = sp.csr_matrix((c.P_up.data, (c.P_up.row, c.P_up.col)), shape=(n, n))
        self.P_full = (up + sp.triu(up, 1).T).tocsr()
        self.P_full.sort_indices()
        self.A = sp.csr_matrix(c.A)
        self.A.sort_indices()
        pos = lambda M: sp.csr_matrix((np.arange(1, M.nnz + 1), M.indices, M.indptr), shape=M.shape)  # noqa: E731
        Pp, Ap = pos(self.P_full), pos(self.A)
        idx = lambda M, r, k: (np.asarray(M[r, k]).ravel().astype(int) - 1 if r.size  # noqa: E731
                               else np.zeros(0, dtype=int))
        self.P_ij = idx(Pp, c.P_up.row, c.P_up.col)
        self.P_ji = idx(Pp, c.P_up.col, c.P_up.row)
        self.P_off = c.P_up.row != c.P_up.col
        self.A_pos = idx(Ap, c.A.row, c.A.col)
        self.nnz_P, self.nnz_A = self.P_full.nnz, self.A.nnz

    def to_ours(self, g: np.ndarray) -> np.ndarray:
        """`DeviceQCP` gradient (per stored entry of the symmetric P, CSR order) -> ours (stored upper values)."""
        gP, gA, rest = g[: self.nnz_P], g[self.nnz_P: self.nnz_P + self.nnz_A], g[self.nnz_P + self.nnz_A:]
        gP_up = gP[self.P_ij] + np.where(self.P_off, gP[self.P_ji], 0.0) if self.P_ij.size else gP[:0]
        return np.concatenate([gP_up, gA[self.A_pos], rest])


def build_device_qcp(c: Canon, layout: DeviceLayout, x, y, s) -> DeviceQCP:
    bcsr = lambda M: BCSR((jnp.asarray(M.data), jnp.asarray(M.indices), jnp.asarray(M.indptr)),  # noqa: E731
                          shape=M.shape)
    P, A = bcsr(layout.P_full), bcsr(layout.A)
    structure = QCPStructureGPU(P, A, c.scs_cones)
    return DeviceQCP(P, A, jnp.asarray(c.q), jnp.asarray(c.b), jnp.asarray(x), jnp.asarray(y), jnp.asarray(s),
                     structure)


def _timed(fn, *args) -> tuple[np.ndarray, float]:
    g = np.asarray(jax.block_until_ready(fn(*args)))  # compile + warm up
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return g, float(np.median(ts))


def diffqcp_vjp(c: Canon, point, cot, solver) -> tuple[np.ndarray, float]:
    """The VJP with `solver`, as `HostQCP` or (`DIFFQCP_BENCH_DEVICE=device`) `DeviceQCP`, in our order."""
    if QCP_KIND == "host":
        return _timed(_vjp, build_qcp(c, *point), *map(jnp.asarray, cot), solver)
    layout = DeviceLayout(c)
    g, t = _timed(_vjp, build_device_qcp(c, layout, *point), *map(jnp.asarray, cot), solver)
    return layout.to_ours(g), t


def diffqcp_nvmath_vjp(c: Canon, point, cot) -> tuple[np.ndarray, float]:
    """`DeviceQCP.vjp(..., solve_method="nvmath-direct")` (cuDSS on the dense augmented system; not jittable)."""
    layout = DeviceLayout(c)
    qcp = build_device_qcp(c, layout, *point)

    def fn(dx, dy, ds):
        dP, dA, dq, db = qcp.vjp(dx, dy, ds, solve_method="nvmath-direct")
        return jnp.concatenate([dP.data, dA.data, dq, db])

    g, t = _timed(fn, *map(jnp.asarray, cot))
    return layout.to_ours(g), t


def reference_vjp(c: Canon, point, cot) -> np.ndarray:
    """Dense SVD reference, always on the CPU with `HostQCP`."""
    with jax.default_device(CPU):
        return np.asarray(_vjp(build_qcp(c, *point), *map(jnp.asarray, cot), SVDSolver()))


def _nvmath_available() -> bool:
    if QCP_KIND != "device":
        return False
    try:
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401
    except ImportError:
        return False
    return True


# ─── diffcp ───────────────────────────────────────────────────────────────


def diffcp_vjp(c: Canon, cot, mode: str):
    """(gradient in our order (dP part zero), median adjoint time, (x, y, s))."""
    A = sp.csc_matrix(c.A)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x, y, s, _, DT = diffcp.solve_and_derivative(
            A, c.b, c.q, dict(c.scs_cones), mode=mode, solve_method="Clarabel",
            tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=400)
        dA, db, dq = DT(*cot)
        ts = []
        for _ in range(REPS):
            t0 = time.perf_counter()
            DT(*cot)
            ts.append(time.perf_counter() - t0)
    dA = sp.csr_matrix(dA)
    gA = np.asarray(dA[c.A.row, c.A.col]).ravel()
    g = np.concatenate([np.zeros(c.P_up.nnz), gA, np.asarray(dq), np.asarray(db)])
    return g, float(np.median(ts)), (x, y, s)


# ─── Moreau ───────────────────────────────────────────────────────────────


def moreau_vjp(c: Canon, cot):
    """(gradient in our order, median backward time, (x, z, s))."""
    n, m = c.n, c.m
    P_up = sp.csr_matrix((c.P_up.data, (c.P_up.row, c.P_up.col)), shape=(n, n))
    P_full = (P_up + sp.triu(P_up, 1).T).tocsr()
    P_full.sort_indices()
    if P_full.nnz == 0:  # Moreau wants a valid (symmetric) pattern; use the diagonal with zeros
        P_full = sp.csr_matrix((np.zeros(n), (np.arange(n), np.arange(n))), shape=(n, n))
    A = sp.csr_matrix(c.A)
    A.sort_indices()
    sc = c.scs_cones
    cones = moreau.Cones(num_zero_cones=sc.get("z", 0), num_nonneg_cones=sc.get("l", 0),
                         so_cone_dims=list(sc.get("q", [])), num_exp_cones=sc.get("ep", 0),
                         power_alphas=list(sc.get("p", [])))
    ipm = moreau.IPMSettings(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, diff_method="exact")
    settings = moreau.Settings(device=MOREAU_DEVICE, enable_grad=True, ipm_settings=ipm, max_iter=400)
    solver = moreau.jax.Solver(n=n, m=m, P_row_offsets=P_full.indptr, P_col_indices=P_full.indices,
                               A_row_offsets=A.indptr, A_col_indices=A.indices, cones=cones, settings=settings)

    def f(Pv, Av, q, b):
        sol = solver.solve(Pv, Av, q, b)
        return sol.x, sol.z, sol.s

    data = tuple(jnp.asarray(v) for v in (P_full.data, A.data, c.q, c.b))
    ct = tuple(jnp.asarray(v) for v in cot)
    (x, z, s), vjp_fn = jax.vjp(f, *data)
    gP, gA, gq, gb = (np.asarray(v) for v in jax.block_until_ready(vjp_fn(ct)))
    ts = []
    for _ in range(REPS):
        _, vjp_fn = jax.vjp(f, *data)  # forward (untimed); the backward reuses its state
        t0 = time.perf_counter()
        jax.block_until_ready(vjp_fn(ct))
        ts.append(time.perf_counter() - t0)
    # Full-symmetric dP -> gradient w.r.t. the stored upper values (P_ij = P_ji = u_ij).
    Gp = sp.csr_matrix((gP, P_full.indices, P_full.indptr), shape=(n, n))
    up = c.P_up
    gP_up = np.asarray(Gp[up.row, up.col]).ravel() + np.where(
        up.row != up.col, np.asarray(Gp[up.col, up.row]).ravel(), 0.0)
    GA = sp.csr_matrix((gA, A.indices, A.indptr), shape=(m, n))
    g = np.concatenate([gP_up, np.asarray(GA[c.A.row, c.A.col]).ravel(), gq, gb])
    return g, float(np.median(ts)), tuple(np.asarray(v) for v in (x, z, s))


# ─── run ──────────────────────────────────────────────────────────────────


def _rel(a, b, mask=None):
    a, b = np.asarray(a), np.asarray(b)
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _fwd_err(point, ref):
    return max(_rel(p, r) for p, r in zip(point, ref, strict=True))


def run_case(family: str, size: str) -> dict:
    c = Canon(make_problem(family, SIZES[family][size]))
    status, *point = c.solve_clarabel()
    rng = np.random.default_rng(1)
    cot = (rng.standard_normal(c.n), rng.standard_normal(c.m), rng.standard_normal(c.m))
    dense_ok = c.N <= DENSE_MAX_N
    out = dict(family=family, size=size, n=c.n, m=c.m, N=c.N, clarabel_status=status, methods={},
               diffqcp_qcp=QCP_KIND, diffqcp_device=str(jax.devices()[0]), moreau_device=MOREAU_DEVICE)
    print(f"{family}/{size}: N={c.N} clarabel {status}", flush=True)
    g_ref = reference_vjp(c, point, cot) if dense_ok else None
    if dense_ok:
        with jax.default_device(CPU):
            _, F, _ = build_qcp(c, *point)._form_atoms()
            Fp_dense = materialise_gauge_operator(gauge_fixed(F))
        sv = np.linalg.svd(np.asarray(Fp_dense), compute_uv=False)
        out["cond_Fp"] = float(sv[0] / sv[-1])
        print(f"   cond(F') = {out['cond_Fp']:.1e}", flush=True)

    def record(name, g, t, own_point=None, mask=None):
        entry = dict(time=t)
        if own_point is not None:
            entry["forward_err"] = _fwd_err(own_point, point)
        if dense_ok:
            ref = g_ref if own_point is None else reference_vjp(c, own_point, cot)
            entry["grad_err"] = _rel(g, ref, mask)
        out["methods"][name] = entry
        print(f"   {name:16s} {1e3 * t:9.2f} ms  err {entry.get('grad_err', float('nan')):.1e}"
              f"  fwd {entry.get('forward_err', 0.0):.1e}", flush=True)

    solvers = {"diffqcp-lsmr": LSMRSolver()}
    if dense_ok:
        solvers |= {"diffqcp-augmented-lu": DenseDirectSolver(), "diffqcp-reduced-lu": ReducedDirectSolver()}
    for name, solver in solvers.items():
        try:
            g, t = diffqcp_vjp(c, point, cot, solver)
            record(name, g, t)
        except Exception as err:
            out["methods"][name] = dict(error=f"{type(err).__name__}: {str(err).splitlines()[0][:200]}")
            print(f"   {name} failed: {out['methods'][name]['error']}", flush=True)
    if dense_ok and _nvmath_available():
        try:
            g, t = diffqcp_nvmath_vjp(c, point, cot)
            record("diffqcp-nvmath-direct", g, t)
        except Exception as err:
            out["methods"]["diffqcp-nvmath-direct"] = dict(error=f"{type(err).__name__}: {str(err)[:200]}")
            print(f"   diffqcp-nvmath-direct failed: {err}", flush=True)
    if family in P_ZERO_FAMILIES:
        for mode in ("lsqr", "dense") if dense_ok else ("lsqr",):
            try:
                g, t, own = diffcp_vjp(c, cot, mode)
                mask = np.r_[np.zeros(c.P_up.nnz, bool), np.ones(c.A.nnz + c.n + c.m, bool)]
                record(f"diffcp-{mode}", g, t, own, mask)
            except Exception as err:
                out["methods"][f"diffcp-{mode}"] = dict(error=f"{type(err).__name__}: {str(err)[:200]}")
                print(f"   diffcp-{mode} failed: {err}", flush=True)
    try:
        g, t, own = moreau_vjp(c, cot)
        record("moreau", g, t, own)
    except Exception as err:
        out["methods"]["moreau"] = dict(error=f"{type(err).__name__}: {str(err)[:200]}")
        print(f"   moreau failed: {err}", flush=True)
    return out


# ─── conditioning sweep ───────────────────────────────────────────────────


def _hp_reference(c: Canon, point, cot, dps: int = 50) -> tuple[np.ndarray, dict]:
    """VJP with w from a 50-digit minimum-norm solve of the float64 F'^T w = -dz[:N-1]."""
    import mpmath

    with jax.default_device(CPU):
        return _hp_reference_cpu(c, point, cot, dps, mpmath)


def _hp_reference_cpu(c: Canon, point, cot, dps, mpmath) -> tuple[np.ndarray, dict]:
    qcp = build_qcp(c, *point)
    _, F, _ = qcp._form_atoms()
    Fp = np.asarray(materialise_gauge_operator(gauge_fixed(F)))

    class _HP(AbstractDerivativeSolver):
        def solve(self, Fg, rhs):
            raise NotImplementedError

        def solve_transpose(self, Fg, rhs):
            def hp(g):
                with mpmath.workdps(dps):
                    M = mpmath.matrix(Fp.tolist())
                    t = mpmath.lu_solve(M.T * M, mpmath.matrix(np.asarray(g).tolist()))
                    return np.array((M * t).tolist(), dtype=float).ravel()
            return jax.pure_callback(hp, jax.ShapeDtypeStruct((Fp.shape[0],), rhs.dtype), rhs)

    g = np.asarray(_vjp(qcp, *map(jnp.asarray, cot), _HP()))
    sv = np.linalg.svd(Fp, compute_uv=False)
    sv_bar = np.linalg.svd(Fp[:-1, :], compute_uv=False)
    K = np.block([[np.eye(Fp.shape[0]), Fp], [Fp.T, np.zeros((Fp.shape[1], Fp.shape[1]))]])
    cond = dict(cond_Fp=float(sv[0] / sv[-1]), cond_Fbar=float(sv_bar[0] / sv_bar[-1]),
                cond_K=float(np.linalg.cond(K)))
    return g, cond


def conditioning_sweep() -> list[dict]:
    """An SOCP made progressively worse-scaled by x = D x', D = diag(10^(t * linspace(0, 1, n)))."""
    results = []
    base = make_problem("socp", 12)
    for t in (0, 1, 2, 3, 4, 5, 6):
        n = Canon(base).n
        c = Canon(base, column_scale=10.0 ** (t * np.linspace(0, 1, n)))
        status, *point = c.solve_clarabel()
        rng = np.random.default_rng(1)
        cot = (rng.standard_normal(c.n), rng.standard_normal(c.m), rng.standard_normal(c.m))
        g_hp, cond = _hp_reference(c, point, cot)
        row = dict(t=t, N=c.N, clarabel_status=status, **cond, methods={})
        for name, solver in (("diffqcp-lsmr", LSMRSolver()), ("diffqcp-augmented-lu", DenseDirectSolver()),
                             ("diffqcp-reduced-lu", ReducedDirectSolver()), ("svd (dense lstsq)", SVDSolver())):
            try:
                g, _ = diffqcp_vjp(c, point, cot, solver)
                row["methods"][name] = dict(grad_err=_rel(g, g_hp))
            except Exception as err:  # e.g. lineax raising on a non-finite LSMR result
                row["methods"][name] = dict(error=f"{type(err).__name__}: {str(err).splitlines()[0][:200]}")
        mask = np.r_[np.zeros(c.P_up.nnz, bool), np.ones(c.A.nnz + c.n + c.m, bool)]
        for name, fn in (("diffcp-lsqr", lambda: diffcp_vjp(c, cot, "lsqr")),
                         ("diffcp-dense", lambda: diffcp_vjp(c, cot, "dense")),
                         ("moreau", lambda: moreau_vjp(c, cot))):
            try:
                g, _, own = fn()
                g_own, _ = _hp_reference(c, own, cot)
                row["methods"][name] = dict(grad_err=_rel(g, g_own, mask if "diffcp" in name else None),
                                            forward_err=_fwd_err(own, point))
            except Exception as err:
                row["methods"][name] = dict(error=f"{type(err).__name__}: {str(err)[:200]}")
        print(f"t={t} N={c.N} {status} cond F'={cond['cond_Fp']:.1e} Fbar={cond['cond_Fbar']:.1e} "
              f"K={cond['cond_K']:.1e} | " + "  ".join(
                  f"{k}={v.get('grad_err', float('nan')):.1e}" for k, v in row["methods"].items()), flush=True)
        results.append(row)
    return results


# ─── report ───────────────────────────────────────────────────────────────

STYLE = {  # name -> (label, color, marker)
    "diffqcp-lsmr": ("diffqcp LSMR (default)", "#2a78d6", "o"),
    "diffqcp-augmented-lu": ("diffqcp augmented LU (K)", "#8a5cd1", "s"),
    "diffqcp-reduced-lu": ("diffqcp reduced LU (F-bar)", "#0f9fb8", "D"),
    "diffqcp-nvmath-direct": ("diffqcp nvmath-direct (cuDSS, K)", "#b8890f", "*"),
    "diffcp-lsqr": ("diffcp lsqr", "#eb6834", "^"),
    "diffcp-dense": ("diffcp dense", "#c2410c", "v"),
    "moreau": ("Moreau (exact)", "#1f9e74", "P"),
    "svd (dense lstsq)": ("dense SVD lstsq", "#7a7873", "x"),
}


def report(run_path: str, cond_path: str, out_png: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(run_path) as f:
        runs = json.load(f)
    with open(cond_path) as f:
        sweep = json.load(f)
    ink, ink2, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(3, len(FAMILIES), figsize=(4 * len(FAMILIES), 11))
    fig.patch.set_facecolor(bg)
    for j, fam in enumerate(FAMILIES):
        rows = sorted((r for r in runs if r["family"] == fam), key=lambda r: r["N"])
        for i, key in enumerate(("time", "grad_err")):
            ax = axes[i, j]
            for name, (label, color, marker) in STYLE.items():
                pts = [(r["N"], r["methods"][name][key]) for r in rows
                       if name in r["methods"] and key in r["methods"][name]]
                if pts:
                    ax.loglog(*zip(*pts, strict=True), marker=marker, color=color, lw=1.6, ms=5, label=label)
            ax.set_title(f"{fam}: {'VJP time (s)' if key == 'time' else 'gradient rel. error'}",
                         fontsize=10, color=ink, loc="left")
            ax.set_xlabel("N = n + m + 1", color=ink2, fontsize=9)
    for ax in axes[:2].ravel():
        ax.set_facecolor(bg)
        ax.grid(True, which="major", color=grid, lw=0.8)
        ax.tick_params(colors=ink2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    gs = axes[2, 0].get_gridspec()
    for ax in axes[2]:
        ax.remove()
    ax = fig.add_subplot(gs[2, :3])
    ax.set_facecolor(bg)
    for name, (label, color, marker) in STYLE.items():
        pts = [(r["cond_Fp"], r["methods"][name]["grad_err"]) for r in sweep
               if name in r["methods"] and "grad_err" in r["methods"][name]]
        if pts:
            ax.loglog(*zip(*pts, strict=True), marker=marker, color=color, lw=1.6, ms=5, label=label)
    ax.set_title("Conditioning sweep (worse-scaled SOCP): error vs. 50-digit reference",
                 fontsize=10, color=ink, loc="left")
    ax.set_xlabel("cond(F')", color=ink2, fontsize=9)
    ax.set_ylabel("gradient rel. error", color=ink2, fontsize=9)
    ax.grid(True, which="major", color=grid, lw=0.8)
    ax2 = fig.add_subplot(gs[2, 3:])
    ax2.set_facecolor(bg)
    cps = [r["cond_Fp"] for r in sweep]
    ax2.loglog(cps, [r["cond_Fbar"] for r in sweep], marker="D", color="#0f9fb8", label="cond(F-bar)")
    ax2.loglog(cps, [r["cond_K"] for r in sweep], marker="s", color="#8a5cd1", label="cond(K)")
    ax2.loglog(cps, cps, ls="--", color=ink2, lw=1, label="cond(F')")
    ax2.set_title("Conditioning of the factored systems", fontsize=10, color=ink, loc="left")
    ax2.set_xlabel("cond(F')", color=ink2, fontsize=9)
    ax2.grid(True, which="major", color=grid, lw=0.8)
    ax2.legend(frameon=False, fontsize=8.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for a in [*axes[0, 1:], ax]:
        for h, lab in zip(*a.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.985))
    where = f"diffqcp: {runs[0].get('diffqcp_qcp', 'host')} on {runs[0].get('diffqcp_device', 'cpu')}, " \
            f"Moreau on {runs[0].get('moreau_device', 'cpu')}, diffcp on cpu"
    fig.suptitle(f"QCP derivative (VJP): accuracy and speed, float64 ({where})", fontsize=12, color=ink, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=140, facecolor=bg)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        families = sys.argv[3].split(",") if len(sys.argv) > 3 else FAMILIES
        sizes = sys.argv[4].split(",") if len(sys.argv) > 4 else ("S", "M", "L", "XL")
        results = [run_case(f, s) for f in families for s in sizes]
        with open(sys.argv[2], "w") as f:
            json.dump(results, f, indent=1)
    elif cmd == "conditioning":
        with open(sys.argv[2], "w") as f:
            json.dump(conditioning_sweep(), f, indent=1)
    elif cmd == "report":
        report(sys.argv[2], sys.argv[3], sys.argv[4])
