"""Learning-curve experiments: recover problem data by gradient descent through a solve.

Mirrors `cpu_experiment.py`, `cpu_baseline.py`, `direct_solve_experiment.py`
and `heterogeneous_experiment2.py`: solve a "target" problem, start from a
different "initial" problem with the same cone structure, and run gradient
descent on

    loss = 0.5 ||x - x*||^2 + 0.5 ||y - y*||^2 + 0.5 ||s - s*||^2

over the data (P, A, q, b), re-solving with Clarabel and applying `qcp.vjp`
each iteration.

It only uses the `HostQCP` / `DeviceQCP` / `QCPStructure*` API, so it runs
against any version of diffqcp on `PYTHONPATH`; that is how the before/after
comparison in `results/` was produced:

    PYTHONPATH=<checkout> python experiments/learning_curves.py run <problem> <variant> <out.npz>
    python experiments/learning_curves.py plot <out.png> <label>=<run.npz> ...
    python experiments/learning_curves.py compare <out.png> <dir with {old,new}_<problem>_<variant>.npz>

Problems:
- `pow`: projection onto a product of 3D power cones (`cpu_experiment.py`).
- `logistic`: group-lasso logistic regression, exp + SOC cones
  (`cpu_baseline.py`, `heterogeneous_experiment2.py`).
- `ls_eq`: simplex-constrained least squares (`direct_solve_experiment.py`).

Variants: `host` (HostQCP, default solve method) and `device` (DeviceQCP on
BCSR data, default solve method; that is the path cvxpylayers' CuClarabel
interface uses).
"""
from __future__ import annotations

import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

import clarabel
import cvxpy as cvx
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import dims_to_solver_cones
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict
from jax.experimental.sparse import BCOO, BCSR

from diffqcp import DeviceQCP, HostQCP, QCPStructureCPU, QCPStructureGPU

# name -> (number of iterations, step size). Same for every diffqcp version.
SETTINGS = {
    "pow": (300, 1e-2),
    "logistic": (300, 1e-6),
    "ls_eq": (300, 3e-3),
}


# ─── problems ─────────────────────────────────────────────────────────────


def _pow_problem(seed: int, alphas: np.ndarray) -> cvx.Problem:
    rng = np.random.default_rng(seed)
    n = 3 * len(alphas)
    x_target = rng.standard_normal(n)
    var = cvx.Variable(n)
    cons = [cvx.PowCone3D(var[3 * i], var[3 * i + 1], var[3 * i + 2], float(a)) for i, a in enumerate(alphas)]
    return cvx.Problem(cvx.Minimize(cvx.sum_squares(var - x_target)), cons)


def _logistic_problem(seed: int, m: int = 20, n: int = 2) -> cvx.Problem:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((m, 10 * n))
    beta_true = np.zeros(10 * n)
    beta_true[: max(1, 10 * n // 10)] = 1.0
    y = np.round(1 / (1 + np.exp(-(X @ beta_true + 0.5 * rng.standard_normal(m)))))
    beta = cvx.Variable(10 * n)
    loss = -cvx.sum(cvx.multiply(y, X @ beta) - cvx.logistic(X @ beta))
    reg = 0.1 * cvx.sum(cvx.norm(beta.reshape((-1, 10), "C"), axis=1))
    return cvx.Problem(cvx.Minimize(loss + reg))


def _ls_eq_problem(seed: int, m: int = 20, n: int = 10) -> cvx.Problem:
    rng = np.random.default_rng(seed)
    x = cvx.Variable(n)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    return cvx.Problem(cvx.Minimize(cvx.sum_squares(A @ x - b)), [x >= 0, cvx.sum(x) == 1.0])


def make_problems(name: str) -> tuple[cvx.Problem, cvx.Problem]:
    """(target, initial): same cone structure, different data."""
    if name == "pow":
        alphas = np.random.default_rng(100).uniform(0.15, 0.85, size=11)
        return _pow_problem(0, alphas), _pow_problem(1, alphas)
    if name == "logistic":
        return _logistic_problem(0), _logistic_problem(1)
    if name == "ls_eq":
        return _ls_eq_problem(0), _ls_eq_problem(1)
    raise ValueError(name)


# ─── canonical data ───────────────────────────────────────────────────────


class Canon:
    """Clarabel-format data with P (upper triangle) and A in fixed COO order."""

    def __init__(self, problem: cvx.Problem):
        data, _, _ = problem.get_problem_data(cvx.CLARABEL, ignore_dpp=True, solver_opts={"use_quad_obj": True})
        self.q = np.asarray(data["c"], dtype=float)
        self.b = np.asarray(data["b"], dtype=float)
        self.n, self.m = self.q.size, self.b.size
        P = data.get("P")
        P = sp.csr_matrix((self.n, self.n)) if P is None else sp.csr_matrix(P)
        self.P_upper = sp.triu(P).tocsr().tocoo()
        self.P_full = P.tocsr().tocoo()
        self.A = sp.csr_matrix(data["A"]).tocoo()
        self.clarabel_cones = dims_to_solver_cones(data["dims"])
        self.scs_cones = dims_to_solver_dict(data["dims"])
        assert not self.scs_cones.get("s"), "PSD cones would need the Clarabel->SCS row permutation"

    def solve(self, P_upper_vals, A_vals, q, b):
        n, m = self.n, self.m
        P = sp.csc_matrix((P_upper_vals, (self.P_upper.row, self.P_upper.col)), shape=(n, n))
        A = sp.csc_matrix((A_vals, (self.A.row, self.A.col)), shape=(m, n))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        soln = clarabel.DefaultSolver(P, q, A, b, self.clarabel_cones, settings).solve()
        return str(soln.status), np.array(soln.x), np.array(soln.z), np.array(soln.s)


def _bcoo(coo, vals):
    return BCOO((jnp.asarray(vals), jnp.stack([jnp.asarray(coo.row), jnp.asarray(coo.col)], axis=1)), shape=coo.shape)


def _bcsr(coo, vals):
    M = sp.csr_matrix((vals, (coo.row, coo.col)), shape=coo.shape)
    M.sort_indices()
    return BCSR((jnp.asarray(M.data), jnp.asarray(M.indices), jnp.asarray(M.indptr)), shape=M.shape)


# ─── learning loop ────────────────────────────────────────────────────────


@eqx.filter_jit
def _vjp(qcp, dx, dy, ds):
    return qcp.vjp(dx, dy, ds)


def run(problem: str, variant: str, num_iter: int | None = None, step: float | None = None) -> dict:
    default_iter, default_step = SETTINGS[problem]
    num_iter = default_iter if num_iter is None else num_iter
    step = default_step if step is None else step
    target, initial = make_problems(problem)
    tgt = Canon(target)
    status, tx, ty, ts = tgt.solve(tgt.P_upper.data, tgt.A.data, tgt.q, tgt.b)
    assert status == "Solved", status

    cur = Canon(initial)
    assert (cur.n, cur.m) == (tgt.n, tgt.m)
    # P and A of the initial problem, in fixed COO order; the iteration updates values only.
    P_vals = cur.P_upper.data.copy()
    A_vals = cur.A.data.copy()
    q, b = cur.q.copy(), cur.b.copy()
    # Map each upper-triangle entry to its position(s) in the full symmetric P (DeviceQCP).
    full_index = {(r, c): k for k, (r, c) in enumerate(zip(cur.P_full.row, cur.P_full.col, strict=True))}
    up_to_full = np.array([full_index[(r, c)] for r, c in zip(cur.P_upper.row, cur.P_upper.col, strict=True)], dtype=int)
    lo_to_full = np.array([full_index[(c, r)] for r, c in zip(cur.P_upper.row, cur.P_upper.col, strict=True)], dtype=int)

    if variant == "host":
        P0 = _bcoo(cur.P_upper, P_vals)
        A0 = _bcoo(cur.A, A_vals)
        structure = QCPStructureCPU(P0, A0, cur.scs_cones)
    else:
        P0 = _bcsr(cur.P_full, cur.P_full.data)
        A0 = _bcsr(cur.A, A_vals)
        structure = QCPStructureGPU(P0, A0, cur.scs_cones)

    losses, grad_norms, statuses = [], [], []
    t0 = time.perf_counter()
    for _ in range(num_iter):
        status, x, y, s = cur.solve(P_vals, A_vals, q, b)
        statuses.append(status)
        if status not in ("Solved", "AlmostSolved"):
            break
        loss = 0.5 * (np.sum((x - tx) ** 2) + np.sum((y - ty) ** 2) + np.sum((s - ts) ** 2))
        losses.append(loss)
        args = (jnp.asarray(q), jnp.asarray(b), jnp.asarray(x), jnp.asarray(y), jnp.asarray(s), structure)
        if variant == "host":
            qcp = HostQCP(_bcoo(cur.P_upper, P_vals), _bcoo(cur.A, A_vals), *args)
        else:
            full_vals = np.zeros(cur.P_full.nnz)
            full_vals[up_to_full] = P_vals
            full_vals[lo_to_full] = P_vals
            Pf = sp.csr_matrix((full_vals, (cur.P_full.row, cur.P_full.col)), shape=cur.P_full.shape)
            qcp = DeviceQCP(_bcsr(Pf.tocoo(), Pf.tocoo().data), _bcsr(cur.A, A_vals), *args)
        try:
            dP, dA, dq, db = _vjp(qcp, jnp.asarray(x - tx), jnp.asarray(y - ty), jnp.asarray(s - ts))
        except Exception as err:  # e.g. lineax raising on a non-finite solve
            statuses.append(f"vjp failed: {type(err).__name__}")
            break
        dq, db = np.asarray(dq), np.asarray(db)
        if variant == "host":
            gP = np.asarray(dP.data)
            gA = np.asarray(dA.data)
        else:
            # Map the BCSR gradients back to the COO value order. For the full
            # symmetric P the stored value u_ij drives both (i, j) and (j, i).
            gP_full = sp.csr_matrix((np.asarray(dP.data), np.asarray(dP.indices), np.asarray(dP.indptr)), shape=dP.shape)
            gP = np.zeros(cur.P_upper.nnz)
            if cur.P_upper.nnz:
                gP = np.asarray(gP_full[cur.P_upper.row, cur.P_upper.col], dtype=float).ravel()
                off = cur.P_upper.row != cur.P_upper.col
                gP = gP + np.where(off, np.asarray(gP_full[cur.P_upper.col, cur.P_upper.row], dtype=float).ravel(), 0.0)
            gA_csr = sp.csr_matrix((np.asarray(dA.data), np.asarray(dA.indices), np.asarray(dA.indptr)), shape=dA.shape)
            gA = np.asarray(gA_csr[cur.A.row, cur.A.col]).ravel()
        g = np.concatenate([gP, gA, dq, db])
        grad_norms.append(float(np.linalg.norm(g)))
        if not np.all(np.isfinite(g)):
            break
        P_vals = P_vals - step * gP
        A_vals = A_vals - step * gA
        q = q - step * dq
        b = b - step * db
    elapsed = time.perf_counter() - t0
    return dict(losses=np.array(losses), grad_norms=np.array(grad_norms), statuses=np.array(statuses),
                elapsed=elapsed, num_iter=num_iter, step=step)


def plot(out_path: str, runs: list[tuple[str, str]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, path in runs:
        r = np.load(path)
        ax.semilogy(np.arange(len(r["losses"])), r["losses"], label=label)
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_comparison(out_path: str, runs_dir: str) -> None:
    """Grid: one row per problem, one column per variant; before vs after per panel."""
    import os

    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    blue, orange = "#2a78d6", "#eb6834"  # categorical slots 1, 2 (validated pair)
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    titles = {
        "pow": "Power cones (cpu_experiment)",
        "logistic": "Exp + SOC cones (cpu_baseline / heterogeneous2)",
        "ls_eq": "Simplex least squares (direct_solve_experiment)",
    }
    variants = {"host": "HostQCP", "device": "DeviceQCP (cvxpylayers path)"}
    fig, axes = plt.subplots(len(titles), len(variants), figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for i, problem in enumerate(titles):
        for j, variant in enumerate(variants):
            ax = axes[i, j]
            ax.set_facecolor("#fcfcfb")
            for tag, label, color in (("old", "before fixes", orange), ("new", "after fixes", blue)):
                path = os.path.join(runs_dir, f"{tag}_{problem}_{variant}.npz")
                r = np.load(path)
                L = r["losses"]
                ax.semilogy(np.arange(len(L)), L, color=color, lw=2, label=label, solid_capstyle="round")
                stopped = len(L) < int(r["num_iter"])
                if stopped:
                    ax.plot(len(L) - 1, L[-1], marker="X", ms=10, color=color, mec="#fcfcfb", mew=1.5)
                    why = [st for st in r["statuses"].tolist() if st != "Solved"]
                    reason = "solver failed" if why and "vjp" not in why[-1] else "VJP failed (NaN/inf)"
                    ax.annotate(f"stopped: {reason}", (len(L) - 1, L[-1]), xytext=(8, 6),
                                textcoords="offset points", fontsize=8.5, color=ink2)
            old_L = np.load(os.path.join(runs_dir, f"old_{problem}_{variant}.npz"))["losses"]
            new_L = np.load(os.path.join(runs_dir, f"new_{problem}_{variant}.npz"))["losses"]
            k = min(len(old_L), len(new_L))
            if k == len(new_L) and np.max(np.abs(old_L[:k] / new_L[:k] - 1)) < 0.02:
                ax.text(0.98, 0.95, "curves overlap (within 2%)", transform=ax.transAxes,
                        ha="right", va="top", fontsize=8.5, color=ink2)
            both = np.concatenate([old_L, new_L])
            if both.max() / both.min() < 10:  # under a decade: a log axis has no labelled ticks
                ax.set_yscale("linear")
                log_note = ""
            else:
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                log_note = " (log scale)"
            ax.set_title(f"{titles[problem]}\n{variants[variant]}", fontsize=10, color=ink, loc="left")
            ax.grid(True, which="major", color=grid, lw=0.8)
            ax.tick_params(colors=ink2, labelsize=8.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.set_ylabel(f"loss{log_note}", color=ink2, fontsize=9)
            if i == len(titles) - 1:
                ax.set_xlabel("gradient-descent iteration", color=ink2, fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Learning data through the solve: diffqcp before vs. after the gradient fixes",
                 fontsize=12, color=ink, y=0.99)
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False,
               fontsize=10, labelcolor=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        _, _, problem, variant, out = sys.argv
        res = run(problem, variant)
        np.savez(out, **res)
        print(problem, variant, f"iters={len(res['losses'])} first={res['losses'][0]:.4e} "
              f"last={res['losses'][-1]:.4e} statuses={sorted(set(res['statuses'].tolist()))} "
              f"time={res['elapsed']:.1f}s")
    elif sys.argv[1] == "plot":
        plot(sys.argv[2], [tuple(a.split("=", 1)) for a in sys.argv[3:]])
    elif sys.argv[1] == "compare":
        plot_comparison(sys.argv[2], sys.argv[3])
