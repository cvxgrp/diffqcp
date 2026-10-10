"""Learning curves through CVXPYlayers: diffqcp vs. diffcp vs. Moreau.

Parameter-level counterpart of `learning_curves.py`. Each problem is a DPP
CVXPY problem with parameters θ. Starting from θ₀, gradient descent on

    loss(θ) = 0.5 ||x(θ) - x(θ*)||^2

(x = the problem's CVXPY variable, as returned by the layer) tries to recover
the target θ*. The gradient of x(θ) comes from one of three backends:

- `diffqcp`: Clarabel solve (CPU) + `diffqcp.make_differentiable` (HostQCP,
  default LSMR solver). CVXPYlayers' JAX layer has no working path for a
  custom canonical-matrix solver (`_solve_with_custom_vjp` reads
  `ctx.solver_ctx`, which is `None` for a `SolverInterface`), so this backend
  reuses the canonicalization of a `solver="MOREAU"` layer (same CSR layout,
  same cones) and swaps the solve + derivative. This mirrors what
  `CvxpyLayer._solve_moreau` does.
- `diffcp`: `CvxpyLayer(..., solver="DIFFCP")` (SCS canonicalization, SOC
  epigraph for quadratics, SCS solve, diffcp derivative).
- `moreau`: `CvxpyLayer(..., solver="MOREAU")` on CPU (Moreau's own IPM and
  its native JAX custom_vjp).

cvxpylayers 1.2.0 + cvxpy 1.9: cvxpy's MOREAU interface moves constraints on
bare variables (`x >= 0`, `PowCone3D(x[0], x[1], x[2], a)`, exp cones, ...)
into Moreau "direct cones" (`data["dir_cones"]`), but cvxpylayers' MOREAU
context builds its cones from `dims` only, so those constraints are silently
dropped and the layer solves a different problem. Layers built on the MOREAU
canonicalization are therefore built with direct-cone extraction disabled
(`_slack_cones_only`); `check_forward` verifies every layer against a plain
CVXPY solve.

Forward solves are run tight for every backend (Clarabel and Moreau defaults,
SCS at eps 1e-9), so differences come from the derivatives, not the solves.

    python experiments/layer_learning_curves.py run <problem> <backend> <out.npz>
    python experiments/layer_learning_curves.py grads <problem>   # forward + gradient agreement at θ₀
    python experiments/layer_learning_curves.py plot <out.png> <runs dir>

Problems: `pow` (learn the point projected onto a product of power cones),
`logistic` (learn the design matrix of a group-lasso logistic regression;
exp + SOC cones), `ls_eq` (learn A, b of a simplex-constrained least squares).
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", os.environ.get("DIFFQCP_PLATFORM", "cpu"))

import clarabel  # noqa: E402
import cvxpy as cvx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from cvxpy.reductions.solvers.conic_solvers.moreau_conif import MOREAU  # noqa: E402
from cvxpy.reductions.solvers.conic_solvers.scs_conif import dims_to_solver_dict  # noqa: E402
from cvxpylayers.jax import CvxpyLayer  # noqa: E402
from cvxpylayers.jax.cvxpylayer import _flatten_and_batch_params, _recover_results  # noqa: E402
from jax.experimental.sparse import BCSR  # noqa: E402

from diffqcp import QCPStructureGPU, make_differentiable  # noqa: E402

BACKENDS = ("diffqcp", "diffcp", "moreau")
# name -> (number of iterations, step size); identical for every backend.
SETTINGS = {
    "pow": (150, 0.5),
    "logistic": (150, 2.0),
    "ls_eq": (150, 0.5),
}
SCS_TIGHT = {"eps_abs": 1e-9, "eps_rel": 1e-9, "max_iters": 100_000}


# ─── problems ─────────────────────────────────────────────────────────────


def make_problem(name: str):
    """(problem, parameters, variable, θ*, θ₀)."""
    if name == "pow":
        alphas = np.random.default_rng(100).uniform(0.15, 0.85, size=11)
        n = 3 * len(alphas)
        v, c = cvx.Variable(n), cvx.Parameter(n)
        cons = [cvx.PowCone3D(v[3 * i], v[3 * i + 1], v[3 * i + 2], float(a)) for i, a in enumerate(alphas)]
        prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(v - c)), cons)
        target = [np.random.default_rng(0).standard_normal(n)]
        init = [np.random.default_rng(1).standard_normal(n)]
        return prob, [c], v, target, init
    if name == "logistic":
        m, p = 20, 20
        beta, X = cvx.Variable(p), cvx.Parameter((m, p))
        rng = np.random.default_rng(0)
        X_true = rng.standard_normal((m, p))
        beta_true = np.zeros(p)
        beta_true[:2] = 1.0
        y = np.round(1 / (1 + np.exp(-(X_true @ beta_true + 0.5 * rng.standard_normal(m)))))
        loss = -cvx.sum(cvx.multiply(y, X @ beta) - cvx.logistic(X @ beta))
        reg = 0.1 * cvx.sum(cvx.norm(cvx.reshape(beta, (2, 10), order="C"), axis=1))
        prob = cvx.Problem(cvx.Minimize(loss + reg))
        X0 = X_true + 0.5 * np.random.default_rng(1).standard_normal((m, p))
        return prob, [X], beta, [X_true], [X0]
    if name == "ls_eq":
        m, n = 20, 10
        x, A, b = cvx.Variable(n), cvx.Parameter((m, n)), cvx.Parameter(m)
        prob = cvx.Problem(cvx.Minimize(cvx.sum_squares(A @ x - b)), [x >= 0, cvx.sum(x) == 1.0])
        r0, r1 = np.random.default_rng(0), np.random.default_rng(1)
        return prob, [A, b], x, [r0.standard_normal((m, n)), r0.standard_normal(m)], \
            [r1.standard_normal((m, n)), r1.standard_normal(m)]
    raise ValueError(name)


# ─── diffqcp as a CVXPYlayers backend ─────────────────────────────────────


class DiffqcpLayer:
    """`CvxpyLayer`-like callable: Clarabel forward, diffqcp backward.

    Reuses the canonicalization of a `solver="MOREAU"` `CvxpyLayer` (CSR
    sparsity of P and A, the row map of b, the cones) and replaces Moreau's
    solve with a Clarabel `pure_callback` wrapped by `make_differentiable`.
    """

    def __init__(self, problem, parameters, variables):
        self.layer = CvxpyLayer(problem, parameters=parameters, variables=variables, solver="MOREAU")
        ctx = self.layer.ctx.solver_ctx
        n, m = ctx.P_shape[0], ctx.A_shape[0]
        self.n, self.m, self.nnz_A, self.b_idx = n, m, ctx.nnz_A, jnp.asarray(ctx.b_idx)
        self.scs_cones = dims_to_solver_dict(ctx.dims)
        assert not self.scs_cones.get("s"), "PSD cones would need the Clarabel->SCS row permutation"
        self.clarabel_cones = _clarabel_cones(self.scs_cones)

        # cvxpylayers stores the upper triangle of P (Clarabel convention);
        # diffqcp's CSR structure wants the full symmetric matrix.
        P_up = sp.csr_matrix((np.arange(1, ctx.P_col_indices.size + 1, dtype=float), ctx.P_col_indices,
                              ctx.P_row_offsets), shape=(n, n))
        assert sp.tril(P_up, -1).nnz == 0, "expected an upper-triangular P"
        P_full = (P_up + sp.triu(P_up, 1).T).tocsr()
        P_full.sort_indices()
        self.P_up = P_up.tocoo()
        self.P_gather = jnp.asarray(P_full.data.astype(int) - 1)  # full CSR value k <- upper value P_gather[k]
        A = sp.csr_matrix((np.ones(ctx.A_col_indices.size), ctx.A_col_indices, ctx.A_row_offsets), shape=(m, n))
        to_bcsr = lambda M: BCSR((jnp.asarray(M.data), jnp.asarray(M.indices), jnp.asarray(M.indptr)),  # noqa: E731
                                 shape=M.shape)
        self.A_csr = A
        self.structure = QCPStructureGPU(to_bcsr(P_full), to_bcsr(A), self.scs_cones)
        self.solve = make_differentiable(self._solve, self.structure)

    def _solve_np(self, P_full_values, A_values, q, b):
        n, m = self.n, self.m
        P = sp.csc_matrix((np.asarray(P_full_values), self.structure.P_csr_indices, self.structure.P_csr_indptr),
                          shape=(n, n))
        A = sp.csc_matrix(sp.csr_matrix((np.asarray(A_values), self.A_csr.indices, self.A_csr.indptr), shape=(m, n)))
        settings = clarabel.DefaultSettings()
        settings.verbose = False
        soln = clarabel.DefaultSolver(sp.triu(P, format="csc"), np.asarray(q), A, np.asarray(b),
                                      self.clarabel_cones, settings).solve()
        if str(soln.status) not in ("Solved", "AlmostSolved"):
            raise RuntimeError(f"Clarabel status {soln.status}")
        return np.array(soln.x), np.array(soln.z), np.array(soln.s)

    def _solve(self, data):
        shapes = tuple(jax.ShapeDtypeStruct((k,), jnp.float64) for k in (self.n, self.m, self.m))
        return jax.pure_callback(self._solve_np, shapes, *data)

    def __call__(self, *params):
        # Same parameter -> canonical data map as `CvxpyLayer._solve_moreau` (unbatched).
        layer = self.layer
        batch = layer.ctx.validate_params(list(params))
        assert batch == (), "unbatched only"
        p = _flatten_and_batch_params(list(params), layer.ctx, batch)
        P_eval = layer.P @ p if layer.P is not None else jnp.zeros(0)
        q_eval, A_eval = layer.q @ p, layer.A @ p
        P_values = P_eval[self.P_gather] if self.P_gather.size else P_eval
        A_values = -A_eval[: self.nnz_A]
        b = jnp.zeros(self.m).at[self.b_idx].set(A_eval[self.nnz_A:])
        x, y, _ = self.solve((P_values, A_values, q_eval[:-1], b))
        return _recover_results(x[None], y[None], layer.ctx, batch)


def _clarabel_cones(scs: dict) -> list:
    cones = []
    if scs.get("z"):
        cones.append(clarabel.ZeroConeT(scs["z"]))
    if scs.get("l"):
        cones.append(clarabel.NonnegativeConeT(scs["l"]))
    cones += [clarabel.SecondOrderConeT(d) for d in scs.get("q", [])]
    cones += [clarabel.ExponentialConeT() for _ in range(scs.get("ep", 0))]
    assert not scs.get("ed"), "dual exponential cones not handled"
    cones += [clarabel.PowerConeT(a) for a in scs.get("p", [])]
    return cones


@contextlib.contextmanager
def _slack_cones_only():
    """Canonicalize for MOREAU without extracting direct (variable-side) cones."""
    kinds = MOREAU.DIR_CONE_KINDS
    MOREAU.DIR_CONE_KINDS = frozenset()
    try:
        yield
    finally:
        MOREAU.DIR_CONE_KINDS = kinds


def make_layer(backend: str, problem, parameters, variable):
    if backend == "diffqcp":
        with _slack_cones_only():
            return DiffqcpLayer(problem, parameters, [variable])
    if backend == "diffcp":
        layer = CvxpyLayer(problem, parameters=parameters, variables=[variable], solver="DIFFCP")
        return lambda *ps: layer(*ps, solver_args=SCS_TIGHT)
    if backend == "moreau":
        with _slack_cones_only():
            return CvxpyLayer(problem, parameters=parameters, variables=[variable], solver="MOREAU")
    raise ValueError(backend)


def check_forward(problem_name: str) -> dict:
    """max |x_layer(θ₀) - x_cvxpy(θ₀)| per backend (the diffcp layer only solves inside a VJP)."""
    problem, parameters, variable, _, init = make_problem(problem_name)
    x_ref = np.asarray(target_solution(problem, parameters, variable, init))
    params = [jnp.asarray(v) for v in init]
    out = {}
    for backend in BACKENDS:
        layer = make_layer(backend, problem, parameters, variable)
        (x,), _ = jax.vjp(lambda *ps: layer(*ps), *params)  # noqa: B023
        out[backend] = float(np.max(np.abs(np.asarray(x) - x_ref)))
    return out


# ─── learning loop ────────────────────────────────────────────────────────


def _loss_fn(layer, x_target):
    def loss(params):
        (x,) = layer(*params)
        return 0.5 * jnp.sum((x - x_target) ** 2)

    return loss


def target_solution(problem, parameters, variable, target):
    for p, v in zip(parameters, target, strict=True):
        p.value = v
    problem.solve(solver=cvx.CLARABEL)
    assert problem.status == "optimal", problem.status
    return jnp.asarray(variable.value)


def run(problem_name: str, backend: str, num_iter: int | None = None, step: float | None = None) -> dict:
    default_iter, default_step = SETTINGS[problem_name]
    num_iter = default_iter if num_iter is None else num_iter
    step = default_step if step is None else step
    problem, parameters, variable, target, init = make_problem(problem_name)
    x_target = target_solution(problem, parameters, variable, target)
    layer = make_layer(backend, problem, parameters, variable)
    value_and_grad = jax.value_and_grad(_loss_fn(layer, x_target))

    params = [jnp.asarray(v) for v in init]
    losses, grad_norms, param_errs, times, status = [], [], [], [], "ok"
    for _ in range(num_iter):
        t0 = time.perf_counter()
        try:
            loss, grads = value_and_grad(params)
            loss = float(loss)
        except Exception as err:  # solver or derivative failure
            status = f"{type(err).__name__}: {str(err)[:200]}"
            break
        times.append(time.perf_counter() - t0)
        g = np.concatenate([np.ravel(gi) for gi in grads])
        if not (np.isfinite(loss) and np.all(np.isfinite(g))):
            status = "non-finite loss or gradient"
            break
        losses.append(loss)
        grad_norms.append(float(np.linalg.norm(g)))
        param_errs.append(float(np.sqrt(sum(np.sum((np.asarray(p) - t) ** 2) for p, t in zip(params, target)))))
        params = [p - step * gi for p, gi in zip(params, grads, strict=True)]
    return dict(losses=np.array(losses), grad_norms=np.array(grad_norms), param_errs=np.array(param_errs),
                times=np.array(times), status=status, num_iter=num_iter, step=step)


def gradient_agreement(problem_name: str) -> dict:
    """Gradient of the loss at θ₀ from each backend; pairwise relative differences and cosines."""
    problem, parameters, variable, target, init = make_problem(problem_name)
    x_target = target_solution(problem, parameters, variable, target)
    params = [jnp.asarray(v) for v in init]
    grads = {}
    for backend in BACKENDS:
        layer = make_layer(backend, problem, parameters, variable)
        g = jax.grad(_loss_fn(layer, x_target))(params)
        grads[backend] = np.concatenate([np.ravel(gi) for gi in g])
    out = {}
    for i, a in enumerate(BACKENDS):
        for b in BACKENDS[i + 1:]:
            ga, gb = grads[a], grads[b]
            out[f"{a} vs {b}"] = dict(
                rel_diff=float(np.linalg.norm(ga - gb) / max(np.linalg.norm(gb), 1e-300)),
                cosine=float(ga @ gb / (np.linalg.norm(ga) * np.linalg.norm(gb))),
            )
    return out


# ─── plotting ─────────────────────────────────────────────────────────────


def plot(out_path: str, runs_dir: str) -> None:
    import matplotlib
    import matplotlib.ticker

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"diffqcp": "#2a78d6", "diffcp": "#eb6834", "moreau": "#1f9e74"}
    labels = {"diffqcp": "diffqcp (Clarabel)", "diffcp": "diffcp (SCS)", "moreau": "Moreau"}
    ink, ink2, grid, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    titles = {
        "pow": "Power cones: learn the projected point",
        "logistic": "Exp + SOC cones: learn the logistic design matrix",
        "ls_eq": "Simplex least squares: learn A, b",
    }
    fig, axes = plt.subplots(len(titles), 2, figsize=(11, 10))
    fig.patch.set_facecolor(bg)
    for i, problem in enumerate(titles):
        for j, (key, ylabel) in enumerate((("losses", "loss  ½‖x(θ) − x(θ*)‖²"), ("param_errs", "‖θ − θ*‖"))):
            ax = axes[i, j]
            ax.set_facecolor(bg)
            for backend in BACKENDS:
                path = os.path.join(runs_dir, f"{problem}_{backend}.npz")
                if not os.path.exists(path):
                    continue
                r = np.load(path)
                ys = r[key]
                ms = 1e3 * float(np.median(r["times"])) if r["times"].size else float("nan")
                ax.plot(np.arange(len(ys)), ys, color=colors[backend], lw=2, solid_capstyle="round",
                        label=f"{labels[backend]} · {ms:.0f} ms/iter")
                if len(ys) < int(r["num_iter"]) and len(ys):
                    ax.plot(len(ys) - 1, ys[-1], marker="X", ms=10, color=colors[backend], mec=bg, mew=1.5)
            if key == "losses":
                ax.set_yscale("log")
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_title(titles[problem] if j == 0 else "", fontsize=10, color=ink, loc="left")
            ax.grid(True, which="major", color=grid, lw=0.8)
            ax.tick_params(colors=ink2, labelsize=8.5)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(grid)
            ax.set_ylabel(ylabel, color=ink2, fontsize=9)
            if j == 0:
                ax.legend(frameon=False, fontsize=8.5, labelcolor=ink)
            if i == len(titles) - 1:
                ax.set_xlabel("gradient-descent iteration", color=ink2, fontsize=9)
    fig.suptitle("Learning parameters through a CVXPYlayers layer: diffqcp vs. diffcp vs. Moreau (CPU)",
                 fontsize=12, color=ink, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        _, _, problem, backend, out = sys.argv
        res = run(problem, backend)
        np.savez(out, **res)
        L = res["losses"]
        print(f"{problem} {backend}: iters={len(L)} first={L[0] if len(L) else float('nan'):.4e} "
              f"last={L[-1] if len(L) else float('nan'):.4e} status={res['status']} "
              f"median {1e3 * np.median(res['times']):.1f} ms/iter")
    elif cmd == "grads":
        name = sys.argv[2]
        print(json.dumps({name: dict(forward_err=check_forward(name), gradients=gradient_agreement(name))}, indent=2))
    elif cmd == "plot":
        plot(sys.argv[2], sys.argv[3])
