"""GPU verification for diffqcp. Run via `scripts/gpu_check.sh`, which also runs the test suite.

Checks, all reported as PASS/FAIL lines in a plain-text report:
1. Environment: devices, library versions, `nvidia-smi`.
2. Solver agreement: `DeviceQCP` on the GPU, with every solve method
   ("jax-lsmr", "jax-lu", "nvmath-direct" if installed), against a CPU
   `HostQCP` reference, on a dense-P QP, an SDP and a simplex least-squares
   problem. Compares JVPs, the VJP's dq/db, and the VJP's directional
   derivative along a random data perturbation (which also covers dP and dA).
3. Learning curves: `experiments/learning_curves.py` on the GPU for every
   problem and both `HostQCP` / `DeviceQCP`, final loss vs. the CPU result.

Usage: python scripts/gpu_checks.py [report_path]
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from importlib import metadata

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental.sparse import BCOO, BCSR  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from diffqcp import DeviceQCP, QCPStructureGPU  # noqa: E402
from tests.helpers import scsr_to_bcsr  # noqa: E402
from tests.problems import (  # noqa: E402
    QCPProbData,
    generate_dense_qp,
    generate_feasible_sdp,
    generate_least_squares_eq,
)
from tests.test_qcp_adjoint import _build_host_qcp  # noqa: E402

# Final losses of `learning_curves.py` on CPU (this branch), for comparison.
CPU_FINAL_LOSS = {"pow": 1.0814e00, "ls_eq": 1.3822e02, "logistic": 5.1877e02}
JVP_RTOL = 1e-6
LOSS_RTOL = 1e-2

lines: list[str] = []
failures: list[str] = []


def out(msg: str = "") -> None:
    print(msg, flush=True)
    lines.append(msg)


def check(name: str, ok: bool, detail: str) -> None:
    out(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        failures.append(name)


def rel(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# ─── 1. environment ───────────────────────────────────────────────────────


def environment() -> bool:
    out("== 1. Environment")
    out(f"  python {platform.python_version()} on {platform.platform()}")
    for pkg in ("jax", "jaxlib", "jax-cuda12-plugin", "equinox", "lineax", "cupy-cuda12x", "nvmath-python",
                "clarabel", "scs", "cvxpy"):
        try:
            out(f"  {pkg} {metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            out(f"  {pkg} (not installed)")
    out(f"  jax default backend: {jax.default_backend()}")
    out(f"  jax devices: {jax.devices()}")
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30)
        out(f"  nvidia-smi: {smi.stdout.strip() or smi.stderr.strip()}")
    except Exception as err:
        out(f"  nvidia-smi unavailable: {err}")
    on_gpu = jax.default_backend() == "gpu"
    if os.environ.get("DIFFQCP_GPU_CHECK_DRY_RUN"):  # exercise the script on a CPU-only machine
        out("  DRY RUN: continuing without a GPU")
        return True
    check("JAX sees a GPU", on_gpu, f"default backend is {jax.default_backend()!r}")
    return on_gpu


# ─── 2. solver agreement ──────────────────────────────────────────────────


PROBLEMS = {
    "dense_qp": lambda: generate_dense_qp(n=8, m=12, rng_or_seed=0),
    "sdp": lambda: generate_feasible_sdp(4, 3, rng_or_seed=0),
    "ls_eq": lambda: generate_least_squares_eq(m=20, n=10, rng_or_seed=0),
}


def solver_agreement() -> None:
    out("\n== 2. DeviceQCP on GPU vs. HostQCP on CPU")
    try:
        from nvmath.sparse.advanced import DirectSolver  # noqa: F401

        methods = ["jax-lsmr", "jax-lu", "nvmath-direct"]
    except ImportError:
        out("  nvmath-python not importable: skipping 'nvmath-direct'")
        methods = ["jax-lsmr", "jax-lu"]

    cpu = jax.devices("cpu")[0]
    for name, generator in PROBLEMS.items():
        out(f"  -- {name}")
        try:
            pd = QCPProbData(generator())
            n, m = pd.n, pd.m
            rng = np.random.default_rng(0)
            A_scs, b, y, s = pd.scs_ordered()
            U = pd.Pupper_coo
            Ac = A_scs.tocoo()
            # One data perturbation u (upper-triangular values of dP, values of dA,
            # dq, db) and one solution perturbation v = (dx, dy, ds).
            uP, uA = rng.standard_normal(U.nnz), rng.standard_normal(Ac.nnz)
            uq, ub = rng.standard_normal(n), rng.standard_normal(m)
            vx, vy, vs = rng.standard_normal(n), rng.standard_normal(m), rng.standard_normal(m)
            dP_full = np.zeros((n, n))
            dP_full[U.row, U.col] = uP
            dP_full[U.col, U.row] = uP
            dA_dense = np.zeros((m, n))
            dA_dense[Ac.row, Ac.col] = uA

            # CPU reference.
            with jax.default_device(cpu):
                host = _build_host_qcp(pd)
                idx = lambda M: jnp.stack([jnp.asarray(M.row), jnp.asarray(M.col)], axis=1)  # noqa: E731
                ref_jvp = host.jvp(BCOO((jnp.asarray(uP), idx(U)), shape=(n, n)),
                                   BCOO((jnp.asarray(uA), idx(Ac)), shape=(m, n)),
                                   jnp.asarray(uq), jnp.asarray(ub))
                rP, rA, rq, rb = host.vjp(jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vs))
                ref_dir = float(np.sum(np.asarray(rP.data) * uP) + np.sum(np.asarray(rA.data) * uA)
                                + np.asarray(rq) @ uq + np.asarray(rb) @ ub)
                ref_jvp = [np.asarray(t) for t in ref_jvp]
                rq, rb = np.asarray(rq), np.asarray(rb)

            # GPU DeviceQCP (full symmetric P, CSR).
            P = scsr_to_bcsr(pd.Pcsr)
            A = scsr_to_bcsr(A_scs.tocsr())
            structure = QCPStructureGPU(P, A, pd.scs_cones)
            dev = DeviceQCP(P, A, jnp.asarray(pd.q), jnp.asarray(b), jnp.asarray(pd.x), jnp.asarray(y),
                            jnp.asarray(s), structure)
            P_rows = np.asarray(structure.P_nonzero_rows)
            P_cols = np.asarray(structure.P_nonzero_cols)
            A_rows = np.asarray(structure.A_nonzero_rows)
            A_cols = np.asarray(structure.A_nonzero_cols)
            dP = BCSR((jnp.asarray(dP_full[P_rows, P_cols]), P.indices, P.indptr), shape=P.shape)
            dA = BCSR((jnp.asarray(dA_dense[A_rows, A_cols]), A.indices, A.indptr), shape=A.shape)
            out(f"     n={n} m={m} cones={ {k: v for k, v in pd.scs_cones.items() if v} } "
                f"device={dev.x.devices()}")

            for method in methods:
                try:
                    jvp = dev.jvp(dP, dA, jnp.asarray(uq), jnp.asarray(ub), solve_method=method)
                    t0 = time.perf_counter()
                    jvp = dev.jvp(dP, dA, jnp.asarray(uq), jnp.asarray(ub), solve_method=method)
                    jax.block_until_ready(jvp)
                    t_jvp = time.perf_counter() - t0
                    vjp = dev.vjp(jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vs), solve_method=method)
                    t0 = time.perf_counter()
                    vjp = dev.vjp(jnp.asarray(vx), jnp.asarray(vy), jnp.asarray(vs), solve_method=method)
                    jax.block_until_ready(vjp)
                    t_vjp = time.perf_counter() - t0
                    gP, gA, gq, gb = vjp
                    dirn = float(np.sum(np.asarray(gP.data) * dP_full[P_rows, P_cols])
                                 + np.sum(np.asarray(gA.data) * dA_dense[A_rows, A_cols])
                                 + np.asarray(gq) @ uq + np.asarray(gb) @ ub)
                    e_jvp = max(rel(a, r) for a, r in zip(jvp, ref_jvp, strict=True))
                    e_vjp = max(rel(gq, rq), rel(gb, rb), abs(dirn - ref_dir) / max(abs(ref_dir), 1e-300))
                    ok = bool(np.isfinite(e_jvp) and np.isfinite(e_vjp) and e_jvp < JVP_RTOL and e_vjp < JVP_RTOL)
                    check(f"{name}/{method}", ok,
                          f"JVP rel err {e_jvp:.1e}, VJP rel err {e_vjp:.1e} "
                          f"(jvp {1e3 * t_jvp:.1f} ms, vjp {1e3 * t_vjp:.1f} ms)")
                except Exception as err:
                    check(f"{name}/{method}", False, f"raised {type(err).__name__}: {str(err)[:300]}")
                    out("     " + traceback.format_exc().strip().replace("\n", "\n     ")[-1500:])
        except Exception as err:
            check(f"{name}/setup", False, f"raised {type(err).__name__}: {str(err)[:300]}")
            out("     " + traceback.format_exc().strip().replace("\n", "\n     ")[-1500:])


# ─── 3. learning curves ───────────────────────────────────────────────────


def learning_curves() -> None:
    out("\n== 3. Learning curves on GPU (experiments/learning_curves.py)")
    script = os.path.join(ROOT, "experiments", "learning_curves.py")
    platforms = "cpu" if os.environ.get("DIFFQCP_GPU_CHECK_DRY_RUN") else "cuda,cpu"
    env = dict(os.environ, DIFFQCP_PLATFORM=platforms, PYTHONPATH=ROOT)
    with tempfile.TemporaryDirectory() as tmp:
        for problem, expected in CPU_FINAL_LOSS.items():
            for variant in ("host", "device"):
                path = os.path.join(tmp, f"{problem}_{variant}.npz")
                t0 = time.perf_counter()
                proc = subprocess.run([sys.executable, script, "run", problem, variant, path],
                                      capture_output=True, text=True, env=env)
                elapsed = time.perf_counter() - t0
                name = f"curve {problem}/{variant}"
                if proc.returncode != 0 or not os.path.exists(path):
                    check(name, False, f"exit {proc.returncode}: {proc.stderr.strip()[-800:]}")
                    continue
                r = np.load(path)
                losses = r["losses"]
                final = float(losses[-1])
                complete = len(losses) == int(r["num_iter"])
                ok = complete and abs(final - expected) / expected < LOSS_RTOL
                check(name, ok, f"{len(losses)}/{int(r['num_iter'])} iters, final loss {final:.4e} "
                      f"(CPU {expected:.4e}), {elapsed:.0f}s, statuses {sorted(set(r['statuses'].tolist()))}")


if __name__ == "__main__":
    report_path = sys.argv[1] if len(sys.argv) > 1 else "gpu_report.txt"
    if environment():
        solver_agreement()
        learning_curves()
    out("\n== Summary")
    out(f"  {len(failures)} failed check(s)" + (f": {', '.join(failures)}" if failures else ""))
    with open(report_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    sys.exit(1 if failures else 0)
