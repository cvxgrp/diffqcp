# diffqcp productionization — status

Living document tracking the multi-wave effort to take `diffqcp` from research
software to production-quality library. Updated at the end of each wave.

## Goal

Make `diffqcp` a high-quality, production-grade JAX library that
1. Continues to back the paper (Healey, Nobel, Boyd 2025).
2. Serves as a clean (CuClarabel-decoupled) backend for `cvxpylayers`.
3. Eventually exports its cones submodule as the standalone `cvxcp` library
   (separate repo under `healeyq3`).

## Wave plan

Each wave is one PR-shaped chunk landing on `main`. Tooling and tests come
before refactors so subsequent waves have a safety net.

All work lives on the long-running `productionization` branch as wave-shaped
commits. One PR off it merges into `main` at the end. (The user is sole
maintainer; this avoids per-wave PR ceremony while preserving wave-shaped
commits for bisection and inspection.)

| #  | Wave                                | Status      |
|----|-------------------------------------|-------------|
| 0  | Branch hygiene                      | done        |
| 1  | Tooling floor (lint/type/CI)        | done        |
| 2  | Test foundation (FD + AD checks)    | done        |
| 2.5| Existing-test cleanup               | done        |
| 3  | Cones cleanup                       | pending     |
| 4  | Unify CPU/GPU problem data          | pending     |
| 5  | Solver dispatch as typed strategy   | pending     |
| 6  | Batching over problem data          | pending     |
| 7  | Sparse F + direct-solve diagnosis   | pending     |
| 8  | `cvxpylayers` interface             | pending     |
| 9  | `cvxcp` extraction                  | pending     |

### Wave details

**Wave 1 — Tooling floor (done).** Landed:
- `lineax>=0.1.0` pinned (release pulled from git pin); Python floor stays at
  `>=3.11` because JAX 0.7.2 requires it.
- `ruff` configured with E/F/W/I/N/B/UP/SIM/RUF; targeted ignores for
  canonical math notation (N802/N803/N806/N815) and jaxtyping shape strings
  (F722, UP037). All 19 PEP 695 `type X = …` aliases rewritten to
  `X: TypeAlias = …` for 3.11 compat. Per-file ignores for files slated
  for refactor in later waves; STATUS tracks paydown.
- `pyright` in `basic` mode, included subset is everything that passes
  cleanly today: `diffqcp/__init__.py`, `_helpers.py`, `linops.py`,
  `qcp_derivs.py`, `cones/__init__.py`, `cones/abstract_projector.py`.
  Files with type debt are explicitly excluded with the wave they get
  added back in.
- `.pre-commit-config.yaml` with ruff (lint+format), pyright, hygiene hooks.
- `.github/workflows/ci.yml`: lint, typecheck, test (ubuntu+macos × 3.11/12/13;
  3.13 informational via `continue-on-error`). Coverage emitted as artifact.
- Old `test.yml` removed.
- `pytest` configured with `--strict-markers`; coverage configured.

**Deferred from Wave 1 (tracked here):**
- `ruff format` not enforced in CI yet — would churn 23 files. Lands in
  Wave 2 alongside test reorg.
- Identifier rename away from canonical math notation (P, A, q, b, x, y, s,
  dP, dA, …) so `N802/N803/N806/N815` can be re-enabled. Big touch; defer
  to a dedicated rename PR (likely after Wave 4).

**Wave 2 — Test foundation (done).** Landed:
- `tests/problems.py` consolidates the prior `experiments/cvx_problem_generator.py`
  and the `QCPProbData` dataclass that lived in `tests/helpers.py`.
  Generators take an `Rng | int | None` and are deterministic.
- `tests/helpers.py` slimmed to converters + `tree_allclose` + `get_zeros_like_*`.
- `experiments/cvx_problem_generator.py` deleted; experiments import from
  `tests.problems` instead.
- `tests/test_qcp_adjoint.py`: JVP/VJP duality identity
  `<v, J(u)> = <J*(v), u>` for least-squares CQPs (12×6 and 20×10).
  Tolerance is `1e-3` relative because `qcp.py` hard-codes LSMR at `1e-8`;
  Wave 5's typed solver dispatch will let us tighten this.
- `tests/test_cones_autodiff.py`: 8 tests cross-checking analytical
  `dproj.mv(dx)` against `jax.jvp` for zero/nonneg/SOC/PSD/product cones.
  Sampling avoids cone boundaries (kinks). EXP and POW deferred to Wave 3
  due to NaN-producing branches under autodiff.

**Caveats discovered in Wave 2 (paid down later):**
- `eqxi.GetKey()` keys are session-scoped, so test ordering can change which
  RNG sequence each test sees. `test_proj_exp_scs` and `test_product_projector`
  pass in isolation and pass in the full suite once trivially-redundant
  `jax.config.update("jax_platform_name", "cpu")` calls are removed from new
  test modules. Wave 3 will fix the underlying NaN-in-unused-branch issue
  in `cones/exp.py` so this becomes a non-issue.

**Deferred from Wave 2:**
- `test_jvp_fd.py` (FD cross-check of QCP solution map). The well-defined
  Jacobian only exists in active-set-stable regions; FD step too large
  crosses kinks, too small loses precision. Cleanest implementation is in
  Wave 7's solver-quality harness, where we already need this kind of test.

**Wave 2.5 — Existing-test cleanup (done).** Bug fixes + simplification on
the pre-existing test files Wave 2 didn't touch:
- `test_problem_data.py:_make_upper_tri_bcoo` was sampling `rng.standard_normal(n)`
  (1-D) instead of `(n, n)`; the test was passing only because comparisons
  coincided trivially on a vector. Fixed to actually exercise an n×n matrix.
- `test_qcp_cpu_analytical.py` and `test_qcp_gpu_analytical.py` had
  `np.random.seed(0)` *inside* the `for _ in range(10):` loop, making every
  iteration test the same problem. Now use `np.random.default_rng(0)` outside
  the loop so the 10 iterations actually exercise distinct problems.
- Same files: dropped the `eqx.partition`/`eqx.combine` ceremony (was
  working around a `P_diag_mask` boolean-indexing issue in `QCPStructureCPU.form_obj`
  that Wave 4 fixes via `eqx.field(static=True)`); call `qcp.jvp` directly.
- Same files: replaced inline `cvx.Problem` construction with a small
  `_ls_problem(A, b)` helper. Use of the shared `tests.problems` API is
  explicitly *not* possible here because the test needs the raw `(A, b)`
  matrices to compute the closed-form Jacobian.
- `test_qcp_cpu_analytical.py:6` redundant `jax.config.update("jax_platform_name", "cpu")`
  removed — same root cause as the test-isolation flakiness we hit in Wave 2.
- Tolerance: with bug-1 fixed, distinct problems revealed that main's
  `atol=1e-8` only passed because of bug-1; the LSMR-bottleneck floor with
  `db = 1e-6·N(0,1)` is ~1e-8 absolute, so we use `atol=1e-7` (5× headroom).
  Wave 5 (configurable solver tolerance) lets us tighten this back down.

Test count: 19 → 29 in Wave 2 → 29 in Wave 2.5 (no new tests; same coverage,
honest tolerances, real bugs fixed).

**Wave 3 — Cones cleanup.** Land per-cone file split cleanly: every projector
final + correct `__check_init__`; replace `jnp.ndim` dispatch in operator
`mv` with 1D implementations called via `eqx.filter_vmap` at the boundary;
fix `ZeroConeProjectorJacobian` static field; implement `as_matrix` for
testability; fill `cones/CONVENTIONS.md`. Defer `cvxcp` extraction.

**Wave 4 — Unify CPU/GPU problem data.** One final `QCPStructure` (or two
sharing only an `AbstractVar` interface, with all init in finals — no
`obj_matrix_init` on the ABC). One `ObjMatrix` type. Bury BCOO/BCSR choice
as strategy or one-time conversion at construction. Delete
`QCPStructureLayers` and `ConstrMatrixCPU` stubs (or implement properly).

**Wave 5 — Solver dispatch.** Replace `solve_method: str` with
`solver: AbstractLinearSolver`. Default = `lx.LSMR(rtol=…, atol=…)`. Move
nvmath/cuDSS behind a thin `AbstractLinearSolver` adapter. Delete the
parallel `_jvp_nvmath` / `_vjp_nvmath` paths.

**Wave 6 — Batching.** Explicit batching contract; primarily `eqx.filter_vmap`
over the unbatched class. Drop ad-hoc `ndim`-dispatch in operator `mv`.
Tests covering batched jvp/vjp consistency vs. per-instance loops.

**Wave 7 — Sparse F + direct-solve.** `bench/` harness reporting condition
number / residual / gradient norm across solvers on fixed problems. Use it
to track down the LU exploding-gradient issue. Move cuDSS path to sparse F.

**Wave 8 — `cvxpylayers` interface.** Clean interface file using `diffqcp`
directly (no CuClarabel coupling). End-to-end test:
CVXPY → cvxpylayers → diffqcp → gradient.

**Wave 9 — `cvxcp` extraction.** Once cones are stable: new repo under
`healeyq3`; `diffqcp` depends on it.

## Items from the original brief not yet assigned to a wave

From `docs/original-brief.md`; fold into a wave or drop explicitly.

- Batched `jvp`/`vjp` over inputs for a single problem (test that this works
  today, independent of Wave 6's batching over problem data).
- `docs/` note on cone architecture patterns, split into: cones that reduce to
  one dimension, products of cones of differing dimension, and products of 3D
  cones (EXP/POW). Partially covered by Wave 3's `CONVENTIONS.md`.
- Consistent, complete docstrings across the package.
- Decide whether `experiments/` and `tests/` should both exist (Wave 2 made
  experiments import from `tests.problems`, which couples them).
- Keep a running learning log: architecture decisions, numerical linear
  algebra notes, DevOps how-tos (`docs/architecture/` is the intended home).

## Known issues to fix as we touch their files

- `diffqcp/problem_data.py:295,298,324,326` — `ObjMatrixCPU/GPU.in_structure`
  returns `None` (`pass`). Real bug.
- `diffqcp/cones/canonical.py:295-299` — `_SecondOrderConeProjector.__check_init__`
  references `self.dims` (typo for `self.dim`).
- `diffqcp/cones/canonical.py` operator `mv` methods dispatch on `jnp.ndim`
  to handle vmap — fragile; replace with single 1D impl + `vmap` at boundary.
- `_jvp_nvmath` / `_vjp_nvmath` instrumentation seams in `qcp.py` — fold
  into solver strategy in Wave 5.
- `QCPStructureLayers` (`problem_data.py:242-261`) and `ConstrMatrixCPU`
  are stubs — decide in Wave 4.
- `diffqcp/problem_data.py:129` — `ObjMatrixCPU.__init__(P, P.T, diag)` calls
  with `diag: Array` while `ObjMatrixCPU` declares `diag: Float[BCOO, " n"]`.
  Fix in Wave 4.

## Branches

- `main` — clean baseline.
- `feature/maintenance-and-hygiene` — parked. Holds an in-flight per-cone file
  split + cvxcp scaffold (mostly empty stubs; not runnable). Its planning notes
  now live in `docs/original-brief.md` and `docs/patterns.md`, so the branch
  can be archived once Wave 3 has used its file layout as a sketch.
- `productionization` — long-running integration branch carrying all wave
  commits. Single PR off this branch lands at the end.

## Conventions for this effort

- One PR per wave; small enough to review.
- Architecture decisions get a `docs/architecture/<name>.md` note before code.
- Abstract/final discipline (see Equinox pattern doc) is enforced; type
  checker catches violations where possible.
