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

| #  | Wave                                | Status      | Branch                       |
|----|-------------------------------------|-------------|------------------------------|
| 0  | Branch hygiene                      | done        | (parked: `wip/cones-split`)  |
| 1  | Tooling floor (lint/type/CI)        | done        | `wave-1/tooling-floor`       |
| 2  | Test foundation (FD + AD checks)    | pending     | —                            |
| 3  | Cones cleanup                       | pending     | —                            |
| 4  | Unify CPU/GPU problem data          | pending     | —                            |
| 5  | Solver dispatch as typed strategy   | pending     | —                            |
| 6  | Batching over problem data          | pending     | —                            |
| 7  | Sparse F + direct-solve diagnosis   | pending     | —                            |
| 8  | `cvxpylayers` interface             | pending     | —                            |
| 9  | `cvxcp` extraction                  | pending     | —                            |

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

**Wave 2 — Test foundation.** Move `experiments/cvx_problem_generator.py`
→ `tests/_problems.py`. Parametrized fixtures: LP/QP/SOCP/SDP/EXP/POW/mixed.
`test_jvp_finite_difference.py`, `test_vjp_adjoint.py` (`<v, J@u> = <Jᵀv, u>`),
`test_cones_via_autodiff.py` (cross-check `dproj` vs `jax.jacrev(proj)`).

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
- `feature/maintenance-and-hygiene` — parked. Holds in-flight per-cone file
  split + cvxcp scaffold + planning notes (`claude-plan.md`, `patterns.md`).
  Will fold the useful parts into Wave 3 cones cleanup; the rest archives.
- `wave-1/tooling-floor` — current.

## Conventions for this effort

- One PR per wave; small enough to review.
- Architecture decisions get a `docs/architecture/<name>.md` note before code.
- Abstract/final discipline (see Equinox pattern doc) is enforced; type
  checker catches violations where possible.
