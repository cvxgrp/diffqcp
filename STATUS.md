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
| 2.6| diffcp oracle + correctness fixes   | done        |
| 3  | Cones cleanup                       | pending     |
| 4  | Unify CPU/GPU problem data          | pending     |
| 5  | Solver dispatch as typed strategy   | done (ahead of 3, 4) |
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

**Wave 2.6 — diffcp oracle + correctness fixes (done).** `diffcp` is now
an independent oracle for cone programs (`P = 0`), and pointing it at the
code immediately found real bugs:
- **PSD Jacobian bug (since PSD support landed, `f911bd1`; also on `main`).**
  `_PSDConeProjector`'s analytical Jacobian put an *identity* on the
  positive-eigenvalue block of `B` where the formula needs a block of *ones*.
  Correct only when exactly one eigenvalue is positive, so every SDP
  derivative with >= 2 positive eigenvalues at `y - s` was wrong (40-800%
  error vs. diffcp and finite differences). Existing tests only sampled
  definite points, where the Jacobian is trivially I or 0. Fixed, with a
  mixed-eigenvalue regression test in `test_cones_autodiff.py`.
- **PSD row order (Clarabel vs. SCS).** `diffqcp` vectorizes PSD blocks in
  SCS order (lower triangle, column-major); Clarabel uses the upper triangle
  column-major. The README tells users to canonicalize with `cvx.CLARABEL`,
  whose data is in Clarabel order, so SDP derivatives from that recipe are
  wrong. Added `diffqcp.clarabel_to_scs_permutation(cone_dims)` and
  `QCPProbData.scs_ordered()`. README/API decision deferred to Wave 4.
- **`DeviceQCP.vjp` defaulted to `solve_method="jax-lu"`** (dense LU of the
  singular F) while `jvp` defaulted to LSMR. cvxpylayers' CuClarabel
  interface calls `vjp` without `solve_method`, so its GPU backward pass used
  the exploding path. Default is now `"jax-lsmr"`.
- **SDP fixtures were unbounded.** `generate_feasible_sdp` used a random
  indefinite `C`, so Clarabel returned an unboundedness certificate rather
  than a solution. Now constructs a strictly dual-feasible `C`.
- **Flaky `test_nonnegative_projector`** (~1 in 8 runs): FD step crossed the
  kink at 0. Samples are now kept >= 1e-3 from zero.
- `tests/test_diffcp_oracle.py`: JVP and VJP vs. diffcp (`dense` mode, SCS at
  eps=1e-10) on LP, SOCPs, two SDPs and an exp-cone problem, plus an
  optimality guard. Exp and portfolio are strict-xfail: an *exact* solve of
  diffqcp's system matches diffcp/FD to ~1e-8, but LSMR at the hard-coded
  1e-8 tolerances stops 1e-4..1e-3 short. Wave 5 removes the xfails.

**Fix — CPU VJP `dP` off-diagonals (done).** `HostQCP.vjp` returned, for
each stored upper-triangular entry of `P`, the gradient with respect to a
single entry of the symmetric matrix; the stored value u_ij (i != j) stands
for both P_ij and P_ji, so its gradient is twice that. Every QP fixture had a
diagonal P (CVXPY canonicalizes `sum_squares` with auxiliary variables), so
the adjoint test never saw it; on a dense-P QP the identity was off by 4e-3
and holds to 5e-12 after the fix. Added `generate_dense_qp` and a
`DeviceQCP` adjoint test (full symmetric P, symmetric perturbations; that
path was already correct).

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

**Wave 5 — Solver strategy + gauge fixing (done; landed before Waves 3/4).**
- `diffqcp/solvers.py`: `AbstractDerivativeSolver` with `LSMRSolver`
  (default) and `DenseDirectSolver`; `jvp`/`vjp` take `solver=`. Legacy
  `solve_method` strings still work (`"jax-lsmr"`, `"jax-lu"`, and
  `"nvmath-direct"` on `DeviceQCP`).
- **Gauge fixing.** Every solve is on `F' = F E` (fix `dz_N = 0`) instead of
  the singular F. JVP: least-squares solve of `F' d = r`. VJP: minimum-norm
  solve of `F'^T w = g`, which is exactly the adjoint of the gauge-fixed JVP
  and also solves `F^T w = -dz` because `dz` is orthogonal to the null
  vector. Derivation in the module docstring.
- `LSMRSolver` tolerances default to 1e-12 (float64) / 1e-6 (float32), with
  `conlim=inf`. 1e-8 left 1e-4..1e-3 error; 1e-12 costs ~5-10% more
  iterations. (lineax's `conlim=1e8` was ruled out as the cause.)
- `DenseDirectSolver` LU-factors the symmetric augmented matrix
  `[[I, F'], [F'^T, 0]]` (one factorization serves JVP and VJP). Degenerate
  points (F' rank deficient) are detected by an ~epsilon LU pivot or a large
  residual and fall back to SVD least squares. A residual check alone is not
  enough: LU on a singular but consistent system returns a valid but
  non-minimum-norm solution.
- nvmath path (untested here, no GPU) now factors the dense augmented
  system instead of F; also fixed its all-zero shortcut, which returned a
  bare vector instead of the output tuple.
- Zero-RHS shortcut is now an exact `== 0` test (was `allclose(., 0)`,
  which zeroed small but legitimate perturbations).
- Tolerances tightened: diffcp oracle 1e-4 -> 1e-6 with no xfails, under
  both solvers; adjoint identity 1e-3 -> 1e-8; closed-form LS tests atol
  1e-7 -> 1e-9 (tighter than main's 1e-8).
- `tests/test_solvers.py`: `F z = 0`, F' full rank, solvers on full-rank and
  rank-deficient systems, augmented symmetry, legacy-string mapping, and the
  `DeviceQCP.vjp` default.
- Deferred: sparse augmented system for cuDSS (Wave 7); exposing solver
  diagnostics (residual, degeneracy flag) to callers.

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

## Findings that shape the remaining waves

- **F is structurally singular.** `F z = 0` for `z = (x, y - s, 1)` (the
  embedding is positively homogeneous), on every problem tested. The output
  map `dz -> (dx, dy, ds)` annihilates `z`, and the VJP right-hand side is
  orthogonal to `z`, so the systems are consistent; LSMR copes, but LU on F
  explodes. Gauge-fixing `dz_N = 0` (drop F's last column, F') removes the
  null direction: cond(F') was 35..4e4 vs. cond(F) ~1e17, and an exact
  solve of the symmetric augmented system `[[I, F'], [F'^T, 0]]` matched the
  truncated pseudo-inverse to 1e-15..1e-6. The same matrix serves the VJP
  with a different right-hand side. Implemented in Wave 5.
- **LSMR at rtol=atol=1e-8 is not accurate enough** on moderately
  conditioned problems (1e-4..1e-3 relative error on portfolio / exp).
- diffcp limitations as an oracle: its linear-solve modes reject `P`; it has
  no power cone; its Clarabel path (1.1.6) returns wrong PSD solutions; its
  default `lsqr` mode was 0.95 off on the exp problem. Use `mode="dense"`
  with SCS at tight tolerance.

## Decisions made overnight (review these)

- Kept SCS vectorization as `diffqcp`'s internal PSD convention (matches
  diffcp and `vec_symm`); converting Clarabel data is the caller's job, via
  the new public `clarabel_to_scs_permutation`.
- `QCPProbData` keeps its Clarabel-ordered fields (experiments re-solve with
  Clarabel from them) and gains `scs_ordered()`; tests build `diffqcp`
  objects from the SCS-ordered view.
- Fixed the PSD Jacobian in `canonical.py` now rather than waiting for the
  Wave 3 file split, since it is a correctness bug.
- Did Wave 5 before Waves 3 and 4: it carries the accuracy fixes (gauge
  fixing, tolerances) and does not depend on the refactors.
- Solver classes are named `LSMRSolver` / `DenseDirectSolver` (not `LSMR`)
  to avoid clashing with `lineax.LSMR`.
- Kept `throw=True` (lineax default): LSMR failing to converge raises rather
  than returning a silently inaccurate derivative.
- Kept `solve_method` strings for backward compatibility (cvxpylayers and
  the experiments use them); `solver=` takes precedence when both are given.

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
- `_jvp_nvmath` / `_vjp_nvmath` in `qcp.py` are still a separate,
  non-jittable path (Wave 5 only switched them to the augmented system).
  Folding them into an `AbstractDerivativeSolver` needs a GPU to test.
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
