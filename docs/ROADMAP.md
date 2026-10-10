# diffqcp roadmap: open questions, PR plan, and task backlog

The single list of everything still open after the 2026-10 working session on
`claude/compassionate-gauss-87v0j0`. `STATUS.md` is the history (what was
done and why); this file is what is left.

That branch is the **target end state**. It lands on `main` through a series
of small PRs (section 3), so every change is reviewed by a person.

Workflow this document is organised around:

1. **Review the math** (section 1), by hand and with a full model sweep of the paper.
2. **Close the open questions** (section 2).
3. **PR the branch into `main` in slices** (section 3).
4. In between, **hand self-contained tasks to models** (section 4: each card
   has a goal, files, a definition of done, and whether it needs you or a GPU).

Status keys: `[ ]` open, `[~]` in progress, `[x]` done (leave done items for a
while so the history is visible, then delete).

---

## 1. Math review checklist

Addendum (rank of D_z N, gauge fixing):
https://claude.ai/code/artifact/46705f4e-d9f2-49ba-a14d-b6a3be600188

- [ ] **M1. Paper §2.4: "D_z N invertible" never holds.** `N(alpha z) = N(z)`
  gives `F z = 0` for every QCP, so F is singular; the assumption should read
  "rank N - 1" (addendum Prop. A.1, Thm. A.5). Check the proof and that the
  JVP/VJP formulas of §3 are otherwise unchanged.
- [ ] **M2. Left null vector `(Pi z)^T F = 0`** (addendum Lemma A.2, from
  `u^T Q(u) = 0`). Holds only *at an exact solution*: at a solver's 1e-10
  solution, `(Pi z)^T F'` is ~1e-8 relative (measured).
- [ ] **M3. The addendum's reduced system (A.6) needs a correction.** Simply
  dropping F's last row assumes M2 holds exactly; with a real solution it adds
  an O(residual x cond) error (1e-7 measured). `ReducedDirectSolver` instead
  computes the exact left null vector `l` of F' from the same LU
  (`F-bar^T l-bar = -f_N`) and projects against it, which recovers exactly the
  least-squares (JVP) / minimum-norm (VJP) answers. The addendum should state
  this; see `diffqcp/solvers.py`.
- [ ] **M4. Conditioning claims.** The addendum (and an earlier
  `solvers.py` note) says cond(K) of the augmented system is about
  sigma_1 / sigma_min^2, i.e. up to cond(F')^2. In the conditioning sweep
  (`experiments/results/derivative_benchmark_conditioning.json`) cond(K) was
  only ~5x cond(F'), and the augmented LU stayed at ~1e-15 up to
  cond(F') = 1e7. Reconcile the bound with the measurement (sigma_1 grows with
  the scaling too) and soften the text accordingly.
- [ ] **M5. cond(F-bar) vs cond(F')** within `sqrt(1 + ||x||^2 + ||y||^2)`:
  check the proof; the sweep shows cond(F-bar) <= cond(F') throughout.
- [ ] **M6. Cone Jacobians vs the paper's appendix.** The code bugs fixed this
  session were deviations from formulas the paper has right (PSD `B` block of
  ones, power cone `alpha (1 - alpha)` terms, dual power cone sign, the polar
  test). Re-derive each cone's Jacobian against the appendix once more (the
  tests now cross-check against diffcp and finite differences, but not
  against the paper text).
- [ ] **M7. Non-differentiable points** (addendum A.6): what the
  least-squares / minimum-norm answer means there, and whether to document
  it as a generalized derivative.
- [ ] **M8. Full paper sweep** (planned with Fable): notation, every
  derivation in §2-3 and the appendix, and the numerical section, against the
  current code.

## 2. Open questions and decisions

- [ ] **Q1. Default derivative solver.** Benchmark
  (`experiments/results/derivative_benchmark*.{json,png}`): reduced LU is the
  most accurate method (1e-14..1e-16) and the fastest exact one up to
  N ~ 3000; LSMR is 1e-8..1e-11 and is the only diffqcp option beyond that.
  Proposal: an `AutoSolver` that uses `ReducedDirectSolver` for N <= ~2000-3000
  and LSMR above (task E1).
- [ ] **Q2. LSMR iteration cap.** lineax's default `max_steps` (10 x min dim)
  is hit from cond(F') ~ 1e6; lineax then raises a misleading "non-finite
  output" error. Raise the cap, fall back to a direct solve, or both (E1).
- [ ] **Q3. Wave 4 CPU/GPU class merge** (`HostQCP` / `DeviceQCP`): API change;
  proposal in `STATUS.md` under Wave 4.
- [ ] **Q4. `ndim` dispatch in cone Jacobian `mv`s** (soc/psd/pow/exp): fragile
  (the power-cone vmap bug was one instance). Replace with 1D operators +
  `vmap` at the boundary? Changes how batched operators are applied (E4).
- [ ] **Q5. Patch release from `main`?** The PSD, power-cone, exp-cone,
  off-diagonal dP and `DeviceQCP.vjp`-default fixes affect cvxpylayers users
  today. Probably the first PRs (section 3) plus a release.
- [ ] **Q6. Upstream reports** (none filed yet; drafts are a model task, E12):
  - cvxpylayers 1.2.0: (a) MOREAU silently drops Moreau "direct cones"
    (constraints on bare variables, e.g. `x >= 0`), so the layer solves a
    different problem; (b) a second CSR-based layer built from the same
    `cvx.Problem` gets corrupted data (in-place permutation of cached
    matrices); (c) the JAX layer cannot use a custom canonical-matrix
    `SolverInterface`, and the DIFFCP JAX forward pass without a gradient
    crashes.
  - diffcp 1.1.x: (d) the implicit modes (`lsqr`/`lsmr`/`dense`) ignore P
    (the derivative is that of the LP embedding) without warning; (e) the
    adjoint crashes on power cones; (f) the Clarabel path returns wrong PSD
    solutions; (g) the default `lsqr` mode loses all accuracy from
    cond ~ 4e3 (benchmark sweep).
  - Moreau 0.4.1 (via Parth): gradient accuracy 1e-4..1e-6 in
    `diff_method="exact"`, degrading to O(1) as conditioning worsens; the
    active-set path (small zero/nonneg problems) is accurate to 1e-12.
- [ ] **Q7. diffcp as the test oracle.** Given (d)-(g), keep it only for P = 0
  problems with `mode="dense"` and SCS at tight tolerance (current tests), and
  lean on the finite-difference and 50-digit references elsewhere?
- [ ] **Q8. `experiments/` vs `tests/`.** Experiments import from
  `tests.problems`; the old scripts (`cpu_experiment.py`, `cpu_baseline.py`,
  `direct_solve_experiment.py`, `gpu_experiment.py`, `heterogeneous_*.py`) are
  superseded by `learning_curves.py`. Delete or archive them?
- [ ] **Q9. How diffqcp reaches CVXPYlayers users on CPU.** Upstream a
  Clarabel + diffqcp backend (fixing (c) along the way), or ship a thin layer
  wrapper in diffqcp (E6)?
- [ ] **Q10. GPU results.** Run `bash scripts/gpu_check.sh` on the work GPU
  and send back `gpu_results.tar.gz`; then compare with the CPU benchmark (E14).

## 3. PR plan: landing the branch on `main`

`main` (3326aeb) is an ancestor of the branch, and the 28 branch commits are
linear, so each PR can be a **contiguous commit range**: create a branch at
the last commit of the slice and open the PR against `main` (or against the
previous slice's branch, while that one is still under review).

```bash
git fetch origin
git branch pr/01-tooling d361bb2 && git push -u origin pr/01-tooling   # PR into main
# after it merges (or stacked on it):
git branch pr/02-tests 526bb81 && git push -u origin pr/02-tests
```

Cut points (oldest first; sizes from `git diff --stat`):

| PR | Last commit | Commits | What |
|----|-------------|---------|------|
| 01 | `d361bb2` | wave 1 | Tooling floor: ruff, pyright, CI |
| 02 | `526bb81` | wave 2, 2.5, uv.lock, planning docs | Test foundation (`tests/problems.py`, adjoint and cone AD tests) |
| 03 | `7e8414f` | wave 2.6 | diffcp oracle tests + first correctness fixes (PSD Jacobian, PSD row order + `clarabel_to_scs_permutation`, `DeviceQCP.vjp` default) |
| 04 | `7f9719a` | wave 5 | Gauge-fixed derivative system, `diffqcp.solvers` (LSMR / dense direct), LSMR tolerance fix |
| 05 | `60d0fd2` | off-diagonal dP fix, QP FD oracle, `differentiable_solution`, power/exp cone fixes | Correctness fixes + `jax.grad` entry point |
| 06 | `be889be` | wave 3 | Per-cone modules, `as_matrix`, cones type-checked |
| 07 | `3450513` | wave 4, wave 6 | Problem-data fixes, stub removal, batching contract |
| 08 | `ad4d971` | deps, STATUS, learning curves, GPU scripts, conditioning note | Experiments + GPU verification scripts |
| 09 | `d68c829` | layer learning curves | diffqcp vs diffcp vs Moreau through CVXPYlayers |
| 10 | `0e05988` | docstrings, power-cone vmap fix | Docstrings + type hints everywhere; `qcp.py` cleanup |
| 11 | `HEAD` | reduced solver, benchmark, GPU benchmark, this roadmap | `ReducedDirectSolver`, derivative benchmark |

Notes:

- PRs 03-05 carry the user-facing bug fixes; if a patch release comes first
  (Q5), cut it after PR 05.
- PR 02 and PR 06 are the largest (test and cone reorganisations); review
  them for structure, not line by line.
- After merges, CI on `main` must stay green: every cut point above passed
  `pytest`, `ruff` and `pyright` when it was committed, but check each slice
  in CI before merging (a model can do this; task P1).
- If a review asks for changes, make them on the slice branch, then rebase
  the remainder of the end-state branch onto it (a model can do this too).

## 4. Task backlog (assignable to models)

Each card is meant to be pasted into a fresh session as-is. Unless a card says
otherwise: work on a new branch off `claude/compassionate-gauss-87v0j0`, keep
`pytest`, `ruff check` and `pyright` (`uv run pyright`) green, don't change
unrelated files, and finish with a short report of what changed and what was
verified. Sizes: S (< 1 h), M (a session), L (several sessions).

### Process

- [ ] **P1. Slice PRs (section 3).** Create the `pr/NN-*` branches at the cut
  points, run the checks at each, and draft each PR description from the
  commits in its range. Needs you: opening/merging the PRs. Size M.

### Engineering: derivative solvers

- [ ] **E1. Solver selection + LSMR cap (Q1, Q2).** Add an `AutoSolver`
  (`ReducedDirectSolver` for N <= a threshold, LSMR above) and decide whether
  it becomes the default; raise LSMR's `max_steps` (or catch lineax's error
  and fall back to the direct solver for small N) and make the error say
  "did not converge". Files: `diffqcp/solvers.py`, `tests/test_solvers.py`,
  README "Selecting solvers". Done: benchmark rerun shows no accuracy or speed
  regression; the conditioning sweep no longer has LSMR failures. Needs you:
  the default (API behaviour). Size M.
- [ ] **E2. Sparse factorization of F-bar.** The dense solvers stop at
  N ~ 3000. F = D_u Q(Pi z) DPi(z) - DPi(z) + I is sparse up to the cone
  blocks (dense per SOC/PSD/exp/pow block), so F-bar can be assembled as CSR
  and factored with a sparse LU (`jax.experimental.sparse.linalg.spsolve`, or
  SciPy SuperLU via `pure_callback` on CPU; cuDSS on GPU). Done: a
  `SparseReducedSolver` matching `ReducedDirectSolver` to 1e-12 on the
  benchmark, timed up to N ~ 30k. Size L.
- [ ] **E3. Fold `nvmath-direct` into a solver class.** `DeviceQCP` still has a
  separate, non-jittable cuDSS path; make it an `AbstractDerivativeSolver`
  (ideally on the reduced system, sparse, from E2). Needs a GPU. Size M.
- [ ] **E7. Performance profiling.** diffqcp LSMR takes 53 ms/iter on the
  logistic layer problem vs 5 ms for Moreau, and 2-5x diffcp's `lsqr` time at
  the largest sizes. Profile (LSMR iterations, Python/dispatch overhead,
  `pure_callback`, operator construction per call) and try a looser LSMR
  tolerance + preconditioning (e.g. diagonal/Jacobi on F'). Done: a written
  breakdown and any wins with benchmark numbers. Size M.

### Engineering: library

- [ ] **E4. Remove `ndim` dispatch from cone operators (Q4).** 1D operators,
  batching only via `vmap` at the boundary; delete the 2D/3D branches in
  `soc.py`, `psd.py`, `pow.py`, `exp.py`, `linops.py`. Done: all tests,
  including `tests/test_batching.py`, pass. Needs your decision on Q4. Size M.
- [ ] **E5. CPU/GPU class merge (Q3).** Per the Wave 4 proposal in
  `STATUS.md`. Needs your decision. Size L.
- [ ] **E6. diffqcp as a CVXPYlayers backend on CPU (Q9).** Start from
  `DiffqcpLayer` in `experiments/layer_learning_curves.py` (Clarabel forward +
  `make_differentiable`, reusing the MOREAU canonicalization). Upstream option:
  a `CLARABEL_DIFFQCP` backend in cvxpylayers' JAX layer plus the fix for
  (c). Needs you: the choice and any upstream contact. Size L.
- [ ] **E8. Learning-curve regression test.** A small (~20 iterations, small
  problems) version of `experiments/learning_curves.py` in CI, asserting the
  loss decreases and matches stored diffqcp curves. Size S-M.
- [ ] **E9. Benchmark extensions.** SDP family (diffqcp + diffcp; Moreau needs
  the Clarabel-to-SCS row permutation), float32 runs, several cotangents per
  case, and the reduced solver in the layer learning curves. Size M.
- [ ] **E10. Small clean-ups found in the docstring pass** (no behaviour
  change unless noted): rename `_proj_onot_polar` -> `_proj_onto_polar`
  (exp.py); fix error-message typos ("espects", "receieved") and the
  "must be 1D or 2D" messages that also accept 3D (soc.py, psd.py); remove the
  dead `jnp.empty_like(v)` assignments in the exp-cone heuristics and the
  unused `EPS` in `canonical.py`; rename the `_ProjPSDConeJacobian.B` field
  (clashes with `B` as the batch-axis letter); clarify `xi` in pow.py's `_gi`.
  Behaviour change, separate commit: the exp-cone Newton search
  (`root_search_newton`) falls back to bisection when it stops on iteration
  20 even if converged; compare with SCS and fix. Size S.
- [ ] **E11. Dependency groups.** Add a `bench` group (cvxpylayers[jax],
  moreau, mpmath) to `pyproject.toml` + `uv.lock` (moreau-cuda12 only has
  manylinux x86_64 wheels, so keep it out of the lock or behind a marker), and
  drop the ad-hoc installs from `scripts/gpu_check.sh`. Size S.
- [ ] **E13. Experiments clean-up (Q8).** Delete/archive superseded scripts;
  break the `experiments -> tests.problems` coupling if decided. Size S.
- [ ] **E15. `STATUS.md` → history.** Trim `STATUS.md` to the wave log and
  findings; keep open items only here. Size S.

### Upstream / communication

- [ ] **E12. Draft upstream issues (Q6).** One minimal, self-contained
  reproduction per item (a)-(g), plus a short note for Parth on Moreau's
  accuracy with the benchmark figure. Output: markdown drafts in
  `docs/upstream/`; filing them needs you. Size M.

### Research

- [ ] **R1. Addendum update (M3, M4).** Add the exact-left-null-vector
  correction to the reduced system and the measured conditioning; decide
  whether the addendum becomes a paper revision or an appendix of a v2.
- [ ] **R2. Reuse the forward solver's factorization.** Moreau's speed likely
  comes from reusing its IPM KKT factorization for the backward pass, at a
  cost in accuracy. Can diffqcp take Clarabel's final factorization (or use it
  as a preconditioner for LSMR on F') and keep 1e-10 accuracy?
- [ ] **R3. Mixed precision on GPU.** Factor F-bar in float32 and refine in
  float64 (iterative refinement); float32 LSMR accuracy vs speed.
- [ ] **R4. Non-differentiable points.** Smoothed derivatives (Moreau has a
  `diff_method="smoothed"` barrier smoothing) vs the least-squares answer at
  degenerate points; what should diffqcp return and document (M7)?
- [ ] **R5. Structure of F.** Exploit the block structure (identity blocks,
  per-cone dense blocks, D_u Q's skew part) for a specialised factorization
  or a better LSMR preconditioner.
- [ ] **R6. `cvxcp` extraction** (Goal 3 in `STATUS.md`): the cones
  submodule as a standalone library.

### Waiting on you

- [ ] **E14. GPU run (Q10).** `git checkout claude/compassionate-gauss-87v0j0 &&
  bash scripts/gpu_check.sh` (optional `DIFFQCP_GPU_BENCH_CPU_BASELINE=1`);
  send back `gpu_results.tar.gz`. A model can then compare it with the CPU
  results and update section 2.
