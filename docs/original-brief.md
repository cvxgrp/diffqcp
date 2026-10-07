# Background context

(For the high-level what `diffqcp` does, see the README.md)

`diffqcp` is a library that I built over the span of about 1 year. It started as a NumPy/SciPy prototype, I then re-implemented it in PyTorch, before finally re-implementing it again with JAX. The primary motivation for the software was to accompany [this paper](https://arxiv.org/pdf/2508.17522) with secondary motivation to become a backend for [this library](https://github.com/cvxpy/cvxpylayers). In its current state, `diffqcp` is still more research software than it is production software. I'd like to make it the latter. To do so, there are a variety of clean-up and maintenance tasks we need to complete, along with larger refactoring, new features, and proper benchmarking. The remainder of this document specifies ideas and constraints for a larger, AI-assisted plan.

**Importantly**, (for claude)
1. **note that you are not currently on a clean branch.** The branch you are on I created to do some basic cleaning and to create a new module `cvxcp`: Convex Cone Projectors, which in the long-term I wish to make a different repository. Please look at `main` so you know what our current status is, and feel free to commit the changes on this branch as "tinkering" and spin-up new branches off `main` to actually complete our work.
    - The `cvxcp` repository will be owned by my `healeyq3` account not `cvxgrp` and it will need its own license, `pyproject.toml`, etc.
2. `diffqcp` currently serves as a backend for CVXPYlayers. Whatever the final status of our changes is needs to work with layers.
    - It is used in this [interface file](https://github.com/cvxpy/cvxpylayers/blob/master/src/cvxpylayers/interfaces/cuclarabel_if.py).
    - Because `diffqcp` can be used on the CPU or GPU, I'd like to create new interface files that leverage `diffqcp` since I don't like it being coupled with CuClarabel.

# Next steps

Firstly, we should thoroughly explore the codebase and explicitly note how the current code works, its shortcomings, and clever implementation details we should mimic. You are more than welcome to also suggest ideas and improvements. Along with whatever you find in this process, the following are TODOs and next steps I have in mind.

**Please inspect the software with intense criticism.** Generally, I like the patterns listed in `patterns.md`.

## Important features

The following are engineering (so productionizing) features that are not currently supported that need to be **and** research studies that we also need to complete.

### Engineering

- Batch everything
    - Allow batching over problem data
    - Should already be able to batch over provided inputs--test this. (So batch `jvp` and `vjp` for a single problem data set.)
- Support auto-diff `jvp`s of cone projectors.

### Research
Much of the remaining research is centered around linear system solves. However, there are some open questions. Specifically,
- Why are direct solvers not computing accurate solutions?
    - Specifically, the jax LU solve and the nvmath solves yield exploding gradients
- How can we do dense solves without materializing the dense matrix?
    - Dense F materialisation — `_jvp_direct_solve_get_F` and `_vjp_direct_solve_get_FT` materialise `F` as a dense (N, N) matrix. This is a prototype path only. The goal is to keep F sparse (CSR). Do not optimise the dense path further.

## Code architecture

As general context, since I implemented this library my maturity writing JAX software has grown. Moreover, there are probably many sub-optimal patterns and general untidy software practices that need to be rectified. (see my note about patterns above.)

Some thoughts I've had:
- We can probably consolidate the CPU and GPU paths some. In particular I think this is the case for the problem data class. We should explore making a single problem data class that is CPU and GPU agnostic. It would contain information that the CPU or GPU QCP classes don't need, but this information would be a one-time at compile-time cost, which is probably worth the reduction in code complexity.
- Overall we need better handling between pure JAX (cpu), JAX (gpu), and JAX + nvmath and CuPy
    - the addition of CuPy and nvmath was very much that--a tacked on addition.
- **We should consolidate the helpers used for testing and experiments**
- Should there be both `experiments` and `tests` directories?

## Code quality

Currently, there is no static type checking, hardly any unit tests, no performance benchmarks, linting is limited, and I'm unsure if our project dependencies are handled properly. We need to rectify all of this.

**Type checking & code quality** (note that we should be using the abstract/final pattern):
- We should definitely be using at least one of `mypy` and `pyright`. If they have complementary features then we should use both.
- `ruff` should be heavily used
- We should explore using `jaxtyping` for real

**Unit tests**
- Need better basic software architecture tests
- Need tests on the quality of solutions generated by `diffqcp`
    - More generally, I think the quality of the derivatives of the cones and of the solution maps to QCPs need to be better understood.
    - Need to be able to easily test the quality of solutions generated by different linear system solvers
- Should leverage the design of these [lineax tests](https://github.com/patrick-kidger/lineax/tree/main/tests)
- Should leverage JAX autodiff for tests

**Documentation**
- The codebase is small, but can be easy to forget certain architecture patterns. The cones in particular can be pretty complicated given their batching behavior. We need to create a `.md` file that explains the architecture patterns the cones adhere to. (Can be separated into cones that reduce to a single dimension, cones that are the cartesian product of different dimensionality cones, and cones that are the cartesian product of 3d cones.)
- Docstrings need to a) be completed and b) be made consistent throughout the repository

**Project dependencies**
- Want to be able to run on Mac or Linux (with or without GPU)
- Need to upgrade to most recent version of `lineax`
- Should allow for older versions of Python as supported by JAX

# General hygiene

- this is going to take probably a couple weeks and various different sessions. Moreover
    - We should generate a status doc that keeps track of our progress
    - We should front-load general exploration and planning
- I want to use this effort to better my skills as a (modern day) software engineer. Moreover, we should keep a repoire going of architecture decisions, efficient numerical linear algebra approaches, DevOps rules and how-tos, etc.