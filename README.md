# General
(Update Jan. 26 2025)

**Status.** The NumPy/SciPy implementation of `diffqcp` has successfully been ported to PyTorch, correctly computing Jacobian-vector products for both small and large instances of least squares and least two-norm problems (see `tests/test_cone_program_diff.py`). Consequently, the previous main branch was overwritten with the `torch_port` branch, and a new branch now houses the NumPy/SciPy backended `diffqcp` prototype (RIP).

It's worth noting that to port `diffqcp` to PyTorch, it was necessary to
- Replace the `PyLops` dependency with `torch-linops`, which also required implementing some linear operators we were relying on the `PyLops` dependency for.
- Switch from CSC format to CSR format for all sparse matrices.

# TODOS
(Update Jan. 28 2025)

## Next steps
To publish our paper on arXiv we need `diffqcp` to support cases where $\mathcal{K}$ is an intersection of zero cones, nonnegative cones, second order cones, exponential cones, and power cones. The current PyTests already validate that `diffqcp` is yielding correct Jacobian-vector products for the case where $\mathcal{K}$ is a zero cone, so to get to a "paper-release `diffqcp` version" we need to complete the following (in the order they are written):

1. Test the projection implementations (such as in `diffcp`). **COMPLETE** (including for the PSD cone)
2. Test the derivative of the projection implementations (such as in `diffcp`). **COMPLETE** (except that JVPs for derivative of projection onto PSD cone are not right; **not impactful to paper release**)
3. Test `proj` and `dproj`. **COMPLETE**
4. Test JVPs for a QCP where $\mathcal{K} = \mathbf{R}^{n}_{+}$. (I'll look for such a QCP that has an analytical expression for the derivative or JVP, or will just test via finite differences.)
5. Test JVPs for a QCP where $\mathcal{K}$ is a SOC. (I'll refer to Theorem 2 and Corollary 5 in RandALO.)
6. Test JVPs for a QCP where $\mathcal{K}$ is an intersection of the zero cone, nonnegative cone, and a SOC. (This time I'll just use finite differences.)
7. Support for exponential cone.
    - we previously discussed supporting the exponential cone by using `CVXPY` to solve the opt. problem required by one case of projecting onto the exponential cone. While we are just going for a "working" version right now, the problem of taking this approach is that then technically `diffqcp` wouldn't be (being very imprecise) "fully on the GPU." Is this fine or do you think we should bite the bullet and build the interior point solver from prox. algs. ourselves? Thinking ahead, it could be the case that building a torch version of the Newton method that uses the Newton step from prox algs. could be better than calling cuclarabel since we'd be exploiting problem structure (out of my depth here)?
8. Support for power cone.
    - Do you have any resources for the projection onto this cone and its derivative? So far I just have [95]-[97] from `scs_long`, which also consider higher-dimensional power cones (it seems like).

Additionally, I also need to:
- Understand why we seemingly need to return `-dx, -dy, -ds` from `diffqcp.qcp.compute_derivative.derivative`. This is what `diffcp` does, despite
those negatives not being in the paper. I reached out to Akshay and he agreed that it is due to some canonicalization specificity, although he couldn't pinpoint it exactly. I found some comments at the bottom of diffcp's `test_cone_prog_diff.py` that I'm planning on looking over to see if that gives a lead. That said, I'm not too concerned about this issue considering there's a history to it.
- Determine why `test_Du_Q_T_is_approximation` and `test_Du_Q_lsqr` are failing in `test_deriv_atoms.py`. Albeit the lsqr test is failing only barely, and it did work for the NumPy/SciPy implementation; I also wouldn't be surprised if the adjoint test failure is just a product of an incorrect test implementation due to my not-complete understanding of the matter. Because the dot test is passing for my $D_uQ(u, \mathcal{D})$ operator, I'm also not too conerned about this issue.

## Medium term goals
To conduct proper numerical experiments (or rather, to position `diffqcp` to do well in numerical experiments) we need to
- GPU accelerate functions such as `proj` in `diffqcp/cones.py`.
- Cache projections and derivatives of projections.

To claim we have a feature complete product, we need to
- Add adjoint support; (since the first of the two options would lead to a fair bit of code we'd end up deleting, I lean toward spending effort toward the second path)
    - if we don't care about numerical performance, we simply need to vectorize our handling of $\mathcal{D}$ so the adjoint can be computed via the `torch-linops` autodiff functionality,
    - otherwise we need to derive the vector-Jacobian product expressions. 
- Add (which also means test, since technically PSD cone support is already added) support for the remaining cones.

## Long term goals
(The first item should be done first, the ordering of the second and third is maybe more ambiguous?):
1. Allow `diffqcp` to compute JVPs and VJPs for batches of QCPs.
2. Integrate `CuClarabel` and `diffqcp`.
3. GPU-accelerate `CVXPYlayers` by replacing its `diffcp` backend with `diffqcp`
