# General
(Update Feb. 02 2025)

**Status.** `diffqcp` is seemingly close to a first release. Currently, (validating using a combination of finite difference testing and analytical expression testing) the software is able to compute Jacobian-vector products (JVPs) of QCPs when $\mathcal{K}$ is an intersection of zero cones, nonnegative cones, and second order cones.


# TODOS

## Next steps
To publish our paper on arXiv we need `diffqcp` to support cases where $\mathcal{K}$ is an intersection of zero cones, nonnegative cones, second order cones, **exponential cones, and power cones**. So to get to a "paper-release `diffqcp` version" we need to complete the following:
- Add analytical JVP tests for QCPs where SOCs and nonnegative cones comprise $\mathcal{K}$.
- Add support for exponential cones.
    - we previously discussed supporting the exponential cone by using `CVXPY` to solve the opt. problem required by one case of projecting onto the exponential cone. While we are just going for a "working" version right now, the problem of taking this approach is that then technically `diffqcp` wouldn't be (being very imprecise) "fully on the GPU." Is this fine or do you think we should bite the bullet and build the interior point solver from prox. algs. ourselves? Thinking ahead/out loud, it could be the case that building a torch version of the Newton method that uses the Newton step from prox algs. could be better than calling cuclarabel since we'd be exploiting problem structure (out of my depth here)?
8. Support for power cone.
    - Do you have any resources for the projection onto this cone and its derivative? So far I just have [95]-[97] from `scs_long`, which also consider higher-dimensional power cones (it seems like).

Additionally, I also need to:
- Determine why `test_Du_Q_T_is_approximation` is failing in `test_deriv_atoms.py`. I also wouldn't be surprised if this adjoint test failure is just a product of an incorrect test implementation due to my not-complete understanding of the matter. Because the dot test is passing for my $D_uQ(u, \mathcal{D})$ operator, I'm also not too conerned about this issue.

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
