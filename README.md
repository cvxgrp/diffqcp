# General
(Update Mar. 1 2025)

**Status.** `diffqcp` is almost ready to be released! Currently:
- The software is able to compute Jacobian-vector products (JVPs) of quadratic cone programs (QCPs) when $\mathcal{K}$ is an intersection of zero cones, nonnegative cones, and second order cones.
- The software fully supports projecting onto the positive semidefinite, exponential, and 3D power cones.
- **HOWEVER**, `diffqcp` will be refactored (specifically how it handles cone functionality) before release. Not expecting this to be too difficult, but updating the tests could potentially be time consuming.


# TODOS

## Next steps
To publish our paper on arXiv we just need to
1. Implement the JVPs for the exponential and power cones. (The latter will probably be more time consuming as that will require truly implementing the JVP. Implementing the JVP for the exponential cone is more of a port.)
2. Ensure the software properly computes JVPs of QCPs when $\mathcal{K}$ is an intersection of the three cones `diffqcp` currently supports **and** exponential and 3D power cones.

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
- Work on PSD `diffqcp` integration (projection and derivative of projection are both working, but finite difference `diffqcp` test is failing.)

## Long term goals
(The first item should be done first, the ordering of the second and third is maybe more ambiguous?):
1. Allow `diffqcp` to compute JVPs and VJPs for batches of QCPs.
2. Integrate `CuClarabel` and `diffqcp`.
3. GPU-accelerate `CVXPYlayers` by replacing its `diffcp` backend with `diffqcp`
