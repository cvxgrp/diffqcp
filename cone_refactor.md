# Big picture
Recall that $z = (x, y - s, 1)$, where $x, y, s$ is the primal-dual solution, which is obtained from a solver.

The cones are used to compute
- $\Pi z = (u, \Pi_{\mathcal{K}^*}(v), \max\{0, w\})$
- $D\Pi(z) = \textbf{blockdiag}(I, D\Pi_{\mathcal{K}^*}(z), \alpha)$, where $\alpha = 1$ if $w \ge 0$ else $0$,

so more precisely, they are used to compute
- $\Pi_{\mathcal{K}^*}(v)$
- $D\Pi_{\mathcal{K}^*}(z)$

**Once $\Pi_{\mathcal{K}^*}(v)$ and $D\Pi_{\mathcal{K}^*}(z)$ are formed, the cone information is not used again.** (That is to say, the iterative component of `diffqcp`, solving the linear system via LSQR, uses these two **fixed** objects and no other computations are done using the cones.) **Additionally,** both $\Pi_{\mathcal{K}^*}(v)$ and $D\Pi_{\mathcal{K}^*}(z)$ are always computed (*i.e.*, it is never the case that just the projection is computed, and vice versa).

# Experimentation considerations
Here are my current ideas for experiments we want to perform (or put another way, the ways we are going benchmark the performance of `diffqcp`.)
- `diffcp` vs. `diffqcp` (CPU only. **`diffqcp` needs to be faster.**)
- `diffqcp` using $P$ vs. `diffqcp` ignoring $P$ <=> reformulated as (non-quadratic) cone program. (also compare to diffcp. This will potentially help gauge if `diffqcp` is faster than `diffcp` due to better implementation, or because of the mathematical structure. Assertion would be that if `diffqcp` without $P$ is faster than `diffcp`, then implementation should take a lot of credit for speed. Less conclusive if `diffqcp` faster than `diffcp` faster than `diffqcp` ignoring $P$)
- `diffcp` vs. `diffqcp` on GPU (not including transfer times)
- `diffqcp` on CPU versus `diffqcp` load and run on GPU

What we wish to show/our selling points
- so firstly, if solving opt. problems on GPU already, diffqcp GPU support (not even fancy accleration) is desired so you don't move data off device.
- however, right now the norm for solving opt. problems is on CPU, thus for `diffqcp` to be a true replacement for `diffcp`, we need to outperform `diffcp` on the CPU (although, arguments can and probably should be made that not having to formulate a problem 2x -- once to solve via a QCP solver and once to differentiate using a non-quadratic supported differentiation engine counts toward something I think)
- There's then the question of whether the mathematical structure of diffqcp has natural computational benefits like QCPs. (Although I think whether it does or doesn't won't matter much to people -- the bottom line is whether diffqcp is faster than diffcp, and it doesn't matter too much why that is.)

Thoughts
- I'm very confident that we can outperform `diffcp` when $\mathcal{K}$ does not include any exponential or power cones (well, `diffcp` doesn't even support power cones, so just exponential cones). I'm thinking that the torch implementation of the projection onto the exponential cone will be a good bit slower than the C++ implementation.


# Numerical enhancements

## Removing redundancy

1. In both `diffcp` and `diffqcp` there are repeated computations of $\Pi_{\mathcal{K}^*}(v)$ and $D\Pi_{\mathcal{K}^*}(z)$. Consequently, the lowest hanging fruit for improving our flop count is removing all redundant function computation.
2. There is also redundant computation when considering both $\Pi_{\mathcal{K}^*}(v)$ and $D\Pi_{\mathcal{K}^*}(z)$; *i.e.*, computations done when finding the projection can and should be reused for computing the derivative of the projection. (A truly egregious example of this type of redundant computation is computing the eigendecomposition 2x between projecting onto a PSD cone and then computing the derivative of this projection.) **Importantly,** beacuse we know we always compute the projection onto the dual cone and the derivative of the projection onto the dual cone, to remove redundant computations we don't even have to cache results. Instead, the computations for $\Pi_{\mathcal{K}^*}(v)$ and $D\Pi_{\mathcal{K}^*}(z)$ should be coupled.

## GPU acceleration

### Good hygiene
1. Allocating as much memory up front and then passing references to the spots in memory the computations should fill in (CuClarabel style)

### Parallelization
1. Zero cones and nonnegative cone computations are GPU accelerated (computed "in parallel") as they are implemented now.
2. SOC computations could theoretically be accelerated via
    - using preprocessing step from CuClarabel
    - **and** doing an additional preprocessing step which groups the cones according to the branching logic. (I'm going back and forth with how much computational upside there is to this.)
3. Exponential cone and power cone computations really cannot be accelerated due to the prevalent branching logic
4. PSD cone can only be accelerated if there are many PSD cones of the same size, in which case we can do a batched eigendecomposition...which seems like it exists, but there are some caveats including synchronization with CPU

# Corresponding code diffs
- Since (as previously stated) the cone information is not carried throughout the life of the program for long, I don't think there's a need from a computational perspective for creating cone classes. Really all that is needed is to
    1. upfront allocate memory for the projection $\Pi_{\mathcal{K}^*}(z)$ and then pass references to indices in this memory for the projection calls to fill in (so similar to CuClarabel data structures)
    2. Just create new functions which couple the projection and (abstract linop) derivative computations.
- That said, I'm not opposed to going the class route -- from a "cleanliness" perspective I actually want to do this. So the question here is **would creating simple cone wrapper classes purely for "cleanliness" have a noticeable computational cost?**
- On a different topic, how much would torch.compile help us?...I don't have experience with it, but from what I was reading, it seems like one of its best practices was not to use it on functions whose input sizes change...so could this work for the exponential and power cone to help us compete with `diffcp`'s C++ backend?