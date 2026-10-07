# Cone conventions

What every projector in `diffqcp/cones/` assumes. Several bugs fixed in the
productionization effort were convention mix-ups (PSD triangle order, alpha
vs. 1 - alpha, dual-cone signs), so check new code against this page.

## The cone vector

`ProductConeProjector` takes an SCS-style cone dictionary and lays the blocks
out in this order (`canonical._CONES`):

| key  | cone                         | value in the dict           | block size     |
|------|------------------------------|-----------------------------|----------------|
| `z`  | zero cone {0}                | int (dimension)             | value          |
| `l`  | nonnegative orthant          | int (dimension)             | value          |
| `q`  | second-order cones           | list of dims                | sum            |
| `s`  | PSD cones                    | list of matrix sizes k      | sum k(k+1)/2   |
| `ep` | exponential cones            | int (count)                 | 3 x count      |
| `ed` | dual exponential cones       | int (count)                 | 3 x count      |
| `p`  | power cones                  | list of alphas in (-1, 1)   | 3 x len        |

This is SCS's order, and also the order CVXPY uses for Clarabel data.

## PSD vectorization

A symmetric k x k matrix X is stored as its **lower triangle, column-major**
(equivalently the upper triangle, row-major), with off-diagonal entries scaled
by sqrt(2) so that `<vec(X), vec(Y)> = trace(X Y)`:

    vec(X) = (X11, sqrt2 X21, ..., sqrt2 Xk1, X22, sqrt2 X32, ..., Xkk)

This matches SCS and diffcp. **Clarabel uses the upper triangle,
column-major**, so data from `get_problem_data(cvx.CLARABEL)` (rows of A, b)
and Clarabel solutions (y, s) must be permuted with
`diffqcp.clarabel_to_scs_permutation(cone_dims)` first.

The projection's Jacobian (`_PSDConeProjector`) uses the eigendecomposition
X = Q diag(lambda) Q^T, ascending eigenvalues, k = index of the last negative
one, and

    DPi(X)[dX] = Q (B o (Q^T dX Q)) Q^T,
    B_ij = 1                            if i, j > k
           lambda_j / (lambda_j - lambda_i)   if i <= k < j   (and symmetric)
           0                            if i, j <= k

The positive block is all ones, not the identity.

## Exponential cone

K_exp = cl{(r, s, t) : s > 0, s exp(r / s) <= t}, in that coordinate order.
Its polar is -K_exp^*. Projections onto the dual cone go through Moreau,
Pi_{K*}(v) = v + Pi_K(-v). The Jacobian has these regions: v in K (identity),
v in the polar (zero), r < 0 and s < 0 (projection (r, 0, max(t, 0))), the
face s = 0 of the projection (also reached from small r > 0 next to a large
negative s), and the general case (a 4 x 4 inverse).

## Power cone

K_alpha = {(x, y, z) : x, y >= 0, x^alpha y^(1 - alpha) >= |z|}, alpha in (0, 1).

- **Dual cones (SCS convention):** a *negative* alpha in the `p` list means the
  dual cone K_|alpha|^*. `PowerConeProjector(alphas, onto_dual=True)` flips
  this for every entry (the cone program needs Pi_{K*}(y - s)).
- Dual projections use Moreau, Pi_{K*}(v) = v + Pi_K(-v); with the
  implementation's sign flip `batch = -v`, that is `-batch + Pi_K(batch)`.
  Its Jacobian is I - DPi_K(-v).
- Polar cone: x, y <= 0 and (-x)^alpha (-y)^(1 - alpha) >= |z| alpha^alpha
  (1 - alpha)^(1 - alpha). Note the **product**.
- In the Newton solve and Jacobian, every quantity attached to y uses
  1 - alpha where the x one uses alpha (e.g. `gy = _gi(r, y, |z|, 1 - alpha)`),
  and the x-y cross term carries alpha (1 - alpha). Mistakes here are invisible
  at alpha = 0.5, so always test with alpha far from 0.5.

## Testing a projector

- Projection: against CVXPY + SCS at tight tolerance (SCS reaches ~1e-12 on
  these small problems; Clarabel only ~1e-6 for exp/pow).
- Jacobian: against `jax.jvp` of the projection where that is NaN-free
  (zero, nonneg, SOC, PSD), otherwise central finite differences, skipping
  points where the one-sided differences disagree (kinks).
- Always include indefinite points (PSD), alphas far from 0.5 and both
  primal and dual cones (power), and points whose projection lies on a face.
