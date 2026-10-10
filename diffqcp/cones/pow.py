"""Projection onto (products of) power cones, or their duals, and its derivative.

The projection follows Hien's Newton method on the scalar `r = |z|` of the
projected point (see `_pow_calc_xi`); the Jacobian is formed in closed form.
Negative alphas denote dual cones (SCS convention); see `CONVENTIONS.md`.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Bool, Float

from diffqcp.linops import _dense_from_mv

from .abstract_projector import AbstractConeProjector

TOL = 1e-12 if jax.config.read("jax_enable_x64") else 1e-06

MAX_ITER = 20

def _pow_calc_xi(
    ri: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """x_i from eq. 4 of the Hien paper: the projected `x` coordinate for a given `r`.

    Here `x` is the original coordinate; the result is clipped below at `TOL`.
    Also used for the `y` coordinate with `1 - alpha` in place of `alpha`.
    """
    x = 0.5 * (x + jnp.sqrt(x*x + 4. * alpha * (abs_z - ri) * ri))
    return jnp.maximum(x, TOL)


def _gi(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """g_i from the diffqcp paper: `2 x_i(r) - x = sqrt(x^2 + 4 alpha (|z| - r) r)`.

    Note that the `xi` argument is the *original* coordinate (`x` or `y`), not an
    iterate.
    """
    return 2. * _pow_calc_xi(ri, xi, abs_z, alpha) - xi


def _pow_calc_f(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """Phi from the Hien paper, `x_i^alpha y_i^(1 - alpha) - r`; its root is the optimal `r`."""
    return xi**alpha * yi**(1.-alpha) - ri


def _pow_calc_dxi_dr(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """Derivative of `_pow_calc_xi` with respect to `r`, `alpha (|z| - 2 r) / (2 x_i - x)`.

    `xi` is the current iterate for the projected `x` (or `y`) coordinate and `x`
    the original coordinate.
    """
    return alpha * (abs_z - 2.0 * ri) / (2.0 * xi - x)


def _pow_calc_fp(
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    dxidri: Float[Array, ""],
    dyidri: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """Derivative of Phi (`_pow_calc_f`) with respect to `r`, for the Newton step."""
    alphac = 1 - alpha
    return (xi**alpha) * (yi**alphac) * (alpha * dxidri / xi + alphac * dyidri / yi) - 1.0


def _in_cone(
    x: Float[Array, ""],
    y: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Bool[Array, ""]:
    """Whether `(x, y, z)` is in the power cone K_alpha, up to `TOL`."""
    return jnp.logical_and(x >= 0,
                           jnp.logical_and(y >= 0,
                                           TOL + x**alpha * y**(1-alpha) >= abs_z))


def _in_polar_cone(
    u: Float[Array, ""],
    v: Float[Array, ""],
    abs_w: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Bool[Array, ""]:
    """Whether `(u, v, w)` is in the polar of the power cone K_alpha, up to `TOL`."""
    # Polar of the power cone: u, v <= 0 and (-u/a)^a (-v/(1-a))^(1-a) >= |w|,
    # i.e. (-u)^a (-v)^(1-a) >= |w| a^a (1-a)^(1-a). (This used to *add* the
    # (1-a)^(1-a) factor, misclassifying ~10% of random points.)
    return jnp.logical_and(u <= 0,
                           jnp.logical_and(v <= 0,
                                           TOL + jnp.pow(-u, alpha) * jnp.pow(-v, 1. - alpha) >=
                                            abs_w * alpha**alpha * jnp.pow(1. - alpha, 1. - alpha)))


def _proj_dproj(
    v: Float[Array, " 3"],
    alpha: Float[Array, ""]
) -> tuple[Float[Array, " 3"], Float[Array, "3 3"]]:
    """Project `v = (x, y, z)` onto the power cone K_alpha and form the 3 x 3 Jacobian.

    Cases: `v` in the cone (identity), `v` in the polar cone (zero), `|z| <= TOL`
    (projection `(max(x, 0), max(y, 0), 0)`), and otherwise Hien's Newton solve
    for `r = |z_proj|` followed by the closed-form Jacobian.
    """
    x, y, z = v
    abs_z = jnp.abs(z)

    def identity_case():
        return (
            v, jnp.eye(3, dtype=x.dtype)
        )

    def zero_case():
        return (
            jnp.zeros_like(v), jnp.zeros((3, 3), dtype=x.dtype)
        )

    def z_zero_case():
        J = jnp.zeros((3, 3), dtype=v.dtype)
        J = J.at[0, 0].set(0.5 * (jnp.sign(x) + 1.0))
        J = J.at[1, 1].set(0.5 * (jnp.sign(y) + 1.0))

        def case1():  # (x > 0 and y < 0 and alpha > 0.5) or (y > 0 and x < 0 and alpha < 0.5)
            return 1.0

        def case2():  # (x > 0 and y < 0 and alpha < 0.5) or (y > 0 and x < 0 and alpha > 0.5)
            return 0.0

        def case3():  # alpha == 0.5 and x > 0 and y < 0
            return x / (2 * jnp.abs(y) + x)

        def case4():  # otherwise (e.g. alpha == 0.5 and y > 0 and x < 0)
            return y / (2 * jnp.abs(x) + y)

        cond1 = ((x > 0) & (y < 0) & (alpha > 0.5)) | ((y > 0) & (x < 0) & (alpha < 0.5))
        cond2 = ((x > 0) & (y < 0) & (alpha < 0.5)) | ((y > 0) & (x < 0) & (alpha > 0.5))
        cond3 = (alpha == 0.5) & (x > 0) & (y < 0)

        J22 = jax.lax.cond(
            cond1, case1,
            lambda: jax.lax.cond(
                cond2, case2,
                lambda: jax.lax.cond(
                    cond3, case3,
                    case4
                )
            )
        )

        J = J.at[2, 2].set(J22)
        proj_v = jnp.array([jnp.maximum(x, 0), jnp.maximum(y, 0), 0.0], dtype=v.dtype)
        return proj_v, J


    def solve_case():

        def _solve_while_body(loop_state):
            # NOTE(quill): we're purposefully using both `i` and `j`.
            #   The former (which is in the function names) is denoting
            #   an element in a vector while the latter is being used to denote
            #   an iteration count.
            loop_state["xj"] = _pow_calc_xi(loop_state["rj"], x, abs_z, alpha)
            loop_state["yj"] = _pow_calc_xi(loop_state["rj"], y, abs_z, 1.0 - alpha)
            fj = _pow_calc_f(loop_state["rj"], loop_state["xj"], loop_state["yj"], alpha)

            dxdr = _pow_calc_dxi_dr(loop_state["rj"], loop_state["xj"], x, abs_z, alpha)
            dydr = _pow_calc_dxi_dr(loop_state["rj"], loop_state["yj"], y, abs_z, 1.-alpha)
            fp = _pow_calc_fp(loop_state["xj"], loop_state["yj"], dxdr, dydr, alpha)

            loop_state["rj"] = jnp.maximum(loop_state["rj"] - fj / fp, 0)
            loop_state["rj"] = jnp.minimum(loop_state["rj"], abs_z)

            loop_state["itn"] += 1
            loop_state["istop"] = jax.lax.select(loop_state["itn"] > MAX_ITER, 2, loop_state["istop"])
            loop_state["istop"] = jax.lax.select(jnp.abs(fj) <= TOL, 1, loop_state["istop"])

            return loop_state

        def condfun(loop_state):
            return loop_state["istop"] == 0

        loop_state = {
            "xj": 0,
            "yj": 0,
            "rj": abs_z / 2,
            "istop": 0,
            "itn": 0
        }

        loop_state = jax.lax.while_loop(condfun, _solve_while_body, loop_state)

        r_star = loop_state["rj"]
        x_star = loop_state["xj"]
        y_star = loop_state["yj"]
        z_star = jax.lax.cond(z < 0, lambda: -r_star, lambda: r_star)
        proj_v = jnp.array([x_star, y_star, z_star])
        a = alpha
        aa = a * a
        ac = 1 - alpha
        acac = ac * ac

        two_r = 2 * r_star
        sign_z = jnp.sign(z)
        gx = _gi(r_star, x, abs_z, a)
        gy = _gi(r_star, y, abs_z, ac)
        frac_x = (a * x) / gx
        frac_y = (ac * y) / gy
        T = - (frac_x + frac_y)
        L = 2 * abs_z - two_r
        L = L / (abs_z + (abs_z - two_r) * (frac_x + frac_y))

        gxgy = gx * gy
        rL = r_star * L
        J = jnp.zeros((3, 3), dtype=x.dtype)
        J = J.at[0, 0].set(0.5 + x / (2 * gx) + (aa * (abs_z - two_r) * rL) / (gx * gx))
        J = J.at[1, 1].set(0.5 + y / (2 * gy) + (acac * (abs_z - two_r) * rL) / (gy * gy))
        J = J.at[2, 2].set(r_star / abs_z + (r_star / abs_z) * T * L)
        J = J.at[0, 1].set(rL * a * ac * (abs_z - two_r) / gxgy)
        J = J.at[1, 0].set(J[0, 1])
        J = J.at[0, 2].set(sign_z * a * rL / gx)
        J = J.at[2, 0].set(J[0, 2])
        J = J.at[1, 2].set(sign_z * ac * rL / gy)
        J = J.at[2, 1].set(J[1, 2])

        return proj_v, J

    return jax.lax.cond(_in_cone(x, y, abs_z, alpha),
                        identity_case,
                        lambda: jax.lax.cond(
                            _in_polar_cone(x, y, abs_z, alpha),
                            zero_case,
                            lambda: jax.lax.cond(
                                abs_z <= TOL,
                                z_zero_case,
                                solve_case
                            )))


def _pow_cone_jacobian_mv(
    dx: Float[Array, " num_cones*3"],
    jacobians: Float[Array, "num_cones 3 3"],
    is_dual: Bool[Array, " num_cones"],
    num_cones: int,
) -> Float[Array, " num_cones*3"]:
    """Apply the power-cone Jacobian blocks to the flat `dx`.

    Block `i` is `jacobians[i]`, or `I - jacobians[i]` where `is_dual[i]` (the
    Moreau form; there `jacobians[i]` was evaluated at the negated point).
    """
    # num cones could be 1.
    dx_batch = jnp.reshape(dx, (num_cones, 3))
    Jdx = eqx.filter_vmap(lambda jac, y: jac @ y,
                            in_axes=(0, 0), out_axes=0)(jacobians, dx_batch)
    mv_dual = dx_batch - Jdx
    mv = jnp.where(is_dual[:, None], mv_dual, Jdx)
    return jnp.ravel(mv)


class _PowerConeJacobianOperator(lx.AbstractLinearOperator):
    """Jacobian of the projection onto a product of power cones (some possibly dual).

    Block diagonal with one dense 3 x 3 block per cone, acting on the flat
    `3 * num_cones` vector: block `i` is `jacobians[i]` for a primal cone and
    `I - jacobians[i]` for a dual one. If the projector was `vmap`ped over a
    batch, `jacobians` is 4D and `mv` takes a 2D `(batch, 3 * num_cones)` input.

    **Arguments:**
    - `jacobians`: the Jacobians of the projections onto the primal cones K_alpha,
        each evaluated at the (sign-flipped for dual cones) point, shape
        `(num_cones, 3, 3)` or `(batch, num_cones, 3, 3)`.
    - `is_dual`: which cones are projected onto their dual.
    """

    jacobians: Float[Array, "*num_batches num_cones 3 3"]
    is_dual: Bool[Array, " num_cones"]
    num_cones: int = eqx.field(static=True)

    def __init__(
        self,
        jacobians: Float[Array, "*num_batches num_cones 3 3"],
        is_dual: Bool[Array, " num_cones"],
    ):
        self.jacobians = jacobians
        self.is_dual = is_dual
        self.num_cones = jnp.size(is_dual)
        ndim = jnp.ndim(jacobians)
        if ndim not in [3, 4]:
            raise ValueError("The `jacobians` argument provided to the `_PowerConeJacobianOperator` "
                             f"is {ndim}D, but it must be 3D or 4D.")

    def mv(self, dx: Float[Array, "*batch num_cones*3"]) -> Float[Array, "*batch num_cones*3"]:
        ndim = jnp.ndim(dx)
        if ndim == 1:
            if jnp.ndim(self.jacobians) == 4:
                raise ValueError("Batched Power cone Jacobians cannot be applied to a 1D input.")

            return _pow_cone_jacobian_mv(dx, self.jacobians, self.is_dual, self.num_cones)
        elif ndim == 2:
            return eqx.filter_vmap(_pow_cone_jacobian_mv,
                                   in_axes=(0, 0, None, None),
                                   out_axes=0)(dx, self.jacobians, self.is_dual, self.num_cones)
        else:
            raise ValueError("The `_PowerConeJacobianOperator` can only be applied to 1D or 2D inputs "
                             f"but the provided vector is {ndim}D.")

    def as_matrix(self) -> Float[Array, "num_cones*3 num_cones*3"]:
        return _dense_from_mv(self)

    def transpose(self) -> "_PowerConeJacobianOperator":
        return self

    # The implicit `None` path is unreachable: `__init__` ensures `jacobians` is 3D or 4D.
    def in_structure(self) -> jax.ShapeDtypeStruct:  # pyright: ignore[reportReturnType]
        ndim = jnp.ndim(self.jacobians)
        shape = jnp.shape(self.jacobians)
        dtype = self.jacobians.dtype

        if ndim == 3:
            # non-batched case
            return jax.ShapeDtypeStruct(shape=(shape[0] * 3,),
                                        dtype=dtype)
        elif ndim == 4:
            # batched case
            return jax.ShapeDtypeStruct(shape=(shape[0], shape[1] * 3),
                                        dtype=dtype)

    def out_structure(self) -> jax.ShapeDtypeStruct:
        return self.in_structure()

@lx.is_symmetric.register(_PowerConeJacobianOperator)
def _(op: _PowerConeJacobianOperator) -> bool:
    return True

class PowerConeProjector(AbstractConeProjector):
    """Projector onto a product of power cones, some of which may be dual cones.

    For `alpha` in (0, 1) the power cone is

        K_alpha = {(x, y, z) : x >= 0, y >= 0, x^alpha y^(1 - alpha) >= |z|},

    and its dual is

        K_alpha^* = {(u, v, w) : u >= 0, v >= 0,
                     (u / alpha)^alpha (v / (1 - alpha))^(1 - alpha) >= |w|}.

    Following SCS, a *negative* entry `-alpha` of `alphas` denotes the dual cone
    K_alpha^*. The input is `len(alphas)` consecutive `(x, y, z)` triples (the `p`
    block of the cone vector). Dual projections use Moreau's decomposition,
    Pi_{K^*}(v) = v + Pi_K(-v), with Jacobian I - DPi_K(-v).

    **Arguments:**
    - `alphas`: one alpha per cone, with `|alpha|` in (0, 1); a negative value
        means the dual cone with parameter `|alpha|`.
    - `onto_dual`: if `True`, every cone is replaced by its dual (so an entry
        `-alpha` then means K_alpha again).
    """

    # NOTE(quill): while similar, this implementation was a bit more challenging than
    # the exponential cone projector implementation as the `cone_dims` dictionary
    # returned by CVXPY has different keys for the exponential cone and its dual, whereas
    # primal vs dual power cone is encoded within the list of `alphas`.

    alphas: Float[Array, " num_cones"]
    num_cones: int = eqx.field(static=True)
    alphas_abs: Float[Array, " num_cones"]
    signs: Float[Array, " num_cones"]
    is_dual: Bool[Array, " num_cones"]

    def __init__(self, alphas: list[float], onto_dual: bool):

        self.alphas = jnp.array(alphas)
        self.num_cones = jnp.size(self.alphas)
        self.is_dual = self.alphas < 0
        self.signs = jnp.where(self.alphas < 0, -1.0, 1.0)
        if onto_dual:
            self.signs = -1.0 * self.signs
            self.is_dual = jnp.logical_not(self.is_dual)
        self.alphas_abs = jnp.abs(self.alphas)

    def proj_dproj(
        self, x: Float[Array, " num_cones*3"]
    ) -> tuple[Float[Array, " num_cones*3"], _PowerConeJacobianOperator]:
        """Project `x` onto the product of (primal or dual) power cones.

        **Arguments:**
        - `x`: a 1D array of length `3 * len(alphas)`. To project a batch, `vmap`
            this method.

        **Returns:**

        A 2-tuple of the projection (same shape as `x`) and its Jacobian, a
        `_PowerConeJacobianOperator`.
        """
        batch = jnp.reshape(x, (self.num_cones, 3))
        # negate points being projected onto dual
        batch = batch * self.signs[:, None]

        proj_primal, jacs = eqx.filter_vmap(_proj_dproj, in_axes=(0, 0), out_axes=(0, 0))(batch, self.alphas_abs)

        # via Moreau: Pi_K^*(v) = v + Pi_K(-v). For dual cones `batch` already
        # holds -v (and `proj_primal` is Pi_K(-v)), so v = -batch.
        proj_dual = -batch + proj_primal

        proj = jnp.where(self.is_dual[:, None], proj_dual, proj_primal)

        return jnp.ravel(proj), _PowerConeJacobianOperator(jacs, self.is_dual)
