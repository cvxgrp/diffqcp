"""
Notation:
- `x` is the 
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
from jaxtyping import Array, Float

from .canonical import AbstractConeProjector

if jax.config.jax_enable_x64:
    TOL = 1e-12
else:
    TOL = 1e-6

def _pow_calc_xi(
    ri: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """x_i from eq 4. from Hien paper"""
    x = 0.5 * (x + jnp.sqrt(x*x + 4. * alpha * (abs_z - ri) * ri))
    return jnp.maximum(x, TOL)


def _gi(
    ri: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """gi from diffqcp paper."""
    return 2. * _pow_calc_xi(ri, x, abs_z, alpha) - x


def _pow_calc_f(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """Phi from Hien paper."""
    return xi**alpha + yi**(1.-alpha) - ri


def _pow_calc_dxi_dr(
    ri: Float[Array, ""],
    xi: Float[Array, ""],
    x: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    """
    `xi` is an iterate toward the projection of `x` or `y` in `(x, y, z)` toward
    the first element or second element, respectively, in `proj(v)`.
    """
    return alpha * (abs_z - 2.0 * ri) / (2.0 * xi - x)


def _pow_calc_fp(
    xi: Float[Array, ""],
    yi: Float[Array, ""],
    dxidri: Float[Array, ""],
    dyidri: Float[Array, ""],
    alpha: Float[Array, ""]
) -> Float[Array, ""]:
    alphac = 1 - alpha
    return xi**alpha + yi**alphac * (alpha * dxidri / xi + alphac * dyidri / yi) - 1


def _in_cone(
    x: Float[Array, ""],
    y: Float[Array, ""],
    abs_z: Float[Array, ""],
    alpha: Float[Array, ""]
) -> bool:
    return (x >= 0 and y >= 0 and TOL + x**alpha * y**(1-alpha) >= abs_z)


def _in_polar_cone(
    u: Float[Array, ""],
    v: Float[Array, ""],
    abs_w: Float[Array, ""],
    alpha: Float[Array, ""]
) -> bool:
    return (u <= 0 and v <= 0 and TOL + jnp.pow(-u, alpha) * jnp.pow(-v, 1. - alpha) >=
            abs_w * alpha**alpha + jnp.pow(1. - alpha, 1. - alpha))


class PowerConeProjector(AbstractConeProjector):

    alpha: Float[Array, ""]
    onto_dual: bool
    
    def proj_dproj(self, v):
        """
        probably need to return a matrix linear operator as derivative?
        """
        x, y, z = v
        abs_z = jnp.abs(z)

        def identity_case():
            return (
                v, lx.MatrixLinearOperator(jnp.eye(3, dtype=x.dtype, device=x.device))
            )

        def zero_case():
            return (
                jnp.zeros_like(v), lx.MatrixLinearOperator(jnp.zeros((3, 3), dtype=x.dtype, device=x.device))
            )

        def z_zero_case():
            J = jnp.zeros((3, 3), dtype=v.dtype)
            J = J.at[0, 0].set(0.5 * (jnp.sign(x) + 1.0))
            J = J.at[1, 1].set(0.5 * (jnp.sign(y) + 1.0))

            def case1():  # (x > 0 and y0 < 0 and a_device > 0.5) or (y0 > 0 and x < 0 and alpha < 0.5)
                return 1.0

            def case2():  # (x > 0 and y < 0 and alpha < 0.5) or (y > 0 and x < 0 and alpha > 0.5)
                return 0.0

            def case3():  # a_device == 0.5 and x0 > 0 and y0 < 0
                return x / (2 * jnp.abs(y) + x)

            def case4():
                return y / (2 * jnp.abs(x) + y)

            cond1 = ((x > 0) & (y < 0) & (self.alpha > 0.5)) | ((y > 0) & (x < 0) & (self.alpha < 0.5))
            cond2 = ((x > 0) & (y < 0) & (self.alpha < 0.5)) | ((y > 0) & (x < 0) & (self.alpha > 0.5))
            cond3 = (self.alpha == 0.5) & (x > 0) & (y < 0)

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
            return proj_v, lx.MatrixLinearOperator(J)
            

        def _solve_while_body(xj, yj, rj):
            # NOTE(quill): we're purposefully using both `i` and `j`.
            #   The former (which is in the function names) is denoting
            #   an element in a vector while the latter is being used to denote
            #   an interation count.
            xj = _pow_calc_xi(rj, x, abs_z, self.alpha)
            yj = _pow_calc_xi(rj, y, abs_z, 1.0 - self.alpha)
            fj = _pow_calc_f(xj, yj, rj, self.alpha)
            
            dxdr = _pow_calc_dxi_dr(rj, xj, x, abs_z, self.alpha)
            dydr = _pow_calc_dxi_dr(rj, yj, y, abs_z)
        
        def solve_case():
            xj = 0
            yj = 0
            rj = abs_z / 2



        return jax.lax.cond(_in_cone(x, y, abs_z, self.alpha),
                            identity_case,
                            lambda: jax.lax.cond(
                                _in_polar_cone(x, y, abs_z, self.alpha),
                                zero_case,
                                lambda: jax.lax.cond(
                                    abs_z <= TOL,
                                    z_zero_case,
                                    solve_case
                                )))