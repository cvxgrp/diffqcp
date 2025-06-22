import equinox as eqx
from jaxtyping import Float, Array
from jax.experimental.sparse import BCOO, BCSR


class QCP(eqx.Module):
    P: Float[Array | BCOO | BCSR, "*batch n n"]
    A: Float[Array | BCOO | BCSR, "*batch m n"]
    q: Float[Array, "*batch n"]
    b: Float[Array, "*batch m"]
