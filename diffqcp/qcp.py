import equinox as eqx
from jaxtyping import Float, Array
from jax.experimental.sparse import BCSR

class QCP(eqx.Module):
    P: Float[BCSR, "*batch n n"]
    A: Float[BCSR, "*batch m n"]
    q: Float[Array, "*batch n"]
    b: Float[Array, "*batch m"]

