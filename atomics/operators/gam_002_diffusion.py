"""
atomics/operators/gam_002_diffusion.py

GAM-002 : Diffusion pure (Laplacien discret)
Forme   : T_{n+1}[i,j] = T_n[i,j] + α·(∑_voisins - 4·T_n[i,j])
Famille : markovian
Rang    : 2 uniquement

Comportement attendu :
  - Homogénéisation progressive
  - Perte totale de diversité
  - Stabilité Von Neumann : α < 0.25
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-002',
    'family'         : 'markovian',
    'rank_constraint': 2,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : False,
}


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur rang 2 (n, m)
        params : {'alpha': float}  défaut 0.05, doit être dans (0, 0.25)
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state

    Raises:
        ValueError : si state.ndim != 2
    """
    if state.ndim != 2:
        raise ValueError(
            f"GAM-002 applicable rang 2 uniquement, reçu ndim={state.ndim}"
        )
    alpha = params.get('alpha', 0.05)

    laplacian = (
        jnp.roll(state,  1, axis=0) +
        jnp.roll(state, -1, axis=0) +
        jnp.roll(state,  1, axis=1) +
        jnp.roll(state, -1, axis=1) -
        4.0 * state
    )
    return state + alpha * laplacian
