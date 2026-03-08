"""
atomics/operators/gam_007_sliding_avg.py

GAM-007 : Régulation par moyenne glissante 8-connexe
Forme   : T_{n+1}[i,j] = (1-ε)·T_n[i,j] + ε·mean(voisins_8)
Famille : markovian
Rang    : 2 uniquement

Comportement attendu :
  - Lissage progressif des structures
  - Homogénéisation plus douce que diffusion pure
  - Conditions périodiques (roll)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-007',
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
        params : {'epsilon': float}  défaut 0.1, dans [0, 1]
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state

    Raises:
        ValueError : si state.ndim != 2

    Note JAX :
        np.roll(state, (1,1), axis=(0,1)) non supporté directement.
        Diagonaux décomposés en deux jnp.roll chainés.
    """
    if state.ndim != 2:
        raise ValueError(
            f"GAM-007 applicable rang 2 uniquement, reçu ndim={state.ndim}"
        )
    epsilon = params.get('epsilon', 0.1)

    # Voisins orthogonaux
    n0p = jnp.roll(state,  1, axis=0)   # nord
    n0m = jnp.roll(state, -1, axis=0)   # sud
    n1p = jnp.roll(state,  1, axis=1)   # est
    n1m = jnp.roll(state, -1, axis=1)   # ouest

    # Voisins diagonaux — deux rolls chainés
    npp = jnp.roll(jnp.roll(state,  1, axis=0),  1, axis=1)
    npm = jnp.roll(jnp.roll(state,  1, axis=0), -1, axis=1)
    nmp = jnp.roll(jnp.roll(state, -1, axis=0),  1, axis=1)
    nmm = jnp.roll(jnp.roll(state, -1, axis=0), -1, axis=1)

    neighbors_sum = n0p + n0m + n1p + n1m + npp + npm + nmp + nmm
    mean_neighbors = neighbors_sum / 8.0

    return (1.0 - epsilon) * state + epsilon * mean_neighbors
