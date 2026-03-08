"""
atomics/operators/gam_013_hebbian.py

GAM-013 : Renforcement hebbien local
Forme   : T_{n+1}[i,j] = T_n[i,j] + η·Σ_k T_n[i,k]·T_n[k,j]
          = T_n + η·(T_n @ T_n)
Famille : structural
Rang    : 2 carré uniquement

Comportement attendu :
  - Émergence possible de clusters (renforcement mutuel)
  - Instable sans régulation additionnelle (risque explosion)
  - Structures hiérarchiques possibles
  - Coût O(N³) — matmul JAX optimisé

Rôle v7 :
  Test émergence structures — GAM le plus susceptible de produire
  des patterns non-triviaux avec encodings asymétriques.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-013',
    'family'         : 'structural',
    'rank_constraint': 'square',
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
        state  : Tenseur rang 2 carré (n, n)
        params : {'eta': float}  défaut 0.01, dans (0, 0.1]
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n, n)

    Raises:
        ValueError : si state.ndim != 2 ou non carré
    """
    if state.ndim != 2:
        raise ValueError(
            f"GAM-013 applicable rang 2 uniquement, reçu ndim={state.ndim}"
        )
    if state.shape[0] != state.shape[1]:
        raise ValueError(
            f"GAM-013 nécessite matrice carrée, reçu shape={state.shape}"
        )
    eta = params.get('eta', 0.01)
    return state + eta * (state @ state)
