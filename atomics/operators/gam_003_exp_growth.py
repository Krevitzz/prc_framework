"""
atomics/operators/gam_003_exp_growth.py

GAM-003 : Croissance exponentielle pointwise
Forme   : T_{n+1}[i,...] = T_n[i,...] · exp(γ)
Famille : markovian

Comportement attendu :
  - Explosion rapide (toutes valeurs → ±∞)
  - Conçu pour tester la robustesse du framework aux explosions
  - NaN passthrough attendu après overflow float32
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-003',
    'family'         : 'markovian',
    'rank_constraint': None,
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
        state  : Tenseur courant, tout rang
        params : {'gamma': float}  défaut 0.05  — doit être > 0
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state — valeurs croissantes, explosion garantie
    """
    gamma = params.get('gamma', 0.05)
    return state * jnp.exp(gamma)
