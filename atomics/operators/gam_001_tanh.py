"""
atomics/operators/gam_001_tanh.py

GAM-001 : Saturation pure pointwise (tanh)
Forme   : T_{n+1}[i,...] = tanh(β · T_n[i,...])
Famille : markovian

Comportement attendu :
  - Convergence rapide vers [-1,1] (saturation)
  - Attracteurs possiblement triviaux selon β
  - Rang-agnostique : opère pointwise

Rôle v7 :
  F1 — rang effectif décroît (saturation → perte rang)
  F2 — S_VN décroît (compression information)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-001',
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
        params : {'beta': float}  défaut 2.0
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state, valeurs dans (-1,1)
    """
    beta = params.get('beta', 2.0)
    return jnp.tanh(beta * state)
