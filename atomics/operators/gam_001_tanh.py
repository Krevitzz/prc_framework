"""
atomics/operators/gam_001_tanh.py

GAM-001 : Saturation pure pointwise (tanh)
Forme   : T_{n+1}[i,...] = tanh(β · T_n[i,...])
Famille : markovian

Comportement attendu :
  - Convergence rapide vers [-1,1] (saturation)
  - Attracteurs possiblement triviaux selon β
  - Rang-agnostique : opère pointwise
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
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,  # ignoré
    params     : dict,
    key        : jax.Array,    # ignoré
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur courant, tout rang
        prev_state : Ignoré (présent pour signature unifiée)
        params     : {'beta': float}  défaut 2.0
        key        : Ignoré

    Returns:
        jnp.ndarray même shape que state, valeurs dans (-1,1)
    """
    beta = params.get('beta', 2.0)
    return jnp.tanh(beta * state)