"""
atomics/operators/gam_004_exp_decay.py

GAM-004 : Décroissance exponentielle pointwise
Forme   : T_{n+1}[i,...] = T_n[i,...] · exp(-γ)
Famille : markovian
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-004',
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
        prev_state : Ignoré
        params     : {'gamma': float}  défaut 0.05
        key        : Ignoré
    """
    gamma = params.get('gamma', 0.05)
    return state * jnp.exp(-gamma)