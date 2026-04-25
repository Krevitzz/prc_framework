"""
atomics/operators/gam_003_exp_growth.py

GAM-003 : Croissance exponentielle pointwise
Forme   : T_{n+1}[i,...] = T_n[i,...] · exp(γ)
Famille : markovian
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
    return state * jnp.exp(gamma)