"""
atomics/operators/gam_008_diff_memory.py

GAM-008 : Mémoire différentielle avec saturation
Forme   : T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)
          = tanh((1+β+γ)·T_n - γ·T_{n-1})
Famille : non_markovian
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-008',
    'family'         : 'non_markovian',
    'rank_constraint': None,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : True,
}

def apply(
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,
    params     : dict,
    key        : jax.Array,    # ignoré
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur courant (T_n)
        prev_state : Tenseur précédent (T_{n-1})
        params     : {'gamma': float, 'beta': float}  défauts 0.3, 1.0
        key        : Ignoré
    """
    gamma = params.get('gamma', 0.3)
    beta  = params.get('beta',  1.0)
    # Forme équivalente : (1+β+γ)*state - γ*prev_state
    return jnp.tanh((1.0 + beta + gamma) * state - gamma * prev_state)