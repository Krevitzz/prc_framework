"""
atomics/operators/gam_006_memory_tanh.py

GAM-006 : Saturation avec mémoire ordre-1
Forme   : T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))
Famille : non_markovian
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-006',
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
        params     : {'beta': float, 'alpha': float}  défauts 1.0, 0.3
        key        : Ignoré
    """
    beta  = params.get('beta',  1.0)
    alpha = params.get('alpha', 0.3)
    velocity = state - prev_state
    return jnp.tanh(beta * state + alpha * velocity)