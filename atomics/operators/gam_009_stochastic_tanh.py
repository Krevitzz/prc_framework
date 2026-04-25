"""
atomics/operators/gam_009_stochastic_tanh.py

GAM-009 : Saturation + bruit additif gaussien
Forme   : T_{n+1} = tanh(β·T_n) + σ·ε,  ε ~ N(0,1)
Famille : stochastic
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-009',
    'family'         : 'stochastic',
    'rank_constraint': None,
    'differentiable' : False,
    'stochastic'     : True,
    'non_markovian'  : False,
}

def apply(
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,  # ignoré
    params     : dict,
    key        : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur courant, tout rang
        prev_state : Ignoré
        params     : {'beta': float, 'sigma': float}  défauts 1.0, 0.01
        key        : PRNGKey JAX
    """
    beta  = params.get('beta',  1.0)
    sigma = params.get('sigma', 0.01)

    saturated = jnp.tanh(beta * state)
    noise = jax.random.normal(key, state.shape) * sigma
    return saturated + noise