"""
atomics/operators/gam_010_mult_noise.py

GAM-010 : Bruit multiplicatif avec saturation
Forme   : T_{n+1} = tanh(T_n · (1 + σ·ε)),  ε ~ N(0,1)
Famille : stochastic
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-010',
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
        params     : {'sigma': float}  défaut 0.05
        key        : PRNGKey JAX
    """
    sigma = params.get('sigma', 0.05)

    noise  = 1.0 + jax.random.normal(key, state.shape) * sigma
    return jnp.tanh(state * noise)