"""
atomics/modifiers/m2_uniform_noise.py

M2 : Bruit uniforme additif sur D avant le scan
Forme   : D' = D + U[-amplitude, +amplitude]
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id': 'M2',
    'family': 'modifier',
    'stochastic': True,
    'differentiable': True,
    'rank_constraint': None,
    'non_markovian': False,
}

def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur D initial, tout rang
        params : {'amplitude': float}  défaut 0.1
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray même shape que state — D perturbé
    """
    amplitude = params.get('amplitude', 0.1)
    noise = jax.random.uniform(key, state.shape, minval=-amplitude, maxval=amplitude)
    return state + noise