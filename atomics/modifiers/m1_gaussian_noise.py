"""
atomics/modifiers/m1_gaussian_noise.py

M1 : Bruit gaussien additif sur D avant le scan
Forme   : D' = D + N(0, σ)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id': 'M1',
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
        params : {'sigma': float}  défaut 0.05
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray même shape que state — D perturbé
    """
    sigma = params.get('sigma', 0.05)
    noise = jax.random.normal(key, state.shape) * sigma
    return state + noise