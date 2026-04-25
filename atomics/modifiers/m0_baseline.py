"""
atomics/modifiers/m0_baseline.py

M0 : Modifier identité (baseline)
Transformation : D' = D

Rôle : placeholder structurel — garantit qu'une composition
       a toujours un modifier, même sans transformation.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id': 'M0',
    'family': 'modifier',
    'stochastic': False,
    'differentiable': True,
    'rank_constraint': None,
    'non_markovian': False,
}

def apply(state: jnp.ndarray, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Retourne state inchangé.

    Args:
        state  : Tenseur d'entrée, tout rang
        params : Ignoré
        key    : Ignorée (stochastic: False)

    Returns:
        state inchangé — même array, même shape, même dtype
    """
    return state