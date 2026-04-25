"""
atomics/operators/gam_012_forced_sym.py

GAM-012 : Préservation symétrie forcée
Forme   : T_{n+1} = (F(T_n) + F(T_n)ᵀ) / 2,  F = tanh(β·)
Famille : structural
Rang    : 2 uniquement
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-012',
    'family'         : 'structural',
    'rank_constraint': 2,
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
        state      : Tenseur rang 2 (n, n)
        prev_state : Ignoré
        params     : {'beta': float}  défaut 2.0
        key        : Ignoré
    """
    beta = params.get('beta', 2.0)
    F = jnp.tanh(beta * state)
    # Utilisation de swapaxes pour transposer les deux dernières dimensions
    return (F + F.swapaxes(-2, -1)) / 2.0