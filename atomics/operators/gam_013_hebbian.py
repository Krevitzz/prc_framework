"""
atomics/operators/gam_013_hebbian.py

GAM-013 : Renforcement hebbien local
Forme   : T_{n+1} = T_n + η·(T_n @ T_n)
Famille : structural
Rang    : 2 carré uniquement
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-013',
    'family'         : 'structural',
    'rank_constraint': 'square',
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
        state      : Tenseur rang 2 carré (n, n)
        prev_state : Ignoré
        params     : {'eta': float}  défaut 0.01
        key        : Ignoré
    """
    eta = params.get('eta', 0.01)
    return state + eta * (state @ state)