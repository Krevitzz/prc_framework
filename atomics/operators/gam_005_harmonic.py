"""
atomics/operators/gam_005_harmonic.py

GAM-005 : Oscillateur harmonique linéaire
Forme   : T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}
Famille : non_markovian
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-005',
    'family'         : 'non_markovian',
    'rank_constraint': None,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : True,
}

_DEFAULT_OMEGA = 0.7853981633974483  # π/4

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
        params     : {'omega': float}  défaut π/4
        key        : Ignoré
    """
    omega = params.get('omega', _DEFAULT_OMEGA)
    return jnp.cos(omega) * state - jnp.sin(omega) * prev_state