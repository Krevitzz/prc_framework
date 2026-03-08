"""
atomics/operators/gam_005_harmonic.py

GAM-005 : Oscillateur harmonique linéaire
Forme   : T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}
Famille : non_markovian

Comportement attendu :
  - Oscillations périodiques (période 2π/ω)
  - Conservation énergie théorique
  - Pas d'émergence de structure

TODO: NON_MARKOVIAN
  Forme complète nécessite carry étendu (state, prev_state, key).
  Implémentation actuelle : fallback markovien cos(ω)·T_n
  (équivalent premier pas, sin(ω)·T_{n-1} ignoré).
  Brancher quand hub_running supporte carry étendu.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-005',
    'family'         : 'non_markovian',
    'rank_constraint': None,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : True,   # carry étendu requis — TODO: NON_MARKOVIAN
}

_DEFAULT_OMEGA = 0.7853981633974483  # π/4


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Fallback markovien : T_{n+1} = cos(ω)·T_n

    Forme complète (TODO: NON_MARKOVIAN) :
        T_{n+1} = cos(ω)·T_n - sin(ω)·T_{n-1}

    Args:
        state  : Tenseur courant, tout rang
        params : {'omega': float}  défaut π/4
        key    : Ignorée

    Returns:
        jnp.ndarray même shape que state
    """
    omega = params.get('omega', _DEFAULT_OMEGA)
    return jnp.cos(omega) * state
