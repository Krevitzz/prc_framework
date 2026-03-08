"""
atomics/operators/gam_008_diff_memory.py

GAM-008 : Mémoire différentielle avec saturation
Forme   : T_{n+1} = tanh(T_n + γ·(T_n - T_{n-1}) + β·T_n)
          = tanh((1+β+γ)·T_n - γ·T_{n-1})
Famille : non_markovian

Comportement attendu :
  - Oscillations amorties possibles
  - Friction informationnelle
  - Premier pas : saturation simple (fallback)

TODO: NON_MARKOVIAN
  Forme complète nécessite carry étendu (state, prev_state, key).
  Implémentation actuelle : fallback markovien tanh((1+β)·T_n)
  (équivalent premier pas, γ·(T_n - T_{n-1}) ignoré).
  Brancher quand hub_running supporte carry étendu.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-008',
    'family'         : 'non_markovian',
    'rank_constraint': None,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : True,   # carry étendu requis — TODO: NON_MARKOVIAN
}


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Fallback markovien : T_{n+1} = tanh((1+β)·T_n)

    Forme complète (TODO: NON_MARKOVIAN) :
        velocity = T_n - T_{n-1}
        T_{n+1}  = tanh(T_n + γ·velocity + β·T_n)

    Args:
        state  : Tenseur courant, tout rang
        params : {'gamma': float, 'beta': float}  défauts 0.3, 1.0
        key    : Ignorée

    Returns:
        jnp.ndarray même shape que state, valeurs dans (-1,1)
    """
    beta = params.get('beta', 1.0)
    return jnp.tanh((1.0 + beta) * state)
