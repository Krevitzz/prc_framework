"""
atomics/operators/gam_006_memory_tanh.py

GAM-006 : Saturation avec mémoire ordre-1
Forme   : T_{n+1} = tanh(β·T_n + α·(T_n - T_{n-1}))
Famille : non_markovian

Comportement attendu :
  - Inertie temporelle (convergence plus lente que markovien)
  - Attracteurs non-triviaux possibles
  - Premier pas : comportement markovien (fallback)

TODO: NON_MARKOVIAN
  Forme complète nécessite carry étendu (state, prev_state, key).
  Implémentation actuelle : fallback markovien tanh(β·T_n)
  (équivalent premier pas, α·(T_n - T_{n-1}) ignoré).
  Brancher quand hub_running supporte carry étendu.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-006',
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
    Fallback markovien : T_{n+1} = tanh(β·T_n)

    Forme complète (TODO: NON_MARKOVIAN) :
        velocity = T_n - T_{n-1}
        T_{n+1}  = tanh(β·T_n + α·velocity)

    Args:
        state  : Tenseur courant, tout rang
        params : {'beta': float, 'alpha': float}  défauts 1.0, 0.3
        key    : Ignorée

    Returns:
        jnp.ndarray même shape que state, valeurs dans (-1,1)
    """
    beta = params.get('beta', 1.0)
    return jnp.tanh(beta * state)
