"""
atomics/operators/gam_010_mult_noise.py

GAM-010 : Bruit multiplicatif avec saturation
Forme   : T_{n+1} = tanh(T_n · (1 + σ·ε)),  ε ~ N(0,1)
Famille : stochastic

Comportement attendu :
  - Amplification différentielle (grandes valeurs amplifiées davantage)
  - Risque avalanche si σ trop fort
  - Structures émergentes possibles

Note JAX :
  seed_run v6 supprimé — aléatoire géré par key JAX splitée dans run_one_jax.
  seed_run dans le YAML fait varier les tirages via _make_run_key.
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
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur courant, tout rang
        params : {'sigma': float}  défaut 0.05
        key    : PRNGKey JAX — source du bruit multiplicatif

    Returns:
        jnp.ndarray même shape que state, valeurs dans (-1,1)
    """
    sigma = params.get('sigma', 0.05)

    noise  = 1.0 + jax.random.normal(key, state.shape) * sigma
    return jnp.tanh(state * noise)
