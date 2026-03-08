"""
atomics/operators/gam_009_stochastic_tanh.py

GAM-009 : Saturation + bruit additif gaussien
Forme   : T_{n+1} = tanh(β·T_n) + σ·ε,  ε ~ N(0,1)
Famille : stochastic

Comportement attendu :
  - Équilibre stochastique si β fort, σ faible
  - Marche aléatoire bornée si σ fort
  - Distribution stationnaire comme attracteur

Note JAX :
  L'aléatoire est géré par la key splitée dans run_one_jax —
  seed_run dans le YAML fait varier les tirages via _make_run_key.
  Pas de seed_run interne — reproductibilité garantie par la key.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-009',
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
        params : {'beta': float, 'sigma': float}
                  défauts : beta=1.0, sigma=0.01
        key    : PRNGKey JAX — source du bruit gaussien

    Returns:
        jnp.ndarray même shape que state
    """
    beta  = params.get('beta',  1.0)
    sigma = params.get('sigma', 0.01)

    saturated = jnp.tanh(beta * state)

    noise = jax.random.normal(key, state.shape) * sigma
    return saturated + noise
