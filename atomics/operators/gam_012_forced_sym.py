"""
atomics/operators/gam_012_forced_sym.py

GAM-012 : Préservation symétrie forcée
Forme   : T_{n+1} = (F(T_n) + F(T_n)ᵀ) / 2,  F = tanh(β·)
Famille : structural
Rang    : 2 uniquement

Comportement attendu :
  - Symétrie garantie à chaque itération
  - Robustesse au bruit asymétrique
  - Convergence similaire GAM-001
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
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur rang 2 (n, n)
        params : {'beta': float}  défaut 2.0
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n, n) symétrique, valeurs dans (-1,1)

    Raises:
        ValueError : si state.ndim != 2
    """
    if state.ndim != 2:
        raise ValueError(
            f"GAM-012 applicable rang 2 uniquement, reçu ndim={state.ndim}"
        )
    beta = params.get('beta', 2.0)
    F    = jnp.tanh(beta * state)
    return (F + F.T) / 2.0
