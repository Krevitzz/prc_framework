"""
atomics/operators/gam_004_exp_decay.py

GAM-004 : Décroissance exponentielle pointwise
Forme   : T_{n+1}[i,...] = T_n[i,...] · exp(-γ)
Famille : markovian

Comportement attendu :
  - Convergence garantie vers zéro
  - Attracteur trivial (norme → 0)
  - Linéaire → jacfwd analytique : J = exp(-γ)·I

Rôle v7 :
  F4 — trace_J = n_dof · exp(-γ)  (analytique, validation Hutchinson)
  Null model A3 : S_VN conservé (scaling uniforme)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-004',
    'family'         : 'markovian',
    'rank_constraint': None,
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
        state  : Tenseur courant, tout rang
        params : {'gamma': float}  défaut 0.05
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state
    """
    gamma = params.get('gamma', 0.05)
    return state * jnp.exp(-gamma)
