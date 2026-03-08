"""
atomics/D_encodings/sym_003_random_gaussian.py

SYM-003 : Matrice symétrique aléatoire gaussienne
Forme   : A = (B + Bᵀ)/2, B_ij ~ N(0, σ)
Rang    : 2
Propriétés : symétrique, non bornée a priori
Usage   : Test continuité, distribution normale
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'SYM-003',
    'rank'      : 2,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'sigma': float}  défaut 0.3
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) symétrique
    """
    sigma = float(params.get('sigma', 0.3))
    B     = jax.random.normal(key, (n_dof, n_dof)) * sigma
    return (B + B.T) / 2.0
