"""
atomics/D_encodings/asy_003_antisymmetric.py

ASY-003 : Matrice antisymétrique aléatoire
Forme   : A = (B - Bᵀ)/2,  A_ij = -A_ji,  A_ii = 0
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'ASY-003',
    'rank'      : 2,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {} — aucun param
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) antisymétrique (A + Aᵀ = 0)
    """
    B = jax.random.uniform(key, (n_dof, n_dof), minval=-1.0, maxval=1.0)
    return (B - B.T) / 2.0
