"""
atomics/D_encodings/sym_004_correlation_matrix.py

SYM-004 : Matrice de corrélation
Forme   : C = A·Aᵀ / diag(A·Aᵀ),  A ~ N(0,1)
          Diagonale = 1, hors-diag ∈ [-1,1]
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'SYM-004',
    'rank'      : 2,
    'stochastic': True,
    'jax_vmappable': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {} — aucun param
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) matrice de corrélation
        Diagonale = 1, symétrique, définie positive
    """
    A    = jax.random.normal(key, (n_dof, n_dof))
    C    = A @ A.T
    diag = jnp.sqrt(jnp.diag(C))
    D    = jnp.outer(diag, diag)
    return C / jnp.where(D > 1e-10, D, 1.0)
