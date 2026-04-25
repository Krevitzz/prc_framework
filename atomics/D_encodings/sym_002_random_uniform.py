"""
atomics/D_encodings/sym_002_random_uniform.py

SYM-002 : Matrice symétrique aléatoire uniforme
Forme   : A = (B + Bᵀ)/2, B_ij ~ U[-1,1]
Rang    : 2
Propriétés : symétrique, bornes [-1,1]
Usage   : Test diversité maximale, générique
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'SYM-002',
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
        jnp.ndarray (n_dof, n_dof) symétrique, valeurs dans [-1,1]
    """
    B = jax.random.uniform(key, (n_dof, n_dof), minval=-1.0, maxval=1.0)
    return (B + B.T) / 2.0
