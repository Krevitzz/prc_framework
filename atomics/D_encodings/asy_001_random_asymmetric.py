"""
atomics/D_encodings/asy_001_random_asymmetric.py

ASY-001 : Matrice asymétrique aléatoire uniforme
Forme   : A_ij ~ U[-1,1] indépendants
Rang    : 2
Propriétés : asymétrique, bornes [-1,1]
Usage   : Test asymétrie générique
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'ASY-001',
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
        jnp.ndarray (n_dof, n_dof) asymétrique, valeurs dans [-1,1]
    """
    return jax.random.uniform(key, (n_dof, n_dof), minval=-1.0, maxval=1.0)
