"""
atomics/D_encodings/asy_002_lower_triangular.py

ASY-002 : Matrice triangulaire inférieure aléatoire
Forme   : L_ij ~ U[-1,1] si i ≥ j, 0 sinon
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'ASY-002',
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
        jnp.ndarray (n_dof, n_dof) triangulaire inférieure, valeurs dans [-1,1]
    """
    A = jax.random.uniform(key, (n_dof, n_dof), minval=-1.0, maxval=1.0)
    return jnp.tril(A)
