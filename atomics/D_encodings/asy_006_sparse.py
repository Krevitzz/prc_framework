"""
atomics/D_encodings/asy_006_sparse.py

ASY-006 : Matrice sparse asymétrique
Forme   : A_ij ~ U[-1,1] avec probabilité density, 0 sinon
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'ASY-006',
    'rank'      : 2,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'density': float}  défaut 0.2, dans (0,1]
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) sparse asymétrique
    """
    density = params.get('density', 0.2)

    k1, k2 = jax.random.split(key)
    mask    = jax.random.uniform(k1, (n_dof, n_dof)) < density
    values  = jax.random.uniform(k2, (n_dof, n_dof), minval=-1.0, maxval=1.0)
    return jnp.where(mask, values, 0.0)
