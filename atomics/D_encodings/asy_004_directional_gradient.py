"""
atomics/D_encodings/asy_004_directional_gradient.py

ASY-004 : Gradient directionnel + bruit
Forme   : A_ij = gradient·(i-j)/n_dof + ε,  ε ~ N(0, noise)
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'ASY-004',
    'rank'      : 2,
    'stochastic': True,
    'jax_vmappable': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'gradient': float, 'noise': float}
                  défauts : gradient=0.1, noise=0.2
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) asymétrique avec structure directionnelle
    """
    gradient = params.get('gradient', 0.1)
    noise    = params.get('noise',    0.2)

    i    = jnp.arange(n_dof, dtype=jnp.float32)
    j    = jnp.arange(n_dof, dtype=jnp.float32)
    base = gradient * (i[:, None] - j[None, :]) / n_dof

    eps = jax.random.normal(key, (n_dof, n_dof)) * noise
    return base + eps
