"""
atomics/D_encodings/sym_008_random_clipped.py

SYM-008 : Matrice symétrique gaussienne clippée
Forme   : A = (B+Bᵀ)/2, B_ij ~ N(mean, std), clippé dans [-1,1]
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'SYM-008',
    'rank'      : 2,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'mean': float, 'std': float}  défauts 0.0, 0.3
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof) symétrique, valeurs dans [-1,1]
    """
    mean = float(params.get('mean', 0.0))
    std  = float(params.get('std',  0.3))

    B    = jax.random.normal(key, (n_dof, n_dof)) * std + mean
    sym  = (B + B.T) / 2.0
    return jnp.clip(sym, -1.0, 1.0)
