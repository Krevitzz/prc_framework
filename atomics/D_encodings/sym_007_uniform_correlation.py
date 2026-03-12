"""
atomics/D_encodings/sym_007_uniform_correlation.py

SYM-007 : Matrice de corrélation uniforme
Forme   : C_ij = ρ (i≠j),  C_ii = 1
Rang    : 2
Stochastique : Non (déterministe)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'SYM-007',
    'rank'      : 2,
    'stochastic': False,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'correlation': float}  défaut 0.5, dans (-1/(n_dof-1), 1)
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n_dof, n_dof) — équicorrélation uniforme, définie positive
    """
    rho = params.get('correlation', 0.5)
    C   = jnp.full((n_dof, n_dof), rho)
    return C.at[jnp.arange(n_dof), jnp.arange(n_dof)].set(1.0)
