"""
atomics/D_encodings/asy_005_circulant.py

ASY-005 : Matrice circulante décalée
Forme   : A_ij = f(j-i mod n_dof), f décroissante positive
          Asymétrie garantie par décalage non-symétrique
Rang    : 2
Stochastique : Non (déterministe)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'ASY-005',
    'rank'      : 2,
    'stochastic': False,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {} — aucun param
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n_dof, n_dof) circulante, asymétrique (A ≠ Aᵀ)

    Construction :
        Première ligne c[k] = exp(-k) pour k=0..n_dof-1
        Chaque ligne i = roll(c, i) → circulante standard
        Asymétrie car exp(-k) ≠ exp(-(n_dof-k)) pour k > 0
    """
    k   = jnp.arange(n_dof, dtype=jnp.float32)
    c   = jnp.exp(-k)   # décroissante → asymétrie garantie

    rows = [jnp.roll(c, i) for i in range(n_dof)]
    return jnp.stack(rows, axis=0)
