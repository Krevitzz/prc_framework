"""
atomics/D_encodings/r3_003_local_coupling.py

R3-003 : Couplage local sparse rang 3
Forme   : T_ijk ≠ 0 ssi |i-j| ≤ radius ET |j-k| ≤ radius
          Valeurs ~ U[-1,1] pour les entrées dans le rayon
Rang    : 3 (fixe)
Shape   : (n_dof, n_dof, n_dof)

Justification conservation :
  Structure spatiale locale rang-3 spécifique —
  non généralisable proprement via RN-*.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'        : 'R3-003',
    'rank'      : 3,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'radius': int}  défaut 2
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof, n_dof, n_dof) sparse local coupling
    """
    radius = int(params.get('radius', 2))

    i = jnp.arange(n_dof)
    j = jnp.arange(n_dof)
    k = jnp.arange(n_dof)

    # Masque 3D : |i-j| ≤ radius ET |j-k| ≤ radius
    dij = jnp.abs(i[:, None, None] - j[None, :, None])
    djk = jnp.abs(j[None, :, None] - k[None, None, :])
    mask = (dij <= radius) & (djk <= radius)

    values = jax.random.uniform(key, (n_dof, n_dof, n_dof), minval=-1.0, maxval=1.0)
    return jnp.where(mask, values, 0.0)
