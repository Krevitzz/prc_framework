"""
atomics/D_encodings/r3_007_block_structure.py

R3-007 : Structure bloc 3D rang 3
Forme   : T_ijk = amplitude_intra si bloc(i)=bloc(j)=bloc(k), sinon amplitude_inter
          + bruit additif faible
Rang    : 3 (fixe)
Shape   : (n_dof, n_dof, n_dof)

Justification conservation :
  Structure bloc 3D spécifique — les interactions triple-blocs
  n'ont pas d'équivalent naturel en RN-*.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'R3-007',
    'rank'      : 3,
    'stochastic': True,
    'jax_vmappable': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'n_blocks': int, 'intra': float, 'inter': float}
                  défauts : n_blocks=4, intra=0.8, inter=0.05
        key    : PRNGKey JAX (bruit)

    Returns:
        jnp.ndarray (n_dof, n_dof, n_dof)
    """
    n_blocks = params.get('n_blocks', 4)
    intra    = params.get('intra', 0.8)
    inter    = params.get('inter', 0.05)

    block_ids = jnp.arange(n_dof) * n_blocks // n_dof

    # Masque triple-bloc : i, j, k dans le même bloc
    same_ijk = (
        (block_ids[:, None, None] == block_ids[None, :, None]) &
        (block_ids[None, :, None] == block_ids[None, None, :])
    )

    base   = jnp.where(same_ijk, intra, inter)
    noise  = jax.random.normal(key, (n_dof, n_dof, n_dof)) * 0.02
    return base + noise
