"""
atomics/D_encodings/sym_006_block_hierarchical.py

SYM-006 : Matrice bloc hiérarchique symétrique
Forme   : A_ij = intra si bloc(i)=bloc(j), inter sinon
          Bruit additif gaussien pour différencier les individus
Rang    : 2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'SYM-006',
    'rank'      : 2,
    'stochastic': True,
    'jax_vmappable': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'n_blocks': int, 'intra': float, 'inter': float}
                  défauts : n_blocks=min(10, n_dof), intra=0.7, inter=0.1
        key    : PRNGKey JAX (bruit intra-bloc)

    Returns:
        jnp.ndarray (n_dof, n_dof) symétrique blocs hiérarchiques
    """
    n_blocks = params.get('n_blocks', min(10, n_dof))
    intra    = params.get('intra', 0.7)
    inter    = params.get('inter', 0.1)

    # Assignation bloc par nœud
    block_ids = jnp.arange(n_dof) * n_blocks // n_dof

    # Matrice de base : intra si même bloc, inter sinon
    same_block = block_ids[:, None] == block_ids[None, :]
    base       = jnp.where(same_block, intra, inter)

    # Bruit symétrique additif faible
    B     = jax.random.normal(key, (n_dof, n_dof)) * 0.02
    noisy = base + (B + B.T) / 2.0

    return noisy
