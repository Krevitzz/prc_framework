"""
atomics/D_encodings/rn_002_partial_symmetric.py

RN-002 : Tenseur partiellement symétrique rang paramétrique
Forme   : T_ij.. = T_ji..  (symétrie sur axes 0 et 1 uniquement)
Rang    : params['rank'] (défaut 3)
Shape   : (n_dof,) * rank

Note :
  Symétrie complète toutes permutations = coûteux, non prioritaire.
  Symétrie partielle axes 0,1 : T = (T + transpose(T, [1,0,2,...]))/2
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': False,
    'id'        : 'RN-002',
    'rank'      : None,
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'rank': int}  défaut 3
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof,) * rank, symétrique sur axes 0,1
    """
    rank  = int(params.get('rank', 3))
    shape = (n_dof,) * rank
    T     = jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)

    # Permutation axes 0 ↔ 1 uniquement
    perm   = list(range(rank))
    perm[0], perm[1] = 1, 0
    T_swap = jnp.transpose(T, perm)

    return (T + T_swap) / 2.0
