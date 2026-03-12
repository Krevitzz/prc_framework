"""
atomics/D_encodings/rn_004_separable.py

RN-004 : Tenseur séparable rang paramétrique
Forme   : T = u₁ ⊗ u₂ ⊗ ... ⊗ uᵣ  (produit extérieur de r vecteurs)
          uₖ ~ N(0,1) normalisés
Rang    : params['rank'] (défaut 3)
Shape   : (n_dof,) * rank

Propriété :
  Entanglement entropy = 0 par construction (état produit pur)
  → Point zéro de référence pour F3 (entanglement)
"""

import jax
import jax.numpy as jnp
import functools

METADATA = {
    'jax_vmappable': False,   # rank détermine la shape → doit être concret
    'id'        : 'RN-004',
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
        jnp.ndarray (n_dof,) * rank — tenseur rang-1 (séparable)

    Note :
        Entanglement entropy = 0 → point zéro F3.
        jnp.einsum construit le produit extérieur de r vecteurs.
    """
    rank = int(params.get('rank', 3))
    keys = jax.random.split(key, rank)

    vecs = []
    for k in keys:
        u = jax.random.normal(k, (n_dof,))
        u = u / (jnp.linalg.norm(u) + 1e-10)
        vecs.append(u)

    # Produit extérieur itératif via jnp.outer puis reshape
    T = vecs[0]
    for v in vecs[1:]:
        T = jnp.tensordot(T, v, axes=0)

    return T