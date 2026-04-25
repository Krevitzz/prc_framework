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

Optimisation :
  - Génération vectorisée des vecteurs (une seule opération random)
  - Normalisation vectorisée
  - Produit tensoriel via einsum avec indices dynamiques
  - La décompression des vecteurs utilise une boucle Python de taille rank (≤5),
    négligeable et inévitable pour passer des arguments séparés à einsum.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'            : 'RN-004',
    'rank'          : None,          # déterminé par params['rank']
    'stochastic'    : True,
    'jax_vmappable' : True,          # vrai mais non utilisé (création unique)
}

def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'rank': int}  défaut 3
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof,) * rank — tenseur séparable (rang-1)
    """
    rank = int(params.get('rank', 3))

    # Génération vectorisée de tous les vecteurs : shape (rank, n_dof)
    keys = jax.random.split(key, rank)
    vecs = jax.vmap(lambda k: jax.random.normal(k, (n_dof,)))(keys) # (rank, n_dof)

    # Normalisation de chaque vecteur
    norms = jnp.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
    vecs = vecs / norms

    # Construction du produit tensoriel via einsum avec indices dynamiques
    # Exemple pour rank=3 : chaîne "a,b,c->abc"
    letters = [chr(97 + i) for i in range(rank)]          # ['a','b','c',...]
    subscripts = ','.join(letters) + '->' + ''.join(letters)

    # Décompresser les vecteurs en arguments séparés (boucle Python courte)
    vec_list = [vecs[i] for i in range(rank)]             # liste de (n_dof,)
    T = jnp.einsum(subscripts, *vec_list)

    return T