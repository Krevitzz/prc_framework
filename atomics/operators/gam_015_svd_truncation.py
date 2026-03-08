"""
atomics/operators/gam_015_svd_truncation.py

GAM-015 : Troncature SVD normalisée
Forme   : T_{n+1} = SVD_k(unfold₀(T)) / σ₁,  reconstruction → reshape original
Famille : structural

Mode-0 unfolding → SVD → troncature rang k → reconstruction → reshape.
Normalisation par σ₁ (valeur singulière dominante).

Propriétés :
  - S_VN décroît garantie (rang effectif → k)
  - Attracteur holographique contrôlé
  - Non-différentiable (SVD truncation non-smooth)

Rôle v7 :
  F2 — S_VN décroît → attracteur holographique candidat contrôlé
  Q3 — produit des structures émergentes stables par construction

Note JAX :
  jnp.linalg.svd compilable dans lax.scan (formes statiques).
  k = params.get('k', n_dof // 2) → k doit être statique pour XLA.
  → k passé dans params, figé avant vmap (valeur scalaire Python).
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-015',
    'family'         : 'structural',
    'rank_constraint': 2,       # SVD mode-0 sur rang≥3 → matrices gigantesques
    'differentiable' : False,
    'stochastic'     : False,
    'non_markovian'  : False,
}


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    original_shape = state.shape
    n0             = original_shape[0]

    # Mode-0 unfolding
    n_cols = 1
    for d in original_shape[1:]:
        n_cols *= d
    mat = state.reshape(n0, n_cols)

    # SVD (statique car full_matrices=False sur shape fixe)
    U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)

    # --- FIX CONCRETIZATION ERROR ---
    # k vient du YAML, JAX le voit comme un Traceur dans le vmap.
    # On ne peut pas slicer s[:k], on doit masquer.
    k = params.get('k', n0 // 2)
    
    # On crée un vecteur d'indices [0, 1, 2, ..., n0-1]
    indices = jnp.arange(len(s))
    
    # Masque binaire : 1.0 si l'indice < k, sinon 0.0
    mask = (indices < k).astype(jnp.float32)
    
    # On annule les valeurs singulières au-delà de k
    s_masked = s * mask
    
    # Normalisation par sigma_1 (pour éviter l'explosion de norme)
    # On ajoute epsilon pour la stabilité si s[0] est nul
    s_norm = s_masked / (s[0] + 1e-10)
    # ---------------------------------

    # Reconstruction (forme toujours statique n0 x n_cols)
    # mat_k = U @ diag(s_norm) @ Vt
    mat_k = (U * s_norm) @ Vt

    # Reshape original
    return mat_k.reshape(original_shape)