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
    'rank_constraint': None,
    'differentiable' : False,
    'stochastic'     : False,
    'non_markovian'  : False,
}


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur courant, tout rang ≥ 2
        params : {'k': int}  défaut = state.shape[0] // 2
                  k = rang de troncature (statique dans XLA)
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray même shape que state, normalisé par σ₁

    Note :
        Si σ₁ < 1e-8 (état nul), retourne state inchangé pour éviter div/0.
    """
    original_shape = state.shape
    n0             = original_shape[0]

    # Mode-0 unfolding : (n0, prod autres dims)
    n_cols = 1
    for d in original_shape[1:]:
        n_cols *= d
    mat = state.reshape(n0, n_cols)

    # SVD complète (full_matrices=False)
    U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)

    k       = int(params.get('k', max(1, n0 // 2)))
    k       = min(k, s.shape[0])

    # Troncature rang k
    U_k  = U[:, :k]
    s_k  = s[:k]
    Vt_k = Vt[:k, :]

    mat_k = (U_k * s_k[None, :]) @ Vt_k

    # Normalisation par σ₁
    sigma1 = s[0]
    mat_k  = jnp.where(sigma1 > 1e-8, mat_k / sigma1, mat_k)

    return mat_k.reshape(original_shape)
