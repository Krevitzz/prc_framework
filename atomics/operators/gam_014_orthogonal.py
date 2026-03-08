"""
atomics/operators/gam_014_orthogonal.py

GAM-014 : Transformation orthogonale
Forme   : T_{n+1} = U @ T @ Uᵀ
Famille : markovian
Rang    : 2 uniquement

U : matrice orthogonale (n_dof, n_dof), QR d'une matrice aléatoire.
    Générée une fois, passée dans params['U'].
    Appelée via prepare_params() avant le scan.

Propriétés :
  - trace_J = 0  (isométrie → pas de contraction ni expansion)
  - S_VN conservé par construction
  - Null model A3 direct — Γ qui ne dissipe PAS D

Rôle v7 :
  F4 — trace_J = 0 → point de référence (Hutchinson doit renvoyer ~0)
  A3 — résistance maximale : D non dissipé par ce Γ
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-014',
    'family'         : 'markovian',
    'rank_constraint': 2,
    'differentiable' : True,
    'stochastic'     : False,
    'non_markovian'  : False,
}


def generate_U(n_dof: int, key: jax.Array) -> jnp.ndarray:
    """
    Génère une matrice orthogonale U via décomposition QR.

    Args:
        n_dof : Dimension de U
        key   : PRNGKey JAX

    Returns:
        U : jnp.ndarray (n_dof, n_dof) orthogonale (Uᵀ@U = I)
    """
    A = jax.random.normal(key, (n_dof, n_dof))
    Q, _ = jnp.linalg.qr(A)
    return Q


def prepare_params(params: dict, n_dof: int, key: jax.Array) -> dict:
    """
    Pré-calcule U depuis key — appelée par hub_running avant vmap.

    Args:
        params : {} — aucun param scalaire
        n_dof  : Dimension de la matrice
        key    : PRNGKey JAX (key_CI du run)

    Returns:
        {'U': jnp.ndarray (n_dof, n_dof)}
    """
    U = generate_U(n_dof=n_dof, key=key)
    return {'U': U}


def apply(
    state : jnp.ndarray,
    params: dict,
    key   : jax.Array,
) -> jnp.ndarray:
    """
    Args:
        state  : Tenseur rang 2 (n_dof, n_dof)
        params : {'U': jnp.ndarray (n_dof, n_dof)}
                  U doit être généré via generate_U() avant le scan
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n_dof, n_dof) — norme de Frobenius conservée

    Raises:
        ValueError : si state.ndim != 2
    """
    if state.ndim != 2:
        raise ValueError(
            f"GAM-014 applicable rang 2 uniquement, reçu ndim={state.ndim}"
        )
    U = params['U']
    return U @ state @ U.T
