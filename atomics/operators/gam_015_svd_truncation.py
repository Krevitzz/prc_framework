"""
atomics/operators/gam_015_svd_truncation.py

GAM-015 : Troncature SVD normalisée
Forme   : T_{n+1} = SVD_k(unfold₀(T)) / σ₁,  reconstruction → reshape original
Famille : structural
Rang    : 2 uniquement
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'             : 'GAM-015',
    'family'         : 'structural',
    'rank_constraint': 2,
    'differentiable' : False,
    'stochastic'     : False,
    'non_markovian'  : False,
}

def apply(
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,  # ignoré
    params     : dict,
    key        : jax.Array,    # ignoré
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur de rang 2 (n, m) ou plus ? Mais rank_constraint=2 donc 2D.
        prev_state : Ignoré
        params     : {'k': int}  défaut n_dof // 2
        key        : Ignoré
    """
    original_shape = state.shape
    n0 = original_shape[0]
    n_cols = state.size // n0
    mat = state.reshape(n0, n_cols)

    U, s, Vt = jnp.linalg.svd(mat, full_matrices=False)

    k = params.get('k', n0 // 2)
    indices = jnp.arange(len(s))
    mask = (indices < k).astype(jnp.float32)
    s_masked = s * mask
    s_norm = s_masked / (s[0] + 1e-10)

    mat_k = (U * s_norm) @ Vt
    return mat_k.reshape(original_shape)