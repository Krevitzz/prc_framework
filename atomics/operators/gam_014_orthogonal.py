"""
atomics/operators/gam_014_orthogonal.py

GAM-014 : Transformation orthogonale
Forme   : T_{n+1} = U @ T @ Uᵀ
Famille : markovian
Rang    : 2 uniquement
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
    A = jax.random.normal(key, (n_dof, n_dof))
    Q, _ = jnp.linalg.qr(A)
    return Q

def prepare_params(params: dict, n_dof: int, key: jax.Array) -> dict:
    U = generate_U(n_dof=n_dof, key=key)
    return {'U': U}

def apply(
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,  # ignoré
    params     : dict,
    key        : jax.Array,    # ignoré
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur rang 2 (n_dof, n_dof)
        prev_state : Ignoré
        params     : {'U': jnp.ndarray (n_dof, n_dof)}
        key        : Ignoré
    """
    U = params['U']
    # Utilisation de swapaxes pour la transposition
    return U @ state @ U.swapaxes(-1, -2)