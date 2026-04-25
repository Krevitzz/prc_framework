"""
atomics/operators/gam_011_linear_tensordot.py

GAM-011 : Transformation linéaire tensordot (mode-0)
Forme   : T_{n+1} = W ⊗₀ T  =  tensordot(W, T, axes=([1],[0]))
Famille : markovian

W est une matrice (n_dof, n_dof) générée une fois via generate_W()
et injectée dans params['W'] avant le scan.
"""

import jax
import jax.numpy as jnp

METADATA = {
    'id'              : 'GAM-011',
    'family'          : 'markovian',
    'rank_constraint' : None,
    'differentiable'  : True,
    'stochastic'      : False,
    'non_markovian'   : False,
}

_EPS = 1e-8

def generate_W(n_dof: int, scale: float = 1.0, key: jax.Array = None) -> jnp.ndarray:
    if key is None:
        key = jax.random.PRNGKey(0)
    W_raw = jax.random.normal(key, (n_dof, n_dof)) / jnp.sqrt(n_dof)
    eigenvalues = jnp.linalg.eigvals(W_raw)
    rho = jnp.max(jnp.abs(eigenvalues))
    return W_raw / (rho + _EPS) * scale

def prepare_params(params: dict, n_dof: int, key: jax.Array) -> dict:
    scale = params.get('scale', 1.0)
    W = generate_W(n_dof=n_dof, scale=float(scale), key=key)
    return {'W': W}

def apply(
    state      : jnp.ndarray,
    prev_state : jnp.ndarray,  # ignoré
    params     : dict,
    key        : jax.Array,    # ignoré
) -> jnp.ndarray:
    """
    Args:
        state      : Tenseur courant, shape (n_dof, ...)
        prev_state : Ignoré
        params     : {'W': jnp.ndarray (n_dof, n_dof)}
        key        : Ignoré
    """
    W = params['W']
    return jnp.tensordot(W, state, axes=([1], [0]))