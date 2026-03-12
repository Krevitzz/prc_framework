"""
atomics/D_encodings/rn_001_random_uniform.py

RN-001 : Tenseur aléatoire uniforme rang paramétrique
Forme   : T_i₁..iᵣ ~ U[-1,1]
Rang    : params['rank'] (défaut 3)
Shape   : (n_dof,) * rank
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': False,
    'id'        : 'RN-001',
    'rank'      : None,   # déterminé par params['rank']
    'stochastic': True,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'rank': int}  défaut 3
        key    : PRNGKey JAX

    Returns:
        jnp.ndarray (n_dof,) * rank, valeurs dans [-1,1]
    """
    rank  = int(params.get('rank', 3))
    shape = (n_dof,) * rank
    return jax.random.uniform(key, shape, minval=-1.0, maxval=1.0)
