"""
atomics/D_encodings/rn_003_diagonal.py

RN-003 : Tenseur diagonal rang paramétrique
Forme   : T_i..i ≠ 0 ssi i₁ = i₂ = ... = iᵣ, valeurs ~ U[-1,1]
Rang    : params['rank'] (défaut 3)
Shape   : (n_dof,) * rank
Stochastique : Non (déterministe — diagonal fixe à 1.0)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'RN-003',
    'rank'      : None,
    'stochastic': False,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension par axe
        params : {'rank': int}  défaut 3
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n_dof,) * rank, non-nul uniquement sur la super-diagonale

    Construction :
        Initialise à zéro, puis T[i,i,...,i] = 1.0 pour i=0..n_dof-1
    """
    rank  = int(params.get('rank', 3))
    shape = (n_dof,) * rank
    T     = jnp.zeros(shape)

    idx = jnp.arange(n_dof)
    indices = tuple(idx for _ in range(rank))
    return T.at[indices].set(1.0)
