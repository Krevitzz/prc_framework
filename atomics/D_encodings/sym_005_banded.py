"""
atomics/D_encodings/sym_005_banded.py

SYM-005 : Matrice bande symétrique
Forme   : A_ij = amplitude · cos(π|i-j|/bandwidth)  si |i-j| ≤ bandwidth, sinon 0
Rang    : 2
Stochastique : Non (déterministe)
"""

import jax
import jax.numpy as jnp

METADATA = {
    'jax_vmappable': True,
    'id'        : 'SYM-005',
    'rank'      : 2,
    'stochastic': False,
}


def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Args:
        n_dof  : Dimension de la matrice
        params : {'bandwidth': int, 'amplitude': float}
                  défauts : bandwidth=3, amplitude=0.5
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray (n_dof, n_dof) symétrique bande creuse
    """
    bandwidth = params.get('bandwidth', 3)
    amplitude = params.get('amplitude', 0.5)

    i = jnp.arange(n_dof)
    j = jnp.arange(n_dof)
    diff = jnp.abs(i[:, None] - j[None, :])

    in_band = diff <= bandwidth
    values  = amplitude * jnp.cos(jnp.pi * diff / jnp.maximum(bandwidth, 1))

    return jnp.where(in_band, values, 0.0)
