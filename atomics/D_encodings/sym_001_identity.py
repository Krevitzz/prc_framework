"""
atomics/D_encodings/sym_001_identity.py

SYM-001 : Matrice identité
Forme   : I[i,j] = δ_ij
Rang    : 2
Propriétés : symétrique, définie positive, valeurs singulières toutes = 1

Rôle v7 :
  F1 — rang effectif = 1, point zéro de référence spectrale
  F2 — S_VN = log(n_dof), entropie maximale uniforme, point zéro F2
  Doublet GAM-011 × SYM-001 : W @ I = W → valeurs propres de W
                               directement lisibles après un pas
"""

import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Métadonnées discovery
# ---------------------------------------------------------------------------

METADATA = {
    'id'         : 'SYM-001',
    'rank'       : 2,
    'stochastic' : False,    # key ignorée — déterministe pur
}

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create(n_dof: int, params: dict, key: jax.Array) -> jnp.ndarray:
    """
    Retourne la matrice identité (n_dof, n_dof).

    Args:
        n_dof  : Dimension
        params : Ignoré — aucun paramètre
        key    : Ignorée (stochastic: False)

    Returns:
        jnp.ndarray shape (n_dof, n_dof), dtype float32
    """
    return jnp.eye(n_dof)