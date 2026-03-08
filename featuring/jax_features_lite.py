"""
featuring/jax_features_lite.py

[_lite] Stub minimal pour valider le squelette pipeline end-to-end.
        Remplacé par jax_features.py (F1→F7 complets) en phase 2.
        TAG: CLEANUP_PHASE2

Deux fonctions publiques :
  measure_state(state, state_next) → dict scalaires  [dans lax.scan]
  post_scan(signals, last_state)   → dict floats     [après scan, vers parquet]

Contrainte JAX : measure_state doit retourner une structure dict fixe
avec des scalaires shape () à chaque appel — lax.scan trace la structure
au premier appel et l'impose pour tous les pas suivants.
"""

import jax.numpy as jnp
from typing import List

# ---------------------------------------------------------------------------
# Colonnes parquet produites par ce module
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    'frob_norm_mean',
    'frob_norm_final',
]

# ---------------------------------------------------------------------------
# measure_state — appelée à chaque pas dans lax.scan
# ---------------------------------------------------------------------------

def measure_state(
    state     : jnp.ndarray,
    state_next: jnp.ndarray,
) -> dict:
    """
    Calcule les mesures du pas courant.

    Appelée à l'intérieur de lax.scan — doit retourner une structure
    dict fixe avec des scalaires shape () à chaque appel.

    Args:
        state      : Tenseur au pas t (ignoré dans _lite)
        state_next : Tenseur au pas t+1

    Returns:
        dict : {'frob_norm': scalar shape ()}
    """
    frob_norm = jnp.linalg.norm(state_next)
    return {'frob_norm': frob_norm}


# ---------------------------------------------------------------------------
# post_scan — appelée une fois après lax.scan, en Python
# ---------------------------------------------------------------------------

def post_scan(
    signals   : dict,
    last_state: jnp.ndarray,
) -> dict:
    """
    Agrège les signaux accumulés pendant le scan en features scalaires.

    Appelée en Python après lax.scan — les arrays JAX sont convertis
    en floats Python purs pour l'écriture parquet.

    Args:
        signals    : {'frob_norm': jnp.array(max_it,)}
        last_state : Dernier état (ignoré dans _lite)

    Returns:
        dict : {feature_name: float}  — colonnes parquet
    """
    return {
        'frob_norm_mean' : float(jnp.mean(signals['frob_norm'])),
        'frob_norm_final': float(signals['frob_norm'][-1]),
    }