"""
featuring/hub_featuring.py

Hub featuring — routage uniquement.
Point d'entrée unique pour run_one_jax vers les features.

Convention hub :
  - Importe uniquement depuis featuring/ (son propre dossier)
  - Zéro logique métier
"""

from featuring.jax_features import (
    measure_state,
    post_scan,
    FEATURE_NAMES,
)

__all__ = ['measure_state', 'post_scan', 'FEATURE_NAMES']
