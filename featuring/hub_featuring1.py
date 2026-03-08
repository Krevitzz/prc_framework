"""
featuring/hub_featuring.py

Hub featuring — routage uniquement.
Point d'entrée unique pour run_one_jax vers les features.

Convention hub :
  - Importe uniquement depuis featuring/ (son propre dossier)
  - Zéro logique métier
  - Permet de basculer jax_features_lite → jax_features (F1→F7)
    sans toucher à run_one_jax.py

Quand jax_features.py sera prêt (phase 2) :
  remplacer l'import ci-dessous — un seul fichier à modifier.
  TAG: CLEANUP_PHASE2
"""

from featuring.jax_features_lite import (   # TAG: CLEANUP_PHASE2
    measure_state,
    post_scan,
    FEATURE_NAMES,
)

__all__ = ['measure_state', 'post_scan', 'FEATURE_NAMES']