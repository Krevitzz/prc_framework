# tests/test_spectral_001.py
"""
Évolution spectre valeurs propres.

Objectif :
- Observer évolution distribution spectrale
- Détecter concentration/dispersion énergie

Métriques :
- eigenvalue_max : Plus grande valeur propre (dominance)
- spectral_gap : Écart λ₁ - λ₂ (séparation)
- spectral_radius : Rayon spectral (stabilité itérations)

Algorithmes utilisés :
- spectral.spectral_radius : Rayon spectral

Exclusions :
- Entropie spectrale : Redondant avec statistical tests
- Valeurs propres individuelles : Trop détaillé pour R0
"""

import numpy as np

TEST_ID = "SPE-002"
TEST_CATEGORY = "SPECTRAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0  
TEST_PHASE = None  # Applicable toutes phases 


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 3,  # Gap nécessite ≥3 valeurs propres
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    
    'spectral_radius': {
        'registry_key': 'spectral.spectral_radius',
        'default_params': {},
        'post_process': 'round_4',
    }
}