# tests/test_spatial_001.py
"""
Rugosité et lissage spatial.

Objectif :
- Mesurer complexité structure spatiale
- Détecter transitions rugosité/lissage

Métriques :
- gradient_magnitude : Amplitude variations spatiales
- laplacian_energy : Rugosité (courbure locale)
- smoothness : Inverse rugosité normalisé

Algorithmes utilisés :
- spatial.gradient_magnitude : Norme gradient moyen
- spatial.laplacian_energy : Énergie laplacien
- spatial.smoothness : Score lissage

Exclusions :
- Variance locale : Corrélée avec gradient
- Détection contours : Trop spécifique pour R0
"""

import numpy as np

TEST_ID = "SPA-001"
TEST_CATEGORY = "SPATIAL"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   
TEST_PHASE = None  # Applicable toutes phases


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 5,  # Gradients nécessitent espace
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'gradient_magnitude': {
        'registry_key': 'spatial.gradient_magnitude',
        'default_params': {
            'normalize': True,
        },
        'post_process': 'round_6',
    },
    
    'laplacian_energy': {
        'registry_key': 'spatial.laplacian_energy',
        'default_params': {
            'normalize': True,
        },
        'post_process': 'round_6',
    },
    
    'smoothness': {
        'registry_key': 'spatial.smoothness',
        'default_params': {},
        'post_process': 'round_4',
    }
}

