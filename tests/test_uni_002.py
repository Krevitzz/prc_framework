# tests/test_univ_002.py
"""
Évolution trace normalisée.

Objectif :
- Mesurer stabilité trace (somme diagonale)
- Détection comportements pathologiques (explosion trace)

Métriques :
- trace_normalized : Trace / dimension (moyenne diagonale)
- trace_variance : Variance trace sur temps

Algorithmes utilisés :
- algebra.trace_value : Calcul standard trace

Exclusions :
- Déterminant : Trop sensible petites variations
- Valeurs propres individuelles : Redondant avec spectral tests
"""

import numpy as np

TEST_ID = "UNIV-002"
TEST_CATEGORY = "UNIV"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'trace_normalized': {
        'registry_key': 'algebra.trace_value',
        'default_params': {
            'normalize': True,  # Diviser par dimension
        },
        'post_process': 'round_4',
    },
    
    'trace_absolute': {
        'registry_key': 'algebra.trace_value',
        'default_params': {
            'normalize': False,
        },
        'post_process': 'round_2',
    }
}