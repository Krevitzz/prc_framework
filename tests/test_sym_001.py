# prc_framework/tests/test_sym_001.py
"""
Évolution asymétrie matrices.

Objectif :
- Mesurer création/destruction/préservation symétrie

Métriques :
- asymmetry_norm : Discriminant principal (global)
- asymmetry_norm_normalized : Comparable entre tailles différentes

Algorithmes utilisés :
- algebra.matrix_asymmetry : Norme ||A - A^T||, paramétrable

Exclusions :
- Trace asymétrie : Masque patterns spatiaux (trop agrégée)
- Max asymétrie : Sensible outliers, peu robuste
"""

import numpy as np

TEST_ID = "SYM-001"
TEST_CATEGORY = "SYM"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   
TEST_PHASE = None  # Applicable toutes phases


APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": True,
    "allowed_d_types": ["SYM", "ASY"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'asymmetry_norm': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {
            'norm_type': 'frobenius',
            'normalize': False,
        },
        'post_process': 'round_6',
    },
    
    'asymmetry_norm_normalized': {
        'registry_key': 'algebra.matrix_asymmetry',
        'default_params': {
            'norm_type': 'frobenius',
            'normalize': True,
        },
        'post_process': 'round_6',
    }
}