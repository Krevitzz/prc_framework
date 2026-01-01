# tests/test_pat_001.py
"""
Diversité et concentration distribution.

Objectif :
- Mesurer dispersion valeurs
- Détecter émergence structures concentrées ou uniformes

Métriques :
- diversity_simpson : Indice diversité Simpson
- concentration_top10 : Concentration énergie dans top 10%
- uniformity : Proximité distribution uniforme

Algorithmes utilisés :
- pattern.diversity : Indice Simpson
- pattern.concentration_ratio : Ratio concentration
- pattern.uniformity : Distance à uniforme

Exclusions :
- Entropie Shannon : Redondant avec diversity
- Coefficient Gini : Approximé par concentration_ratio
"""

import numpy as np

TEST_ID = "PAT-001"
TEST_CATEGORY = "PAT"
TEST_VERSION = "5.4"

APPLICABILITY_SPEC = {
    "requires_rank": None,  # Tout rang
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'diversity_simpson': {
        'registry_key': 'pattern.diversity',
        'default_params': {
            'bins': 50,
        },
        'post_process': 'round_4',
    },
    
    'concentration_top10': {
        'registry_key': 'pattern.concentration_ratio',
        'default_params': {
            'top_percent': 0.1,
        },
        'post_process': 'round_4',
    },
    
    'uniformity': {
        'registry_key': 'pattern.uniformity',
        'default_params': {
            'bins': 50,
        },
        'post_process': 'round_4',
    }
}
