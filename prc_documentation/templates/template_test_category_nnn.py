# tests/test_category_nnn.py
"""
[Titre descriptif]

Objectif :
- [Description phénomène mesuré]

Métriques :
- [métrique1] : [Pourquoi pertinente]
- [métrique2] : [Pourquoi pertinente]

Algorithmes utilisés :
- [registry_key1] : [Justification]

Exclusions :
- [Alternatives non retenues] : [Pourquoi]
"""

import numpy as np

TEST_ID = "CATEGORY-NNN"
TEST_CATEGORY = "CATEGORY"
TEST_VERSION = "5.4"

APPLICABILITY_SPEC = {
    "requires_rank": 2,              # None = tous rangs
    "requires_square": True,         # True si matrice carrée requise
    "allowed_d_types": ["SYM", "ASY"],  # ["ALL"] ou liste
    "minimum_dimension": 10,         # None si pas de min
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'metric_name': {
        'registry_key': 'registre.fonction',
        'default_params': {
            'param1': value1,
            'param2': value2,
        },
        'post_process': 'round_4',  # Optionnel
    },
    # 1 à 5 métriques max
}