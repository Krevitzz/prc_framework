# tests/test_topological_001.py
"""
Invariants topologiques simplifiés.

Objectif :
- Observer changements topologiques
- Détecter création/destruction structures

Métriques :
- connected_components : Fragmentation
- holes_count : Nombre trous
- euler_characteristic : Invariant χ

Algorithmes utilisés :
- topological.connected_components : Comptage composantes
- topological.holes_count : Détection trous
- topological.euler_characteristic : Calcul χ

Exclusions :
- Homologie persistante : Hors scope R0
- Betti numbers : Nécessite bibliothèques spécialisées
"""

import numpy as np

TEST_ID = "TOP-001"
TEST_CATEGORY = "TOPOLOGICAL"
TEST_VERSION = "5.4"

APPLICABILITY_SPEC = {
    "requires_rank": 2,
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": 10,  # Topologie nécessite espace
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'connected_components': {
        'registry_key': 'topological.connected_components',
        'default_params': {
            'threshold': 0.0,
            'connectivity': 1,
        },
        'post_process': 'round_2',
    },
    
    'holes_count': {
        'registry_key': 'topological.holes_count',
        'default_params': {
            'threshold': 0.0,
            'min_hole_size': 4,
        },
        'post_process': 'round_2',
    },
    
    'euler_characteristic': {
        'registry_key': 'topological.euler_characteristic',
        'default_params': {
            'threshold': 0.0,
        },
        'post_process': 'round_2',
    }
}