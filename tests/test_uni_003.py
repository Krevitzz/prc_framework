# tests/test_uni_003.py
"""
Évolution norme euclidienne généralisée (tous rangs).

Objectif :
- Mesurer stabilité globale tenseur (rang 2, 3, N)
- Généralisation norme Frobenius via aplatissement

Métriques :
- euclidean_norm : Norme L2 vecteur aplati (tous rangs)

Algorithmes utilisés :
- algebra.matrix_norm (norm_type=2) : Norme euclidienne généralisée

Exclusions :
- Norme Frobenius native (UNIV-001) : Rang 2 uniquement
- Norme spectrale : Trop coûteuse, peu discriminante ici

Différence UNIV-001 :
- UNIV-001 : Frobenius pur (np.linalg.norm(A, 'fro')) → Rang 2
- UNIV-003 : Euclidienne (np.linalg.norm(T.flatten())) → Tous rangs
- Sur rang 2 : Équivalence mathématique vérifiable
"""

import numpy as np

TEST_ID = "UNIV-003"
TEST_CATEGORY = "UNIV"
TEST_VERSION = "5.5" 
TEST_WEIGHT = 1.0   
TEST_PHASE = None  # Applicable toutes phases


APPLICABILITY_SPEC = {
    "requires_rank": None,  # ✅ Tous rangs (2, 3, N)
    "requires_square": False,
    "allowed_d_types": ["ALL"],
    "minimum_dimension": None,
    "requires_even_dimension": False,
}

COMPUTATION_SPECS = {
    'euclidean_norm': {
        'registry_key': 'algebra.matrix_norm',
        'default_params': {
            'norm_type': 2,  # ✅ INT → déclenche elif isinstance(norm_type, int)
            #            ^   → np.linalg.norm(state.flatten(), 2)
            #                → Support tous rangs
        },
        'post_process': 'round_4',
    }
}