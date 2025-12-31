# tests/test_xxx_yyy.py
"""
[Titre descriptif]

Objectif :
- [Description claire du phénomène mesuré]

Métriques :
- [nom_métrique_1] : [Pertinence, discrimination attendue]
- [nom_métrique_2] : [Pertinence, discrimination attendue]
- [nom_métrique_3] : [Pertinence, discrimination attendue]

Algorithmes utilisés :
- [registry_key_1] : [Justification du choix]
- [registry_key_2] : [Justification du choix]

Exclusions :
- [Métriques évidentes NON incluses] : [Pourquoi]

Notes :
- [Contexte additionnel si pertinent]
"""

import numpy as np

# ============================================================================
# IDENTIFICATION
# ============================================================================

TEST_ID = "CAT-NNN"          # Format: CAT-001 (SYM-001, PAT-001, etc.)
TEST_CATEGORY = "CAT"        # UNIV | SYM | PAT | TOP | SPA
TEST_VERSION = "5.4"         # Version architecture

# ============================================================================
# APPLICABILITÉ
# ============================================================================

APPLICABILITY_SPEC = {
    "requires_rank": int | None,           # None = tous rangs, 2 = matrices, 3 = tenseurs
    "requires_square": bool,               # Matrice carrée requise ?
    "allowed_d_types": List[str],          # ["SYM", "ASY", "R3"] ou ["ALL"]
    "minimum_dimension": int | None,       # Dimension minimale (ex: 10)
    "requires_even_dimension": bool,       # Toutes dimensions paires ?
}

# ============================================================================
# SPÉCIFICATIONS DE CALCUL
# ============================================================================

COMPUTATION_SPECS = {
    'nom_metrique_unique': {
        # OBLIGATOIRE : référence au registre
        'registry_key': 'registre.fonction',
        
        # OBLIGATOIRE : paramètres par défaut
        'default_params': {
            'param1': valeur1,
            'param2': valeur2,
        },
        
        # OPTIONNEL : post-traitement
        'post_process': 'round_4',  # Clé POST_PROCESSORS
        
        # OPTIONNEL : validation supplémentaire
        'validation': {
            'expected_range': [min, max],
            'tolerance': float,
        }
    },
    
    # 1 à 5 métriques maximum
}