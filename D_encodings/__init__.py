# D_encodings/__init__.py

"""
Package D_encodings - Encodages spécifiques pour les tenseurs

Ce package contient des créateurs d'états D^(base) pour différents types de tenseurs.
Chaque module contient des présuppositions explicites sur la structure des tenseurs.

PRINCIPE FONDAMENTAL :
Ces présuppositions sont HORS du core - le kernel et state_preparation sont aveugles
à ces propriétés spécifiques.

CATALOGUE COMPLET:
- rank2_symmetric: 6 générateurs (D-SYM-001 à D-SYM-006)
- rank2_asymmetric: 4+ générateurs (D-ASY-001 à D-ASY-004)
- rank3_correlations: 3+ générateurs (D-R3-001 à D-R3-003)
- verification_tests: Tests de vérification pour tous les générateurs
"""

# ============================================================================
# RANG 2 SYMÉTRIQUE
# ============================================================================

from .rank2_symmetric import (
    # Catalogue officiel
    create_identity,              # D-SYM-001
    create_random_uniform,        # D-SYM-002
    create_random_gaussian,       # D-SYM-003
    create_correlation_matrix,    # D-SYM-004
    create_banded,                # D-SYM-005
    create_block_hierarchical,    # D-SYM-006
    
    # Alias legacy (compatibilité)
    create_uniform,
    create_random,
)

# ============================================================================
# RANG 2 ASYMÉTRIQUE
# ============================================================================

from .rank2_asymmetric import (
    # Catalogue officiel
    create_random_asymmetric,     # D-ASY-001
    create_lower_triangular,      # D-ASY-002
    create_antisymmetric,         # D-ASY-003
    create_directional_gradient,  # D-ASY-004
    
    # Générateurs bonus
    create_circulant_asymmetric,
    create_sparse_asymmetric,
)

# ============================================================================
# RANG 3
# ============================================================================

from .rank3_correlations import (
    # Catalogue officiel
    create_random_rank3,             # D-R3-001
    create_partial_symmetric_rank3,  # D-R3-002
    create_local_coupling_rank3,     # D-R3-003
    
    # Générateurs bonus
    create_fully_symmetric_rank3,
    create_diagonal_rank3,
    create_separable_rank3,
    create_block_rank3,
)

# ============================================================================
# TESTS DE VÉRIFICATION
# ============================================================================

from .verification_tests import (
    # Classes
    VerificationResult,
    
    # Tests génériques
    verify_shape,
    verify_bounds,
    verify_not_constant,
    
    # Tests symétrie
    verify_symmetry,
    verify_antisymmetry,
    verify_diagonal_ones,
    verify_positive_definite,
    
    # Tests structure
    verify_sparsity,
    verify_banded,
    verify_triangular,
    
    # Tests rang 3
    verify_partial_symmetry_23,
    verify_full_symmetry_rank3,
    
    # Tests reproductibilité
    verify_reproducibility,
    
    # Suite complète
    run_verification_suite,
    print_verification_report,
)

# ============================================================================
# MÉTADONNÉES
# ============================================================================

__all__ = [
    # Rang 2 symétrique
    'create_identity',
    'create_random_uniform',
    'create_random_gaussian',
    'create_correlation_matrix',
    'create_banded',
    'create_block_hierarchical',
    'create_uniform',  # Legacy
    'create_random',   # Legacy
    
    # Rang 2 asymétrique
    'create_random_asymmetric',
    'create_lower_triangular',
    'create_antisymmetric',
    'create_directional_gradient',
    'create_circulant_asymmetric',
    'create_sparse_asymmetric',
    
    # Rang 3
    'create_random_rank3',
    'create_partial_symmetric_rank3',
    'create_local_coupling_rank3',
    'create_fully_symmetric_rank3',
    'create_diagonal_rank3',
    'create_separable_rank3',
    'create_block_rank3',
    
    # Vérification
    'VerificationResult',
    'verify_shape',
    'verify_bounds',
    'verify_not_constant',
    'verify_symmetry',
    'verify_antisymmetry',
    'verify_diagonal_ones',
    'verify_positive_definite',
    'verify_sparsity',
    'verify_banded',
    'verify_triangular',
    'verify_partial_symmetry_23',
    'verify_full_symmetry_rank3',
    'verify_reproducibility',
    'run_verification_suite',
    'print_verification_report',
]

__version__ = '2.0.0'
__author__ = 'PRC Encodings Team'
__description__ = 'Créateurs d\'états D^(base) avec vérification complète'

# ============================================================================
# DOCUMENTATION DES PRÉSUPPOSITIONS
# ============================================================================

__catalog__ = {
    "symmetric_rank2": {
        "count": 6,
        "ids": ["SYM-001", "SYM-002", "SYM-003", "SYM-004", "SYM-005", "SYM-006"],
        "properties": ["symmetric", "rank=2", "bounds=[-1,1]"],
    },
    "asymmetric_rank2": {
        "count": 4,
        "ids": ["ASY-001", "ASY-002", "ASY-003", "ASY-004"],
        "properties": ["asymmetric", "rank=2", "bounds=[-1,1]"],
    },
    "rank3": {
        "count": 3,
        "ids": ["R3-001", "R3-002", "R3-003"],
        "properties": ["rank=3", "bounds=[-1,1]"],
        "note": "Memory intensive: N³ elements"
    }
}

# Note d'avertissement
__warning__ = """
Ces créateurs imposent des présuppositions spécifiques.
Toujours vérifier la compatibilité avec les tests de vérification.
Utiliser verification_tests.run_verification_suite() avant production.
"""

# ============================================================================
# HELPERS
# ============================================================================

def list_generators():
    """Liste tous les générateurs disponibles avec leurs IDs."""
    print("\n" + "="*60)
    print("CATALOGUE DES GÉNÉRATEURS D^(base)")
    print("="*60)
    
    for category, info in __catalog__.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print(f"  Count: {info['count']}")
        print(f"  IDs: {', '.join(info['ids'])}")
        print(f"  Properties: {', '.join(info['properties'])}")
        if 'note' in info:
            print(f"  Note: {info['note']}")
    
    print("\n" + "="*60 + "\n")


def get_generator_by_id(base_id: str):
    """
    Retourne le générateur correspondant à un ID du catalogue.
    
    Args:
        base_id: ID du catalogue (ex: "SYM-001", "ASY-002", "R3-003")
    
    Returns:
        Fonction générateur
    
    Raises:
        ValueError: Si ID inconnu
    
    Exemple:
        gen = get_generator_by_id("SYM-001")
        D = gen(n_dof=50)
    """
    mapping = {
        # Symétriques
        "SYM-001": create_identity,
        "SYM-002": create_random_uniform,
        "SYM-003": create_random_gaussian,
        "SYM-004": create_correlation_matrix,
        "SYM-005": create_banded,
        "SYM-006": create_block_hierarchical,
        
        # Asymétriques
        "ASY-001": create_random_asymmetric,
        "ASY-002": create_lower_triangular,
        "ASY-003": create_antisymmetric,
        "ASY-004": create_directional_gradient,
        
        # Rang 3
        "R3-001": create_random_rank3,
        "R3-002": create_partial_symmetric_rank3,
        "R3-003": create_local_coupling_rank3,
    }
    
    if base_id not in mapping:
        available = ', '.join(mapping.keys())
        raise ValueError(f"Unknown base_id '{base_id}'. Available: {available}")
    
    return mapping[base_id]


def get_recommended_tests(base_id: str) -> list:
    """
    Retourne la liste des tests recommandés pour un générateur.
    
    Args:
        base_id: ID du catalogue
    
    Returns:
        Liste de noms de tests
    
    Exemple:
        tests = get_recommended_tests("SYM-004")
        # ['shape_2d', 'symmetry', 'diagonal_ones', 'positive_definite']
    """
    recommendations = {
        # Symétriques
        "SYM-001": ['shape_2d', 'symmetry', 'diagonal_ones', 'positive_definite'],
        "SYM-002": ['shape_2d', 'symmetry', 'bounds', 'non_constant'],
        "SYM-003": ['shape_2d', 'symmetry', 'non_constant'],
        "SYM-004": ['shape_2d', 'symmetry', 'diagonal_ones', 'positive_definite'],
        "SYM-005": ['shape_2d', 'symmetry', 'diagonal_ones', 'banded_3'],
        "SYM-006": ['shape_2d', 'symmetry', 'diagonal_ones', 'bounds'],
        
        # Asymétriques
        "ASY-001": ['shape_2d', 'bounds', 'non_constant'],
        "ASY-002": ['shape_2d', 'bounds', 'triangular_lower'],
        "ASY-003": ['shape_2d', 'antisymmetry', 'bounds'],
        "ASY-004": ['shape_2d', 'non_constant'],
        
        # Rang 3
        "R3-001": ['shape_3d', 'bounds', 'non_constant'],
        "R3-002": ['shape_3d', 'bounds', 'partial_symmetry_23'],
        "R3-003": ['shape_3d', 'bounds'],
    }
    
    return recommendations.get(base_id, ['shape_2d'])


# ============================================================================
# EXPORTS ADDITIONNELS
# ============================================================================

__all__.extend([
    'list_generators',
    'get_generator_by_id',
    'get_recommended_tests',
])