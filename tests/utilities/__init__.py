# tests/utilities/__init__.py

"""
Package tests/utilities - Tests modulaires réutilisables

Ce package contient tous les tests observationnels pour analyser
le comportement de Γ sur D.

PRINCIPE FONDAMENTAL:
Les tests OBSERVENT sans prescrire. Le TM décide de l'interprétation.

CATALOGUE COMPLET:
- test_symmetry: Tests symétrie (TEST-SYM-001, 002, 003, ANTI)
- test_norm: Tests normes/bornes (TEST-UNIV-001, TEST-BND-001, STR-002)
- test_diversity: Tests diversité (TEST-UNIV-002 + compléments)
- test_convergence: Tests convergence (TEST-UNIV-003 + compléments)
"""

# ============================================================================
# SYMÉTRIE
# ============================================================================

from .test_symmetry import (
    # Classes
    SymmetryResult,
    
    # Tests officiels
    test_symmetry_preservation,      # TEST-SYM-001
    test_symmetry_creation,           # TEST-SYM-002
    test_asymmetry_evolution,         # TEST-SYM-003
    
    # Test bonus
    test_antisymmetry_preservation,
    
    # Utilitaires
    compute_asymmetry,
    is_symmetric,
    compute_antisymmetry,
    is_antisymmetric,
)

# ============================================================================
# NORMES ET BORNES
# ============================================================================

from .test_norm import (
    # Classes
    NormResult,
    BoundsResult,
    
    # Tests officiels
    test_norm_evolution,              # TEST-UNIV-001
    test_bounds_preservation,         # TEST-BND-001
    test_spectral_evolution,          # TEST-STR-002
    
    # Tests complémentaires
    test_element_wise_bounds,
)

# ============================================================================
# DIVERSITÉ
# ============================================================================

from .test_diversity import (
    # Classes
    DiversityResult,
    
    # Tests officiels
    test_diversity_preservation,      # TEST-UNIV-002
    
    # Tests complémentaires
    test_entropy_evolution,
    test_range_evolution,
    test_uniformity,
    test_distinct_values,
)

# ============================================================================
# CONVERGENCE
# ============================================================================

from .test_convergence import (
    # Classes
    ConvergenceResult,
    
    # Tests officiels
    test_convergence_to_fixed_point,  # TEST-UNIV-003
    
    # Tests complémentaires
    test_lyapunov_exponent,
    test_periodic_orbit,
    test_distance_to_custom_target,
)

# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Classes de résultats
    'SymmetryResult',
    'NormResult',
    'BoundsResult',
    'DiversityResult',
    'ConvergenceResult',
    
    # Tests symétrie
    'test_symmetry_preservation',
    'test_symmetry_creation',
    'test_asymmetry_evolution',
    'test_antisymmetry_preservation',
    'compute_asymmetry',
    'is_symmetric',
    'compute_antisymmetry',
    'is_antisymmetric',
    
    # Tests normes/bornes
    'test_norm_evolution',
    'test_bounds_preservation',
    'test_spectral_evolution',
    'test_element_wise_bounds',
    
    # Tests diversité
    'test_diversity_preservation',
    'test_entropy_evolution',
    'test_range_evolution',
    'test_uniformity',
    'test_distinct_values',
    
    # Tests convergence
    'test_convergence_to_fixed_point',
    'test_lyapunov_exponent',
    'test_periodic_orbit',
    'test_distance_to_custom_target',
]

__version__ = '1.0.0'
__author__ = 'PRC Tests Team'
__description__ = 'Tests modulaires observationnels pour analyse Γ'

# ============================================================================
# HELPERS
# ============================================================================

def get_applicable_tests(d_base_type: str, gamma_type: str = "any") -> list:
    """
    Retourne la liste des tests applicables pour un type de D donné.
    
    Args:
        d_base_type: "SYM" | "ASY" | "R3" | "any"
        gamma_type: "pointwise" | "coupled" | "any" (pour extensions futures)
    
    Returns:
        Liste de noms de tests applicables
    
    Exemple:
        tests = get_applicable_tests("SYM")
        # ['UNIV-001', 'UNIV-002', 'UNIV-003', 'SYM-001', 'BND-001', ...]
    """
    # Tests universels (toujours applicables)
    applicable = [
        'UNIV-001',  # Norme
        'UNIV-002',  # Diversité
        'UNIV-003',  # Convergence
    ]
    
    # Tests rang 2
    if d_base_type in ['SYM', 'ASY']:
        applicable.extend([
            'SYM-003',   # Évolution asymétrie (observation)
            'STR-002',   # Évolution spectrale
            'BND-001',   # Bornes
        ])
        
        # Tests spécifiques symétrie
        if d_base_type == 'SYM':
            applicable.append('SYM-001')  # Préservation symétrie
        elif d_base_type == 'ASY':
            applicable.append('SYM-002')  # Création symétrie
    
    # Tests rang 3
    elif d_base_type == 'R3':
        applicable.extend([
            'BND-001',   # Bornes
        ])
    
    return applicable


def run_all_applicable_tests(history, D_base, d_base_id: str, 
                             gamma_id: str = "any") -> dict:
    """
    Execute automatiquement tous les tests applicables.
    
    Args:
        history: Liste d'états [D_0, D_1, ..., D_T]
        D_base: État initial D^(base)
        d_base_id: ID du générateur (ex: "SYM-001")
        gamma_id: ID de Γ (pour extensions futures)
    
    Returns:
        dict {test_id: result}
    
    Exemple:
        results = run_all_applicable_tests(history, D_base, "SYM-002")
        for test_id, result in results.items():
            print(f"{test_id}: {result.status}")
    """
    results = {}
    
    # Déterminer type de D
    if d_base_id.startswith('SYM'):
        d_type = 'SYM'
    elif d_base_id.startswith('ASY'):
        d_type = 'ASY'
    elif d_base_id.startswith('R3'):
        d_type = 'R3'
    else:
        d_type = 'any'
    
    # Tests universels
    results['UNIV-001'] = test_norm_evolution(history)
    results['UNIV-002'] = test_diversity_preservation(history)
    results['UNIV-003'] = test_convergence_to_fixed_point(history)
    
    # Tests rang 2
    if D_base.ndim == 2:
        results['SYM-003'] = test_asymmetry_evolution(history)
        results['STR-002'] = test_spectral_evolution(history)
        
        if d_type == 'SYM':
            results['SYM-001'] = test_symmetry_preservation(history)
        elif d_type == 'ASY':
            results['SYM-002'] = test_symmetry_creation(history)
    
    # Tests bornes
    results['BND-001'] = test_bounds_preservation(
        history, 
        initial_bounds=(-1.0, 1.0)
    )
    
    return results


def print_test_summary(results: dict):
    """
    Affiche un résumé des résultats de tests.
    
    Args:
        results: dict {test_id: result}
    """
    print("\n" + "="*70)
    print("RÉSUMÉ DES TESTS")
    print("="*70)
    
    # Statistiques
    n_pass = sum(1 for r in results.values() if r.status == "PASS")
    n_fail = sum(1 for r in results.values() if r.status == "FAIL")
    n_neutral = sum(1 for r in results.values() if r.status == "NEUTRAL")
    n_total = len(results)
    
    # Afficher chaque test
    for test_id, result in sorted(results.items()):
        print(result)
    
    # Résumé
    print("\n" + "-"*70)
    print(f"Total: {n_total} tests")
    print(f"  ✓ PASS:    {n_pass} ({100*n_pass/n_total:.0f}%)")
    print(f"  ✗ FAIL:    {n_fail} ({100*n_fail/n_total:.0f}%)")
    print(f"  ○ NEUTRAL: {n_neutral} ({100*n_neutral/n_total:.0f}%)")
    
    # Blockers
    blockers = [r for r in results.values() if getattr(r, 'blocking', False) and r.status == "FAIL"]
    if blockers:
        print(f"\n⚠ {len(blockers)} tests bloquants échoués:")
        for b in blockers:
            print(f"    - {b.test_name}: {b.message}")
    
    print("="*70 + "\n")


# ============================================================================
# MÉTADONNÉES
# ============================================================================

__test_catalog__ = {
    'universal': {
        'count': 3,
        'tests': ['UNIV-001', 'UNIV-002', 'UNIV-003'],
        'description': 'Applicables à tous D et Γ',
    },
    'symmetry': {
        'count': 3,
        'tests': ['SYM-001', 'SYM-002', 'SYM-003'],
        'description': 'Rang 2 uniquement',
    },
    'structure': {
        'count': 1,
        'tests': ['STR-002'],
        'description': 'Matrices rang 2 (valeurs propres)',
    },
    'bounds': {
        'count': 1,
        'tests': ['BND-001'],
        'description': 'Tous tenseurs',
    },
    'complementary': {
        'count': 7,
        'tests': ['DIV-ENTROPY', 'DIV-UNIFORM', 'CONV-LYAPUNOV', 'etc.'],
        'description': 'Tests additionnels pour analyses approfondies',
    }
}

def list_available_tests():
    """Liste tous les tests disponibles avec leurs descriptions."""
    print("\n" + "="*70)
    print("CATALOGUE DES TESTS DISPONIBLES")
    print("="*70)
    
    for category, info in __test_catalog__.items():
        print(f"\n{category.upper().replace('_', ' ')}")
        print(f"  Count: {info['count']}")
        print(f"  Tests: {', '.join(info['tests'])}")
        print(f"  Description: {info['description']}")
    
    print("\n" + "="*70 + "\n")