"""
tests/utilities/__init__.py

Exports des tests modulaires et fonctions de scoring (Section 14.4).

Ce module contient tous les tests observationnels réutilisables
et les fonctions de scoring contextuel.

PRINCIPE FONDAMENTAL (Section 7 Charte) :
  - Tests observent sans juger
  - Scoring interprète selon contexte
  - Séparation stricte observation/interprétation
"""

# =============================================================================
# IMPORTS DES TESTS
# =============================================================================

# Tests de norme
from .test_norm import (
    test_norm_evolution,
    test_bounds_preservation,
    test_spectral_evolution,
    test_element_wise_bounds,
    NormResult,
    BoundsResult,
)

# Tests de symétrie
from .test_symmetry import (
    test_symmetry_preservation,
    test_symmetry_creation,
    test_asymmetry_evolution,
    test_antisymmetry_preservation,
    SymmetryResult,
)

# Tests de diversité
from .test_diversity import (
    test_diversity_preservation,
    test_entropy_evolution,
    test_range_evolution,
    test_uniformity,
    test_distinct_values,
    DiversityResult,
)

# Tests de convergence
from .test_convergence import (
    test_convergence_to_fixed_point,
    test_lyapunov_exponent,
    test_oscillation_detection,
    test_convergence_speed,
    ConvergenceResult,
)

# =============================================================================
# IMPORTS APPLICABILITÉ ET SCORING
# =============================================================================

from .applicability import (
    TEST_APPLICABILITY,
    is_test_applicable,
    get_applicable_tests,
)

from .scoring import (
    score_observation,
    compute_global_score,
    load_weights_config,
)

# =============================================================================
# FONCTION PRINCIPALE : Exécution complète
# =============================================================================

def run_all_applicable_tests(history, D_base, d_base_id, gamma_id):
    """
    Exécute tous les tests applicables pour un run.
    
    Args:
        history: Liste des états [D_0, D_1, ..., D_T]
        D_base: État initial avant modifiers
        d_base_id: ID de la base (ex: "SYM-001")
        gamma_id: ID du Γ (ex: "GAM-001")
    
    Returns:
        dict {test_name: observation_result}
    
    Cette fonction :
      1. Détermine quels tests sont applicables
      2. Exécute uniquement les tests applicables
      3. Retourne les observations brutes (PAS de scores)
    """
    if not history:
        return {}
    
    results = {}
    state_shape = history[0].shape
    
    # Récupérer liste des tests applicables
    applicable_tests = get_applicable_tests(d_base_id, state_shape, gamma_id)
    
    # Exécuter chaque test applicable
    for test_name in applicable_tests:
        try:
            # Tests universels (toujours applicables)
            if test_name == "UNIV-001":
                results[test_name] = test_norm_evolution(
                    history, 
                    norm_type="frobenius",
                    name=test_name
                )
            
            elif test_name == "UNIV-002":
                results[test_name] = test_diversity_preservation(
                    history,
                    name=test_name
                )
            
            elif test_name == "UNIV-003":
                results[test_name] = test_convergence_to_fixed_point(
                    history,
                    name=test_name
                )
            
            elif test_name == "UNIV-004":
                # Nécessite plusieurs seeds - skip pour l'instant
                # TODO: implémenter test inter-seeds
                pass
            
            # Tests symétrie (rang 2 uniquement)
            elif test_name == "SYM-001":
                results[test_name] = test_symmetry_preservation(
                    history,
                    name=test_name
                )
            
            elif test_name == "SYM-002":
                results[test_name] = test_symmetry_creation(
                    history,
                    name=test_name
                )
            
            elif test_name == "SYM-003":
                results[test_name] = test_asymmetry_evolution(
                    history,
                    name=test_name
                )
            
            # Tests structure
            elif test_name == "STR-002":
                results[test_name] = test_spectral_evolution(
                    history,
                    name=test_name
                )
            
            # Tests bornes
            elif test_name == "BND-001":
                results[test_name] = test_bounds_preservation(
                    history,
                    initial_bounds=(-1.0, 1.0),
                    name=test_name
                )
            
            # Tests diversité complémentaires
            elif test_name == "DIV-ENTROPY":
                results[test_name] = test_entropy_evolution(
                    history,
                    name=test_name
                )
            
            elif test_name == "DIV-UNIFORM":
                results[test_name] = test_uniformity(
                    history,
                    name=test_name
                )
            
            # Tests convergence complémentaires
            elif test_name == "CONV-LYAPUNOV":
                results[test_name] = test_lyapunov_exponent(
                    history,
                    name=test_name
                )
        
        except Exception as e:
            # En cas d'erreur, logger mais continuer
            print(f"⚠ Test {test_name} failed: {str(e)}")
            continue
    
    return results


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tests norme
    'test_norm_evolution',
    'test_bounds_preservation',
    'test_spectral_evolution',
    'test_element_wise_bounds',
    'NormResult',
    'BoundsResult',
    
    # Tests symétrie
    'test_symmetry_preservation',
    'test_symmetry_creation',
    'test_asymmetry_evolution',
    'test_antisymmetry_preservation',
    'SymmetryResult',
    
    # Tests diversité
    'test_diversity_preservation',
    'test_entropy_evolution',
    'test_range_evolution',
    'test_uniformity',
    'test_distinct_values',
    'DiversityResult',
    
    # Tests convergence
    'test_convergence_to_fixed_point',
    'test_lyapunov_exponent',
    'test_oscillation_detection',
    'test_convergence_speed',
    'ConvergenceResult',
    
    # Applicabilité
    'TEST_APPLICABILITY',
    'is_test_applicable',
    'get_applicable_tests',
    
    # Scoring
    'score_observation',
    'compute_global_score',
    'load_weights_config',
    
    # Fonction principale
    'run_all_applicable_tests',
]

__version__ = '1.0.0'
__charter_compliance__ = 'Section 14.4'