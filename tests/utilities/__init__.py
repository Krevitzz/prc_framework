"""
tests/utilities/__init__.py

Exports des tests modulaires et fonctions de scoring (Section 14.4).

Ce module contient tous les tests observationnels réutilisables
et les fonctions de scoring contextuel.
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
    test_local_diversity_preservation,  # NOUVEAU
    test_spatial_heterogeneity,         # NOUVEAU
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
    score_all_observations,
    compute_global_score,
    load_weights_config,
)

# =============================================================================
# MAPPING DES TESTS - CRITIQUE !
# =============================================================================

# Ce mapping relie chaque test_id à sa fonction d'exécution
# C'est le cœur du système d'exécution des tests
TEST_FUNCTION_MAP = {
    # Tests universels
    "UNIV-001": lambda history, **kwargs: test_norm_evolution(
        history, norm_type="frobenius", name="UNIV-001"
    ),
    "UNIV-002": lambda history, **kwargs: test_diversity_preservation(
        history, name="UNIV-002"
    ),
    "UNIV-003": lambda history, **kwargs: test_convergence_to_fixed_point(
        history, name="UNIV-003"
    ),
    "UNIV-004": lambda history, **kwargs: TestObservation(
        test_name="UNIV-004",
        status="SKIPPED",
        message="Test multi-seeds non implémenté dans cette version",
        applicable=True
    ),
    
    # Tests symétrie
    "SYM-001": lambda history, **kwargs: test_symmetry_preservation(
        history, name="SYM-001"
    ),
    "SYM-002": lambda history, **kwargs: test_symmetry_creation(
        history, name="SYM-002"
    ),
    "SYM-003": lambda history, **kwargs: test_asymmetry_evolution(
        history, name="SYM-003"
    ),
    
    # Tests structure
    "STR-001": lambda history, **kwargs: test_rank_preservation(
        history, name="STR-001"
    ),
    "STR-002": lambda history, **kwargs: test_spectral_evolution(
        history, name="STR-002"
    ),
    "STR-003": lambda history, **kwargs: test_temporal_correlations(
        history, name="STR-003"
    ),
    
    # Tests bornes
    "BND-001": lambda history, **kwargs: test_bounds_preservation(
        history, initial_bounds=(-1.0, 1.0), name="BND-001"
    ),
    "BND-002": lambda history, **kwargs: test_spectral_evolution(  # Approximation
        history, check_positivity=True, name="BND-002"
    ),
    
    # Tests localité
    "LOC-001": lambda history, **kwargs: test_information_propagation(
        history, name="LOC-001"
    ),
    "LOC-002": lambda history, **kwargs: test_sparsity_preservation(
        history, epsilon=1e-6, name="LOC-002"
    ),
    
    # Tests diversité complémentaires
    "DIV-ENTROPY": lambda history, **kwargs: test_entropy_evolution(
        history, name="DIV-ENTROPY"
    ),
    "DIV-UNIFORM": lambda history, **kwargs: test_uniformity(
        history, name="DIV-UNIFORM"
    ),
    "DIV-RANGE": lambda history, **kwargs: test_range_evolution(
        history, name="DIV-RANGE"
    ),
    "DIV-DISTINCT": lambda history, **kwargs: test_distinct_values(
        history, tolerance=1e-6, name="DIV-DISTINCT"
    ),
    
    # Nouveaux tests diversité
    "UNIV-002b": lambda history, **kwargs: test_local_diversity_preservation(
        history, patch_size=5, n_patches=20, name="UNIV-002b"
    ),
    "DIV-HETERO": lambda history, **kwargs: test_spatial_heterogeneity(
        history, grid_size=10, name="DIV-HETERO"
    ),
    
    # Tests convergence complémentaires
    "CONV-LYAPUNOV": lambda history, **kwargs: test_lyapunov_exponent(
        history, name="CONV-LYAPUNOV"
    ),
    "CONV-SPEED": lambda history, **kwargs: test_convergence_speed(
        history, epsilon=1e-6, name="CONV-SPEED"
    ),
    "CONV-OSCILLATION": lambda history, **kwargs: test_oscillation_detection(
        history, name="CONV-OSCILLATION"
    ),
    
    # Tests pointwise
    "PW-001": lambda history, **kwargs: test_pointwise_independence(
        history, name="PW-001"
    ),
}

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
    """
    if not history or len(history) < 2:
        print(f"⚠ Historique insuffisant: {len(history) if history else 0} états")
        return {}
    
    results = {}
    state_shape = history[0].shape
    
    # Récupérer liste des tests applicables
    applicable_tests = get_applicable_tests(d_base_id, state_shape, gamma_id)
    
    print(f"📊 {len(applicable_tests)} tests applicables pour D={d_base_id}, Γ={gamma_id}, shape={state_shape}")
    
    # Exécuter chaque test applicable
    for test_name in applicable_tests:
        try:
            # Vérifier si test est implémenté
            if test_name not in TEST_FUNCTION_MAP:
                print(f"  ⚠ Test non implémenté: {test_name}")
                results[test_name] = TestObservation(
                    test_name=test_name,
                    status="NOT_IMPLEMENTED",
                    message=f"Test {test_name} non implémenté",
                    applicable=True
                )
                continue
            
            # Exécuter le test
            test_func = TEST_FUNCTION_MAP[test_name]
            result = test_func(history)
            
            # S'assurer que le test a bien un nom
            if hasattr(result, 'test_name'):
                result.test_name = test_name
            
            results[test_name] = result
            
            print(f"  ✓ {test_name}: {result.status}")
            
        except Exception as e:
            print(f"  ❌ Test {test_name} échoué: {str(e)}")
            import traceback
            traceback.print_exc()
            
            results[test_name] = TestObservation(
                test_name=test_name,
                status="ERROR",
                message=f"Exception: {str(e)}",
                applicable=True
            )
    
    print(f"✅ {len(results)} tests exécutés avec succès")
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
    'test_local_diversity_preservation',
    'test_spatial_heterogeneity', 
    
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
    'score_all_observations',
    'compute_global_score',
    'load_weights_config',
    
    # Fonctions principales
    'run_all_applicable_tests'
]

__version__ = '1.1.0'