# tests/utilities/HUB/profiling_runner.py
"""
Orchestration profiling multi-axes avec découverte automatique.

RESPONSABILITÉ :
- Exécuter profiling tous axes (test, gamma, modifier, encoding)
- Enrichir axe test (discriminant_power)
- Format retour unifié strict

ARCHITECTURE :
- Délégation profiling_common (profiling individuel)
- Délégation cross_profiling (enrichissements)
- Zéro calcul inline (orchestration pure)

UTILISATEURS :
- verdict_reporter.py (génération rapports complets)
- Scripts analyse (profiling_analysis.py si futur)

INTERDICTIONS (voir PRC_DEPENDENCY_RULES.md) :
- Implémenter profiling (→ profiling_common)
- Implémenter calculs (→ UTIL, registries)
- Modifier observations (lecture seule)
"""

from typing import Dict, List
from collections import defaultdict

# Profiling modules
from ..utils.profiling_common import (
    profile_all_tests,
    profile_all_gammas,
    profile_all_modifiers,
    profile_all_encodings,
    compare_tests_summary,
    compare_gammas_summary,
    compare_modifiers_summary,
    compare_encodings_summary,
)

# Cross-profiling
from ..utils.cross_profiling import compute_all_discriminant_powers


# ============================================================================
# CONFIGURATION
# ============================================================================

# Ordre exécution axes (Charter R5.1-D)
DEFAULT_AXES_ORDER = ['test', 'gamma', 'modifier', 'encoding']


# ============================================================================
# DÉCOUVERTE AUTOMATIQUE (si besoin futur)
# ============================================================================

def discover_profiling_axes() -> List[str]:
    """
    Découvre axes profiling disponibles.
    
    R0 : Retourne liste hardcodée (4 axes)
    R1+ : Introspection profiling_common (detect profile_all_*)
    
    Returns:
        Liste axes disponibles
    """
    # R0 : Hardcodé (4 axes connus)
    return DEFAULT_AXES_ORDER.copy()


# ============================================================================
# ORCHESTRATION PRINCIPALE
# ============================================================================

def run_all_profiling(
    observations: List[dict],
    axes: List[str] = None
) -> Dict[str, Dict]:
    """
    Exécute profiling tous axes demandés.
    
    DÉLÉGATION STRICTE :
    - Profiling individuel → profiling_common.profile_all_{axis}()
    - Comparaisons → profiling_common.compare_{axis}_summary()
    - Enrichissement test → cross_profiling.compute_all_discriminant_powers()
    
    Args:
        observations: Liste observations SUCCESS
        axes: Axes à profiler (None = tous)
    
    Returns:
        {
            'test': {
                'profiles': {...},
                'summary': {...},
                'discriminant_powers': {...},  # Enrichissement spécifique
                'metadata': {...}
            },
            'gamma': {...},
            'modifier': {...},
            'encoding': {...}
        }
    
    Raises:
        ValueError: Si axe inconnu
    
    Examples:
        >>> results = run_all_profiling(observations)
        >>> results['gamma']['profiles']['GAM-001']
        {...}
        
        >>> results_partial = run_all_profiling(observations, axes=['gamma', 'test'])
    """
    if axes is None:
        axes = DEFAULT_AXES_ORDER
    
    # Validation axes
    valid_axes = discover_profiling_axes()
    for axis in axes:
        if axis not in valid_axes:
            raise ValueError(
                f"Axe inconnu '{axis}'. Valides: {valid_axes}"
            )
    
    results = {}
    
    for axis in axes:
        # Appel dynamique fonctions profiling_common
        # profile_all_{axis}() + compare_{axis}_summary()
        
        axis_plural = f"{axis}s" if axis != 'encoding' else 'encodings'
        
        profile_func_name = f"profile_all_{axis_plural}"
        compare_func_name = f"compare_{axis_plural}_summary"
        
        # Récupération fonctions
        profile_func = globals()[profile_func_name]
        compare_func = globals()[compare_func_name]
        
        # Profiling axe
        profiles = profile_func(observations)
        summary = compare_func(profiles)
        
        # Structure retour
        result = {
            'profiles': profiles,
            'summary': summary,
            'metadata': {
                'axis': axis,
                'n_entities': len(profiles),
                'n_observations': len(observations),
                'profiling_version': '6.1',
                'profiling_module': 'profiling_common'
            }
        }
        
        # Enrichissement spécifique axe test
        if axis == 'test':
            result['discriminant_powers'] = compute_all_discriminant_powers(
                profiles, reference_axis='gamma'
            )
        
        results[axis] = result
    
    return results


def run_profiling_single_axis(
    observations: List[dict],
    axis: str
) -> Dict:
    """
    Profiling un seul axe (helper).
    
    Args:
        observations: Liste observations
        axis: Axe à profiler ('test', 'gamma', 'modifier', 'encoding')
    
    Returns:
        Résultat profiling axe
    """
    results = run_all_profiling(observations, axes=[axis])
    return results[axis]


# ============================================================================
# HELPERS EXTRACTION (si besoin reporting)
# ============================================================================

def get_entity_profile(
    profiling_results: Dict,
    axis: str,
    entity_id: str
) -> Dict:
    """
    Extrait profil d'une entité spécifique.
    
    Args:
        profiling_results: Résultats run_all_profiling()
        axis: Axe concerné
        entity_id: ID entité (ex: 'GAM-001', 'SYM-001')
    
    Returns:
        Profil entité (tests + metadata)
    
    Raises:
        KeyError: Si axe ou entité introuvable
    """
    if axis not in profiling_results:
        raise KeyError(f"Axe '{axis}' non trouvé dans résultats")
    
    axis_results = profiling_results[axis]
    profiles = axis_results['profiles']
    
    if entity_id not in profiles:
        raise KeyError(
            f"Entité '{entity_id}' non trouvée dans axe '{axis}'. "
            f"Disponibles: {list(profiles.keys())}"
        )
    
    return profiles[entity_id]


def get_test_profile_for_entity(
    profiling_results: Dict,
    axis: str,
    entity_id: str,
    test_name: str
) -> Dict:
    """
    Extrait profil d'un test spécifique pour une entité.
    
    Args:
        profiling_results: Résultats run_all_profiling()
        axis: Axe concerné
        entity_id: ID entité
        test_name: ID test (ex: 'SYM-001')
    
    Returns:
        Profil test pour entité
    
    Raises:
        KeyError: Si non trouvé
    """
    entity_profile = get_entity_profile(profiling_results, axis, entity_id)
    
    if test_name not in entity_profile['tests']:
        raise KeyError(
            f"Test '{test_name}' non trouvé pour {entity_id}. "
            f"Disponibles: {list(entity_profile['tests'].keys())}"
        )
    
    return entity_profile['tests'][test_name]