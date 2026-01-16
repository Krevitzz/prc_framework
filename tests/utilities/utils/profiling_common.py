# tests/utilities/UTIL/profiling_common.py
"""
Profiling Common Module - Charter 6.1

ARCHITECTURE UNIFIÉE :
- Moteur générique profiling tous axes (gamma, modifier, encoding, test)
- API publique conventionnelle (profile_all_{axis}, compare_{axis}_summary)
- Délégation timeline_utils, aggregation_utils, regime_utils
- Zéro duplication code inter-axes

RESPONSABILITÉS :
- Agrégation signatures dynamiques
- Calcul profil PRC complet
- Profiling entités × tests (moteur générique)
- Comparaisons inter-entités
- API publique découvrable (8 fonctions : 4 axes × 2 fonctions)

AXES SUPPORTÉS :
- test (test_name) : Observations, pouvoir discriminant via cross_profiling
- gamma (gamma_id) : Mécanismes Γ
- modifier (modifier_id) : Perturbations D
- encoding (d_encoding_id) : Structure D

PHILOSOPHIE R0 :
- Format retour unifié strict (profiles, summary, metadata)
- Aucune extension spécifique axe (cross_profiling pour enrichissements)
- Découverte dynamique entités depuis observations
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

# ============================================================================
# IMPORTS UTILITIES
# ============================================================================

# Timelines et événements dynamiques
from .timeline_utils import (
    extract_dynamic_events,
    compute_timeline_descriptor,
    TIMELINE_THRESHOLDS
)

# Agrégations statistiques
from .aggregation_utils import (
    aggregate_summary_metrics,
    aggregate_run_dispersion
)

# Classification régimes
from .regime_utils import (
    classify_regime
)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Mapping axes → clés DB (normalisation stricte)
ENTITY_KEY_MAP = {
    'test': 'test_name',        # DB utilise 'test_name'
    'gamma': 'gamma_id',        # DB utilise 'gamma_id'
    'modifier': 'modifier_id',  # DB utilise 'modifier_id'
    'encoding': 'd_encoding_id' # DB utilise 'd_encoding_id' (pas d'alias)
}


# ============================================================================
# FONCTIONS COMMUNES (extraction gamma_profiling)
# ============================================================================

def aggregate_dynamic_signatures(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège signatures événements + timelines compositionnels.
    
    EXTRACTION : Identique gamma_profiling.aggregate_dynamic_signatures()
    RÉUTILISABLE : Tous axes profiling
    
    Utilise compute_timeline_descriptor() pour chaque run,
    puis agrège par Counter.
    
    Args:
        observations: Liste observations pour (entity, test) fixés
        metric_name: Nom métrique principale
    
    Returns:
        {
            'dynamic_signature': {
                'deviation_onset_median': float | None,
                'instability_onset_median': float | None,
                'oscillatory_fraction': float,
                'saturation_fraction': float,
                'collapse_fraction': float
            },
            'timeline_distribution': {
                'dominant_timeline': str,
                'timeline_confidence': float,
                'timeline_variants': {timeline: count, ...}
            }
        }
    """
    n_runs = len(observations)
    
    # Compteurs événements booléens
    oscillatory_count = 0
    saturation_count = 0
    collapse_count = 0
    
    # Onsets (pour médiane)
    deviation_onsets = []
    instability_onsets = []
    
    # Timelines compositionnels
    timelines = []
    
    for obs in observations:
        events = extract_dynamic_events(obs, metric_name)
        
        # Booléens
        if events.get('oscillatory'):
            oscillatory_count += 1
        if events.get('saturation'):
            saturation_count += 1
        if events.get('collapse'):
            collapse_count += 1
        
        # Onsets
        if events.get('deviation_onset') is not None:
            deviation_onsets.append(events['deviation_onset'])
        if events.get('instability_onset') is not None:
            instability_onsets.append(events['instability_onset'])
        
        # Timeline descriptor
        timeline_desc = compute_timeline_descriptor(
            events['sequence'],
            events['sequence_timing_relative'],
            events.get('oscillatory_global', False)
        )
        timelines.append(timeline_desc['timeline_compact'])
    
    # Fractions
    oscillatory_frac = oscillatory_count / n_runs if n_runs > 0 else 0.0
    saturation_frac = saturation_count / n_runs if n_runs > 0 else 0.0
    collapse_frac = collapse_count / n_runs if n_runs > 0 else 0.0
    
    # Onsets médianes (absolus, pour info)
    deviation_onset_median = float(np.median(deviation_onsets)) if deviation_onsets else None
    instability_onset_median = float(np.median(instability_onsets)) if instability_onsets else None
    
    # Timeline dominante
    counter = Counter(timelines)
    if counter:
        dominant, count = counter.most_common(1)[0]
        confidence = count / n_runs
    else:
        dominant = 'no_significant_dynamics'
        confidence = 0.0
    
    return {
        'dynamic_signature': {
            'deviation_onset_median': deviation_onset_median,
            'instability_onset_median': instability_onset_median,
            'oscillatory_fraction': oscillatory_frac,
            'saturation_fraction': saturation_frac,
            'collapse_fraction': collapse_frac
        },
        'timeline_distribution': {
            'dominant_timeline': dominant,
            'timeline_confidence': confidence,
            'timeline_variants': dict(counter)
        }
    }


def compute_prc_profile(
    metrics: dict,
    dynamic_sig: dict,
    timeline_dist: dict,
    dispersion: dict,
    n_runs: int,
    n_valid: int,
    test_name: str
) -> dict:
    """
    Génère profil PRC complet avec confidence heuristique.
    
    EXTRACTION : Identique gamma_profiling.compute_prc_profile()
    MODIFICATION : Suppression parameter extensions (pas de cas d'usage R0)
    RÉUTILISABLE : Tous axes profiling
    
    Args:
        metrics: Métriques agrégées (aggregate_summary_metrics)
        dynamic_sig: Signature dynamique (aggregate_dynamic_signatures)
        timeline_dist: Distribution timelines (aggregate_dynamic_signatures)
        dispersion: Dispersion inter-runs (aggregate_run_dispersion)
        n_runs: Nombre total runs
        n_valid: Nombre runs SUCCESS
        test_name: ID test
    
    Returns:
        {
            'regime': str,
            'behavior': str,
            'dominant_timeline': {...},
            'robustness': {...},
            'pathologies': {...},
            'n_runs': int,
            'n_valid': int,
            'confidence': str,
            'confidence_metadata': {...}
        }
    """
    if not metrics:
        return {
            'regime': 'NO_DATA',
            'behavior': 'unknown',
            'n_runs': n_runs,
            'n_valid': n_valid,
            'confidence': 'none',
            'confidence_metadata': {}
        }
    
    # Régime (avec qualificatif MIXED si applicable)
    regime = classify_regime(metrics, dynamic_sig, timeline_dist, dispersion, test_name)
    
    # Behavior
    base_regime = regime.split('::')[-1] if '::' in regime else regime
    
    if base_regime.startswith('CONSERVES_'):
        behavior = 'stable'
    elif base_regime in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE']:
        behavior = 'unstable'
    elif base_regime == 'DEGRADING':
        behavior = 'degrading'
    elif base_regime == 'TRIVIAL':
        behavior = 'stable'  # Techniquement stable (pas de dynamique)
    elif base_regime == 'SATURATES_HIGH':
        behavior = 'stable'  # Converge vers plateau
    else:
        behavior = 'uncategorized'
    
    # Qualificatif MIXED
    if regime.startswith('MIXED::'):
        behavior = 'mixed'
    
    # Timeline dominante
    dominant_timeline = timeline_dist['dominant_timeline']
    timeline_confidence = timeline_dist['timeline_confidence']
    
    # Robustness
    robustness = {
        'homogeneous': not dispersion['bimodal_detected'] and dispersion['cv_across_runs'] < 0.5,
        'mixed_behavior': dispersion['bimodal_detected'],
        'numerically_stable': dynamic_sig['collapse_fraction'] < 0.1 and 
                              dynamic_sig.get('instability_onset_median') is None
    }
    
    # Pathologies
    pathologies = {
        'numeric_instability': 'NUMERIC_INSTABILITY' in regime,
        'oscillatory': dynamic_sig['oscillatory_fraction'] > 0.3,
        'collapse': dynamic_sig['collapse_fraction'] > 0.1,
        'trivial': base_regime == 'TRIVIAL',
        'degrading': base_regime == 'DEGRADING'
    }
    
    # Confidence heuristique
    confidence_criteria = {
        'min_runs_high': 20,
        'min_runs_medium': 10,
        'max_cv_homogeneous': 0.5,
        'min_timeline_confidence': 0.7
    }
    
    if (n_valid >= confidence_criteria['min_runs_high'] and 
        not dispersion['bimodal_detected'] and
        timeline_confidence >= confidence_criteria['min_timeline_confidence']):
        confidence = 'high'
        rationale = f"n_valid={n_valid}, bimodal=False, timeline_conf={timeline_confidence:.2f}"
    elif n_valid >= confidence_criteria['min_runs_medium']:
        confidence = 'medium'
        rationale = f"n_valid={n_valid}"
    else:
        confidence = 'low'
        rationale = f"n_valid={n_valid} < {confidence_criteria['min_runs_medium']}"
    
    return {
        'regime': regime,
        'behavior': behavior,
        'dominant_timeline': {
            'timeline_compact': dominant_timeline,
            'confidence': timeline_confidence,
            'variants': timeline_dist['timeline_variants']
        },
        'robustness': robustness,
        'pathologies': pathologies,
        'n_runs': n_runs,
        'n_valid': n_valid,
        'confidence': confidence,
        'confidence_metadata': {
            'level': confidence,
            'criteria': confidence_criteria,
            'rationale': rationale
        }
    }


# ============================================================================
# MOTEUR GÉNÉRIQUE (privé)
# ============================================================================

def _profile_test_for_entity(
    observations: List[dict],
    test_name: str,
    entity_id: str,
    axis: str
) -> dict:
    """
    Profil UN test sous UNE entité.
    
    GÉNÉRALISATION : gamma_profiling.profile_test_for_gamma()
    MOTEUR INTERNE : Logique partagée tous axes
    
    Args:
        observations: Observations pour (entity_id, test_name) fixés
        test_name: ID test (ex: 'SYM-001')
        entity_id: ID entité (ex: 'GAM-001', 'M1', 'SYM-001')
        axis: Nom axe ('gamma', 'modifier', 'encoding', 'test')
    
    Returns:
        {
            'test_name': str,
            '{axis}_id': str,  # Clé dynamique selon axe
            'prc_profile': {...},
            'diagnostic_signature': {...},
            'instrumentation': {...}
        }
    """
    if not observations:
        return {
            'test_name': test_name,
            f'{axis}_id': entity_id,
            'prc_profile': {
                'regime': 'NO_DATA',
                'behavior': 'unknown',
                'n_runs': 0,
                'n_valid': 0,
                'confidence': 'none'
            },
            'diagnostic_signature': {},
            'instrumentation': {}
        }
    
    # Métrique principale
    first_obs = observations[0]
    obs_data = first_obs.get('observation_data', {})
    stats = obs_data.get('statistics', {})
    
    metric_name = list(stats.keys())[0] if stats else 'unknown'
    
    n_runs = len(observations)
    n_valid = len([o for o in observations if o.get('status') == 'SUCCESS'])
    
    # Vérifier présence dynamic_events
    dynamic_events_present = 'dynamic_events' in obs_data
    timeseries_present = 'timeseries' in obs_data
    
    # NIVEAU 3 : Instrumentation
    summary_metrics = aggregate_summary_metrics(observations, metric_name)
    
    instrumentation = {
        'metric_name': metric_name,
        'summary_metrics': summary_metrics,
        'data_completeness': {
            'dynamic_events_present': dynamic_events_present,
            'timeseries_present': timeseries_present,
            'fallback_used': not dynamic_events_present
        },
        'computation_metadata': {
            'profiling_version': '6.1',
            'profiling_module': 'profiling_common',
            'profiling_axis': axis,
            'timeline_architecture': 'compositional_relative',
            'n_runs': n_runs,
            'n_valid': n_valid
        }
    }
    
    # NIVEAU 2 : Diagnostic signature
    run_dispersion = aggregate_run_dispersion(observations, metric_name)
    event_aggregates = aggregate_dynamic_signatures(observations, metric_name)
    
    diagnostic_signature = {
        'dynamic_signature': event_aggregates['dynamic_signature'],
        'timeline_distribution': event_aggregates['timeline_distribution'],
        'run_dispersion': run_dispersion,
        'thresholds_used': {
            'timeline_early': TIMELINE_THRESHOLDS['early'],
            'timeline_mid': TIMELINE_THRESHOLDS['mid'],
            'timeline_late': TIMELINE_THRESHOLDS['late'],
            'instability_detection': 'P90 * 10 (relative)',
            'oscillatory_threshold': '10% sign changes',
            'saturation_cv': '5%',
            'bimodal_iqr_ratio': '3.0'
        }
    }
    
    # NIVEAU 1 : PRC Profile
    prc_profile = compute_prc_profile(
        summary_metrics,
        event_aggregates['dynamic_signature'],
        event_aggregates['timeline_distribution'],
        run_dispersion,
        n_runs,
        n_valid,
        test_name
    )
    
    return {
        'test_name': test_name,
        f'{axis}_id': entity_id,
        'prc_profile': prc_profile,
        'diagnostic_signature': diagnostic_signature,
        'instrumentation': instrumentation
    }


def _profile_entity_axis(
    observations: List[dict],
    axis: str,
    entity_key: str
) -> dict:
    """
    Moteur générique profiling tous axes.
    
    GÉNÉRALISATION : gamma_profiling.profile_all_gammas()
    CŒUR ARCHITECTURE : Point central unification
    
    Args:
        observations: Toutes observations (découverte dynamique entités)
        axis: Nom axe ('gamma', 'modifier', 'encoding', 'test')
        entity_key: Clé DB selon axe (depuis ENTITY_KEY_MAP)
    
    Returns:
        {
            'entity_id_1': {
                'tests': {
                    'test_name_1': {...},  # Résultat _profile_test_for_entity
                    'test_name_2': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            'entity_id_2': {...},
            ...
        }
    """
    profiles = {}
    
    # Groupement observations par entité (découverte dynamique)
    obs_by_entity = defaultdict(list)
    for obs in observations:
        entity_id = obs.get(entity_key)
        if entity_id:
            obs_by_entity[entity_id].append(obs)
    
    # Profiling chaque entité
    for entity_id, entity_obs in obs_by_entity.items():
        # Groupement observations par test
        obs_by_test = defaultdict(list)
        for obs in entity_obs:
            test_name = obs.get('test_name')
            if test_name:
                obs_by_test[test_name].append(obs)
        
        # Profiling chaque test
        test_profiles = {}
        for test_name, test_obs in obs_by_test.items():
            profile = _profile_test_for_entity(
                test_obs, test_name, entity_id, axis
            )
            test_profiles[test_name] = profile
        
        profiles[entity_id] = {
            'tests': test_profiles,
            'n_tests': len(test_profiles),
            'n_total_runs': len(entity_obs)
        }
    
    return profiles


def _compare_entities_summary(profiles: dict, axis: str) -> dict:
    """
    Comparaisons cross-entities.
    
    GÉNÉRALISATION : gamma_profiling.compare_gammas_summary()
    
    Args:
        profiles: Résultats _profile_entity_axis()
        axis: Nom axe
    
    Returns:
        {
            'by_regime': {
                'regime_name': ['entity_id_1', 'entity_id_2', ...],
                ...
            },
            'by_test': {
                'test_name': {
                    'best_conservation': 'entity_id',
                    'worst_conservation': 'entity_id',
                    'ranking': ['entity_id_1', ...]
                },
                ...
            }
        }
    """
    by_regime = defaultdict(list)
    
    # Groupement par régime dominant
    for entity_id, entity_data in profiles.items():
        regime_counts = Counter()
        for test_profile in entity_data['tests'].values():
            regime = test_profile['prc_profile']['regime']
            regime_counts[regime] += 1
        
        if regime_counts:
            dominant_regime = regime_counts.most_common(1)[0][0]
            by_regime[dominant_regime].append(entity_id)
    
    # Rankings par test
    by_test = {}
    
    all_tests = set()
    for entity_data in profiles.values():
        all_tests.update(entity_data['tests'].keys())
    
    for test_name in all_tests:
        # Scoring conservation simple (aligné gamma_profiling)
        scores = []
        
        for entity_id, entity_data in profiles.items():
            test_profile = entity_data['tests'].get(test_name)
            
            if not test_profile:
                continue
            
            prc = test_profile['prc_profile']
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            
            # Score conservation
            if regime.startswith('CONSERVES_'):
                score = 1.0
            elif regime == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
            
            scores.append((entity_id, score))
        
        if scores:
            scores.sort(key=lambda x: x[1], reverse=True)
            
            by_test[test_name] = {
                'best_conservation': scores[0][0],
                'worst_conservation': scores[-1][0],
                'ranking': [eid for eid, _ in scores]
            }
    
    return {
        'by_regime': dict(by_regime),
        'by_test': by_test
    }


# ============================================================================
# API PUBLIQUE (conventions naming - découvrable)
# ============================================================================

def profile_all_tests(observations: List[dict]) -> dict:
    """
    Profil comportemental tests individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'test_id': {
                'tests': {...},  # Note: "tests" contient ici entities observées
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    
    Note R0:
        Pouvoir discriminant tests calculable via cross_profiling
        (variance inter-entities, effect_size, etc.)
        Pas d'enrichissement spécifique axe test pour R0.
    """
    return _profile_entity_axis(observations, 'test', 'test_name')


def compare_tests_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-tests.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_tests()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'test')


def profile_all_gammas(observations: List[dict]) -> dict:
    """
    Profil comportemental gammas individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'GAM-001': {
                'tests': {
                    'SYM-001': {...},
                    'SPE-001': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    """
    return _profile_entity_axis(observations, 'gamma', 'gamma_id')


def compare_gammas_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-gammas.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_gammas()
    
    Returns:
        {
            'by_regime': {
                'CONSERVES_SYMMETRY': ['GAM-001', 'GAM-004'],
                ...
            },
            'by_test': {
                'SYM-001': {
                    'best_conservation': 'GAM-001',
                    'worst_conservation': 'GAM-013',
                    'ranking': ['GAM-001', ...]
                },
                ...
            }
        }
    """
    return _compare_entities_summary(profiles, 'gamma')


def profile_all_modifiers(observations: List[dict]) -> dict:
    """
    Profil comportemental modifiers individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'M0': {
                'tests': {
                    'SYM-001': {...},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            'M1': {...},
            'M2': {...}
        }
    """
    return _profile_entity_axis(observations, 'modifier', 'modifier_id')


def compare_modifiers_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-modifiers.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_modifiers()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'modifier')


def profile_all_encodings(observations: List[dict]) -> dict:
    """
    Profil comportemental encodings individuels.
    
    CONVENTION : profile_all_{axis}()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        observations: Toutes observations
    
    Returns:
        Structure unifiée conforme R5.1-A :
        {
            'SYM-001': {
                'tests': {...},
                'n_tests': int,
                'n_total_runs': int
            },
            'ASY-001': {...},
            'R3-001': {...},
            ...
        }
    """
    return _profile_entity_axis(observations, 'encoding', 'd_encoding_id')


def compare_encodings_summary(profiles: dict) -> dict:
    """
    Comparaisons inter-encodings.
    
    CONVENTION : compare_{axis}_summary()
    DÉCOUVRABLE : profiling_runner détecte automatiquement
    
    Args:
        profiles: Résultats profile_all_encodings()
    
    Returns:
        {
            'by_regime': {...},
            'by_test': {...}
        }
    """
    return _compare_entities_summary(profiles, 'encoding')