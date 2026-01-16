# tests/utilities/modifier_profiling.py
"""
Modifier Profiling Module - Charter R0

ARCHITECTURE PARALLÈLE À gamma_profiling.py :
- Délégation timelines → timeline_utils.py
- Délégation agrégations → aggregation_utils.py
- Délégation régimes → regime_utils.py
- Cœur métier : profiling comportemental modifiers individuels

RESPONSABILITÉS :
- Agrégation signatures dynamiques par modifier
- Calcul profil PRC modifier
- Profiling tous modifiers × tests
- Détection sensibilités cross-gamma et cross-encoding
- Comparaisons inter-modifiers
- Rankings descriptifs

PHILOSOPHIE R0 :
- Cataloguer signatures modifiers (pas hiérarchiser)
- Détecter patterns modifier-spécifiques
- Identifier sensibilités contextuelles
- Aucun verdict décisionnel
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict

# ============================================================================
# IMPORTS MODULES REFACTORISÉS
# ============================================================================

# Timelines et événements dynamiques
from .timeline_utils import (
    TIMELINE_THRESHOLDS,
    classify_timing,
    compute_timeline_descriptor,
    extract_dynamic_events,
    extract_metric_timeseries
)

# Agrégations statistiques
from .aggregation_utils import (
    aggregate_summary_metrics,
    aggregate_run_dispersion
)

# Classification régimes
from .regime_utils import (
    classify_regime,
    extract_conserved_properties
)


# =============================================================================
# AGRÉGATION SIGNATURES DYNAMIQUES (LOCAL - spécifique profiling)
# =============================================================================

def aggregate_dynamic_signatures(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège signatures événements + timelines pour un modifier.
    
    Identique à gamma_profiling (réutilisation pattern établi).
    
    ⚠️ RESTE LOCAL : Spécifique au profiling modifier (composition timeline + événements)
    
    Returns:
        {
            'dynamic_signature': {...},
            'timeline_distribution': {
                'dominant_timeline': 'early_instability_then_collapse',
                'timeline_confidence': 0.75,
                'timeline_variants': {...}
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


# =============================================================================
# ANALYSE SENSIBILITÉS CONTEXTUELLES (NOUVEAU - spécifique modifier)
# =============================================================================

def compute_contextual_sensitivities(observations: List[dict], metric_name: str) -> dict:
    """
    Détecte sensibilités cross-gamma et cross-encoding d'un modifier.
    
    NOUVEAU : Spécifique profiling modifier (pas dans gamma_profiling).
    
    Analyse variance inter-gammas et inter-encodings pour détecter
    si le modifier amplifie ou masque différences contextuelles.
    
    Returns:
        {
            'cross_gamma_variance': float,      # Variance inter-gammas (encodings fixés)
            'cross_encoding_variance': float,   # Variance inter-encodings (gammas fixés)
            'sensitive_gammas': List[str],      # Gammas très affectés
            'sensitive_encodings': List[str],   # Encodings très affectés
            'interaction_strength': float       # Mesure interaction modifier × contexte
        }
    """
    # Extraire final values par contexte
    final_by_gamma = defaultdict(list)
    final_by_encoding = defaultdict(list)
    
    for obs in observations:
        if obs.get('status') != 'SUCCESS':
            continue
        
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        final_val = stats.get('final')
        
        if final_val is None or not np.isfinite(final_val):
            continue
        
        gamma_id = obs.get('gamma_id')
        encoding_id = obs.get('d_encoding_id')
        
        if gamma_id:
            final_by_gamma[gamma_id].append(final_val)
        if encoding_id:
            final_by_encoding[encoding_id].append(final_val)
    
    # Variance inter-gammas (agrégée sur encodings)
    gamma_medians = {gid: np.median(vals) for gid, vals in final_by_gamma.items() if vals}
    cross_gamma_variance = np.var(list(gamma_medians.values())) if gamma_medians else 0.0
    
    # Variance inter-encodings (agrégée sur gammas)
    encoding_medians = {eid: np.median(vals) for eid, vals in final_by_encoding.items() if vals}
    cross_encoding_variance = np.var(list(encoding_medians.values())) if encoding_medians else 0.0
    
    # Gammas sensibles (écart > P75 variance)
    if gamma_medians:
        global_median = np.median(list(gamma_medians.values()))
        deviations = {gid: abs(med - global_median) for gid, med in gamma_medians.items()}
        threshold = np.percentile(list(deviations.values()), 75) if deviations else 0.0
        sensitive_gammas = [gid for gid, dev in deviations.items() if dev > threshold]
    else:
        sensitive_gammas = []
    
    # Encodings sensibles
    if encoding_medians:
        global_median = np.median(list(encoding_medians.values()))
        deviations = {eid: abs(med - global_median) for eid, med in encoding_medians.items()}
        threshold = np.percentile(list(deviations.values()), 75) if deviations else 0.0
        sensitive_encodings = [eid for eid, dev in deviations.items() if dev > threshold]
    else:
        sensitive_encodings = []
    
    # Interaction strength : ratio variance contextuelles / variance totale
    total_variance = np.var([
        val for vals in final_by_gamma.values() for val in vals
    ]) if final_by_gamma else 0.0
    
    interaction_strength = (
        (cross_gamma_variance + cross_encoding_variance) / (total_variance + 1e-10)
        if total_variance > 0 else 0.0
    )
    
    return {
        'cross_gamma_variance': float(cross_gamma_variance),
        'cross_encoding_variance': float(cross_encoding_variance),
        'sensitive_gammas': sorted(sensitive_gammas),
        'sensitive_encodings': sorted(sensitive_encodings),
        'interaction_strength': float(interaction_strength)
    }


# =============================================================================
# PROFIL PRC MODIFIER (adapté de gamma_profiling)
# =============================================================================

def compute_prc_profile(
    metrics: dict,
    dynamic_sig: dict,
    timeline_dist: dict,
    dispersion: dict,
    sensitivities: dict,
    n_runs: int,
    n_valid: int,
    test_name: str
) -> dict:
    """
    Génère profil PRC modifier avec sensibilités contextuelles.
    
    ADAPTÉ : Ajoute sensitivities aux dimensions gamma_profiling standard.
    
    Returns profil complet incluant :
    - Régime dominant
    - Comportement (stable/unstable/mixed)
    - Timeline dominante
    - Robustness
    - Pathologies
    - Sensibilités contextuelles (NOUVEAU)
    - Confidence
    """
    if not metrics:
        return {
            'regime': 'NO_DATA',
            'behavior': 'unknown',
            'n_runs': n_runs,
            'n_valid': n_valid,
            'confidence': 'none',
            'confidence_metadata': {},
            'contextual_sensitivities': sensitivities
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
        behavior = 'stable'
    elif base_regime == 'SATURATES_HIGH':
        behavior = 'stable'
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
                              dynamic_sig.get('instability_onset_median') is None,
        'context_independent': sensitivities['interaction_strength'] < 0.3  # NOUVEAU
    }
    
    # Pathologies
    pathologies = {
        'numeric_instability': 'NUMERIC_INSTABILITY' in regime,
        'oscillatory': dynamic_sig['oscillatory_fraction'] > 0.3,
        'collapse': dynamic_sig['collapse_fraction'] > 0.1,
        'trivial': base_regime == 'TRIVIAL',
        'degrading': base_regime == 'DEGRADING',
        'context_dependent': sensitivities['interaction_strength'] > 0.7  # NOUVEAU
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
        'contextual_sensitivities': sensitivities,  # NOUVEAU
        'n_runs': n_runs,
        'n_valid': n_valid,
        'confidence': confidence,
        'confidence_metadata': {
            'level': confidence,
            'criteria': confidence_criteria,
            'rationale': rationale
        }
    }


# =============================================================================
# PROFILING PRINCIPAL
# =============================================================================

def profile_test_for_modifier(
    observations: List[dict],
    test_name: str,
    modifier_id: str
) -> dict:
    """
    Profil UN test sous UN modifier.
    
    ARCHITECTURE PARALLÈLE À gamma_profiling.profile_test_for_gamma
    AJOUT : Analyse sensibilités contextuelles cross-gamma/encoding
    
    Args:
        observations: Liste observations pour (modifier_id, test_name)
                     Contient TOUS gammas × TOUS encodings
        test_name: ID test (ex: "SYM-001")
        modifier_id: ID modifier (ex: "M1")
    
    Returns:
        {
            'test_name': str,
            'modifier_id': str,
            'prc_profile': {...},
            'diagnostic_signature': {...},
            'instrumentation': {...}
        }
    """
    if not observations:
        return {
            'test_name': test_name,
            'modifier_id': modifier_id,
            'prc_profile': {
                'regime': 'NO_DATA',
                'behavior': 'unknown',
                'n_runs': 0,
                'n_valid': 0,
                'confidence': 'none',
                'contextual_sensitivities': {}
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
    # ✅ DÉLÉGUÉ → aggregation_utils
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
            'profiling_version': '6.0',
            'profiling_module': 'modifier_profiling',
            'timeline_architecture': 'compositional_relative',
            'n_runs': n_runs,
            'n_valid': n_valid,
            'n_gammas': len(set(o.get('gamma_id') for o in observations if o.get('gamma_id'))),
            'n_encodings': len(set(o.get('d_encoding_id') for o in observations if o.get('d_encoding_id')))
        }
    }
    
    # NIVEAU 2 : Diagnostic signature
    # ✅ DÉLÉGUÉ → aggregation_utils
    run_dispersion = aggregate_run_dispersion(observations, metric_name)
    # ⚠️ LOCAL : Spécifique profiling
    event_aggregates = aggregate_dynamic_signatures(observations, metric_name)
    # 🆕 NOUVEAU : Sensibilités contextuelles
    contextual_sensitivities = compute_contextual_sensitivities(observations, metric_name)
    
    diagnostic_signature = {
        'dynamic_signature': event_aggregates['dynamic_signature'],
        'timeline_distribution': event_aggregates['timeline_distribution'],
        'run_dispersion': run_dispersion,
        'contextual_sensitivities': contextual_sensitivities,  # NOUVEAU
        'thresholds_used': {
            'timeline_early': TIMELINE_THRESHOLDS['early'],
            'timeline_mid': TIMELINE_THRESHOLDS['mid'],
            'timeline_late': TIMELINE_THRESHOLDS['late'],
            'instability_detection': 'P90 * 10 (relative)',
            'oscillatory_threshold': '10% sign changes',
            'saturation_cv': '5%',
            'bimodal_iqr_ratio': '3.0',
            'sensitivity_threshold': 'P75 deviation'  # NOUVEAU
        }
    }
    
    # NIVEAU 1 : PRC Profile
    # ⚠️ LOCAL : Cœur métier profiling (utilise regime_utils.classify_regime)
    prc_profile = compute_prc_profile(
        summary_metrics,
        event_aggregates['dynamic_signature'],
        event_aggregates['timeline_distribution'],
        run_dispersion,
        contextual_sensitivities,  # NOUVEAU
        n_runs,
        n_valid,
        test_name
    )
    
    return {
        'test_name': test_name,
        'modifier_id': modifier_id,
        'prc_profile': prc_profile,
        'diagnostic_signature': diagnostic_signature,
        'instrumentation': instrumentation
    }


# =============================================================================
# PROFILING COMPLET + COMPARAISONS
# =============================================================================

def profile_all_modifiers(observations: List[dict]) -> dict:
    """
    Profil tous modifiers × tous tests.
    
    ARCHITECTURE PARALLÈLE À gamma_profiling.profile_all_gammas
    
    Args:
        observations: Liste complète observations (tous modifiers × gammas × encodings)
    
    Returns:
        {
            'M0': {
                'tests': {
                    'SYM-001': {...},
                    'UNIV-001': {...},
                    ...
                },
                'n_tests': 9,
                'n_total_runs': 1200
            },
            'M1': {...},
            'M2': {...}
        }
    """
    profiles = {}
    
    obs_by_modifier = defaultdict(list)
    for obs in observations:
        modifier_id = obs.get('modifier_id')
        if modifier_id:
            obs_by_modifier[modifier_id].append(obs)
    
    for modifier_id, modifier_obs in obs_by_modifier.items():
        obs_by_test = defaultdict(list)
        for obs in modifier_obs:
            test_name = obs.get('test_name')
            if test_name:
                obs_by_test[test_name].append(obs)
        
        test_profiles = {}
        for test_name, test_obs in obs_by_test.items():
            profile = profile_test_for_modifier(test_obs, test_name, modifier_id)
            test_profiles[test_name] = profile
        
        profiles[modifier_id] = {
            'tests': test_profiles,
            'n_tests': len(test_profiles),
            'n_total_runs': len(modifier_obs)
        }
    
    return profiles


def rank_modifiers_by_test(
    profiles: dict,
    test_name: str,
    criterion: str = 'conservation'
) -> list:
    """
    Classe modifiers pour un test donné (DESCRIPTIF, pas décisionnel).
    
    ARCHITECTURE PARALLÈLE À gamma_profiling.rank_gammas_by_test
    
    Critères disponibles :
    - 'conservation' : Fraction régimes CONSERVES_*
    - 'stability' : Absence instabilités numériques
    - 'context_independence' : Faible interaction_strength
    - 'homogeneity' : Faible variance cross-runs
    
    Returns:
        [(modifier_id, score), ...] sorted descending
    """
    scores = []
    
    for modifier_id, modifier_data in profiles.items():
        test_profile = modifier_data['tests'].get(test_name)
        
        if not test_profile:
            continue
        
        prc = test_profile['prc_profile']
        
        if criterion == 'conservation':
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            if regime.startswith('CONSERVES_'):
                score = 1.0
            elif regime == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
        
        elif criterion == 'stability':
            diag = test_profile['diagnostic_signature']
            instab = diag['dynamic_signature'].get('instability_onset_median') is not None
            collapse = diag['dynamic_signature']['collapse_fraction']
            score = 1.0 - (float(instab) * 0.5 + collapse * 0.5)
        
        elif criterion == 'context_independence':
            sens = prc.get('contextual_sensitivities', {})
            interaction = sens.get('interaction_strength', 1.0)
            score = 1.0 - interaction  # Plus faible = meilleur
        
        elif criterion == 'homogeneity':
            diag = test_profile['diagnostic_signature']
            cv = diag['run_dispersion'].get('cv_across_runs', 1.0)
            bimodal = diag['run_dispersion'].get('bimodal_detected', False)
            score = (1.0 - min(cv, 1.0)) * (0.5 if bimodal else 1.0)
        
        else:
            score = 0.0
        
        scores.append((modifier_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compare_modifiers_summary(profiles: dict) -> dict:
    """
    Synthèse comparative modifiers.
    
    ARCHITECTURE PARALLÈLE À gamma_profiling.compare_gammas_summary
    AJOUT : Analyse sensibilités contextuelles agrégées
    
    Returns:
        {
            'by_regime': {...},           # Distribution régimes par modifier
            'by_test': {...},             # Rankings par test
            'by_sensitivity': {...}       # Sensibilités contextuelles (NOUVEAU)
        }
    """
    by_regime = defaultdict(list)
    
    for modifier_id, modifier_data in profiles.items():
        regime_counts = Counter()
        for test_profile in modifier_data['tests'].values():
            regime = test_profile['prc_profile']['regime']
            regime_counts[regime] += 1
        
        if regime_counts:
            dominant_regime = regime_counts.most_common(1)[0][0]
            by_regime[dominant_regime].append(modifier_id)
    
    by_test = {}
    
    all_tests = set()
    for modifier_data in profiles.values():
        all_tests.update(modifier_data['tests'].keys())
    
    for test_name in all_tests:
        ranking = rank_modifiers_by_test(profiles, test_name, 'conservation')
        
        if ranking:
            by_test[test_name] = {
                'best_conservation': ranking[0][0],
                'worst_conservation': ranking[-1][0],
                'ranking': [m for m, _ in ranking]
            }
    
    # NOUVEAU : Analyse sensibilités contextuelles agrégées
    by_sensitivity = {}
    
    for test_name in all_tests:
        sens_scores = []
        
        for modifier_id, modifier_data in profiles.items():
            test_profile = modifier_data['tests'].get(test_name)
            if not test_profile:
                continue
            
            sens = test_profile['prc_profile'].get('contextual_sensitivities', {})
            interaction = sens.get('interaction_strength', 0.0)
            sens_scores.append((modifier_id, interaction))
        
        if sens_scores:
            sens_scores.sort(key=lambda x: x[1])
            by_sensitivity[test_name] = {
                'most_context_independent': sens_scores[0][0],
                'most_context_dependent': sens_scores[-1][0],
                'ranking': [m for m, _ in sens_scores]
            }
    
    return {
        'by_regime': dict(by_regime),
        'by_test': by_test,
        'by_sensitivity': by_sensitivity  # NOUVEAU
    }


# =============================================================================
# DÉTECTION PATTERNS MODIFIER-SPÉCIFIQUES (NOUVEAU)
# =============================================================================

def detect_modifier_signatures(profiles: dict) -> dict:
    """
    Détecte patterns récurrents spécifiques à chaque modifier.
    
    NOUVEAU : Spécifique modifier_profiling
    
    Analyse :
    - Tests systématiquement affectés
    - Gammas/encodings sensibles récurrents
    - Timelines caractéristiques
    - Pathologies induites
    
    Returns:
        {
            'M0': {
                'signature': 'neutral_baseline',
                'affected_tests': [],
                'induced_pathologies': [],
                'characteristic_timeline': 'no_significant_dynamics'
            },
            'M1': {
                'signature': 'amplifies_instabilities',
                'affected_tests': ['TOP-001', 'SPE-002'],
                'induced_pathologies': ['oscillatory'],
                'characteristic_timeline': 'early_deviation_then_oscillatory',
                'sensitive_gammas': ['GAM-004', 'GAM-009'],
                'sensitive_encodings': ['ASY-003']
            },
            ...
        }
    """
    signatures = {}
    
    for modifier_id, modifier_data in profiles.items():
        # Tests affectés (régime change vs baseline attendu)
        affected_tests = []
        
        # Pathologies induites (fréquence > 20%)
        pathology_counts = Counter()
        
        # Timelines dominantes
        timeline_counts = Counter()
        
        # Sensibilités contextuelles agrégées
        all_sensitive_gammas = []
        all_sensitive_encodings = []
        
        for test_name, test_profile in modifier_data['tests'].items():
            prc = test_profile['prc_profile']
            
            # Tests affectés
            if prc['behavior'] in ['unstable', 'degrading', 'mixed']:
                affected_tests.append(test_name)
            
            # Pathologies
            for pathology, present in prc['pathologies'].items():
                if present:
                    pathology_counts[pathology] += 1
            
            # Timelines
            timeline = prc['dominant_timeline']['timeline_compact']
            timeline_counts[timeline] += 1
            
            # Sensibilités
            sens = prc.get('contextual_sensitivities', {})
            all_sensitive_gammas.extend(sens.get('sensitive_gammas', []))
            all_sensitive_encodings.extend(sens.get('sensitive_encodings', []))
        
        n_tests = len(modifier_data['tests'])
        
        # Pathologies induites (fréquence > 20%)
        induced_pathologies = [
            p for p, count in pathology_counts.items()
            if count / n_tests > 0.2
        ]
        
        # Timeline caractéristique (plus fréquente)
        if timeline_counts:
            characteristic_timeline = timeline_counts.most_common(1)[0][0]
        else:
            characteristic_timeline = 'unknown'
        
        # Sensibilités récurrentes (présentes dans > 30% tests)
        gamma_counter = Counter(all_sensitive_gammas)
        encoding_counter = Counter(all_sensitive_encodings)
        
        sensitive_gammas = [
            g for g, count in gamma_counter.items()
            if count / n_tests > 0.3
        ]
        sensitive_encodings = [
            e for e, count in encoding_counter.items()
            if count / n_tests > 0.3
        ]
        
        # Signature textuelle
        if modifier_id == 'M0':
            signature = 'neutral_baseline'
        elif len(affected_tests) == 0:
            signature = 'minimal_impact'
        elif 'oscillatory' in induced_pathologies:
            signature = 'amplifies_instabilities'
        elif 'trivial' in induced_pathologies:
            signature = 'induces_collapse'
        elif len(sensitive_gammas) > 0 or len(sensitive_encodings) > 0:
            signature = 'context_dependent'
        else:
            signature = 'uniform_perturbation'
        
        signatures[modifier_id] = {
            'signature': signature,
            'affected_tests': affected_tests,
            'induced_pathologies': induced_pathologies,
            'characteristic_timeline': characteristic_timeline,
            'sensitive_gammas': sensitive_gammas,
            'sensitive_encodings': sensitive_encodings,
            'n_tests_analyzed': n_tests
        }
    
    return signatures