# tests/utilities/gamma_profiling.py
"""
Gamma Profiling Module - Charter R0 avec Timelines Compositionnels.

ARCHITECTURE TIMELINES :
- Seuils globaux relatifs (early/mid/late)
- Composition automatique événements
- Structure intermédiaire exploitable
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


# =============================================================================
# CONFIGURATION TIMELINES (globale, unique, documentée)
# =============================================================================

TIMELINE_THRESHOLDS = {
    'early': 0.20,  # onset < 20% durée
    'mid':   0.60,  # 20% ≤ onset ≤ 60%
    'late':  0.60   # onset > 60%
}

"""
PRINCIPE TIMELINES R0 :
- Toute notion temporelle est RELATIVE (jamais absolue)
- Seuils globaux uniques (pas de variation par test)
- Composition automatique : {timing}_{event}_then_{event}
- Descriptif pas causal ("then" pas "causes")

Exemples :
- early_instability_then_collapse
- mid_deviation_then_saturation
- late_instability_then_plateau

Structure intermédiaire disponible pour exploitation :
{
    'phases': [
        {'event': 'instability', 'timing': 'early', 'onset_relative': 0.05},
        {'event': 'collapse', 'timing': 'late', 'onset_relative': 0.85}
    ],
    'timeline_compact': 'early_instability_then_collapse',
    'n_phases': 2
}
"""


# =============================================================================
# CHARGEMENT DONNÉES (depuis observations enrichies)
# =============================================================================

def extract_metric_timeseries(observation: dict, metric_name: str) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extrait série temporelle avec marqueur fallback.
    
    Returns:
        (values, is_fallback)
        - values: array ou None
        - is_fallback: True si proxy linéaire utilisé
    """
    obs_data = observation.get('observation_data', {})
    timeseries = obs_data.get('timeseries', {})
    
    values = timeseries.get(metric_name)
    
    if values is not None:
        return np.array(values), False
    
    # Fallback : proxy linéaire depuis statistics
    stats = obs_data.get('statistics', {}).get(metric_name, {})
    initial = stats.get('initial')
    final = stats.get('final')
    
    if initial is not None and final is not None:
        n_iterations = obs_data.get('metadata', {}).get('n_iterations', 200)
        return np.linspace(initial, final, n_iterations), True
    
    return None, True


def extract_dynamic_events(observation: dict, metric_name: str) -> dict:
    """
    Extrait événements dynamiques depuis observation.
    
    Lit depuis observation_data['dynamic_events'][metric_name].
    
    Returns:
        {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool,
            'sequence': [...],
            'sequence_timing': [...],
            'sequence_timing_relative': [...]
        }
    """
    obs_data = observation.get('observation_data', {})
    dynamic_events = obs_data.get('dynamic_events', {})
    
    events = dynamic_events.get(metric_name, {})
    
    # Valeurs par défaut si absentes
    return {
        'deviation_onset': events.get('deviation_onset'),
        'instability_onset': events.get('instability_onset'),
        'oscillatory': events.get('oscillatory', False),
        'saturation': events.get('saturation', False),
        'collapse': events.get('collapse', False),
        'sequence': events.get('sequence', []),
        'sequence_timing': events.get('sequence_timing', []),
        'sequence_timing_relative': events.get('sequence_timing_relative', []),
        'saturation_onset_estimated': events.get('saturation_onset_estimated', False),
        'oscillatory_global': events.get('oscillatory_global', False)
    }


# =============================================================================
# TIMELINES COMPOSITIONNELS (réécrit complet)
# =============================================================================

def classify_timing(onset_relative: float) -> str:
    """
    Classifie timing selon seuils globaux.
    
    Args:
        onset_relative: Onset normalisé [0, 1]
    
    Returns:
        'early' | 'mid' | 'late'
    
    Examples:
        >>> classify_timing(0.05)
        'early'
        >>> classify_timing(0.45)
        'mid'
        >>> classify_timing(0.85)
        'late'
    """
    if onset_relative < TIMELINE_THRESHOLDS['early']:
        return 'early'
    elif onset_relative <= TIMELINE_THRESHOLDS['mid']:
        return 'mid'
    else:
        return 'late'


def compute_timeline_descriptor(
    sequence: list,
    sequence_timing_relative: list,
    oscillatory_global: bool = False
) -> dict:
    """
    Génère descriptor timeline compositionnel.
    
    PRINCIPE :
    - Composition automatique : {timing}_{event}_then_{event}
    - Pas de patterns hardcodés
    - Structure intermédiaire pour exploitation
    
    Args:
        sequence: ['deviation', 'saturation']
        sequence_timing_relative: [0.05, 0.80]
        oscillatory_global: Comportement oscillatoire global
    
    Returns:
        {
            'phases': [
                {'event': 'deviation', 'timing': 'early', 'onset_relative': 0.05},
                {'event': 'saturation', 'timing': 'late', 'onset_relative': 0.80}
            ],
            'timeline_compact': 'early_deviation_then_saturation',
            'n_phases': 2,
            'oscillatory_global': False
        }
    
    Cas spéciaux :
    - Aucun événement → 'no_significant_dynamics'
    - 1 événement → '{timing}_{event}_only'
    - 2+ événements → '{timing1}_{event1}_then_{event2}'
    - Oscillatoire global → préfixe 'oscillatory_'
    """
    if not sequence:
        return {
            'phases': [],
            'timeline_compact': 'no_significant_dynamics',
            'n_phases': 0,
            'oscillatory_global': oscillatory_global
        }
    
    # Construire phases structurées
    phases = []
    for event, onset_rel in zip(sequence, sequence_timing_relative):
        timing = classify_timing(onset_rel)
        phases.append({
            'event': event,
            'timing': timing,
            'onset_relative': float(onset_rel)
        })
    
    # Composition timeline_compact
    if len(phases) == 1:
        # Format : {timing}_{event}_only
        p = phases[0]
        compact = f"{p['timing']}_{p['event']}_only"
    
    elif len(phases) == 2:
        # Format : {timing1}_{event1}_then_{event2}
        p1, p2 = phases[0], phases[1]
        compact = f"{p1['timing']}_{p1['event']}_then_{p2['event']}"
    
    else:
        # 3+ phases : simplifier
        p1 = phases[0]
        pN = phases[-1]
        compact = f"{p1['timing']}_{p1['event']}_to_{pN['event']}_complex"
    
    # Préfixe oscillatoire si global
    if oscillatory_global:
        compact = f"oscillatory_{compact}"
    
    return {
        'phases': phases,
        'timeline_compact': compact,
        'n_phases': len(phases),
        'oscillatory_global': oscillatory_global
    }


# =============================================================================
# AGRÉGATION INTER-RUNS
# =============================================================================

def aggregate_summary_metrics(observations: List[dict], metric_name: str) -> dict:
    """Agrège métriques statistiques (inchangé)."""
    final_values = []
    initial_values = []
    mean_values = []
    
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        
        final = stats.get('final')
        initial = stats.get('initial')
        mean = stats.get('mean')
        
        if final is not None:
            final_values.append(final)
        if initial is not None:
            initial_values.append(initial)
        if mean is not None:
            mean_values.append(mean)
    
    if not final_values:
        return {}
    
    final_values = np.array(final_values)
    
    final_stats = {
        'median': float(np.median(final_values)),
        'q1': float(np.percentile(final_values, 25)),
        'q3': float(np.percentile(final_values, 75)),
        'mean': float(np.mean(final_values)),
        'std': float(np.std(final_values))
    }
    
    initial_value = float(np.median(initial_values)) if initial_values else 0.0
    mean_value = float(np.mean(mean_values)) if mean_values else 0.0
    
    cv = float(np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10))
    
    return {
        'final_value': final_stats,
        'initial_value': initial_value,
        'mean_value': mean_value,
        'cv': cv
    }


def aggregate_run_dispersion(observations: List[dict], metric_name: str) -> dict:
    """Calcule indicateurs multimodalité (inchangé)."""
    final_values = []
    
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        final = stats.get('final')
        
        if final is not None:
            final_values.append(final)
    
    if len(final_values) < 2:
        return {
            'final_value_iqr_ratio': 0.0,
            'cv_across_runs': 0.0,
            'bimodal_detected': False
        }
    
    final_values = np.array(final_values)
    
    q1 = np.percentile(final_values, 25)
    q3 = np.percentile(final_values, 75)
    iqr_ratio = q3 / max(q1, 1e-10)
    
    cv = np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10)
    
    bimodal = iqr_ratio > 3.0
    
    return {
        'final_value_iqr_ratio': float(iqr_ratio),
        'cv_across_runs': float(cv),
        'bimodal_detected': bool(bimodal)
    }


def aggregate_dynamic_signatures(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège signatures événements + timelines.
    
    Utilise compute_timeline_descriptor() pour chaque run,
    puis agrège par Counter.
    
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
# CLASSIFICATION RÉGIME (Patch #2 : MIXED comme qualificatif)
# =============================================================================

def classify_regime(
    metrics: dict,
    dynamic_sig: dict,
    timeline_dist: dict,
    dispersion: dict
) -> str:
    """
    Classification régime R0 avec MIXED::BASE.
    
    Patch #2 appliqué : Multimodalité = qualificatif pas régime de base.
    
    Régimes de base :
    - CONSERVES_X, NUMERIC_INSTABILITY, SATURATES_HIGH,
      OSCILLATORY_UNSTABLE, TRIVIAL, DEGRADING, UNCATEGORIZED
    
    Si bimodal détecté → MIXED::{régime_base}
    """
    if not metrics:
        return "NO_DATA"
    
    final = metrics['final_value']['median']
    initial = metrics['initial_value']
    cv = metrics['cv']
    
    # Classifier régime de base d'abord
    instability_onset = dynamic_sig.get('instability_onset_median')
    
    if instability_onset is not None and instability_onset < 20 and final > 1e20:
        base_regime = "NUMERIC_INSTABILITY"
    elif dynamic_sig['oscillatory_fraction'] > 0.3:
        base_regime = "OSCILLATORY_UNSTABLE"
    elif cv < 0.01:
        base_regime = "TRIVIAL"
    elif final < 2 * initial and cv < 0.1:
        base_regime = "CONSERVES_X"
    elif 'saturation' in timeline_dist.get('dominant_timeline', '') and dynamic_sig['saturation_fraction'] > 0.7:
        if final > 10 * initial:
            base_regime = "SATURATES_HIGH"
        else:
            base_regime = "CONSERVES_X"
    elif final < 0.5 * initial and dynamic_sig['collapse_fraction'] < 0.1:
        base_regime = "DEGRADING"
    else:
        base_regime = "UNCATEGORIZED"
    
    # Qualificatif multimodalité (Patch #2)
    if dispersion['bimodal_detected']:
        return f"MIXED::{base_regime}"
    
    return base_regime


# =============================================================================
# PROFIL PRC (Patch #4 : Confidence metadata)
# =============================================================================

def compute_prc_profile(
    metrics: dict,
    dynamic_sig: dict,
    timeline_dist: dict,
    dispersion: dict,
    n_runs: int,
    n_valid: int
) -> dict:
    """
    Génère profil PRC avec confidence heuristique documentée.
    
    Patch #4 appliqué : Confidence metadata exposée.
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
    regime = classify_regime(metrics, dynamic_sig, timeline_dist, dispersion)
    
    # Behavior
    base_regime = regime.split('::')[-1] if '::' in regime else regime
    
    if base_regime in ['CONSERVES_X', 'TRIVIAL', 'SATURATES_HIGH']:
        behavior = 'stable'
    elif base_regime in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE']:
        behavior = 'unstable'
    elif base_regime == 'DEGRADING':
        behavior = 'degrading'
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
        'trivial': base_regime == 'TRIVIAL'
    }
    
    # Confidence heuristique (Patch #4)
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


# =============================================================================
# PROFILING PRINCIPAL (Patch #1 : Fallback marqué)
# =============================================================================

def profile_test_for_gamma(
    observations: List[dict],
    test_name: str,
    gamma_id: str
) -> dict:
    """
    Profil UN test sous UN gamma.
    
    Patch #1 appliqué : Fallback timeseries marqué dans instrumentation.
    """
    if not observations:
        return {
            'test_name': test_name,
            'gamma_id': gamma_id,
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
    
    # NIVEAU 3 : Instrumentation (Patch #1)
    summary_metrics = aggregate_summary_metrics(observations, metric_name)
    
    instrumentation = {
        'metric_name': metric_name,
        'summary_metrics': summary_metrics,
        'data_completeness': {
            'dynamic_events_present': dynamic_events_present,
            'timeseries_present': timeseries_present,
            'fallback_used': not dynamic_events_present  # Si absent = fallback
        },
        'computation_metadata': {
            'profiling_version': '5.5',
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
        n_valid
    )
    
    return {
        'test_name': test_name,
        'gamma_id': gamma_id,
        'prc_profile': prc_profile,
        'diagnostic_signature': diagnostic_signature,
        'instrumentation': instrumentation
    }


# =============================================================================
# PROFILING COMPLET + COMPARAISONS (inchangé)
# =============================================================================

def profile_all_gammas(observations: List[dict]) -> dict:
    """Profil tous gammas × tous tests."""
    profiles = {}
    
    obs_by_gamma = defaultdict(list)
    for obs in observations:
        gamma_id = obs.get('gamma_id')
        if gamma_id:
            obs_by_gamma[gamma_id].append(obs)
    
    for gamma_id, gamma_obs in obs_by_gamma.items():
        obs_by_test = defaultdict(list)
        for obs in gamma_obs:
            test_name = obs.get('test_name')
            if test_name:
                obs_by_test[test_name].append(obs)
        
        test_profiles = {}
        for test_name, test_obs in obs_by_test.items():
            profile = profile_test_for_gamma(test_obs, test_name, gamma_id)
            test_profiles[test_name] = profile
        
        profiles[gamma_id] = {
            'tests': test_profiles,
            'n_tests': len(test_profiles),
            'n_total_runs': len(gamma_obs)
        }
    
    return profiles


def rank_gammas_by_test(
    profiles: dict,
    test_name: str,
    criterion: str = 'conservation'
) -> list:
    """Classe gammas pour un test donné."""
    scores = []
    
    for gamma_id, gamma_data in profiles.items():
        test_profile = gamma_data['tests'].get(test_name)
        
        if not test_profile:
            continue
        
        prc = test_profile['prc_profile']
        
        if criterion == 'conservation':
            regime = prc['regime'].split('::')[-1]  # Strip MIXED::
            if regime == 'CONSERVES_X':
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
        
        elif criterion == 'final_value':
            instr = test_profile['instrumentation']
            final = instr['summary_metrics'].get('final_value', {}).get('median', 1e100)
            score = 1.0 / (1.0 + np.log10(abs(final) + 1.0))
        
        else:
            score = 0.0
        
        scores.append((gamma_id, score))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compare_gammas_summary(profiles: dict) -> dict:
    """Synthèse comparative gammas."""
    by_regime = defaultdict(list)
    
    for gamma_id, gamma_data in profiles.items():
        regime_counts = Counter()
        for test_profile in gamma_data['tests'].values():
            regime = test_profile['prc_profile']['regime']
            regime_counts[regime] += 1
        
        if regime_counts:
            dominant_regime = regime_counts.most_common(1)[0][0]
            by_regime[dominant_regime].append(gamma_id)
    
    by_test = {}
    
    all_tests = set()
    for gamma_data in profiles.values():
        all_tests.update(gamma_data['tests'].keys())
    
    for test_name in all_tests:
        ranking = rank_gammas_by_test(profiles, test_name, 'conservation')
        
        if ranking:
            by_test[test_name] = {
                'best_conservation': ranking[0][0],
                'worst_conservation': ranking[-1][0],
                'ranking': [g for g, _ in ranking]
            }
    
    return {
        'by_regime': dict(by_regime),
        'by_test': by_test
    }