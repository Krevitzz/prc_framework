# tests/utilities/gamma_profiling.py
"""
Gamma Profiling Module - Charter R0 avec Timelines Compositionnels.

ARCHITECTURE REFACTORISÉE (Phase 2.2) :
- Délégation timelines → timeline_utils.py
- Délégation agrégations → aggregation_utils.py
- Délégation régimes → regime_utils.py
- Cœur métier : profiling comportemental gammas individuels

RESPONSABILITÉS CONSERVÉES :
- Agrégation signatures dynamiques (spécifique profiling)
- Calcul profil PRC complet
- Profiling tous gammas × tests
- Comparaisons inter-gammas
- Rankings
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
    Agrège signatures événements + timelines.
    
    Utilise compute_timeline_descriptor() pour chaque run,
    puis agrège par Counter.
    
    ⚠️ RESTE LOCAL : Spécifique au profiling gamma (composition timeline + événements)
    
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
# PROFIL PRC (LOCAL - cœur métier profiling)
# =============================================================================

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
    Génère profil PRC avec confidence heuristique documentée.
    
    ⚠️ RESTE LOCAL : Cœur métier spécifique gamma profiling
    
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
    
    # Régime (avec qualificatif MIXED si applicable + propriété spécifique)
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
        behavior = 'stable'  # ← Techniquement stable (pas de dynamique)
    elif base_regime == 'SATURATES_HIGH':
        behavior = 'stable'  # ← Converge vers plateau
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
    
    ⚠️ REFACTORISÉ : Utilise aggregation_utils, timeline_utils, regime_utils
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
    # ✅ DÉLÉGUÉ → aggregation_utils
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
    # ✅ DÉLÉGUÉ → aggregation_utils
    run_dispersion = aggregate_run_dispersion(observations, metric_name)
    # ⚠️ LOCAL : Spécifique profiling
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
    # ⚠️ LOCAL : Cœur métier profiling (utilise regime_utils.classify_regime)
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
        'gamma_id': gamma_id,
        'prc_profile': prc_profile,
        'diagnostic_signature': diagnostic_signature,
        'instrumentation': instrumentation
    }


# =============================================================================
# PROFILING COMPLET + COMPARAISONS
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
    """
    Synthèse comparative gammas.
    
    ⚠️ REFACTORISÉ : Utilise regime_utils.extract_conserved_properties
    """
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