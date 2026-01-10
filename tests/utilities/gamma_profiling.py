# tests/utilities/gamma_profiling.py
"""
Gamma Profiling Module - Charter R0 (LECTEUR PUR)

Architecture 3 niveaux :
- Niveau 1 : PRC Profile (exposé, assertions interprétées)
- Niveau 2 : Diagnostic Signature (expert/audit)
- Niveau 3 : Instrumentation (debug/R2.5)

PRINCIPE FONDAMENTAL :
Ce module NE CALCULE PAS la dynamique, il la LIT depuis db_results.
Détection événements = responsabilité TestEngine.
Gamma_profiling = agrégation + classification + exposition.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


# =============================================================================
# CHARGEMENT DONNÉES (depuis observations enrichies)
# =============================================================================

def extract_metric_timeseries(observation: dict, metric_name: str) -> Optional[np.ndarray]:
    """
    Extrait série temporelle d'une métrique depuis observation.
    
    #\TODO TEST_ENGINE: Ajouter dans observation_data lors de execute_test()
    Structure attendue :
    observation_data = {
        'statistics': {...},
        'evolution': {...},
        'timeseries': {  # ← NOUVEAU
            'metric_name': [val_0, val_1, ..., val_199],
            ...
        }
    }
    
    Args:
        observation: Dict observation complète
        metric_name: Nom métrique (ex: 'frobenius_norm')
    
    Returns:
        array (n_iterations,) ou None si absent
    """
    obs_data = observation.get('observation_data', {})
    timeseries = obs_data.get('timeseries', {})
    
    values = timeseries.get(metric_name)
    
    if values is None:
        # Fallback : reconstruire proxy depuis statistics (temporaire R0)
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        initial = stats.get('initial')
        final = stats.get('final')
        
        if initial is not None and final is not None:
            # Proxy linéaire (sera remplacé par vraies données)
            return np.linspace(initial, final, 200)
    
    return np.array(values) if values else None


def extract_dynamic_events(observation: dict, metric_name: str) -> dict:
    """
    Extrait événements dynamiques depuis observation.
    
    #\TODO TEST_ENGINE: Ajouter dans observation_data lors de execute_test()
    Structure attendue :
    observation_data = {
        'statistics': {...},
        'evolution': {...},
        'dynamic_events': {  # ← NOUVEAU
            'metric_name': {
                'deviation_onset': int | None,
                'instability_onset': int | None,
                'oscillatory': bool,
                'saturation': bool,
                'collapse': bool
            },
            ...
        }
    }
    
    CALCULS À FAIRE DANS TEST_ENGINE (detect_dynamic_events) :
    - deviation_onset : première iter où |val - initial| > 0.1 * |initial|
    - instability_onset : première iter où |diff| > P90(diffs) * 10
    - oscillatory : nb changements de signe > 10% iterations
    - saturation : std(last_20%) / mean(last_20%) < 0.05
    - collapse : any(|last_10_vals| < 1e-10) and max(|all_vals|) > 1.0
    
    Args:
        observation: Dict observation complète
        metric_name: Nom métrique
    
    Returns:
        {
            'deviation_onset': int | None,
            'instability_onset': int | None,
            'oscillatory': bool,
            'saturation': bool,
            'collapse': bool
        }
    """
    obs_data = observation.get('observation_data', {})
    dynamic_events = obs_data.get('dynamic_events', {})
    
    events = dynamic_events.get(metric_name, {})
    
    # Valeurs par défaut si absentes (R0 transitoire)
    return {
        'deviation_onset': events.get('deviation_onset'),
        'instability_onset': events.get('instability_onset'),
        'oscillatory': events.get('oscillatory', False),
        'saturation': events.get('saturation', False),
        'collapse': events.get('collapse', False)
    }


def extract_event_sequence(observation: dict, metric_name: str) -> dict:
    """
    Extrait séquence événements depuis observation.
    
    #\TODO TEST_ENGINE: Calculer dans execute_test() après detect_dynamic_events()
    Structure attendue :
    observation_data = {
        'dynamic_events': {
            'metric_name': {
                'deviation_onset': 3,
                'instability_onset': 7,
                'oscillatory': False,
                'saturation': True,
                'collapse': False,
                'sequence': ['deviation', 'instability', 'saturation'],  # ← NOUVEAU
                'sequence_timing': [3, 7, 150]  # ← NOUVEAU
            }
        }
    }
    
    LOGIQUE À IMPLÉMENTER DANS TEST_ENGINE :
    1. Extraire événements avec onset
    2. Trier par timing
    3. Construire liste ordonnée ['event1', 'event2', ...]
    
    Args:
        observation: Dict observation
        metric_name: Nom métrique
    
    Returns:
        {
            'sequence': ['deviation', 'saturation'],
            'sequence_timing': [3, 150]
        }
    """
    obs_data = observation.get('observation_data', {})
    dynamic_events = obs_data.get('dynamic_events', {}).get(metric_name, {})
    
    sequence = dynamic_events.get('sequence', [])
    timing = dynamic_events.get('sequence_timing', [])
    
    return {
        'sequence': sequence,
        'sequence_timing': timing
    }


# =============================================================================
# AGRÉGATION INTER-RUNS (lecture pure)
# =============================================================================

def aggregate_summary_metrics(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège métriques statistiques sur ensemble runs.
    
    Lit depuis observation_data['statistics'][metric_name] (existant).
    
    Returns:
        {
            'final_value': {'median': ..., 'q1': ..., 'q3': ...},
            'initial_value': float,
            'mean_value': float,
            'cv': float
        }
    """
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
    
    # Métriques agrégées
    final_stats = {
        'median': float(np.median(final_values)),
        'q1': float(np.percentile(final_values, 25)),
        'q3': float(np.percentile(final_values, 75)),
        'mean': float(np.mean(final_values)),
        'std': float(np.std(final_values))
    }
    
    initial_value = float(np.median(initial_values)) if initial_values else 0.0
    mean_value = float(np.mean(mean_values)) if mean_values else 0.0
    
    # CV global
    all_values = final_values  # Proxy, idéalement toutes itérations
    cv = float(np.std(all_values) / (np.abs(np.mean(all_values)) + 1e-10))
    
    return {
        'final_value': final_stats,
        'initial_value': initial_value,
        'mean_value': mean_value,
        'cv': cv
    }


def aggregate_run_dispersion(observations: List[dict], metric_name: str) -> dict:
    """
    Calcule indicateurs multimodalité inter-runs.
    
    Returns:
        {
            'final_value_iqr_ratio': float,
            'cv_across_runs': float,
            'bimodal_detected': bool
        }
    """
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
    Agrège signatures événements sur ensemble runs.
    
    Lit dynamic_events depuis observations (calculé par TestEngine).
    
    Returns:
        {
            'dynamic_signature': {
                'deviation_onset': float | None,
                'instability_onset': float | None,
                'oscillatory_fraction': float,
                'saturation_fraction': float,
                'collapse_fraction': float
            },
            'event_sequence': {
                'dominant_sequence': [...],
                'sequence_confidence': float,
                'sequence_variants': {...}
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
    
    # Séquences
    sequences = []
    
    for obs in observations:
        events = extract_dynamic_events(obs, metric_name)
        seq_info = extract_event_sequence(obs, metric_name)
        
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
        
        # Séquence
        sequences.append(tuple(seq_info['sequence']))
    
    # Fractions
    oscillatory_frac = oscillatory_count / n_runs if n_runs > 0 else 0.0
    saturation_frac = saturation_count / n_runs if n_runs > 0 else 0.0
    collapse_frac = collapse_count / n_runs if n_runs > 0 else 0.0
    
    # Onsets médianes
    deviation_onset_median = float(np.median(deviation_onsets)) if deviation_onsets else None
    instability_onset_median = float(np.median(instability_onsets)) if instability_onsets else None
    
    # Séquence dominante
    counter = Counter(sequences)
    if counter:
        dominant, count = counter.most_common(1)[0]
        confidence = count / n_runs
    else:
        dominant = ()
        confidence = 0.0
    
    return {
        'dynamic_signature': {
            'deviation_onset': deviation_onset_median,
            'instability_onset': instability_onset_median,
            'oscillatory_fraction': oscillatory_frac,
            'saturation_fraction': saturation_frac,
            'collapse_fraction': collapse_frac
        },
        'event_sequence': {
            'dominant_sequence': list(dominant),
            'sequence_confidence': confidence,
            'sequence_variants': dict(counter)
        }
    }


# =============================================================================
# CLASSIFICATION & PROFILING (niveau 1-2-3)
# =============================================================================

def describe_timeline(sequence: list, onset_median: Optional[float]) -> str:
    """
    Projection : séquence → prose structurée.
    
    Args:
        sequence: ['deviation', 'saturation']
        onset_median: Médiane onset premier événement
    """
    if not sequence:
        return "no_significant_dynamics"
    
    early_threshold = 20  # Itérations
    
    # Patterns simples
    if sequence == ['deviation', 'saturation']:
        if onset_median and onset_median < early_threshold:
            return "early_deviation_then_plateau"
        else:
            return "late_deviation_then_plateau"
    
    if sequence == ['instability', 'collapse']:
        return "early_failure"
    
    if sequence == ['instability', 'saturation']:
        if onset_median and onset_median < early_threshold:
            return "early_instability_then_recovery"
        else:
            return "late_instability_then_recovery"
    
    # Patterns complexes
    if len(sequence) > 2:
        return "complex_dynamics"
    
    # Simple saturation
    if sequence == ['saturation']:
        return "direct_saturation"
    
    # Simple deviation
    if sequence == ['deviation']:
        return "deviation_only"
    
    return "atypical_dynamics"


def classify_regime(
    metrics: dict,
    dynamic_sig: dict,
    event_seq: dict,
    dispersion: dict
) -> str:
    """
    Classification régime R0 (8 catégories).
    
    RÉGIMES :
    - CONSERVES_X : Stable, faible variation
    - NUMERIC_INSTABILITY : Rupture précoce + explosion
    - SATURATES_HIGH : Croissance → plateau
    - OSCILLATORY_UNSTABLE : Oscillations persistantes
    - TRIVIAL : Pas de dynamique
    - DEGRADING : Décroissance monotone
    - MIXED_BEHAVIOR : Bimodalité inter-runs
    - UNCATEGORIZED : Aucun pattern
    """
    if not metrics:
        return "NO_DATA"
    
    final = metrics['final_value']['median']
    initial = metrics['initial_value']
    cv = metrics['cv']
    
    # Multimodalité (priorité)
    if dispersion['bimodal_detected']:
        return "MIXED_BEHAVIOR"
    
    # Séquence dominante
    dominant_seq = event_seq['dominant_sequence']
    seq_confidence = event_seq['sequence_confidence']
    
    # NUMERIC_INSTABILITY
    instability_onset = dynamic_sig['instability_onset']
    if instability_onset is not None and instability_onset < 20 and final > 1e20:
        return "NUMERIC_INSTABILITY"
    
    # OSCILLATORY_UNSTABLE
    if dynamic_sig['oscillatory_fraction'] > 0.3:
        return "OSCILLATORY_UNSTABLE"
    
    # TRIVIAL
    if cv < 0.01:
        return "TRIVIAL"
    
    # CONSERVES_X
    if final < 2 * initial and cv < 0.1:
        return "CONSERVES_X"
    
    # SATURATES_HIGH
    if 'saturation' in dominant_seq and dynamic_sig['saturation_fraction'] > 0.7:
        if final > 10 * initial:
            return "SATURATES_HIGH"
    
    # DEGRADING
    if final < 0.5 * initial and dynamic_sig['collapse_fraction'] < 0.1:
        return "DEGRADING"
    
    return "UNCATEGORIZED"


def compute_prc_profile(
    metrics: dict,
    dynamic_sig: dict,
    event_seq: dict,
    dispersion: dict,
    n_runs: int,
    n_valid: int
) -> dict:
    """
    Génère profil PRC (niveau 1) : assertions interprétées.
    """
    if not metrics:
        return {
            'regime': 'NO_DATA',
            'behavior': 'unknown',
            'n_runs': n_runs,
            'n_valid': n_valid,
            'confidence': 'none'
        }
    
    # Régime
    regime = classify_regime(metrics, dynamic_sig, event_seq, dispersion)
    
    # Behavior
    if regime in ['CONSERVES_X', 'TRIVIAL', 'SATURATES_HIGH']:
        behavior = 'stable'
    elif regime in ['NUMERIC_INSTABILITY', 'OSCILLATORY_UNSTABLE']:
        behavior = 'unstable'
    elif regime == 'DEGRADING':
        behavior = 'degrading'
    elif regime == 'MIXED_BEHAVIOR':
        behavior = 'mixed'
    else:
        behavior = 'uncategorized'
    
    # Timeline
    sequence = event_seq['dominant_sequence']
    onset = dynamic_sig.get('deviation_onset') or dynamic_sig.get('instability_onset')
    timeline = describe_timeline(sequence, onset)
    
    # Dominant dynamics
    dominant_dynamics = {
        'sequence': sequence,
        'confidence': event_seq['sequence_confidence'],
        'typical_timeline': timeline
    }
    
    # Robustness
    robustness = {
        'homogeneous': not dispersion['bimodal_detected'] and dispersion['cv_across_runs'] < 0.5,
        'mixed_behavior': dispersion['bimodal_detected'],
        'numerically_stable': dynamic_sig['collapse_fraction'] < 0.1 and 
                              dynamic_sig['instability_onset'] is None
    }
    
    # Pathologies
    pathologies = {
        'numeric_instability': regime == 'NUMERIC_INSTABILITY',
        'oscillatory': dynamic_sig['oscillatory_fraction'] > 0.3,
        'collapse': dynamic_sig['collapse_fraction'] > 0.1,
        'trivial': regime == 'TRIVIAL'
    }
    
    # Confidence
    if n_valid >= 20 and not dispersion['bimodal_detected']:
        confidence = 'high'
    elif n_valid >= 10:
        confidence = 'medium'
    else:
        confidence = 'low'
    
    return {
        'regime': regime,
        'behavior': behavior,
        'dominant_dynamics': dominant_dynamics,
        'robustness': robustness,
        'pathologies': pathologies,
        'n_runs': n_runs,
        'n_valid': n_valid,
        'confidence': confidence
    }


# =============================================================================
# PROFILING PRINCIPAL
# =============================================================================

def profile_test_for_gamma(
    observations: List[dict],
    test_name: str,
    gamma_id: str
) -> dict:
    """
    Profil UN test sous UN gamma (lecture pure).
    
    Args:
        observations: Liste observations filtrées (ce gamma × ce test)
        test_name: Ex 'SYM-001'
        gamma_id: Ex 'GAM-001'
    
    Returns:
        {
            'test_name': str,
            'gamma_id': str,
            'prc_profile': {...},          # Niveau 1
            'diagnostic_signature': {...}, # Niveau 2
            'instrumentation': {...}       # Niveau 3
        }
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
    
    # Identifier métrique principale (première dans statistics)
    first_obs = observations[0]
    obs_data = first_obs.get('observation_data', {})
    stats = obs_data.get('statistics', {})
    
    if not stats:
        metric_name = 'unknown'
    else:
        metric_name = list(stats.keys())[0]
    
    n_runs = len(observations)
    n_valid = len([o for o in observations if o.get('status') == 'SUCCESS'])
    
    # NIVEAU 3 : Instrumentation
    summary_metrics = aggregate_summary_metrics(observations, metric_name)
    
    instrumentation = {
        'metric_name': metric_name,
        'summary_metrics': summary_metrics,
        'computation_metadata': {
            'profiling_version': '5.5',
            'n_runs': n_runs,
            'n_valid': n_valid
        }
    }
    
    # NIVEAU 2 : Diagnostic signature
    run_dispersion = aggregate_run_dispersion(observations, metric_name)
    event_aggregates = aggregate_dynamic_signatures(observations, metric_name)
    
    diagnostic_signature = {
        'dynamic_signature': event_aggregates['dynamic_signature'],
        'event_sequence': event_aggregates['event_sequence'],
        'run_dispersion': run_dispersion,
        'thresholds_used': {
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
        event_aggregates['event_sequence'],
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


def profile_all_gammas(observations: List[dict]) -> dict:
    """
    Profil tous gammas × tous tests.
    
    Args:
        observations: Toutes observations chargées
    
    Returns:
        {
            'GAM-001': {
                'tests': {
                    'SYM-001': {profil complet},
                    ...
                },
                'n_tests': int,
                'n_total_runs': int
            },
            ...
        }
    """
    profiles = {}
    
    # Grouper par gamma
    obs_by_gamma = defaultdict(list)
    for obs in observations:
        gamma_id = obs.get('gamma_id')
        if gamma_id:
            obs_by_gamma[gamma_id].append(obs)
    
    # Profiler chaque gamma
    for gamma_id, gamma_obs in obs_by_gamma.items():
        
        # Grouper par test
        obs_by_test = defaultdict(list)
        for obs in gamma_obs:
            test_name = obs.get('test_name')
            if test_name:
                obs_by_test[test_name].append(obs)
        
        # Profiler chaque test
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


# =============================================================================
# COMPARAISONS INTER-GAMMAS
# =============================================================================

def rank_gammas_by_test(
    profiles: dict,
    test_name: str,
    criterion: str = 'conservation'
) -> list:
    """
    Classe gammas pour un test donné.
    
    Args:
        profiles: Retour de profile_all_gammas()
        test_name: Ex 'SYM-001'
        criterion: 'conservation' | 'stability' | 'final_value'
    
    Returns:
        [('GAM-001', score), ('GAM-002', score), ...]
        Trié meilleur → pire
    """
    scores = []
    
    for gamma_id, gamma_data in profiles.items():
        test_profile = gamma_data['tests'].get(test_name)
        
        if not test_profile:
            continue
        
        prc = test_profile['prc_profile']
        
        # Score selon critère
        if criterion == 'conservation':
            if prc['regime'] == 'CONSERVES_X':
                score = 1.0
            elif prc['regime'] == 'TRIVIAL':
                score = 0.5
            else:
                score = 0.0
        
        elif criterion == 'stability':
            diag = test_profile['diagnostic_signature']
            instab = diag['dynamic_signature'].get('instability_onset') is not None
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
    
    Returns:
        {
            'by_regime': {
                'CONSERVES_X': ['GAM-001', 'GAM-002'],
                ...
            },
            'by_test': {
                'SYM-001': {
                    'best_conservation': 'GAM-001',
                    'worst_conservation': 'GAM-013',
                    'ranking': [...]
                },
                ...
            }
        }
    """
    # Regrouper par régime dominant
    by_regime = defaultdict(list)
    
    for gamma_id, gamma_data in profiles.items():
        regime_counts = Counter()
        for test_profile in gamma_data['tests'].values():
            regime = test_profile['prc_profile']['regime']
            regime_counts[regime] += 1
        
        if regime_counts:
            dominant_regime = regime_counts.most_common(1)[0][0]
            by_regime[dominant_regime].append(gamma_id)
    
    # Comparaisons par test
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