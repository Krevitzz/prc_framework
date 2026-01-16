# tests/utilities/timeline_utils.py
"""
Timeline Utilities - Construction timelines dynamiques compositionnels.

RESPONSABILITÉS :
- Classification timing (early/mid/late)
- Composition descripteurs {timing}_{event}_then_{event}
- Extraction séquences depuis dynamic_events
- Seuils globaux relatifs (pas absolus)

ARCHITECTURE :
- classify_timing() : early/mid/late selon seuils globaux
- compute_timeline_descriptor() : Composition automatique phases
- extract_dynamic_events() : Parsing observation_data

PRINCIPE R0 :
- Toute notion temporelle est RELATIVE (jamais absolue)
- Seuils globaux uniques (pas de variation par test)
- Composition automatique : {timing}_{event}_then_{event}
- Descriptif pas causal ("then" pas "causes")

UTILISATEURS :
- gamma_profiling.py (timelines gamma)
- Futurs : analyses temporelles avancées
"""

from typing import Dict, List, Tuple


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
# CLASSIFICATION TIMING
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


# =============================================================================
# COMPOSITION TIMELINES
# =============================================================================

def compute_timeline_descriptor(
    sequence: List[str],
    sequence_timing_relative: List[float],
    oscillatory_global: bool = False
) -> Dict:
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
    
    Examples:
        >>> desc = compute_timeline_descriptor(
        ...     ['deviation', 'saturation'],
        ...     [0.05, 0.80]
        ... )
        >>> desc['timeline_compact']
        'early_deviation_then_saturation'
        
        >>> desc = compute_timeline_descriptor([], [], True)
        >>> desc['timeline_compact']
        'no_significant_dynamics'
    """
    # Vérifier cohérence listes
    if not sequence or not sequence_timing_relative:
        return {
            'phases': [],
            'timeline_compact': 'no_significant_dynamics',
            'n_phases': 0,
            'oscillatory_global': oscillatory_global
        }
    
    # Vérifier longueurs égales
    if len(sequence) != len(sequence_timing_relative):
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
# EXTRACTION ÉVÉNEMENTS
# =============================================================================

def extract_dynamic_events(observation: Dict, metric_name: str) -> Dict:
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
            'sequence_timing_relative': [...],
            'saturation_onset_estimated': bool,
            'oscillatory_global': bool
        }
    
    Notes:
        - Retourne valeurs par défaut si dynamic_events absent
        - Compatible fallback (test_engine avant enrichissement)
    
    Examples:
        >>> events = extract_dynamic_events(obs, 'asymmetry_norm')
        >>> events['sequence']
        ['deviation', 'instability', 'saturation']
        >>> events['sequence_timing_relative']
        [0.05, 0.15, 0.80]
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


def extract_metric_timeseries(
    observation: Dict,
    metric_name: str
) -> Tuple[List[float], bool]:
    """
    Extrait série temporelle avec marqueur fallback.
    
    Returns:
        (values, is_fallback)
        - values: liste valeurs ou None
        - is_fallback: True si proxy linéaire utilisé
    
    Notes:
        - Fallback : linspace(initial, final) si timeseries absent
        - Utilisé pour visualisations/analyses temporelles avancées
    
    Examples:
        >>> values, is_fallback = extract_metric_timeseries(obs, 'asymmetry_norm')
        >>> is_fallback
        False  # Timeseries disponible
        >>> len(values)
        200  # n_iterations
    """
    obs_data = observation.get('observation_data', {})
    timeseries = obs_data.get('timeseries', {})
    
    values = timeseries.get(metric_name)
    
    if values is not None:
        return list(values), False
    
    # Fallback : proxy linéaire depuis statistics
    stats = obs_data.get('statistics', {}).get(metric_name, {})
    initial = stats.get('initial')
    final = stats.get('final')
    
    if initial is not None and final is not None:
        import numpy as np
        n_iterations = obs_data.get('metadata', {}).get('n_iterations', 200)
        return list(np.linspace(initial, final, n_iterations)), True
    
    return None, True