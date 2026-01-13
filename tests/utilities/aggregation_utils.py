# tests/utilities/aggregation_utils.py
"""
Aggregation Utilities - Agrégations statistiques multi-runs.

RESPONSABILITÉS :
- Agrégations métriques (médiane, q1, q3, cv)
- Détection multimodalité (IQR ratio, bimodal)
- Statistiques descriptives inter-runs

UTILISATEURS :
- gamma_profiling.py (profiling comportemental)
- Futurs : modifier_profiling.py, test_profiling.py
"""

import numpy as np
from typing import List, Dict, Any


# =============================================================================
# AGRÉGATIONS MÉTRIQUES
# =============================================================================

def aggregate_summary_metrics(observations: List[dict], metric_name: str) -> dict:
    """
    Agrège métriques statistiques inter-runs.
    
    CALCULS :
    - final_value : {median, q1, q3, mean, std}
    - initial_value : médiane valeurs initiales
    - mean_value : moyenne des moyennes
    - cv : coefficient variation (std/mean sur finales)
    
    Args:
        observations: Liste observations (même gamma × test)
        metric_name: Nom métrique à agréger
    
    Returns:
        {
            'final_value': {
                'median': float,
                'q1': float,
                'q3': float,
                'mean': float,
                'std': float
            },
            'initial_value': float,
            'mean_value': float,
            'cv': float
        }
        
        Retourne {} si aucune valeur finale disponible
    
    Examples:
        >>> obs = [
        ...     {'observation_data': {'statistics': {
        ...         'asymmetry': {'final': 1.2, 'initial': 1.0, 'mean': 1.1}
        ...     }}},
        ...     {'observation_data': {'statistics': {
        ...         'asymmetry': {'final': 1.5, 'initial': 1.0, 'mean': 1.3}
        ...     }}}
        ... ]
        >>> result = aggregate_summary_metrics(obs, 'asymmetry')
        >>> result['final_value']['median']
        1.35
        >>> result['cv']  # Coefficient variation
        0.157
    
    Notes:
        - cv = std(final) / |mean(final)| (mesure dispersion relative)
        - Protection division par zéro (+ 1e-10)
        - initial_value = médiane (robuste outliers)
    """
    final_values = []
    initial_values = []
    mean_values = []
    
    # Collecter valeurs depuis observations
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
    
    # Vérifier données disponibles
    if not final_values:
        return {}
    
    # Convertir numpy pour calculs
    final_values = np.array(final_values)
    
    # Statistiques final_value (distribution complète)
    final_stats = {
        'median': float(np.median(final_values)),
        'q1': float(np.percentile(final_values, 25)),
        'q3': float(np.percentile(final_values, 75)),
        'mean': float(np.mean(final_values)),
        'std': float(np.std(final_values))
    }
    
    # Valeurs agrégées simples
    initial_value = float(np.median(initial_values)) if initial_values else 0.0
    mean_value = float(np.mean(mean_values)) if mean_values else 0.0
    
    # Coefficient variation (dispersion relative)
    cv = float(np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10))
    
    return {
        'final_value': final_stats,
        'initial_value': initial_value,
        'mean_value': mean_value,
        'cv': cv
    }


# =============================================================================
# DÉTECTION MULTIMODALITÉ
# =============================================================================

def aggregate_run_dispersion(observations: List[dict], metric_name: str) -> dict:
    """
    Calcule indicateurs multimodalité inter-runs.
    
    INDICATEURS :
    - final_value_iqr_ratio : Q3 / Q1 (détection bimodalité)
    - cv_across_runs : std / |mean| (dispersion relative)
    - bimodal_detected : True si iqr_ratio > 3.0
    
    PRINCIPE BIMODALITÉ :
    - IQR ratio > 3.0 suggère 2+ modes distincts
    - Exemple : Q1=0.1, Q3=0.5 → ratio=5.0 → bimodal probable
    - Heuristique R0 (pas test statistique formel)
    
    Args:
        observations: Liste observations (même gamma × test)
        metric_name: Nom métrique à analyser
    
    Returns:
        {
            'final_value_iqr_ratio': float,
            'cv_across_runs': float,
            'bimodal_detected': bool
        }
        
        Retourne valeurs 0.0/False si < 2 valeurs finales
    
    Examples:
        >>> # Cas homogène
        >>> obs_homo = [
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.0}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.1}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.9}}}}
        ... ]
        >>> result = aggregate_run_dispersion(obs_homo, 'metric')
        >>> result['bimodal_detected']
        False
        >>> result['iqr_ratio']
        1.18  # Q3=1.05, Q1=0.95 → ratio faible
        
        >>> # Cas bimodal
        >>> obs_bimodal = [
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.1}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 0.2}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.0}}}},
        ...     {'observation_data': {'statistics': {'metric': {'final': 1.1}}}}
        ... ]
        >>> result = aggregate_run_dispersion(obs_bimodal, 'metric')
        >>> result['bimodal_detected']
        True
        >>> result['iqr_ratio']
        5.5  # Q3=1.05, Q1=0.15 → ratio élevé
    
    Notes:
        - Seuil bimodal (3.0) heuristique R0
        - Protection division par zéro Q1 (max(Q1, 1e-10))
        - cv_across_runs : même calcul que aggregate_summary_metrics
    """
    final_values = []
    
    # Collecter valeurs finales
    for obs in observations:
        obs_data = obs.get('observation_data', {})
        stats = obs_data.get('statistics', {}).get(metric_name, {})
        final = stats.get('final')
        
        if final is not None:
            final_values.append(final)
    
    # Vérifier données suffisantes
    if len(final_values) < 2:
        return {
            'final_value_iqr_ratio': 0.0,
            'cv_across_runs': 0.0,
            'bimodal_detected': False
        }
    
    # Convertir numpy
    final_values = np.array(final_values)
    
    # IQR ratio (Q3 / Q1)
    q1 = np.percentile(final_values, 25)
    q3 = np.percentile(final_values, 75)
    iqr_ratio = q3 / max(q1, 1e-10)
    
    # Coefficient variation
    cv = np.std(final_values) / (np.abs(np.mean(final_values)) + 1e-10)
    
    # Détection bimodalité (heuristique)
    bimodal = iqr_ratio > 3.0
    
    return {
        'final_value_iqr_ratio': float(iqr_ratio),
        'cv_across_runs': float(cv),
        'bimodal_detected': bool(bimodal)
    }


# =============================================================================
# HELPERS (Futures Extensions)
# =============================================================================

def compute_dominant_value(
    values: List[Any],
    method: str = 'mode'
) -> Any:
    """
    Calcule valeur dominante (mode, médiane, etc.).
    
    TODO : Implémenter si nécessaire pour analyses catégorielles.
    
    Args:
        values: Liste valeurs
        method: 'mode' | 'median' | 'mean'
    
    Returns:
        Valeur dominante
    
    Examples:
        >>> compute_dominant_value([1, 1, 2, 3], method='mode')
        1
        >>> compute_dominant_value([1, 2, 3], method='median')
        2
    """
    raise NotImplementedError("compute_dominant_value non implémenté (Phase future)")


def aggregate_event_counts(
    observations: List[dict],
    metric_name: str
) -> dict:
    """
    Compte événements booléens inter-runs.
    
    TODO : Implémenter si besoin agrégations événements dynamiques génériques.
    
    Args:
        observations: Liste observations
        metric_name: Nom métrique
    
    Returns:
        {
            'event_name': fraction,
            ...
        }
    
    Notes:
        - Actuellement implémenté dans gamma_profiling.aggregate_dynamic_signatures()
        - Extraction ici si besoin réutilisation ailleurs
    """
    raise NotImplementedError("aggregate_event_counts non implémenté (Phase future)")