# tests/test_xxx_nnn.py
"""
Test XXX-NNN : [Description courte]

Applicable : [Conditions]
Mesure : [Formule/métrique]
"""

import numpy as np
from tests.utilities.parameters import get_test_parameters

TEST_ID = "XXX-NNN"
REQUIRES_RANK = None  # ou 2, 3
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 1.0


def _safe_compute(state):
    """
    Helper avec gestion NaN/Inf.
    
    Retourne:
        float ou None si erreur
    """
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return None
    
    try:
        # Calcul sécurisé
        return float(np.linalg.norm(state))
    except:
        return None


def run_test(history, context):
    """
    Mesure [propriété factuelle].
    
    Args:
        history: Liste états [D_0, ..., D_T]
        context: dict {d_base_id, gamma_id, seed, modifier_id}
    
    Returns:
        dict conforme Section 14.8.2 v2
    """
    # =========================================================================
    # CHARGEMENT PARAMÈTRES (depuis YAML)
    # =========================================================================
    params = get_test_parameters(TEST_ID)
    threshold_param = params['some_threshold']
    
    # =========================================================================
    # VALIDATION HISTORIQUE
    # =========================================================================
    if not history or len(history) < 2:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Historique insuffisant',
            'statistics': {
                'initial': 0.0, 'final': 0.0,
                'min': 0.0, 'max': 0.0,
                'mean': 0.0, 'std': 0.0,
            },
            'evolution': {
                'transition': 'error',
                'trend': 'unknown',
                'trend_coefficient': 0.0,
            },
            'metadata': {}
        }
    
    # =========================================================================
    # VÉRIFICATION APPLICABILITÉ
    # =========================================================================
    if context['d_base_id'].startswith('R3') and REQUIRES_RANK == 2:
        return {
            'test_name': TEST_ID,
            'status': 'NOT_APPLICABLE',
            'message': 'Test nécessite rang 2',
            'statistics': {
                'initial': 0.0, 'final': 0.0,
                'min': 0.0, 'max': 0.0,
                'mean': 0.0, 'std': 0.0,
            },
            'evolution': {
                'transition': 'not_applicable',
                'trend': 'not_applicable',
                'trend_coefficient': 0.0,
            },
            'metadata': {'required_rank': 2}
        }
    
    # =========================================================================
    # CALCUL TIMESERIES (LOCAL - PAS STOCKÉ)
    # =========================================================================
    values = []
    for i, state in enumerate(history):
        val = _safe_compute(state)
        if val is None:
            # Erreur numérique
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf détecté à itération {i}',
                'statistics': {
                    'initial': values[0] if values else 0.0,
                    'final': 0.0,
                    'min': min(values) if values else 0.0,
                    'max': max(values) if values else 0.0,
                    'mean': float(np.mean(values)) if values else 0.0,
                    'std': float(np.std(values)) if values else 0.0,
                },
                'evolution': {
                    'transition': 'nan_detected',
                    'trend': 'unknown',
                    'trend_coefficient': 0.0,
                },
                'metadata': {'first_error_iteration': i}
            }
        values.append(val)
    
    # =========================================================================
    # STATISTIQUES
    # =========================================================================
    statistics = {
        'initial': values[0],
        'final': values[-1],
        'min': min(values),
        'max': max(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
    }
    
    # =========================================================================
    # ÉVOLUTION/TENDANCE
    # =========================================================================
    # Régression linéaire
    if len(values) > 1:
        trend_coef = float(np.polyfit(range(len(values)), values, 1)[0])
    else:
        trend_coef = 0.0
    
    # Classification tendance
    if abs(trend_coef) < 1e-8:
        trend = "stable"
    elif trend_coef > 0:
        trend = "increasing"
    else:
        trend = "decreasing"
    
    # Transition factuelle (utilise params YAML)
    relative_change = abs(statistics['final'] - statistics['initial']) / \
                      (statistics['initial'] + 1e-10)
    
    if statistics['max'] > threshold_param:
        transition = "explosive"
    elif relative_change < 0.1:
        transition = "stable"
    elif statistics['final'] > statistics['initial']:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    # =========================================================================
    # RETOUR FORMAT STANDARDISÉ
    # =========================================================================
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f"Mesure: {statistics['initial']:.2f} → {statistics['final']:.2f}",
        
        'statistics': statistics,
        
        'evolution': {
            'transition': transition,
            'trend': trend,
            'trend_coefficient': trend_coef,
        },
        
        'metadata': {
            'threshold_used': threshold_param,
            'relative_change': relative_change,
        }
    }