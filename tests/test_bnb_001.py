"""
Test BND-001 : Respect bornes initiales

Applicable : Tous rangs, tous types D
Mesure : min(D_t), max(D_t) à chaque itération
"""

import numpy as np

TEST_ID = "BND-001"
REQUIRES_RANK = None
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 1.0


def _compute_bounds(state):
    """
    Calcule min/max d'un état.
    
    Returns:
        (min_val, max_val) ou (None, None) si erreur
    """
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return None, None
    
    try:
        min_val = float(np.min(state))
        max_val = float(np.max(state))
        return min_val, max_val
    except:
        return None, None


def run_test(history, context):
    """
    Observe le respect des bornes initiales.
    
    Args:
        history: Liste des états [D_0, ..., D_T]
        context: dict {d_base_id, gamma_id, seed, modifier_id}
    
    Returns:
        dict conforme Section 14.8.2
    """
    # Vérification historique
    if not history or len(history) < 2:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Historique insuffisant',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    # Bornes initiales
    initial_min, initial_max = _compute_bounds(history[0])
    if initial_min is None:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'NaN/Inf dans état initial',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    initial_range = initial_max - initial_min
    
    # Bornes finales
    final_min, final_max = _compute_bounds(history[-1])
    if final_min is None:
        # Trouver où ça a planté
        for i, state in enumerate(history):
            if _compute_bounds(state)[0] is None:
                return {
                    'test_name': TEST_ID,
                    'status': 'ERROR',
                    'message': f'NaN/Inf à itération {i}',
                    'initial_value': initial_range,
                    'final_value': 0.0,
                    'transition': 'nan_detected',
                    'value': 0.0,
                    'metadata': {'first_error_iteration': i}
                }
    
    final_range = final_max - final_min
    
    # Calcul violation
    tolerance_factor = 1.1
    tolerance = initial_range * (tolerance_factor - 1.0)
    
    tolerated_min = initial_min - tolerance
    tolerated_max = initial_max + tolerance
    
    max_violation_low = max(0, tolerated_min - final_min)
    max_violation_high = max(0, final_max - tolerated_max)
    max_violation = max(max_violation_low, max_violation_high)
    
    # Transition factuelle
    if final_min >= tolerated_min and final_max <= tolerated_max:
        transition = "respected"
    elif max_violation > 2 * tolerance:
        transition = "severe_violation"
    else:
        transition = "minor_violation"
    
    # Changement de plage
    range_ratio = final_range / (initial_range + 1e-10)
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Bornes: [{initial_min:.3f}, {initial_max:.3f}] → [{final_min:.3f}, {final_max:.3f}]',
        'initial_value': initial_range,
        'final_value': final_range,
        'transition': transition,
        'value': max_violation,  # Violation comme valeur principale
        'metadata': {
            'initial_bounds': (initial_min, initial_max),
            'final_bounds': (final_min, final_max),
            'max_violation': max_violation,
            'range_ratio': range_ratio,
            'tolerance_factor': tolerance_factor
        }
    }