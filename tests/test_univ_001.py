"""
Test UNIV-001 : Évolution norme Frobenius

Applicable : Tous rangs, tous types D
Mesure : ||D||_F à chaque itération
"""

import numpy as np

TEST_ID = "UNIV-001"
REQUIRES_RANK = None  # Tous rangs
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 2.0


def _safe_norm(state):
    """
    Calcul sécurisé de la norme Frobenius.
    
    Returns:
        float ou None si erreur
    """
    if np.any(np.isnan(state)) or np.any(np.isinf(state)):
        return None
    
    try:
        if state.ndim == 2:
            return float(np.linalg.norm(state, 'fro'))
        else:
            return float(np.linalg.norm(state.flatten()))
    except:
        return None


def run_test(history, context):
    """
    Observe l'évolution de la norme Frobenius.
    
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
    
    # Calcul normes
    norms = []
    for i, state in enumerate(history):
        norm = _safe_norm(state)
        if norm is None:
            # NaN/Inf détecté
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf détecté à itération {i}',
                'initial_value': norms[0] if norms else 0.0,
                'final_value': 0.0,
                'transition': 'nan_detected',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        norms.append(norm)
    
    # Calcul transition (factuelle)
    initial = norms[0]
    final = norms[-1]
    max_norm = max(norms)
    min_norm = min(norms)
    
    relative_change = abs(final - initial) / (initial + 1e-10)
    
    if max_norm > 1000.0:
        transition = "explosive"
    elif relative_change < 0.1:
        transition = "stable"
    elif final > initial:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    # Tendance
    if len(norms) > 1:
        trend_coef = np.polyfit(range(len(norms)), norms, 1)[0]
    else:
        trend_coef = 0.0
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Norme: {initial:.2f} → {final:.2f}',
        'initial_value': initial,
        'final_value': final,
        'transition': transition,
        'value': final,
        'metadata': {
            'max_norm': max_norm,
            'min_norm': min_norm,
            'relative_change': relative_change,
            'trend_coefficient': trend_coef,
            'norms': norms
        }
    }