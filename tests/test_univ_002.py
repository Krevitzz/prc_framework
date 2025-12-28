"""
Test UNIV-002 : Diversité globale (écart-type)

Applicable : Tous rangs, tous types D
Mesure : σ(D_flat) à chaque itération
"""

import numpy as np

TEST_ID = "UNIV-002"
REQUIRES_RANK = None  # Tous rangs
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 2.0


def _safe_std(state):
    """
    Calcul sécurisé de l'écart-type.
    
    Returns:
        float ou None si erreur
    """
    flat = state.flatten()
    
    # Vérifier valeurs finies
    if not np.all(np.isfinite(flat)):
        return None
    
    # Vérifier si constant
    if np.allclose(flat, flat[0]):
        return 0.0
    
    try:
        # Méthode robuste pour éviter overflow
        mean = np.mean(flat)
        centered = flat - mean
        max_abs = np.max(np.abs(centered))
        
        if max_abs > 0:
            centered_normalized = centered / max_abs
            std_normalized = np.std(centered_normalized)
            return float(std_normalized * max_abs)
        else:
            return 0.0
    except:
        return None


def run_test(history, context):
    """
    Observe l'évolution de la diversité globale.
    
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
    
    # Calcul diversités
    diversities = []
    for i, state in enumerate(history):
        div = _safe_std(state)
        if div is None:
            # NaN/Inf détecté
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf détecté à itération {i}',
                'initial_value': diversities[0] if diversities else 0.0,
                'final_value': 0.0,
                'transition': 'nan_detected',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        diversities.append(div)
    
    # Calcul transition (factuelle)
    initial = diversities[0]
    final = diversities[-1]
    max_div = max(diversities)
    min_div = min(diversities)
    
    # Ratio (avec protection division par zéro)
    if initial < 1e-10:
        if final < 1e-10:
            ratio = 1.0
        else:
            ratio = float('inf')
    else:
        ratio = final / initial
    
    # Transition factuelle
    if ratio < 0.1:
        transition = "collapsed"
    elif abs(ratio - 1.0) < 0.1:
        transition = "stable"
    elif ratio > 1.0:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    # Tendance
    if len(diversities) > 1:
        valid_idx = np.where(np.isfinite(diversities))[0]
        if len(valid_idx) > 1:
            try:
                trend_coef = np.polyfit(valid_idx, np.array(diversities)[valid_idx], 1)[0]
            except:
                trend_coef = 0.0
        else:
            trend_coef = 0.0
    else:
        trend_coef = 0.0
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Diversité: {initial:.3e} → {final:.3e} (ratio={ratio:.3f})',
        'initial_value': initial,
        'final_value': final,
        'transition': transition,
        'value': ratio,  # Ratio comme valeur principale
        'metadata': {
            'max_diversity': max_div,
            'min_diversity': min_div,
            'ratio': ratio,
            'trend_coefficient': trend_coef,
            'diversities': diversities
        }
    }