"""
Test UNIV-003 : Convergence vers point fixe

Applicable : Tous rangs, tous types D
Mesure : ||D_{t+1} - D_t|| à chaque itération
"""

import numpy as np

TEST_ID = "UNIV-003"
REQUIRES_RANK = None
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 2.0


def _safe_distance(state1, state2):
    """
    Calcul sécurisé de la distance entre deux états.
    
    Returns:
        float ou None si erreur
    """
    diff = state2 - state1
    
    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        return None
    
    try:
        if diff.ndim == 2:
            return float(np.linalg.norm(diff, 'fro'))
        else:
            return float(np.linalg.norm(diff.flatten()))
    except:
        return None


def run_test(history, context):
    """
    Observe la convergence vers un point fixe.
    
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
    
    # Calcul distances successives
    distances = []
    for i in range(len(history) - 1):
        dist = _safe_distance(history[i], history[i+1])
        if dist is None:
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf à itération {i}',
                'initial_value': distances[0] if distances else 0.0,
                'final_value': 0.0,
                'transition': 'nan_detected',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        distances.append(dist)
    
    # Statistiques
    initial_dist = distances[0]
    final_dist = distances[-1]
    mean_dist = float(np.mean(distances))
    
    # Détection convergence (fenêtre de 10 itérations)
    converged = False
    convergence_iter = None
    threshold = 1e-6
    window = 10
    
    for i in range(len(distances) - window + 1):
        if all(d < threshold for d in distances[i:i+window]):
            converged = True
            convergence_iter = i
            break
    
    # Tendance
    if len(distances) > 1:
        try:
            trend_coef = np.polyfit(range(len(distances)), distances, 1)[0]
        except:
            trend_coef = 0.0
        
        if trend_coef < -1e-6:
            trend = "decreasing"
        elif trend_coef > 1e-6:
            trend = "increasing"
        else:
            trend = "stable"
    else:
        trend = "stable"
        trend_coef = 0.0
    
    # Transition factuelle
    if converged:
        # Mesurer distance au point initial (trivialité)
        initial_state = history[0]
        final_state = history[-1]
        
        if final_state.ndim == 2:
            dist_to_initial = np.linalg.norm(final_state - initial_state, 'fro')
        else:
            dist_to_initial = np.linalg.norm((final_state - initial_state).flatten())
        
        if dist_to_initial < 1e-3:
            transition = "trivial_convergence"
        else:
            transition = "non_trivial_convergence"
    else:
        if trend == "increasing":
            transition = "divergent"
        else:
            transition = "no_convergence"
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Distance finale: {final_dist:.2e}, trend: {trend}',
        'initial_value': initial_dist,
        'final_value': final_dist,
        'transition': transition,
        'value': final_dist,
        'metadata': {
            'mean_distance': mean_dist,
            'converged': converged,
            'convergence_iteration': convergence_iter,
            'trend': trend,
            'trend_coefficient': trend_coef,
            'distances': distances
        }
    }