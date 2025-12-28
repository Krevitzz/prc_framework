"""
Test CONV-LYAPUNOV : Exposant de Lyapunov

Applicable : Tous rangs, tous types D
Mesure : λ = lim (1/t) log(||δD_t|| / ||δD_0||)
"""

import numpy as np

TEST_ID = "CONV-LYAPUNOV"
REQUIRES_RANK = None
D_TYPES = ["SYM", "ASY", "R3"]
WEIGHT_DEFAULT = 1.5


def run_test(history, context):
    """
    Estime l'exposant de Lyapunov (proxy via distances successives).
    
    Args:
        history: Liste des états [D_0, ..., D_T]
        context: dict {d_base_id, gamma_id, seed, modifier_id}
    
    Returns:
        dict conforme Section 14.8.2
    """
    # Vérification historique
    if not history or len(history) < 10:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Historique trop court (nécessite ≥10 itérations)',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    # Calcul distances successives
    distances = []
    for i in range(len(history) - 1):
        diff = history[i+1] - history[i]
        
        if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf à itération {i}',
                'initial_value': 0.0,
                'final_value': 0.0,
                'transition': 'nan_detected',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        
        if diff.ndim == 2:
            dist = np.linalg.norm(diff, 'fro')
        else:
            dist = np.linalg.norm(diff.flatten())
        
        distances.append(dist)
    
    # Filtrer zéros (log impossible)
    distances_nonzero = [d for d in distances if d > 1e-12]
    
    if len(distances_nonzero) < 5:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Pas assez de distances non-nulles',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'insufficient_data',
            'value': 0.0,
            'metadata': {}
        }
    
    # Estimer λ via régression log(dist) vs t
    log_distances = np.log(distances_nonzero)
    t = np.arange(len(log_distances))
    
    try:
        lambda_estimate = float(np.polyfit(t, log_distances, 1)[0])
    except:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Erreur régression linéaire',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    # Transition factuelle
    if lambda_estimate < -0.01:
        transition = "stable"
    elif lambda_estimate > 0.1:
        transition = "chaotic"
    else:
        transition = "neutral"
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Exposant Lyapunov: λ={lambda_estimate:.4f}',
        'initial_value': distances_nonzero[0],
        'final_value': distances_nonzero[-1],
        'transition': transition,
        'value': lambda_estimate,  # λ comme valeur principale
        'metadata': {
            'lambda_estimate': lambda_estimate,
            'n_distances': len(distances_nonzero),
            'log_distances': log_distances.tolist()
        }
    }