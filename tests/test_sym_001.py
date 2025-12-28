"""
Test SYM-001 : Symétrie (préservation/création/destruction)

Applicable : Rang 2 uniquement
Mesure : ||D - D^T|| à chaque itération
"""

import numpy as np

TEST_ID = "SYM-001"
REQUIRES_RANK = 2
D_TYPES = ["SYM", "ASY"]
WEIGHT_DEFAULT = 1.5


def _compute_asymmetry(matrix, tol=1e-6):
    """
    Calcule l'asymétrie d'une matrice.
    
    Returns:
        (asymmetry_norm, is_symmetric) ou (None, None) si erreur
    """
    if matrix.ndim != 2:
        return None, None
    
    diff = matrix - matrix.T
    
    if np.any(np.isnan(diff)) or np.any(np.isinf(diff)):
        return None, None
    
    try:
        asym_norm = float(np.linalg.norm(diff, 'fro'))
        is_sym = asym_norm < tol
        return asym_norm, is_sym
    except:
        return None, None


def run_test(history, context):
    """
    Observe l'évolution de la symétrie.
    
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
    
    # Vérifier applicabilité
    if history[0].ndim != 2:
        return {
            'test_name': TEST_ID,
            'status': 'NOT_APPLICABLE',
            'message': 'Test nécessite rang 2',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'not_applicable',
            'value': 0.0,
            'metadata': {}
        }
    
    # Calcul asymétries
    asymmetries = []
    for i, state in enumerate(history):
        asym, is_sym = _compute_asymmetry(state)
        if asym is None:
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'NaN/Inf à itération {i}',
                'initial_value': 0.0,
                'final_value': 0.0,
                'transition': 'error',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        asymmetries.append(asym)
    
    # Normaliser par norme de l'état
    norms = [np.linalg.norm(state, 'fro') for state in history]
    relative_asymmetries = [a / (n + 1e-10) for a, n in zip(asymmetries, norms)]
    
    # État initial/final
    initial_asym, initial_sym = _compute_asymmetry(history[0])
    final_asym, final_sym = _compute_asymmetry(history[-1])
    
    max_asym = max(relative_asymmetries)
    mean_asym = float(np.mean(relative_asymmetries))
    
    # Déterminer transition (factuelle)
    if initial_sym and final_sym:
        transition = "preserved"
    elif not initial_sym and final_sym:
        transition = "created"
    elif initial_sym and not final_sym:
        transition = "destroyed"
    else:
        transition = "absent"
    
    # Tendance
    if len(asymmetries) > 1:
        try:
            trend_coef = np.polyfit(range(len(asymmetries)), asymmetries, 1)[0]
        except:
            trend_coef = 0.0
        
        if abs(trend_coef) < 1e-8:
            trend = "stable"
        elif trend_coef > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"
        trend_coef = 0.0
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Symétrie: {transition}, asym_max={max_asym:.2e}',
        'initial_value': 1.0 if initial_sym else 0.0,  # 1=symétrique, 0=asymétrique
        'final_value': 1.0 if final_sym else 0.0,
        'transition': transition,
        'value': max_asym,  # Max asymétrie comme valeur principale
        'metadata': {
            'max_asymmetry': max_asym,
            'mean_asymmetry': mean_asym,
            'trend': trend,
            'trend_coefficient': trend_coef,
            'asymmetries': asymmetries
        }
    }