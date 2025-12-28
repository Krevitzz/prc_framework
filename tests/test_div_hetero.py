"""
Test DIV-HETERO : Hétérogénéité spatiale

Applicable : Rang 2 uniquement, shape >= 10x10
Mesure : std(std_locaux) via grille régulière
"""

import numpy as np

TEST_ID = "DIV-HETERO"
REQUIRES_RANK = 2
D_TYPES = ["SYM", "ASY"]
WEIGHT_DEFAULT = 1.0


def _compute_spatial_heterogeneity(state, grid_size=10):
    """
    Calcule hétérogénéité spatiale via grille.
    
    Returns:
        float ou None si erreur
    """
    if state.ndim != 2:
        return None
    
    h, w = state.shape
    if h < grid_size or w < grid_size:
        return None
    
    cell_h = max(1, h // grid_size)
    cell_w = max(1, w // grid_size)
    
    local_stds = []
    
    for i in range(0, h, cell_h):
        for j in range(0, w, cell_w):
            cell = state[i:min(i+cell_h, h), j:min(j+cell_w, w)]
            cell_clean = cell[np.isfinite(cell)]
            
            if len(cell_clean) > 1:
                local_stds.append(np.std(cell_clean))
    
    if len(local_stds) > 1:
        # Hétérogénéité = std des std locaux
        return float(np.std(local_stds))
    else:
        return None


def run_test(history, context):
    """
    Observe l'évolution de l'hétérogénéité spatiale.
    
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
    initial_state = history[0]
    if initial_state.ndim != 2:
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
    
    if initial_state.shape[0] < 10 or initial_state.shape[1] < 10:
        return {
            'test_name': TEST_ID,
            'status': 'NOT_APPLICABLE',
            'message': 'État trop petit pour grille 10x10',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'not_applicable',
            'value': 0.0,
            'metadata': {}
        }
    
    # Calcul hétérogénéités
    initial_hetero = _compute_spatial_heterogeneity(history[0])
    final_hetero = _compute_spatial_heterogeneity(history[-1])
    
    if initial_hetero is None or final_hetero is None:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Erreur calcul hétérogénéité',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    # Ratio
    if initial_hetero > 1e-10:
        ratio = final_hetero / initial_hetero
    else:
        ratio = 0.0 if final_hetero < 1e-10 else float('inf')
    
    # Transition factuelle
    if abs(ratio - 1.0) < 0.1:
        transition = "stable"
    elif ratio > 1.1:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Hétérogénéité: {initial_hetero:.3f} → {final_hetero:.3f} (ratio={ratio:.3f})',
        'initial_value': initial_hetero,
        'final_value': final_hetero,
        'transition': transition,
        'value': ratio,
        'metadata': {
            'ratio': ratio,
            'grid_size': 10
        }
    }