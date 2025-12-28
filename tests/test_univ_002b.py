"""
Test UNIV-002b : Diversité locale (patches)

Applicable : Rang 2 uniquement, shape >= 5x5
Mesure : Moyenne des std locaux sur patches échantillonnés
"""

import numpy as np

TEST_ID = "UNIV-002b"
REQUIRES_RANK = 2
D_TYPES = ["SYM", "ASY"]
WEIGHT_DEFAULT = 1.5


def _compute_local_diversity(state, patch_size=5, n_patches=20):
    """
    Calcule diversité locale via échantillonnage de patches.
    
    Returns:
        float ou None si erreur
    """
    if state.ndim != 2:
        return None
    
    h, w = state.shape
    if h < patch_size or w < patch_size:
        return None
    
    local_stds = []
    
    # Seed reproductible basé sur contenu état
    try:
        state_sum = np.nansum(state)
        if not np.isfinite(state_sum):
            state_sum = 0.0
        seed = int(abs(state_sum) * 1000) % (2**31)
        rng = np.random.RandomState(seed)
    except:
        rng = np.random.RandomState(42)
    
    for _ in range(n_patches):
        # Position aléatoire
        i = rng.randint(0, max(1, h - patch_size + 1))
        j = rng.randint(0, max(1, w - patch_size + 1))
        
        # Extraire patch
        patch = state[i:i+patch_size, j:j+patch_size]
        
        # Std du patch
        patch_clean = patch[np.isfinite(patch)]
        if len(patch_clean) > 1:
            local_stds.append(np.std(patch_clean))
    
    if local_stds:
        return float(np.nanmean(local_stds))
    else:
        return None


def run_test(history, context):
    """
    Observe l'évolution de la diversité locale.
    
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
    
    if initial_state.shape[0] < 5 or initial_state.shape[1] < 5:
        return {
            'test_name': TEST_ID,
            'status': 'NOT_APPLICABLE',
            'message': 'État trop petit pour patches 5x5',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'not_applicable',
            'value': 0.0,
            'metadata': {}
        }
    
    # Calcul diversités locales
    initial_local = _compute_local_diversity(history[0])
    final_local = _compute_local_diversity(history[-1])
    
    if initial_local is None or final_local is None:
        return {
            'test_name': TEST_ID,
            'status': 'ERROR',
            'message': 'Erreur calcul diversité locale',
            'initial_value': 0.0,
            'final_value': 0.0,
            'transition': 'error',
            'value': 0.0,
            'metadata': {}
        }
    
    # Ratio
    if initial_local > 1e-10:
        ratio = final_local / initial_local
    else:
        ratio = 0.0 if final_local < 1e-10 else float('inf')
    
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
        'message': f'Diversité locale: {initial_local:.3f} → {final_local:.3f} (ratio={ratio:.3f})',
        'initial_value': initial_local,
        'final_value': final_local,
        'transition': transition,
        'value': ratio,
        'metadata': {
            'ratio': ratio,
            'patch_size': 5,
            'n_patches': 20
        }
    }