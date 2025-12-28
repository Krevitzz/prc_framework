"""
Test STR-002 : Évolution du spectre

Applicable : Rang 2 uniquement (matrices carrées symétriques de préférence)
Mesure : max(|λ_i|) à chaque itération (rayon spectral)
"""

import numpy as np

TEST_ID = "STR-002"
REQUIRES_RANK = 2
D_TYPES = ["SYM", "ASY"]
WEIGHT_DEFAULT = 1.5


def _compute_spectral_radius(matrix):
    """
    Calcule le rayon spectral d'une matrice.
    
    Returns:
        float ou None si erreur
    """
    if matrix.ndim != 2:
        return None
    
    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
        return None
    
    try:
        # Essayer décomposition symétrique (plus rapide)
        eigenvalues = np.linalg.eigvalsh(matrix)
        spectral_radius = float(np.max(np.abs(eigenvalues)))
        return spectral_radius
    except np.linalg.LinAlgError:
        # Si non symétrique, utiliser eig général
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            return spectral_radius
        except np.linalg.LinAlgError:
            return None


def run_test(history, context):
    """
    Observe l'évolution du rayon spectral.
    
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
    
    # Calcul rayons spectraux
    spectral_radii = []
    for i, state in enumerate(history):
        sr = _compute_spectral_radius(state)
        if sr is None:
            return {
                'test_name': TEST_ID,
                'status': 'ERROR',
                'message': f'Erreur calcul valeurs propres à itération {i}',
                'initial_value': spectral_radii[0] if spectral_radii else 0.0,
                'final_value': 0.0,
                'transition': 'error',
                'value': 0.0,
                'metadata': {'first_error_iteration': i}
            }
        spectral_radii.append(sr)
    
    # Statistiques
    initial_sr = spectral_radii[0]
    final_sr = spectral_radii[-1]
    max_sr = max(spectral_radii)
    min_sr = min(spectral_radii)
    
    # Tendance
    if len(spectral_radii) > 1:
        try:
            trend_coef = np.polyfit(range(len(spectral_radii)), spectral_radii, 1)[0]
        except:
            trend_coef = 0.0
        
        if abs(trend_coef) < 0.01:
            trend = "stable"
        elif trend_coef > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
    else:
        trend = "stable"
        trend_coef = 0.0
    
    # Transition factuelle
    if max_sr > 100:
        transition = "explosive"
    elif abs(final_sr - initial_sr) / (initial_sr + 1e-10) < 0.1:
        transition = "stable"
    elif final_sr > initial_sr:
        transition = "increasing"
    else:
        transition = "decreasing"
    
    return {
        'test_name': TEST_ID,
        'status': 'SUCCESS',
        'message': f'Rayon spectral: {initial_sr:.3f} → {final_sr:.3f}',
        'initial_value': initial_sr,
        'final_value': final_sr,
        'transition': transition,
        'value': max_sr,  # Max rayon spectral comme valeur principale
        'metadata': {
            'max_spectral_radius': max_sr,
            'min_spectral_radius': min_sr,
            'trend': trend,
            'trend_coefficient': trend_coef,
            'spectral_radii': spectral_radii
        }
    }