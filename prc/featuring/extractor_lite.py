"""
prc.featuring.extractor_lite

Responsabilité : Extraction features scalaires depuis history np.ndarray

Minimal : projections temporelles + appel registres
"""

import numpy as np
from typing import Dict, List

from featuring.registries import universal_lite, matrix_2d_lite, tensor_3d_lite


# =============================================================================
# PROJECTIONS TEMPORELLES
# =============================================================================

def compute_projection(history: np.ndarray, projection: str) -> np.ndarray:
    """
    Calcule projection temporelle sur history.
    
    Args:
        history    : Séquence états (T, *dims)
        projection : 'initial' | 'final' | 'mean' | 'max' | 'min'
    
    Returns:
        État projeté (*dims)
    
    Examples:
        >>> compute_projection(history, 'initial')  # history[0]
        >>> compute_projection(history, 'final')    # history[-1]
        >>> compute_projection(history, 'mean')     # np.mean(history, axis=0)
    """
    if projection == 'initial':
        return history[0]
    elif projection == 'final':
        return history[-1]
    elif projection == 'mean':
        return np.mean(history, axis=0)
    elif projection == 'max':
        return np.max(history, axis=0)
    elif projection == 'min':
        return np.min(history, axis=0)
    else:
        raise ValueError(f"Projection inconnue: {projection}")


# =============================================================================
# EXTRACTION FEATURES PAR LAYER
# =============================================================================

def extract_universal_features(history: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Extrait features universelles (tout rank).
    
    Args:
        history : Séquence états (T, *dims)
        config  : Dict {'functions': [{name, projections, params}, ...]}
    
    Returns:
        {'euclidean_norm_initial': 5.0, 'entropy_final': 2.5, ...}
    
    Notes:
        - Applicable tout rank
        - Projections configurables via YAML
    """
    features = {}
    
    for func_config in config['functions']:
        func_name = func_config['name']
        projections = func_config.get('projections', ['final'])
        params = func_config.get('params', {})
        
        # Résoudre fonction depuis registre
        func = getattr(universal_lite, func_name)
        
        # Appliquer sur chaque projection
        for proj in projections:
            state_proj = compute_projection(history, proj)
            
            try:
                value = func(state_proj, **params)
                features[f'{func_name}_{proj}'] = value
            except Exception as e:
                # Skip si erreur (ex: entropy sur valeurs constantes)
                pass
    
    return features


def extract_matrix_2d_features(history: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Extrait features matrix_2d (rank 2).
    
    Args:
        history : Séquence états (T, n, m)
        config  : Dict {'functions': [{name, projections, params}, ...]}
    
    Returns:
        {'trace_final': 12.3, 'eigenvalue_max_initial': 5.0, ...}
    
    Notes:
        - Applicable rank 2 uniquement
        - ValueError si rank ≠ 2
    """
    features = {}
    
    for func_config in config['functions']:
        func_name = func_config['name']
        projections = func_config.get('projections', ['final'])
        params = func_config.get('params', {})
        
        # Résoudre fonction depuis registre
        func = getattr(matrix_2d_lite, func_name)
        
        # Appliquer sur chaque projection
        for proj in projections:
            state_proj = compute_projection(history, proj)
            
            try:
                value = func(state_proj, **params)
                features[f'{func_name}_{proj}'] = value
            except Exception as e:
                # Skip si erreur (ex: matrice singulière)
                pass
    
    return features


def extract_tensor_3d_features(history: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Extrait features tensor_3d (rank 3).
    
    Args:
        history : Séquence états (T, n, m, p)
        config  : Dict {'functions': [{name, projections, params}, ...]}
    
    Returns:
        {'mode_variance_0_final': 0.5, 'mode_variance_1_final': 0.3, ...}
    
    Notes:
        - Applicable rank 3 uniquement
        - ValueError si rank ≠ 3
    """
    features = {}
    
    for func_config in config['functions']:
        func_name = func_config['name']
        projections = func_config.get('projections', ['final'])
        params = func_config.get('params', {})
        
        # Résoudre fonction depuis registre
        func = getattr(tensor_3d_lite, func_name)
        
        # Appliquer sur chaque projection
        for proj in projections:
            state_proj = compute_projection(history, proj)
            
            try:
                value = func(state_proj, **params)
                features[f'{func_name}_{proj}'] = value
            except Exception as e:
                # Skip si erreur
                pass
    
    return features
