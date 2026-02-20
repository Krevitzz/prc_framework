"""
prc.featuring.extractor_lite

Responsabilité : Extraction features scalaires depuis history np.ndarray

Minimal : projections temporelles + appel registre universal
"""

import numpy as np
from typing import Dict, Callable

from featuring.registries.universal_lite import (
    euclidean_norm,
    entropy,
    mean_value,
    std_value,
)


# Mapping fonctions disponibles
UNIVERSAL_FUNCTIONS = {
    'euclidean_norm': euclidean_norm,
    'entropy': entropy,
    'mean_value': mean_value,
    'std_value': std_value,
}


def compute_projection(history: np.ndarray, projection: str) -> np.ndarray:
    """
    Calcule projection temporelle depuis history.
    
    Args:
        history    : np.ndarray (T, *state_shape)
        projection : 'initial' | 'final' | 'mean' | 'max' | 'min'
    
    Returns:
        State projeté (même shape que state)
    
    Examples:
        >>> history = np.random.rand(201, 10, 10)
        >>> state_initial = compute_projection(history, 'initial')
        >>> state_initial.shape
        (10, 10)
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
        raise ValueError(f"Projection inconnue : {projection}")


def extract_universal_features(
    history: np.ndarray,
    config: Dict
) -> Dict[str, float]:
    """
    Extrait features layer universal avec projections.
    
    Args:
        history : np.ndarray (T, *state_shape)
        config  : Dict config features (depuis YAML)
    
    Returns:
        Dict features scalaires
        Ex: {
            'euclidean_norm_initial': 5.3,
            'euclidean_norm_final': 12.7,
            'entropy_initial': 3.2,
            ...
        }
    
    Notes:
        - Lit config['functions'] : liste des fonctions à appliquer
        - Chaque fonction spécifie 'name' et 'projections'
        - Produit : len(functions) × len(projections) features
    """
    features = {}
    
    functions_config = config.get('functions', [])
    
    for func_cfg in functions_config:
        func_name = func_cfg['name']
        projections = func_cfg.get('projections', ['final'])
        params = func_cfg.get('params', {})
        
        # Récupérer fonction
        if func_name not in UNIVERSAL_FUNCTIONS:
            raise ValueError(f"Fonction universal inconnue : {func_name}")
        
        func = UNIVERSAL_FUNCTIONS[func_name]
        
        # Appliquer sur chaque projection
        for projection in projections:
            state_proj = compute_projection(history, projection)
            
            # Calculer feature
            try:
                value = func(state_proj, **params)
            except Exception as e:
                # Log et skip (pas crash batch)
                print(f"Warning: {func_name}({projection}) failed: {e}")
                value = np.nan
            
            # Stocker
            feature_key = f"{func_name}_{projection}"
            features[feature_key] = value
    
    return features
