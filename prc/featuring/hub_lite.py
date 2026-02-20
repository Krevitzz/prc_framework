"""
prc.featuring.hub_lite

Responsabilité : Orchestration extraction features minimales (3-5 features scalaires)

Minimal : universal layer uniquement
"""

import numpy as np
from typing import Dict

from featuring.layers_lite import inspect_history
from featuring.extractor_lite import extract_universal_features


def extract_features(
    history: np.ndarray,
    config: Dict
) -> Dict[str, float]:
    """
    Extraction features minimale (orchestrateur).
    
    Args:
        history : np.ndarray (T, *state_shape)
        config  : Dict config features (depuis YAML)
    
    Returns:
        Dict features scalaires (~5 features)
        Ex: {
            'euclidean_norm_initial': 5.3,
            'euclidean_norm_final': 12.7,
            'euclidean_norm_mean': 8.9,
            'entropy_initial': 3.2,
            'entropy_final': 2.8,
        }
    
    Workflow:
        1. Inspect history (rank, shape)
        2. Extract universal features (seul layer POC)
        3. Return dict scalaires
    
    Notes:
        - Minimal : universal layer uniquement
        - Futur : ajouter layers matrix_2d, tensor_3d, etc.
        - Layer = fichier registre → maintenance simple
    
    Examples:
        >>> history = np.random.rand(201, 10, 10)
        >>> config = {'universal': {'functions': [...]}}
        >>> features = extract_features(history, config)
        >>> len(features)
        5
    """
    features = {}
    
    # 1. Inspect history
    info = inspect_history(history)
    
    # Validation basique
    if not np.all(np.isfinite(history)):
        features['has_nan_inf'] = True
    else:
        features['has_nan_inf'] = False
    
    # 2. Extract universal features (seul layer POC)
    universal_config = config.get('universal', {})
    
    if universal_config:
        universal_features = extract_universal_features(history, universal_config)
        features.update(universal_features)
    
    # 3. Futur : ajouter autres layers selon rank
    # if info['rank'] == 2:
    #     features.update(extract_matrix_2d_features(history, config))
    
    return features
