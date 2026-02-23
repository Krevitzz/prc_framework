"""
prc.featuring.hub_featuring

Responsabilité : Orchestration extraction features (router layers selon rank)

Minimal : découverte automatique layers depuis configs/
"""

import numpy as np
from pathlib import Path
from typing import Dict

from utils.data_loading_lite import load_yaml
from featuring.layers_lite import inspect_history
from featuring.extractor_lite import (
    extract_universal_features,
    extract_matrix_2d_features,
    extract_tensor_3d_features,
)


def load_all_configs() -> Dict:
    """
    Découvre et charge automatiquement tous configs layers.
    
    Returns:
        {
            'universal': {...},
            'matrix_2d': {...},
            'tensor_3d': {...},
            ...
        }
    
    Notes:
        - Scan featuring/configs/minimal/*.yaml
        - Layer name = nom fichier sans extension
        - Ajout layer = créer YAML, pas toucher code
    
    Examples:
        >>> configs = load_all_configs()
        >>> configs.keys()
        dict_keys(['universal', 'matrix_2d', 'tensor_3d'])
    """
    configs = {}
    config_dir = Path('featuring/configs/minimal')
    
    if not config_dir.exists():
        return configs
    
    for yaml_file in sorted(config_dir.glob('*.yaml')):
        layer_name = yaml_file.stem
        configs[layer_name] = load_yaml(yaml_file)
    
    return configs


def extract_features(history: np.ndarray, config: Dict) -> Dict[str, float]:
    """
    Extrait features depuis history (routing layers automatique).
    
    Args:
        history : Séquence états (T, *dims)
        config  : Dict {
                    'universal': {...},
                    'matrix_2d': {...},
                    'tensor_3d': {...},
                  }
    
    Returns:
        {'feature_name': value, ...}
    
    Workflow:
        1. Inspect history → rank, shape
        2. Validation NaN/Inf → flag has_nan_inf
        3. Extract universal (tout rank)
        4. Extract matrix_2d si rank=2
        5. Extract tensor_3d si rank=3
    
    Notes:
        - Layers appelés selon rank détecté
        - Features agrégées dans dict unique
        - Erreurs extraction → skip silencieux
    
    Examples:
        >>> features = extract_features(history, config)
        >>> features.keys()
        dict_keys(['has_nan_inf', 'euclidean_norm_initial', 'trace_final', ...])
    """
    # 1. Inspect history
    info = inspect_history(history)
    rank = info['rank']
    
    # 2. Validation NaN/Inf
    has_nan_inf = not np.all(np.isfinite(history))
    
    features = {
        'has_nan_inf': has_nan_inf,
    }
    
    # 3. Universal features (tout rank)
    if 'universal' in config:
        universal_features = extract_universal_features(history, config['universal'])
        features.update(universal_features)
    
    # 4. Matrix 2D features (rank 2)
    if rank == 2 and 'matrix_2d' in config:
        matrix_features = extract_matrix_2d_features(history, config['matrix_2d'])
        features.update(matrix_features)
    
    # 5. Tensor 3D features (rank 3)
    if rank == 3 and 'tensor_3d' in config:
        tensor_features = extract_tensor_3d_features(history, config['tensor_3d'])
        features.update(tensor_features)
    
    return features
