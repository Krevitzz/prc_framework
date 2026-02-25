"""
prc.featuring.hub_featuring

Responsabilité : Orchestration extraction features (router layers selon rank)

Minimal : découverte automatique layers depuis configs/
"""

import numpy as np
from pathlib import Path
from typing import Dict

from utils.data_loading_lite import load_yaml
from featuring.extractor_lite import extract_for_layer
from featuring.layers_lite import check_applicability, inspect_history



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


def extract_features(history: np.ndarray, config: Dict) -> Dict:
    """
    Extrait features + détecte layers applicables.
    
    Returns:
        {
            'features': {...},
            'layers': ['universal', 'matrix_2d']
        }
    """
    
    # 1. Inspect history
    info = inspect_history(history)
    
    # 2. Boucle layers : vérifier applicabilité + extraire
    applicable_layers = []
    features = {}
    
    for layer_name, layer_config in config.items():
        # Vérifier applicabilité (logique dans extractor_lite)
        if check_applicability(info, layer_config):
            applicable_layers.append(layer_name)
            
            # Extraire features (dispatch dans extractor_lite)
            try:
                layer_features = extract_for_layer(history, layer_name, layer_config)
                features.update(layer_features)
            except Exception as e:
                print(f"[WARNING] Extraction {layer_name} échouée: {e}")
    
    # 3. Validation NaN/Inf
    has_nan_inf = not np.all(np.isfinite(history))
    features['has_nan_inf'] = has_nan_inf
    
    # Return features + layers
    return {
        'features': features,
        'layers': applicable_layers
    }
