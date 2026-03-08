"""
prc.featuring.hub_featuring

Responsabilité : Routage extraction features (router layers selon applicabilité)

Architecture :
    load_all_configs() : découverte automatique YAML depuis configs/minimal/
    extract_features() : boucle layers → check applicabilité → extract_for_layer

Notes :
    Routage pur — aucun calcul ici.
    Toute logique d'extraction est dans les registres ({layer}_lite.py).
    Ajout layer = créer YAML + registre, zéro touche ici.
"""

import numpy as np
from pathlib import Path
from typing import Dict

from utils.data_loading_lite import load_yaml
from featuring.extractor_lite import extract_for_layer
from featuring.layers_lite import check_applicability, inspect_history


def load_all_configs() -> Dict:
    """
    Découvre et charge automatiquement tous les configs layers.

    Returns:
        {
            'timeline': {...},
            'matrix_2d': {...},
            ...
        }

    Notes:
        - Scan featuring/configs/minimal/*.yaml
        - Layer name = nom fichier sans extension
        - Ajout layer = créer YAML, pas toucher code
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

    Args:
        history : np.ndarray (T, *dims)
        config  : Dict depuis load_all_configs()

    Returns:
        {
            'features': {feature_name: float, ...},
            'layers'  : ['timeline', ...]
        }
    """
    info = inspect_history(history)

    applicable_layers = []
    features = {}

    for layer_name, layer_config in config.items():
        if check_applicability(info, layer_config):
            applicable_layers.append(layer_name)

            layer_features = extract_for_layer(history, layer_name, layer_config)
            features.update(layer_features)

    return {
        'features': features,
        'layers'  : applicable_layers
    }
