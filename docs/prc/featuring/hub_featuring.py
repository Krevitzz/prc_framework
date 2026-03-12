"""
prc.featuring.hub_featuring

Responsabilité : Routage extraction features (router layers selon applicabilité)

Architecture :
    load_all_configs()       : découverte automatique YAML depuis configs/minimal/
    get_active_layers()      : détecte layers applicables depuis shape D_initial
    measure_state()          : scalaires sur état courant — appelé à chaque itération kernel
    extract_from_signals()   : catch22 + dérivées depuis signaux accumulés — appelé une fois
    extract_features()       : wrapper rétrocompat (history complet → streaming interne)

Notes :
    Routage pur — aucun calcul ici.
    Toute logique d'extraction est dans les registres ({layer}_lite.py).
    Ajout layer = créer YAML + registre + entrée _STREAMING_LAYERS, zéro touche ailleurs.
"""

import numpy as np
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from utils.data_loading_lite import load_yaml
from featuring.layers_lite import check_applicability, inspect_history

# Import des fonctions streaming depuis les registres
from featuring.registries.timeline_lite import (
    measure_state        as _timeline_measure_state,
    extract_from_signals as _timeline_extract_from_signals,
)


# Mapping layer → fonctions streaming
# Ajouter une entrée ici quand un nouveau layer implémente measure_state/extract_from_signals
_STREAMING_LAYERS = {
    'timeline': {
        'measure_state'       : _timeline_measure_state,
        'extract_from_signals': _timeline_extract_from_signals,
    },
}


def load_all_configs() -> Dict:
    """
    Découvre et charge automatiquement tous les configs layers.

    Returns:
        {'timeline': {...}, ...}

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


def get_active_layers(state: np.ndarray, config: Dict) -> List[str]:
    """
    Détecte layers applicables depuis shape d'un état.

    Args:
        state  : np.ndarray — D_initial ou tout état (shape-based)
        config : Dict depuis load_all_configs()

    Returns:
        Liste noms layers applicables

    Notes:
        - inspect_history accepte state reshape (1, *state.shape) — shape-based uniquement
        - Appelé une fois avant la boucle streaming depuis D_initial
    """
    info = inspect_history(state.reshape(1, *state.shape))
    return [
        layer_name for layer_name, layer_config in config.items()
        if check_applicability(info, layer_config)
    ]


def measure_state(
    state        : np.ndarray,
    active_layers: List[str],
    config       : Dict,
) -> Dict[str, float]:
    """
    Applique fonctions de mesure de chaque layer actif sur état courant.

    Args:
        state         : np.ndarray — état courant (un pas de temps)
        active_layers : depuis get_active_layers()
        config        : Dict depuis load_all_configs()

    Returns:
        Dict[str, float] — {fn_name: scalaire} — NaN si fonction échoue

    Notes:
        - Appelé à chaque itération kernel — doit être rapide
    """
    measures: Dict[str, float] = {}

    for layer_name in active_layers:
        streaming = _STREAMING_LAYERS.get(layer_name)
        if streaming is None:
            continue

        layer_config   = config.get(layer_name, {})
        func_configs   = layer_config.get('functions', [])
        layer_measures = streaming['measure_state'](state, func_configs)
        measures.update(layer_measures)

    return measures


def extract_from_signals(
    signals      : Dict[str, List[float]],
    has_nan_inf  : bool,
    last_state   : np.ndarray,
    active_layers: List[str],
    config       : Dict,
) -> Dict[str, float]:
    """
    catch22 + dérivées + santé depuis signaux accumulés.

    Args:
        signals       : {fn_name: [val_t0, val_t1, ...]}
        has_nan_inf   : bool accumulé pendant streaming
        last_state    : dernier état capturé (pour is_collapsed)
        active_layers : depuis get_active_layers()
        config        : Dict depuis load_all_configs()

    Returns:
        Dict[str, float] — toutes features

    Notes:
        - Appelé une fois après la boucle kernel
    """
    features: Dict[str, float] = {}

    for layer_name in active_layers:
        streaming = _STREAMING_LAYERS.get(layer_name)
        if streaming is None:
            continue

        layer_config   = config.get(layer_name, {})
        layer_features = streaming['extract_from_signals'](
            signals, has_nan_inf, last_state, layer_config
        )
        features.update(layer_features)

    return features


def extract_features(history: np.ndarray, config: Dict) -> Dict:
    """
    Wrapper rétrocompat — streaming interne sur history complet.

    Args:
        history : np.ndarray (T, *dims)
        config  : Dict depuis load_all_configs()

    Returns:
        {'features': Dict[str, float], 'layers': List[str]}

    Notes:
        - Conservé pour compatibilité tests et verdict depuis parquet
    """
    if len(history) == 0:
        return {'features': {}, 'layers': []}

    active_layers = get_active_layers(history[0], config)

    signals     = defaultdict(list)
    has_nan_inf = False
    last_state  = None

    for state in history:
        if not np.all(np.isfinite(state)):
            has_nan_inf = True
        measures = measure_state(state, active_layers, config)
        for k, v in measures.items():
            signals[k].append(v)
        last_state = state

    features = extract_from_signals(
        dict(signals), has_nan_inf, last_state, active_layers, config
    )

    return {'features': features, 'layers': active_layers}