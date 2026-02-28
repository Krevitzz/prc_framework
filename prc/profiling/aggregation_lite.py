"""
prc.profiling.aggregation_lite

Responsabilité : Aggregation basique cross-runs (median, IQR, n_runs)

RECÂBLAGE timeline :
    find_feature_variants() adapté au nommage catch22 ({fn}__{catch22_name}).
    Recherche par préfixe au lieu de suffixes _initial/_final/_mean.

FIX : Filtrage features booléennes (has_*, is_*) avant percentile
"""

import numpy as np
from typing import Dict, List


def find_feature_variants(features: Dict, base_name: str) -> List[str]:
    """
    Trouve variantes feature pour un nom de base donné.

    Nommage timeline  : {base_name}__{catch22_name} → préfixe '{base_name}__'
    Dérivées simples  : norm_ratio, entropy_delta... → correspondance exacte
    Rétrocompat legacy: {base_name}_{suffix} → gardé pour transition
    """
    variants = []
    prefix = f'{base_name}__'

    for key in features:
        if key.startswith(prefix):
            variants.append(key)
            continue
        if key == base_name:
            variants.append(key)
            continue
        for suffix in ['_initial', '_final', '_mean', '_max', '_min']:
            if key == base_name + suffix:
                variants.append(key)
                break

    return variants


def aggregate_feature_by_entity(
    rows: List[Dict],
    entity_key: str,
    feature_name: str
) -> Dict[str, Dict]:
    """Agrège feature par entité (median, Q1, Q3, n_runs)."""
    grouped = {}

    for row in rows:
        entity_id = row['composition'][entity_key]
        features  = row['features']
        variants  = find_feature_variants(features, feature_name)

        for variant in variants:
            value = features[variant]
            if isinstance(value, bool):
                continue
            if not np.isfinite(value):
                continue
            if entity_id not in grouped:
                grouped[entity_id] = []
            grouped[entity_id].append(float(value))

    aggregated = {}
    for entity_id, values in grouped.items():
        if not values:
            continue
        arr = np.array(values, dtype=float)
        aggregated[entity_id] = {
            'median': float(np.median(arr)),
            'q1'    : float(np.percentile(arr, 25)),
            'q3'    : float(np.percentile(arr, 75)),
            'n_runs': len(values),
        }

    return aggregated


def aggregate_all_features_by_entity(
    rows: List[Dict],
    entity_key: str
) -> Dict[str, Dict[str, Dict]]:
    """Agrège toutes features numériques par entité."""
    all_features = set()
    for row in rows:
        all_features.update(row['features'].keys())

    numeric_features = sorted([
        f for f in all_features
        if not f.startswith('has_') and not f.startswith('is_')
    ])

    result = {}
    for feature_name in numeric_features:
        feature_agg = aggregate_feature_by_entity(rows, entity_key, feature_name)
        for entity_id, stats in feature_agg.items():
            if entity_id not in result:
                result[entity_id] = {}
            result[entity_id][feature_name] = stats

    return result
