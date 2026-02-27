"""
prc.profiling.aggregation_lite

Responsabilité : Aggregation basique cross-runs (median, IQR, n_runs)

Minimal : 3 stats uniquement

FIX : Filtrage features booléennes (has_*, is_*) avant percentile
"""

import numpy as np
from typing import Dict, List


def find_feature_variants(features: Dict, base_name: str) -> List[str]:
    """
    Trouve variantes feature avec projections (_initial, _final, _mean).

    Args:
        features  : Dict features d'un run
        base_name : Nom base (ex: 'euclidean_norm', 'trace')

    Returns:
        Liste variantes trouvées (ex: ['euclidean_norm_final', 'euclidean_norm_initial'])

    Examples:
        >>> find_feature_variants({'euclidean_norm_final': 12.3}, 'euclidean_norm')
        ['euclidean_norm_final']
    """
    variants = []

    for suffix in ['_initial', '_final', '_mean', '_max', '_min']:
        variant = base_name + suffix
        if variant in features:
            variants.append(variant)

    if base_name in features:
        variants.append(base_name)

    return variants


def aggregate_feature_by_entity(
    rows: List[Dict],
    entity_key: str,
    feature_name: str
) -> Dict[str, Dict]:
    """
    Agrège feature par entité (median, Q1, Q3, n_runs).

    Args:
        rows         : Liste {composition, features}
        entity_key   : 'gamma_id' | 'encoding_id' | 'modifier_id'
        feature_name : Nom feature à agréger (ex: 'euclidean_norm_final')

    Returns:
        {
            'GAM-001': {
                'median': 12.3,
                'q1': 10.5,
                'q3': 15.8,
                'n_runs': 21
            },
            ...
        }

    Notes:
        - Skip si feature absente ou NaN/Inf
        - Skip si valeurs non numériques (bool → filtré en amont)
        - Calcul sur toutes valeurs valides cross-runs
    """
    grouped = {}

    for row in rows:
        entity_id = row['composition'][entity_key]
        features = row['features']

        variants = find_feature_variants(features, feature_name)

        for variant in variants:
            value = features[variant]

            # Skip booléens — np.percentile ne supporte pas dtype bool
            if isinstance(value, bool):
                continue

            # Skip NaN/Inf
            if not np.isfinite(value):
                continue

            if entity_id not in grouped:
                grouped[entity_id] = []

            grouped[entity_id].append(float(value))

    aggregated = {}

    for entity_id, values in grouped.items():
        if len(values) == 0:
            continue

        values_array = np.array(values, dtype=float)

        aggregated[entity_id] = {
            'median': float(np.median(values_array)),
            'q1': float(np.percentile(values_array, 25)),
            'q3': float(np.percentile(values_array, 75)),
            'n_runs': len(values),
        }

    return aggregated


def aggregate_all_features_by_entity(
    rows: List[Dict],
    entity_key: str
) -> Dict[str, Dict[str, Dict]]:
    """
    Agrège toutes features numériques par entité.

    Args:
        rows       : Liste {composition, features}
        entity_key : 'gamma_id' | 'encoding_id' | 'modifier_id'

    Returns:
        {
            'GAM-001': {
                'euclidean_norm_final': {median, q1, q3, n_runs},
                ...
            },
            ...
        }

    Notes:
        - Exclut features booléennes : has_* et is_* (flags état)
        - Détecte features automatiquement depuis rows
        - Skip features avec <1 valeur valide
    """
    all_features = set()
    for row in rows:
        all_features.update(row['features'].keys())

    # Filtrer flags booléens (has_* et is_*) — non agrégeables par percentile
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
