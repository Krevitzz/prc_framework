"""
prc.analysing.profiling_lite

Responsabilité : Agrégation cross-runs par entité (gamma, encoding, modifier)

Fusion de :
    profiling/aggregation_lite.py  — fonctions d'agrégation
    profiling/hub_profiling.py     — orchestration

Pourquoi la fusion :
    profiling/ contenait 2 fichiers pour 3 fonctions sans dépendance propre.
    La responsabilité "caractérisation cross-runs" appartient à analysing —
    le profiling était la surcouche ML que le peeling + namer remplacent.
    Les fonctions d'agrégation restent utiles comme input au cluster_namer
    (build_cluster_profile() les utilise indirectement).

Notes :
    - Aucun changement fonctionnel sur les fonctions héritées
    - profiling/ supprimé du pipeline (retirer du charter)
"""

import numpy as np
from typing import Dict, List


# =============================================================================
# HELPERS FEATURES
# =============================================================================

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


# =============================================================================
# AGRÉGATION PAR ENTITÉ
# =============================================================================

def aggregate_feature_by_entity(
    rows        : List[Dict],
    entity_key  : str,
    feature_name: str,
) -> Dict[str, Dict]:
    """
    Agrège une feature par entité (median, Q1, Q3, n_runs).

    Args:
        rows         : Liste {composition, features}
        entity_key   : 'gamma_id' | 'encoding_id' | 'modifier_id'
        feature_name : Nom de base (ex: 'norm_ratio', 'euclidean_norm')

    Returns:
        {entity_id: {median, q1, q3, n_runs}}
    """
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
    rows      : List[Dict],
    entity_key: str,
) -> Dict[str, Dict[str, Dict]]:
    """
    Agrège toutes features numériques par entité.

    Returns:
        {entity_id: {feature_name: {median, q1, q3, n_runs}}}
    """
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


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_profiling(rows: List[Dict]) -> Dict:
    """
    Profiling complet : agrégation par gamma, encoding, modifier.

    Args:
        rows : Liste {composition, features}

    Returns:
        {
            'gamma'         : {gamma_id: {feature: {median, q1, q3, n_runs}}},
            'encoding'      : {encoding_id: {...}},
            'modifier'      : {modifier_id: {...}},
            'n_observations': int,
        }
    """
    print(f"=== Profiling cross-runs ===")
    print(f"Observations: {len(rows)}\n")

    result = {'n_observations': len(rows)}

    for entity_key, label in [
        ('gamma_id',    'gamma'),
        ('encoding_id', 'encoding'),
        ('modifier_id', 'modifier'),
    ]:
        print(f"Aggregating by {label}...")
        profiles = aggregate_all_features_by_entity(rows, entity_key)
        result[label] = profiles
        print(f"  {len(profiles)} {label}s profiled")

    return result
