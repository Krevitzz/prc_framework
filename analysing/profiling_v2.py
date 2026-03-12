"""
analysing/profiling_v2.py

Responsabilité : Agrégation cross-runs par entité (gamma, encoding, modifier).

Reçoit AnalysingData — pas de reconstruction de matrice.
Groupby via arrays meta numpy, np.nanmedian/nanpercentile sur slices.
"""

import numpy as np
from typing import Dict, List

from analysing.parquet_filter_v2 import AnalysingData


def _aggregate_by_meta_array(
    M          : np.ndarray,
    feat_names : List[str],
    entity_arr : np.ndarray,
) -> Dict[str, Dict]:
    """
    Agrège M par entité via groupby sur entity_arr.

    Args:
        M          : (n, F) float32 — features sanitizées
        feat_names : noms colonnes de M
        entity_arr : (n,) array des IDs entité (gamma_ids, encoding_ids, ...)

    Returns:
        {entity_id: {feature_name: {median, q1, q3, n_runs}}}
    """
    unique_ids = np.unique(entity_arr)
    profiles   = {}

    for eid in unique_ids:
        mask = entity_arr == eid
        sub  = M[mask].astype(np.float64)   # (n_entity, F)

        profiles[str(eid)] = {}
        for j, fname in enumerate(feat_names):
            col = sub[:, j]
            col = col[np.isfinite(col)]
            if len(col) == 0:
                continue
            profiles[str(eid)][fname] = {
                'median': float(np.median(col)),
                'q1'    : float(np.percentile(col, 25)),
                'q3'    : float(np.percentile(col, 75)),
                'n_runs': len(col),
            }

    return profiles


def run_profiling(data: AnalysingData) -> Dict:
    """
    Profiling complet : agrégation par gamma, encoding, modifier.

    Args:
        data : AnalysingData depuis parquet_filter.load_analysing_data()

    Returns:
        {
            'gamma'         : {gamma_id: {feature: {median, q1, q3, n_runs}}},
            'encoding'      : {encoding_id: {...}},
            'modifier'      : {modifier_id: {...}},
            'n_observations': int,
        }
    """
    print(f"=== Profiling cross-runs ===")
    print(f"Observations: {data.n}\n")

    if data.n == 0:
        return {'n_observations': 0, 'gamma': {}, 'encoding': {}, 'modifier': {}}

    M, feat_names = data.features_for_ml()

    result = {'n_observations': data.n}

    for entity_arr, label in [
        (data.gamma_ids,    'gamma'),
        (data.encoding_ids, 'encoding'),
        (data.modifier_ids, 'modifier'),
    ]:
        print(f"Aggregating by {label}...")
        profiles      = _aggregate_by_meta_array(M, feat_names, entity_arr)
        result[label] = profiles
        print(f"  {len(profiles)} {label}s profiled")

    return result
