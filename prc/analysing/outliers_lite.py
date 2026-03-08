"""
prc.analysing.outliers_lite

Responsabilité : Détection outliers + récurrence atomics

Minimal : IsolationForest sklearn sur features COMMUNES uniquement

FIX : inf remplacé par valeur sentinelle finie avant IsolationForest
      — préserve le signal "très mal conditionné" sans exclure le run
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.ensemble import IsolationForest
from featuring.layers_lite import extract_common_features

# Valeur sentinelle pour inf — suffisamment grande pour discriminer
# sans overflow dans IsolationForest
_INF_SENTINEL = 1e15




def _sanitize_vector(vector: List[float]) -> List[float]:
    """
    Remplace inf par sentinelle finie, NaN par 0.0 placeholder.

    Args:
        vector : Valeurs features d'un run

    Returns:
        vector_sanitized — aucun run exclu

    Notes:
        - NaN  → 0.0 placeholder neutre, documenté par {fn}__signal_finite_ratio
                 et {fn}__signal_norm_absolute présents dans la même matrice
        - inf  → ±_INF_SENTINEL (signal physique préservé)
        - Cohérent avec clustering_lite : même traitement, même documentation
    """
    sanitized = []
    for v in vector:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sanitized.append(0.0)
        elif isinstance(v, float) and np.isposinf(v):
            sanitized.append(_INF_SENTINEL)
        elif isinstance(v, float) and np.isneginf(v):
            sanitized.append(-_INF_SENTINEL)
        else:
            sanitized.append(float(v))

    return sanitized


def detect_outliers(
    rows: List[Dict],
    contamination: float = 0.1
) -> Tuple[List[int], List[int]]:
    """
    Détecte outliers via IsolationForest sur features COMMUNES.

    Args:
        rows          : Liste {composition, features}
        contamination : Fraction attendue outliers (défaut 10%)

    Returns:
        (outlier_indices, stable_indices)

    Notes:
        - inf  → sentinelle ±1e15 (signal préservé)
        - NaN  → 0.0 placeholder, documenté par {fn}__signal_finite_ratio
        - 216/216 runs analysés — aucune exclusion silencieuse
        - Features communes : intersection stricte, sans has_* et is_*
    """
    common_features = extract_common_features(rows)

    if len(common_features) == 0:
        print("  WARNING: Aucune feature commune — outliers detection skipped")
        return [], list(range(len(rows)))

    feature_names = sorted(common_features)

    features_matrix = []
    valid_indices = []
    n_inf_replaced = 0
    n_nan_placeholder = 0

    for i, row in enumerate(rows):
        features = row['features']
        vector = [features.get(k) for k in feature_names]

        sanitized = _sanitize_vector(vector)

        n_inf_replaced += sum(
            1 for v, s in zip(vector, sanitized)
            if v is not None and isinstance(v, float) 
            and not np.isnan(v) and abs(v) > _INF_SENTINEL / 2
        )
        n_nan_placeholder += sum(
            1 for v in vector
            if v is None or (isinstance(v, float) and np.isnan(v))
        )

        features_matrix.append(sanitized)
        valid_indices.append(i)

    if n_inf_replaced > 0:
        print(f"  Inf remplacés par sentinelle (±{_INF_SENTINEL:.0e}) : {n_inf_replaced}")
    if n_nan_placeholder > 0:
        print(f"  NaN → 0.0 placeholder (documentés par signal_finite_ratio) : {n_nan_placeholder}")

    if len(features_matrix) < 2:
        print(f"  WARNING: Moins de 2 samples valides — outliers skipped")
        return [], list(range(len(rows)))

    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(features_matrix)

    outlier_indices = [valid_indices[i] for i, p in enumerate(predictions) if p == -1]
    stable_indices  = [valid_indices[i] for i, p in enumerate(predictions) if p == 1]

    print(f"  Features communes utilisées: {len(feature_names)}")
    print(f"  Samples valides analysés: {len(features_matrix)}/{len(rows)}")

    return outlier_indices, stable_indices


def compute_atomic_recurrence(
    rows: List[Dict],
    indices: List[int],
    atomic_key: str = 'gamma_id'
) -> Dict[str, Dict]:
    """
    Calcule récurrence atomics dans subset.

    Args:
        rows       : Liste complète rows
        indices    : Indices subset (ex: outliers)
        atomic_key : 'gamma_id' | 'encoding_id' | 'modifier_id'

    Returns:
        {
            'GAM-001': {
                'count': 5,
                'fraction': 0.42,
                'total_subset': 12
            },
            ...
        }
    """
    subset_rows = [rows[i] for i in indices]
    n_total = len(subset_rows)

    if n_total == 0:
        return {}

    counts = {}
    for row in subset_rows:
        atomic_id = row['composition'][atomic_key]
        counts[atomic_id] = counts.get(atomic_id, 0) + 1

    recurrence = {}
    for atomic_id, count in counts.items():
        recurrence[atomic_id] = {
            'count': count,
            'fraction': count / n_total,
            'total_subset': n_total
        }

    recurrence = dict(
        sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True)
    )

    return recurrence


def analyze_outliers(rows: List[Dict], contamination: float = 0.1) -> Dict:
    """
    Analyse complète outliers.

    Returns:
        {
            'n_outliers'      : int,
            'n_stables'       : int,
            'outlier_fraction': float,
            'outlier_indices' : List[int],
            'stable_indices'  : List[int],
            'recurrence'      : {
                'gamma'   : {...},
                'encoding': {...},
                'modifier': {...}
            },
            'n_features_used' : int
        }
    """
    outlier_idx, stable_idx = detect_outliers(rows, contamination)

    n_outliers = len(outlier_idx)
    n_stables  = len(stable_idx)
    n_total    = n_outliers + n_stables

    gamma_rec    = compute_atomic_recurrence(rows, outlier_idx, 'gamma_id')
    encoding_rec = compute_atomic_recurrence(rows, outlier_idx, 'encoding_id')
    modifier_rec = compute_atomic_recurrence(rows, outlier_idx, 'modifier_id')

    common_features = extract_common_features(rows)

    return {
        'n_outliers'      : n_outliers,
        'n_stables'       : n_stables,
        'outlier_fraction': n_outliers / n_total if n_total > 0 else 0.0,
        'outlier_indices' : outlier_idx,
        'stable_indices'  : stable_idx,
        'recurrence': {
            'gamma'   : gamma_rec,
            'encoding': encoding_rec,
            'modifier': modifier_rec
        },
        'n_features_used': len(common_features)
    }