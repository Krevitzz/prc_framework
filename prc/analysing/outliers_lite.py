"""
prc.analysing.outliers_lite

Responsabilité : Détection outliers + récurrence atomics

Minimal : IsolationForest sklearn sur features COMMUNES uniquement

FIX : inf remplacé par valeur sentinelle finie avant IsolationForest
      — préserve le signal "très mal conditionné" sans exclure le run
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.ensemble import IsolationForest

# Valeur sentinelle pour inf — suffisamment grande pour discriminer
# sans overflow dans IsolationForest
_INF_SENTINEL = 1e15


def _extract_common_features(rows: List[Dict]) -> Set[str]:
    """
    Identifie features présentes dans TOUS les runs.

    Args:
        rows : Liste {composition, features}

    Returns:
        Set features communes (intersection)

    Notes:
        - Exclut features non numériques (has_*, is_*)
        - Retourne intersection stricte (présent partout)
    """
    if len(rows) == 0:
        return set()

    common = set(rows[0]['features'].keys())

    for row in rows[1:]:
        common &= set(row['features'].keys())

    # Filtrer flags booléens
    common = {
        k for k in common
        if not k.startswith('has_') and not k.startswith('is_')
    }

    return common


# Marqueur interne pour NaN avant imputation médiane
_NAN_MARKER = float('nan')


def _sanitize_vector(vector: List[float]) -> Tuple[List[float], bool]:
    """
    Remplace inf par sentinelle finie. NaN conservé pour imputation ultérieure.

    Args:
        vector : Valeurs features d'un run

    Returns:
        (vector_sanitized, has_nan)
        has_nan=True → run contient des NaN (sera imputé par médiane colonne)
        inf → remplacé par ±_INF_SENTINEL (signal préservé)

    Notes:
        - NaN = feature non calculable (ex: log_condition_delta si source inf)
          → imputation médiane colonne dans detect_outliers (run conservé)
        - inf = signal physique (ex: condition_number singulier) → sentinelle
        - -inf → -_INF_SENTINEL
        - Seul cas d'exclusion réelle : run entier vide (géré dans detect_outliers)
    """
    sanitized = []
    has_nan = False
    for v in vector:
        if np.isnan(v):
            sanitized.append(_NAN_MARKER)
            has_nan = True
        elif np.isposinf(v):
            sanitized.append(_INF_SENTINEL)
        elif np.isneginf(v):
            sanitized.append(-_INF_SENTINEL)
        else:
            sanitized.append(float(v))

    return sanitized, has_nan


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
        - inf remplacé par sentinelle ±1e15 (signal préservé, run inclus)
        - NaN → run exclu (donnée manquante)
        - Features communes : intersection stricte, sans has_* et is_*
    """
    common_features = _extract_common_features(rows)

    if len(common_features) == 0:
        print("  WARNING: Aucune feature commune — outliers detection skipped")
        return [], list(range(len(rows)))

    feature_names = sorted(common_features)

    features_matrix = []
    valid_indices = []
    n_inf_replaced = 0
    n_nan_imputed = 0

    for i, row in enumerate(rows):
        features = row['features']
        vector = [features[k] for k in feature_names]

        sanitized, has_nan = _sanitize_vector(vector)

        # Compter les inf remplacés (pour info)
        n_inf_replaced += sum(
            1 for v, s in zip(vector, sanitized)
            if not np.isnan(s) and not np.isnan(v) and v != s
        )

        features_matrix.append(sanitized)
        valid_indices.append(i)

    # Imputation médiane colonne pour NaN résiduels
    # NaN = feature non calculable (ex: log_condition_delta si source inf)
    # → médiane colonne = valeur neutre pour IsolationForest
    if features_matrix:
        n_cols = len(features_matrix[0])
        for col in range(n_cols):
            col_vals = [
                row[col] for row in features_matrix
                if not np.isnan(row[col])
            ]
            median_val = float(np.median(col_vals)) if col_vals else 0.0
            for row in features_matrix:
                if np.isnan(row[col]):
                    row[col] = median_val
                    n_nan_imputed += 1

    if n_inf_replaced > 0:
        print(f"  Inf remplacés par sentinelle (±{_INF_SENTINEL:.0e}) : {n_inf_replaced}")
    if n_nan_imputed > 0:
        print(f"  NaN imputés (médiane colonne) : {n_nan_imputed}")

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

    common_features = _extract_common_features(rows)

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