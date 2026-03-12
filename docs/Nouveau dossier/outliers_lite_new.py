"""
analysing/outliers_lite_new.py

Responsabilité : Détection outliers + récurrence atomics.

Reçoit AnalysingData — M passé directement à IsolationForest.
Pas de reconstruction de matrice. inf déjà → nan dans parquet_filter.
"""

import numpy as np
from typing import Dict, Tuple

from sklearn.ensemble import IsolationForest

from analysing.parquet_filter import AnalysingData


# Valeur sentinelle pour nan → 0.0 dans IsolationForest
# (IsolationForest sklearn n'accepte pas nan)
_NAN_FILL = 0.0


def _sanitize_for_isolation_forest(M: np.ndarray) -> np.ndarray:
    """
    Remplace nan par 0.0 pour IsolationForest.
    inf déjà converti en nan par parquet_filter.
    Retourne une copie — M original non modifié.
    """
    M_clean = M.copy()
    M_clean[~np.isfinite(M_clean)] = _NAN_FILL
    return M_clean


def detect_outliers(
    data         : AnalysingData,
    contamination: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Détecte outliers via IsolationForest.

    Args:
        data          : AnalysingData
        contamination : Fraction attendue outliers (défaut 10%)

    Returns:
        (outlier_mask, stable_mask) — booléens numpy (n,)
    """
    if data.n < 2:
        print(f"  WARNING: Moins de 2 samples — outliers skipped")
        return np.zeros(data.n, dtype=bool), np.ones(data.n, dtype=bool)

    # Features sans health_*
    feat_mask = np.array([not f.startswith('health_') for f in data.feat_names])
    M_clean   = _sanitize_for_isolation_forest(data.M[:, feat_mask])

    n_nan = int(np.sum(np.isnan(data.M[:, feat_mask])))
    if n_nan > 0:
        print(f"  NaN → 0.0 placeholder : {n_nan} valeurs")

    clf         = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    predictions = clf.fit_predict(M_clean)

    outlier_mask = predictions == -1
    stable_mask  = predictions ==  1

    print(f"  Features utilisées : {int(feat_mask.sum())}")
    print(f"  Samples analysés   : {data.n}")

    return outlier_mask, stable_mask


def compute_atomic_recurrence(
    entity_arr  : np.ndarray,
    mask        : np.ndarray,
) -> Dict[str, Dict]:
    """
    Calcule récurrence atomics dans le subset masqué.

    Args:
        entity_arr : (n,) array IDs entité (gamma_ids, encoding_ids, ...)
        mask       : (n,) booléen — subset à analyser (ex: outliers)

    Returns:
        {atomic_id: {count, fraction, total_subset}}
    """
    subset     = entity_arr[mask]
    n_total    = len(subset)

    if n_total == 0:
        return {}

    unique, counts = np.unique(subset, return_counts=True)
    recurrence = {
        str(uid): {
            'count'       : int(cnt),
            'fraction'    : float(cnt / n_total),
            'total_subset': n_total,
        }
        for uid, cnt in zip(unique, counts)
    }
    return dict(sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True))


def analyze_outliers(data: AnalysingData, contamination: float = 0.1) -> Dict:
    """
    Analyse complète outliers.

    Args:
        data          : AnalysingData
        contamination : Fraction attendue outliers

    Returns:
        {
            'n_outliers'      : int,
            'n_stables'       : int,
            'outlier_fraction': float,
            'outlier_mask'    : np.ndarray (n,) bool,
            'stable_mask'     : np.ndarray (n,) bool,
            'recurrence'      : {'gamma': {...}, 'encoding': {...}, 'modifier': {...}},
            'n_features_used' : int,
        }
    """
    outlier_mask, stable_mask = detect_outliers(data, contamination)

    n_outliers = int(outlier_mask.sum())
    n_stables  = int(stable_mask.sum())
    n_total    = n_outliers + n_stables

    feat_mask = np.array([not f.startswith('health_') for f in data.feat_names])

    return {
        'n_outliers'      : n_outliers,
        'n_stables'       : n_stables,
        'outlier_fraction': n_outliers / n_total if n_total > 0 else 0.0,
        'outlier_mask'    : outlier_mask,
        'stable_mask'     : stable_mask,
        'recurrence': {
            'gamma'   : compute_atomic_recurrence(data.gamma_ids,    outlier_mask),
            'encoding': compute_atomic_recurrence(data.encoding_ids, outlier_mask),
            'modifier': compute_atomic_recurrence(data.modifier_ids, outlier_mask),
        },
        'n_features_used': int(feat_mask.sum()),
    }
