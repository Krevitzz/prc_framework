"""
prc.analysing.outliers_lite

Responsabilité : Détection outliers + récurrence atomics

Minimal : IsolationForest sklearn sur features COMMUNES uniquement
"""

import numpy as np
from typing import Dict, List, Tuple, Set
from sklearn.ensemble import IsolationForest


def _extract_common_features(rows: List[Dict]) -> Set[str]:
    """
    Identifie features présentes dans TOUS les runs.
    
    Args:
        rows : Liste {composition, features}
    
    Returns:
        Set features communes (intersection)
    
    Notes:
        - Exclut features non numériques (has_*)
        - Retourne intersection stricte (présent partout)
    
    Examples:
        >>> # Rank 2 : euclidean_norm, trace, condition_number
        >>> # Rank 3 : euclidean_norm, mode_variance_0
        >>> # → Communes : euclidean_norm uniquement
    """
    if len(rows) == 0:
        return set()
    
    # Features premier run
    common = set(rows[0]['features'].keys())
    
    # Intersection avec tous les autres
    for row in rows[1:]:
        common &= set(row['features'].keys())
    
    # Filtrer features non numériques
    common = {k for k in common if not k.startswith('has_')}
    
    return common


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
        - Compare uniquement features présentes dans TOUS runs
        - Pas de remplissage artificiel (NaN=0)
        - Features communes typiques : euclidean_norm, entropy
    
    Examples:
        >>> rows_mixed = [
        ...     # Rank 2 : euclidean_norm + trace + condition_number
        ...     {'features': {'euclidean_norm_final': 5.0, 'trace_final': 10.0}},
        ...     # Rank 3 : euclidean_norm + mode_variance_0
        ...     {'features': {'euclidean_norm_final': 6.0, 'mode_variance_0_final': 0.5}}
        ... ]
        >>> # Seules features communes : euclidean_norm_*
    """
    # Identifier features communes
    common_features = _extract_common_features(rows)
    
    if len(common_features) == 0:
        print("  WARNING: Aucune feature commune — outliers detection skipped")
        return [], list(range(len(rows)))
    
    # Trier pour ordre stable
    feature_names = sorted(common_features)
    
    # Extraire features matrix
    features_matrix = []
    valid_indices = []
    
    for i, row in enumerate(rows):
        features = row['features']
        
        # Extraire valeurs features communes
        vector = [features[k] for k in feature_names]
        
        # Skip si NaN/Inf présents (garder conceptuel, pas remplacer par 0)
        if not all(np.isfinite(v) for v in vector):
            continue
        
        features_matrix.append(vector)
        valid_indices.append(i)
    
    if len(features_matrix) < 2:
        print(f"  WARNING: Moins de 2 samples valides ({len(features_matrix)}) — outliers detection skipped")
        return [], list(range(len(rows)))
    
    # IsolationForest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(features_matrix)
    
    # Séparer outliers (-1) vs stables (1)
    outlier_indices = [valid_indices[i] for i, pred in enumerate(predictions) if pred == -1]
    stable_indices = [valid_indices[i] for i, pred in enumerate(predictions) if pred == 1]
    
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
    
    Examples:
        >>> recurrence = compute_atomic_recurrence(rows, outlier_indices, 'gamma_id')
        >>> recurrence['GAM-003']['fraction']
        0.75  # Présent dans 75% des outliers
    """
    subset_rows = [rows[i] for i in indices]
    n_total = len(subset_rows)
    
    if n_total == 0:
        return {}
    
    # Compter occurrences
    counts = {}
    for row in subset_rows:
        atomic_id = row['composition'][atomic_key]
        counts[atomic_id] = counts.get(atomic_id, 0) + 1
    
    # Calculer fractions
    recurrence = {}
    for atomic_id, count in counts.items():
        recurrence[atomic_id] = {
            'count': count,
            'fraction': count / n_total,
            'total_subset': n_total
        }
    
    # Trier par fraction décroissante
    recurrence = dict(sorted(recurrence.items(), key=lambda x: x[1]['fraction'], reverse=True))
    
    return recurrence


def analyze_outliers(rows: List[Dict], contamination: float = 0.1) -> Dict:
    """
    Analyse complète outliers.
    
    Returns:
        {
            'n_outliers': int,
            'n_stables': int,
            'outlier_fraction': float,
            'outlier_indices': List[int],
            'stable_indices': List[int],
            'recurrence': {
                'gamma': {...},
                'encoding': {...},
                'modifier': {...}
            },
            'n_features_used': int
        }
    """
    outlier_idx, stable_idx = detect_outliers(rows, contamination)
    
    n_outliers = len(outlier_idx)
    n_stables = len(stable_idx)
    n_total = n_outliers + n_stables
    
    # Récurrence atomics dans outliers
    gamma_rec = compute_atomic_recurrence(rows, outlier_idx, 'gamma_id')
    encoding_rec = compute_atomic_recurrence(rows, outlier_idx, 'encoding_id')
    modifier_rec = compute_atomic_recurrence(rows, outlier_idx, 'modifier_id')
    
    # Compter features communes utilisées
    common_features = _extract_common_features(rows)
    
    return {
        'n_outliers': n_outliers,
        'n_stables': n_stables,
        'outlier_fraction': n_outliers / n_total if n_total > 0 else 0.0,
        'outlier_indices': outlier_idx,
        'stable_indices': stable_idx,
        'recurrence': {
            'gamma': gamma_rec,
            'encoding': encoding_rec,
            'modifier': modifier_rec
        },
        'n_features_used': len(common_features)
    }
