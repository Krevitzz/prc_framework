"""
prc.analysing.clustering_lite

Responsabilité : Clustering (HDBSCAN) sur features COMMUNES cross-runs

Pipeline normalisation :
    1. _sanitize_value  : inf → sentinelle ±1e15, nan → np.nan
    2. _log_transform   : compression log si dynamique > DYNAMIC_THRESHOLD (1e6)
                          sign(x) * log1p(|x|) — préserve signe, ordre, variations
    3. RobustScaler     : normalisation median/IQR sur espace log
    4. HDBSCAN          : clusters découverts automatiquement, pas de n fixe

Pourquoi pas KMeans :
    - KMeans overflow sur features finies extrêmes (1e238) en float32 interne
    - KMeans impose n_clusters fixe → arbitraire
    - KMeans distances euclidiennes → non comparables entre features multi-échelles

Pourquoi pas clipping :
    - Clip arbitraire détruit les variations relatives sur grands nombres
    - 1e238 vs 1e200 est une différence physique réelle

Pourquoi log_transform :
    - log1p(1e238) ≈ 548, log1p(1e200) ≈ 460 → différence préservée, finie
    - Appliqué uniquement si dynamique > 1e6 — pas de transformation inutile
"""

import numpy as np
from typing import Dict, List, Set, Tuple
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import RobustScaler

from featuring.layers_lite import group_rows_by_layers


# Seuil de dynamique au-dessus duquel log-transform est appliqué par feature
DYNAMIC_THRESHOLD = 1e6

# Seuil de corrélation au-dessus duquel deux features sont considérées redondantes
# |r| > CORRELATION_THRESHOLD → une seule gardée (la plus discriminante)
CORRELATION_THRESHOLD = 0.85

# Ratio pour min_cluster_size adaptatif :
# min_cluster_size = max(5, n_valid // (n_features_ortho * MIN_CLUSTER_RATIO))
# Valeur basse → plus permissif | Valeur haute → plus strict
MIN_CLUSTER_RATIO = 2


# =============================================================================
# HELPERS
# =============================================================================

def _extract_common_features(rows: List[Dict]) -> Set[str]:
    """
    Features présentes dans TOUS les runs, flags booléens exclus.

    Returns:
        Set features communes (has_* et is_* exclus)
    """
    if len(rows) == 0:
        return set()

    common = set(rows[0]['features'].keys())
    for row in rows[1:]:
        common &= set(row['features'].keys())

    return {
        k for k in common
        if not k.startswith('has_') and not k.startswith('is_')
    }


def _sanitize_value(v: float) -> float:
    """
    Prépare valeur pour log_transform + RobustScaler.

    - NaN → np.nan  (run exclu en aval)
    - +inf → +1e15  (fini, log1p gérable : log1p(1e15) ≈ 35)
    - -inf → -1e15
    - Finies extrêmes (1e238...) → inchangées, log_transform s'en charge

    Notes:
        Pas de clipping arbitraire — log_transform préserve les variations.
    """
    if np.isnan(v):
        return np.nan
    elif np.isposinf(v):
        return 1e15
    elif np.isneginf(v):
        return -1e15
    return float(v)


def _log_transform(matrix: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    """
    Compression logarithmique par feature si grande dynamique.

    Pour chaque colonne :
        dynamique = max(|col|) / (min(|col| > 0) + ε)
        Si dynamique > DYNAMIC_THRESHOLD → sign(x) * log1p(|x|)
        Sinon → colonne inchangée

    sign(x) * log1p(|x|) :
        - log1p(0) = 0         (zéro préservé)
        - log1p(1e238) ≈ 548   (fini, comparable)
        - log1p(1e15)  ≈ 35    (inf remplacés → finies)
        - Ordre et variations relatives préservés
        - Signe préservé

    Args:
        matrix : np.ndarray (n_samples, n_features)

    Returns:
        (matrix_transformed, transformed_cols: List[bool])
    """
    result = matrix.copy()
    n_features = matrix.shape[1]
    transformed_cols = []

    for j in range(n_features):
        col = matrix[:, j]
        abs_col = np.abs(col)

        col_max = np.max(abs_col)
        nonzero = abs_col[abs_col > 0]
        col_min_nonzero = np.min(nonzero) if len(nonzero) > 0 else 1.0

        dynamic = col_max / (col_min_nonzero + 1e-10)

        if dynamic > DYNAMIC_THRESHOLD:
            result[:, j] = np.sign(col) * np.log1p(abs_col)
            transformed_cols.append(True)
        else:
            transformed_cols.append(False)

    return result, transformed_cols


def _select_orthogonal_features(
    matrix_scaled: np.ndarray,
    feature_names: List[str],
    threshold: float = CORRELATION_THRESHOLD
) -> tuple:
    """
    Sélectionne features orthogonales — élimine redondances colinéaires.

    Algorithme glouton :
        1. Matrice corrélation absolute sur matrix_scaled
        2. Parcours features dans l'ordre décroissant de variance
           (les plus discriminantes en premier)
        3. Ajouter une feature si |r| < threshold avec toutes déjà gardées
        4. Sinon : l'écarter (redondante avec une feature déjà représentée)

    Args:
        matrix_scaled  : np.ndarray (n_samples, n_features) — déjà normalisé
        feature_names  : List[str] longueur n_features
        threshold      : float seuil corrélation (défaut CORRELATION_THRESHOLD)

    Returns:
        (kept_indices, kept_names, exclusion_report)
        kept_indices      : List[int] indices colonnes gardées
        kept_names        : List[str] noms features gardées
        exclusion_report  : Dict {feature_écartée: feature_représentante}

    Notes:
        - Variance décroissante → on garde la plus informative du groupe
        - |r| utilisé (corrélation négative = redondance aussi)
        - features avec variance nulle écartées d'office (constantes)
        - Si < 2 features : retourner toutes (pas de corrélation calculable)
    """
    n_features = matrix_scaled.shape[1]

    if n_features < 2:
        return list(range(n_features)), list(feature_names), {}

    # Variance par feature — ordre décroissant = plus discriminant en premier
    variances = np.var(matrix_scaled, axis=0)
    order = np.argsort(variances)[::-1]  # indices triés variance décroissante

    # Matrice corrélation absolute
    # Protection : features constantes → variance 0 → corrélation NaN
    with np.errstate(invalid='ignore'):
        corr_matrix = np.abs(np.corrcoef(matrix_scaled.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)  # constante = redondante

    kept_indices = []
    exclusion_report = {}

    for idx in order:
        if variances[idx] == 0:
            # Feature constante — inutile pour clustering
            exclusion_report[feature_names[idx]] = 'constant_feature'
            continue

        # Vérifier corrélation avec toutes les features déjà gardées
        if len(kept_indices) == 0:
            kept_indices.append(idx)
            continue

        max_corr = np.max(corr_matrix[idx, kept_indices])

        if max_corr < threshold:
            kept_indices.append(idx)
        else:
            # Trouver la feature représentante (celle avec laquelle corrélée)
            most_correlated_pos = np.argmax(corr_matrix[idx, kept_indices])
            representative = feature_names[kept_indices[most_correlated_pos]]
            exclusion_report[feature_names[idx]] = representative

    # Trier indices pour reproductibilité (ordre colonne original)
    kept_indices_sorted = sorted(kept_indices)
    kept_names = [feature_names[i] for i in kept_indices_sorted]

    return kept_indices_sorted, kept_names, exclusion_report


# =============================================================================
# CLUSTERING
# =============================================================================

def run_clustering(rows: List[Dict], **kwargs) -> Dict:
    """
    Clustering HDBSCAN sur features COMMUNES.

    Pipeline :
        1. Extraction features communes (sans has_* / is_*)
        2. Sanitize : inf → sentinelle, nan → exclu
        3. _log_transform : compression features à grande dynamique
        4. RobustScaler (median/IQR) sur espace log
        5. HDBSCAN : découverte automatique clusters

    Args:
        rows   : Liste {composition, features}
        kwargs : Paramètres HDBSCAN (ex: min_cluster_size=10)

    Returns:
        {
            'n_clusters'      : int,        # découverts (label != -1)
            'n_noise'         : int,        # bruit (label == -1)
            'n_samples'       : int,        # samples valides
            'n_features'      : int,
            'labels'          : List[int],  # -1 = bruit
            'valid_indices'   : List[int],
            'feature_names'   : List[str],
            'transformed_cols': List[bool]  # quelles features log-transformées
        }
        None si échec

    Notes:
        - min_cluster_size = max(5, n_samples // 10) si non fourni
        - n_noise élevé = dataset trop hétérogène ou min_cluster_size trop grand
    """
    common_features = _extract_common_features(rows)

    if len(common_features) == 0:
        print("  WARNING: Aucune feature commune — clustering skipped")
        return None

    feature_names = sorted(common_features)

    features_matrix = []
    valid_indices = []

    for i, row in enumerate(rows):
        vector = [_sanitize_value(row['features'][k]) for k in feature_names]

        if any(np.isnan(v) for v in vector):
            continue

        features_matrix.append(vector)
        valid_indices.append(i)

    n_valid = len(features_matrix)

    if n_valid < 5:
        print(f"  WARNING: Pas assez de samples valides ({n_valid})")
        return None

    matrix = np.array(features_matrix, dtype=float)

    # Log-transform features à grande dynamique
    matrix, transformed_cols = _log_transform(matrix)
    n_transformed = sum(transformed_cols)
    if n_transformed > 0:
        print(f"  Log-transform : {n_transformed}/{len(feature_names)} features")

    # RobustScaler sur espace log
    try:
        matrix_scaled = RobustScaler().fit_transform(matrix)
    except Exception as e:
        print(f"  WARNING: RobustScaler failed ({e}) — données brutes")
        matrix_scaled = matrix

    # Sélection features orthogonales — élimine redondances colinéaires
    threshold = kwargs.get('correlation_threshold', CORRELATION_THRESHOLD)
    kept_indices, kept_names, exclusion_report = _select_orthogonal_features(
        matrix_scaled, feature_names, threshold
    )

    if len(kept_indices) < len(feature_names):
        n_dropped = len(feature_names) - len(kept_indices)
        print(f"  Features orthogonales : {len(kept_indices)}/{len(feature_names)} "
              f"({n_dropped} redondantes écartées)")

    matrix_ortho = matrix_scaled[:, kept_indices]

    # HDBSCAN sur espace orthogonal
    n_ortho = len(kept_names)
    min_cluster_size = kwargs.get(
        'min_cluster_size',
        max(5, n_valid // (n_ortho * MIN_CLUSTER_RATIO))
    )

    try:
        labels = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(matrix_ortho)

        n_clusters = len(set(labels) - {-1})
        n_noise    = int(np.sum(labels == -1))

        print(f"  Samples : {n_valid} | Features : {len(kept_names)} | "
              f"Clusters : {n_clusters} | Bruit : {n_noise}")

        return {
            'n_clusters'       : n_clusters,
            'n_noise'          : n_noise,
            'n_samples'        : n_valid,
            'n_features'       : len(kept_names),
            'n_features_total' : len(feature_names),
            'labels'           : labels.tolist(),
            'valid_indices'    : valid_indices,
            'feature_names'    : kept_names,
            'feature_names_all': feature_names,
            'transformed_cols' : transformed_cols,
            'exclusion_report' : exclusion_report,
        }

    except Exception as e:
        print(f"  WARNING: HDBSCAN failed — {e}")
        return None


def run_clustering_stratified(rows: List[Dict], **kwargs) -> Dict:
    """
    Clustering par layers (découverte automatique).

    Args:
        rows   : Liste {composition, features, layers}
        kwargs : Transmis à run_clustering

    Returns:
        {layer_name: result_dict, ...}
    """
    layers_groups = group_rows_by_layers(rows)
    results = {}

    for layer_name, layer_rows in layers_groups.items():
        if len(layer_rows) < 5:
            print(f"  [SKIP] {layer_name}: {len(layer_rows)} samples < 5")
            continue

        result = run_clustering(layer_rows, **kwargs)

        if result is not None:
            results[layer_name] = result
            print(f"  ✓ {layer_name}: {len(layer_rows)} samples")

    return results