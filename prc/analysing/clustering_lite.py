"""
prc.analysing.clustering_lite

Responsabilité : Helpers pipeline features + interface vers clustering_peeling

Ce module expose :
    - Les helpers de préparation matrice (sanitize, log_transform, orthogonal)
      utilisés par clustering_peeling.py
    - prepare_matrix() : pipeline complet sanitize→log→scale→ortho
    - compute_tsne()   : projection 2D avec cache
    - run_clustering() : wrapper vers run_peeling()
    - run_clustering_stratified() : une passe peeling par layer actif

Pourquoi pas HDBSCAN direct ici :
    Le peeling résiduel (clustering_peeling.py) remplace le HDBSCAN simple.
    Les helpers restent ici — logique de préparation matrice indépendante
    de la stratégie de clustering.

Pipeline normalisation (inchangé) :
    1. _sanitize_value  : inf → ±1e15, nan → np.nan
    2. _log_transform   : sign(x)*log1p(|x|) si dynamique > DYNAMIC_THRESHOLD
    3. RobustScaler     : normalisation median/IQR
    4. _select_orthogonal_features : élimination redondances colinéaires
"""

import os
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from featuring.layers_lite import group_rows_by_layers


# Seuil dynamique log-transform
DYNAMIC_THRESHOLD = 1e6

# Seuil corrélation sélection orthogonale
CORRELATION_THRESHOLD = 0.85

# Features physiques — jamais exclues par la sélection orthogonale
PROTECTED_FEATURES = {
    'norm_ratio', 'norm_final', 'entropy_delta', 'effective_rank_delta',
    'log_condition_delta', 'spread_ratio', 'condition_number_svd_final',
}
PROTECTED_SUFFIXES = ('signal_finite_ratio', 'signal_norm_absolute')


# =============================================================================
# HELPERS PRÉPARATION MATRICE
# =============================================================================

def _extract_common_features(rows: List[Dict]) -> Set[str]:
    """Features présentes dans TOUS les runs, flags booléens exclus."""
    if not rows:
        return set()
    common = set(rows[0]['features'].keys())
    for row in rows[1:]:
        common &= set(row['features'].keys())
    return {k for k in common
            if not k.startswith('has_') and not k.startswith('is_')}


def _sanitize_value(v) -> float:
    """inf → ±1e15, nan → np.nan, finies → inchangées."""
    if v is None:         return np.nan
    v = float(v)
    if np.isnan(v):       return np.nan
    if np.isposinf(v):    return  1e15
    if np.isneginf(v):    return -1e15
    return v


def _log_transform(matrix: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    """
    Compression logarithmique par feature si dynamique > DYNAMIC_THRESHOLD.

    sign(x) * log1p(|x|) — zéro, signe et ordre préservés.
    """
    result = matrix.copy()
    transformed_cols = []

    for j in range(matrix.shape[1]):
        col     = matrix[:, j]
        abs_col = np.abs(col)
        nz      = abs_col[abs_col > 0]
        col_min = np.min(nz) if len(nz) > 0 else 1.0
        dynamic = np.max(abs_col) / (col_min + 1e-10)

        if dynamic > DYNAMIC_THRESHOLD:
            result[:, j] = np.sign(col) * np.log1p(abs_col)
            transformed_cols.append(True)
        else:
            transformed_cols.append(False)

    return result, transformed_cols


def _select_orthogonal_features(
    matrix_scaled  : np.ndarray,
    feature_names  : List[str],
    threshold      : float = CORRELATION_THRESHOLD,
    protected_names: Optional[Set[str]] = None,
) -> Tuple[List[int], List[str], Dict]:
    """
    Sélection features orthogonales — élimine redondances colinéaires.

    Features protégées insérées en premier (toutes conservées sauf constantes).
    Non-protégées évaluées par variance décroissante.

    Returns:
        (kept_indices, kept_names, exclusion_report)
    """
    n_features = matrix_scaled.shape[1]
    if n_features < 2:
        return list(range(n_features)), list(feature_names), {}

    if protected_names is None:
        protected_names = set(PROTECTED_FEATURES)
        for name in feature_names:
            if any(name.endswith(s) for s in PROTECTED_SUFFIXES):
                protected_names.add(name)

    protected_idx   = [i for i, k in enumerate(feature_names) if k in protected_names]
    unprotected_idx = [i for i, k in enumerate(feature_names) if k not in protected_names]

    variances    = np.var(matrix_scaled, axis=0)
    order_unprot = sorted(unprotected_idx, key=lambda i: variances[i], reverse=True)

    with np.errstate(invalid='ignore'):
        corr_matrix = np.abs(np.corrcoef(matrix_scaled.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)

    kept_indices     = []
    exclusion_report = {}

    for idx in protected_idx:
        if variances[idx] == 0:
            exclusion_report[feature_names[idx]] = 'constant_feature'
        else:
            kept_indices.append(idx)

    for idx in order_unprot:
        if variances[idx] == 0:
            exclusion_report[feature_names[idx]] = 'constant_feature'
            continue
        if not kept_indices or np.max(corr_matrix[idx, kept_indices]) < threshold:
            kept_indices.append(idx)
        else:
            rep_pos = np.argmax(corr_matrix[idx, kept_indices])
            exclusion_report[feature_names[idx]] = feature_names[kept_indices[rep_pos]]

    kept_sorted = sorted(kept_indices)
    return kept_sorted, [feature_names[i] for i in kept_sorted], exclusion_report


def prepare_matrix(
    rows          : List[Dict],
    feature_names : Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Pipeline complet préparation matrice.

    Sanitize → NaN→0.0 → log_transform → RobustScaler → orthogonal.

    Returns:
        (M_ortho, kept_names, meta)
    """
    if feature_names is None:
        common = _extract_common_features(rows)
        if not common:
            return np.empty((0, 0)), [], {}
        feature_names = sorted(common)

    M_raw = np.zeros((len(rows), len(feature_names)), dtype=float)
    for i, row in enumerate(rows):
        for j, k in enumerate(feature_names):
            v = _sanitize_value(row['features'].get(k))
            M_raw[i, j] = 0.0 if np.isnan(v) else v

    M_log, transformed_cols = _log_transform(M_raw)
    n_transformed = sum(transformed_cols)
    if n_transformed > 0:
        print(f"  Log-transform : {n_transformed}/{len(feature_names)} features")

    try:
        M_scaled = RobustScaler().fit_transform(M_log)
    except Exception as e:
        print(f"  WARNING: RobustScaler failed ({e}) — données brutes")
        M_scaled = M_log

    kept_indices, kept_names, exclusion_report = _select_orthogonal_features(
        M_scaled, feature_names
    )
    n_dropped = len(feature_names) - len(kept_names)
    if n_dropped > 0:
        print(f"  Features orthogonales : {len(kept_names)}/{len(feature_names)} "
              f"({n_dropped} redondantes écartées)")

    return M_scaled[:, kept_indices], kept_names, {
        'transformed_cols' : transformed_cols,
        'exclusion_report' : exclusion_report,
        'n_features_total' : len(feature_names),
        'n_features_ortho' : len(kept_names),
    }


def compute_tsne(
    M_ortho   : np.ndarray,
    cache_path: Optional[str] = None,
    perplexity: int = 30,
) -> np.ndarray:
    """
    Projection t-SNE 2D avec cache .npy optionnel.

    Returns:
        M_2d : np.ndarray (n, 2)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  t-SNE chargé depuis cache : {cache_path}")
        return np.load(cache_path)

    print("  Calcul t-SNE 2D...")
    n_pca = min(50, M_ortho.shape[1])
    M_pca = PCA(n_components=n_pca, random_state=42).fit_transform(M_ortho)
    M_2d  = TSNE(n_components=2, perplexity=perplexity,
                 max_iter=1000, random_state=42).fit_transform(M_pca)

    if cache_path:
        np.save(cache_path, M_2d)
        print(f"  t-SNE sauvegardé → {cache_path}")

    return M_2d


# =============================================================================
# INTERFACE PEELING
# =============================================================================

def _load_peeling_config(cfg_path: Optional[str] = None) -> Dict:
    """Charge config peeling depuis YAML ou chemin par défaut."""
    from analysing.clustering_peeling import load_config
    if cfg_path and os.path.exists(cfg_path):
        return load_config(cfg_path)
    default = os.path.join(os.path.dirname(__file__),
                           'configs', 'clustering_peeling.yaml')
    if os.path.exists(default):
        return load_config(default)
    raise FileNotFoundError(
        "Config peeling introuvable — fournir peeling_cfg_path ou "
        "placer clustering_peeling.yaml dans analysing/configs/"
    )


def run_clustering(
    rows             : List[Dict],
    peeling_config   : Optional[Dict] = None,
    peeling_cfg_path : Optional[str]  = None,
    run_regimes      : Optional[List[str]] = None,
    M_2d             : Optional[np.ndarray] = None,
    output_dir       : Optional[str] = None,
    label            : str = 'clustering',
    **kwargs,
) -> Optional[Dict]:
    """
    Clustering via residual peeling — remplace l'ancien HDBSCAN direct.

    Args:
        rows             : Liste {composition, features}
        peeling_config   : Config dict déjà chargée
        peeling_cfg_path : Chemin YAML (si config non fournie)
        run_regimes      : Régimes par run pour enrichissement namer
        M_2d             : t-SNE pour visualisations
        output_dir       : Répertoire sorties
        label            : Préfixe fichiers
        **kwargs         : Ignorés (rétrocompatibilité)

    Returns:
        {
            'n_clusters'     : int,
            'n_noise'        : int,        # résidu non résolu
            'n_samples'      : int,
            'n_features'     : int,
            'n_features_total': int,
            'labels'         : List[int],
            'valid_indices'  : List[int],
            'feature_names'  : List[str],
            'peeling_result' : Dict,
            'matrix_meta'    : Dict,
        }
        None si échec
    """
    from analysing.clustering_peeling import run_peeling

    if not rows:
        print("  WARNING: Aucune observation — clustering skipped")
        return None

    M_ortho, kept_names, matrix_meta = prepare_matrix(rows)

    if M_ortho.shape[0] < 5:
        print(f"  WARNING: Pas assez de samples ({M_ortho.shape[0]})")
        return None

    if peeling_config is None:
        peeling_config = _load_peeling_config(peeling_cfg_path)

    n   = len(rows)
    mcs = max(peeling_config.get('mcs_floor', 8), n // 30)
    ms  = max(peeling_config.get('ms_floor',  3), n // 100)

    peeling_result = run_peeling(
        M_ortho     = M_ortho,
        cfg         = peeling_config,
        mcs_global  = mcs,
        ms_global   = ms,
        M_2d        = M_2d,
        run_regimes = run_regimes,
        output_dir  = output_dir,
        label       = label,
    )

    labels     = peeling_result['labels'].tolist()
    n_clusters = peeling_result['n_clusters']
    n_noise    = peeling_result['n_unresolved']

    print(f"  Samples : {n} | Features : {len(kept_names)} | "
          f"Clusters : {n_clusters} | Résidu : {n_noise}")

    return {
        'n_clusters'      : n_clusters,
        'n_noise'         : n_noise,
        'n_samples'       : n,
        'n_features'      : len(kept_names),
        'n_features_total': matrix_meta.get('n_features_total', len(kept_names)),
        'labels'          : labels,
        'valid_indices'   : list(range(n)),
        'feature_names'   : kept_names,
        'peeling_result'  : peeling_result,
        'matrix_meta'     : matrix_meta,
    }


def run_clustering_stratified(
    rows             : List[Dict],
    peeling_config   : Optional[Dict] = None,
    peeling_cfg_path : Optional[str]  = None,
    run_regimes      : Optional[List[str]] = None,
    M_2d             : Optional[np.ndarray] = None,
    output_dir       : Optional[str] = None,
    label            : str = 'clustering',
    **kwargs,
) -> Dict:
    """
    Clustering par layers — une passe peeling par layer actif.

    Returns:
        {layer_name: result_dict}
    """
    layers_groups = group_rows_by_layers(rows)
    results       = {}

    for layer_name, layer_rows in layers_groups.items():
        if len(layer_rows) < 5:
            print(f"  [SKIP] {layer_name}: {len(layer_rows)} samples < 5")
            continue

        layer_label   = f'{label}_{layer_name}'
        layer_regimes = None
        if run_regimes and len(run_regimes) == len(rows):
            layer_indices = [i for i, r in enumerate(rows) if r in layer_rows]
            layer_regimes = [run_regimes[i] for i in layer_indices]

        result = run_clustering(
            rows             = layer_rows,
            peeling_config   = peeling_config,
            peeling_cfg_path = peeling_cfg_path,
            run_regimes      = layer_regimes,
            M_2d             = M_2d,
            output_dir       = output_dir,
            label            = layer_label,
        )

        if result is not None:
            results[layer_name] = result
            print(f"  ✓ {layer_name}: {len(layer_rows)} samples "
                  f"→ {result['n_clusters']} clusters")

    return results
