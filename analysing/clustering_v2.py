"""
analysing/clustering_v2.py

Responsabilité : Préparation matrice + interface vers clustering_peeling.

Reçoit AnalysingData — prepare_matrix opère sur data.M directement.
run_clustering reçoit M_ortho directement — plus de prepare_matrix interne.

Pipeline normalisation (inchangé) :
    1. sanitize  : nan → 0.0, inf → déjà nan depuis parquet_filter
    2. log_transform : sign(x)*log1p(|x|) si dynamique > DYNAMIC_THRESHOLD
    3. RobustScaler  : normalisation median/IQR
    4. _select_orthogonal_features : élimination redondances colinéaires
"""

import os
import numpy as np
from typing import Dict, List, Optional, Set, Tuple

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    _UMAP_AVAILABLE = True
except ImportError:
    _UMAP_AVAILABLE = False

from analysing.parquet_filter_v2 import AnalysingData


DYNAMIC_THRESHOLD    = 1e6
CORRELATION_THRESHOLD = 0.85

PROTECTED_FEATURES_V7: Set[str] = {
    'ps_norm_ratio',
    'f2_von_neumann_entropy_delta',
    'ps_rank_delta',
    'ps_condition_ratio',
    'f1_nuclear_frobenius_ratio_final',
    'f1_condition_number_final',
    'f4_lyapunov_empirical_mean',
    'f7_dmd_spectral_radius',
}


# =============================================================================
# HELPERS PRÉPARATION MATRICE
# =============================================================================

def _log_transform(matrix: np.ndarray) -> Tuple[np.ndarray, List[bool]]:
    """sign(x)*log1p(|x|) si dynamique > DYNAMIC_THRESHOLD."""
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
    """Sélection features orthogonales — élimine redondances colinéaires."""
    n_features = matrix_scaled.shape[1]
    if n_features < 2:
        return list(range(n_features)), list(feature_names), {}

    if protected_names is None:
        protected_names = PROTECTED_FEATURES_V7

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


# =============================================================================
# PREPARE MATRIX — opère sur AnalysingData
# =============================================================================

def prepare_matrix(
    data              : AnalysingData,
    corr_threshold    : float = CORRELATION_THRESHOLD,
    protected_features: Optional[Set[str]] = None,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Pipeline complet préparation matrice depuis AnalysingData.

    Sanitize (nan→0) → log_transform → RobustScaler → orthogonal.
    data.M non modifié — opère sur une copie.

    Args:
        data           : AnalysingData
        corr_threshold : Seuil corrélation orthogonalisation

    Returns:
        (M_ortho, kept_names, matrix_meta)
        M_ortho    : (n, F_ortho) float64 — prêt pour clustering
        kept_names : noms features conservées
        matrix_meta: dict stats
    """
    M_ml, feat_names = data.features_for_ml()
    M_raw            = M_ml.astype(np.float64)

    # nan → 0.0 pour RobustScaler
    n_nan = int(np.sum(np.isnan(M_raw)))
    M_raw[np.isnan(M_raw)] = 0.0

    M_log, transformed_cols = _log_transform(M_raw)
    n_transformed = sum(transformed_cols)
    if n_transformed > 0:
        print(f"  Log-transform : {n_transformed}/{len(feat_names)} features")

    try:
        M_scaled = RobustScaler().fit_transform(M_log)
    except Exception as e:
        print(f"  WARNING: RobustScaler failed ({e}) — données brutes")
        M_scaled = M_log

    pf = protected_features if protected_features is not None else PROTECTED_FEATURES_V7
    kept_indices, kept_names, exclusion_report = _select_orthogonal_features(
        M_scaled, feat_names, threshold=corr_threshold, protected_names=pf
    )
    n_dropped = len(feat_names) - len(kept_names)
    if n_dropped > 0:
        print(f"  Features orthogonales : {len(kept_names)}/{len(feat_names)} "
              f"({n_dropped} redondantes écartées)")

    return M_scaled[:, kept_indices], kept_names, {
        'transformed_cols' : transformed_cols,
        'exclusion_report' : exclusion_report,
        'n_features_total' : len(feat_names),
        'n_features_ortho' : len(kept_names),
        'n_nan_filled'     : n_nan,
    }


# =============================================================================
# T-SNE
# =============================================================================

def compute_projection(
    M_ortho   : np.ndarray,
    cfg       : Optional[Dict] = None,
    cache_path: Optional[str] = None,
) -> np.ndarray:
    """
    Projection 2D avec cache .npy optionnel.

    Algorithme :
      n < umap_threshold → t-SNE (perplexity auto)
      n ≥ umap_threshold → UMAP si disponible, sinon t-SNE

    Perplexity auto : min(50, max(5, sqrt(n) // 3))
    umap_threshold  : cfg.get('umap_threshold', 50000)
    """
    cfg = cfg or {}

    if cache_path and os.path.exists(cache_path):
        print(f"  Projection chargée depuis cache : {cache_path}")
        return np.load(cache_path)

    n              = M_ortho.shape[0]
    umap_threshold = cfg.get('umap_threshold', 50_000)
    use_umap       = n >= umap_threshold and _UMAP_AVAILABLE

    n_pca = min(50, M_ortho.shape[1])
    M_pca = PCA(n_components=n_pca, random_state=42).fit_transform(M_ortho)

    if use_umap:
        print(f"  Calcul UMAP 2D (n={n})...")
        M_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(M_pca)
    else:
        perplexity = min(50, max(5, int(np.sqrt(n)) // 3))
        print(f"  Calcul t-SNE 2D (n={n}, perplexity={perplexity})...")
        M_2d = TSNE(n_components=2, perplexity=perplexity,
                    max_iter=1000, random_state=42).fit_transform(M_pca)

    if cache_path:
        np.save(cache_path, M_2d)
        print(f"  Projection sauvegardée → {cache_path}")

    return M_2d


# =============================================================================
# INTERFACE PEELING — reçoit M_ortho directement
# =============================================================================

def run_clustering(
    M_ortho     : np.ndarray,
    feat_names  : List[str],
    peeling_cfg : Dict,
    M_2d        : Optional[np.ndarray] = None,
    output_dir  : Optional[str] = None,
    label       : str = 'clustering',
    save_debug  : bool = False,
    verbose     : bool = False,
) -> Optional[Dict]:
    """
    Clustering via residual peeling.

    Reçoit M_ortho directement — pas de prepare_matrix interne.

    Args:
        M_ortho    : (n, F_ortho) depuis prepare_matrix
        feat_names : noms colonnes M_ortho
        peeling_cfg: cfg['peeling'] depuis analysing_default.yaml
        M_2d       : projection 2D optionnelle pour visualisations
        output_dir : répertoire sorties debug
        label      : préfixe fichiers
        save_debug : sauvegarder labels + trace JSON
        verbose    : affichage détaillé peeling

    Returns:
        dict résultat peeling ou None si échec
    """
    from analysing.clustering_peeling_v2 import run_peeling

    if M_ortho.shape[0] < 5:
        print(f"  WARNING: Pas assez de samples ({M_ortho.shape[0]})")
        return None

    n   = M_ortho.shape[0]
    mcs = max(peeling_cfg.get('mcs_floor', 8), n // 30)
    ms  = max(peeling_cfg.get('ms_floor',  3), n // 100)

    peeling_result = run_peeling(
        M_ortho    = M_ortho,
        cfg        = peeling_cfg,
        mcs_global = mcs,
        ms_global  = ms,
        M_2d       = M_2d,
        output_dir = output_dir,
        label      = label,
        save_debug = save_debug,
        verbose    = verbose,
    )

    labels     = peeling_result['labels'].tolist()
    n_clusters = peeling_result['n_clusters']
    n_noise    = peeling_result['n_unresolved']

    print(f"  Samples : {n} | Features : {len(feat_names)} | "
          f"Clusters : {n_clusters} | Résidu : {n_noise}")

    return {
        'n_clusters'  : n_clusters,
        'n_noise'     : n_noise,
        'n_samples'   : n,
        'n_features'  : len(feat_names),
        'labels'      : labels,
        'feat_names'  : feat_names,
        'peeling_result': peeling_result,
    }
