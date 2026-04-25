"""
Matérialisation et transformation de la matrice pour le clustering.

Matérialise les features à la demande depuis AnalysingData,
transforme (imputation NaN, log, RobustScaler, orthogonalisation),
retourne M_ortho. L'appelant libère explicitement après usage.

@ROLE    Matérialisation + transformation : AnalysingData → M_ortho (une seule matrice)
@LAYER   analysing

@EXPORTS
  MatrixMeta                                         → dataclass | métadonnées transform
  build_ml_feature_list(features_applicable)          → List[str] | features pour le ML
  materialize_and_transform(data, strate, cfg)        → tuple     | M_ortho + meta

@LIFECYCLE
  CREATES  M_ortho        np.ndarray (n, F) float64 — UNE SEULE matrice lourde
  RECEIVES AnalysingData   depuis hub
  PASSES   M_ortho         vers hub → clustering (puis libéré)

@CONFORMITY
  OK   health_* et meta_* exclus du ML, restent accessibles pour namer (Q1)
  OK   Une seule matrice lourde à la fois (SD-3)
  OK   Constantes de transform dans le cfg YAML (P4)
  OK   NaN runtime imputés médiane, NaN structurels déjà retirés par stratification
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import warnings

import numpy as np
from sklearn.preprocessing import RobustScaler


# =========================================================================
# PREFIXES EXCLUS DU ML
# =========================================================================

# health_* : diagnostiques (has_inf, is_collapsed) — pas des mesures physiques
# meta_*   : métadonnées numériques du run (n_svd, turbulence, t_effective)
#
# Ces features restent accessibles via AnalysingData.materialize_features()
# pour le namer (CND utilise health_has_inf) et la validation.

ML_EXCLUDED_PREFIXES: Tuple[str, ...] = ('health_', 'meta_')


# =========================================================================
# MATRIX META — métadonnées de la transformation
# =========================================================================

@dataclass
class MatrixMeta:
    """Métadonnées de la transformation matrice.

    Permet de tracer exactement ce qui a été fait :
    quelles features log-transformées, lesquelles exclues par ortho,
    combien de NaN imputés.
    """
    n_runs:             int
    n_features_input:   int               # avant ortho
    n_features_ortho:   int               # après ortho
    feat_names_input:   List[str]         # noms avant ortho
    feat_names_ortho:   List[str]         # noms après ortho
    log_transformed:    List[str]         # noms des features log-transformées
    ortho_excluded:     Dict[str, str]    # {feature: corrélée_avec}
    n_nan_imputed:      int
    nan_mask:           np.ndarray        # (n, F_input) bool — positions NaN avant imputation
    ortho_threshold:    Optional[float]


# =========================================================================
# SÉLECTION FEATURES ML
# =========================================================================

def build_ml_feature_list(features_applicable: List[str]) -> List[str]:
    """Retire health_* et meta_* de la liste des features pour le clustering.

    Ces features ne sont pas des mesures physiques :
    - health_* sont des diagnostiques binaires
    - meta_* sont des métadonnées numériques du run (n_svd, turbulence, t_effective)

    Elles restent accessibles via AnalysingData.materialize_features() pour
    le namer et la validation.
    """
    return [f for f in features_applicable
            if not any(f.startswith(p) for p in ML_EXCLUDED_PREFIXES)]


# =========================================================================
# LOG TRANSFORM
# =========================================================================

def _log_transform(M: np.ndarray,
                    feat_names: List[str],
                    dynamic_threshold: float) -> Tuple[np.ndarray, List[str]]:
    """Log-transform sign-preserving pour les features à haute dynamique.

    Pour chaque colonne : si max/min > dynamic_threshold,
    applique sign(x) * log1p(|x|).

    Args:
        M : matrice (n, F) float64. Modifiée in-place.
        feat_names : noms des features.
        dynamic_threshold : seuil de dynamique pour déclencher le log.

    Returns:
        (M_transformed, noms_features_transformées)
    """
    transformed_names = []

    for j in range(M.shape[1]):
        col = M[:, j]
        abs_col = np.abs(col)
        nz = abs_col[abs_col > 0]
        if len(nz) == 0:
            continue

        col_min = np.min(nz)
        dynamic = np.max(abs_col) / (col_min + 1e-10)

        if dynamic > dynamic_threshold:
            M[:, j] = np.sign(col) * np.log1p(abs_col)
            transformed_names.append(feat_names[j])

    return M, transformed_names


# =========================================================================
# ORTHOGONALISATION (suppression quasi-doublons)
# =========================================================================

def _select_orthogonal_features(M_scaled: np.ndarray,
                                  feat_names: List[str],
                                  threshold: float,
                                  protected_names: Optional[Set[str]] = None
                                  ) -> Tuple[List[int], List[str], Dict[str, str]]:
    """Sélection de features orthogonales par corrélation.

    Les features protégées sont gardées en priorité.
    Parmi les non-protégées, sélection par variance décroissante,
    rejet si corrélation > threshold avec une feature déjà gardée.

    Returns:
        (kept_indices, kept_names, {excluded_name: correlated_with})
    """
    n_features = M_scaled.shape[1]
    if n_features < 2:
        return list(range(n_features)), list(feat_names), {}

    protected = protected_names or set()
    protected_idx = [i for i, k in enumerate(feat_names) if k in protected]
    unprotected_idx = [i for i, k in enumerate(feat_names) if k not in protected]

    variances = np.var(M_scaled, axis=0)
    order_unprot = sorted(unprotected_idx, key=lambda i: variances[i], reverse=True)

    with np.errstate(invalid='ignore'):
        corr_matrix = np.abs(np.corrcoef(M_scaled.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)

    kept_indices = []
    exclusion_report = {}

    # Protected first
    for idx in protected_idx:
        if variances[idx] == 0:
            exclusion_report[feat_names[idx]] = 'constant_feature'
        else:
            kept_indices.append(idx)

    # Unprotected by variance
    for idx in order_unprot:
        if variances[idx] == 0:
            exclusion_report[feat_names[idx]] = 'constant_feature'
            continue
        if not kept_indices or np.max(corr_matrix[idx, kept_indices]) < threshold:
            kept_indices.append(idx)
        else:
            rep_pos = np.argmax(corr_matrix[idx, kept_indices])
            exclusion_report[feat_names[idx]] = feat_names[kept_indices[rep_pos]]

    kept_sorted = sorted(kept_indices)
    return kept_sorted, [feat_names[i] for i in kept_sorted], exclusion_report


# =========================================================================
# POINT D'ENTRÉE — MATÉRIALISATION + TRANSFORMATION
# =========================================================================

def materialize_and_transform(data,  # AnalysingData
                                strate,  # Strate
                                cfg: Dict
                                ) -> Tuple[np.ndarray, List[str], MatrixMeta]:
    """Matérialise et transforme pour le clustering.

    Flux :
    1. Sélection features ML (sans health_*, meta_*)
    2. Matérialisation numpy depuis PyArrow (colonnes + lignes de la strate)
    3. Imputation NaN runtime → médiane colonne
    4. Log-transform (features haute dynamique)
    5. RobustScaler
    6. Orthogonalisation (corrélations > seuil)

    Args:
        data : AnalysingData avec références lazy.
        strate : Strate définissant les indices et features applicables.
        cfg : section du YAML analysing (orthogonalization, etc.)

    Returns:
        (M_ortho, feat_names_ortho, matrix_meta)
        L'APPELANT est responsable de libérer M_ortho (del M_ortho) après usage.
    """
    # --- Config ---
    ortho_cfg = cfg.get('orthogonalization', {})
    corr_threshold = ortho_cfg.get('correlation_threshold', 0.98)
    ortho_enabled = ortho_cfg.get('enabled', True)
    protected = set(ortho_cfg.get('protected_features', []))
    dynamic_threshold = cfg.get('log_dynamic_threshold', 1e6)

    # --- 1. Features ML ---
    ml_features = build_ml_feature_list(strate.features_applicable)

    print(f"  [prepare] {strate.strate_id} : {strate.n_runs} runs × "
          f"{len(ml_features)} features ML")

    # --- 2. Matérialisation ---
    M_raw = data.materialize_features(
        columns=ml_features,
        rows=strate.run_indices
    ).astype(np.float64)

    # --- 3. NaN mask + imputation médiane ---
    nan_mask = np.isnan(M_raw)
    n_nan = int(np.sum(nan_mask))

    for j in range(M_raw.shape[1]):
        col_nans = nan_mask[:, j]
        if np.any(col_nans):
            finite_vals = M_raw[~col_nans, j]
            if len(finite_vals) > 0:
                M_raw[col_nans, j] = np.median(finite_vals)
            else:
                M_raw[col_nans, j] = 0.0

    if n_nan > 0:
        print(f"  [prepare] NaN runtime → médiane : {n_nan} valeurs imputées")

    # --- 4. Log-transform ---
    M_log, log_names = _log_transform(M_raw, ml_features, dynamic_threshold)
    if log_names:
        print(f"  [prepare] Log-transform : {len(log_names)}/{len(ml_features)} features")
    # M_raw est maintenant M_log (in-place)

    # --- 5. RobustScaler ---
    try:
        M_scaled = RobustScaler().fit_transform(M_log)
    except Exception as e:
        warnings.warn(f"[prepare] RobustScaler failed ({e}) — données log brutes")
        M_scaled = M_log
    del M_log  # Libérer l'intermédiaire

    # --- 6. Orthogonalisation ---
    if ortho_enabled:
        kept_idx, kept_names, exclusion_report = _select_orthogonal_features(
            M_scaled, ml_features,
            threshold=corr_threshold,
            protected_names=protected,
        )
        n_dropped = len(ml_features) - len(kept_names)
        if n_dropped > 0:
            print(f"  [prepare] Ortho ({corr_threshold}) : {len(kept_names)}/{len(ml_features)} "
                  f"({n_dropped} quasi-doublons écartés)")
        M_ortho = M_scaled[:, kept_idx]
        nan_mask_ortho = nan_mask[:, kept_idx]
    else:
        kept_names = list(ml_features)
        exclusion_report = {}
        M_ortho = M_scaled
        nan_mask_ortho = nan_mask
        print(f"  [prepare] Ortho : désactivé ({len(ml_features)} features)")

    del M_scaled  # Libérer l'intermédiaire

    # --- Meta ---
    meta = MatrixMeta(
        n_runs=M_ortho.shape[0],
        n_features_input=len(ml_features),
        n_features_ortho=len(kept_names),
        feat_names_input=ml_features,
        feat_names_ortho=kept_names,
        log_transformed=log_names,
        ortho_excluded=exclusion_report,
        n_nan_imputed=n_nan,
        nan_mask=nan_mask_ortho,
        ortho_threshold=corr_threshold if ortho_enabled else None,
    )

    return M_ortho, kept_names, meta
