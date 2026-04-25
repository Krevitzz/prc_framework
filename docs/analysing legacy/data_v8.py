"""
Chargement, filtrage et préparation des données analysing.

Parquet v8 → AnalysingData columnar → prepare_matrix (NaN→médiane + nan_mask).
Projection 2D (UMAP si disponible, t-SNE fallback).

@ROLE    Données : chargement → filtrage → préparation matrice ML → projection
@LAYER   analysing

"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import warnings

import numpy as np
import pyarrow.parquet as pq
from sklearn.preprocessing import RobustScaler

from utils.io_v8 import load_yaml, _META_COLS_V8


# =========================================================================
# ANALYSING DATA — conteneur pivot
# =========================================================================

@dataclass
class AnalysingData:
    M            : np.ndarray
    feat_names   : List[str]
    gamma_ids    : np.ndarray
    encoding_ids : np.ndarray
    modifier_ids : np.ndarray
    n_dofs       : np.ndarray
    rank_effs    : np.ndarray
    max_its      : np.ndarray
    run_statuses : np.ndarray
    phases       : np.ndarray
    seed_CIs     : np.ndarray
    seed_runs    : np.ndarray
    recording_modes : np.ndarray
    _n_cached : int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        if self.M is not None:
            self._n_cached = self.M.shape[0]

    @property
    def n(self):
        if self.M is not None:
            return self.M.shape[0]
        return self._n_cached

    @property
    def F(self):
        if self.M is not None:
            return self.M.shape[1]
        return 0

    def features_for_ml(self):
        """Retourne (M, feat_names) sans colonnes health_*."""
        mask = [not f.startswith('health_') for f in self.feat_names]
        feat_names = [f for f, m in zip(self.feat_names, mask) if m]
        return self.M[:, mask], feat_names


# =========================================================================
# CONSTRUCTION DEPUIS PYARROW
# =========================================================================

def _df_to_analysing_data(df, feature_cols):
    cols = df.to_pydict()
    n = len(df)
    F = len(feature_cols)

    M = np.empty((n, F), dtype=np.float32)
    for j, col_name in enumerate(feature_cols):
        col = np.array(cols[col_name], dtype=np.float32)
        col[~np.isfinite(col)] = np.nan
        M[:, j] = col

    def _arr(key, dtype):
        c = cols.get(key)
        if c is None:
            return np.full(n, -1 if dtype in (np.int32, np.int64) else '', dtype=dtype)
        if dtype in (np.int32, np.int64):
            c = [v if v is not None else -1 for v in c]
        return np.array(c, dtype=dtype)

    return AnalysingData(
        M=M, feat_names=list(feature_cols),
        gamma_ids=_arr('gamma_id', object),
        encoding_ids=_arr('encoding_id', object),
        modifier_ids=_arr('modifier_id', object),
        n_dofs=_arr('n_dof', np.int32),
        rank_effs=_arr('rank_eff', np.int32),
        max_its=_arr('max_it', np.int32),
        run_statuses=_arr('run_status', object),
        phases=_arr('phase', object),
        seed_CIs=_arr('seed_CI', np.int64),
        seed_runs=_arr('seed_run', np.int64),
        recording_modes=_arr('recording_mode', object),
    )


# =========================================================================
# FILTRES
# =========================================================================

def build_pyarrow_filters(scope):
    if not scope:
        return []
    filters = []
    for key, col in [('run_status', 'run_status'), ('n_dof', 'n_dof'),
                      ('rank_eff', 'rank_eff'), ('modifiers', 'modifier_id')]:
        val = scope.get(key)
        if val and val != 'all':
            vals = val if isinstance(val, list) else [val]
            filters.append((col, 'in', vals))
    return filters


def _mask_seeds_one(data):
    keys = np.array(
        [f'{g}||{e}' for g, e in zip(data.gamma_ids, data.encoding_ids)],
        dtype=object,
    )
    _, first_idx = np.unique(keys, return_index=True)
    mask = np.zeros(data.n, dtype=bool)
    mask[first_idx] = True
    return mask


def _apply_mask(data, mask):
    return AnalysingData(
        M=data.M[mask], feat_names=data.feat_names,
        gamma_ids=data.gamma_ids[mask], encoding_ids=data.encoding_ids[mask],
        modifier_ids=data.modifier_ids[mask], n_dofs=data.n_dofs[mask],
        rank_effs=data.rank_effs[mask], max_its=data.max_its[mask],
        run_statuses=data.run_statuses[mask], phases=data.phases[mask],
        seed_CIs=data.seed_CIs[mask], seed_runs=data.seed_runs[mask],
        recording_modes=data.recording_modes[mask],
    )


def load_pool_requirements(path=None):
    candidates = [path, Path(__file__).parent.parent / 'configs' / 'pool_requirements.yaml']
    for p in candidates:
        if p is not None and Path(p).exists():
            return load_yaml(Path(p))
    return {'n_dof': {'min': None, 'max': None}, 'deprecated': {}}


def _mask_pool_requirements(data, req):
    mask = np.ones(data.n, dtype=bool)
    n_dof_cfg = req.get('n_dof', {})
    deprecated = req.get('deprecated', {})
    if n_dof_cfg.get('min') is not None:
        mask &= data.n_dofs >= n_dof_cfg['min']
    if n_dof_cfg.get('max') is not None:
        mask &= data.n_dofs <= n_dof_cfg['max']
    for arr, key in [(data.gamma_ids, 'gammas'), (data.encoding_ids, 'encodings'),
                      (data.modifier_ids, 'modifiers')]:
        dep = list(deprecated.get(key, []))
        if dep:
            mask &= ~np.isin(arr, dep)
    return mask


# =========================================================================
# POINT D'ENTRÉE CHARGEMENT
# =========================================================================

def load_analysing_data(parquet_path, scope=None, apply_pool=False,
                        pool_path=None, verbose=True):
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    filters = build_pyarrow_filters(scope or {})
    if verbose:
        print(f"\n[data_v8] Lecture : {parquet_path.name}")

    df = pq.read_table(str(parquet_path), filters=filters if filters else None)
    all_cols = df.schema.names
    feature_cols = [c for c in all_cols if c not in _META_COLS_V8 and c != 'iteration']

    if verbose:
        print(f"  Rows après pushdown : {len(df)}")

    data = _df_to_analysing_data(df, feature_cols)
    del df

    if scope and scope.get('seeds') == 'one':
        n_before = data.n
        data = _apply_mask(data, _mask_seeds_one(data))
        if verbose:
            print(f"  seeds: one → {data.n}/{n_before}")

    if apply_pool:
        req = load_pool_requirements(pool_path)
        n_before = data.n
        data = _apply_mask(data, _mask_pool_requirements(data, req))
        if verbose and data.n < n_before:
            print(f"  pool_requirements → {data.n}/{n_before}")

    if verbose:
        n_ok = int(np.sum(data.run_statuses == 'OK'))
        n_exp = int(np.sum(data.run_statuses == 'EXPLOSION'))
        print(f"  Total : {data.n} rows (OK={n_ok}, EXP={n_exp})")

    return data


# =========================================================================
# PREPARE MATRIX — NaN→médiane + nan_mask + log + scale + ortho
# =========================================================================

DYNAMIC_THRESHOLD = 1e6
CORRELATION_THRESHOLD_DEFAULT = 0.98

PROTECTED_FEATURES_V8: Set[str] = {
    'ps_norm_ratio',
    'f2_von_neumann_entropy_delta',
    'f1_effective_rank_delta',
    'ps_condition_ratio',
    'f1_nuclear_frobenius_ratio_final',
    'f1_condition_number_mean',
    'f4_lyapunov_empirical_mean',
    'f7_dmd_spectral_radius',
}


def _log_transform(matrix):
    result = matrix.copy()
    transformed = []
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        abs_col = np.abs(col)
        nz = abs_col[abs_col > 0]
        col_min = np.min(nz) if len(nz) > 0 else 1.0
        dynamic = np.max(abs_col) / (col_min + 1e-10)
        if dynamic > DYNAMIC_THRESHOLD:
            result[:, j] = np.sign(col) * np.log1p(abs_col)
            transformed.append(True)
        else:
            transformed.append(False)
    return result, transformed


def _select_orthogonal_features(matrix_scaled, feature_names, threshold=0.98,
                                 protected_names=None):
    n_features = matrix_scaled.shape[1]
    if n_features < 2:
        return list(range(n_features)), list(feature_names), {}

    if protected_names is None:
        protected_names = PROTECTED_FEATURES_V8

    protected_idx = [i for i, k in enumerate(feature_names) if k in protected_names]
    unprotected_idx = [i for i, k in enumerate(feature_names) if k not in protected_names]
    variances = np.var(matrix_scaled, axis=0)
    order_unprot = sorted(unprotected_idx, key=lambda i: variances[i], reverse=True)

    with np.errstate(invalid='ignore'):
        corr_matrix = np.abs(np.corrcoef(matrix_scaled.T))
    corr_matrix = np.nan_to_num(corr_matrix, nan=1.0)

    kept_indices = []
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


def prepare_matrix(data, cfg=None):
    """
    NaN→médiane + log_transform + RobustScaler + orthogonalisation.
    Retourne (M_ortho, feat_names, nan_mask, matrix_meta).
    """
    cfg = cfg or {}
    ortho_cfg = cfg.get('orthogonalization', {})
    corr_threshold = ortho_cfg.get('correlation_threshold', CORRELATION_THRESHOLD_DEFAULT)
    ortho_enabled = ortho_cfg.get('enabled', True)

    M_ml, feat_names = data.features_for_ml()
    M_raw = M_ml.astype(np.float64)

    # NaN mask — conservé pour le namer
    nan_mask = np.isnan(M_raw)
    n_nan = int(np.sum(nan_mask))

    # NaN → médiane colonne
    for j in range(M_raw.shape[1]):
        col_nans = nan_mask[:, j]
        if np.any(col_nans):
            finite_vals = M_raw[~col_nans, j]
            if len(finite_vals) > 0:
                M_raw[col_nans, j] = np.median(finite_vals)
            else:
                M_raw[col_nans, j] = 0.0

    if n_nan > 0:
        print(f"  NaN → médiane : {n_nan} valeurs imputées (sentinel neutre)")

    M_log, transformed_cols = _log_transform(M_raw)
    if sum(transformed_cols) > 0:
        print(f"  Log-transform : {sum(transformed_cols)}/{len(feat_names)} features")

    try:
        M_scaled = RobustScaler().fit_transform(M_log)
    except Exception as e:
        warnings.warn(f"RobustScaler failed ({e}) — données brutes")
        M_scaled = M_log

    if ortho_enabled:
        kept_idx, kept_names, exclusion_report = _select_orthogonal_features(
            M_scaled, feat_names, threshold=corr_threshold,
            protected_names=PROTECTED_FEATURES_V8,
        )
        n_dropped = len(feat_names) - len(kept_names)
        if n_dropped > 0:
            print(f"  Ortho ({corr_threshold}) : {len(kept_names)}/{len(feat_names)} "
                  f"({n_dropped} quasi-doublons écartés)")
        M_ortho = M_scaled[:, kept_idx]
        nan_mask_ortho = nan_mask[:, [feat_names.index(k) for k in kept_names]]
    else:
        kept_names = list(feat_names)
        exclusion_report = {}
        M_ortho = M_scaled
        nan_mask_ortho = nan_mask
        print(f"  Ortho : désactivé ({len(feat_names)} features)")

    return M_ortho, kept_names, nan_mask_ortho, {
        'transformed_cols': transformed_cols,
        'exclusion_report': exclusion_report,
        'n_features_total': len(feat_names),
        'n_features_ortho': len(kept_names),
        'n_nan_imputed': n_nan,
        'ortho_threshold': corr_threshold if ortho_enabled else None,
    }


# =========================================================================
# PROJECTION 2D — t-SNE / UMAP
# =========================================================================

def compute_projection(M_ortho, cfg=None, cache_path=None):
    import os
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    try:
        import umap
        _umap = True
    except ImportError:
        _umap = False

    cfg = cfg or {}
    if cache_path and os.path.exists(cache_path):
        return np.load(cache_path)

    n = M_ortho.shape[0]
    max_proj = cfg.get('max_projection_samples', 5000)
    umap_threshold = cfg.get('umap_threshold', 50000)

    # Subsample if too large for t-SNE (memory-safe)
    if n > max_proj:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_proj, replace=False)
        M_proj = M_ortho[idx]
        print(f"  Projection subsampled : {n} → {max_proj}")
    else:
        idx = None
        M_proj = M_ortho

    n_proj = M_proj.shape[0]
    n_pca = min(50, M_proj.shape[1], n_proj - 1)
    if n_pca < 2:
        M_pca = M_proj[:, :2] if M_proj.shape[1] >= 2 else np.column_stack([M_proj[:, 0], np.zeros(n_proj)])
    else:
        M_pca = PCA(n_components=n_pca, random_state=42).fit_transform(M_proj)

    if n_proj >= umap_threshold and _umap:
        print(f"  UMAP 2D (n={n_proj})...")
        M_2d_sub = umap.UMAP(n_components=2, random_state=42).fit_transform(M_pca)
    else:
        perplexity = min(50, max(5, int(np.sqrt(n_proj)) // 3))
        print(f"  t-SNE 2D (n={n_proj}, perplexity={perplexity})...")
        M_2d_sub = TSNE(n_components=2, perplexity=perplexity,
                        max_iter=1000, random_state=42).fit_transform(M_pca)

    # If subsampled, map back to full array (non-sampled → NaN)
    if idx is not None:
        M_2d = np.full((n, 2), np.nan)
        M_2d[idx] = M_2d_sub
    else:
        M_2d = M_2d_sub

    if cache_path:
        np.save(cache_path, M_2d)
    return M_2d
