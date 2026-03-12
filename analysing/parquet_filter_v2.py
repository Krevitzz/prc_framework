"""
analysing/parquet_filter_v2.py

Responsabilité : Chargement ciblé parquet → AnalysingData.

Flux :
    parquet (disque)
      → pushdown pyarrow (run_status, n_dof, rank_eff, modifier_id)
      → AnalysingData (columnar — une seule matrice numpy)
      → masque seeds:one (si demandé)
      → masque pool_requirements (si apply: true)
      → _apply_mask → AnalysingData filtrée

Format pivot : AnalysingData — remplace List[Dict].
  M (n, F) float32 + arrays meta numpy — jamais de dicts par row.
  Zéro reconstruction de matrice dans les modules aval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pyarrow.parquet as pq

from utils.data_loading_new import load_yaml


# =============================================================================
# ANALYSING DATA — conteneur pivot
# =============================================================================

@dataclass
class AnalysingData:
    """
    Conteneur columnar unique pour le pipeline analysing.

    M[i, j]        : feature j du run i — float32, inf→nan à la construction
    feat_names[j]  : nom de la feature j
    *[i]           : métadonnée du run i — arrays numpy longueur n

    Invariant : toutes les arrays ont longueur n = M.shape[0].
    """
    M            : np.ndarray   # (n, F) float32
    feat_names   : List[str]    # longueur F

    gamma_ids    : np.ndarray   # (n,) object
    encoding_ids : np.ndarray   # (n,) object
    modifier_ids : np.ndarray   # (n,) object
    n_dofs       : np.ndarray   # (n,) int32
    rank_effs    : np.ndarray   # (n,) int32
    max_its      : np.ndarray   # (n,) int32
    run_statuses : np.ndarray   # (n,) object
    phases       : np.ndarray   # (n,) object
    seed_CIs     : np.ndarray   # (n,) int64
    seed_runs    : np.ndarray   # (n,) int64

    @property
    def n(self) -> int:
        return self.M.shape[0]

    @property
    def F(self) -> int:
        return self.M.shape[1]

    def features_for_ml(self):
        """
        Retourne (M, feat_names) sans colonnes health_*.

        Source unique — remplace les filtres dupliqués dans
        profiling_lite, outliers_lite, clustering_lite.

        Returns:
            (np.ndarray (n, F_ml), List[str])
        """
        mask       = [not f.startswith('health_') for f in self.feat_names]
        feat_names = [f for f, m in zip(self.feat_names, mask) if m]
        return self.M[:, mask], feat_names


# =============================================================================
# CONSTANTES
# =============================================================================

META_COLS_V7 = {
    'run_status', 'phase',
    'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'rank_eff', 'max_it',
    'gamma_params', 'encoding_params', 'modifier_params',
    'seed_CI', 'seed_run',
}


# =============================================================================
# POOL REQUIREMENTS
# =============================================================================

def load_pool_requirements(path: Optional[Path] = None) -> Dict:
    """Charge pool_requirements.yaml via load_yaml."""
    candidates = [path, Path('configs/pool_requirements.yaml')]
    for p in candidates:
        if p is not None and Path(p).exists():
            return load_yaml(Path(p))
    return {
        'n_dof'     : {'min': None, 'max': None},
        'deprecated': {'gammas': [], 'encodings': [], 'modifiers': []},
    }


# =============================================================================
# FILTRES PYARROW PUSHDOWN
# =============================================================================

def build_pyarrow_filters(scope: Dict) -> List:
    if not scope:
        return []

    filters = []

    statuses = scope.get('run_status')
    if statuses and statuses != 'all':
        vals = statuses if isinstance(statuses, list) else [statuses]
        filters.append(('run_status', 'in', vals))

    n_dof = scope.get('n_dof')
    if n_dof and n_dof != 'all':
        vals = n_dof if isinstance(n_dof, list) else [int(n_dof)]
        filters.append(('n_dof', 'in', vals))

    rank_eff = scope.get('rank_eff')
    if rank_eff and rank_eff != 'all':
        vals = rank_eff if isinstance(rank_eff, list) else [int(rank_eff)]
        filters.append(('rank_eff', 'in', vals))

    modifiers = scope.get('modifiers')
    if modifiers and modifiers != 'all':
        vals = modifiers if isinstance(modifiers, list) else [modifiers]
        filters.append(('modifier_id', 'in', vals))

    return filters


# =============================================================================
# CONSTRUCTION AnalysingData DEPUIS PYARROW
# =============================================================================

def _df_to_analysing_data(df, feature_cols: List[str]) -> AnalysingData:
    """
    Construit AnalysingData directement depuis colonnes pyarrow.

    Opérations vectorisées colonne par colonne — pas de boucle row par row.
    inf → nan (sklearn et HDBSCAN n'acceptent pas inf).
    """
    cols = df.to_pydict()
    n    = len(df)
    F    = len(feature_cols)

    M = np.empty((n, F), dtype=np.float32)
    for j, col_name in enumerate(feature_cols):
        col = np.array(cols[col_name], dtype=np.float32)
        col[~np.isfinite(col)] = np.nan
        M[:, j] = col

    def _arr(key, dtype):
        col = cols[key]
        if dtype in (np.int32, np.int64):
            # seed_CI / seed_run peuvent être None si non définis
            col = [v if v is not None else -1 for v in col]
        return np.array(col, dtype=dtype)

    return AnalysingData(
        M            = M,
        feat_names   = list(feature_cols),
        gamma_ids    = _arr('gamma_id',    object),
        encoding_ids = _arr('encoding_id', object),
        modifier_ids = _arr('modifier_id', object),
        n_dofs       = _arr('n_dof',       np.int32),
        rank_effs    = _arr('rank_eff',    np.int32),
        max_its      = _arr('max_it',      np.int32),
        run_statuses = _arr('run_status',  object),
        phases       = _arr('phase',       object),
        seed_CIs     = _arr('seed_CI',     np.int64),
        seed_runs    = _arr('seed_run',    np.int64),
    )


# =============================================================================
# MASQUES POST-FILTRES
# =============================================================================

def _mask_seeds_one(data: AnalysingData) -> np.ndarray:
    """
    Masque : un seul sample par (gamma_id × encoding_id), premier vu.

    Vectorisé via np.unique — O(n log n) au lieu de boucle Python O(n).
    np.unique retourne les indices dans l'ordre de première occurrence.
    """
    keys = np.array(
        [f'{g}||{e}' for g, e in zip(data.gamma_ids, data.encoding_ids)],
        dtype=object,
    )
    _, first_idx = np.unique(keys, return_index=True)
    mask         = np.zeros(data.n, dtype=bool)
    mask[first_idx] = True
    return mask


def _mask_pool_requirements(data: AnalysingData, req: Dict) -> np.ndarray:
    """Masque : filtre n_dof min/max + atomics deprecated."""
    mask       = np.ones(data.n, dtype=bool)
    n_dof_cfg  = req.get('n_dof', {})
    deprecated = req.get('deprecated', {})
    dep_gammas = list(deprecated.get('gammas',    []))
    dep_encs   = list(deprecated.get('encodings', []))
    dep_mods   = list(deprecated.get('modifiers', []))

    if n_dof_cfg.get('min') is not None:
        mask &= data.n_dofs >= n_dof_cfg['min']
    if n_dof_cfg.get('max') is not None:
        mask &= data.n_dofs <= n_dof_cfg['max']
    if dep_gammas:
        mask &= ~np.isin(data.gamma_ids,    dep_gammas)
    if dep_encs:
        mask &= ~np.isin(data.encoding_ids, dep_encs)
    if dep_mods:
        mask &= ~np.isin(data.modifier_ids, dep_mods)

    return mask


def _apply_mask(data: AnalysingData, mask: np.ndarray) -> AnalysingData:
    """Retourne un AnalysingData filtré — slices numpy sans copie des données."""
    return AnalysingData(
        M            = data.M[mask],
        feat_names   = data.feat_names,
        gamma_ids    = data.gamma_ids[mask],
        encoding_ids = data.encoding_ids[mask],
        modifier_ids = data.modifier_ids[mask],
        n_dofs       = data.n_dofs[mask],
        rank_effs    = data.rank_effs[mask],
        max_its      = data.max_its[mask],
        run_statuses = data.run_statuses[mask],
        phases       = data.phases[mask],
        seed_CIs     = data.seed_CIs[mask],
        seed_runs    = data.seed_runs[mask],
    )


# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

def load_analysing_data(
    parquet_path : Path,
    scope        : Optional[Dict] = None,
    apply_pool   : bool = False,
    pool_path    : Optional[Path] = None,
    verbose      : bool = True,
) -> AnalysingData:
    """
    Charge les données parquet v7 → AnalysingData.

    Flux :
        1. pushdown pyarrow
        2. _df_to_analysing_data → AnalysingData columnar (del df immédiat)
        3. _mask_seeds_one si scope['seeds'] == 'one'
        4. _mask_pool_requirements si apply_pool
        5. _apply_mask → AnalysingData filtrée

    Returns:
        AnalysingData — M (n, F) float32 + arrays meta numpy
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    filters = build_pyarrow_filters(scope or {})

    if verbose:
        print(f"\n[parquet_filter] Lecture : {parquet_path.name}")
        if filters:
            print(f"  Pushdown filters : {filters}")

    df           = pq.read_table(str(parquet_path), filters=filters if filters else None)
    all_cols     = df.schema.names
    feature_cols = [c for c in all_cols if c not in META_COLS_V7]

    if verbose:
        print(f"  Rows après pushdown : {len(df)}")

    data = _df_to_analysing_data(df, feature_cols)
    del df  # libérer la table pyarrow immédiatement

    if scope and scope.get('seeds') == 'one':
        n_before = data.n
        data     = _apply_mask(data, _mask_seeds_one(data))
        if verbose:
            print(f"  seeds: one → {data.n}/{n_before} rows conservées")

    if apply_pool:
        req      = load_pool_requirements(pool_path)
        n_before = data.n
        data     = _apply_mask(data, _mask_pool_requirements(data, req))
        if verbose and data.n < n_before:
            print(f"  [pool_requirements] {n_before - data.n}/{n_before} rows filtrées")

    if verbose:
        n_ok  = int(np.sum(data.run_statuses == 'OK'))
        n_exp = int(np.sum(data.run_statuses == 'EXPLOSION'))
        print(f"  Total final : {data.n} rows (OK={n_ok}, EXPLOSION={n_exp})")

    return data