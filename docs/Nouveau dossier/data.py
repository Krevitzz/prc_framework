"""
Chargement structuré et conteneur pivot des données analysing.

Parquet → classification colonnes (registre) → AnalysingData lazy.
La matérialisation numpy se fait à la demande via materialize_features()
et materialize_timelines(). Aucune matrice lourde en RAM au chargement.

@ROLE    Conteneur pivot + chargement structuré parquet → AnalysingData lazy
@LAYER   analysing

@EXPORTS
  AnalysingData                          → dataclass | conteneur pivot lazy
  classify_columns(schema)               → Dict      | typage colonnes via registre
  load_analysing_data(source, scope)     → AnalysingData | point d'entrée chargement

@LIFECYCLE
  CREATES  AnalysingData    contient refs PyArrow (lazy), metadata numpy (léger)
  RECEIVES pa.Table         depuis pool.merge_parquets ou lecture directe
  PASSES   AnalysingData    vers hub → tous les modules

@CONFORMITY
  OK   Registre = source de vérité pour typage colonnes (P4)
  OK   Colonnes non-registrées chargées et documentées, jamais ignorées (point 2 validé)
  OK   Matérialisation lazy — pas de matrice complète au chargement (point 4 validé)
  OK   Aucune constante hardcodée
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from running.features_registry import (
    FEATURE_NAMES,
    METADATA_COLUMNS,
    TIMELINE_COLUMNS,
)


# =========================================================================
# CLASSIFICATION DES COLONNES
# =========================================================================

def classify_columns(parquet_schema: pa.Schema) -> Dict[str, List[str]]:
    """Classe chaque colonne du parquet via le registre features_registry.

    Retourne :
        {
            'metadata':     [...],   # dans METADATA_COLUMNS
            'features':     [...],   # dans FEATURE_NAMES
            'timelines':    [...],   # dans TIMELINE_COLUMNS
            'unregistered': [...],   # présentes dans parquet, absentes du registre
        }

    Si unregistered non vide → warning documenté.
    Aucune colonne n'est ignorée — les non-registrées sont chargées et tracées.
    """
    all_cols = set(parquet_schema.names)

    metadata_set = set(METADATA_COLUMNS.keys())
    features_set = set(FEATURE_NAMES)
    timelines_set = set(TIMELINE_COLUMNS)

    classified = {
        'metadata':     [c for c in parquet_schema.names if c in metadata_set],
        'features':     [c for c in parquet_schema.names if c in features_set],
        'timelines':    [c for c in parquet_schema.names if c in timelines_set],
        'unregistered': [],
    }

    known = metadata_set | features_set | timelines_set
    unregistered = [c for c in parquet_schema.names if c not in known]
    classified['unregistered'] = unregistered

    if unregistered:
        warnings.warn(
            f"[data] {len(unregistered)} colonnes parquet non-registrées "
            f"(chargées mais non typées) : {unregistered}"
        )

    # Vérification inverse : features du registre absentes du parquet
    missing_features = features_set - all_cols
    if missing_features:
        warnings.warn(
            f"[data] {len(missing_features)} features du registre absentes du parquet : "
            f"{sorted(missing_features)}"
        )

    missing_meta = metadata_set - all_cols
    if missing_meta:
        warnings.warn(
            f"[data] {len(missing_meta)} metadata du registre absentes du parquet : "
            f"{sorted(missing_meta)}"
        )

    return classified


# =========================================================================
# ANALYSING DATA — conteneur pivot lazy
# =========================================================================

@dataclass
class AnalysingData:
    """Conteneur pivot — référence lazy aux données parquet.

    Metadata : arrays numpy légers (toujours en RAM).
    Features : référence PyArrow Table (matérialisée à la demande).
    Timelines : référence PyArrow Table (matérialisée à la demande).
    Unregistered : référence PyArrow Table (documentées, accessibles).

    Usage mémoire :
        Chargement → metadata numpy (~Ko) + refs PyArrow (~0)
        materialize_features() → numpy float32 (n × F) — appelant libère après usage
        materialize_timelines() → listes de arrays — appelant libère après usage
    """

    # --- Metadata (toujours en RAM, léger) ---
    n_runs:           int
    gamma_ids:        np.ndarray           # (n,) object
    encoding_ids:     np.ndarray           # (n,) object
    modifier_ids:     np.ndarray           # (n,) object
    n_dofs:           np.ndarray           # (n,) int32
    rank_effs:        np.ndarray           # (n,) int32
    max_its:          np.ndarray           # (n,) int32
    seed_CIs:         np.ndarray           # (n,) int64
    seed_runs:        np.ndarray           # (n,) int64
    run_statuses:     np.ndarray           # (n,) object
    p1_regime_class:  np.ndarray           # (n,) object
    phases:           np.ndarray           # (n,) object

    # --- Features scalaires (lazy) ---
    _features_table:  pa.Table
    feature_names:    List[str]

    # --- Timelines (lazy) ---
    _timelines_table: Optional[pa.Table]
    timeline_names:   List[str]

    # --- Colonnes non-registrées (lazy, documentées) ---
    _unregistered_table: Optional[pa.Table]
    unregistered_names:  List[str]

    def materialize_features(self,
                              columns: Optional[List[str]] = None,
                              rows: Optional[np.ndarray] = None) -> np.ndarray:
        """Matérialise un sous-ensemble de features en numpy float32.

        Args:
            columns : noms des features à extraire (None = toutes).
            rows : indices des runs à extraire (None = tous).

        Returns:
            np.ndarray (n_rows, len(columns)) float32.
            Les non-finis sont convertis en NaN.
        """
        cols = columns if columns is not None else self.feature_names

        # Extraction colonnes
        arrays = []
        for col_name in cols:
            if col_name not in self._features_table.schema.names:
                warnings.warn(f"[data] Feature '{col_name}' absente de la table")
                arr = np.full(self._features_table.num_rows, np.nan, dtype=np.float32)
            else:
                arr = self._features_table.column(col_name).to_numpy(
                    zero_copy_only=False
                ).astype(np.float32)
            arrays.append(arr)

        M = np.column_stack(arrays) if len(arrays) > 1 else arrays[0].reshape(-1, 1)

        # Filtrage lignes
        if rows is not None:
            M = M[rows]

        # Non-finis → NaN
        M[~np.isfinite(M)] = np.nan

        return M

    def materialize_timelines(self,
                               columns: Optional[List[str]] = None,
                               rows: Optional[np.ndarray] = None
                               ) -> Dict[str, List[np.ndarray]]:
        """Matérialise des timelines en listes de arrays numpy.

        Args:
            columns : noms des colonnes timeline (None = toutes).
            rows : indices des runs (None = tous).

        Returns:
            {col_name: [array_run_0, array_run_1, ...]}
            Les arrays peuvent avoir des longueurs différentes (max_it variable).
        """
        if self._timelines_table is None:
            return {}

        cols = columns if columns is not None else self.timeline_names
        result = {}

        for col_name in cols:
            if col_name not in self._timelines_table.schema.names:
                warnings.warn(f"[data] Timeline '{col_name}' absente de la table")
                continue

            col = self._timelines_table.column(col_name)
            n = col.length()

            if rows is not None:
                indices = rows
            else:
                indices = range(n)

            arrays = []
            for i in indices:
                element = col[int(i)]
                if element is None or element.as_py() is None:
                    arrays.append(np.array([], dtype=np.float32))
                else:
                    arrays.append(np.array(element.as_py(), dtype=np.float32))
            result[col_name] = arrays

        return result

    def subset(self, indices: np.ndarray) -> 'AnalysingData':
        """Crée un sous-ensemble léger (nouvelles refs PyArrow, metadata filtrée).

        Ne copie pas les données lourdes — crée des vues PyArrow.
        """
        indices_list = indices.tolist()

        features_sub = self._features_table.take(indices_list)
        timelines_sub = (self._timelines_table.take(indices_list)
                         if self._timelines_table is not None else None)
        unreg_sub = (self._unregistered_table.take(indices_list)
                     if self._unregistered_table is not None else None)

        return AnalysingData(
            n_runs=len(indices),
            gamma_ids=self.gamma_ids[indices],
            encoding_ids=self.encoding_ids[indices],
            modifier_ids=self.modifier_ids[indices],
            n_dofs=self.n_dofs[indices],
            rank_effs=self.rank_effs[indices],
            max_its=self.max_its[indices],
            seed_CIs=self.seed_CIs[indices],
            seed_runs=self.seed_runs[indices],
            run_statuses=self.run_statuses[indices],
            p1_regime_class=self.p1_regime_class[indices],
            phases=self.phases[indices],
            _features_table=features_sub,
            feature_names=self.feature_names,
            _timelines_table=timelines_sub,
            timeline_names=self.timeline_names,
            _unregistered_table=unreg_sub,
            unregistered_names=self.unregistered_names,
        )


# =========================================================================
# EXTRACTION METADATA NUMPY (léger)
# =========================================================================

def _extract_metadata_array(table: pa.Table, col_name: str,
                             dtype, default) -> np.ndarray:
    """Extrait une colonne metadata en numpy. Gère les absences et les None."""
    n = table.num_rows
    if col_name not in table.schema.names:
        return np.full(n, default, dtype=dtype)

    col = table.column(col_name)

    if dtype == object:
        # String columns — convert None to empty string
        result = np.empty(n, dtype=object)
        for i in range(n):
            val = col[i].as_py()
            result[i] = val if val is not None else ''
        return result
    else:
        arr = col.to_numpy(zero_copy_only=False)
        # Replace None with default for numeric
        if dtype in (np.int32, np.int64):
            mask = arr == None  # noqa: E711 — PyArrow can return None
            if hasattr(arr, 'filled'):
                arr = arr.filled(default)
            else:
                arr = np.where(mask, default, arr)
        return np.array(arr, dtype=dtype)


# =========================================================================
# FILTRES PYARROW (pushdown)
# =========================================================================

def _build_pyarrow_filter(scope: Dict) -> Optional[pc.Expression]:
    """Construit un filtre PyArrow depuis le scope config.

    Clés supportées :
        run_status : list[str] ou 'all'
        n_dof : list[int] ou 'all'
        rank_eff : list[int] ou 'all'
        modifiers : list[str] ou 'all'
    """
    if not scope:
        return None

    filters = []

    for key, col in [('run_status', 'run_status'), ('n_dof', 'n_dof'),
                      ('rank_eff', 'rank_eff'), ('modifiers', 'modifier_id')]:
        val = scope.get(key)
        if val is None or val == 'all':
            continue
        vals = val if isinstance(val, list) else [val]
        filters.append(pc.field(col).isin(vals))

    if not filters:
        return None

    combined = filters[0]
    for f in filters[1:]:
        combined = combined & f
    return combined


# =========================================================================
# POINT D'ENTRÉE CHARGEMENT
# =========================================================================

def load_analysing_data(source: Union[Path, pa.Table],
                         scope: Optional[Dict] = None,
                         verbose: bool = True) -> AnalysingData:
    """Point d'entrée chargement structuré.

    Args:
        source : chemin parquet OU pa.Table (sortie de merge_parquets).
        scope : filtres optionnels (run_status, n_dof, etc.)
        verbose : afficher les stats de chargement.

    Returns:
        AnalysingData avec références lazy.
    """
    # --- Lecture ---
    if isinstance(source, (str, Path)):
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Parquet introuvable : {source}")

        pf = pq.ParquetFile(str(source))
        pa_filter = _build_pyarrow_filter(scope or {})

        if pa_filter is not None:
            table = pq.read_table(str(source), filters=pa_filter)
        else:
            table = pf.read()

        if verbose:
            print(f"\n[data] Lecture : {source.name}")
            print(f"  {table.num_rows} rows après pushdown")
    else:
        # pa.Table directe (depuis pool)
        table = source
        if scope:
            pa_filter = _build_pyarrow_filter(scope)
            if pa_filter is not None:
                table = table.filter(pa_filter)
        if verbose:
            print(f"\n[data] Table directe : {table.num_rows} rows")

    # --- Classification colonnes ---
    classified = classify_columns(table.schema)

    if verbose:
        print(f"  Features : {len(classified['features'])}")
        print(f"  Timelines : {len(classified['timelines'])}")
        if classified['unregistered']:
            print(f"  Non-registrées : {classified['unregistered']}")

    # --- Extraction metadata (léger, numpy) ---
    n = table.num_rows

    gamma_ids = _extract_metadata_array(table, 'gamma_id', object, '')
    encoding_ids = _extract_metadata_array(table, 'encoding_id', object, '')
    modifier_ids = _extract_metadata_array(table, 'modifier_id', object, '')
    n_dofs = _extract_metadata_array(table, 'n_dof', np.int32, -1)
    rank_effs = _extract_metadata_array(table, 'rank_eff', np.int32, -1)
    max_its = _extract_metadata_array(table, 'max_it', np.int32, -1)
    seed_CIs = _extract_metadata_array(table, 'seed_CI', np.int64, -1)
    seed_runs = _extract_metadata_array(table, 'seed_run', np.int64, -1)
    run_statuses = _extract_metadata_array(table, 'run_status', object, '')
    p1_regime_class = _extract_metadata_array(table, 'p1_regime_class', object, '')
    phases = _extract_metadata_array(table, 'phase', object, '')

    # --- Séparation tables par type (refs légères) ---
    feat_cols = classified['features']
    features_table = table.select(feat_cols) if feat_cols else pa.table({})

    tl_cols = classified['timelines']
    timelines_table = table.select(tl_cols) if tl_cols else None

    unreg_cols = classified['unregistered']
    unreg_table = table.select(unreg_cols) if unreg_cols else None

    # Libérer la table complète — on a extrait ce qu'il faut
    del table

    if verbose:
        n_ok = int(np.sum(run_statuses == 'OK'))
        n_trunc = int(np.sum(run_statuses == 'OK_TRUNCATED'))
        n_exp = int(np.sum(run_statuses == 'EXPLOSION'))
        n_col = int(np.sum(run_statuses == 'COLLAPSED'))
        print(f"  Statuts : OK={n_ok} TRUNC={n_trunc} EXP={n_exp} COL={n_col}")

    return AnalysingData(
        n_runs=n,
        gamma_ids=gamma_ids,
        encoding_ids=encoding_ids,
        modifier_ids=modifier_ids,
        n_dofs=n_dofs,
        rank_effs=rank_effs,
        max_its=max_its,
        seed_CIs=seed_CIs,
        seed_runs=seed_runs,
        run_statuses=run_statuses,
        p1_regime_class=p1_regime_class,
        phases=phases,
        _features_table=features_table,
        feature_names=feat_cols,
        _timelines_table=timelines_table,
        timeline_names=tl_cols,
        _unregistered_table=unreg_table,
        unregistered_names=unreg_cols,
    )
