"""
Fusion multi-parquets et filtrage pool.

Lit N parquets, filtre les atomics exclus (YAML pool), déduplique,
fusionne en un seul pa.Table. Si un seul parquet sans pool → pass-through.

@ROLE    Data preparation : N parquets + YAML pool → 1 pa.Table unifié
@LAYER   analysing

@EXPORTS
  load_pool_config(yaml_path)         → Dict      | charge le YAML pool
  scan_parquets(results_dir)          → List[Path] | liste les parquets disponibles
  merge_parquets(paths, pool_config)  → pa.Table   | fusion filtrée et dédupliquée

@LIFECYCLE
  CREATES  pa.Table   table fusionnée, passée au hub → load_analysing_data
  RECEIVES parquets   depuis le disque
  PASSES   pa.Table   vers data.load_analysing_data

@CONFORMITY
  OK   Le pool ne juge pas les atomics — applique le YAML (point 1 validé)
  OK   Déduplication sur clé composite complète
  OK   Aucune constante hardcodée
"""

from pathlib import Path
from typing import Dict, List, Optional
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

from utils.io_v8 import load_yaml


# =========================================================================
# CONFIGURATION POOL
# =========================================================================

def load_pool_config(yaml_path: Optional[Path] = None) -> Dict:
    """Charge le YAML pool : atomics exclus, filtres.

    Structure attendue :
        excluded:
            gammas: [gamma_id_1, ...]
            encodings: [encoding_id_1, ...]
            modifiers: [modifier_id_1, ...]
        filters:
            n_dof: {min: N, max: M}    # optionnel

    Si yaml_path est None ou le fichier n'existe pas, retourne un config vide
    (aucun filtrage).
    """
    if yaml_path is None:
        return {'excluded': {}, 'filters': {}}

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        warnings.warn(f"[pool] YAML pool introuvable : {yaml_path}")
        return {'excluded': {}, 'filters': {}}

    cfg = load_yaml(yaml_path)
    return {
        'excluded': cfg.get('excluded', cfg.get('deprecated', {})),
        'filters': cfg.get('filters', {}),
    }


# =========================================================================
# SCAN PARQUETS
# =========================================================================

def scan_parquets(results_dir: Optional[Path] = None) -> List[Path]:
    """Liste les parquets disponibles dans le répertoire résultats.

    Retourne les chemins triés par nom.
    """
    if results_dir is None:
        results_dir = Path('data/results')

    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    return sorted(results_dir.glob('*.parquet'))


# =========================================================================
# FILTRAGE ATOMICS EXCLUS
# =========================================================================

def _filter_excluded(table: pa.Table, excluded: Dict) -> pa.Table:
    """Filtre les runs dont les atomics sont dans la liste d'exclusion.

    Le layer analysing ne décide pas quels atomics exclure.
    Il applique le YAML pool — la décision est prise en amont.
    """
    n_before = table.num_rows

    for col_name, key in [('gamma_id', 'gammas'),
                           ('encoding_id', 'encodings'),
                           ('modifier_id', 'modifiers')]:
        excluded_ids = excluded.get(key, [])
        if not excluded_ids or col_name not in table.schema.names:
            continue
        mask = pc.invert(pc.is_in(table.column(col_name),
                                   value_set=pa.array(excluded_ids, type=pa.string())))
        table = table.filter(mask)

    n_after = table.num_rows
    if n_after < n_before:
        print(f"  [pool] Exclusion atomics : {n_before} → {n_after} "
              f"({n_before - n_after} runs filtrés)")

    return table


# =========================================================================
# FILTRAGE N_DOF
# =========================================================================

def _filter_n_dof(table: pa.Table, filters: Dict) -> pa.Table:
    """Filtre par n_dof si spécifié dans le pool config."""
    n_dof_cfg = filters.get('n_dof', {})
    if not n_dof_cfg or 'n_dof' not in table.schema.names:
        return table

    n_before = table.num_rows
    n_dof_min = n_dof_cfg.get('min')
    n_dof_max = n_dof_cfg.get('max')

    if n_dof_min is not None:
        table = table.filter(pc.greater_equal(table.column('n_dof'), n_dof_min))
    if n_dof_max is not None:
        table = table.filter(pc.less_equal(table.column('n_dof'), n_dof_max))

    n_after = table.num_rows
    if n_after < n_before:
        print(f"  [pool] Filtre n_dof : {n_before} → {n_after}")

    return table


# =========================================================================
# DÉDUPLICATION
# =========================================================================

# Clé de déduplication : identifie un run unique.
DEDUP_KEY_COLUMNS = [
    'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'max_it', 'seed_CI', 'seed_run',
]


def _deduplicate(table: pa.Table) -> pa.Table:
    """Déduplique sur la clé composite.

    Si deux runs ont la même clé (même gamma, encoding, modifier, dof, max_it, seeds),
    on garde le premier. Cela arrive quand deux parquets couvrent des plages
    qui se chevauchent.
    """
    # Vérifier que toutes les colonnes de clé existent
    available = set(table.schema.names)
    key_cols = [c for c in DEDUP_KEY_COLUMNS if c in available]

    if len(key_cols) < len(DEDUP_KEY_COLUMNS):
        missing = set(DEDUP_KEY_COLUMNS) - available
        warnings.warn(f"[pool] Colonnes de dédup manquantes : {missing}. "
                      f"Déduplication partielle.")

    if not key_cols:
        return table

    n_before = table.num_rows

    # Construire une clé composite string pour chaque row
    # PyArrow n'a pas de group_by + first natif → on passe par numpy
    import numpy as np

    key_arrays = []
    for col in key_cols:
        arr = table.column(col)
        key_arrays.append(pc.cast(arr, pa.string()).to_pylist())

    n = table.num_rows
    keys = ['||'.join(str(key_arrays[j][i]) for j in range(len(key_cols)))
            for i in range(n)]

    keys_np = np.array(keys)
    _, first_indices = np.unique(keys_np, return_index=True)
    first_indices = np.sort(first_indices)

    if len(first_indices) < n_before:
        table = table.take(first_indices.tolist())
        print(f"  [pool] Déduplication : {n_before} → {table.num_rows} "
              f"({n_before - table.num_rows} doublons)")

    return table


# =========================================================================
# FUSION MULTI-PARQUETS
# =========================================================================

def merge_parquets(parquet_paths: List[Path],
                    pool_config: Optional[Dict] = None,
                    verbose: bool = True) -> pa.Table:
    """Fusionne N parquets avec filtrage et déduplication.

    1. Lecture séquentielle (PyArrow)
    2. Validation schéma compatible
    3. Concaténation
    4. Filtrage atomics exclus (depuis pool_config)
    5. Filtrage n_dof (depuis pool_config)
    6. Déduplication sur clé composite

    Args:
        parquet_paths : chemins vers les parquets.
        pool_config : config pool (excluded, filters). None = pas de filtrage.
        verbose : afficher les stats.

    Returns:
        pa.Table unifié, filtré, dédupliqué.
    """
    if not parquet_paths:
        raise ValueError("[pool] Aucun parquet fourni")

    if verbose:
        print(f"\n[pool] Fusion de {len(parquet_paths)} parquet(s)")

    tables = []
    reference_schema = None

    for path in parquet_paths:
        path = Path(path)
        if not path.exists():
            warnings.warn(f"[pool] Parquet introuvable, ignoré : {path}")
            continue

        t = pq.read_table(str(path))

        if verbose:
            print(f"  {path.name} : {t.num_rows} rows, {t.num_columns} cols")

        if reference_schema is None:
            reference_schema = t.schema
        else:
            # Vérification schéma compatible (mêmes colonnes, même ordre pas requis)
            ref_names = set(reference_schema.names)
            cur_names = set(t.schema.names)
            if ref_names != cur_names:
                only_ref = ref_names - cur_names
                only_cur = cur_names - ref_names
                if only_ref:
                    warnings.warn(f"[pool] {path.name} manque : {only_ref}")
                if only_cur:
                    warnings.warn(f"[pool] {path.name} a en plus : {only_cur}")
                # Harmoniser : garder l'intersection
                common = sorted(ref_names & cur_names)
                t = t.select(common)

        tables.append(t)

    if not tables:
        raise ValueError("[pool] Aucun parquet lisible")

    # Harmoniser les colonnes si nécessaire
    if len(tables) > 1:
        common_cols = set(tables[0].schema.names)
        for t in tables[1:]:
            common_cols &= set(t.schema.names)
        common_cols = sorted(common_cols)
        tables = [t.select(common_cols) for t in tables]

    # Concaténation
    merged = pa.concat_tables(tables, promote_options='default')
    del tables

    if verbose:
        print(f"  Concaténé : {merged.num_rows} rows")

    # Filtrage
    cfg = pool_config or {'excluded': {}, 'filters': {}}

    excluded = cfg.get('excluded', {})
    if excluded:
        merged = _filter_excluded(merged, excluded)

    filters = cfg.get('filters', {})
    if filters:
        merged = _filter_n_dof(merged, filters)

    # Déduplication
    merged = _deduplicate(merged)

    if verbose:
        print(f"  Final : {merged.num_rows} rows")

    return merged
