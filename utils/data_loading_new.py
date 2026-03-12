"""
utils/data_loading_new.py

Source unique de vérité pour le chargement de données PRC v7.

Responsabilités :
  - Discovery atomics JAX (gammas, encodings, modifiers)
  - Lecture YAML (path direct uniquement — pas de modes)
  - Écriture Parquet par append (schéma v7, pyarrow natif)

Différences vs data_loading_jax.py :
  1. prepare_params capturé dans _discover_from_dir (optionnel, None si absent)
  2. write_parquet (pandas) supprimé
     → remplacé par open_parquet_writer / write_rows_to_parquet / close_parquet_writer
  3. import pandas supprimé

Convention callable :
  - Gammas    : apply(state, params, key)
  - Encodings : create(n_dof, params, key)
  - Modifiers : apply(state, params, key)
  - prepare_params (optionnel) : prepare_params(params, key) → params enrichis

Usage : toujours lancer depuis prc_framework/ (python -m ...)
"""

import importlib
import warnings
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CriticalDiscoveryError(Exception):
    """METADATA ou callable obligatoire manquant — arrêt obligatoire."""
    pass


class ValidationError(Exception):
    """Structure atomique invalide."""
    pass


# =============================================================================
# SECTION 1 — DISCOVERY ATOMICS
# =============================================================================

def _discover_from_dir(
    directory    : Path,
    pattern      : Union[str, List[str]],
    module_prefix: str,
    callable_name: str,
) -> Dict[str, Dict[str, Any]]:
    """
    Moteur commun discovery.

    Args:
        directory     : Dossier à scanner
        pattern       : Glob pattern(s) — str ou liste de str
        module_prefix : Préfixe import (ex: 'atomics.operators')
        callable_name : Nom du callable attendu ('apply' ou 'create')

    Règles :
        - Skip *_deprecated_* et __init__
        - METADATA absent → CriticalDiscoveryError
        - METADATA['id'] absent → CriticalDiscoveryError
        - callable_name absent → warning + skip
        - prepare_params : capturé si présent, None sinon (optionnel)

    Returns:
        dict {id → {'callable': fn, 'metadata': dict, 'prepare_params': fn|None}}
    """
    if not directory.exists():
        raise FileNotFoundError(
            f"Dossier discovery introuvable : {directory}"
        )

    patterns = [pattern] if isinstance(pattern, str) else pattern
    files = []
    for p in patterns:
        files.extend(directory.glob(p))

    registry = {}

    for filepath in sorted(files):
        stem = filepath.stem

        # Skip deprecated et __init__
        if '_deprecated_' in stem or stem == '__init__':
            continue

        module_name = f"{module_prefix}.{stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"[discovery] Import échoué {module_name} : {e}")
            continue

        # METADATA obligatoire
        if not hasattr(module, 'METADATA'):
            raise CriticalDiscoveryError(
                f"{module_name} : METADATA manquant (obligatoire)"
            )

        metadata  = module.METADATA
        entity_id = metadata.get('id')

        if not entity_id:
            raise CriticalDiscoveryError(
                f"{module_name} : METADATA['id'] manquant (obligatoire)"
            )

        # callable obligatoire
        fn = getattr(module, callable_name, None)
        if fn is None:
            warnings.warn(
                f"[discovery] {module_name} : {callable_name}() absent — skipped"
            )
            continue

        # prepare_params optionnel — None si absent
        prepare_params_fn = getattr(module, 'prepare_params', None)

        registry[entity_id] = {
            'callable'       : fn,
            'metadata'       : metadata,
            'prepare_params' : prepare_params_fn,
        }

    return registry


def discover_gammas_jax() -> Dict[str, Dict[str, Any]]:
    """
    Scan atomics/operators/gam_*.py → registre gammas JAX.

    Returns:
        {'GAM-011': {'callable': apply_fn, 'metadata': {...},
                     'prepare_params': fn|None}, ...}
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'operators',
        pattern='gam_*.py',
        module_prefix='atomics.operators',
        callable_name='apply',
    )


def discover_encodings_jax() -> Dict[str, Dict[str, Any]]:
    """
    Scan atomics/D_encodings/{sym,asy,r3,rn}_*.py → registre encodings JAX.

    Returns:
        {'SYM-001': {'callable': create_fn, 'metadata': {...},
                     'prepare_params': fn|None}, ...}
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'D_encodings',
        pattern=['sym_*.py', 'asy_*.py', 'r3_*.py', 'rn_*.py'],
        module_prefix='atomics.D_encodings',
        callable_name='create',
    )


def discover_modifiers_jax() -> Dict[str, Dict[str, Any]]:
    """
    Scan atomics/modifiers/m*.py → registre modifiers JAX.

    Returns:
        {'M0': {'callable': apply_fn, 'metadata': {...},
                'prepare_params': fn|None}, ...}
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'modifiers',
        pattern='m*.py',
        module_prefix='atomics.modifiers',
        callable_name='apply',
    )


# =============================================================================
# SECTION 2 — LECTURE YAML
# =============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Charge un fichier YAML depuis son path absolu.

    Args:
        path : Path vers le fichier YAML

    Returns:
        dict config

    Raises:
        FileNotFoundError : Si fichier introuvable
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable : {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# SECTION 3 — ÉCRITURE PARQUET (pyarrow, append)
# =============================================================================

def _build_schema(sample_row: Dict) -> pa.Schema:
    """
    Construit le schéma pyarrow depuis une row d'exemple.

    Colonnes fixes : phase, gamma_id, encoding_id, modifier_id,
                     n_dof, rank_eff, max_it,
                     gamma_params, encoding_params, modifier_params,
                     seed_CI, seed_run
    Colonnes features : toutes float32 (NaN autorisé)

    Args:
        sample_row : dict {'composition': {...}, 'features': {...}}

    Returns:
        pa.Schema
    """
    fields = [
        pa.field('phase'          , pa.string()),
        pa.field('gamma_id'       , pa.string()),
        pa.field('encoding_id'    , pa.string()),
        pa.field('modifier_id'    , pa.string()),
        pa.field('n_dof'          , pa.int32()),
        pa.field('rank_eff'       , pa.int32()),
        pa.field('max_it'         , pa.int32()),
        pa.field('gamma_params'   , pa.string()),
        pa.field('encoding_params', pa.string()),
        pa.field('modifier_params', pa.string()),
        pa.field('seed_CI'        , pa.int64()),
        pa.field('seed_run'       , pa.int64()),
    ]
    # Colonnes features : float32 dans l'ordre de FEATURE_NAMES
    for k in sample_row['features']:
        fields.append(pa.field(k, pa.float32()))

    return pa.schema(fields)


def open_parquet_writer(
    phase      : str,
    output_dir : Path,
    sample_row : Dict,
) -> pq.ParquetWriter:
    """
    Ouvre un ParquetWriter en append.

    Doit être fermé via close_parquet_writer() en fin de phase.
    Le fichier est créé (ou écrasé) à l'ouverture.

    Args:
        phase      : Nom phase (ex: 'poc_v7') — détermine le nom de fichier
        output_dir : Dossier output
        sample_row : Première row — utilisée pour inférer le schéma

    Returns:
        pq.ParquetWriter ouvert
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filepath = output_dir / f'{phase}.parquet'
    schema   = _build_schema(sample_row)

    return pq.ParquetWriter(str(filepath), schema=schema)


def write_rows_to_parquet(
    writer : pq.ParquetWriter,
    rows   : List[Dict],
    phase  : str,
) -> int:
    """
    Écrit un batch de rows dans le ParquetWriter ouvert.

    Args:
        writer : ParquetWriter ouvert via open_parquet_writer()
        rows   : Liste de dicts {'composition': {...}, 'features': {...}}
        phase  : Nom phase (ajouté à chaque record)

    Returns:
        Nombre de rows écrites

    Notes:
        Appelé une ou plusieurs fois par phase.
        Pas de fermeture — le writer reste ouvert pour les batches suivants.
        Écriture avant verdict intra-run (resilience si crash verdict).
    """
    if not rows:
        return 0

    # Construire les colonnes fixes
    comps     = [r['composition'] for r in rows]
    features  = [r['features']    for r in rows]
    n         = len(rows)

    col_phase           = [phase]                                     * n
    col_gamma_id        = [c.get('gamma_id',    '')                  for c in comps]
    col_encoding_id     = [c.get('encoding_id', '')                  for c in comps]
    col_modifier_id     = [c.get('modifier_id', '')                  for c in comps]
    col_n_dof           = [c.get('n_dof',       0)                   for c in comps]
    col_rank_eff        = [c.get('rank_eff',    2)                   for c in comps]
    col_max_it          = [c.get('max_it',      0)                   for c in comps]
    col_gamma_params    = [json.dumps(c.get('gamma_params',    {}))  for c in comps]
    col_encoding_params = [json.dumps(c.get('encoding_params', {}))  for c in comps]
    col_modifier_params = [json.dumps(c.get('modifier_params', {}))  for c in comps]
    col_seed_CI         = [c.get('seed_CI',  None)                   for c in comps]
    col_seed_run        = [c.get('seed_run', None)                   for c in comps]

    arrays = [
        pa.array(col_phase,           type=pa.string()),
        pa.array(col_gamma_id,        type=pa.string()),
        pa.array(col_encoding_id,     type=pa.string()),
        pa.array(col_modifier_id,     type=pa.string()),
        pa.array(col_n_dof,           type=pa.int32()),
        pa.array(col_rank_eff,        type=pa.int32()),
        pa.array(col_max_it,          type=pa.int32()),
        pa.array(col_gamma_params,    type=pa.string()),
        pa.array(col_encoding_params, type=pa.string()),
        pa.array(col_modifier_params, type=pa.string()),
        pa.array(col_seed_CI,         type=pa.int64()),
        pa.array(col_seed_run,        type=pa.int64()),
    ]

    # Colonnes features dans l'ordre du schéma
    feature_keys = list(features[0].keys())
    for k in feature_keys:
        col = [float(f[k]) if f[k] is not None else float('nan') for f in features]
        arrays.append(pa.array(col, type=pa.float32()))

    # writer.schema est le pa.Schema passé à l'ouverture
    # (attribut correct de pq.ParquetWriter — pas schema_arrow)
    schema = writer.schema
    table  = pa.table(
        {schema.field(i).name: arrays[i] for i in range(len(arrays))},
        schema=schema,
    )
    writer.write_table(table)
    return n


def close_parquet_writer(writer: pq.ParquetWriter) -> None:
    """
    Ferme le ParquetWriter proprement.

    À appeler en fin de phase, dans un bloc finally pour garantir la fermeture
    même en cas d'exception.

    Args:
        writer : ParquetWriter ouvert via open_parquet_writer()
    """
    writer.close()


# =============================================================================
# SECTION 4 — UTILITAIRES
# =============================================================================

def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge dicts — override prime sur base.

    Dicts imbriqués : merge récursif.
    Listes/scalaires : override remplace base entièrement.
    """
    result = base.copy()
    for key, value in override.items():
        if (key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def fix_str(s: str) -> str:
    """Corrige double-encodage cp1252/utf-8."""
    if not isinstance(s, str):
        return str(s)
    try:
        return s.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


# =============================================================================
# SECTION 5 — LECTURE PARQUET CIBLÉE
# =============================================================================

# Colonnes parquet v7 qui ne sont pas des features
_META_COLS_V7 = {
    'run_status', 'phase',
    'gamma_id', 'encoding_id', 'modifier_id',
    'n_dof', 'rank_eff', 'max_it',
    'gamma_params', 'encoding_params', 'modifier_params',
    'seed_CI', 'seed_run',
}


def read_parquet_rows(
    parquet_path : Path,
    filters      : list = None,
) -> list:
    """
    Lecture parquet v7 avec filtres pyarrow pushdown optionnels.

    Interface bas niveau — utilisée par analysing/parquet_filter.py.
    Retourne les rows brutes avant post-filtres.

    Les filtres pyarrow sont construits par parquet_filter.build_pyarrow_filters().
    Ne pas appeler directement — utiliser analysing.parquet_filter.load_rows().

    Args:
        parquet_path : Path vers .parquet v7
        filters      : Liste de tuples pyarrow (col, op, val) ou None

    Returns:
        List[Dict] — {composition: dict, features: dict}

    Notes:
        Zéro dépendance vers analysing/ — data_loading reste découplé.
        Les post-filtres (seeds: one, pool_requirements) sont dans parquet_filter.py.
    """
    import json as _json

    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")

    df = pq.read_table(str(parquet_path), filters=filters or None)

    all_cols     = df.schema.names
    feature_cols = [c for c in all_cols if c not in _META_COLS_V7]
    cols         = df.to_pydict()
    n            = len(df)

    def _pj(s):
        if s is None: return {}
        try: return _json.loads(s)
        except: return {}

    def _ff(v):
        if v is None: return float('nan')
        try: return float(v)
        except: return float('nan')

    rows = []
    for i in range(n):
        rows.append({
            'composition': {
                'gamma_id'       : cols['gamma_id'][i],
                'encoding_id'    : cols['encoding_id'][i],
                'modifier_id'    : cols['modifier_id'][i],
                'n_dof'          : int(cols['n_dof'][i]),
                'rank_eff'       : int(cols['rank_eff'][i]),
                'max_it'         : int(cols['max_it'][i]),
                'run_status'     : cols['run_status'][i],
                'phase'          : cols['phase'][i],
                'seed_CI'        : cols['seed_CI'][i],
                'seed_run'       : cols['seed_run'][i],
                'gamma_params'   : _pj(cols['gamma_params'][i]),
                'encoding_params': _pj(cols['encoding_params'][i]),
                'modifier_params': _pj(cols['modifier_params'][i]),
            },
            'features': {k: _ff(cols[k][i]) for k in feature_cols},
        })

    return rows