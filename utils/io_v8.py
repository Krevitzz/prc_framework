"""
I/O générique pipeline PRC v15 — discovery, YAML, parquet.

Utilisé par running ET analysing. Aucune dépendance vers d'autres layers PRC.
Le schéma parquet est construit depuis les définitions passées par l'appelant
(typiquement issues de features_registry.py).

@ROLE    Discovery atomics, YAML, Parquet v15 — utils transversal
@LAYER   utils

@EXPORTS
  discover_gammas_jax()         → Dict   | registre gammas
  discover_encodings_jax()      → Dict   | registre encodings
  discover_modifiers_jax()      → Dict   | registre modifiers
  load_yaml(path)               → Dict   | chargement YAML
  build_schema_v15(meta, feat, tl) → Schema | schéma parquet v15
  open_parquet_writer(phase, dir, schema) → ParquetWriter
  write_rows_to_parquet(writer, rows)     → int
  close_parquet_writer(writer)            → None
  read_parquet(path, columns)             → Table

@LIFECYCLE
  CREATES  ParquetWriter   via open_parquet_writer, fermé par close_parquet_writer
  PASSES   ParquetWriter   vers le hub qui gère son lifecycle

@CONFORMITY
  OK   Aucune dépendance vers d'autres layers PRC (P8)
  OK   Schéma construit depuis paramètres, pas hardcodé (P4)
"""

import importlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional
import ast
import yaml
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# =========================================================================
# EXCEPTIONS
# =========================================================================

class CriticalDiscoveryError(Exception):
    """Erreur fatale lors de la découverte d'atomics."""
    pass

# =========================================================================
# DISCOVERY METADATA-ONLY (AST, zéro import JAX)
# =========================================================================

def _extract_metadata_from_file(filepath):
    """
    Lit METADATA depuis un fichier Python par analyse AST, sans importer le module.
    Retourne le dict METADATA ou None si absent/illisible.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
    except (SyntaxError, IOError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'METADATA':
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def _discover_metadata_from_dir(directory, pattern):
    """
    Scanne un répertoire, extrait METADATA par AST sans import de modules.
    Même filtrage que _discover_from_dir (_deprecated_, __init__).

    Retourne : {id: {'metadata': dict, 'callable': None, 'prepare_params': None}}
    Structure compatible avec resolve_axis_atomic.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Dossier discovery introuvable : {directory}")

    patterns = [pattern] if isinstance(pattern, str) else pattern
    files = []
    for p in patterns:
        files.extend(directory.glob(p))

    registry = {}
    for filepath in sorted(files):
        stem = filepath.stem
        if '_deprecated_' in stem or stem == '__init__':
            continue

        metadata = _extract_metadata_from_file(filepath)
        if metadata is None:
            warnings.warn(f"[discovery-metadata] METADATA illisible : {filepath}")
            continue

        entity_id = metadata.get('id')
        if not entity_id:
            warnings.warn(f"[discovery-metadata] METADATA sans 'id' : {filepath}")
            continue

        registry[entity_id] = {
            'metadata': metadata,
            'callable': None,
            'prepare_params': None,
        }

    return registry


def discover_gammas_metadata():
    """Gammas metadata-only depuis atomics/operators/gam_*.py. Zéro import JAX."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_metadata_from_dir(atomics_dir / 'operators', 'gam_*.py')


def discover_encodings_metadata():
    """Encodings metadata-only depuis atomics/D_encodings/. Zéro import JAX."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_metadata_from_dir(
        atomics_dir / 'D_encodings',
        ['sym_*.py', 'asy_*.py', 'r3_*.py', 'rn_*.py'])


def discover_modifiers_metadata():
    """Modifiers metadata-only depuis atomics/modifiers/m*.py. Zéro import JAX."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_metadata_from_dir(atomics_dir / 'modifiers', 'm*.py')
# =========================================================================
# DISCOVERY ATOMICS
# =========================================================================

def _discover_from_dir(directory, pattern, module_prefix, callable_name):
    """
    Scanne un répertoire, importe les modules, construit un registre.

    Retourne : {id: {'callable': fn, 'metadata': dict, 'prepare_params': fn|None}}

    Filtre : _deprecated_, __init__.
    Exige : METADATA avec 'id', callable_name dans le module.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Dossier discovery introuvable : {directory}")

    patterns = [pattern] if isinstance(pattern, str) else pattern
    files = []
    for p in patterns:
        files.extend(directory.glob(p))

    registry = {}
    for filepath in sorted(files):
        stem = filepath.stem
        if '_deprecated_' in stem or stem == '__init__':
            continue

        module_name = f"{module_prefix}.{stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            warnings.warn(f"[discovery] Import échoué {module_name} : {e}")
            continue

        if not hasattr(module, 'METADATA'):
            raise CriticalDiscoveryError(f"{module_name} : METADATA manquant")

        metadata = module.METADATA
        entity_id = metadata.get('id')
        if not entity_id:
            raise CriticalDiscoveryError(f"{module_name} : METADATA['id'] manquant")

        fn = getattr(module, callable_name, None)
        if fn is None:
            warnings.warn(f"[discovery] {module_name} : {callable_name}() absent — skipped")
            continue

        prepare_params_fn = getattr(module, 'prepare_params', None)

        registry[entity_id] = {
            'callable': fn,
            'metadata': metadata,
            'prepare_params': prepare_params_fn,
        }

    return registry


def discover_gammas_jax():
    """Découvre les gammas JAX depuis atomics/operators/gam_*.py."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        atomics_dir / 'operators', 'gam_*.py', 'atomics.operators', 'apply')


def discover_encodings_jax():
    """Découvre les encodings JAX depuis atomics/D_encodings/."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        atomics_dir / 'D_encodings', ['sym_*.py', 'asy_*.py', 'r3_*.py', 'rn_*.py'],
        'atomics.D_encodings', 'create')


def discover_modifiers_jax():
    """Découvre les modifiers JAX depuis atomics/modifiers/m*.py."""
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        atomics_dir / 'modifiers', 'm*.py', 'atomics.modifiers', 'apply')


# =========================================================================
# YAML
# =========================================================================

def load_yaml(path):
    """Charge un fichier YAML. Erreur explicite si absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable : {path}")
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =========================================================================
# PARQUET V15 — SCHEMA EXPLICITE
# =========================================================================

# Mapping type_string → pyarrow type (pour les metadata_columns)
_PA_TYPE_MAP = {
    'string':  pa.string(),
    'int32':   pa.int32(),
    'int64':   pa.int64(),
    'float32': pa.float32(),
    'float64': pa.float64(),
}


def build_schema_v15(metadata_columns, feature_names, timeline_columns,
                     timeline_int_columns=None):
    """
    Construit le schéma parquet v15 depuis les définitions de colonnes.

    L'appelant passe typiquement les valeurs de features_registry :
      metadata_columns     = METADATA_COLUMNS
      feature_names        = FEATURE_NAMES
      timeline_columns     = TIMELINE_COLUMNS_A + TIMELINE_COLUMNS_B
      timeline_int_columns = [MASK_INDICES_COLUMN]

    Ce module n'importe PAS features_registry (indépendance de layer).

    Args:
        metadata_columns     : Dict[str, str]    — {nom: type_string}
        feature_names        : List[str]         — noms des colonnes float32
        timeline_columns     : List[str]         — noms des colonnes list<float32>
        timeline_int_columns : List[str] | None  — noms des colonnes list<int32>

    Returns:
        pa.Schema
    """
    fields = []

    # Metadata — types hétérogènes
    for name, type_str in metadata_columns.items():
        if type_str not in _PA_TYPE_MAP:
            raise ValueError(
                f"Type inconnu '{type_str}' pour colonne '{name}'. "
                f"Types valides : {list(_PA_TYPE_MAP.keys())}")
        fields.append(pa.field(name, _PA_TYPE_MAP[type_str]))

    # Features — toutes float32
    for name in feature_names:
        fields.append(pa.field(name, pa.float32()))

    # Timelines — list<float32>
    for name in timeline_columns:
        fields.append(pa.field(name, pa.list_(pa.float32())))

    # Timelines entières — list<int32> (mask indices)
    if timeline_int_columns:
        for name in timeline_int_columns:
            fields.append(pa.field(name, pa.list_(pa.int32())))

    return pa.schema(fields)


def open_parquet_writer(phase, output_dir, schema):
    """
    Ouvre un ParquetWriter v15 avec schéma explicite.

    Args:
        phase      : nom de la phase (utilisé comme nom de fichier)
        output_dir : répertoire de sortie
        schema     : pa.Schema (issu de build_schema_v15)

    Returns:
        pq.ParquetWriter
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f'{phase}.parquet'
    return pq.ParquetWriter(str(filepath), schema=schema)


def write_rows_to_parquet(writer, rows):
    """
    Écrit des rows plates dans le parquet v15.

    Chaque row est un dict plat {col_name: value}.
    Gère les types : string, int32, int64, float32, list<float32>.
    Les valeurs None sont converties en -1 (int) ou NaN (float) ou None (list).

    Returns:
        int — nombre de rows écrites
    """
    if not rows:
        return 0

    schema = writer.schema
    arrays = {}

    for i in range(len(schema)):
        field = schema.field(i)
        col_name = field.name
        col_type = field.type
        col_values = [r.get(col_name) for r in rows]

        if col_type == pa.int32() or col_type == pa.int64():
            col_values = [int(v) if v is not None else -1 for v in col_values]

        elif col_type == pa.float32():
            col_values = [float(v) if v is not None else float('nan')
                          for v in col_values]

        elif col_type == pa.string():
            col_values = [str(v) if v is not None else '' for v in col_values]

        elif isinstance(col_type, pa.ListType):
            # Timeline columns : list<float32>
            # Valeur attendue : list de float, ou None
            col_values = [v if v is not None else [] for v in col_values]

        arrays[col_name] = pa.array(col_values, type=col_type)

    table = pa.table(arrays, schema=schema)
    writer.write_table(table)
    return len(rows)


def close_parquet_writer(writer):
    """Ferme le ParquetWriter."""
    writer.close()


def read_parquet(parquet_path, columns=None):
    """Lecture parquet. Retourne pyarrow Table."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet introuvable : {parquet_path}")
    return pq.read_table(str(parquet_path), columns=columns)

# Ajouter à la fin du fichier io_v8.py

def write_col_data_to_parquet(writer, col_data):
    """
    Écrit un dictionnaire de colonnes (arrays numpy ou listes) dans le parquet.

    Args:
        writer: pq.ParquetWriter
        col_data: dict {col_name: array-like} (numpy array ou list)

    Returns:
        int: nombre de lignes écrites
    """
    import pyarrow as pa
    schema = writer.schema
    arrays = {}

    for i in range(len(schema)):
        field = schema.field(i)
        col_name = field.name
        col_type = field.type
        if col_name not in col_data:
            raise KeyError(f"Colonne '{col_name}' manquante dans col_data")
        values = col_data[col_name]

        if col_type == pa.int32() or col_type == pa.int64():
            if isinstance(values, np.ndarray):
                values = values.astype(np.int64) if col_type == pa.int64() else values.astype(np.int32)
            else:
                values = np.array(values, dtype=np.int64 if col_type == pa.int64() else np.int32)
            arrays[col_name] = pa.array(values, type=col_type)

        elif col_type == pa.float32():
            if isinstance(values, np.ndarray):
                values = values.astype(np.float32)
            else:
                values = np.array(values, dtype=np.float32)
            arrays[col_name] = pa.array(values, type=col_type)

        elif col_type == pa.string():
            if isinstance(values, np.ndarray):
                values = values.astype(str)
            else:
                values = [str(v) for v in values]
            arrays[col_name] = pa.array(values, type=col_type)

        elif isinstance(col_type, pa.ListType):
            # Pour les timelines : listes de listes
            if not isinstance(values, list):
                values = values.tolist() if hasattr(values, 'tolist') else list(values)
            arrays[col_name] = pa.array(values, type=col_type)

        else:
            raise TypeError(f"Type non géré pour colonne {col_name}: {col_type}")

    table = pa.table(arrays, schema=schema)
    writer.write_table(table)
    # Déterminer le nombre de lignes
    first_key = next(iter(col_data))
    if isinstance(col_data[first_key], np.ndarray):
        n = col_data[first_key].shape[0]
    else:
        n = len(col_data[first_key])
    return n
