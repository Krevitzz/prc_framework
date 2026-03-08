"""
utils/data_loading_jax.py

Source unique de vérité pour le chargement de données PRC v7.

Responsabilités :
  - Discovery atomics JAX (gammas, encodings, modifiers)
  - Lecture YAML (path direct uniquement — pas de modes)
  - Écriture Parquet (schéma v7)

Convention callable :
  - Gammas    : apply(state, params, key)
  - Encodings : create(n_dof, params, key)
  - Modifiers : apply(state, params, key)

Usage : toujours lancer depuis prc_framework/ (python -m ...)
"""

import importlib
import warnings
import json
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml
import pandas as pd


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

    Returns:
        dict {id → {'callable': fn, 'metadata': dict}}
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

        registry[entity_id] = {
            'callable': fn,
            'metadata': metadata,
        }

    return registry


def discover_gammas_jax() -> Dict[str, Dict[str, Any]]:
    """
    Scan atomics/operators/gam_*.py → registre gammas JAX.

    Returns:
        {'GAM-011': {'callable': apply_fn, 'metadata': {...}}, ...}
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
        {'SYM-001': {'callable': create_fn, 'metadata': {...}}, ...}
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
        {'M0': {'callable': apply_fn, 'metadata': {...}}, ...}
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
# SECTION 3 — ÉCRITURE PARQUET
# =============================================================================

def write_parquet(
    rows      : List[Dict],
    phase     : str,
    output_dir: Path,
) -> Path:
    """
    Écrit les observations en Parquet — schéma v7.

    Args:
        rows       : Liste de dicts {'composition': {...}, 'features': {...}}
        phase      : Nom phase (ex: 'poc_v7')
        output_dir : Dossier output

    Schéma colonnes :
        phase, gamma_id, encoding_id, modifier_id,
        n_dof, rank_eff, max_it,
        gamma_params (json), encoding_params (json), modifier_params (json),
        seed_CI, seed_run,
        + features plates (une colonne par feature)

    Returns:
        Path du fichier Parquet créé
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for row in rows:
        comp     = row['composition']
        features = row['features']

        record = {
            'phase'          : phase,
            'gamma_id'       : comp.get('gamma_id', ''),
            'encoding_id'    : comp.get('encoding_id', ''),
            'modifier_id'    : comp.get('modifier_id', ''),
            'n_dof'          : comp.get('n_dof', 0),
            'rank_eff'       : comp.get('rank_eff', 2),
            'max_it'         : comp.get('max_it', 0),
            'gamma_params'   : json.dumps(comp.get('gamma_params', {})),
            'encoding_params': json.dumps(comp.get('encoding_params', {})),
            'modifier_params': json.dumps(comp.get('modifier_params', {})),
            'seed_CI'        : comp.get('seed_CI', None),
            'seed_run'       : comp.get('seed_run', None),
            **features,
        }

        records.append(record)

    df       = pd.DataFrame(records)
    filepath = output_dir / f'{phase}.parquet'
    df.to_parquet(filepath, index=False, engine='pyarrow')

    print(f"\n✓ Parquet écrit : {filepath}")
    print(f"  Observations  : {len(df)}")
    print(f"  Colonnes      : {len(df.columns)}")

    return filepath


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