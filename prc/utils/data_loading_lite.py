"""
prc.utils.data_loading

Responsabilité : Source unique de vérité pour tout chargement de données.
- Discovery atomics (gammas, encodings, modifiers)
- Lecture YAML
- I/O Parquet (à peupler)

Divergences actives : D1 (Parquet), D3 (colonnes features)
Convention : Tout callable atomique s'appelle create()

Usage : Toujours lancer depuis prc/ (python -m ...) — sys.path géré par Python.
"""

import importlib
import warnings
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CriticalDiscoveryError(Exception):
    """METADATA['id'] manquant — arrêt obligatoire."""
    pass


class ValidationError(Exception):
    """Structure atomique invalide."""
    pass


# =============================================================================
# SECTION 1 — DISCOVERY ATOMICS
# =============================================================================

def discover_gammas() -> List[Dict[str, Any]]:
    """
    Scan atomics/operators/gamma_hyp_*.py → liste entités disponibles.

    Retour : [{'id', 'module', 'callable', 'metadata'}, ...]
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'operators',
        pattern='gamma_hyp_*.py',
        module_prefix='atomics.operators',
    )


def discover_encodings() -> List[Dict[str, Any]]:
    """
    Scan atomics/D_encodings/{sym,asy,r3}_*.py → liste entités disponibles.

    Retour : [{'id', 'module', 'callable', 'metadata'}, ...]
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'D_encodings',
        pattern=['sym_*.py', 'asy_*.py', 'r3_*.py'],
        module_prefix='atomics.D_encodings',
    )


def discover_modifiers() -> List[Dict[str, Any]]:
    """
    Scan atomics/modifiers/m*.py → liste entités disponibles.

    Retour : [{'id', 'module', 'callable', 'metadata'}, ...]
    """
    atomics_dir = Path(__file__).parent.parent / 'atomics'
    return _discover_from_dir(
        directory=atomics_dir / 'modifiers',
        pattern='m*.py',
        module_prefix='atomics.modifiers',
    )


def _discover_from_dir(
    directory: Path,
    pattern: Union[str, List[str]],
    module_prefix: str,
) -> List[Dict[str, Any]]:
    """
    Moteur commun discovery.

    Args:
        directory     : Dossier à scanner
        pattern       : Glob pattern(s) — str ou liste de str
        module_prefix : Préfixe import (ex: 'atomics.operators')

    Règles :
    - Skip *_deprecated_* et __init__
    - METADATA absent ou METADATA['id'] absent → CriticalDiscoveryError
    - create() absent → warning + skip
    - PHASE ni requis ni filtré

    Retour : [{'id', 'module', 'callable', 'metadata'}, ...]
    """
    if not directory.exists():
        raise FileNotFoundError(f"Dossier discovery introuvable : {directory}")

    # Collecter fichiers selon pattern(s)
    patterns = [pattern] if isinstance(pattern, str) else pattern
    files = []
    for p in patterns:
        files.extend(directory.glob(p))

    entities = []

    for filepath in sorted(files):
        stem = filepath.stem

        # Skip deprecated et __init__
        if '_deprecated_' in stem or stem == '__init__':
            continue

        # Import module
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

        metadata = module.METADATA

        # METADATA['id'] obligatoire
        entity_id = metadata.get('id')
        if not entity_id:
            raise CriticalDiscoveryError(
                f"{module_name} : METADATA['id'] manquant (obligatoire)"
            )

        # create() obligatoire
        create_fn = getattr(module, 'create', None)
        if create_fn is None:
            warnings.warn(
                f"[discovery] {module_name} : create() absent — skipped"
            )
            continue

        entities.append({
            'id': entity_id,
            'module': module,
            'callable': create_fn,
            'metadata': metadata,
        })

    return entities


# =============================================================================
# SECTION 2 — LECTURE YAML
# =============================================================================

def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Charge fichier YAML → dict.

    Raises : FileNotFoundError si path inexistant
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable : {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# SECTION 3 — I/O PARQUET (D1/D3)
# =============================================================================

# À peupler selon besoins pipeline (divergences D1/D3)