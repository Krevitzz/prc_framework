"""
prc.utils.data_loading_lite

Responsabilité : Source unique de vérité pour tout chargement de données.
- Discovery atomics (gammas, encodings, modifiers)
- Lecture YAML générique avec modes (default/laxe/strict)
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
# SECTION 2 — LECTURE YAML GÉNÉRIQUE
# =============================================================================

# Mapping identifier → path pattern
# NOTE IMPORTANTE : Ajouter une ligne pour chaque nouveau module utilisant YAML
_YAML_PATHS = {
    # Atomics
    'operators'   : 'atomics/operators/configs/operators_{mode}.yaml',
    'D_encodings' : 'atomics/D_encodings/configs/D_encodings_{mode}.yaml',
    'modifiers'   : 'atomics/modifiers/configs/modifiers_{mode}.yaml',
    
    # Featuring (placeholders — à activer selon besoins)
    # 'layers'      : 'configs/features/default/layers.yaml',
    # 'algebra'     : 'configs/features/default/algebra.yaml',
    # 'statistical' : 'configs/features/default/statistical.yaml',
    
    # Thresholds (placeholders)
    # 'regimes'     : 'configs/thresholds/default/regimes.yaml',
    # 'aggregation' : 'configs/thresholds/default/aggregation.yaml',
    
    # Verdict (placeholders)
    # 'verdict'     : 'configs/verdict/default/default.yaml',
}


def resolve_yaml_path(identifier: str, mode: str = 'default') -> Path:
    """
    Résout path YAML depuis identifier et mode.
    
    Args:
        identifier : Clé mapping ('operators', 'D_encodings', etc.)
        mode       : 'default' | 'laxe' | 'strict'
    
    Returns:
        Path relatif vers le YAML (depuis prc/)
    
    Raises:
        KeyError : Si identifier inconnu dans _YAML_PATHS
    
    Examples:
        >>> resolve_yaml_path('operators', 'default')
        Path('configs/atomics/operators/default.yaml')
        >>> resolve_yaml_path('operators', 'laxe')
        Path('configs/atomics/operators/laxe.yaml')
    """
    if identifier not in _YAML_PATHS:
        raise KeyError(
            f"Identifier YAML inconnu : '{identifier}'\n"
            f"Disponibles : {list(_YAML_PATHS.keys())}\n"
            f"→ Ajouter une ligne dans _YAML_PATHS si nouveau module"
        )
    
    pattern = _YAML_PATHS[identifier]
    path_str = pattern.format(mode=mode)
    return Path(path_str)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """
    Deep merge dicts — override prime sur base.
    
    Stratégie :
    - Dicts imbriqués : merge récursif
    - Listes/scalaires : override remplace base entièrement
    
    Args:
        base     : Config de base (default)
        override : Config override (laxe/strict)
    
    Returns:
        Config fusionnée
    
    Examples:
        >>> base = {'a': 1, 'b': {'x': 10, 'y': 20}}
        >>> override = {'b': {'x': 99}, 'c': 3}
        >>> merge_configs(base, override)
        {'a': 1, 'b': {'x': 99, 'y': 20}, 'c': 3}
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Merge récursif pour dicts
            result[key] = merge_configs(result[key], value)
        else:
            # Remplacement direct pour scalaires/listes
            result[key] = value
    
    return result


def load_yaml(identifier: Union[str, Path], mode: str = 'default') -> Dict[str, Any]:
    """
    Charge YAML avec support mode override.
    
    Deux modes d'utilisation :
    1. Identifier string → charge depuis _YAML_PATHS avec mode
    2. Path absolu → charge directement (ex: YAML de run)
    
    Args:
        identifier : Clé mapping ('operators', 'D_encodings', ...)
                     OU Path absolu (ex: Path('configs/phases/poc/poc.yaml'))
        mode       : 'default' | 'laxe' | 'strict' (ignoré si identifier est Path)
    
    Returns:
        Dict config (avec merge default + mode si identifier string)
    
    Raises:
        FileNotFoundError : Si fichier YAML introuvable
        KeyError          : Si identifier inconnu dans _YAML_PATHS
    
    Examples:
        >>> load_yaml('operators')                          # default
        >>> load_yaml('operators', mode='laxe')             # default + laxe merged
        >>> load_yaml(Path('configs/phases/poc/poc.yaml'))  # path absolu
    """
    # Cas 1 : Path absolu (YAML de run, etc.)
    if isinstance(identifier, Path):
        if not identifier.exists():
            raise FileNotFoundError(f"Fichier YAML introuvable : {identifier}")
        
        with open(identifier, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    # Cas 2 : Identifier atomics/features/thresholds/etc.
    # Charger default (obligatoire)
    base_path = resolve_yaml_path(identifier, 'default')
    
    if not base_path.exists():
        raise FileNotFoundError(
            f"Config default introuvable : {base_path}\n"
            f"→ Créer {identifier}/default.yaml en premier"
        )
    
    with open(base_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Merge avec mode override si nécessaire
    if mode != 'default':
        override_path = resolve_yaml_path(identifier, mode)
        
        if override_path.exists():
            with open(override_path, 'r', encoding='utf-8') as f:
                override = yaml.safe_load(f)
            config = merge_configs(config, override)
        else:
            warnings.warn(
                f"Mode '{mode}' demandé mais {override_path} absent — "
                f"utilise default uniquement"
            )
    
    return config


# =============================================================================
# SECTION 3 — I/O PARQUET (D1/D3)
# =============================================================================

# À peupler selon besoins pipeline (divergences D1/D3)
