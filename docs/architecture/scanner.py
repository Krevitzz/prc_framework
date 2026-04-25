# scanner.py
import os
from pathlib import Path
from typing import List, Set

# ----- Constantes d'exclusion -----
EXCLUDE_DIRS = {
    'tests', 'docs', 'data', '__pycache__', '.git', '.ipynb_checkpoints'
}
EXCLUDE_PATHS = {
    'configs/phases'
}
VALID_EXTS = {'.py', '.yaml'}

# ----- Règles de filtrage pour les analyseurs -----
EXCLUDED_LAYERS = {'atomics', 'tests', 'docs', 'data'}
EXCLUDED_MODULES = {'batch','parquet_to_json'}
EXCLUDED_PREFIXES = {}


def is_constant_name(name: str) -> bool:
    """Retourne True si le nom est considéré comme une constante (tout en majuscules)."""
    return name.isupper() and '_' in name


def should_exclude_module(module_id: str, layer: str, name: str) -> bool:
    """Détermine si un module doit être exclu des analyses."""
    if layer in EXCLUDED_LAYERS:
        return True
    if name in EXCLUDED_MODULES:
        return True
    if any(name.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    return False


def should_exclude_function(func_name: str) -> bool:
    """Exclut les méthodes spéciales comme __init__, __repr__, etc."""
    if func_name.startswith('__') and func_name.endswith('__'):
        return True
    return False


# ----- Fonction de scan -----
def scan_files(root_path: str) -> List[Path]:
    """
    Parcourt récursivement root_path, applique les exclusions,
    et retourne la liste triée des chemins absolus des fichiers .py et .yaml.
    """
    root = Path(root_path).resolve()
    results: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        # Élagage des dossiers exclus
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDE_DIRS and not d.startswith('__')
        ]

        # Vérifier si le chemin courant est dans un sous-chemin exclu
        try:
            rel_current = current.relative_to(root).as_posix()
        except ValueError:
            rel_current = ''
        if any(rel_current == ep or rel_current.startswith(ep + '/') for ep in EXCLUDE_PATHS):
            dirnames.clear()
            continue

        for fname in filenames:
            fpath = current / fname
            if fpath.suffix in VALID_EXTS:
                results.append(fpath)

    return sorted(results)
