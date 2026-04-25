# yaml_parser.py
from pathlib import Path
from typing import List, Dict, Any, Set

from .models import YamlInfo

def _extract_header_comments(filepath: Path) -> str:
    """
    Extrait les commentaires consécutifs au début du fichier,
    jusqu'à la première ligne non vide qui n'est pas un commentaire.
    """
    lines = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    # Ligne vide : on continue (ignore)
                    continue
                if stripped.startswith('#'):
                    # Enlever le # et les espaces
                    comment = stripped.lstrip('#').strip()
                    if comment:
                        lines.append(comment)
                else:
                    # Première ligne non commentaire, on arrête
                    break
    except Exception:
        pass
    return '\n'.join(lines)

def _collect_scalars(obj: Any, collection: Set[Any]) -> None:
    """Parcourt récursivement un objet YAML et ajoute les scalaires à collection."""
    if isinstance(obj, dict):
        for v in obj.values():
            _collect_scalars(v, collection)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_scalars(item, collection)
    elif isinstance(obj, (int, float, str, bool)):
        collection.add(obj)

def parse_yaml(filepath: Path) -> YamlInfo:
    """
    Charge un fichier YAML et retourne ses clés top-level, toutes les valeurs scalaires,
    et une description extraite des commentaires d'en-tête.
    """
    description = _extract_header_comments(filepath)
    try:
        import yaml
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            keys = list(data.keys())
        else:
            keys = []  # YAML peut être une liste, etc.
        scalars = set()
        _collect_scalars(data, scalars)
    except Exception as e:
        keys = [f'PARSE_ERROR: {e}']
        scalars = set()
    return YamlInfo(
        path=filepath,
        top_keys=keys,
        scalar_values=scalars,
        description=description
    )
