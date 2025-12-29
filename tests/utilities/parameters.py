# tests/utilities/parameters.py
"""
Chargement paramètres tests depuis config YAML.

USAGE:
    params = get_test_parameters('UNIV-001')
    threshold = params['explosion_threshold']
"""

import yaml
from pathlib import Path
from typing import Dict, Any

CONFIG_PATH = Path("config/test_parameters.yaml")

_PARAMS_CACHE = None


def load_all_test_parameters() -> Dict[str, Dict[str, Any]]:
    """
    Charge tous les paramètres tests depuis YAML.
    
    Returns:
        dict {test_id: {param_name: value}}
    
    Raises:
        FileNotFoundError: Si config non trouvée
    """
    global _PARAMS_CACHE
    
    if _PARAMS_CACHE is None:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(
                f"Configuration tests non trouvée: {CONFIG_PATH}\n"
                f"Créer le fichier avec les paramètres requis."
            )
        
        with open(CONFIG_PATH, 'r') as f:
            _PARAMS_CACHE = yaml.safe_load(f)
    
    return _PARAMS_CACHE


def get_test_parameters(test_id: str) -> Dict[str, Any]:
    """
    Récupère paramètres d'un test spécifique.
    
    Args:
        test_id: ID du test (ex: "UNIV-001")
    
    Returns:
        dict des paramètres
    
    Raises:
        KeyError: Si test_id inconnu
    
    Example:
        >>> params = get_test_parameters('UNIV-001')
        >>> params['explosion_threshold']
        1000.0
    """
    all_params = load_all_test_parameters()
    
    if test_id not in all_params:
        raise KeyError(
            f"Paramètres non trouvés pour test: {test_id}\n"
            f"Tests disponibles: {list(all_params.keys())}"
        )
    
    return all_params[test_id]


def reload_parameters():
    """Force rechargement paramètres (utile pour tests)."""
    global _PARAMS_CACHE
    _PARAMS_CACHE = None