"""
prc.featuring.extractor_lite

Responsabilité : Dispatch extraction features vers registres

Architecture :
    extract_for_layer() : importe dynamiquement {layer_name}_lite
                          et appelle module.extract(history, layer_config)

Contrat registre :
    Chaque fichier {layer}_lite.py doit exposer :
        extract(history: np.ndarray, config: Dict) -> Dict[str, float]

Extensibilité :
    Nouveau layer = créer {layer}_lite.py + {layer}.yaml
    Zéro touche ici.

Notes :
    compute_projection() supprimé — cas dégénéré du layer timeline.
    Toute logique d'extraction appartient au registre, pas ici.
"""

import importlib
from typing import Dict

import numpy as np


def extract_for_layer(
    history: np.ndarray,
    layer_name: str,
    layer_config: Dict
) -> Dict[str, float]:
    """
    Dispatch extraction features vers registre dynamique.

    Args:
        history      : np.ndarray (T, *dims) — séquence temporelle états
        layer_name   : Nom du layer (ex: 'timeline', 'matrix_2d')
        layer_config : Config YAML du layer

    Returns:
        Dict[str, float] — features extraites par le registre
        Dict vide si registre introuvable ou sans extract()

    Notes :
        - Import dynamique : featuring.registries.{layer_name}_lite
        - Contrat : module.extract(history, config) → Dict[str, float]
        - Nouveau layer = nouveau fichier, zéro touche ici
    """
    try:
        module = importlib.import_module(f'featuring.registries.{layer_name}_lite')
    except ImportError:
        print(f"[WARNING] extractor_lite: registre '{layer_name}_lite' introuvable")
        return {}

    if not hasattr(module, 'extract'):
        print(f"[WARNING] extractor_lite: '{layer_name}_lite' n'expose pas extract()")
        return {}

    try:
        return module.extract(history, layer_config)
    except Exception as e:
        print(f"[WARNING] extractor_lite: extract() échoué pour '{layer_name}': {e}")
        return {}
