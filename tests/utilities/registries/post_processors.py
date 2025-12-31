# prc_framework/tests/utilities/registries/post_processors.py

import numpy as np
from typing import Callable, Dict

POST_PROCESSORS: Dict[str, Callable] = {
    # Identity
    'identity': lambda x: x,
    
    # Rounding
    'round_2': lambda x: round(float(x), 2),
    'round_4': lambda x: round(float(x), 4),
    'round_6': lambda x: round(float(x), 6),
    
    # Absolute value
    'abs': lambda x: abs(float(x)),
    
    # Logarithmic
    'log': lambda x: float(np.log(x + 1e-10)),
    'log10': lambda x: float(np.log10(x + 1e-10)),
    'log1p': lambda x: float(np.log1p(x)),
    
    # Clipping
    'clip_01': lambda x: float(np.clip(x, 0, 1)),
    'clip_positive': lambda x: float(max(0, x)),
    
    # Scientific notation
    'scientific_3': lambda x: float(f"{x:.3e}"),
}


def get_post_processor(key: str) -> Callable:
    """
    Récupère un post-processor par clé.
    
    Args:
        key: Identifiant du post-processor
    
    Returns:
        Callable: Fonction de transformation (float -> float)
    
    Raises:
        KeyError: Si clé inconnue
    """
    if key not in POST_PROCESSORS:
        available = list(POST_PROCESSORS.keys())
        raise KeyError(
            f"Post-processor '{key}' inconnu. "
            f"Disponibles: {available}"
        )
    
    return POST_PROCESSORS[key]


def add_post_processor(key: str, func: Callable) -> None:
    """
    Ajoute un post-processor custom.
    
    Args:
        key: Identifiant unique
        func: Fonction (float -> float)
    
    Raises:
        ValueError: Si clé déjà existante
    """
    if key in POST_PROCESSORS:
        raise ValueError(f"Post-processor '{key}' déjà existant")
    
    POST_PROCESSORS[key] = func