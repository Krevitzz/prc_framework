"""
core/state_preparation.py

Composition aveugle d'états D.

RESPONSABILITÉ UNIQUE:
- Appliquer séquentiellement encoding → modifier_1 → modifier_2 → ...
- AUCUNE connaissance du contenu, de la dimension, ou de la structure

FONCTION AVEUGLE:
- Ne connaît ni dimension ni structure
- Pas de gestion seed (chaque atomic gère ses propres paramètres via YAML)
"""

import numpy as np
from typing import Callable, Optional, Dict, List


def prepare_state(
    encoding_func: Callable[..., np.ndarray],
    encoding_params: Dict,
    modifiers: Optional[List[Callable]] = None,
    modifier_configs: Optional[Dict[Callable, Dict]] = None,
) -> np.ndarray:
    """
    Compose un état D initial.

    FONCTION AVEUGLE:
    - Ne connaît ni dimension ni structure
    - Applique séquentiellement : encoding → modifier_1 → modifier_2 → ...
    - Aucune gestion seed (paramètres gérés par YAML via runner)

    Args:
        encoding_func   : Fonction création tenseur base
                         Signature : (**params) -> np.ndarray
        encoding_params : Paramètres encoding (depuis YAML)
                         Ex : {'n_dof': 10, 'gradient': 0.1}
        modifiers       : Liste fonctions modification (optionnel)
                         Signature : (state, **params) -> np.ndarray
        modifier_configs: Params par modifier (optionnel)
                         Ex : {create_noise: {'sigma': 0.05}}

    Returns:
        Tenseur composé final (np.ndarray)

    Examples:
        # Sans modifier
        D = prepare_state(
            encoding_func=create,
            encoding_params={'n_dof': 10},
        )

        # Avec modifier
        D = prepare_state(
            encoding_func=create,
            encoding_params={'n_dof': 10},
            modifiers=[apply_noise],
            modifier_configs={apply_noise: {'sigma': 0.05}},
        )
    """
    # Créer tenseur base
    state = encoding_func(**encoding_params)

    # Appliquer modifiers séquentiellement
    if modifiers is not None and len(modifiers) > 0:
        for modifier_func in modifiers:
            modifier_params = {}
            if modifier_configs and modifier_func in modifier_configs:
                modifier_params = modifier_configs[modifier_func]
            state = modifier_func(state, **modifier_params)

    return state
