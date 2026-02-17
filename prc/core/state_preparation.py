"""
core/state_preparation.py

Composition aveugle d'états D - REFONTE PHASE 10.

CHANGEMENT MAJEUR:
- Centralisation seed (R0)
- Encodings + modifiers appelés depuis prepare_state
- Plus de gestion seed dispersée

BREAKING CHANGE:
- Ancienne signature abandonnée
- Encodings n'ont plus de param seed
"""

import numpy as np
from typing import Callable, Optional, Dict, List


def prepare_state(
    encoding_func: Callable[..., np.ndarray],
    encoding_params: Dict,
    modifiers: Optional[List[Callable]] = None,
    modifier_configs: Optional[Dict[Callable, Dict]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Compose un état D avec centralisation seed.
    
    SEED MANAGEMENT:
    - Seed fixé UNE FOIS au début
    - Tous appels aléatoires reproductibles
    - Encodings et modifiers n'ont PLUS de param seed
    
    FONCTION AVEUGLE:
    - Ne connaît ni dimension ni structure
    - Applique séquentiellement: encoding → modifier_1 → modifier_2 → ...
    
    Args:
        encoding_func: Fonction création tenseur base
                      Signature: (**params) -> np.ndarray
                      Ex: create(n_dof=10)
        
        encoding_params: Paramètres encoding
                        Ex: {'n_dof': 10}
        
        modifiers: Liste fonctions modification (optionnel)
                  Signature: (state, **params) -> np.ndarray
                  Ex: [apply_gaussian_noise, apply_uniform_noise]
        
        modifier_configs: Params par modifier (optionnel)
                         Ex: {apply_gaussian_noise: {'sigma': 0.05}}
        
        seed: Graine aléatoire globale (optionnel)
             Si None, état aléatoire non fixé
    
    Returns:
        Tenseur composé final (np.ndarray)
    
    Examples:
        # Sans modifier
        D = prepare_state(
            encoding_func=create_symmetric,
            encoding_params={'n_dof': 10},
            seed=42
        )
        
        # Avec modifiers
        D = prepare_state(
            encoding_func=create_symmetric,
            encoding_params={'n_dof': 10},
            modifiers=[apply_noise],
            modifier_configs={apply_noise: {'sigma': 0.05}},
            seed=42
        )
        
        # M0 baseline (sans modifier)
        D = prepare_state(
            encoding_func=create_symmetric,
            encoding_params={'n_dof': 10},
            modifiers=None,
            seed=42
        )
    
    Notes:
        - Seed fixé AVANT encoding
        - Modifiers appliqués séquentiellement
        - Core reste aveugle au contenu
    """
    # Fixer seed global si fourni
    if seed is not None:
        np.random.seed(seed)
    
    # Créer tenseur base
    state = encoding_func(**encoding_params)
    
    # Appliquer modifiers séquentiellement
    if modifiers is not None and len(modifiers) > 0:
        for modifier_func in modifiers:
            # Extraire params pour ce modifier
            modifier_params = {}
            if modifier_configs and modifier_func in modifier_configs:
                modifier_params = modifier_configs[modifier_func]
            
            # Appliquer
            state = modifier_func(state, **modifier_params)
    
    return state