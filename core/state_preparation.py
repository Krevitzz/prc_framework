"""
core/state_preparation.py

Composition aveugle d'états D à partir de sources multiples.

RESPONSABILITÉ UNIQUE:
- Appliquer séquentiellement des modificateurs sur un tenseur de base
- AUCUNE connaissance du contenu, dimension, ou interprétation

USAGE:
    D_base = create_base_state(50)
    D_final = prepare_state(D_base, [
        add_noise(sigma=0.05),
        apply_constraint(params)
    ])
"""

import numpy as np
from typing import List, Callable, Optional


def prepare_state(base: np.ndarray,
                  modifiers: Optional[List[Callable[[np.ndarray], np.ndarray]]] = None) -> np.ndarray:
    """
    Compose un état D par application séquentielle de modificateurs.
    
    FONCTION AVEUGLE:
    - Ne connaît ni la dimension du tenseur
    - Ne connaît ni sa structure (symétrie, bornes, etc.)
    - Ne connaît ni son interprétation (corrélations, champ, etc.)
    
    Applique simplement: state → modifier_1 → modifier_2 → ... → state_final
    
    Args:
        base: Tenseur de base (np.ndarray de shape quelconque)
        modifiers: Liste de fonctions (np.ndarray → np.ndarray)
                   Chaque fonction transforme le tenseur
                   Si None ou [], retourne base inchangé
    
    Returns:
        Tenseur composé final (np.ndarray)
    
    Exemples:
        # Sans modificateur
        D = prepare_state(base_state)
        
        # Avec modificateurs
        D = prepare_state(base_state, [
            add_gaussian_noise(sigma=0.05),
            apply_periodic_constraint()
        ])
    
    Notes:
        - Chaque modifier reçoit le résultat du précédent
        - Les modifiers sont définis HORS du core (dans modifiers/)
        - Le core ne valide RIEN sur le contenu
    """
    state = base.copy()
    
    if modifiers is not None and len(modifiers) > 0:
        for modifier in modifiers:
            state = modifier(state)
    
    return state