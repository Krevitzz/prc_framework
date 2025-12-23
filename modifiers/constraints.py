"""
modifiers/constraints.py

Modificateurs appliquant des contraintes expérimentales sur D.

NOTE: Ces fonctions sont des PLACEHOLDERS pour exploration future.
La notion de "position", "barrière", etc. est pré-géométrique.
Toute la rétro-ingénierie réalité → information reste à faire.
"""

import numpy as np
from typing import Callable, List


def apply_barrier_constraint(barrier_indices: List[int], 
                             aperture_size: int = 1) -> Callable[[np.ndarray], np.ndarray]:
    """
    PLACEHOLDER: Réduit corrélations trans-barrière.
    
    NOTE CRITIQUE:
    Cette fonction utilise "indices", "positions" qui sont des concepts
    pré-géométriques. L'exploration de comment les contraintes réelles
    (double fente, etc.) réduisent les DOF informationnels reste à faire.
    
    Args:
        barrier_indices: Indices des DOF où la contrainte s'applique
        aperture_size: Nombre de DOF "ouverts" par la contrainte
    
    Returns:
        Fonction qui modifie le tenseur
    """
    def modifier(state: np.ndarray) -> np.ndarray:
        """
        Applique contrainte barrière (LOGIQUE PLACEHOLDER).
        """
        modified = state.copy()
        
        # LOGIQUE À DÉVELOPPER
        # Pour l'instant: juste annule certaines corrélations
        if state.ndim == 2:
            for idx in barrier_indices:
                # Réduit corrélations impliquant cet indice
                # (sauf ouverture)
                # ...
                pass
        
        return modified
    
    return modifier


def apply_periodic_boundary() -> Callable[[np.ndarray], np.ndarray]:
    """
    PLACEHOLDER: Applique contrainte de périodicité.
    
    Concept à explorer: comment "périodicité" réduit DOF ?
    """
    def modifier(state: np.ndarray) -> np.ndarray:
        """Applique périodicité (LOGIQUE PLACEHOLDER)."""
        # À développer
        return state
    
    return modifier